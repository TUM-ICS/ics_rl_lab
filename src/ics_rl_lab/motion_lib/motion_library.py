"""Multi-clip motion libraries for reference-motion tasks (tracking, ...).

Reads pre-baked retargeted ``.npz`` clips (``joint_pos``/``joint_vel`` + world-frame
``body_pos_w``/``body_quat_w``/``body_lin_vel_w``/``body_ang_vel_w``, all already
computed -- no FK here; the same format ``AMPMotionLoader`` reads).

Two storage backends behind one interface (``MotionLibrary``):

- ``ResidentMotionLibrary``: every clip loaded once, kept resident (default fp16 on
  the GPU -- the files themselves are fp16, so nothing is lost; ~8 GB for 48k clips).
- ``WindowedMotionLibrary``: only a pool of ``window_size`` clips is resident. Every
  ``refresh_interval_s`` a new pool is drawn from the full catalog (weighted) and
  loaded in a background thread; once ready, new samples come from it while envs
  still playing a clip of the previous pool keep reading that pool until their next
  reset. The old pool is freed as soon as no env references it -- no forced
  synchronized reset.

Callers hold opaque ``handles`` (LongTensor, one per env) returned by
``sample_clips`` and query frames with ``get_frames(handles, times[N, K])`` -- the
``K`` axis is how multi-frame horizons (e.g. a critic's future-motion lookahead) are
served in one vectorized call. Frames are linearly interpolated between the two
neighbouring samples (quaternions: normalized lerp with hemisphere fix); times past
the clip end are held at the last frame and flagged ``valid=False``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import yaml

_FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass
class MotionFrames:
  """Reference frames for a batch of (clip, time) queries. All float32, shape [N, K, ...]."""

  joint_pos: torch.Tensor  # [N, K, J]
  joint_vel: torch.Tensor  # [N, K, J]
  body_pos_w: torch.Tensor  # [N, K, B, 3]  (clip frame, no env origin applied)
  body_quat_w: torch.Tensor  # [N, K, B, 4]  (w, x, y, z)
  body_lin_vel_w: torch.Tensor  # [N, K, B, 3]
  body_ang_vel_w: torch.Tensor  # [N, K, B, 3]
  valid: torch.Tensor  # [N, K] bool, False where the query time is past the clip end


##
# Configs.
##


@dataclass(kw_only=True)
class MotionLibraryCfg:
  path: str
  """Directory walked (recursively, following symlinks) for ``*.npz`` clips."""

  selection_file: str | None = None
  """Optional YAML ``{selected_files: [...], motion_weights: [...]}`` (paths relative
  to ``path``) narrowing and weighting the catalog. Without it every clip has weight 1."""

  dtype: Literal["float16", "bfloat16", "float32"] = "float16"
  """Storage dtype. Queries are always returned as float32."""

  storage_device: Literal["output", "cpu"] = "output"
  """Where the resident frames live. "cpu" gathers on the CPU and copies the result
  to the output device every query (slower, saves VRAM)."""

  num_load_workers: int = 16

  def build(
    self, joint_names: Sequence[str], body_names: Sequence[str], device: str, num_envs: int
  ) -> MotionLibrary:
    raise NotImplementedError


@dataclass(kw_only=True)
class ResidentMotionLibraryCfg(MotionLibraryCfg):
  def build(self, joint_names, body_names, device, num_envs) -> ResidentMotionLibrary:
    return ResidentMotionLibrary(self, joint_names, body_names, device)


@dataclass(kw_only=True)
class WindowedMotionLibraryCfg(MotionLibraryCfg):
  window_size: int | None = None
  """Clips resident per pool. None: one per env (as in HOVER/PHC)."""

  refresh_interval_s: float = 1000.0
  """Simulated seconds between pool refreshes (counted from the moment the previous
  pool went live)."""

  background_load: bool = True
  """Read the next pool's files in a background thread (training keeps stepping on
  the current pool meanwhile). False: load synchronously when the refresh is due."""

  def build(self, joint_names, body_names, device, num_envs) -> WindowedMotionLibrary:
    return WindowedMotionLibrary(self, joint_names, body_names, device, num_envs)


##
# Loading.
##


def discover_motion_files(cfg: MotionLibraryCfg) -> tuple[list[str], np.ndarray]:
  """Return (file paths, per-file weights) of the catalog described by ``cfg``."""
  weights = None
  if cfg.selection_file is not None:
    with open(cfg.selection_file) as f:
      selection = yaml.safe_load(f)
    files = [os.path.join(cfg.path, p) for p in selection["selected_files"]]
    weights = selection.get("motion_weights")
  else:
    files = []
    for root, _, names in os.walk(cfg.path, followlinks=True):
      files.extend(os.path.join(root, n) for n in names if n.endswith(".npz"))
    files.sort()
  if not files:
    raise FileNotFoundError(
      f"No motion files (*.npz) found for path={cfg.path!r}. Motion data is not part of the"
      " repository -- see ics_rl_lab/motion_lib/paths.py ($ICS_MOTION_DATA)."
    )
  weights = np.ones(len(files)) if weights is None else np.asarray(weights, dtype=np.float64)
  if len(weights) != len(files):
    raise ValueError(f"{len(files)} motion files but {len(weights)} motion weights")
  return files, weights


class _ClipReader:
  """Reads one clip into a packed [T, D] float array, joints/bodies in the requested order."""

  def __init__(self, joint_names: Sequence[str], body_names: Sequence[str]):
    self.joint_names = list(joint_names)
    self.body_names = list(body_names)
    self.num_joints = len(self.joint_names)
    self.num_bodies = len(self.body_names)
    j, b = self.num_joints, self.num_bodies
    widths = (j, j, 3 * b, 4 * b, 3 * b, 3 * b)
    offsets = np.cumsum((0,) + widths)
    self.slices = {f: slice(int(offsets[i]), int(offsets[i + 1])) for i, f in enumerate(_FIELDS)}
    self.width = int(offsets[-1])
    self._index_cache: dict[tuple, tuple[list[int], list[int]]] = {}

  def _indices(self, file_joints: list[str], file_bodies: list[str], path: str):
    key = (tuple(file_joints), tuple(file_bodies))
    if key not in self._index_cache:
      missing = [n for n in self.joint_names if n not in file_joints]
      missing += [n for n in self.body_names if n not in file_bodies]
      if missing:
        raise KeyError(f"{path}: names {missing} not in the clip's joint_names/body_names")
      self._index_cache[key] = (
        [file_joints.index(n) for n in self.joint_names],
        [file_bodies.index(n) for n in self.body_names],
      )
    return self._index_cache[key]

  def __call__(self, path: str) -> tuple[np.ndarray, float]:
    raw = np.load(path, allow_pickle=True)
    fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    joint_ids, body_ids = self._indices(list(raw["joint_names"]), list(raw["body_names"]), path)
    parts = [raw["joint_pos"][:, joint_ids], raw["joint_vel"][:, joint_ids]]
    for field in _FIELDS[2:]:
      arr = raw[field][:, body_ids]
      parts.append(arr.reshape(arr.shape[0], -1))
    return np.concatenate(parts, axis=1), fps


@dataclass
class _HostPool:
  """A loaded set of clips, packed on the host, ready to upload."""

  files: list[str]
  frames: np.ndarray  # [F, D]
  num_frames: np.ndarray  # [C]
  fps: np.ndarray  # [C]


def _load_host_pool(reader: _ClipReader, files: list[str], num_workers: int) -> _HostPool:
  with ThreadPoolExecutor(max(1, num_workers)) as pool:
    clips = list(pool.map(reader, files))
  num_frames = np.array([c.shape[0] for c, _ in clips], dtype=np.int64)
  if (num_frames < 2).any():
    bad = [f for f, n in zip(files, num_frames, strict=True) if n < 2]
    raise ValueError(f"Clips need at least 2 frames: {bad[:5]}")
  frames = np.concatenate([c for c, _ in clips], axis=0)
  fps = np.array([f for _, f in clips], dtype=np.float64)
  return _HostPool(files=list(files), frames=frames, num_frames=num_frames, fps=fps)


class _ClipPool:
  """Resident, packed clip storage + the interpolated frame gather."""

  def __init__(self, host: _HostPool, reader: _ClipReader, dtype: torch.dtype, storage_device, output_device):
    self.files = host.files
    self.reader = reader
    self.output_device = torch.device(output_device)
    self.storage_device = torch.device(storage_device)
    self.num_clips = len(host.files)
    self.data = torch.as_tensor(host.frames).to(device=self.storage_device, dtype=dtype)
    dev = self.output_device
    self.num_frames = torch.as_tensor(host.num_frames, device=dev)
    self.fps = torch.as_tensor(host.fps, device=dev, dtype=torch.float32)
    starts = np.concatenate(([0], np.cumsum(host.num_frames)[:-1]))
    self.starts = torch.as_tensor(starts, device=dev)
    self.lengths_s = (self.num_frames - 1).to(torch.float32) / self.fps

  @property
  def nbytes(self) -> int:
    return self.data.numel() * self.data.element_size()

  def get_frames(self, clip_ids: torch.Tensor, times: torch.Tensor) -> MotionFrames:
    n, k = times.shape
    fps = self.fps[clip_ids].unsqueeze(-1)
    last = (self.num_frames[clip_ids] - 1).unsqueeze(-1)
    frame_f = (times * fps).clamp(min=0.0)
    valid = frame_f <= last + 1e-4
    frame_f = torch.minimum(frame_f, last.to(frame_f.dtype))
    i0 = frame_f.floor().to(torch.long)
    i1 = torch.minimum(i0 + 1, last)
    blend = (frame_f - i0).unsqueeze(-1)
    start = self.starts[clip_ids].unsqueeze(-1)
    idx = torch.stack((start + i0, start + i1), dim=0).reshape(-1)
    rows = self.data[idx.to(self.storage_device)].to(self.output_device, torch.float32)
    rows = rows.reshape(2, n, k, self.reader.width)
    r0, r1 = rows[0], rows[1]
    lerp = r0 + blend * (r1 - r0)

    s, b = self.reader.slices, self.reader.num_bodies
    q0 = r0[..., s["body_quat_w"]].reshape(n, k, b, 4)
    q1 = r1[..., s["body_quat_w"]].reshape(n, k, b, 4)
    q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
    quat = q0 + blend.unsqueeze(-1) * (q1 - q0)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return MotionFrames(
      joint_pos=lerp[..., s["joint_pos"]],
      joint_vel=lerp[..., s["joint_vel"]],
      body_pos_w=lerp[..., s["body_pos_w"]].reshape(n, k, b, 3),
      body_quat_w=quat,
      body_lin_vel_w=lerp[..., s["body_lin_vel_w"]].reshape(n, k, b, 3),
      body_ang_vel_w=lerp[..., s["body_ang_vel_w"]].reshape(n, k, b, 3),
      valid=valid,
    )


##
# Libraries.
##


class MotionLibrary:
  """Interface the motion command talks to. Handles are opaque per-env clip ids."""

  def __init__(self, cfg: MotionLibraryCfg, joint_names, body_names, device: str):
    self.cfg = cfg
    self.device = torch.device(device)
    self.reader = _ClipReader(joint_names, body_names)
    self.dtype = _DTYPES[cfg.dtype]
    self.storage_device = self.device if cfg.storage_device == "output" else torch.device("cpu")
    self.catalog_files, weights = discover_motion_files(cfg)
    self.catalog_weights = torch.as_tensor(weights / weights.sum(), dtype=torch.float32, device=self.device)

  @property
  def num_catalog_clips(self) -> int:
    return len(self.catalog_files)

  def sample_clips(self, n: int) -> torch.Tensor:
    raise NotImplementedError

  def clip_lengths_s(self, handles: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  def get_frames(self, handles: torch.Tensor, times: torch.Tensor) -> MotionFrames:
    """Interpolated frames for clips ``handles`` [N] at clip times ``times`` [N, K] (s)."""
    raise NotImplementedError

  def clip_names(self, handles: torch.Tensor) -> list[str]:
    raise NotImplementedError

  def update(self, dt: float, active_handles: torch.Tensor) -> None:
    """Advance library bookkeeping by one env step. ``active_handles`` are the handles
    every env currently plays."""

  def resident_bytes(self) -> int:
    raise NotImplementedError


class ResidentMotionLibrary(MotionLibrary):
  """All catalog clips resident. Handle = catalog index."""

  def __init__(self, cfg: ResidentMotionLibraryCfg, joint_names, body_names, device: str):
    super().__init__(cfg, joint_names, body_names, device)
    print(f"[ResidentMotionLibrary] Loading {self.num_catalog_clips} clips from {cfg.path} ...")
    host = _load_host_pool(self.reader, self.catalog_files, cfg.num_load_workers)
    self.pool = _ClipPool(host, self.reader, self.dtype, self.storage_device, self.device)
    del host
    print(
      f"[ResidentMotionLibrary] {self.pool.num_clips} clips, {self.pool.data.shape[0]} frames,"
      f" {self.pool.nbytes / 2**30:.2f} GiB ({cfg.dtype}, {self.storage_device})"
    )

  def sample_clips(self, n: int) -> torch.Tensor:
    return torch.multinomial(self.catalog_weights, n, replacement=True)

  def clip_lengths_s(self, handles: torch.Tensor) -> torch.Tensor:
    return self.pool.lengths_s[handles]

  def get_frames(self, handles: torch.Tensor, times: torch.Tensor) -> MotionFrames:
    return self.pool.get_frames(handles, times)

  def clip_names(self, handles: torch.Tensor) -> list[str]:
    return [self.pool.files[i] for i in handles.tolist()]

  def resident_bytes(self) -> int:
    return self.pool.nbytes


class WindowedMotionLibrary(MotionLibrary):
  """A rolling pool of ``window_size`` clips out of the full catalog.

  Two pool slots (ping-pong). Handle = slot * window_size + clip index in that slot's
  pool. New samples always come from the live slot; the other slot is either empty,
  loading, or still referenced by envs that have not reset since the last swap.
  """

  def __init__(self, cfg: WindowedMotionLibraryCfg, joint_names, body_names, device: str, num_envs: int):
    super().__init__(cfg, joint_names, body_names, device)
    self.cfg: WindowedMotionLibraryCfg = cfg
    self.window_size = min(cfg.window_size or num_envs, self.num_catalog_clips)
    self._pools: list[_ClipPool | None] = [None, None]
    self._live = 0
    self._time_since_swap = 0.0
    self._executor = ThreadPoolExecutor(1) if cfg.background_load else None
    self._pending: Future[_HostPool] | None = None
    self.num_refreshes = 0
    self._install(self._load_host(self._draw_window()))

  # -- window selection / loading --

  def _draw_window(self) -> list[str]:
    ids = torch.multinomial(self.catalog_weights, self.window_size, replacement=False)
    return [self.catalog_files[i] for i in ids.tolist()]

  def _load_host(self, files: list[str]) -> _HostPool:
    return _load_host_pool(self.reader, files, self.cfg.num_load_workers)

  def _install(self, host: _HostPool) -> None:
    target = 0 if self._pools[0] is None and self._pools[1] is None else 1 - self._live
    assert self._pools[target] is None, "target pool slot still in use"
    self._pools[target] = _ClipPool(host, self.reader, self.dtype, self.storage_device, self.device)
    self._live = target
    self._time_since_swap = 0.0
    self.num_refreshes += 1
    print(
      f"[WindowedMotionLibrary] pool #{self.num_refreshes} live: {self.window_size} of"
      f" {self.num_catalog_clips} clips, {self._pools[target].nbytes / 2**30:.2f} GiB"
    )

  # -- interface --

  def _split(self, handles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return handles // self.window_size, handles % self.window_size

  def sample_clips(self, n: int) -> torch.Tensor:
    local = torch.randint(0, self.window_size, (n,), device=self.device)
    return local + self._live * self.window_size

  def clip_lengths_s(self, handles: torch.Tensor) -> torch.Tensor:
    slot, local = self._split(handles)
    out = torch.zeros(handles.shape, device=self.device)
    for s, pool in enumerate(self._pools):
      if pool is not None:
        out = torch.where(slot == s, pool.lengths_s[local.clamp(max=pool.num_clips - 1)], out)
    return out

  def get_frames(self, handles: torch.Tensor, times: torch.Tensor) -> MotionFrames:
    slot, local = self._split(handles)
    other = self._pools[1 - self._live]
    live_frames = self._pools[self._live].get_frames(local, times)
    if other is None:
      return live_frames
    # Both pools resident (transition): gather from both, select per env. No host sync.
    other_frames = other.get_frames(local.clamp(max=other.num_clips - 1), times)
    use_other = slot != self._live
    merged = {}
    for name in MotionFrames.__dataclass_fields__:
      a, b = getattr(live_frames, name), getattr(other_frames, name)
      mask = use_other.view(-1, *([1] * (a.dim() - 1)))
      merged[name] = torch.where(mask, b, a)
    return MotionFrames(**merged)

  def clip_names(self, handles: torch.Tensor) -> list[str]:
    slot, local = self._split(handles)
    return [self._pools[s].files[i] for s, i in zip(slot.tolist(), local.tolist(), strict=True)]

  def update(self, dt: float, active_handles: torch.Tensor) -> None:
    self._time_since_swap += dt
    other = 1 - self._live
    # Free the previous pool once no env plays one of its clips any more (host sync,
    # only while a previous pool is still resident).
    if self._pools[other] is not None and not bool((active_handles // self.window_size == other).any()):
      self._pools[other] = None
      if self.device.type == "cuda":
        torch.cuda.empty_cache()
    if self._time_since_swap < self.cfg.refresh_interval_s:
      return
    if self._executor is None:
      if self._pools[other] is None:
        self._install(self._load_host(self._draw_window()))
      return
    if self._pending is None:
      self._pending = self._executor.submit(self._load_host, self._draw_window())
    elif self._pending.done() and self._pools[other] is None:
      host = self._pending.result()
      self._pending = None
      self._install(host)

  def resident_bytes(self) -> int:
    return sum(p.nbytes for p in self._pools if p is not None)
