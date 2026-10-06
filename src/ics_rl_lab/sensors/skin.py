"""Skin (proximity taxel) sensor: one ray per skin cell, cast along the cell normal.

Built on mjlab's ``RayCastSensor``. Cell poses come from the skin driver's patch files
(``sp_extrinsics.tf_base_rsc`` @ ``sc_patch.sc_poses[i].htf``: position = ray origin,
local z-axis = ray direction), expressed in the patch's ``base_frame`` body.

Cells sit on the body surface and would start inside whatever they touch (e.g. the
ground under a loaded foot). So every ray origin is pulled back by ``offset`` along its
normal -- into the parent body -- and the parent body is excluded from the ray test
(MuJoCo ``bodyexclude``). Every other geom is visible (all geom groups: terrain, props, the
robot's other links -- visual and collision geoms alike, the nearest one wins). The reported distance is measured from the real cell position:
``clamp(hit_distance - offset, 0, max_distance)``, and ``max_distance`` when nothing
is hit (mjlab's RayCastSensor itself reports -1 for a miss).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np
import torch
from mjlab.sensor import RayCastSensor, RayCastSensorCfg


def load_skin_patch(path: str) -> tuple[str, np.ndarray]:
  """Return (base_frame, cell transforms [N, 4, 4] in the base frame) of one patch file."""
  with open(path) as f:
    data = json.load(f)  # the patch files are JSON (also valid YAML)
  base_frame = data["sp_extrinsics"]["base_frame"]
  tf_base_rsc = np.asarray(data["sp_extrinsics"]["tf_base_rsc"], dtype=np.float64)
  tf_cells = np.stack([np.asarray(c["htf"], dtype=np.float64) for c in data["sc_patch"]["sc_poses"]])
  return base_frame, tf_base_rsc @ tf_cells


@dataclass
class SkinPatchPatternCfg:
  """Ray pattern from skin patch files: origins at the cells, directions = cell z-axes."""

  cfg_files: Sequence[str] = ()
  offset: float = 0.0
  """Pull each origin back along its normal by this much (into the parent body)."""

  expected_base_frame: str | None = None
  """If set, every file's ``base_frame`` must equal it."""

  def load(self) -> tuple[np.ndarray, np.ndarray]:
    """Cell positions [N, 3] and normals [N, 3] in the base frame (no offset applied)."""
    positions, normals = [], []
    for path in self.cfg_files:
      base_frame, tf = load_skin_patch(path)
      if self.expected_base_frame is not None and base_frame != self.expected_base_frame:
        raise ValueError(f"{path}: base_frame {base_frame!r} != {self.expected_base_frame!r}")
      positions.append(tf[:, :3, 3])
      normals.append(tf[:, :3, 2])
    return np.concatenate(positions), np.concatenate(normals)

  def generate_rays(self, mj_model: mujoco.MjModel | None, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    del mj_model
    positions, normals = self.load()
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    origins = positions - self.offset * normals
    return (
      torch.as_tensor(origins, dtype=torch.float32, device=device),
      torch.as_tensor(normals, dtype=torch.float32, device=device),
    )


class SkinSensor(RayCastSensor):
  """RayCastSensor whose distances are re-referenced to the cell surface (see module doc).

  ``data.distances``: [B, N] in [0, max_distance]; ``max_distance`` = nothing in range.
  ``data.hit_pos_w`` / ``normals_w`` keep RayCastSensor's meaning (origin / 0 on a miss).
  """

  cfg: SkinSensorCfg

  def postprocess_rays(self) -> None:
    super().postprocess_rays()
    offset = self.cfg.pattern.offset
    raw = self._distances
    hit = raw >= 0.0
    self._distances = torch.where(
      hit, (raw - offset).clamp(0.0, self.cfg.skin_max_distance), torch.full_like(raw, self.cfg.skin_max_distance)
    )


@dataclass
class SkinSensorCfg(RayCastSensorCfg):
  pattern: SkinPatchPatternCfg = field(default_factory=SkinPatchPatternCfg)
  skin_max_distance: float = 0.1
  """Sensing range measured from the cell surface (source ``max_distance``)."""
  include_geom_groups: tuple[int, ...] | None = None
  """All groups: the skin sees everything except its parent body."""

  def build(self) -> SkinSensor:
    # Rays start ``offset`` behind the surface, so search that much further.
    return SkinSensor(replace(self, max_distance=self.skin_max_distance + self.pattern.offset))
