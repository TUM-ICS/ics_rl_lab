"""Replay retargeted .pkl motions on PM01 / S2 / Ultra / G1 in mjlab and record the policy-rate reference states.

Input: retargeted .pkl clips (`fps`, `root_pos`, `root_rot` (xyzw), `dof_pos`, `dof_names`), as
written by `retarget.py`. Output: the .npz layout `AMPMotionLoader` reads. The replay is purely
kinematic -- set root + joint state, `sim.forward()`,
read the robot's link states -- so the recorded features use exactly the definitions the policy's
observation terms use (`body_link_*_w`, `projected_gravity_b`).

Notes:
  * root quaternions are normalized on load (slerp would otherwise preserve a non-unit norm and
    corrupt every root-frame field);
  * world positions are saved relative to the env origin;
  * body-frame fields are saved by default (the AMP task needs them).

`retarget.py` runs this automatically at the end. On its own:
  uv run scripts/motion/replay_motions.py --input data/motions/pm01/retargeted --output-dir data/motions/pm01/amp
  uv run scripts/motion/replay_motions.py --robot s2 --input data/motions/s2/retargeted --output-dir data/motions/s2/amp
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import mjlab
import numpy as np
import torch
import tyro
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_inv,
  quat_mul,
)
from scipy.ndimage import convolve1d
from tqdm import tqdm

from ics_rl_lab.assets.g1.g1_cfg import get_g1_robot_cfg
from ics_rl_lab.assets.pm01.pm01_cfg import get_pm01_robot_cfg
from ics_rl_lab.assets.s2.s2_cfg import get_s2_robot_cfg
from ics_rl_lab.assets.ultra.ultra_cfg import get_ultra_robot_cfg

ROBOTS = {"pm01": get_pm01_robot_cfg, "s2": get_s2_robot_cfg, "ultra": get_ultra_robot_cfg, "g1": get_g1_robot_cfg}


def _slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Batched shortest-path slerp between unit quaternions a, b (N, 4) at t (N,)."""
  dot = (a * b).sum(-1, keepdim=True)
  b = torch.where(dot < 0, -b, b)
  dot = dot.abs().clamp(max=1.0)
  theta = torch.acos(dot)
  sin_theta = torch.sin(theta)
  t = t.unsqueeze(-1)
  small = sin_theta < 1e-6
  w_a = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta.clamp(min=1e-6))
  w_b = torch.where(small, t, torch.sin(t * theta) / sin_theta.clamp(min=1e-6))
  q = w_a * a + w_b * b
  return q / q.norm(dim=-1, keepdim=True)


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
  """World-frame angular velocity of a (T, 4) quaternion sequence, central differences."""
  q_rel = quat_mul(rotations[2:], quat_conjugate(rotations[:-2]))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


class BatchMotion:
  """All clips of one batch, resampled to the output fps and padded to the longest clip
  (padding repeats each clip's own last frame and is never saved)."""

  def __init__(
    self,
    motion_files: list[Path],
    output_fps: float,
    device: str,
    frame_range: tuple[int, int] | None,
  ):
    self.output_dt = 1.0 / output_fps
    self.dof_names: list[str] | None = None
    fields: dict[str, list[torch.Tensor]] = {k: [] for k in ("pos", "rot", "dof", "lin_vel", "ang_vel", "dof_vel")}
    frames = []
    for motion_file in motion_files:
      with open(motion_file, "rb") as f:
        motion = pickle.load(f)
      if self.dof_names is None:
        self.dof_names = list(motion["dof_names"])
      elif list(motion["dof_names"]) != self.dof_names:
        raise ValueError(f"{motion_file}: dof_names differ from the rest of the batch.")

      sl = slice(frame_range[0] - 1, frame_range[1]) if frame_range is not None else slice(None)
      as_t = lambda x: torch.tensor(np.asarray(x)[sl], device=device, dtype=torch.float32)  # noqa: E731
      pos_in, dof_in = as_t(motion["root_pos"]), as_t(motion["dof_pos"])
      rot_in = as_t(motion["root_rot"])[:, [3, 0, 1, 2]]  # xyzw -> wxyz
      rot_in = rot_in / rot_in.norm(dim=-1, keepdim=True)

      input_dt = 1.0 / float(motion["fps"])
      n_in = dof_in.shape[0]
      duration = (n_in - 1) * input_dt
      times = torch.arange(0, duration, self.output_dt, device=device, dtype=torch.float32)
      phase = times / duration
      i0 = (phase * (n_in - 1)).floor().long()
      i1 = torch.clamp(i0 + 1, max=n_in - 1)
      blend = phase * (n_in - 1) - i0

      pos = torch.lerp(pos_in[i0], pos_in[i1], blend[:, None])
      dof = torch.lerp(dof_in[i0], dof_in[i1], blend[:, None])
      rot = _slerp(rot_in[i0], rot_in[i1], blend)
      fields["pos"].append(pos)
      fields["rot"].append(rot)
      fields["dof"].append(dof)
      fields["lin_vel"].append(torch.gradient(pos, spacing=self.output_dt, dim=0)[0])
      fields["dof_vel"].append(torch.gradient(dof, spacing=self.output_dt, dim=0)[0])
      fields["ang_vel"].append(_so3_derivative(rot, self.output_dt))
      frames.append(times.shape[0])

    self.output_frames = frames
    self.max_frames = max(frames)

    def pad(ts: list[torch.Tensor]) -> torch.Tensor:
      out = ts[0].new_empty(len(ts), self.max_frames, ts[0].shape[-1])
      for i, t in enumerate(ts):
        out[i, : t.shape[0]] = t
        out[i, t.shape[0] :] = t[-1]
      return out

    self.pos, self.rot, self.dof = pad(fields["pos"]), pad(fields["rot"]), pad(fields["dof"])
    self.lin_vel, self.ang_vel, self.dof_vel = pad(fields["lin_vel"]), pad(fields["ang_vel"]), pad(fields["dof_vel"])


def compute_statistics(data: np.ndarray, dt: float, window_dur: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
  """Min/max of a moving average over `window_dur`."""
  kernel_size = int(round(window_dur / dt))
  if kernel_size % 2 == 0:
    kernel_size += 1
  data_conv = convolve1d(data, np.ones(kernel_size) / kernel_size, axis=0, mode="constant", cval=0.0)
  trim = kernel_size // 2
  data_conv = data_conv[trim:-trim]
  return data_conv.min(axis=0), data_conv.max(axis=0)


def replay_batch(
  sim: Simulation,
  scene: Scene,
  motion_files: list[Path],
  output_fps: float,
  output_dir: Path,
  frame_range: tuple[int, int] | None,
  link_names: list[str] | None,
  save_body_frame: bool,
  fp16: bool,
) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
  num_envs = scene.num_envs
  num_real = len(motion_files)
  padded = list(motion_files) + [motion_files[-1]] * (num_envs - num_real)
  motion = BatchMotion(padded, output_fps, str(sim.device), frame_range)

  robot: Entity = scene["robot"]
  joint_ids = robot.find_joints(motion.dof_names, preserve_order=True)[0]
  if link_names is not None:
    body_ids, body_names = robot.find_bodies(link_names, preserve_order=True)
    if 0 not in body_ids:  # the root body is always kept (the loader reads the root from it)
      body_ids, body_names = [0, *body_ids], [robot.body_names[0], *body_names]
  else:
    body_ids, body_names = list(range(len(robot.body_names))), list(robot.body_names)

  keys = ["root_pos_z", "projected_gravity", "joint_pos", "joint_vel",
          "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
  if save_body_frame:
    keys += ["body_pos_b", "body_quat_b", "body_lin_vel_b", "body_ang_vel_b"]
  log: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
  root_vel_b_log: list[torch.Tensor] = []
  origins = scene.env_origins

  for step in range(motion.max_frames):
    root_state = torch.cat(
      [motion.pos[:, step] + origins, motion.rot[:, step], motion.lin_vel[:, step], motion.ang_vel[:, step]],
      dim=-1,
    )
    robot.write_root_state_to_sim(root_state)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = motion.dof[:, step]
    joint_vel[:, joint_ids] = motion.dof_vel[:, step]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    d = robot.data
    pos_w = d.body_link_pos_w[:, body_ids] - origins[:, None]
    quat_w = d.body_link_quat_w[:, body_ids]
    lin_w = d.body_link_lin_vel_w[:, body_ids]
    ang_w = d.body_link_ang_vel_w[:, body_ids]
    root_pos, root_quat = d.root_link_pos_w - origins, d.root_link_quat_w

    log["root_pos_z"].append(root_pos[:, 2:3])
    log["projected_gravity"].append(d.projected_gravity_b.clone())
    log["joint_pos"].append(d.joint_pos.clone())
    log["joint_vel"].append(d.joint_vel.clone())
    log["body_pos_w"].append(pos_w)
    log["body_quat_w"].append(quat_w.clone())
    log["body_lin_vel_w"].append(lin_w.clone())
    log["body_ang_vel_w"].append(ang_w.clone())
    root_q = root_quat[:, None].expand(-1, len(body_ids), -1)
    if save_body_frame:
      log["body_pos_b"].append(quat_apply_inverse(root_q, pos_w - root_pos[:, None]))
      log["body_quat_b"].append(quat_mul(quat_inv(root_q), quat_w))
      log["body_lin_vel_b"].append(quat_apply_inverse(root_q, lin_w))
      log["body_ang_vel_b"].append(quat_apply_inverse(root_q, ang_w))
    root_vel_b_log.append(
      torch.cat([quat_apply_inverse(root_quat, d.root_link_lin_vel_w), quat_apply_inverse(root_quat, d.root_link_ang_vel_w)], dim=-1)
    )

  stacked = {}
  for k in keys:
    arr = torch.stack(log[k]).cpu().numpy()  # (T, num_envs, ...)
    stacked[k] = arr.astype(np.float16) if fp16 else arr
  root_vel_b = torch.stack(root_vel_b_log).cpu().numpy()

  results = []
  for i in range(num_real):
    n = motion.output_frames[i]
    clip = {"fps": [output_fps], "body_names": body_names, "joint_names": list(robot.joint_names)}
    for k in keys:
      clip[k] = stacked[k][:n, i]
    window_dur = min(1.0, n / output_fps - 0.1)
    vmin, vmax = compute_statistics(root_vel_b[:n, i], dt=motion.output_dt, window_dur=window_dur)
    clip["min_root_velocity"], clip["max_root_velocity"] = vmin, vmax
    name = motion_files[i].stem
    np.savez(output_dir / f"{name}.npz", **clip)
    results.append((name, n / output_fps, vmin[[0, 1, 5]], vmax[[0, 1, 5]]))
  return results


def main(
  input: str,
  robot: Literal["pm01", "s2", "ultra", "g1"] = "pm01",
  output_dir: str | None = None,
  output_fps: float = 50.0,
  frame_range: tuple[int, int] | None = None,
  link_names: list[str] | None = None,
  save_body_frame: bool = True,
  fp16: bool = False,
  batch_size: int = 64,
  device: str = "cuda:0",
) -> None:
  """Replay retargeted .pkl motion(s) on a robot and save policy-rate reference .npz files.

  Args:
    input: A .pkl motion file or a directory of .pkl files.
    robot: Robot the motions were retargeted to.
    output_dir: Where to write the .npz files (default: next to the input).
    output_fps: Output frame rate; must match the policy rate (50 Hz for the AMP tasks).
    frame_range: START END input frames to use, 1-indexed, both inclusive (default: all).
    link_names: Only save these bodies (root always included). Default: all bodies.
    save_body_frame: Also save body_*_b (root-frame) fields; the AMP tasks need them.
    fp16: Save float arrays as float16.
    batch_size: Clips replayed in parallel, one per mjlab world.
    device: Torch/warp device.
  """
  input_path = Path(input)
  motion_files = sorted(input_path.glob("*.pkl")) if input_path.is_dir() else [input_path]
  if not motion_files:
    raise FileNotFoundError(f"No .pkl files found at {input_path}")
  out = Path(output_dir) if output_dir is not None else (input_path if input_path.is_dir() else input_path.parent)
  out.mkdir(parents=True, exist_ok=True)
  if device.startswith("cuda") and not torch.cuda.is_available():
    device = "cpu"

  num_envs = max(1, min(batch_size, len(motion_files)))
  scene = Scene(SceneCfg(num_envs=num_envs, env_spacing=2.0, entities={"robot": ROBOTS[robot]()}), device=device)
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=scene.compile(), device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  results = []
  for start in tqdm(range(0, len(motion_files), num_envs), desc="Replaying", unit="batch"):
    results += replay_batch(
      sim, scene, motion_files[start : start + num_envs], output_fps, out, frame_range,
      link_names, save_body_frame, fp16,
    )

  with open(out / "stats.txt", "w") as f:
    f.write("name, duration, vx min, vx max, vy min, vy max, omega min, omega max\n")
    for name, dur, vmin, vmax in results:
      f.write(
        f"{name}, {dur:.4f}, {vmin[0]:.4f}, {vmax[0]:.4f}, {vmin[1]:.4f}, {vmax[1]:.4f}, {vmin[2]:.4f}, {vmax[2]:.4f}\n"
      )
  print(f"[INFO] Wrote {len(results)} clips to {out}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
