"""Hoverboard event terms. Pushes use mjlab's stock velocity push.
"""

from __future__ import annotations

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform


def reset_hoverboard_root_state(
  env,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  robot_offset_pos: tuple[float, float, float] | list[float],
  robot_offset_range: dict[str, tuple[float, float]] | None = None,
  robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  hoverboard_asset_cfg: SceneEntityCfg = SceneEntityCfg("hoverboard"),
) -> None:
  """Reset robot and hoverboard co-located, with shared planar pose and velocity randomization.

  Both assets are placed at the same sampled XY position (default + env origin + random
  offset). Each keeps its own Z from init_state. The sampled yaw delta is applied on top of
  each asset's default orientation, and `robot_offset_pos` (hoverboard frame) is rotated by
  it. The same planar velocity (vx, vy, omega_z) is written to both, so the robot starts on
  the hoverboard with matching initial motion.

  pose_range keys: ``x``, ``y``, ``yaw``. velocity_range keys: ``x``, ``y``, ``yaw``.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  robot = env.scene[robot_asset_cfg.name]
  hoverboard = env.scene[hoverboard_asset_cfg.name]
  n = len(env_ids)
  dev = env.device

  robot_default = robot.data.default_root_state[env_ids].clone()
  hoverboard_default = hoverboard.data.default_root_state[env_ids].clone()
  env_origins = env.scene.env_origins[env_ids]

  # Shared planar pose offset (x, y, yaw).
  pose_ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "yaw"]], device=dev)
  pose_samples = sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (n, 3), device=dev)
  xy_offset = pose_samples[:, 0:2]
  yaw = pose_samples[:, 2]
  zeros = torch.zeros(n, device=dev)
  yaw_delta = quat_from_euler_xyz(zeros, zeros, yaw)

  # Hoverboard: default XY + env origin + random XY, own Z from init_state.
  hoverboard_base_xy = hoverboard_default[:, 0:2] + env_origins[:, 0:2] + xy_offset
  hoverboard_positions = torch.cat([hoverboard_base_xy, hoverboard_default[:, 2:3] + env_origins[:, 2:3]], dim=-1)
  hoverboard_orientations = quat_mul(hoverboard_default[:, 3:7], yaw_delta)

  # Rotate robot_offset_pos (hoverboard frame) into world frame using the sampled yaw.
  off = torch.as_tensor(robot_offset_pos, dtype=torch.float32, device=dev).unsqueeze(0).expand(n, -1).clone()
  if robot_offset_range is not None:
    for i, k in enumerate(["x", "y", "z"]):
      if k in robot_offset_range:
        lo, hi = robot_offset_range[k]
        off[:, i] += torch.empty(n, device=dev).uniform_(lo, hi)
  cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
  rotated_offset_xy = torch.stack(
    [cos_yaw * off[:, 0] - sin_yaw * off[:, 1], sin_yaw * off[:, 0] + cos_yaw * off[:, 1]], dim=-1
  )

  # Robot: hoverboard XY + rotated offset, own Z from init_state (+ offset z).
  robot_positions = torch.cat(
    [hoverboard_base_xy + rotated_offset_xy, robot_default[:, 2:3] + env_origins[:, 2:3] + off[:, 2:3]], dim=-1
  )
  robot_orientations = quat_mul(robot_default[:, 3:7], yaw_delta)

  # Shared planar velocity (vx, vy, omega_z); vz and roll/pitch rates stay 0.
  vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "yaw"]], device=dev)
  vel_samples = sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 3), device=dev)
  planar_vel = torch.zeros(n, 6, device=dev)
  planar_vel[:, 0:2] = vel_samples[:, 0:2]
  planar_vel[:, 5] = vel_samples[:, 2]

  # Written as the root *link* velocity like mjlab's own reset events (mjlab 1.2.0's
  # write_root_com_velocity_to_sim crashes on an un-batched body_ipos). Link and COM
  # velocity differ by omega x com_offset only, and the configured velocity ranges are
  # all zero.
  robot.write_root_link_pose_to_sim(torch.cat([robot_positions, robot_orientations], dim=-1), env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(planar_vel, env_ids=env_ids)
  hoverboard.write_root_link_pose_to_sim(
    torch.cat([hoverboard_positions, hoverboard_orientations], dim=-1), env_ids=env_ids
  )
  hoverboard.write_root_link_velocity_to_sim(planar_vel, env_ids=env_ids)
