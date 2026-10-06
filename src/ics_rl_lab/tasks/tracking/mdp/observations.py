"""Tracking observations.

The reference terms are recomputed against the robot's current state every step and
return (N, D).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
  transform_points,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg

  from .commands import MotionCommand


def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
  """Quaternion (w, x, y, z) -> rotated x-axis (tangent) and z-axis (normal), (..., 6).
  As in ProtoMotions."""
  ref_tan = torch.zeros_like(q[..., :3])
  ref_tan[..., 0] = 1.0
  ref_norm = torch.zeros_like(q[..., :3])
  ref_norm[..., 2] = 1.0
  return torch.cat([quat_apply(q, ref_tan), quat_apply(q, ref_norm)], dim=-1)


def motion_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]


##
# Reference (command) terms.
##


def ref_joint_pos(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  return motion_command(env, command_name).joint_pos


def ref_joint_vel(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  return motion_command(env, command_name).joint_vel


def ref_base_pos_b(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Reference base position in the robot's root frame (PositionRefCommand, anchor_frame="robot")."""
  cmd = motion_command(env, command_name)
  inv_pos, inv_quat = subtract_frame_transforms(cmd.robot.data.root_link_pos_w, cmd.robot.data.root_link_quat_w)
  return transform_points(cmd.base_pos_w.unsqueeze(1), inv_pos, inv_quat).squeeze(1)


def ref_base_rot_tannorm_b(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Reference base orientation in the robot's root frame as tan-norm. Zeroed past the
  clip end."""
  cmd = motion_command(env, command_name)
  q = quat_mul(quat_inv(cmd.robot.data.root_link_quat_w), cmd.base_quat_w)
  return quat_to_tan_norm(q) * cmd.valid.unsqueeze(-1)


##
# Proprioception (critic).
##


def link_tannorm_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """Link orientations in the root frame as tan-norm, (N, L, 6)."""
  asset: Entity = env.scene[asset_cfg.name]
  link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
  root_inv = quat_inv(asset.data.root_link_quat_w)[:, None, :].expand(-1, link_quat_w.shape[1], -1)
  return quat_to_tan_norm(quat_mul(root_inv, link_quat_w))


class future_motion_sequence(ManagerTermBase):
  """Privileged, critic-only lookahead: reference base/joint state at ``step_offsets`` env
  steps ahead, relative to the robot's *current* root pose: (N, K, 16 + 2*J) = rel_pos_b(3) + rel_rot_tannorm(6) +
  rel_lin_vel_b(3) + rel_ang_vel_b(3) + joint_pos(J) + joint_vel(J) + validity(1).
  Frames past the clip end are held at the last frame, validity=0."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    offsets = cfg.params.get("step_offsets", tuple(range(5, 101, 5)))
    self._offsets_s = torch.tensor(offsets, device=env.device, dtype=torch.float32) * env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    step_offsets: tuple[int, ...] = tuple(range(5, 101, 5)),
  ) -> torch.Tensor:
    cmd = motion_command(env, command_name)
    frames = cmd.get_frames(self._offsets_s)
    root = cmd.root_index
    k = self._offsets_s.shape[0]
    robot = cmd.robot.data
    inv_pos, inv_quat = subtract_frame_transforms(robot.root_link_pos_w, robot.root_link_quat_w)
    rel_pos_b = transform_points(frames.body_pos_w[:, :, root], inv_pos, inv_quat)
    root_inv = quat_inv(robot.root_link_quat_w)[:, None, :].expand(-1, k, -1)
    rel_rot = quat_to_tan_norm(quat_mul(root_inv, frames.body_quat_w[:, :, root]))
    root_quat = robot.root_link_quat_w[:, None, :].expand(-1, k, -1)
    rel_lin_vel_b = quat_apply_inverse(root_quat, frames.body_lin_vel_w[:, :, root])
    rel_ang_vel_b = quat_apply_inverse(root_quat, frames.body_ang_vel_w[:, :, root])
    validity = frames.valid.unsqueeze(-1).to(rel_pos_b.dtype)
    return torch.cat(
      [rel_pos_b, rel_rot, rel_lin_vel_b, rel_ang_vel_b, frames.joint_pos, frames.joint_vel, validity], dim=-1
    )
