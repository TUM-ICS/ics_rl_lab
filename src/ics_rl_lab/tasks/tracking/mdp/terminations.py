"""Tracking terminations. Every frame of a clip is checked (no keyframe gating).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from .observations import motion_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def dataset_exhausted(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """The reference time went past the clip end."""
  return ~motion_command(env, command_name).valid


def pos_far_from_ref(
  env: ManagerBasedRlEnv, distance_threshold: float = 0.5, height_only: bool = False, command_name: str = "motion"
) -> torch.Tensor:
  cmd = motion_command(env, command_name)
  diff = cmd.robot.data.root_link_pos_w - cmd.base_pos_w
  distance = diff[:, 2].abs() if height_only else diff.norm(dim=-1)
  return distance > distance_threshold


def projected_gravity_far_from_ref(
  env: ManagerBasedRlEnv, projected_gravity_threshold: float = 0.5, z_only: bool = True, command_name: str = "motion"
) -> torch.Tensor:
  cmd = motion_command(env, command_name)
  gravity = cmd.robot.data.gravity_vec_w
  pg = quat_apply_inverse(cmd.robot.data.root_link_quat_w, gravity)
  ref_pg = quat_apply_inverse(cmd.base_quat_w, gravity)
  diff = (pg[:, 2] - ref_pg[:, 2]).abs() if z_only else (pg - ref_pg).norm(dim=-1)
  return diff > projected_gravity_threshold


def link_pos_far_from_ref(
  env: ManagerBasedRlEnv,
  body_names: tuple[str, ...],
  distance_threshold: float = 0.5,
  height_only: bool = False,
  command_name: str = "motion",
) -> torch.Tensor:
  """World-frame link position error (in_base_frame=False), max over ``body_names``."""
  cmd = motion_command(env, command_name)
  ids = [cmd.body_names.index(n) for n in body_names]
  diff = cmd.robot_body_pos_w[:, ids] - cmd.body_pos_w[:, ids]
  distance = diff[..., 2].abs() if height_only else diff.norm(dim=-1)
  return distance.max(dim=-1).values > distance_threshold
