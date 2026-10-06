"""Tracking rewards.

Every term compares the robot with the *current* reference frame of the motion
command. Link terms are in the world / relative-world frame (the clips carry no
body-frame link data).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

from .observations import motion_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

Combine = Literal["prod", "sum", "mean_prod"]


def _gauss(square_err: torch.Tensor, std: float, combine: Combine) -> torch.Tensor:
  """engineai's combine_method over the last dim of a per-element squared error."""
  if combine == "prod":
    return torch.exp(-square_err.sum(-1) / std**2)
  if combine == "mean_prod":
    return torch.exp(-square_err.mean(-1) / std**2)
  return torch.exp(-square_err / std**2).sum(-1)


def _link_ids(cmd, body_names: tuple[str, ...] | None) -> slice | list[int]:
  if body_names is None:
    return slice(None)
  return [cmd.body_names.index(n) for n in body_names]


def tracking_base_position(env: ManagerBasedRlEnv, std: float = 0.1, command_name: str = "motion") -> torch.Tensor:
  cmd = motion_command(env, command_name)
  err = torch.sum(torch.square(cmd.robot.data.root_link_pos_w - cmd.base_pos_w), dim=-1)
  return torch.exp(-err / std**2) * cmd.valid


def tracking_base_rot(env: ManagerBasedRlEnv, std: float = 0.3, command_name: str = "motion") -> torch.Tensor:
  """difference_type="axis_angle": rotation angle of q_ref * conj(q)."""
  cmd = motion_command(env, command_name)
  err = quat_error_magnitude(cmd.base_quat_w, cmd.robot.data.root_link_quat_w)
  return torch.exp(-torch.square(err) / std**2)


def tracking_base_velocity(env: ManagerBasedRlEnv, std: float = 0.5, command_name: str = "motion") -> torch.Tensor:
  """Base linear velocity, each in its own base frame (in_base_frame=True)."""
  cmd = motion_command(env, command_name)
  ref = quat_apply_inverse(cmd.base_quat_w, cmd.base_lin_vel_w)
  err = torch.sum(torch.square(cmd.robot.data.root_link_lin_vel_b - ref), dim=-1)
  return torch.exp(-err / std**2)


def tracking_base_ang_vel(env: ManagerBasedRlEnv, std: float = 3.14, command_name: str = "motion") -> torch.Tensor:
  cmd = motion_command(env, command_name)
  ref = quat_apply_inverse(cmd.base_quat_w, cmd.base_ang_vel_w)
  err = torch.sum(torch.square(cmd.robot.data.root_link_ang_vel_b - ref), dim=-1)
  return torch.exp(-err / std**2)


def tracking_joint_pos(
  env: ManagerBasedRlEnv, std: float = 0.7, combine_method: Combine = "prod", command_name: str = "motion"
) -> torch.Tensor:
  cmd = motion_command(env, command_name)
  return _gauss(torch.square(cmd.robot.data.joint_pos - cmd.joint_pos), std, combine_method)


def tracking_joint_vel(
  env: ManagerBasedRlEnv, std: float = 10.0, combine_method: Combine = "prod", command_name: str = "motion"
) -> torch.Tensor:
  cmd = motion_command(env, command_name)
  return _gauss(torch.square(cmd.robot.data.joint_vel - cmd.joint_vel), std, combine_method)


def tracking_link_pos(
  env: ManagerBasedRlEnv,
  std: float = 0.1,
  in_relative_world_frame: bool = True,
  combine_method: Combine = "prod",
  body_names: tuple[str, ...] | None = None,
  command_name: str = "motion",
) -> torch.Tensor:
  cmd = motion_command(env, command_name)
  ref = cmd.relative_body_pose_w()[0] if in_relative_world_frame else cmd.body_pos_w
  err = torch.sum(torch.square(cmd.robot_body_pos_w - ref), dim=-1)[:, _link_ids(cmd, body_names)]
  return _gauss(err, std, combine_method)


def tracking_link_rot(
  env: ManagerBasedRlEnv,
  std: float = 0.4,
  in_relative_world_frame: bool = True,
  combine_method: Combine = "prod",
  body_names: tuple[str, ...] | None = None,
  command_name: str = "motion",
) -> torch.Tensor:
  cmd = motion_command(env, command_name)
  ref = cmd.relative_body_pose_w()[1] if in_relative_world_frame else cmd.body_quat_w
  robot = cmd.robot_body_quat_w
  err = quat_error_magnitude(robot.reshape(-1, 4), ref.reshape(-1, 4)).reshape(robot.shape[:2])
  return _gauss(torch.square(err[:, _link_ids(cmd, body_names)]), std, combine_method)


def tracking_link_lin_vel(
  env: ManagerBasedRlEnv,
  std: float = 0.4,
  combine_method: Combine = "prod",
  body_names: tuple[str, ...] | None = None,
  command_name: str = "motion",
) -> torch.Tensor:
  """World frame (in_base_frame=False)."""
  cmd = motion_command(env, command_name)
  err = torch.sum(torch.square(cmd.robot_body_lin_vel_w - cmd.body_lin_vel_w), dim=-1)
  return _gauss(err[:, _link_ids(cmd, body_names)], std, combine_method)


def tracking_link_ang_vel(
  env: ManagerBasedRlEnv,
  std: float = 0.4,
  combine_method: Combine = "prod",
  body_names: tuple[str, ...] | None = None,
  command_name: str = "motion",
) -> torch.Tensor:
  """World frame (in_base_frame=False)."""
  cmd = motion_command(env, command_name)
  err = torch.sum(torch.square(cmd.robot_body_ang_vel_w - cmd.body_ang_vel_w), dim=-1)
  return _gauss(err[:, _link_ids(cmd, body_names)], std, combine_method)
