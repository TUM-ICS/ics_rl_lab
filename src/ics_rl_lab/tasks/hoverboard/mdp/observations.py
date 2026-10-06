"""Hoverboard observation terms, on mjlab's Entity/ContactSensor API.

`link_pos_b`/`link_quat_b` are reused from the locomotion task (via `mdp/__init__.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
  transform_points,
)

if TYPE_CHECKING:
  from ics_rl_lab.assets.hoverboard.hoverboard_driver import HoverboardActionTerm


def link_relative_pos(
  env,
  parent_asset_cfg: SceneEntityCfg = SceneEntityCfg("hoverboard"),
  child_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  flatten: bool = False,
) -> torch.Tensor:
  """Position of child links expressed in the parent's root frame.

  Returns (num_envs, num_child_bodies, 3), or (num_envs, num_child_bodies * 3) if flatten.
  """
  parent = env.scene[parent_asset_cfg.name]
  child = env.scene[child_asset_cfg.name]
  child_pos_w = child.data.body_link_pos_w[:, child_asset_cfg.body_ids]
  pos_inv, quat_inv_ = subtract_frame_transforms(parent.data.root_link_pos_w, parent.data.root_link_quat_w)
  child_pos_parent = transform_points(child_pos_w, pos_inv, quat_inv_)
  return child_pos_parent.flatten(1, 2) if flatten else child_pos_parent


def link_relative_quat(
  env,
  parent_asset_cfg: SceneEntityCfg = SceneEntityCfg("hoverboard"),
  child_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  flatten: bool = False,
) -> torch.Tensor:
  """Orientation of child links relative to the parent's root frame (q_parent^-1 * q_child).

  Returns (num_envs, num_child_bodies, 4), or (num_envs, num_child_bodies * 4) if flatten.
  """
  parent = env.scene[parent_asset_cfg.name]
  child = env.scene[child_asset_cfg.name]
  child_quat_w = child.data.body_link_quat_w[:, child_asset_cfg.body_ids]
  parent_quat_inv = quat_inv(parent.data.root_link_quat_w).unsqueeze(1).expand(-1, child_quat_w.shape[1], -1)
  child_quat_parent = quat_mul(parent_quat_inv, child_quat_w)
  return child_quat_parent.flatten(1, 2) if flatten else child_quat_parent


def contact_force(
  env,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  in_base_frame: bool = True,
  flatten: bool = False,
) -> torch.Tensor:
  """Net contact force per sensor slot, optionally rotated into the asset's base frame.

  mjlab's `reduce="netforce"` contact force is already in the world frame. Returns (num_envs, num_slots, 3) or flattened.
  """
  asset = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  net_force = sensor.data.force  # (N, S, 3), world frame
  if in_base_frame:
    root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, net_force.shape[1], -1)
    net_force = quat_apply_inverse(root_quat.reshape(-1, 4), net_force.reshape(-1, 3)).view(net_force.shape)
  return net_force.flatten(1, 2) if flatten else net_force


def feet_contact_binary(env, sensor_name: str, threshold: float = 5.0) -> torch.Tensor:
  """Binary foot contact state: 1.0 if contact force magnitude exceeds threshold, else 0.0.

  Maps to a physical binary contact switch on the foot plate. Returns (N, num_feet).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  return (torch.norm(sensor.data.force, dim=-1) > threshold).float()


def hoverboard_state(env, action_name: str) -> torch.Tensor:
  """Hoverboard LQR state [pitch, wheel_omega, pitch_dot] (true, undelayed)."""
  action: HoverboardActionTerm = env.action_manager.get_term(action_name)  # type: ignore[assignment]
  return action.state


def hoverboard_torque(env, action_name: str) -> torch.Tensor:
  """Hoverboard torque applied by the LQR controller, one column per board joint."""
  action: HoverboardActionTerm = env.action_manager.get_term(action_name)  # type: ignore[assignment]
  return action.torque
