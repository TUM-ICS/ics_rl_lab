"""Custom locomotion observation terms, on mjlab's Entity/ContactSensor API.

Everything else the locomotion task observes (base_ang_vel, projected_gravity,
joint_pos_rel, joint_vel_rel, last_action, generated_commands) is stock
`mjlab.envs.mdp` and is used directly in `locomotion_env_cfg.py` -- only the
critic-only, feet-frame terms below are custom.
"""

from __future__ import annotations

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    transform_points,
)


def link_pos_b(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
    flatten: bool = False,
) -> torch.Tensor:
    """Link positions in the robot base frame. Returns (num_envs, num_links, 3)."""
    asset = env.scene[asset_cfg.name]
    link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    if in_base_frame:
        pos_b, quat_b = subtract_frame_transforms(
            asset.data.root_link_pos_w, asset.data.root_link_quat_w
        )
        link_pos = transform_points(link_pos_w, pos_b, quat_b)
    else:
        link_pos = link_pos_w
    return link_pos.flatten(1, 2) if flatten else link_pos


def link_quat_b(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
    flatten: bool = False,
) -> torch.Tensor:
    """Link orientations in the robot base frame. Returns (num_envs, num_links, 4)."""
    asset = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    if in_base_frame:
        root_quat_inv = quat_inv(asset.data.root_link_quat_w)
        link_quat = quat_mul(
            root_quat_inv.unsqueeze(1).expand(-1, link_quat_w.shape[1], -1), link_quat_w
        )
    else:
        link_quat = link_quat_w
    return link_quat.flatten(1, 2) if flatten else link_quat


def link_lin_vel_b(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
    flatten: bool = False,
) -> torch.Tensor:
    """Link linear velocity in the robot base frame. Returns (num_envs, num_links, 3)."""
    asset = env.scene[asset_cfg.name]
    link_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    if in_base_frame:
        root_quat_inv = quat_inv(asset.data.root_link_quat_w)
        link_lin_vel = quat_apply(
            root_quat_inv.unsqueeze(1).expand(-1, link_lin_vel_w.shape[1], -1), link_lin_vel_w
        )
    else:
        link_lin_vel = link_lin_vel_w
    return link_lin_vel.flatten(1, 2) if flatten else link_lin_vel


def link_ang_vel_b(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_base_frame: bool = True,
    flatten: bool = False,
) -> torch.Tensor:
    """Link angular velocity in the robot base frame. Returns (num_envs, num_links, 3)."""
    asset = env.scene[asset_cfg.name]
    link_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    if in_base_frame:
        root_quat_inv = quat_inv(asset.data.root_link_quat_w)
        link_ang_vel = quat_apply(
            root_quat_inv.unsqueeze(1).expand(-1, link_ang_vel_w.shape[1], -1), link_ang_vel_w
        )
    else:
        link_ang_vel = link_ang_vel_w
    return link_ang_vel.flatten(1, 2) if flatten else link_ang_vel


def contact_force_norm(env, sensor_name: str) -> torch.Tensor:
    """Norm of each contact slot's net force. Returns (num_envs, num_slots)."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    return torch.norm(contact_sensor.data.force, dim=-1)


def base_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base height above the world origin (mjlab has no stock term for it). Returns
    (num_envs, 1)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]
