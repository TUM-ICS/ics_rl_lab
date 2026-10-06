"""AMP-specific reward terms."""

from __future__ import annotations

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


def body_orientation_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
    """Penalize non-flat orientation of a specific body (e.g. the torso).

    ``scale`` = (tilt forward/backward weight, tilt right/left weight).
    """
    asset = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    g = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)[:, :2]
    return g[:, 0] ** 2 * scale[0] + g[:, 1] ** 2 * scale[1]


def ang_vel_xy_scaled_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
    """Penalize base roll/pitch angular velocity, independently scaled per axis."""
    asset = env.scene[asset_cfg.name]
    w = asset.data.root_link_ang_vel_b[:, :2]
    return w[:, 0] ** 2 * scale[0] + w[:, 1] ** 2 * scale[1]


def stand_still_vel(
    env,
    command_name: str,
    threshold: float = 0.05,
    joint_vel_scale: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base drift (xy / yaw velocity) and joint motion while the command is zero.

    Complements ``stand_still`` (joint-position deviation only), which lets the robot
    creep or sway slowly without cost.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = (torch.norm(command[:, :2], dim=1) < threshold) & (torch.abs(command[:, 2]) < threshold)
    motion = (
        torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
        + torch.square(asset.data.root_link_ang_vel_b[:, 2])
        + joint_vel_scale * torch.sum(torch.square(asset.data.joint_vel), dim=1)
    )
    return motion * standing
