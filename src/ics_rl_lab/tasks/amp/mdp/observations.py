"""AMP discriminator-input observation terms.

Only `amp_joint_pos`/
`amp_joint_vel` need dedicated functions here -- ABSOLUTE joint values (unlike
locomotion's `joint_pos_rel`/`joint_vel_rel`, which subtract the default pose),
since the `.npz` motion clips store absolute joint angles. The body-frame body
pose/twist terms (`body_pos_b`/`quat_b`/`lin_vel_b`/`ang_vel_b`) and
`root_pos_z`/`projected_gravity`/`root_lin_vel_b`/`root_ang_vel_b` are identical to
what locomotion already needs, so the env cfg wires those straight to
`locomotion.mdp` (`base_height`, stock `projected_gravity`/`base_lin_vel`/
`base_ang_vel`, and `link_pos_b`/`link_quat_b`/`link_lin_vel_b`/`link_ang_vel_b`)
rather than duplicating them.
"""

from __future__ import annotations

import torch
from mjlab.managers import SceneEntityCfg


def amp_joint_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute joint positions for the selected joints (matches raw motion-clip values)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def amp_joint_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities for the selected joints (matches raw motion-clip values)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]
