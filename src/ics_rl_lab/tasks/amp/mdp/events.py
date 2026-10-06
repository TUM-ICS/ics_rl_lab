"""AMP-specific reset event (reference state initialization, RSI)."""

from __future__ import annotations

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform


def reset_expert_state(
    env,
    env_ids: torch.Tensor,
    expert_reset_prob: float = 0.5,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    dof_position_range: tuple[float, float] | None = None,
    dof_velocity_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motionloader_cfg: SceneEntityCfg = SceneEntityCfg("motionloader"),
) -> None:
    """Hybrid reset: reference-state init (RSI) from expert motion data for
    ``expert_reset_prob`` of resets, default-pose + randomization for the rest.

    - Expert envs: expert quat/vel/joint state + linear (x,y,z) position noise only.
      Their velocities stay the clip's: velocity noise here would make the reset state
      inconsistent with the motion (a root velocity the clip's pose and joint velocities
      don't match).
    - Random envs: default state + full pose/vel noise (incl. rotation) + joint
      position/velocity randomization.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[motionloader_cfg.name]

    device = env.device
    num_reset = len(env_ids)

    probs = torch.rand(num_reset, device=device)
    expert_mask = probs < expert_reset_prob
    random_mask = ~expert_mask

    default_root = asset.data.default_root_state[env_ids].clone()
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()

    root_pos = default_root[:, 0:3]
    root_quat = default_root[:, 3:7]
    root_lin_vel = default_root[:, 7:10]
    root_ang_vel = default_root[:, 10:13]

    if expert_mask.any():
        expert_states = sensor.data.get_state_for_reset(int(expert_mask.sum()))
        num_dof = asset.data.default_joint_pos.shape[1]
        root_quat[expert_mask] = expert_states[:, 0:4]
        root_lin_vel[expert_mask] = expert_states[:, 4:7]
        root_ang_vel[expert_mask] = expert_states[:, 7:10]
        joint_pos[expert_mask] = expert_states[:, 10 : 10 + num_dof]
        joint_vel[expert_mask] = expert_states[:, 10 + num_dof : 10 + 2 * num_dof]

    if random_mask.any():
        if dof_position_range is not None:
            random_jpos = joint_pos[random_mask].clone()
            random_jpos *= sample_uniform(*dof_position_range, random_jpos.shape, device)
            lims = asset.data.soft_joint_pos_limits[env_ids][random_mask][:, asset_cfg.joint_ids]
            joint_pos[random_mask] = random_jpos.clamp_(lims[..., 0], lims[..., 1])
        if dof_velocity_range is not None:
            random_jvel = joint_vel[random_mask].clone()
            # mjlab has no live soft_joint_vel_limits tensor (unlike soft_joint_pos_limits);
            # the pm01 config runs this at (0.0, 0.0) anyway, so a clamp would never
            # activate in practice -- scale only.
            random_jvel *= sample_uniform(*dof_velocity_range, random_jvel.shape, device)
            joint_vel[random_mask] = random_jvel

    root_pos = root_pos + env.scene.env_origins[env_ids]

    if pose_range is not None:
        pos_keys, rot_keys = ("x", "y", "z"), ("roll", "pitch", "yaw")
        pos_ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in pos_keys], device=device)
        rot_ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in rot_keys], device=device)
        rand_pos = sample_uniform(pos_ranges[:, 0], pos_ranges[:, 1], (num_reset, 3), device)
        rand_rot = sample_uniform(rot_ranges[:, 0], rot_ranges[:, 1], (num_reset, 3), device)

        # Linear position noise applies to everyone; rotation noise only to random envs
        # (expert envs keep the expert's own orientation).
        root_pos = root_pos + rand_pos
        if random_mask.any():
            rot_delta = quat_from_euler_xyz(
                rand_rot[random_mask, 0], rand_rot[random_mask, 1], rand_rot[random_mask, 2]
            )
            root_quat[random_mask] = quat_mul(root_quat[random_mask], rot_delta)

    if velocity_range is not None:
        lin_keys, ang_keys = ("x", "y", "z"), ("roll", "pitch", "yaw")
        lin_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in lin_keys], device=device)
        ang_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in ang_keys], device=device)
        rand_lin = sample_uniform(lin_ranges[:, 0], lin_ranges[:, 1], (num_reset, 3), device)
        rand_ang = sample_uniform(ang_ranges[:, 0], ang_ranges[:, 1], (num_reset, 3), device)

        if random_mask.any():
            root_lin_vel[random_mask] = root_lin_vel[random_mask] + rand_lin[random_mask]
            root_ang_vel[random_mask] = root_ang_vel[random_mask] + rand_ang[random_mask]

    asset.write_root_state_to_sim(
        torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
    )
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=device)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids)
