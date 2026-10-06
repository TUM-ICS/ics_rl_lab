"""Custom locomotion reward terms, on mjlab's Entity/ContactSensor API.

Reward terms already covered by mjlab's stock `mjlab.envs.mdp` (is_terminated,
flat_orientation_l2, joint_pos_limits, action_rate_l2, joint_vel_l2, joint_acc_l2)
are used directly from there in `locomotion_env_cfg.py` and are not repeated here.
"""

from __future__ import annotations

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from mjlab.utils.string import resolve_field


def feet_air_time(
    env,
    command_name: str,
    vel_threshold: float,
    threshold: float,
    sensor_name: str,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds, one foot in the air at a time.

    Zero reward while the command is small (agent isn't supposed to take a step).
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    command = env.command_manager.get_command(command_name)
    reward *= torch.logical_or(
        torch.norm(command[:, :2], dim=1) > vel_threshold,
        torch.abs(command[:, 2]) > vel_threshold,
    )
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track xy linear velocity commands in the gravity-aligned (yaw) robot frame."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track yaw angular velocity commands in the world frame."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def stand_still(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Penalize moving when there is no velocity command."""
    asset = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    command = env.command_manager.get_command(command_name)
    return (
        (dof_error - offset)
        * (torch.norm(command[:, :2], dim=1) < threshold)
        * (torch.abs(command[:, 2]) < threshold)
    )


def dont_wait(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    asset = env.scene[asset_cfg.name]
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_link_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * (
        (lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float()
    )


def dont_fly(env, threshold: float, sensor_name: str) -> torch.Tensor:
    """Penalize flying (both feet off the ground at the same time)."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    is_contact = torch.norm(contact_sensor.data.force, dim=-1) > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def feet_stumble(env, sensor_name: str) -> torch.Tensor:
    """Penalize impacting the ground with large horizontal (vs. vertical) forces."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    force = contact_sensor.data.force
    return torch.any(
        torch.norm(force[..., :2], dim=-1) > 5 * torch.abs(force[..., 2]),
        dim=1,
    )


def feet_too_near(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    """Penalize the two feet getting too close together (xy distance)."""
    assert len(asset_cfg.body_ids) == 2
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    return (threshold - distance).clamp(min=0)


def _resolve_joint_names(asset, joint_ids) -> tuple[str, ...]:
    if isinstance(joint_ids, slice):
        return asset.joint_names[joint_ids]
    return tuple(asset.joint_names[i] for i in joint_ids)


def _applied_torque_by_joint_name(asset, joint_names: tuple[str, ...]) -> torch.Tensor:
    """Actuation force per named joint, shape (num_envs, len(joint_names)).

    mjlab has no live per-joint torque tensor on Entity.data (`joint_torques` raises
    NotImplementedError); `actuator_force` exists but is indexed by *actuator*
    order, not joint order. Each `Actuator.target_names` is already resolved in the
    exact order its targets were added to the model (matching `actuator_force`'s
    column order), so concatenating them across `asset.actuators` reconstructs the
    joint name for every column.
    """
    force = asset.data.actuator_force
    column_names = [name for act in asset.actuators for name in act.target_names]
    name_to_col = {name: i for i, name in enumerate(column_names)}
    cols = [name_to_col[n] for n in joint_names]
    return force[:, cols]


def energy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize mechanical power (|torque * joint velocity|)."""
    asset = env.scene[asset_cfg.name]
    joint_names = _resolve_joint_names(asset, asset_cfg.joint_ids)
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = _applied_torque_by_joint_name(asset, joint_names)
    return torch.norm(torch.abs(qfrc * qvel), dim=-1)


def contact_slide(
    env,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ang_vel_penalty: bool = False,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize a body sliding along the ground while in contact."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    force_history = contact_sensor.data.force_history
    if force_history is not None:
        contacts = force_history.norm(dim=-1).max(dim=2)[0] > threshold
    else:
        contacts = contact_sensor.data.force.norm(dim=-1) > threshold

    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    if ang_vel_penalty:
        body_ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :2]
        reward = reward + torch.sum(body_ang_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def contact_force_threshold(
    env,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.25,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """Penalize foot contact forces exceeding ``threshold`` times the robot's own weight.

    ``threshold``/``max_reward`` are multiples of body weight (mass * g), read from
    the sim's default mass field so it stays meaningful across mass-randomization.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    asset = env.scene[asset_cfg.name]

    body_mass = env.sim.get_default_field("body_mass")
    entity_body_ids = asset.indexing.body_ids
    total_mass = body_mass[entity_body_ids].sum()
    weight = total_mass * 9.81

    force_norm = torch.norm(contact_sensor.data.force, dim=-1).sum(dim=-1)
    out_of_limits = (force_norm - threshold * weight).clip(min=0) / weight
    return torch.clamp(out_of_limits, max=max_reward)


def applied_torque_limits_by_ratio(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    effort_limits: dict[str, float] | None = None,
    limit_ratio: float = 0.8,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """Penalize applied torque exceeding ``limit_ratio`` of the joint's effort limit.

    ``effort_limits`` maps joint-name regex -> N*m (mjlab has no live per-joint
    effort-limit tensor on Entity.data, since limits live on each actuator config instead -- resolved here the same way action scale is).
    """
    asset = env.scene[asset_cfg.name]
    joint_names = _resolve_joint_names(asset, asset_cfg.joint_ids)
    limits = torch.tensor(
        resolve_field(effort_limits or {}, joint_names, 0.0),
        dtype=torch.float32,
        device=env.device,
    )

    applied_torque = torch.abs(_applied_torque_by_joint_name(asset, joint_names))
    out_of_limits = (applied_torque - limits * limit_ratio).clip(min=0)
    out_of_limits_err = torch.sum(torch.square(out_of_limits), dim=-1)
    return torch.clamp(out_of_limits_err, max=max_reward)


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base roll/pitch angular velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base vertical linear velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def joint_deviation_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize absolute deviation from the default joint pose."""
    asset = env.scene[asset_cfg.name]
    joint_error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(joint_error), dim=1)


def undesired_contacts(env, sensor_name: str, threshold: float = 1.0) -> torch.Tensor:
    """Penalize (count of) contacts on bodies that should not touch anything."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    force_history = contact_sensor.data.force_history
    if force_history is not None:
        force_norm = force_history.norm(dim=-1).max(dim=2)[0]
    else:
        force_norm = contact_sensor.data.force.norm(dim=-1)
    return torch.sum((force_norm > threshold).float(), dim=1)
