"""Hoverboard reward terms, on mjlab's Entity/ContactSensor API.

Reused rather than duplicated:
`track_lin_vel_xy_yaw_frame_exp`/`track_ang_vel_z_world_exp` (locomotion mdp) and
`ang_vel_xy_scaled_l2` (AMP mdp). All come in through `mdp/__init__.py`.
"""

from __future__ import annotations

import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.delayed_actuator import DelayedActuatorCfg
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from ics_rl_lab.tasks.locomotion.mdp.rewards import _applied_torque_by_joint_name, _resolve_joint_names


def dont_drift(
  env,
  command_name: str,
  cmd_threshold: float = 0.1,
  scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize base motion (L1) in the yaw frame, gated per-axis by command.

  vx is penalized only when the commanded vx is ~0, yaw rate only when the commanded
  yaw rate is ~0; vy is always penalized (the board cannot move sideways on purpose).
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)  # (N, >=3)

  vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3])
  v_x = vel_yaw[:, 0].abs()
  v_y = vel_yaw[:, 1].abs()
  w_z = asset.data.root_link_ang_vel_w[:, 2].abs()

  gate_x = (command[:, 0].abs() < cmd_threshold).float()
  gate_yaw = (command[:, 2].abs() < cmd_threshold).float()

  return scale[0] * v_x * gate_x + scale[1] * v_y + scale[2] * w_z * gate_yaw


def flat_orientation_scaled_l2(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  scale: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
  """Penalize non-flat base orientation, scale = (forward/backward tilt, left/right tilt)."""
  asset = env.scene[asset_cfg.name]
  g = asset.data.projected_gravity_b[:, :2]
  return g[:, 0] ** 2 * scale[0] + g[:, 1] ** 2 * scale[1]


def feet_contact_loss(env, sensor_name: str, threshold: float = 5.0) -> torch.Tensor:
  """Number of feet NOT in contact (multiply by a negative weight).

  A foot counts as in contact if its peak force over the sensor's substep history
  exceeds `threshold`, so momentary contact fluctuations are not penalized. The sensor
  must have `history_length > 0`.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force_history = sensor.data.force_history  # (N, F, H, 3)
  assert force_history is not None, f"{sensor_name} needs history_length > 0"
  max_force = torch.norm(force_history, dim=-1).max(dim=2)[0]  # (N, F)
  return torch.sum(max_force <= threshold, dim=-1).float()


def hoverboard_ang_vel_l2(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("hoverboard"),
  scale: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
  """Penalize hoverboard roll/pitch angular velocity in its body frame, scale = (roll, pitch)."""
  asset = env.scene[asset_cfg.name]
  w = asset.data.root_link_ang_vel_b[:, :2]
  return w[:, 0] ** 2 * scale[0] + w[:, 1] ** 2 * scale[1]


def motors_power_square(
  env,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  normalize_by_stiffness: bool = True,
  normalize_by_num_joints: bool = False,
) -> torch.Tensor:
  """Sum of squared per-joint mechanical power (tau * qvel), optionally divided by the
  joint's position-gain stiffness first.

  Uses the *nominal* stiffness from each actuator config, not the live (possibly
  `randomize_actuator_gains`-scaled) one, i.e. at most a 0.8-1.2x difference per joint.
  """
  asset = env.scene[asset_cfg.name]
  joint_names = _resolve_joint_names(asset, asset_cfg.joint_ids)
  qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  power_j = _applied_torque_by_joint_name(asset, joint_names) * qvel
  if normalize_by_stiffness:
    stiffness = _nominal_stiffness(asset, joint_names, power_j.device)
    power_j = power_j / stiffness
  power = torch.sum(torch.square(power_j), dim=-1)
  if normalize_by_num_joints:
    power /= power_j.shape[-1]
  return power


def _nominal_stiffness(asset, joint_names: tuple[str, ...], device) -> torch.Tensor:
  cache = getattr(asset, "_ics_nominal_stiffness_cache", None)
  if cache is None:
    cache = {}
    for act in asset.actuators:
      act_cfg = act.cfg.base_cfg if isinstance(act.cfg, DelayedActuatorCfg) else act.cfg
      assert isinstance(act_cfg, BuiltinPositionActuatorCfg), "needs position actuators"
      for name in act.target_names:
        cache[name] = float(act_cfg.stiffness)
    asset._ics_nominal_stiffness_cache = cache
  return torch.tensor([cache[n] for n in joint_names], device=device)
