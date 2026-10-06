"""Tienkung Ultra AMP (walk and run).

Same skeleton (`make_amp_env_cfg`) and run-variant changes as PM01; only the robot wiring
and a few Ultra-specific overrides differ.

uv run scripts/train.py Ics-AMP-Walk-Ultra --num-envs 4096
"""

from __future__ import annotations

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ics_rl_lab.assets.ultra.ultra_cfg import (
  ANKLE_BODIES,
  FOOT_BOX_GEOMS,
  ROOT_BODY,
  ULTRA_ACTION_SCALE,
  ULTRA_EFFORT_LIMITS,
  get_ultra_robot_cfg,
)
from ics_rl_lab.motion_lib import MotionLoaderSensorCfg, motion_data_dir
from ics_rl_lab.tasks.amp import mdp
from ics_rl_lab.tasks.amp.amp_env_cfg import MOTIONLOADER, make_amp_env_cfg
from ics_rl_lab.tasks.amp.config.pm01.env_cfgs import (
  AMP_OBS_NAMES,
  RUN_MOTION_NAMES,
  RUN_MOTION_WEIGHTS,
  RUN_VELOCITY_COMMAND,
)
from ics_rl_lab.tasks.locomotion.locomotion_env_cfg import (
  FEET_GROUND_CONTACT,
  UNDESIRED_CONTACT,
  make_feet_contact_sensors,
)

MOTION_DIR = motion_data_dir("ultra", "amp")

AMP_BODY_NAMES = ("ankle_roll_l_link", "ankle_roll_r_link", "elbow_pitch_l_link", "elbow_pitch_r_link")
# Source order; ankles excluded, as for PM01. 12 of Ultra's 16 joints.
AMP_JOINT_NAMES = (
  "hip_roll_l_joint", "hip_roll_r_joint",
  "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
  "hip_yaw_l_joint", "hip_yaw_r_joint",
  "elbow_pitch_l_joint", "elbow_pitch_r_joint",
  "hip_pitch_l_joint", "hip_pitch_r_joint",
  "knee_pitch_l_joint", "knee_pitch_r_joint",
)


def make_motionloader_cfg(motion_names: tuple[str, ...], motion_weights: tuple[float, ...]) -> MotionLoaderSensorCfg:
  return MotionLoaderSensorCfg(
    name=MOTIONLOADER,
    entity_name="robot",
    motion_directory=str(MOTION_DIR),
    motion_names=motion_names,
    motion_weights=motion_weights,
    root_body_name=ROOT_BODY,
    selected_obs_names=AMP_OBS_NAMES,
    selected_body_names=AMP_BODY_NAMES,
    selected_joint_names=AMP_JOINT_NAMES,
  )


def make_walk_motionloader_cfg() -> MotionLoaderSensorCfg:
  return make_motionloader_cfg(
    ("humanoid_standing", "humanoid_long_walk1", "humanoid_long_walk1_mirror"), (1.0, 2.0, 2.0)
  )


def apply_ultra_robot(cfg: ManagerBasedRlEnvCfg, motionloader_cfg: MotionLoaderSensorCfg) -> None:
  """Attach Ultra, its contact sensors and the motion loader; wire robot-specific names.

  No actuator delay.
  """
  cfg.scene.entities = {"robot": get_ultra_robot_cfg()}

  feet_ground, undesired = make_feet_contact_sensors(ANKLE_BODIES)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, undesired, motionloader_cfg)

  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.scale = ULTRA_ACTION_SCALE

  critic = cfg.observations["critic"].terms
  for term in ("feet_rel_pos", "feet_rel_ori", "feet_rel_lin_vel", "feet_rel_ang_vel"):
    critic[term].params["asset_cfg"].body_names = ANKLE_BODIES
  critic["feet_contact"].params["sensor_name"] = FEET_GROUND_CONTACT

  amp = cfg.observations["amp_policy"].terms
  amp["joint_pos"].params["asset_cfg"].joint_names = AMP_JOINT_NAMES
  amp["joint_vel"].params["asset_cfg"].joint_names = AMP_JOINT_NAMES
  for term in ("body_pos_b", "body_quat_b", "body_lin_vel_b", "body_ang_vel_b"):
    amp[term].params["asset_cfg"].body_names = AMP_BODY_NAMES

  cfg.events["physics_material"].params["asset_cfg"].geom_names = FOOT_BOX_GEOMS
  cfg.events["place_on_terrain"].params["asset_cfg"].geom_names = FOOT_BOX_GEOMS
  # Source: base mass/COM randomization on the pelvis for Ultra (the torso for PM01).
  cfg.events["add_base_mass"].params["asset_cfg"].body_names = (ROOT_BODY,)
  cfg.events["base_com"].params["asset_cfg"].body_names = (ROOT_BODY,)

  rewards = cfg.rewards
  for term in ("feet_slide", "feet_fly", "feet_stumble", "feet_air_time", "penalty_feet_force"):
    rewards[term].params["sensor_name"] = FEET_GROUND_CONTACT
  rewards["feet_slide"].params["asset_cfg"].body_names = ANKLE_BODIES
  rewards["penalty_feet_too_near"].params["asset_cfg"].body_names = ANKLE_BODIES
  rewards["undesired_contacts"].params["sensor_name"] = UNDESIRED_CONTACT
  rewards["dof_torque_limits"].params["effort_limits"] = ULTRA_EFFORT_LIMITS

  cfg.viewer.body_name = ROOT_BODY
  cfg.sim.mujoco.ccd_iterations = 50


def _apply_play(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.episode_length_s = int(1e9)
  cfg.observations["policy"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.events["reset_expert_state"].params["expert_reset_prob"] = 1.0


def ultra_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Ultra AMP flat-walk task."""
  cfg = make_amp_env_cfg()
  apply_ultra_robot(cfg, make_walk_motionloader_cfg())
  # Source: no posture terms for Ultra.
  for name in ("joint_deviation_hip", "joint_deviation_arms", "joint_deviation_legs"):
    cfg.rewards.pop(name)

  if play:
    _apply_play(cfg)
  return cfg


def ultra_amp_run_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Ultra AMP flat-run task: walk task + run motions, hull velocity command, running rewards."""
  cfg = ultra_amp_env_cfg(play=False)
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != MOTIONLOADER) + (
    make_motionloader_cfg(RUN_MOTION_NAMES, RUN_MOTION_WEIGHTS),
  )

  cfg.commands["base_velocity"] = copy.deepcopy(RUN_VELOCITY_COMMAND)

  cfg.rewards["penalty_action_rate_l2"].weight = -0.02
  # Ultra has no waist: only the pelvis' roll tilt (scale (0, 1)) and roll rate
  # (scale (1, 0)) are penalized, so the whole body may pitch forward when running.
  cfg.rewards["penalty_torso_orientation_l2"] = RewardTermCfg(
    func=mdp.body_orientation_l2,
    weight=-2.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(ROOT_BODY,)), "scale": (0.0, 1.0)},
  )
  cfg.rewards["penalty_ang_vel_xy_l2"] = RewardTermCfg(
    func=mdp.ang_vel_xy_scaled_l2, weight=-1.0, params={"scale": (1.0, 0.0)}
  )
  cfg.rewards["penalty_feet_force"] = RewardTermCfg(
    func=mdp.contact_force_threshold,
    weight=-1.0,
    params={"sensor_name": FEET_GROUND_CONTACT, "threshold": 3.0, "max_reward": 1.0},
  )
  for name in ("penalty_feet_too_near", "stand_still", "stand_still_vel", "feet_fly", "feet_air_time"):
    cfg.rewards.pop(name, None)

  if play:
    _apply_play(cfg)
  return cfg


__all__ = ["ultra_amp_env_cfg", "ultra_amp_run_env_cfg", "apply_ultra_robot"]
