"""PM01 AMP (walk and run).

uv run scripts/train.py Ics-AMP-Walk-PM01 --num-envs 4096
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ics_rl_lab.assets.pm01 import robot
from ics_rl_lab.tasks.amp import mdp
from ics_rl_lab.tasks.amp.amp_env_cfg import MOTIONLOADER, make_amp_env_cfg
from ics_rl_lab.tasks.amp.mdp.commands import HullVelocityCommandCfg
from ics_rl_lab.motion_lib import MotionLoaderSensorCfg, motion_data_dir
from ics_rl_lab.tasks.locomotion.locomotion_env_cfg import FEET_GROUND_CONTACT, UNDESIRED_CONTACT
from ics_rl_lab.assets.pm01.pm01_cfg import PM01_ACTION_SCALE, PM01_EFFORT_LIMITS, apply_actuator_delay, get_pm01_robot_cfg

TORSO_BODY = "torso_yaw_link"
ROOT_BODY = "base_link"
ANKLE_BODIES = ("ankle_roll_l_link", "ankle_roll_r_link")
FOOT_GEOMS = robot.FOOT_SOLE_GEOMS

MOTION_DIR = motion_data_dir("pm01", "amp")

# Excludes ankle_pitch/roll joints from the discriminator's joint features (the
# ankle *bodies* are still tracked via the body terms below), 20 of pm01's 24
# joints.
AMP_JOINT_NAMES = (
  "hip_pitch_l_joint", "hip_pitch_r_joint",
  "waist_yaw_joint",
  "hip_roll_l_joint", "hip_roll_r_joint",
  "head_yaw_joint",
  "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
  "hip_yaw_l_joint", "hip_yaw_r_joint",
  "shoulder_roll_l_joint", "shoulder_roll_r_joint",
  "knee_pitch_l_joint", "knee_pitch_r_joint",
  "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
  "elbow_pitch_l_joint", "elbow_pitch_r_joint",
  "elbow_yaw_l_joint", "elbow_yaw_r_joint",
)
AMP_BODY_NAMES = ("ankle_roll_l_link", "ankle_roll_r_link", "elbow_yaw_l_link", "elbow_yaw_r_link")

# Term declaration order in amp_env_cfg.make_amp_env_cfg()'s amp_policy group --
# must match AMPMotionLoader.obs_motions' column order exactly.
AMP_OBS_NAMES = (
  "root_pos_z", "projected_gravity", "joint_pos", "joint_vel",
  "root_lin_vel_b", "root_ang_vel_b", "body_pos_b", "body_quat_b", "body_lin_vel_b", "body_ang_vel_b",
)


def make_motionloader_cfg() -> MotionLoaderSensorCfg:
  return MotionLoaderSensorCfg(
    name=MOTIONLOADER,
    entity_name="robot",
    motion_directory=str(MOTION_DIR),
    motion_names=("humanoid_standing", "humanoid_long_walk1", "humanoid_long_walk1_mirror"),
    motion_weights=(1.0, 2.0, 2.0),
    root_body_name=ROOT_BODY,
    selected_obs_names=AMP_OBS_NAMES,
    selected_body_names=AMP_BODY_NAMES,
    selected_joint_names=AMP_JOINT_NAMES,
  )


def apply_pm01_robot(cfg: ManagerBasedRlEnvCfg, motionloader_cfg: MotionLoaderSensorCfg | None = None) -> None:
  """Attach PM01, its contact sensors and the motion loader; wire robot-specific names."""
  cfg.scene.entities = {"robot": get_pm01_robot_cfg()}
  apply_actuator_delay(cfg)

  from ics_rl_lab.tasks.locomotion.config.pm01.env_cfgs import make_pm01_contact_sensors

  feet_ground, undesired = make_pm01_contact_sensors()
  loader = motionloader_cfg or make_motionloader_cfg()
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, undesired, loader)

  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.actuator_names = tuple(f"^{n}$" for n in robot.POLICY_JOINTS)
  joint_pos.scale = PM01_ACTION_SCALE

  cfg.observations["critic"].terms["feet_rel_pos"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.observations["critic"].terms["feet_rel_ori"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.observations["critic"].terms["feet_rel_lin_vel"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.observations["critic"].terms["feet_rel_ang_vel"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.observations["critic"].terms["feet_contact"].params["sensor_name"] = FEET_GROUND_CONTACT

  cfg.observations["amp_policy"].terms["joint_pos"].params["asset_cfg"].joint_names = AMP_JOINT_NAMES
  cfg.observations["amp_policy"].terms["joint_vel"].params["asset_cfg"].joint_names = AMP_JOINT_NAMES
  for term in ("body_pos_b", "body_quat_b", "body_lin_vel_b", "body_ang_vel_b"):
    cfg.observations["amp_policy"].terms[term].params["asset_cfg"].body_names = AMP_BODY_NAMES

  cfg.events["physics_material"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["place_on_terrain"].params["asset_cfg"].geom_names = robot.FOOT_BOX_GEOMS
  cfg.events["add_base_mass"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  cfg.rewards["feet_slide"].params["sensor_name"] = FEET_GROUND_CONTACT
  cfg.rewards["feet_slide"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.rewards["feet_fly"].params["sensor_name"] = FEET_GROUND_CONTACT
  cfg.rewards["feet_stumble"].params["sensor_name"] = FEET_GROUND_CONTACT
  cfg.rewards["feet_air_time"].params["sensor_name"] = FEET_GROUND_CONTACT
  cfg.rewards["penalty_feet_too_near"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.rewards["penalty_feet_force"].params["sensor_name"] = FEET_GROUND_CONTACT
  cfg.rewards["undesired_contacts"].params["sensor_name"] = UNDESIRED_CONTACT
  cfg.rewards["dof_torque_limits"].params["effort_limits"] = PM01_EFFORT_LIMITS

  cfg.viewer.body_name = TORSO_BODY
  cfg.sim.mujoco.ccd_iterations = 50


def pm01_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """PM01 AMP (adversarial motion prior) flat-walk task."""
  cfg = make_amp_env_cfg()
  apply_pm01_robot(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events["reset_expert_state"].params["expert_reset_prob"] = 1.0

  return cfg


##
# Run variant -- same base env, larger (run) motion set, hull-bounded velocity
# command, and reduced reward set.
##

# 14 forward + 15 left + 15 right clips (forward12/13 are left out).
_RUN_FORWARD_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15)
_RUN_TURN_IDS = tuple(range(15))

RUN_MOTION_NAMES = (
  ("humanoid_standing",)
  + ("humanoid_long_walk1", "humanoid_long_walk1_mirror")
  + tuple(f"humanoid_sie_run_forward{i:02d}" for i in _RUN_FORWARD_IDS)
  + tuple(f"humanoid_sie_run_left{i:02d}" for i in _RUN_TURN_IDS)
  + tuple(f"humanoid_sie_run_right{i:02d}" for i in _RUN_TURN_IDS)
)
RUN_MOTION_WEIGHTS = (
  (1.0,)
  + (2.0, 2.0)
  + (10.0,) * len(_RUN_FORWARD_IDS)
  + (1.0,) * len(_RUN_TURN_IDS)
  + (1.0,) * len(_RUN_TURN_IDS)
)

RUN_VELOCITY_COMMAND = HullVelocityCommandCfg(
  resampling_time_range=(20.0, 20.0),
  asset_name="robot",
  rel_standing_envs=0.2,
  rel_heading_envs=1.0,
  heading_command=True,
  heading_control_stiffness=0.5,
  vy_bound_points=[[0.0, 0.3], [1.5, 0.3], [2.0, 0.2], [3.0, 0.1], [3.7, 0.05]],
  omega_bound_points=[[0.0, 1.5], [1.0, 1.0], [2.0, 1.0], [3.0, 0.6], [3.7, 0.6]],
  velocity_bins=[0.0, 1.5, 3.0, 4.5],
  heading=(-math.pi, math.pi),
)


def make_run_motionloader_cfg() -> MotionLoaderSensorCfg:
  return MotionLoaderSensorCfg(
    name=MOTIONLOADER,
    entity_name="robot",
    motion_directory=str(MOTION_DIR),
    motion_names=RUN_MOTION_NAMES,
    motion_weights=RUN_MOTION_WEIGHTS,
    root_body_name=ROOT_BODY,
    selected_obs_names=AMP_OBS_NAMES,
    selected_body_names=AMP_BODY_NAMES,
    selected_joint_names=AMP_JOINT_NAMES,
  )


def pm01_amp_run_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """PM01 AMP (adversarial motion prior) flat-run task."""
  cfg = make_amp_env_cfg()
  apply_pm01_robot(cfg, motionloader_cfg=make_run_motionloader_cfg())

  # Commands: hull-bounded velocity (vy/omega shrink as vx grows) instead of the
  # walk task's uniform-range command.
  cfg.commands["base_velocity"] = RUN_VELOCITY_COMMAND

  # Rewards: softer action-rate penalty, torso allowed to lean forward while the
  # base stays level underneath it, higher foot-force tolerance (running impacts
  # harder than walking), and postural/standing terms that don't apply to running
  # dropped entirely.
  cfg.rewards["penalty_action_rate_l2"].weight = -0.01
  cfg.rewards["penalty_torso_orientation_l2"] = RewardTermCfg(
    func=mdp.body_orientation_l2,
    weight=-2.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY,)), "scale": (1.0, 1.0)},
  )
  cfg.rewards["penalty_ang_vel_xy_l2"] = RewardTermCfg(
    func=mdp.ang_vel_xy_scaled_l2, weight=-1.0, params={"scale": (1.0, 1.0)}
  )
  cfg.rewards["penalty_feet_force"] = RewardTermCfg(
    func=mdp.contact_force_threshold,
    weight=-1.0,
    params={"sensor_name": FEET_GROUND_CONTACT, "threshold": 3.0, "max_reward": 1.0},
  )
  for name in (
    "penalty_feet_too_near", "joint_deviation_hip", "joint_deviation_arms",
    "joint_deviation_legs", "stand_still", "stand_still_vel", "feet_fly", "feet_air_time",
  ):
    cfg.rewards.pop(name, None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events["reset_expert_state"].params["expert_reset_prob"] = 1.0

  return cfg


__all__ = [
  "pm01_amp_env_cfg", "pm01_amp_run_env_cfg", "apply_pm01_robot", "AMP_JOINT_NAMES", "AMP_BODY_NAMES",
]
