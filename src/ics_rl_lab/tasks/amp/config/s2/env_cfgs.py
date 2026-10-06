"""S2 AMP (walk and run).

Same skeleton (`make_amp_env_cfg`) and run-variant changes as PM01; only the robot wiring
and a few S2-specific overrides differ.

uv run scripts/train.py Ics-AMP-Walk-S2 --num-envs 4096
"""

from __future__ import annotations

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ics_rl_lab.assets.s2.s2_cfg import (
  ANKLE_BODIES,
  FOOT_BOX_GEOMS,
  ROOT_BODY,
  S2_ACTION_SCALE,
  S2_EFFORT_LIMITS,
  TORSO_BODY,
  apply_s2_actuator_delay,
  get_s2_robot_cfg,
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

MOTION_DIR = motion_data_dir("s2", "amp")

AMP_BODY_NAMES = ("L_ankle_roll_link", "R_ankle_roll_link", "L_elbow_roll_link", "R_elbow_roll_link")

# Ankles excluded in both tasks (as for PM01).
_UPPER_JOINTS = (
  "waist_yaw_joint", "waist_pitch_joint",
  "head_yaw_joint", "head_pitch_joint",
  "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
  "L_elbow_roll_joint", "L_elbow_yaw_joint",
  "L_wrist_pitch_joint", "L_wrist_roll_joint",
  "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
  "R_elbow_roll_joint", "R_elbow_yaw_joint",
  "R_wrist_pitch_joint", "R_wrist_roll_joint",
)
_R_LEG = ("R_hip_roll_joint", "R_hip_yaw_joint", "R_hip_pitch_joint", "R_knee_pitch_joint")
_L_LEG = ("L_hip_roll_joint", "L_hip_yaw_joint", "L_hip_pitch_joint", "L_knee_pitch_joint")

AMP_JOINT_NAMES = _UPPER_JOINTS + _R_LEG + _L_LEG  # 26 of S2's 30 joints


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
    ("humanoid_standing", "humanoid_long_walk1", "humanoid_long_walk1_mirror"),
    (1.0, 2.0, 2.0),
  )


def apply_s2_robot(cfg: ManagerBasedRlEnvCfg, motionloader_cfg: MotionLoaderSensorCfg) -> None:
  """Attach S2, its contact sensors and the motion loader; wire robot-specific names."""
  cfg.scene.entities = {"robot": get_s2_robot_cfg()}
  apply_s2_actuator_delay(cfg)

  feet_ground, undesired = make_feet_contact_sensors(ANKLE_BODIES)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, undesired, motionloader_cfg)

  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.scale = S2_ACTION_SCALE

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
  # Source: base mass/COM randomization on base_link for S2 (the torso for PM01).
  cfg.events["add_base_mass"].params["asset_cfg"].body_names = (ROOT_BODY,)
  cfg.events["base_com"].params["asset_cfg"].body_names = (ROOT_BODY,)

  rewards = cfg.rewards
  for term in ("feet_slide", "feet_fly", "feet_stumble", "feet_air_time", "penalty_feet_force"):
    rewards[term].params["sensor_name"] = FEET_GROUND_CONTACT
  rewards["feet_slide"].params["asset_cfg"].body_names = ANKLE_BODIES
  rewards["penalty_feet_too_near"].params["asset_cfg"].body_names = ANKLE_BODIES
  rewards["undesired_contacts"].params["sensor_name"] = UNDESIRED_CONTACT
  rewards["dof_torque_limits"].params["effort_limits"] = S2_EFFORT_LIMITS

  cfg.viewer.body_name = TORSO_BODY
  cfg.sim.mujoco.ccd_iterations = 50


def _apply_play(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.episode_length_s = int(1e9)
  cfg.observations["policy"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.events["reset_expert_state"].params["expert_reset_prob"] = 1.0


def s2_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """S2 AMP flat-walk task."""
  cfg = make_amp_env_cfg()
  apply_s2_robot(cfg, make_walk_motionloader_cfg())

  cfg.rewards["feet_air_time"].weight = 1.0
  cfg.rewards["joint_deviation_hip"].params["asset_cfg"].joint_names = (
    r"[LR]_hip_yaw_joint", r"[LR]_hip_roll_joint", r"[LR]_shoulder_pitch_joint", r"[LR]_elbow_.*_joint",
  )
  cfg.rewards["joint_deviation_arms"].params["asset_cfg"].joint_names = (
    r"waist_.*", r"[LR]_shoulder_roll_joint", r"[LR]_shoulder_yaw_joint", r"[LR]_wrist_.*_joint", r"head_.*_joint",
  )
  cfg.rewards["joint_deviation_legs"].params["asset_cfg"].joint_names = (
    r"[LR]_hip_pitch_joint", r"[LR]_knee_pitch_joint", r"[LR]_ankle_.*_joint",
  )

  if play:
    _apply_play(cfg)
  return cfg


def s2_amp_run_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """S2 AMP flat-run task: walk task + run motions, hull velocity command, running rewards."""
  cfg = s2_amp_env_cfg(play=False)
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != MOTIONLOADER) + (
    make_motionloader_cfg(RUN_MOTION_NAMES, RUN_MOTION_WEIGHTS),
  )

  cfg.commands["base_velocity"] = copy.deepcopy(RUN_VELOCITY_COMMAND)

  cfg.rewards["penalty_action_rate_l2"].weight = -0.02
  # Torso (waist_pitch_link) may lean forward while the base stays level underneath it.
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
    _apply_play(cfg)
  return cfg


__all__ = ["s2_amp_env_cfg", "s2_amp_run_env_cfg", "apply_s2_robot"]
