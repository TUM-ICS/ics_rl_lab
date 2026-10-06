"""Unitree G1 AMP (flat-walk, flat-run).

Same skeleton (`make_amp_env_cfg`) and run-variant changes as PM01, with G1 names. Motion
clips are expected at `<motion data root>/g1/amp/` under the same clip names as the other
robots (generate them with `scripts/motion/retarget.py --robot g1`); the tasks register
without them, but building an env needs the files.

uv run scripts/train.py Ics-AMP-Walk-G1 --num-envs 4096
"""

from __future__ import annotations

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ics_rl_lab.assets.g1.g1_cfg import (
  ANKLE_BODIES,
  FOOT_GEOMS,
  G1_ACTION_SCALE,
  G1_EFFORT_LIMITS,
  ROOT_BODY,
  TORSO_BODY,
  apply_g1_actuator_delay,
  get_g1_robot_cfg,
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

MOTION_DIR = motion_data_dir("g1", "amp")

# Feet and hands, as for the other robots (G1's last arm link is the wrist).
AMP_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
# Every joint except the ankles (as for PM01/S2/Ultra): 25 of G1's 29.
AMP_JOINT_NAMES = (
  "left_hip_pitch_joint", "right_hip_pitch_joint",
  "left_hip_roll_joint", "right_hip_roll_joint",
  "left_hip_yaw_joint", "right_hip_yaw_joint",
  "left_knee_joint", "right_knee_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
  "left_shoulder_roll_joint", "right_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
  "left_elbow_joint", "right_elbow_joint",
  "left_wrist_roll_joint", "right_wrist_roll_joint",
  "left_wrist_pitch_joint", "right_wrist_pitch_joint",
  "left_wrist_yaw_joint", "right_wrist_yaw_joint",
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


def apply_g1_robot(cfg: ManagerBasedRlEnvCfg, motionloader_cfg: MotionLoaderSensorCfg) -> None:
  """Attach G1, its contact sensors and the motion loader; wire robot-specific names."""
  cfg.scene.entities = {"robot": get_g1_robot_cfg()}
  apply_g1_actuator_delay(cfg)

  feet_ground, undesired = make_feet_contact_sensors(ANKLE_BODIES)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, undesired, motionloader_cfg)

  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.scale = G1_ACTION_SCALE

  critic = cfg.observations["critic"].terms
  for term in ("feet_rel_pos", "feet_rel_ori", "feet_rel_lin_vel", "feet_rel_ang_vel"):
    critic[term].params["asset_cfg"].body_names = ANKLE_BODIES
  critic["feet_contact"].params["sensor_name"] = FEET_GROUND_CONTACT

  amp = cfg.observations["amp_policy"].terms
  amp["joint_pos"].params["asset_cfg"].joint_names = AMP_JOINT_NAMES
  amp["joint_vel"].params["asset_cfg"].joint_names = AMP_JOINT_NAMES
  for term in ("body_pos_b", "body_quat_b", "body_lin_vel_b", "body_ang_vel_b"):
    amp[term].params["asset_cfg"].body_names = AMP_BODY_NAMES

  cfg.events["physics_material"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["place_on_terrain"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["add_base_mass"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  rewards = cfg.rewards
  for term in ("feet_slide", "feet_fly", "feet_stumble", "feet_air_time", "penalty_feet_force"):
    rewards[term].params["sensor_name"] = FEET_GROUND_CONTACT
  rewards["feet_slide"].params["asset_cfg"].body_names = ANKLE_BODIES
  rewards["penalty_feet_too_near"].params["asset_cfg"].body_names = ANKLE_BODIES
  rewards["undesired_contacts"].params["sensor_name"] = UNDESIRED_CONTACT
  rewards["dof_torque_limits"].params["effort_limits"] = G1_EFFORT_LIMITS
  # The skeleton's posture groups use PM01 names; G1's knee is `*_knee_joint` and it has wrists.
  rewards["joint_deviation_arms"].params["asset_cfg"].joint_names = (
    r".*waist.*", r".*shoulder_roll.*", r".*shoulder_yaw.*", r".*wrist.*",
  )
  rewards["joint_deviation_legs"].params["asset_cfg"].joint_names = (r".*hip_pitch.*", r".*knee.*", r".*ankle.*")

  cfg.viewer.body_name = TORSO_BODY
  cfg.sim.mujoco.ccd_iterations = 50
  # 7 capsules per foot on the heightfield: up to ~10 contacts per capsule (88 in the model's
  # initial state, where the straight-legged XML pose sits in the gravel); the skeleton's 35
  # is sized for PM01's two boxes per foot.
  cfg.sim.nconmax = 100


def _apply_play(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.episode_length_s = int(1e9)
  cfg.observations["policy"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.events["reset_expert_state"].params["expert_reset_prob"] = 1.0


def g1_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 AMP flat-walk task."""
  cfg = make_amp_env_cfg()
  apply_g1_robot(cfg, make_walk_motionloader_cfg())
  if play:
    _apply_play(cfg)
  return cfg


def g1_amp_run_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 AMP flat-run task: PM01's run-variant changes with G1 names."""
  cfg = make_amp_env_cfg()
  apply_g1_robot(cfg, make_motionloader_cfg(RUN_MOTION_NAMES, RUN_MOTION_WEIGHTS))

  cfg.commands["base_velocity"] = copy.deepcopy(RUN_VELOCITY_COMMAND)

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
    _apply_play(cfg)
  return cfg


__all__ = ["g1_amp_env_cfg", "g1_amp_run_env_cfg", "apply_g1_robot"]
