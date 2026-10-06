"""PM01 whole-body motion tracking on flat ground.

uv run scripts/train.py Ics-Tracking-Flat-PM01 --num-envs 4096
uv run scripts/play.py  Ics-Tracking-Flat-PM01 --num-envs 4 --checkpoint <path>
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ics_rl_lab.assets.pm01 import robot
from ics_rl_lab.assets.pm01.pm01_cfg import PM01_EFFORT_LIMITS, apply_actuator_delay, get_pm01_robot_cfg
from ics_rl_lab.motion_lib import (
  MotionLibraryCfg,
  ResidentMotionLibraryCfg,
  WindowedMotionLibraryCfg,
  motion_data_dir,
)
from ics_rl_lab.tasks.tracking.mdp import MotionCommandCfg
from ics_rl_lab.tasks.tracking.tracking_env_cfg import FEET_CONTACT, UNDESIRED_CONTACT, make_tracking_env_cfg

# The retargeted pm01 seed set (48k clips, ~8.3 GB of fp16 .npz), see motion_lib/paths.py.
SEED_MOTION_PATH = str(motion_data_dir("pm01", "seed_motion"))

ROOT_BODY = "base_link"
TORSO_BODY = "torso_yaw_link"
LINK_OF_INTERESTS = (
  ROOT_BODY,
  TORSO_BODY,
  "shoulder_roll_l_link",
  "shoulder_roll_r_link",
  "elbow_pitch_l_link",
  "elbow_pitch_r_link",
  "elbow_yaw_l_link",  # end-effector / hand-equivalent (no separate hand body)
  "elbow_yaw_r_link",
  "hip_roll_l_link",
  "hip_roll_r_link",
  "knee_pitch_l_link",
  "knee_pitch_r_link",
  "ankle_roll_l_link",
  "ankle_roll_r_link",
)
FOOT_LINKS = ("ankle_roll_l_link", "ankle_roll_r_link")
HAND_LINKS = ("elbow_yaw_l_link", "elbow_yaw_r_link")
EXTREMITY_LINKS = FOOT_LINKS + HAND_LINKS


def seed_motion_lib(windowed: bool) -> MotionLibraryCfg:
  if windowed:
    # One clip per env, refreshed every 1000 s.
    return WindowedMotionLibraryCfg(path=SEED_MOTION_PATH, window_size=None, refresh_interval_s=1000.0)
  return ResidentMotionLibraryCfg(path=SEED_MOTION_PATH)


def make_pm01_contact_sensors() -> tuple[ContactSensorCfg, ContactSensorCfg]:
  """Per-foot contact, and per-body contact of everything but feet and hands. Secondary is
  unrestricted (ground *and* self-contact)."""
  feet = ContactSensorCfg(
    name=FEET_CONTACT,
    primary=ContactMatch(mode="subtree", pattern=rf"^({'|'.join(FOOT_LINKS)})$", entity="robot"),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=3,
  )
  # Body mode: mjlab's `exclude` only filters matched root names, so a subtree sensor
  # rooted at base_link would still include the feet.
  undesired = ContactSensorCfg(
    name=UNDESIRED_CONTACT,
    primary=ContactMatch(
      mode="body", pattern=r".*", entity="robot", exclude=tuple(rf"^{n}$" for n in EXTREMITY_LINKS)
    ),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=3,
  )
  return feet, undesired


def pm01_tracking_env_cfg(play: bool = False, windowed: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_pm01_robot_cfg()}
  apply_actuator_delay(cfg)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + make_pm01_contact_sensors()

  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.actuator_names = tuple(f"^{n}$" for n in robot.POLICY_JOINTS)

  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_lib = seed_motion_lib(windowed)
  motion.body_names = LINK_OF_INTERESTS
  motion.root_body_name = ROOT_BODY
  # Reference state init anywhere in the first 90% of the clip.
  motion.start_range = (0.0, 0.9)

  critic = cfg.observations["critic"].terms
  critic["link_pos"].params["asset_cfg"].body_names = LINK_OF_INTERESTS
  critic["link_rot"].params["asset_cfg"].body_names = LINK_OF_INTERESTS

  cfg.events["physics_material"].params["asset_cfg"].geom_names = robot.FOOT_SOLE_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  cfg.rewards["feet_slide"].params["asset_cfg"].body_names = FOOT_LINKS
  cfg.rewards["dof_torque_limits"].params["effort_limits"] = PM01_EFFORT_LIMITS
  cfg.terminations["link_pos_too_far"].params["body_names"] = EXTREMITY_LINKS

  cfg.viewer.body_name = TORSO_BODY
  cfg.sim.mujoco.ccd_iterations = 50  # FULL_COLLISION self-collision, as locomotion

  if play:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion.start_range = (0.0, 0.0)
    motion.ghost_offset = (0.0, 1.5, 0.0)  # source visualizing_robot_offset
  return cfg


__all__ = ["pm01_tracking_env_cfg", "LINK_OF_INTERESTS", "EXTREMITY_LINKS", "SEED_MOTION_PATH"]
