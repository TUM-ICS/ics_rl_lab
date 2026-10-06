"""PM01 velocity-tracking locomotion.

uv run scripts/train.py Ics-Locomotion-Flat-PM01 --num-envs 4096
uv run scripts/play.py Ics-Locomotion-Flat-PM01 --num-envs 4 --checkpoint <path>
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import randomize_terrain
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ics_rl_lab.assets.pm01 import robot
from ics_rl_lab.tasks.locomotion.locomotion_env_cfg import (
  FEET_GROUND_CONTACT,
  UNDESIRED_CONTACT,
  make_feet_contact_sensors,
  make_locomotion_env_cfg,
)
from ics_rl_lab.assets.pm01.pm01_cfg import PM01_ACTION_SCALE, PM01_EFFORT_LIMITS, apply_actuator_delay, get_pm01_robot_cfg

TORSO_BODY = "torso_yaw_link"
ROOT_BODY = "base_link"
ANKLE_BODIES = ("ankle_roll_l_link", "ankle_roll_r_link")
FOOT_GEOMS = robot.FOOT_SOLE_GEOMS


def make_pm01_contact_sensors() -> tuple[ContactSensorCfg, ContactSensorCfg]:
  """Per-foot ground contact (with air-time tracking) and undesired (non-foot) contact."""
  return make_feet_contact_sensors(ANKLE_BODIES)


def apply_pm01_robot(cfg: ManagerBasedRlEnvCfg) -> None:
  """Attach PM01 and wire every robot-specific name into the generic terms."""
  cfg.scene.entities = {"robot": get_pm01_robot_cfg()}
  apply_actuator_delay(cfg)

  feet_ground, undesired = make_pm01_contact_sensors()
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, undesired)

  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.actuator_names = tuple(f"^{n}$" for n in robot.POLICY_JOINTS)
  joint_pos.scale = PM01_ACTION_SCALE

  cfg.observations["critic"].terms["feet_rel_pos"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.observations["critic"].terms["feet_rel_ori"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.observations["critic"].terms["feet_rel_lin_vel"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.observations["critic"].terms["feet_rel_ang_vel"].params["asset_cfg"].body_names = ANKLE_BODIES

  cfg.events["physics_material"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["place_on_terrain"].params["asset_cfg"].geom_names = robot.FOOT_BOX_GEOMS
  cfg.events["add_base_mass"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  cfg.rewards["penalty_feet_too_near"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.rewards["feet_slide"].params["asset_cfg"].body_names = ANKLE_BODIES
  cfg.rewards["dof_torque_limits"].params["effort_limits"] = PM01_EFFORT_LIMITS

  cfg.viewer.body_name = TORSO_BODY

  # Self-collision (FULL_COLLISION, mjlab default in get_pm01_robot_cfg) needed more
  # CCD iterations than the mjlab default to resolve leg-cylinder overlaps cleanly
  # The limbs are capsules now, which
  # may make this unnecessary (the box soles still go through CCD) -- not re-tested.
  cfg.sim.mujoco.ccd_iterations = 50


def pm01_locomotion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """PM01 velocity tracking on generated rough terrain."""
  cfg = make_locomotion_env_cfg()
  apply_pm01_robot(cfg)

  velocity = cfg.commands["base_velocity"]
  assert isinstance(velocity, UniformVelocityCommandCfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # First, so the reset terms (and place_on_terrain) see the new env origin.
    cfg.events = {"randomize_terrain": EventTermCfg(func=randomize_terrain, mode="reset", params={}), **cfg.events}
    velocity.ranges.lin_vel_x = (-0.5, 1.2)
    velocity.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg


__all__ = ["pm01_locomotion_env_cfg", "apply_pm01_robot", "TORSO_BODY", "ANKLE_BODIES"]
