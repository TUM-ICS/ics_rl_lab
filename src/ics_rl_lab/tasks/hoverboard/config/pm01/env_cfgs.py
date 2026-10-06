"""PM01 driving the hoverboard.

uv run scripts/train.py Ics-Hoverboard-Driving-PM01 --num-envs 4096
uv run scripts/play.py  Ics-Hoverboard-Driving-PM01 --num-envs 4 --checkpoint <path>
"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ics_rl_lab.assets.hoverboard.hoverboard_cfg import PLATE_TOP_Z
from ics_rl_lab.assets.pm01 import robot
from ics_rl_lab.assets.pm01.pm01_cfg import apply_actuator_delay, get_pm01_robot_cfg
from ics_rl_lab.tasks.hoverboard.hoverboard_env_cfg import (
  FEET_CONTACT,
  UNDESIRED_CONTACT,
  make_hoverboard_env_cfg,
)

TORSO_BODY = "torso_yaw_link"
ROOT_BODY = "base_link"
ANKLE_BODIES = ("ankle_roll_l_link", "ankle_roll_r_link")  # (left, right) -- plate order
FOOT_GEOMS = robot.FOOT_SOLE_GEOMS

#: Straight-legged stance on the plates (source: PM01_HB_CFG.init_state.joint_pos).
#: Our model puts the soles at base z - 0.821 in this pose and at y = +-0.1435, i.e.
#: exactly on the plates' nominal foot offset (0.1135 + 0.03).
HOVERBOARD_QPOS: dict[str, float] = {
  "hip_pitch_l_joint": -0.0,
  "hip_roll_l_joint": 0.025,
  "hip_yaw_l_joint": 0.0,
  "knee_pitch_l_joint": 0.0,
  "ankle_pitch_l_joint": -0.0,
  "ankle_roll_l_joint": -0.025,
  "hip_pitch_r_joint": -0.0,
  "hip_roll_r_joint": -0.025,
  "hip_yaw_r_joint": 0.0,
  "knee_pitch_r_joint": 0.0,
  "ankle_pitch_r_joint": -0.0,
  "ankle_roll_r_joint": 0.025,
  "waist_yaw_joint": 0.0,
  "head_yaw_joint": 0.0,
  "shoulder_pitch_l_joint": 0.24,
  "shoulder_roll_l_joint": 0.20,
  "shoulder_yaw_l_joint": 0.0,
  "elbow_pitch_l_joint": -0.68,
  "elbow_yaw_l_joint": 0.0,
  "shoulder_pitch_r_joint": 0.24,
  "shoulder_roll_r_joint": -0.20,
  "shoulder_yaw_r_joint": 0.0,
  "elbow_pitch_r_joint": -0.68,
  "elbow_yaw_r_joint": 0.0,
}


def _base_over_soles(qpos: dict[str, float]) -> float:
  """Height of base_link above the lowest sole point in the given pose (0.821 m here)."""
  model = robot.load_model()
  data = mujoco.MjData(model)
  for name, q in qpos.items():
    data.qpos[robot.joint_qpos_adr(model, name)] = q
  mujoco.mj_forward(model, data)
  return float(data.qpos[2]) - robot.lowest_sole_z(model, data)


#: The reset event adds `robot_offset_pos` z = 0.1 on top of this. Derived so the soles
#: start exactly on the plate tops: a robot starting even ~13 mm above the plates makes
#: the board run its balancer unloaded until the robot lands (plates slam past their
#: limits).
ROBOT_OFFSET_Z = 0.1
HOVERBOARD_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, PLATE_TOP_Z + _base_over_soles(HOVERBOARD_QPOS) - ROBOT_OFFSET_Z),
  joint_pos=HOVERBOARD_QPOS,
  joint_vel={".*": 0.0},
)


def make_pm01_contact_sensors() -> tuple[ContactSensorCfg, ContactSensorCfg]:
  """Per-foot contact against anything, and per-body contact for every non-ankle body.

  Feet are `.*ankle_roll.*`, undesired contacts `(?!.*ankle.*).*`. The undesired sensor
  uses mjlab's *body* mode, one slot per body, so `undesired_contacts` counts bodies. (A
  subtree match on the root can't express "minus the feet": mjlab's
  `exclude` filters the matched root names, not bodies inside the subtree.)
  """
  feet = ContactSensorCfg(
    name=FEET_CONTACT,
    primary=ContactMatch(mode="subtree", pattern=ANKLE_BODIES, entity="robot"),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=3,
  )
  undesired = ContactSensorCfg(
    name=UNDESIRED_CONTACT,
    primary=ContactMatch(mode="body", pattern=r".*", entity="robot", exclude=(r".*ankle.*",)),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=3,
  )
  return feet, undesired


def apply_pm01_robot(cfg: ManagerBasedRlEnvCfg) -> None:
  """Attach PM01 and wire every robot-specific name into the generic terms."""
  robot_cfg = get_pm01_robot_cfg()
  robot_cfg.init_state = HOVERBOARD_KEYFRAME
  cfg.scene.entities = {"robot": robot_cfg, **cfg.scene.entities}
  apply_actuator_delay(cfg)

  cfg.scene.sensors = (cfg.scene.sensors or ()) + make_pm01_contact_sensors()

  joint_pos = cfg.actions["joint_pos"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  joint_pos.actuator_names = tuple(f"^{n}$" for n in robot.POLICY_JOINTS)
  # Flat 0.25 rad, NOT PM01_ACTION_SCALE (the locomotion/AMP convention). With
  # PM01_ACTION_SCALE and the stiffer 10 Hz gains, one unit of action is 2-4x the
  # torque on ankles/hip-yaw, and early exploration kicked the feet off the
  # plates within ~3 steps (feet_outside_plate), teaching the policy to end episodes early.

  critic = cfg.observations["critic"].terms
  critic["hb_robot_rel_pos"].params["child_asset_cfg"].body_names = (ROOT_BODY, *ANKLE_BODIES)
  critic["hb_robot_rel_quat"].params["child_asset_cfg"].body_names = (ROOT_BODY, *ANKLE_BODIES)

  cfg.terminations["feet_outside_plate"].params["feet_asset_cfg"].body_names = ANKLE_BODIES
  cfg.events["reset_base"].params["robot_offset_pos"] = [0.0, 0.0, ROBOT_OFFSET_Z]

  cfg.events["physics_material"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["add_base_mass"].params["asset_cfg"].body_names = (TORSO_BODY,)
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  cfg.viewer.body_name = TORSO_BODY

  # Same CCD tuning as the other pm01 tasks (self-collision on).
  cfg.sim.mujoco.ccd_iterations = 50


def pm01_hoverboard_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_hoverboard_env_cfg()
  apply_pm01_robot(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


__all__ = ["pm01_hoverboard_env_cfg", "apply_pm01_robot", "HOVERBOARD_QPOS"]
