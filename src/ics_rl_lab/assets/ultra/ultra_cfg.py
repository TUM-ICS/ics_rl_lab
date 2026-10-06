"""Tienkung Ultra as an mjlab `Entity`.

* MJCF: `ultra.xml`, converted from the robot's URDF (`ultra_simplified.urdf`) with MuJoCo's
  URDF compiler, same conventions as `s2.xml`.
* Gains, armature and effort limits: hand-set per joint. mjlab takes scalar gains per
  actuator config, so each joint type is its own config.
* No actuator delay.

Joint velocity limits are not enforced in simulation.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from ics_rl_lab.assets.robot_utils import effort_limits, make_articulation, standing_root_height

ULTRA_XML = Path(__file__).resolve().parent / "ultra.xml"
assert ULTRA_XML.exists(), ULTRA_XML


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ULTRA_XML))
  assets: dict[str, bytes] = {}
  update_assets(assets, ULTRA_XML.parent / "meshes", spec.meshdir)
  spec.assets = assets
  return spec


##
# Names.
##

ROOT_BODY = "pelvis"
ANKLE_BODIES = ("ankle_roll_l_link", "ankle_roll_r_link")
FOOT_SOLE_GEOMS = ("foot_l_sole_col", "foot_r_sole_col")
#: Every foot collision geom (sole + two toe boxes per foot) -- the lowest points of the robot.
FOOT_BOX_GEOMS = FOOT_SOLE_GEOMS + ("foot_l_toe_col", "foot_r_toe_col", "foot_l_tip_col", "foot_r_tip_col")

##
# Actuators (source values; effort N*m, stiffness N*m/rad, damping N*m*s/rad).
##


def _actuator(effort: float, stiffness: float, damping: float, armature: float, *targets: str):
  return BuiltinPositionActuatorCfg(
    target_names_expr=targets, stiffness=stiffness, damping=damping, effort_limit=effort, armature=armature
  )


ULTRA_ACTUATORS = (
  _actuator(235, 700, 10, 0.22468, r"^hip_roll_[lr]_joint$"),
  _actuator(300, 700, 10, 0.28329, r"^hip_pitch_[lr]_joint$", r"^knee_pitch_[lr]_joint$"),
  _actuator(150, 500, 5, 0.16773, r"^hip_yaw_[lr]_joint$"),
  _actuator(100, 30, 2.5, 2 * 0.013879, r"^ankle_pitch_[lr]_joint$"),
  _actuator(55, 16.8, 1.4, 2 * 0.013879, r"^ankle_roll_[lr]_joint$"),
  _actuator(55, 20, 1, 0.013879, r"^shoulder_pitch_[lr]_joint$"),
  _actuator(55, 10, 1, 0.013879, r"^elbow_pitch_[lr]_joint$"),
)

ULTRA_EFFORT_LIMITS = effort_limits(ULTRA_ACTUATORS)

#: PM01's rule, 0.25 * effort_limit / stiffness: every joint saturates at 4 action units.
#: A flat 0.25 rad gives 125-175 N*m per unit on the hips/knees (saturating at ~1.5 units)
#: but 4-7.5 N*m on the ankles (~13 units); with it, the policy never learned to stand or
#: walk slowly.
ULTRA_ACTION_SCALE: dict[str, float] = {
  expr: 0.25 * a.effort_limit / a.stiffness for a in ULTRA_ACTUATORS for expr in a.target_names_expr
}

##
# Initial state: default joint pose, root height from the model.
##

NOMINAL_QPOS: dict[str, float] = {
  **{f"hip_pitch_{s}_joint": -0.5 for s in "lr"},
  **{f"knee_pitch_{s}_joint": 1.0 for s in "lr"},
  **{f"ankle_pitch_{s}_joint": -0.5 for s in "lr"},
  **{f"elbow_pitch_{s}_joint": -0.3 for s in "lr"},
}

NOMINAL_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, standing_root_height(ULTRA_XML, NOMINAL_QPOS, FOOT_BOX_GEOMS)),
  joint_pos={**NOMINAL_QPOS, ".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collisions: self-collision on (source: `enabled_self_collisions=True`); only the feet get
# friction (condim 3), as for PM01.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_col$",),
  condim={r"^foot_[lr]_.*_col$": 3, r".*_col$": 1},
  priority={r"^foot_[lr]_.*_col$": 1},
  friction={r"^foot_[lr]_.*_col$": (0.6,)},
)


def get_ultra_robot_cfg(collisions: CollisionCfg | None = None) -> EntityCfg:
  """A fresh Ultra `EntityCfg`."""
  return EntityCfg(
    init_state=NOMINAL_KEYFRAME,
    collisions=(collisions or FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=make_articulation(ULTRA_ACTUATORS),
  )
