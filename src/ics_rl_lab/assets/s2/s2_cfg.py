"""EngineAI S2 as an mjlab `Entity`.

* MJCF: `s2.xml`, converted from the robot's URDF (`s2_feet_collision.urdf`) with MuJoCo's
  URDF compiler and hand-checked (collision primitives named `*_col`, cylinders ->
  capsules/spheres, fixed hand/force-sensor links kept as joint-less bodies, visual meshes
  decimated).
* Gains, armature and effort limits: hand-set per joint group, not derived from a natural
  frequency like PM01's. mjlab takes scalar gains per actuator config, so the shoulder/waist-yaw
  group (one gain set, two effort limits) is split in two.
* Delay: 0-20 ms position-target delay through `apply_s2_actuator_delay`, sampled once per
  episode.

Joint velocity limits (8.4 rad/s on the legs) are not enforced in simulation.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from ics_rl_lab.assets.robot_utils import apply_actuator_delay, effort_limits, make_articulation, standing_root_height

S2_XML = Path(__file__).resolve().parent / "s2.xml"
assert S2_XML.exists(), S2_XML


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(S2_XML))
  assets: dict[str, bytes] = {}
  update_assets(assets, S2_XML.parent / "meshes", spec.meshdir)
  spec.assets = assets
  return spec


##
# Names.
##

ROOT_BODY = "base_link"
TORSO_BODY = "waist_pitch_link"
ANKLE_BODIES = ("L_ankle_roll_link", "R_ankle_roll_link")
FOOT_SOLE_GEOMS = ("foot_l_sole_col", "foot_r_sole_col")
#: Every foot collision geom (soles + toe boxes) -- the lowest points of the robot.
FOOT_BOX_GEOMS = FOOT_SOLE_GEOMS + ("foot_l_toe_col", "foot_r_toe_col")

##
# Actuators (source groups; effort N*m, stiffness N*m/rad, damping N*m*s/rad).
##


def _actuator(effort: float, stiffness: float, damping: float, armature: float, *targets: str):
  return BuiltinPositionActuatorCfg(
    target_names_expr=targets, stiffness=stiffness, damping=damping, effort_limit=effort, armature=armature
  )


S2_ACTUATORS = (
  _actuator(225, 700, 10, 0.22, r"^[LR]_hip_roll_joint$", r"^[LR]_hip_pitch_joint$", r"^[LR]_knee_pitch_joint$"),
  _actuator(65, 20, 1.6, 0.028, r"^[LR]_hip_yaw_joint$", r"^[LR]_ankle_pitch_joint$", r"^[LR]_ankle_roll_joint$"),
  _actuator(265, 620, 9, 0.25, r"^waist_pitch_joint$"),
  _actuator(80, 30, 1.5, 0.02, r"^[LR]_shoulder_pitch_joint$", r"^[LR]_shoulder_roll_joint$"),
  _actuator(85, 30, 1.5, 0.02, r"^waist_yaw_joint$"),
  _actuator(45, 10, 0.8, 0.011, r"^[LR]_shoulder_yaw_joint$", r"^[LR]_elbow_roll_joint$"),
  _actuator(20, 4, 0.4, 0.005, r"^[LR]_elbow_yaw_joint$", r"^[LR]_wrist_pitch_joint$", r"^[LR]_wrist_roll_joint$"),
  _actuator(4.5, 8, 0.3, 0.002, r"^head_yaw_joint$", r"^head_pitch_joint$"),
)

S2_EFFORT_LIMITS = effort_limits(S2_ACTUATORS)

#: PM01's rule, 0.25 * effort_limit / stiffness: every joint saturates at 4 action units.
#: A flat 0.25 rad gives ~175 N*m per unit on the legs but 5 N*m on hip yaw and ankles; on
#: Ultra the same imbalance kept the policy from learning to stand or walk slowly.
S2_ACTION_SCALE: dict[str, float] = {
  expr: 0.25 * a.effort_limit / a.stiffness for a in S2_ACTUATORS for expr in a.target_names_expr
}


def apply_s2_actuator_delay(cfg) -> None:
  """0-20 ms position-target delay, one lag per env per episode."""
  apply_actuator_delay(cfg, S2_ACTUATORS)


##
# Initial state: default joint pose, root height from the model.
##

NOMINAL_QPOS: dict[str, float] = {
  "L_shoulder_yaw_joint": 1.57,
  "L_elbow_roll_joint": -0.68,
  "R_shoulder_yaw_joint": -1.57,
  "R_elbow_roll_joint": 0.68,
  **{f"{s}_hip_pitch_joint": -0.24 for s in "LR"},
  **{f"{s}_knee_pitch_joint": 0.48 for s in "LR"},
  **{f"{s}_ankle_pitch_joint": -0.24 for s in "LR"},
}

NOMINAL_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, standing_root_height(S2_XML, NOMINAL_QPOS, FOOT_BOX_GEOMS)),
  joint_pos={**NOMINAL_QPOS, ".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collisions: self-collision on (source: `enabled_self_collisions=True`). As for PM01,
# only the feet get friction (condim 3); the rest are frictionless interpenetration stops.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_col$",),
  condim={r"^foot_[lr]_.*_col$": 3, r".*_col$": 1},
  priority={r"^foot_[lr]_.*_col$": 1},
  friction={r"^foot_[lr]_.*_col$": (0.6,)},
)


def get_s2_robot_cfg(collisions: CollisionCfg | None = None) -> EntityCfg:
  """A fresh S2 `EntityCfg` (no delay; training tasks add it with `apply_s2_actuator_delay`)."""
  return EntityCfg(
    init_state=NOMINAL_KEYFRAME,
    collisions=(collisions or FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=make_articulation(S2_ACTUATORS),
  )
