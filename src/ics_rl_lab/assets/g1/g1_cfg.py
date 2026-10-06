"""Unitree G1 (29 DoF) as an mjlab `Entity`.

The model (`g1.xml`, `meshes/`) is a local copy of mjlab's asset-zoo G1 (see README.md), so it
can be edited here like the other robots. Motor data, actuator groups, keyframe and collision
setup are mjlab's own (`mjlab.asset_zoo.robots.unitree_g1.g1_constants`): Unitree rotor
inertias and gear ratios -> armature, 10 Hz / zeta 2 gains (the recipe PM01 uses), ankles and
waist pitch/roll as two-motor linkages. The action scale follows the same 0.25 * effort / K
rule as the other robots (every joint saturates at 4 action units).

Added here, as for PM01 and S2: the 0-20 ms position-target delay (`apply_g1_actuator_delay`),
sampled once per episode. The foot is seven capsules per side, so `place_on_terrain` samples
capsule axes (see `tasks/locomotion/mdp/events.py`).
"""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as mjlab_g1
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from ics_rl_lab.assets.robot_utils import apply_actuator_delay, effort_limits, make_articulation

G1_XML = Path(__file__).resolve().parent / "g1.xml"
assert G1_XML.exists(), G1_XML


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_XML.parent / "meshes", spec.meshdir)
  spec.assets = assets
  return spec


##
# Names.
##

ROOT_BODY = "pelvis"
TORSO_BODY = "torso_link"
ANKLE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
#: Every foot collision geom (7 capsules per foot) -- the lowest points of the robot.
FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))

##
# Actuators: mjlab's G1 groups (5020, 7520-14, 7520-22, 4010, waist and ankle linkages).
##

G1_ACTUATORS = mjlab_g1.G1_ARTICULATION.actuators
G1_EFFORT_LIMITS = effort_limits(G1_ACTUATORS)

#: 0.25 * effort_limit / stiffness, as for PM01, S2 and Ultra (= mjlab's G1_ACTION_SCALE).
G1_ACTION_SCALE: dict[str, float] = {
  expr: 0.25 * a.effort_limit / a.stiffness for a in G1_ACTUATORS for expr in a.target_names_expr
}


def apply_g1_actuator_delay(cfg) -> None:
  """0-20 ms position-target delay, one lag per env per episode (as for PM01/S2)."""
  apply_actuator_delay(cfg, G1_ACTUATORS)


##
# Initial state and collisions (mjlab's).
##

#: Knees-bent standing pose, mjlab's default for G1 tasks.
NOMINAL_KEYFRAME = mjlab_g1.KNEES_BENT_KEYFRAME

#: Self-collision on; feet condim 3 / friction 0.6 / priority 1, everything else frictionless.
FULL_COLLISION = mjlab_g1.FULL_COLLISION


def get_g1_robot_cfg(collisions: CollisionCfg | None = None) -> EntityCfg:
  """A fresh G1 `EntityCfg` (no delay; training tasks add it with `apply_g1_actuator_delay`)."""
  return EntityCfg(
    init_state=copy.deepcopy(NOMINAL_KEYFRAME),
    collisions=(copy.deepcopy(collisions or FULL_COLLISION),),
    spec_fn=get_spec,
    articulation=make_articulation(G1_ACTUATORS),
  )
