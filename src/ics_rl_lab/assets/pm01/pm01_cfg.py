"""PM01 as an mjlab `Entity`.

Gains follow the effective-inertia recipe mjlab uses for its own robots: pick a
natural frequency and damping ratio, then

    K = J * w_n^2        D = 2 * zeta * J * w_n

with `J` the *reflected rotor inertia* (MuJoCo's joint `armature`), not the link
inertia. The motor constants below are EngineAI's specifications for this robot, so `J`
is measured rather than inferred.

mjlab's 10 Hz / zeta 2, except the ankles, which use values tuned on the hardware: K=20 /
D=0.2 with doubled armature. A 7 Hz natural frequency for the other joints trained worse:
with the stiffness-derived action scale it doubles the target excursion per action unit,
so exploration noise moved the joints 8-19% more.

Two facts about mjlab that shape this file:

* `BuiltinPositionActuatorCfg` takes **scalar** stiffness/damping/effort_limit, not
  regex->value dicts. A group that needs two different gain sets has to become two
  configs -- which is why `hip_yaw` is split out of the leg group here.
* mjlab *appends* `<position>` actuators to whatever the XML already declares, so
  `pm01.xml` declares none; with the robot's 24 `<motor>` elements in the XML, nu
  would be 48.

Reproduce the derived numbers with `uv run python -m ics_rl_lab.assets.pm01.pm01_cfg`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.delayed_actuator import DelayedActuatorCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from ics_rl_lab.assets.pm01 import robot

##
# MJCF and assets.
##

PM01_XML = robot.PM01_XML
assert PM01_XML.exists(), PM01_XML


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PM01_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PM01_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Motor specs (EngineAI).
##


@dataclass(frozen=True)
class MotorSpec:
  """Physical motor specification for a PM01 actuator type."""

  armature: float  # reflected rotor inertia (kg*m^2)
  effort_limit: float  # max torque (N*m)
  velocity_limit: float  # max joint velocity (rad/s); recorded, unused by mjlab


# Q90 -- the high-torque joints: hip pitch, hip roll, knee.
Q90 = MotorSpec(armature=0.0453, effort_limit=164.0, velocity_limit=26.3)
# Q25 -- everything else: hip yaw, ankles, waist, shoulders, elbows, head.
Q25 = MotorSpec(armature=0.0067, effort_limit=61.0, velocity_limit=35.2)

NATURAL_FREQ = 10.0 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_Q90 = Q90.armature * NATURAL_FREQ**2  # ~178.8
STIFFNESS_Q25 = Q25.armature * NATURAL_FREQ**2  # ~26.5
DAMPING_Q90 = 2.0 * DAMPING_RATIO * Q90.armature * NATURAL_FREQ  # ~11.39
DAMPING_Q25 = 2.0 * DAMPING_RATIO * Q25.armature * NATURAL_FREQ  # ~1.68

# Ankles hand-tuned on hardware: D=0.2 was needed to stop ankle
# oscillation on hard ground -- keep it at 0.2, not larger. Armature doubled.
STIFFNESS_ANKLE = 20.0
DAMPING_ANKLE = 0.2
ARMATURE_ANKLE = 2.0 * Q25.armature

# Position-target delay, sampled once per env at every reset and shared by all actuator
# groups: 0-20 ms, kept as a time so tasks with a different physics step get the same
# delay.
MAX_DELAY_S = 0.020


def delay_steps(physics_dt: float) -> int:
  """`MAX_DELAY_S` in physics steps of `physics_dt`."""
  return round(MAX_DELAY_S / physics_dt)


def _actuator(
  motor: MotorSpec,
  stiffness: float,
  damping: float,
  *targets: str,
  armature: float | None = None,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=targets,
    stiffness=stiffness,
    damping=damping,
    effort_limit=motor.effort_limit,
    armature=motor.armature if armature is None else armature,
  )


# One group per motor type. hip_yaw is split out of the leg group because it is a Q25 motor and mjlab
# cannot express per-joint gains within one actuator config.
LEG_ACTUATOR = _actuator(
  Q90,
  STIFFNESS_Q90,
  DAMPING_Q90,
  r"^hip_pitch_[lr]_joint$",
  r"^hip_roll_[lr]_joint$",
  r"^knee_pitch_[lr]_joint$",
)
HIP_YAW_ACTUATOR = _actuator(Q25, STIFFNESS_Q25, DAMPING_Q25, r"^hip_yaw_[lr]_joint$")
ANKLE_ACTUATOR = _actuator(
  Q25,
  STIFFNESS_ANKLE,
  DAMPING_ANKLE,
  r"^ankle_pitch_[lr]_joint$",
  r"^ankle_roll_[lr]_joint$",
  armature=ARMATURE_ANKLE,
)
WAIST_ACTUATOR = _actuator(Q25, STIFFNESS_Q25, DAMPING_Q25, r"^waist_yaw_joint$")
ARM_ACTUATOR = _actuator(
  Q25,
  STIFFNESS_Q25,
  DAMPING_Q25,
  r"^shoulder_(pitch|roll|yaw)_[lr]_joint$",
  r"^elbow_(pitch|yaw)_[lr]_joint$",
)
# In the action space like every other actuator (`robot.POLICY_JOINTS`, 24). It is
# not a balance channel, so a posture term holds it near zero
# rather than the action space excluding it.
HEAD_ACTUATOR = _actuator(Q25, STIFFNESS_Q25, DAMPING_Q25, r"^head_yaw_joint$")

PM01_ACTUATORS = (
  LEG_ACTUATOR,
  HIP_YAW_ACTUATOR,
  ANKLE_ACTUATOR,
  WAIST_ACTUATOR,
  ARM_ACTUATOR,
  HEAD_ACTUATOR,
)


def make_pm01_articulation(max_delay_steps: int = 0) -> EntityArticulationInfoCfg:
  """PM01 actuators, each wrapped in a position-target delay of 0..`max_delay_steps`.

  `delay_hold_prob=1` stops mjlab from resampling the lag every physics step (its
  default); the lag is set per episode by `actuator_delay_event` instead.
  """
  return EntityArticulationInfoCfg(
    actuators=tuple(
      DelayedActuatorCfg(
        base_cfg=a, delay_min_lag=0, delay_max_lag=max_delay_steps, delay_hold_prob=1.0
      )
      for a in PM01_ACTUATORS
    ),
    soft_joint_pos_limit_factor=0.9,
  )



def actuator_delay_event(max_delay_steps: int) -> EventTermCfg:
  """Reset event sampling one delay per env for all PM01 actuators.

  Reset events run after the actuators' own reset (which zeroes the lag), so this
  sets the lag for the whole episode. Must match the robot's `max_delay_steps`.
  """
  return EventTermCfg(
    func=dr.sync_actuator_delays, mode="reset", params={"lag_range": (0, max_delay_steps)}
  )


def apply_actuator_delay(cfg) -> None:
  """Give a task's PM01 (`cfg.scene.entities["robot"]`) the 0-`MAX_DELAY_S` delay.

  Call after the robot is attached; uses the task's physics step.
  """
  n = delay_steps(cfg.sim.mujoco.timestep)
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.articulation = make_pm01_articulation(n)
  cfg.events["actuator_delay"] = actuator_delay_event(n)


##
# Action scale, per mjlab's convention: a quarter of the joint excursion that
# saturates the motor, i.e. 0.25 * effort_limit / stiffness.
##

PM01_ACTION_SCALE: dict[str, float] = {}
PM01_EFFORT_LIMITS: dict[str, float] = {}
for _a in PM01_ACTUATORS:
  assert _a.effort_limit is not None
  for _n in _a.target_names_expr:
    PM01_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness
    PM01_EFFORT_LIMITS[_n] = _a.effort_limit


##
# Keyframes.
##

NOMINAL_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, robot.nominal_base_height()),
  joint_pos={**robot.NOMINAL_QPOS, ".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collisions.
##

# Self-collision ON, mirroring mjlab's own G1 config: non-foot contacts are
# frictionless (condim=1) so they act as interpenetration stops rather than
# surfaces the policy can push off. Needed: the measured -94 mm swing-foot/stance-shank
# penetration is only detectable with the leg capsules live.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_col$",),
  condim={r"^foot_[lr]_sole_col$": 3, r".*_col$": 1},
  priority={r"^foot_[lr]_sole_col$": 1},
  friction={r"^foot_[lr]_sole_col$": (0.6,)},
)

# Self-collision OFF. Every
# geom still collides with the terrain (terrain contype=1 meets conaffinity=1),
# but robot-robot pairs resolve to 0. Kept for the speed/fidelity ablation.
NO_SELF_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_col$",),
  contype=0,
  conaffinity=1,
  condim={r"^foot_[lr]_sole_col$": 3, r".*_col$": 1},
  priority={r"^foot_[lr]_sole_col$": 1},
  friction={r"^foot_[lr]_sole_col$": (0.6,)},
)

# Only the soles collide with anything at all. Cheapest, and unusable for the
# beam task, but it is the floor of the cost comparison.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^foot_[lr]_sole_col$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


def get_pm01_robot_cfg(collisions: CollisionCfg | None = None) -> EntityCfg:
  """Build a fresh PM01 `EntityCfg`.

  A new instance each call, so callers that mutate the config cannot affect
  each other. No actuator delay; training tasks add it with `apply_actuator_delay`.
  """
  return EntityCfg(
    init_state=NOMINAL_KEYFRAME,
    collisions=(collisions or FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=make_pm01_articulation(),
  )


if __name__ == "__main__":
  print(f"natural freq {NATURAL_FREQ / (2 * math.pi):.1f} Hz, zeta {DAMPING_RATIO}")
  print(f"  Q90  J={Q90.armature:.4f}  K={STIFFNESS_Q90:7.2f}  D={DAMPING_Q90:6.2f}")
  print(f"  Q25  J={Q25.armature:.4f}  K={STIFFNESS_Q25:7.2f}  D={DAMPING_Q25:6.2f}")
  print(f"  ankle K={STIFFNESS_ANKLE}  D={DAMPING_ANKLE}  J={ARMATURE_ANKLE:.4f}")
  print(f"  delay 0-{MAX_DELAY_S * 1e3:.0f} ms")
  print("\naction scale (0.25 * effort / stiffness):")
  for _n, _s in PM01_ACTION_SCALE.items():
    print(f"  {_n:40s} {_s:.3f}")
