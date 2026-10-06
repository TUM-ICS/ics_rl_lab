"""Hoverboard as an mjlab `Entity`.

The board is a floating base with two passive plate joints
(pitch, +-0.17 rad) and two torque-driven wheels. The wheels are not driven by the
policy: a zero-dim `HoverboardActionTerm` (the board's own per-side LQR balancer,
`hoverboard_driver.py` next to this file) writes their torque every physics step.

* wheels: pure torque sources capped at the rated speed ->
  `DcMotorActuatorCfg(stiffness=0, damping=0, effort_limit=30)` with a torque-speed
  curve: full 30 N*m up to ~80 rad/s, falling to 0 at 83.8 rad/s (MuJoCo has no joint
  velocity limit). Without it an unloaded board (robot fallen off) spins its
  frictionless wheels up to thousands of rad/s. Armature and friction are 0.
* plates: passive -> no actuator, joint damping 0.1 set directly in `hoverboard.xml`.
* no self-collision -> `contype=0, conaffinity=1` on every collision geom: board-board
  pairs never collide, but the board still collides with the terrain and with the robot
  (whose geoms have `contype=1`).

Known gap: the URDF's plate-joint velocity limit (50 rad/s) has no MuJoCo equivalent for a
passive joint and is not enforced. The wheel cap is a torque
curve, not a hard cap: external forces can still push a wheel past 83.8 rad/s.

Reproduce the numbers with `uv run python -m ics_rl_lab.assets.hoverboard.hoverboard_cfg`.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

HOVERBOARD_XML = Path(__file__).resolve().parent / "hoverboard.xml"
assert HOVERBOARD_XML.exists(), HOVERBOARD_XML


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, HOVERBOARD_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(HOVERBOARD_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Names and geometry.
##

PLATE_BODIES = ("left_plate_link", "right_plate_link")
WHEEL_BODIES = ("left_wheel_link", "right_wheel_link")
PLATE_JOINTS = ("left_plate_joint", "right_plate_joint")
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")

WHEEL_RADIUS = 0.0814  # [m], wheel collision cylinder
AXLE_WIDTH = 2 * (0.1135 + 0.14365)  # [m], wheel-to-wheel joint distance (0.5143)

##
# Motor spec. Similar motor: YK_B_1972 (hub motor, 1:1).
##


def rpm_to_rad_per_s(rpm: float) -> float:
  return rpm * 2.0 * math.pi / 60.0


WHEEL_RATED_SPEED = rpm_to_rad_per_s(800)  # [rad/s]
WHEEL_RATED_TORQUE = 4.18  # [N*m]; tau = P/w = 350 W / 83.8 rad/s
WHEEL_PEAK_TORQUE = 30.0  # [N*m]; the effort limit actually applied

# Stall torque 20x the effort limit puts the curve's corner at 95% of the rated speed:
# full torque below ~80 rad/s, linear drop to zero at 83.8 -- a soft stand-in for a hard
# joint velocity cap, not a real motor curve.
_CAP_SHARPNESS = 20.0

WHEEL_ACTUATOR = DcMotorActuatorCfg(
  target_names_expr=(r"^.*_wheel_joint$",),
  stiffness=0.0,
  damping=0.0,
  effort_limit=WHEEL_PEAK_TORQUE,
  saturation_effort=_CAP_SHARPNESS * WHEEL_PEAK_TORQUE,
  velocity_limit=WHEEL_RATED_SPEED,
)

HOVERBOARD_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(WHEEL_ACTUATOR,),
  soft_joint_pos_limit_factor=1.0,
)

##
# Keyframe.
##

PLATE_HALF_HEIGHT = 0.04  # [m], plate collision box half-height (plate top = axle + 0.04)

# Resting on the wheels (source: 0.086, a ~5 mm drop). Spawning at rest matters: while
# the board is unloaded, the balancer's torque slams the light plates past their limits.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, WHEEL_RADIUS + 0.001),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

#: World z of the plate tops when the board rests level on flat ground at z=0.
PLATE_TOP_Z = WHEEL_RADIUS + PLATE_HALF_HEIGHT

##
# Collisions.
##

# No self-collision; see module docstring.
#
# priority=1 on plates and wheels: MuJoCo combines friction as the *max* of both geoms
# (or takes the higher-priority geom's). Without priority the terrain's default friction
# 1.0 would override the task's wheel-friction randomization; with it, the wheel's value
# wins. The plates tie with PM01's soles (also priority 1), so foot-plate
# friction is max(sole, plate), which is the plate's value for the task's ranges.
NO_SELF_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_col$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority={r"^.*_(plate|wheel)_col$": 1, r".*_col$": 0},
)


def get_hoverboard_cfg(collisions: CollisionCfg | None = None) -> EntityCfg:
  """Hoverboard entity config.

  A new instance each call, so callers that mutate the config cannot affect
  each other.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(collisions or NO_SELF_COLLISION,),
    spec_fn=get_spec,
    articulation=HOVERBOARD_ARTICULATION,
  )


if __name__ == "__main__":
  model = get_spec().compile()
  print(f"nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom}")
  print(f"total mass {sum(model.body_mass):.3f} kg")
  print(f"wheel rated speed {WHEEL_RATED_SPEED:.1f} rad/s "
        f"= {WHEEL_RATED_SPEED * WHEEL_RADIUS:.2f} m/s at the rim")
  print(f"axle width {AXLE_WIDTH:.4f} m, wheel radius {WHEEL_RADIUS} m")
