"""PM01 robot constants and pose helpers.

Single source of truth for the nominal pose, joint grouping and foot geometry, so the
planner, the reference generator and the env all agree. Numbers that are *derived* from
the model (CoM height, stance width, nominal base height) are computed here rather than
hardcoded -- run `scripts/pm01_facts.py` to print them.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

PM01_XML = Path(__file__).resolve().parent / "pm01.xml"

# --- joint groups -----------------------------------------------------------------

LEG_JOINTS = tuple(
  f"{j}_{s}_joint"
  for s in ("l", "r")
  for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
)
ARM_JOINTS = tuple(
  f"{j}_{s}_joint"
  for s in ("l", "r")
  for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_yaw")
)
WAIST_JOINTS = ("waist_yaw_joint",)
HEAD_JOINTS = ("head_yaw_joint",)

#: Joints the policy controls -- every actuated joint. The head (yaw-only, +-35 deg)
#: contributes nothing to balance, but including it keeps the action space identical
#: to the actuator set, which avoids a class of name-resolution mismatch in mjlab and
#: costs one action dimension. A posture term keeps it near zero.
POLICY_JOINTS = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS + HEAD_JOINTS

FOOT_SOLE_GEOMS = ("foot_l_sole_col", "foot_r_sole_col")
#: Every foot collision geom (soles + toe boxes) -- the lowest points of the robot.
FOOT_BOX_GEOMS = FOOT_SOLE_GEOMS + ("foot_l_ankle_col", "foot_r_ankle_col")
FOOT_SITES = ("foot_l_link", "foot_r_link")

# --- nominal pose -----------------------------------------------------------------

#: Exactly `default_position` from pm01_description/configs/pm01.yaml. Slight crouch,
#: arms straight down; everything not listed is 0.
#:
#: WARNING: this pose is *self-penetrating* in the collision model. Each hand
#: sphere (elbow_yaw_{l,r}, r=0.05) overlaps its own-side thigh capsule
#: (hip_yaw_{l,r}, r=0.06) by 19.9 mm. Do not use it as an RL default pose -- see
#: NOMINAL_QPOS.
DEPLOY_QPOS: dict[str, float] = {
  "hip_pitch_l_joint": -0.24,
  "knee_pitch_l_joint": 0.48,
  "ankle_pitch_l_joint": -0.24,
  "hip_pitch_r_joint": -0.24,
  "knee_pitch_r_joint": 0.48,
  "ankle_pitch_r_joint": -0.24,
}

#: Shoulder abduction added to lift the hands clear of the thighs. 4 deg is the break-even
#: point; 8.6 deg leaves margin for the arm motion the policy will use.
SHOULDER_CLEARANCE_RAD = 0.15

#: Our default pose: DEPLOY_QPOS plus arm clearance. Self-collision free.
NOMINAL_QPOS: dict[str, float] = {
  **DEPLOY_QPOS,
  "shoulder_roll_l_joint": +SHOULDER_CLEARANCE_RAD,
  "shoulder_roll_r_joint": -SHOULDER_CLEARANCE_RAD,
}

#: Deploy-time PD gains (EngineAI deployment config). Recorded for reference only -- these are
#: NOT suitable for RL training (arms are K=250/D=1, wildly underdamped); use
#: effective-inertia matching at a 1-2 Hz natural frequency instead.
DEPLOY_STIFFNESS: dict[str, float] = {
  **{j: k for j, k in zip(LEG_JOINTS, [70, 50, 50, 70, 20, 20] * 2, strict=True)},
  "waist_yaw_joint": 200.0,
  **{j: 250.0 for j in ARM_JOINTS},
  "head_yaw_joint": 100.0,
}
DEPLOY_DAMPING: dict[str, float] = {
  **{j: d for j, d in zip(LEG_JOINTS, [7.0, 5.0, 5.0, 7.0, 0.2, 0.2] * 2, strict=True)},
  "waist_yaw_joint": 1.0,
  **{j: 1.0 for j in ARM_JOINTS},
  "head_yaw_joint": 1.0,
}


# --- helpers ------------------------------------------------------------------------


def load_model(xml: Path = PM01_XML) -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(xml))


def joint_qpos_adr(model: mujoco.MjModel, name: str, prefix: str = "") -> int:
  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
  if jid < 0:
    raise KeyError(f"joint not found: {prefix + name}")
  return int(model.jnt_qposadr[jid])


def set_nominal_joints(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> None:
  """Write the nominal joint angles into `data.qpos` (leaves the free joint alone)."""
  for name, q in NOMINAL_QPOS.items():
    data.qpos[joint_qpos_adr(model, name, prefix)] = q


def lowest_sole_z(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> float:
  """World z of the lowest sole corner, given current kinematics."""
  zs = []
  for name in FOOT_SOLE_GEOMS:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name)
    if gid < 0:
      raise KeyError(f"geom not found: {prefix + name}")
    half = model.geom_size[gid]
    rot = data.geom_xmat[gid].reshape(3, 3)
    corners = np.array([[sx * half[0], sy * half[1], -half[2]] for sx in (-1, 1) for sy in (-1, 1)])
    zs.append(float((data.geom_xpos[gid] + corners @ rot.T)[:, 2].min()))
  return min(zs)


def nominal_base_height(model: mujoco.MjModel | None = None, prefix: str = "") -> float:
  """Height of base_link above the support surface at the nominal pose (~0.7985 m)."""
  model = model or load_model()
  data = mujoco.MjData(model)
  set_nominal_joints(model, data, prefix)
  mujoco.mj_forward(model, data)
  base_z = float(data.qpos[2])
  return base_z - lowest_sole_z(model, data, prefix)


def foot_half_extents(model: mujoco.MjModel | None = None, prefix: str = "") -> np.ndarray:
  """(half_length, half_width, half_thickness) of the sole box, metres."""
  model = model or load_model()
  gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + FOOT_SOLE_GEOMS[0])
  return np.array(model.geom_size[gid], dtype=float)


def place_free_base(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  xy: tuple[float, float],
  yaw: float,
  surface_z: float,
  prefix: str = "",
) -> None:
  """Put the robot at (x, y, yaw) with its soles resting on `surface_z`, nominal pose.

  Assumes the robot's free joint occupies qpos[0:7], which holds both for the standalone
  model and for a spec where the robot is the only attached body with a free joint.
  """
  set_nominal_joints(model, data, prefix)
  half = 0.5 * yaw
  data.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
  data.qpos[0:2] = xy
  data.qpos[2] = 0.0
  mujoco.mj_forward(model, data)
  # One forward pass gives the sole height for this orientation; shift to land on it.
  data.qpos[2] = surface_z - lowest_sole_z(model, data, prefix)
  mujoco.mj_forward(model, data)


def foot_collision_extents(
  model: mujoco.MjModel | None = None, prefix: str = ""
) -> dict[str, float]:
  """Extent of the foot's *collision* geometry in the sole frame.

  The sole box is not the whole foot: the ankle box sits above and ahead of it, reaching
  further forward than the toe. Two feet placed heel-to-toe therefore interfere before
  their soles do, so the planner's non-overlap constraint must use these numbers, not the
  sole alone. The CoP/support polygon still uses the sole -- that is a
  different rectangle for a different purpose.
  """
  model = model or load_model()
  data = mujoco.MjData(model)
  set_nominal_joints(model, data, prefix)
  mujoco.mj_forward(model, data)

  sole_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + FOOT_SOLE_GEOMS[0])
  rot_s = data.geom_xmat[sole_gid].reshape(3, 3)
  org = data.geom_xpos[sole_gid]
  body = model.geom_bodyid[sole_gid]

  lo = np.full(3, np.inf)
  hi = np.full(3, -np.inf)
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != body:
      continue
    if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
      continue
    half = model.geom_size[g]
    rot = data.geom_xmat[g].reshape(3, 3)
    corners = np.array(
      [
        [sx * half[0], sy * half[1], sz * half[2]]
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
      ]
    )
    local = (data.geom_xpos[g] + corners @ rot.T - org) @ rot_s
    lo = np.minimum(lo, local.min(axis=0))
    hi = np.maximum(hi, local.max(axis=0))
  return {
    "back": float(-lo[0]),
    "front": float(hi[0]),
    "half_width": float(max(hi[1], -lo[1])),
  }
