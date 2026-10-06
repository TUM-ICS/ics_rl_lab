"""Retarget motion clips from a source kinematic model onto a robot (MuJoCo + scipy only).

The source is any MJCF with a free joint (here MimicKit's humanoid, `assets/humanoid/`) and a
clip is a sequence of its qpos. A handful of source bodies are mapped onto robot bodies
(`Target.frame_map`). Then, per clip:

1. Link-length fit (once per robot): the source's limb offsets are scaled so the mapped bodies
   line up with the robot's in a shared calibration pose (`source_pose` / `target_pose`). This
   is linear least squares, since forward kinematics in a fixed pose is linear in the offsets.
2. Calibration: the constant transform from each source body to its robot body is recorded in
   the calibration pose. Every frame of the clip then gives a world pose target for each robot
   body (`T_source_body(t) @ offset`).
3. Ground: the whole clip is shifted vertically so the robot's lowest foot point touches z=0
   in the first frame.
4. Per-frame IK (scipy SLSQP, warm-started from the previous frame):
     50 * position error^2 + 1 * orientation error^2 (mapped bodies)
     + 0.02 * |joint angles|^2 + 0.1 * |change from the previous frame|^2,
   subject to joint limits and the foot collision geoms (boxes, capsules, spheres) staying
   at or above z=0.

Output: one .pkl per clip (`fps`, `root_pos`, `root_rot` xyzw, `dof_pos`, `dof_names`) in
`<data root>/<robot>/retargeted/`, plus `humanoid_standing.pkl` (the robot's default pose held still,
which the AMP tasks use as a clip). Clips run in parallel (10 workers by default); clips whose .pkl
already exists are skipped, so an interrupted run just resumes. Finally `replay_motions.py` turns
all .pkl files into the AMP .npz clips in `<data root>/<robot>/amp/`.

To retarget from another source (another mocap skeleton, another robot), give it an MJCF with a
free joint, a loader returning qpos per frame (see `load_mimickit`), and a `frame_map` from its
bodies to the robot's.

Usage (MimicKit_Data.zip unzipped into the data root, see the Readme):
  uv run scripts/motion/retarget.py --robot g1
  uv run scripts/motion/retarget.py --robot g1 --input /path/to/clip.pkl --view --no-replay
"""

from __future__ import annotations

# ruff: noqa: E402
import os

# The IK works on tiny matrices: multithreaded BLAS only adds overhead (~1.7x slower, all cores busy).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing as mp
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro
from scipy.optimize import lsq_linear, minimize
from tqdm import tqdm

ASSETS = Path(__file__).resolve().parents[2] / "src" / "ics_rl_lab" / "assets"
SOURCE_XML = ASSETS / "humanoid" / "humanoid.xml"

#: Source joints whose body offsets the link-length fit may scale (one per limb segment).
SOURCE_SCALED_JOINTS = (
  "abdomen_x", "neck_x",
  "right_shoulder_x", "right_elbow", "left_shoulder_x", "left_elbow",
  "right_hip_x", "right_knee", "right_ankle_x", "left_hip_x", "left_knee", "left_ankle_x",
)

#: MimicKit's humanoid clips, below the motion data root.
MIMICKIT_DIR = ("MimicKit_Data", "motions", "humanoid")
#: The AMP clips retargeted by default (116), relative to MIMICKIT_DIR.
MIMICKIT_CLIPS = (
  "long/humanoid_long_walk[0-9]*.pkl",  # 8 walks (4 + mirrored)
  "parc/humanoid_run[0-9][0-9].pkl",  # 62 runs
  "sie/humanoid_sie_run_*.pkl",  # 46 runs: forward, left, right
)

#: Standing clip: 5 s at the 50 Hz policy rate.
STANDING_FRAMES = 250

# Cost weights of the per-frame IK.
W_POS, W_ROT, W_POSTURE, W_SMOOTH = 50.0, 1.0, 0.02, 0.1


@dataclass(frozen=True)
class Target:
  xml: Path
  #: source body -> robot body
  frame_map: dict[str, str]
  #: Calibration pose (unlisted joints at 0) in which source and robot look alike.
  target_pose: dict[str, float]
  source_pose: dict[str, float] = field(default_factory=dict)  # humanoid at 0 is a T-pose
  #: Pose for the standing clip (joint names or regexes, as in mjlab keyframes).
  #: Given as "module:attribute" of the robot's config and loaded only when needed: importing
  #: ics_rl_lab/mjlab costs ~2.5 GB, which the retargeting workers must not pay.
  standing_pose: str = ""
  #: Regex of the robot's foot collision geoms that must stay above the ground.
  foot_geoms: str = r"^foot_[lr]_.*_col$"


TARGETS = {
  "pm01": Target(
    xml=ASSETS / "pm01" / "pm01.xml",
    frame_map={
      "pelvis": "base_link", "torso": "torso_yaw_link", "head": "head_yaw_link",
      "left_shin": "knee_pitch_l_link", "right_shin": "knee_pitch_r_link",
      "left_foot": "ankle_pitch_l_link", "right_foot": "ankle_pitch_r_link",
      "left_upper_arm": "shoulder_roll_l_link", "right_upper_arm": "shoulder_roll_r_link",
      "left_lower_arm": "elbow_yaw_l_link", "right_lower_arm": "elbow_yaw_r_link",
    },
    target_pose={"shoulder_roll_l_joint": 1.57, "shoulder_roll_r_joint": -1.57},
    standing_pose="ics_rl_lab.assets.pm01.robot:NOMINAL_QPOS",
  ),
  "s2": Target(
    xml=ASSETS / "s2" / "s2.xml",
    frame_map={
      "pelvis": "base_link", "torso": "waist_pitch_link", "head": "head_pitch_link",
      "left_shin": "L_knee_pitch_link", "right_shin": "R_knee_pitch_link",
      "left_foot": "L_ankle_roll_link", "right_foot": "R_ankle_roll_link",
      "left_upper_arm": "L_shoulder_yaw_link", "right_upper_arm": "R_shoulder_yaw_link",
      "left_lower_arm": "L_elbow_roll_link", "right_lower_arm": "R_elbow_roll_link",
    },
    target_pose={
      "L_shoulder_roll_joint": 1.40, "R_shoulder_roll_joint": -1.40,
      "L_shoulder_yaw_joint": 1.57, "R_shoulder_yaw_joint": -1.57,
    },
    standing_pose="ics_rl_lab.assets.s2.s2_cfg:NOMINAL_QPOS",
  ),
  # Ultra has no shoulder roll (it cannot T-pose), so both calibrate with the arms down.
  "ultra": Target(
    xml=ASSETS / "ultra" / "ultra.xml",
    frame_map={
      "pelvis": "pelvis",
      "left_shin": "knee_pitch_l_link", "right_shin": "knee_pitch_r_link",
      "left_foot": "ankle_roll_l_link", "right_foot": "ankle_roll_r_link",
      "left_upper_arm": "shoulder_pitch_l_link", "right_upper_arm": "shoulder_pitch_r_link",
      "left_lower_arm": "elbow_pitch_l_link", "right_lower_arm": "elbow_pitch_r_link",
    },
    source_pose={"left_shoulder_x": -1.57, "right_shoulder_x": 1.57},
    target_pose={},
    standing_pose="ics_rl_lab.assets.ultra.ultra_cfg:NOMINAL_QPOS",
  ),
  "g1": Target(
    xml=ASSETS / "g1" / "g1.xml",
    frame_map={
      "pelvis": "pelvis", "torso": "torso_link",
      "left_shin": "left_knee_link", "right_shin": "right_knee_link",
      "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link",
      "left_upper_arm": "left_shoulder_roll_link", "right_upper_arm": "right_shoulder_roll_link",
      "left_lower_arm": "left_elbow_link", "right_lower_arm": "right_elbow_link",
    },
    # G1's elbow is bent ~80 deg at 0; 1.43 straightens the arm like the humanoid's T-pose.
    target_pose={
      "left_shoulder_roll_joint": 1.57, "right_shoulder_roll_joint": -1.57, ".*_elbow_joint": 1.43,
    },
    standing_pose="ics_rl_lab.assets.g1.g1_cfg:NOMINAL_KEYFRAME.joint_pos",
    foot_geoms=r"^(left|right)_foot\d_collision$",
  ),
}


##
# Clip loading.
##


def load_mimickit(path: Path, model: mujoco.MjModel) -> tuple[np.ndarray, float]:
  """MimicKit .pkl -> (qpos (T, nq), fps). Frames are [root pos, root exp-map, 28 dofs]."""
  with open(path, "rb") as f:
    clip = pickle.load(f)
  frames = np.asarray(clip["frames"], dtype=float)
  if frames.shape[1] - 6 != model.nq - 7:
    raise ValueError(f"{path.name}: {frames.shape[1] - 6} dofs, source model has {model.nq - 7}")
  qpos = np.zeros((len(frames), model.nq))
  qpos[:, :3] = frames[:, :3]
  for t, rot in enumerate(frames[:, 3:6]):
    angle = np.linalg.norm(rot)
    axis = rot / angle if angle > 1e-9 else np.array([1.0, 0.0, 0.0])
    mujoco.mju_axisAngle2Quat(qpos[t, 3:7], axis, angle)
  qpos[:, 7:] = frames[:, 6:]
  return qpos, float(clip["fps"])


##
# Small SO(3) helpers.
##


def _skew(v: np.ndarray) -> np.ndarray:
  return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _log(R: np.ndarray) -> np.ndarray:
  q = np.empty(4)
  mujoco.mju_mat2Quat(q, R.ravel())
  v = np.empty(3)
  mujoco.mju_quat2Vel(v, q, 1.0)
  return v


def _jr(phi: np.ndarray) -> np.ndarray:
  """SO(3) right Jacobian."""
  th = np.linalg.norm(phi)
  K = _skew(phi)
  if th < 1e-6:
    return np.eye(3) - 0.5 * K
  return np.eye(3) - (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * K @ K


def _jr_inv(phi: np.ndarray) -> np.ndarray:
  """Inverse of the SO(3) right Jacobian."""
  th = np.linalg.norm(phi)
  K = _skew(phi)
  if th < 1e-6:
    return np.eye(3) + 0.5 * K
  return np.eye(3) + 0.5 * K + (1 / th**2 - (1 + np.cos(th)) / (2 * th * np.sin(th))) * K @ K


def _kinematic_model(xml: Path) -> mujoco.MjModel:
  """The robot without its visual meshes: same kinematics, ~55 MB instead of up to 1.4 GB."""
  spec = mujoco.MjSpec.from_file(str(xml))
  for geom in [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]:
    spec.delete(geom)
  for mesh in list(spec.meshes):
    spec.delete(mesh)
  return spec.compile()


def _load(path: str):
  """"package.module:attr.attr" -> object, imported on first use."""
  import importlib

  import mjlab  # noqa: F401  (registers the tasks before the asset modules; avoids an import cycle)

  module, _, attrs = path.partition(":")
  obj = importlib.import_module(module)
  for attr in attrs.split("."):
    obj = getattr(obj, attr)
  return obj


def _pose(model: mujoco.MjModel, joints: dict[str, float]) -> np.ndarray:
  """qpos at the origin, identity root, `joints` (names or regexes) set, every other joint at 0."""
  q = np.zeros(model.nq)
  q[3] = 1.0
  for pattern, value in joints.items():
    matched = [j for j in range(1, model.njnt) if re.fullmatch(pattern, model.joint(j).name)]
    if not matched:
      raise ValueError(f"no joint matches {pattern!r}")
    q[model.jnt_qposadr[matched]] = value
  return q


def _foot_points(model: mujoco.MjModel, geom: int) -> tuple[np.ndarray, float]:
  """Points (geom frame) and radius whose spheres bound the geom from below."""
  size, kind = model.geom_size[geom], model.geom_type[geom]
  if kind == mujoco.mjtGeom.mjGEOM_BOX:
    return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size, 0.0
  if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
    return np.array([[0.0, 0.0, -size[1]], [0.0, 0.0, size[1]]]), float(size[0])
  if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
    return np.zeros((1, 3)), float(size[0])
  raise ValueError(f"foot geom {model.geom(geom).name}: only boxes, capsules and spheres are supported")


def _body_pose(data: mujoco.MjData, body: int) -> np.ndarray:
  T = np.eye(4)
  T[:3, :3], T[:3, 3] = data.xmat[body].reshape(3, 3), data.xpos[body]
  return T


def _check_floating(model: mujoco.MjModel, name: str) -> None:
  types = model.jnt_type
  if types[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
    raise ValueError(f"{name}: the first joint must be the free root joint")
  if any(t not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for t in types[1:]):
    raise ValueError(f"{name}: only hinge/slide joints are supported below the root")


##
# Retargeting.
##


class Retargeter:
  def __init__(self, target: Target, verbose: bool = True):
    self.src = mujoco.MjModel.from_xml_path(str(SOURCE_XML))
    self.tgt = _kinematic_model(target.xml)
    _check_floating(self.src, SOURCE_XML.name)
    _check_floating(self.tgt, target.xml.name)
    self.src_data, self.tgt_data = mujoco.MjData(self.src), mujoco.MjData(self.tgt)
    self.src_bodies = [self.src.body(n).id for n in target.frame_map]
    self.tgt_bodies = [self.tgt.body(n).id for n in target.frame_map.values()]
    self.q_src_cal = _pose(self.src, target.source_pose)
    self.q_tgt_cal = _pose(self.tgt, target.target_pose)
    self.dof_names = [self.tgt.joint(j).name for j in range(1, self.tgt.njnt)]
    self.standing_pose = target.standing_pose

    self._fit_link_lengths(verbose)

    # Calibration offsets: robot body pose expressed in its source body's frame.
    src_cal, tgt_cal = self._fk(self.src, self.src_data, self.q_src_cal), self._fk(self.tgt, self.tgt_data, self.q_tgt_cal)
    self.offsets = [np.linalg.inv(_body_pose(src_cal, s)) @ _body_pose(tgt_cal, t)
                    for s, t in zip(self.src_bodies, self.tgt_bodies, strict=True)]

    # Foot collision geoms as points + radius, in their geom frames.
    pattern = re.compile(target.foot_geoms)
    self.foot_geoms = [g for g in range(self.tgt.ngeom) if pattern.match(self.tgt.geom(g).name or "")]
    if not self.foot_geoms:
      raise ValueError(f"{target.xml.name}: no geoms match {target.foot_geoms}")
    self.foot_points = [_foot_points(self.tgt, g) for g in self.foot_geoms]

    # Joint limits as qpos bounds (the root and unlimited joints are free).
    self.lo, self.hi = np.full(self.tgt.nq, -np.inf), np.full(self.tgt.nq, np.inf)
    for j in range(1, self.tgt.njnt):
      if self.tgt.jnt_limited[j]:
        self.lo[self.tgt.jnt_qposadr[j]], self.hi[self.tgt.jnt_qposadr[j]] = self.tgt.jnt_range[j]

  @staticmethod
  def _fk(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> mujoco.MjData:
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    return data

  def _fit_link_lengths(
    self, verbose: bool, w_pos: float = 100.0, w_scale: float = 1.0, w_offset: float = 10.0
  ) -> None:
    """Per scaled source joint, new body offset = nominal * S + O (S, O per axis), fitting the
    mapped bodies' calibration positions to the robot's, regularized towards S=1, O=0."""
    src, tgt = self.src, self.tgt
    bodies = [src.jnt_bodyid[src.joint(n).id] for n in SOURCE_SCALED_JOINTS]
    nominal = src.body_pos[bodies].copy()

    def positions() -> np.ndarray:
      return self._fk(src, self.src_data, self.q_src_cal).xpos[self.src_bodies].ravel().copy()

    goal = self._fk(tgt, self.tgt_data, self.q_tgt_cal).xpos[self.tgt_bodies].ravel()
    base = positions()
    # Positions are linear in the body offsets, so unit perturbations give the exact Jacobian.
    J = np.zeros((base.size, nominal.size))
    for i, b in enumerate(bodies):
      for a in range(3):
        src.body_pos[b, a] += 1.0
        J[:, 3 * i + a] = positions() - base
        src.body_pos[b, a] -= 1.0
    # Unknowns: u = S - 1 and O, both flattened (body, axis).
    A = np.vstack([
      np.sqrt(w_pos) * np.hstack([J * nominal.ravel(), J]),
      np.sqrt(w_scale) * np.hstack([np.eye(nominal.size), np.zeros((nominal.size,) * 2)]),
      np.sqrt(w_offset) * np.hstack([np.zeros((nominal.size,) * 2), np.eye(nominal.size)]),
    ])
    b = np.concatenate([np.sqrt(w_pos) * (goal - base), np.zeros(2 * nominal.size)])
    lo = np.concatenate([np.full(nominal.size, 0.2 - 1.0), np.full(nominal.size, -0.2)])
    hi = np.concatenate([np.full(nominal.size, 5.0 - 1.0), np.full(nominal.size, 0.2)])
    u, offset = np.split(lsq_linear(A, b, bounds=(lo, hi)).x, 2)
    src.body_pos[bodies] = nominal * (1.0 + u.reshape(-1, 3)) + offset.reshape(-1, 3)

    if not verbose:
      return
    err = np.linalg.norm((positions() - goal).reshape(-1, 3), axis=1)
    names = [src.body(s).name for s in self.src_bodies]
    print("[retarget] calibration-pose error after the link-length fit (cm): "
          + ", ".join(f"{n} {100 * e:.1f}" for n, e in zip(names, err, strict=True)))

  def _references(self, q_src: np.ndarray) -> list[np.ndarray]:
    d = self._fk(self.src, self.src_data, q_src)
    return [_body_pose(d, s) @ off for s, off in zip(self.src_bodies, self.offsets, strict=True)]

  def _ground_offset(self, q_src0: np.ndarray) -> float:
    """Clip height shift putting the lowest foot point at z=0, from the frame-0 references
    (each foot geom rigidly attached to the mapped robot body it hangs below)."""
    d = self._fk(self.tgt, self.tgt_data, self.q_tgt_cal)
    refs = dict(zip(self.tgt_bodies, self._references(q_src0), strict=True))
    lowest = np.inf
    for g, (points, radius) in zip(self.foot_geoms, self.foot_points, strict=True):
      body = self.tgt.geom_bodyid[g]
      while body not in refs:
        if body == 0:
          raise ValueError(f"foot geom {self.tgt.geom(g).name} is not below a mapped body")
        body = self.tgt.body_parentid[body]
      T_geom = np.eye(4)
      T_geom[:3, :3], T_geom[:3, 3] = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
      T = refs[body] @ np.linalg.inv(_body_pose(d, body)) @ T_geom
      lowest = min(lowest, (T[:3, 3] + points @ T[:3, :3].T)[:, 2].min() - radius)
    return -lowest

  def _solve(self, q_prev: np.ndarray, refs: list[np.ndarray]) -> np.ndarray:
    """One IK frame. Unknown: x in the tangent space at q_prev, q = q_prev (+) x."""
    m, d, nv = self.tgt, self.tgt_data, self.tgt.nv
    jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
    cache: dict[str, object] = {}

    def evaluate(x: np.ndarray):
      if cache.get("x") is not None and np.array_equal(cache["x"], x):
        return cache["out"]
      q = q_prev.copy()
      mujoco.mj_integratePos(m, q, x, 1.0)
      d.qpos[:] = q
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      root_fix = _jr(x[3:6])  # d(root angular velocity)/dx on the root rotation
      rows, jac = [], []
      for b, T_ref in zip(self.tgt_bodies, refs, strict=True):
        mujoco.mj_jacBody(m, d, jacp, jacr, b)
        jp, jr = jacp.copy(), jacr.copy()
        jp[:, 3:6] = jp[:, 3:6] @ root_fix
        jr[:, 3:6] = jr[:, 3:6] @ root_fix
        e = _log(T_ref[:3, :3] @ d.xmat[b].reshape(3, 3).T)
        rows += [np.sqrt(W_POS) * (d.xpos[b] - T_ref[:3, 3]), np.sqrt(W_ROT) * e]
        jac += [np.sqrt(W_POS) * jp, -np.sqrt(W_ROT) * _jr_inv(e) @ jr]
      rows += [np.sqrt(W_POSTURE) * q[7:], np.sqrt(W_SMOOTH) * x]
      jac += [np.sqrt(W_POSTURE) * np.eye(nv)[6:], np.sqrt(W_SMOOTH) * np.eye(nv)]
      r, J = np.concatenate(rows), np.vstack(jac)

      heights, height_jac = [], []
      for g, (points, radius) in zip(self.foot_geoms, self.foot_points, strict=True):
        b = m.geom_bodyid[g]
        mujoco.mj_jacBody(m, d, jacp, jacr, b)
        jacp[:, 3:6] = jacp[:, 3:6] @ root_fix
        jacr[:, 3:6] = jacr[:, 3:6] @ root_fix
        pts = d.geom_xpos[g] + points @ d.geom_xmat[g].reshape(3, 3).T
        heights.append(pts[:, 2] - radius)
        # z row of jacp - skew(p - xpos) @ jacr, for every point p
        lever = pts - d.xpos[b]
        height_jac.append(jacp[2] + lever[:, 1:2] * jacr[0] - lever[:, 0:1] * jacr[1])
      out = (float(r @ r), 2.0 * J.T @ r, np.concatenate(heights), np.vstack(height_jac))
      cache["x"], cache["out"] = x.copy(), out
      return out

    bounds = list(zip(np.r_[np.full(6, -np.inf), self.lo[7:] - q_prev[7:]],
                      np.r_[np.full(6, np.inf), self.hi[7:] - q_prev[7:]], strict=True))
    bounds = [(None if np.isinf(lo) else lo, None if np.isinf(hi) else hi) for lo, hi in bounds]
    res = minimize(
      lambda x: evaluate(x)[:2], np.zeros(nv), jac=True, method="SLSQP", bounds=bounds,
      constraints={"type": "ineq", "fun": lambda x: evaluate(x)[2], "jac": lambda x: evaluate(x)[3]},
      options={"maxiter": 50, "ftol": 1e-7},
    )
    q = q_prev.copy()
    mujoco.mj_integratePos(m, q, res.x, 1.0)
    q[3:7] /= np.linalg.norm(q[3:7])
    return q

  def retarget(
    self, q_src: np.ndarray, viewer: Viewer | None = None, fps: float = 30.0, progress: bool = True
  ) -> np.ndarray:
    q_src = q_src.copy()
    q_src[:, 2] += self._ground_offset(q_src[0])
    q = np.zeros(self.tgt.nq)
    q[:7] = q_src[0, :7]
    q[7:] = np.clip(0.0, self.lo[7:], self.hi[7:])
    out = np.empty((len(q_src), self.tgt.nq))
    for t in tqdm(range(len(q_src)), desc="[retarget]", leave=False, disable=not progress):
      start = time.perf_counter()
      q = self._solve(q, self._references(q_src[t]))
      out[t] = q
      if viewer is not None:
        viewer.show(q_src[t], q, 1.0 / fps - (time.perf_counter() - start))
    return out

  def standing(self) -> np.ndarray:
    """Robot default pose with the lowest foot point at z=0."""
    q = _pose(self.tgt, _load(self.standing_pose))
    d = self._fk(self.tgt, self.tgt_data, q)
    q[2] = -min((d.geom_xpos[g] + p @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min() - r
                for g, (p, r) in zip(self.foot_geoms, self.foot_points, strict=True))
    return q


##
# Viewer: the scaled source as a ghost next to the robot.
##


class ViewerClosed(Exception):
  pass


class Viewer:
  def __init__(self, rt: Retargeter, target: Target):
    import mujoco.viewer

    spec = mujoco.MjSpec.from_file(str(target.xml))
    src_spec = mujoco.MjSpec.from_file(str(SOURCE_XML))
    for g in src_spec.geoms:
      g.rgba = [0.2, 0.6, 1.0, 0.4]
    spec.worldbody.add_frame().attach_body(src_spec.body("pelvis"), "src/", "")
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], width=300, height=300)
    mat = spec.add_material(name="grid", texrepeat=[1, 1], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid",
                            contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    for b in range(1, rt.src.nbody):  # copy the fitted link lengths
      self.model.body_pos[self.model.body("src/" + rt.src.body(b).name).id] = rt.src.body_pos[b]
    self.src_qadr = self.model.jnt_qposadr[self.model.joint("src/root").id]
    self.nq_tgt = rt.tgt.nq

    self.handle = mujoco.viewer.launch_passive(self.model, self.data)
    self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    self.handle.cam.trackbodyid = 1
    self.handle.cam.distance = 4.0

  def show(self, q_src: np.ndarray, q_tgt: np.ndarray, sleep: float) -> None:
    if not self.handle.is_running():
      raise ViewerClosed
    self.data.qpos[: self.nq_tgt] = q_tgt
    self.data.qpos[self.src_qadr : self.src_qadr + len(q_src)] = q_src
    mujoco.mj_kinematics(self.model, self.data)
    self.handle.sync()
    time.sleep(max(0.0, sleep))

  def wait_and_close(self) -> None:
    """Keep the window open until the user closes it. Exiting while the viewer thread still
    renders crashes X (GLXBadDrawable)."""
    if self.handle.is_running():
      print("[retarget] done -- close the viewer window to exit")
    while self.handle.is_running():
      time.sleep(0.05)
    self.handle.close()


##
# Entry point.
##


def save(path: Path, qpos: np.ndarray, fps: float, dof_names: list[str]) -> None:
  clip = {
    "dof_names": dof_names,
    "fps": fps,
    "root_pos": qpos[:, :3].tolist(),
    "root_rot": qpos[:, [4, 5, 6, 3]].tolist(),  # wxyz -> xyzw
    "dof_pos": qpos[:, 7:].tolist(),
  }
  with open(path, "wb") as f:
    pickle.dump(clip, f)


_worker: Retargeter | None = None


def _init_worker(robot: str) -> None:
  global _worker
  _worker = Retargeter(TARGETS[robot], verbose=False)


def _retarget_file(path: Path, out_dir: Path) -> tuple[str, int, float]:
  assert _worker is not None
  q_src, fps = load_mimickit(path, _worker.src)
  start = time.perf_counter()
  q = _worker.retarget(q_src, fps=fps, progress=False)
  save(out_dir / path.name, q, fps, _worker.dof_names)
  return path.name, len(q), time.perf_counter() - start


def main(
  input: str | None = None,
  robot: Literal["pm01", "s2", "ultra", "g1"] = "pm01",
  output_dir: str | None = None,
  workers: int = 10,
  overwrite: bool = False,
  replay: bool = True,
  view: bool = False,
) -> None:
  """Retarget MimicKit humanoid clips onto a robot and build its AMP clips.

  Args:
    input: A MimicKit .pkl clip or a directory of them (default: the MIMICKIT_CLIPS list).
    robot: Target robot.
    output_dir: Where to write the retargeted .pkl files (default: <data root>/<robot>/retargeted).
      The AMP .npz files go to its sibling `amp/`.
    workers: Clips retargeted in parallel (at most the number of CPU cores).
    overwrite: Redo clips that already have a .pkl (default: skip them, i.e. resume).
    replay: Afterwards, run replay_motions.py to write the AMP .npz clips.
    view: Show the robot and the scaled source while retargeting (real time, one clip at a time,
      existing outputs are redone). Close the window to stop.
  """
  # Heavy (~2.5 GB), so main process only. mjlab first: it and ics_rl_lab import each other.
  import mjlab  # noqa: F401

  from ics_rl_lab.motion_lib.paths import motion_data_dir

  out = Path(output_dir) if output_dir is not None else motion_data_dir(robot, "retargeted")
  out.mkdir(parents=True, exist_ok=True)
  if input is None:
    clip_dir = motion_data_dir(*MIMICKIT_DIR)
    if not clip_dir.exists():
      raise FileNotFoundError(f"{clip_dir} not found: download MimicKit_Data.zip and unzip it into "
                              f"{motion_data_dir()} (see the Readme), or pass --input")
    files = sorted(f for pattern in MIMICKIT_CLIPS for f in clip_dir.glob(pattern))
  else:
    input_path = Path(input)
    files = sorted(input_path.glob("*.pkl")) if input_path.is_dir() else [input_path]
  if not files:
    raise FileNotFoundError(f"No .pkl files at {input or clip_dir}")
  todo = [f for f in files if overwrite or view or not (out / f.name).exists()]
  print(f"[retarget] {robot}: {len(todo)} of {len(files)} clips to retarget -> {out}")

  target = TARGETS[robot]
  rt = Retargeter(target)
  standing = rt.standing()
  save(out / "humanoid_standing.pkl", np.tile(standing, (STANDING_FRAMES, 1)), 50.0, rt.dof_names)

  if view or workers == 1:
    viewer = Viewer(rt, target) if view else None
    try:
      for path in todo:
        q_src, fps = load_mimickit(path, rt.src)
        start = time.perf_counter()
        q = rt.retarget(q_src, viewer, fps)
        save(out / path.name, q, fps, rt.dof_names)
        print(f"[retarget] {path.name}: {len(q)} frames in {time.perf_counter() - start:.1f} s")
    except ViewerClosed:
      print("[retarget] viewer closed -- stopping (the current clip is not saved)")
      replay = False
    finally:
      if viewer is not None:
        viewer.wait_and_close()
  elif todo:
    todo.sort(key=lambda f: f.stat().st_size, reverse=True)  # longest clips first
    n = min(len(todo), workers, os.cpu_count() or 1)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(n, mp_context=ctx, initializer=_init_worker, initargs=(robot,)) as pool:
      jobs = [pool.submit(_retarget_file, f, out) for f in todo]
      for job in tqdm(as_completed(jobs), total=len(jobs), desc=f"[retarget] {n} workers", unit="clip"):
        name, frames, seconds = job.result()
        tqdm.write(f"[retarget] {name}: {frames} frames in {seconds:.1f} s")

  if replay:
    from replay_motions import main as replay_main  # mjlab + torch; only needed here

    replay_main(input=str(out), robot=robot, output_dir=str(out.parent / "amp"))


if __name__ == "__main__":
  tyro.cli(main)
