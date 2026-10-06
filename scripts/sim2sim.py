"""Run an exported ONNX policy in plain MuJoCo (no mjlab, torch or GPU).

A quick sim2sim check: everything the policy needs -- joint order, PD gains, armature,
effort limits, default pose, action scale, observation layout/scales/history and command
ranges -- is read from the ONNX metadata that `train.py`/`play.py` embed on export. The
robot MJCF is picked from `src/ics_rl_lab/assets/` by matching the policy's joint names
(override with `--xml`).

Deliberately minimal, and different from training in these ways:
  * flat ground, no terrain, no domain randomization, no observation noise, no actuator delay;
  * PD is computed explicitly every physics step and clamped to the effort limit (as the
    robot's motor drivers do), instead of mjlab's implicit position actuators;
  * contact/solver settings are MuJoCo defaults plus the robot XML's own.

Supported policy observation terms: base_ang_vel, base_lin_vel, projected_gravity,
commands / velocity_commands, joint_pos_rel, joint_vel_rel, last_action (the AMP and
locomotion tasks). Tasks needing anything else (tracking references, hoverboard state)
are rejected.

Usage:
  uv run scripts/sim2sim.py path/to/policy.onnx
  uv run scripts/sim2sim.py path/to/policy.onnx --vx 1.5 --headless --duration 10

Viewer keys: UP/DOWN vx -/+ 0.25 m/s, PAGE_UP/PAGE_DOWN vy, LEFT/RIGHT yaw rate,
END zero command, HOME reset the robot.
"""

from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ASSETS = Path(__file__).resolve().parents[1] / "src" / "ics_rl_lab" / "assets"
ROBOT_XMLS = ("pm01/pm01.xml", "s2/s2.xml", "ultra/ultra.xml", "g1/g1.xml")

SUPPORTED_OBS = {
  "base_ang_vel", "base_lin_vel", "projected_gravity", "commands", "velocity_commands",
  "joint_pos_rel", "joint_vel_rel", "last_action",
}

# GLFW key codes (mujoco.viewer passes them to key_callback).
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME, KEY_END = 266, 267, 268, 269


##
# Metadata.
##


def _parse(value: str):
  """ONNX metadata values are strings: '1,2,3' lists, '[a, b],[c, d]' point lists, dict reprs."""
  try:
    return ast.literal_eval(value) if value[:1] in "{[" else [float(v) for v in value.split(",")]
  except (ValueError, SyntaxError):
    return value.split(",") if "," in value else value


def load_policy(path: Path) -> tuple[ort.InferenceSession, dict]:
  session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  meta = {k: _parse(v) for k, v in session.get_modelmeta().custom_metadata_map.items()}
  if isinstance(meta.get("vy_bound_points"), tuple):  # '[..],[..]' parses as a tuple of lists
    meta["vy_bound_points"] = list(meta["vy_bound_points"])
    meta["omega_bound_points"] = list(meta["omega_bound_points"])
  for key in ("decimation", "observation_history_length"):
    meta[key] = int(meta[key][0])
  for key in ("dt", "step_dt"):
    meta[key] = float(meta[key][0])
  missing = [k for k in ("joint_names", "joint_stiffness", "joint_damping", "joint_effort_limits",
                         "default_joint_pos", "action_joint_names", "action_scale", "action_offset",
                         "observation_names", "observation_scales") if k not in meta]
  if missing:
    raise ValueError(f"{path.name}: metadata lacks {missing} -- re-export it with scripts/play.py")
  unsupported = set(meta["observation_names"]) - SUPPORTED_OBS
  if unsupported:
    raise ValueError(f"{path.name}: unsupported observation terms {sorted(unsupported)}")
  if "joint_armature" not in meta:
    print("[WARN] metadata has no joint_armature (older export); joints get the XML's armature. "
          "Re-export with scripts/play.py for faithful joint dynamics.")
  return session, meta


def find_robot_xml(joint_names: list[str]) -> Path:
  for rel in ROBOT_XMLS:
    xml = ASSETS / rel
    model = mujoco.MjModel.from_xml_path(str(xml))
    names = {model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE}
    if names == set(joint_names):
      return xml
  raise ValueError("no robot XML in assets/ matches the policy's joints; pass --xml")


##
# Simulation.
##


def add_scene(spec: mujoco.MjSpec) -> None:
  """Checker floor, gradient skybox, sun + headlight, shadows and a far clipping plane."""
  vis = spec.visual
  vis.headlight.ambient = [0.3, 0.3, 0.3]
  vis.headlight.diffuse = [0.5, 0.5, 0.5]
  vis.headlight.specular = [0.0, 0.0, 0.0]
  vis.rgba.haze = [0.15, 0.25, 0.35, 1.0]
  vis.global_.azimuth = 135
  vis.global_.elevation = -20
  vis.global_.offwidth, vis.global_.offheight = 1920, 1080
  vis.quality.shadowsize = 8192
  vis.quality.offsamples = 8
  vis.map.znear = 0.01
  vis.map.zfar = 500.0
  vis.map.fogstart, vis.map.fogend = 40.0, 120.0  # fog only shows if the viewer enables it
  spec.stat.extent = 4.0  # sets the default camera distance and light/shadow frustum scale

  spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                   builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[0.3, 0.5, 0.7], rgb2=[0.0, 0.0, 0.0], width=512, height=3072)
  spec.add_texture(name="checker", type=mujoco.mjtTexture.mjTEXTURE_2D,
                   builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                   rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
                   width=300, height=300)
  floor_mat = spec.add_material(name="floor", texrepeat=[1, 1], texuniform=True, reflectance=0.2)
  floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "checker"

  # size 0 = infinite plane for physics; the third entry is the rendered grid spacing.
  spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor")
  # Directional sun from above-front, casting shadows, plus a soft fill light.
  spec.worldbody.add_light(name="sun", pos=[0, 0, 10], dir=[-0.3, -0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                           castshadow=True, diffuse=[0.6, 0.6, 0.6], specular=[0.2, 0.2, 0.2])
  spec.worldbody.add_light(name="fill", pos=[0, 0, 4], dir=[0.3, 0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                           castshadow=False, diffuse=[0.2, 0.2, 0.2])


class Sim2Sim:
  def __init__(self, session: ort.InferenceSession, meta: dict, xml: Path):
    self.session, self.meta = session, meta
    self.input_name = session.get_inputs()[0].name

    spec = mujoco.MjSpec.from_file(str(xml))
    spec.option.timestep = meta["dt"]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    add_scene(spec)
    joint_names = meta["joint_names"]
    if "joint_armature" in meta:
      for name, arm in zip(joint_names, meta["joint_armature"], strict=True):
        spec.joint(name).armature = arm
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    m = self.model

    free = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(free) == 1, "expected one floating base"
    self.root_qadr, self.root_vadr = m.jnt_qposadr[free[0]], m.jnt_dofadr[free[0]]
    self.root_body = m.jnt_bodyid[free[0]]

    # Joint-order (observations, PD) and action-order index maps.
    self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in joint_names])
    self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in joint_names])
    self.kp = np.array(meta["joint_stiffness"])
    self.kd = np.array(meta["joint_damping"])
    self.effort = np.array(meta["joint_effort_limits"])
    self.default_q = np.array(meta["default_joint_pos"])
    self.action_to_joint = np.array([joint_names.index(n) for n in meta["action_joint_names"]])
    self.action_scale = np.array(meta["action_scale"])
    self.action_offset = np.array(meta["action_offset"])

    self.obs_names = list(meta["observation_names"])
    self.obs_scales = [np.asarray(s, dtype=float) for s in meta["observation_scales"]]
    self.history = meta["observation_history_length"]
    term_hist = meta.get("observation_history_lengths", [1] * len(self.obs_names))
    if self.history == 1 and any(int(h) > 1 for h in term_hist):
      raise ValueError("per-term observation history is not supported")

    self.vx_range = tuple(meta.get("cmd_range_vx", (-1.0, 1.0)))
    self.vy_range = tuple(meta.get("cmd_range_vy", (-0.5, 0.5)))
    self.wz_range = tuple(meta.get("cmd_range_omega", (-1.0, 1.0)))
    self.command = np.zeros(3)
    self.reset()

  def reset(self) -> None:
    m, d = self.model, self.data
    mujoco.mj_resetData(m, d)
    d.qpos[self.root_qadr + 3 : self.root_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    d.qpos[self.qadr] = self.default_q
    d.qpos[self.root_qadr + 2] = 2.0
    mujoco.mj_forward(m, d)
    # Drop the robot so its lowest collision geom sits 2 mm above the floor.
    lowest = np.inf
    for g in range(m.ngeom):
      if m.geom_bodyid[g] == 0 or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
        continue
      center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
      corners = center + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
      lowest = min(lowest, (d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
    d.qpos[self.root_qadr + 2] += 0.002 - lowest
    mujoco.mj_forward(m, d)
    self.action = np.zeros(len(self.action_scale))
    self.target_q = self.default_q.copy()
    self.obs_history: list[np.ndarray] | None = None

  def set_command(self, vx: float, vy: float, wz: float) -> None:
    vx = float(np.clip(vx, *self.vx_range))
    vy_max, wz_max = max(abs(self.vy_range[0]), self.vy_range[1]), max(abs(self.wz_range[0]), self.wz_range[1])
    if "vy_bound_points" in self.meta:  # hull command: vy / yaw-rate bounds shrink as vx grows
      vy_pts, wz_pts = np.array(self.meta["vy_bound_points"]), np.array(self.meta["omega_bound_points"])
      vy_max = np.interp(vx, vy_pts[:, 0], vy_pts[:, 1])
      wz_max = np.interp(vx, wz_pts[:, 0], wz_pts[:, 1])
    self.command[:] = vx, float(np.clip(vy, -vy_max, vy_max)), float(np.clip(wz, -wz_max, wz_max))

  # -- observations (same definitions as the mjlab terms)

  def _root_rot(self) -> np.ndarray:
    return self.data.xmat[self.root_body].reshape(3, 3)

  def _term(self, name: str) -> np.ndarray:
    d = self.data
    if name == "base_ang_vel":  # free-joint angular velocity is already in the body frame
      return d.qvel[self.root_vadr + 3 : self.root_vadr + 6].copy()
    if name == "base_lin_vel":
      return self._root_rot().T @ d.qvel[self.root_vadr : self.root_vadr + 3]
    if name == "projected_gravity":
      return self._root_rot().T @ np.array([0.0, 0.0, -1.0])
    if name in ("commands", "velocity_commands"):
      return self.command.copy()
    if name == "joint_pos_rel":
      return d.qpos[self.qadr] - self.default_q
    if name == "joint_vel_rel":
      return d.qvel[self.vadr].copy()
    if name == "last_action":
      return self.action.copy()
    raise KeyError(name)

  def observation(self) -> np.ndarray:
    frame = [self._term(n) * s for n, s in zip(self.obs_names, self.obs_scales)]
    if self.obs_history is None:  # first frame fills the whole history (as mjlab does)
      self.obs_history = [frame] * self.history
    else:
      self.obs_history = self.obs_history[1:] + [frame]
    # Per term: its history oldest -> newest, then the next term.
    return np.concatenate([np.concatenate([h[i] for h in self.obs_history]) for i in range(len(frame))])

  # -- control

  def policy_step(self) -> None:
    obs = self.observation().astype(np.float32)[None]
    self.action = self.session.run(None, {self.input_name: obs})[0][0].astype(float)
    target = self.action_offset + self.action_scale * self.action  # action order
    self.target_q = np.empty_like(self.default_q)
    self.target_q[self.action_to_joint] = target

  def physics_step(self) -> None:
    d = self.data
    tau = self.kp * (self.target_q - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr]
    d.qfrc_applied[self.vadr] = np.clip(tau, -self.effort, self.effort)
    mujoco.mj_step(self.model, d)

  def step(self) -> None:
    self.policy_step()
    for _ in range(self.meta["decimation"]):
      self.physics_step()

  def fallen(self) -> bool:
    return self._root_rot()[2, 2] < np.cos(0.8)  # same 0.8 rad tilt as the bad_orientation term

  def base_velocity(self) -> np.ndarray:
    """(vx, vy) in the yaw frame and the yaw rate, for reporting."""
    rot = self._root_rot()
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    v = self.data.qvel[self.root_vadr : self.root_vadr + 3]
    w = self.data.qvel[self.root_vadr + 3 : self.root_vadr + 6]
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], (rot @ w)[2]])


##
# Entry points.
##


def run_headless(sim: Sim2Sim, duration: float) -> None:
  steps = int(duration / sim.meta["step_dt"])
  vels = []
  for i in range(steps):
    sim.step()
    if sim.fallen():
      print(f"[sim2sim] fell after {i * sim.meta['step_dt']:.2f} s")
      return
    if i >= steps // 2:
      vels.append(sim.base_velocity())
  v = np.mean(vels, axis=0)
  print(f"[sim2sim] command vx {sim.command[0]:.2f} vy {sim.command[1]:.2f} wz {sim.command[2]:.2f} | "
        f"second-half mean vx {v[0]:.2f} vy {v[1]:.2f} wz {v[2]:.2f} | upright for {duration:.1f} s")


def run_viewer(sim: Sim2Sim) -> None:
  import mujoco.viewer

  def on_key(key: int) -> None:
    vx, vy, wz = sim.command
    if key == KEY_UP: vx += 0.25  # noqa: E701
    elif key == KEY_DOWN: vx -= 0.25  # noqa: E701
    elif key == KEY_PAGE_UP: vy += 0.1  # noqa: E701
    elif key == KEY_PAGE_DOWN: vy -= 0.1  # noqa: E701
    elif key == KEY_LEFT: wz += 0.25  # noqa: E701
    elif key == KEY_RIGHT: wz -= 0.25  # noqa: E701
    elif key == KEY_END: vx = vy = wz = 0.0  # noqa: E701
    elif key == KEY_HOME: sim.reset()  # noqa: E701
    else: return  # noqa: E701
    sim.set_command(vx, vy, wz)
    print(f"[sim2sim] command vx {sim.command[0]:.2f} vy {sim.command[1]:.2f} wz {sim.command[2]:.2f}")

  with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = sim.root_body
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    while viewer.is_running():
      start = time.perf_counter()
      sim.step()
      if sim.fallen():
        print("[sim2sim] fell -- resetting")
        sim.reset()
      viewer.sync()
      time.sleep(max(0.0, sim.meta["step_dt"] - (time.perf_counter() - start)))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("policy", type=Path, help="exported .onnx policy")
  parser.add_argument("--xml", type=Path, default=None, help="robot MJCF (default: matched from assets/)")
  parser.add_argument("--vx", type=float, default=0.0)
  parser.add_argument("--vy", type=float, default=0.0)
  parser.add_argument("--wz", type=float, default=0.0)
  parser.add_argument("--headless", action="store_true", help="no viewer; print the tracked velocity")
  parser.add_argument("--duration", type=float, default=10.0, help="headless run length (s)")
  args = parser.parse_args()

  session, meta = load_policy(args.policy)
  xml = args.xml or find_robot_xml(meta["joint_names"])
  print(f"[sim2sim] {args.policy.name}: {len(meta['action_joint_names'])} actions, obs {meta['observation_names']}, "
        f"history {meta['observation_history_length']}, dt {meta['dt']} x {meta['decimation']} | robot {xml.name}")
  sim = Sim2Sim(session, meta, xml)
  sim.set_command(args.vx, args.vy, args.wz)
  if args.headless:
    run_headless(sim, args.duration)
  else:
    run_viewer(sim)


if __name__ == "__main__":
  main()
