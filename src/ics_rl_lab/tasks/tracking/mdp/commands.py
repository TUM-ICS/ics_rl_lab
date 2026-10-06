"""Reference-motion command for the tracking task.

One mjlab ``CommandTerm`` owns every env's playback state
(clip handle, start time) on top of a ``MotionLibrary`` (resident or windowed
storage, see ``ics_rl_lab.motion_lib``). Observations/rewards/terminations read
the reference from here (``env.command_manager.get_term("motion")``), the same
layout as mjlab's own tracking task.

Timing differs from mjlab's MotionCommand: the reference time is
``start_time + (common_step_counter - reset_step) * step_dt``. The env step counter
is incremented right after physics, so rewards/terminations compare the robot at
``t`` with the reference at ``t`` (mjlab's MotionCommand advances its time step only
after rewards, i.e. rewards see the previous frame). Deliberately not
``episode_length_buf``: the runner randomizes it at start (init_at_random_ep_len).
The current frame is computed lazily and cached per env step (invalidated on
resample).

Reset-to-reference (engineai's ``reset_robot_state_by_reference`` event) lives here,
in ``_resample_command``: mjlab runs reset events *before* the command manager's
reset, so an event would still see the previous clip.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

from ics_rl_lab.motion_lib import MotionFrames, MotionLibraryCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.body_names = tuple(cfg.body_names)
    self.root_index = self.body_names.index(cfg.root_body_name)
    self.robot_body_ids = torch.tensor(
      self.robot.find_bodies(self.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
    )
    self.robot_root_body_id = self.robot.body_names.index(cfg.root_body_name)
    self.num_bodies = len(self.body_names)

    # Library joint order == the robot's joint order (joint_pos obs/rewards index both alike).
    self.library = cfg.motion_lib.build(self.robot.joint_names, self.body_names, self.device, self.num_envs)

    self.handles = self.library.sample_clips(self.num_envs)
    self.start_time = torch.zeros(self.num_envs, device=self.device)
    self.reset_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._cache_step = -1
    self._frame: MotionFrames | None = None
    self._relative: tuple[torch.Tensor, torch.Tensor] | None = None

    for name in ("error_base_pos", "error_base_rot", "error_joint_pos", "error_joint_vel", "error_link_pos"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model: mujoco.MjModel | None = None

  # -- time / frames --

  @property
  def motion_time(self) -> torch.Tensor:
    """Current reference time within each env's clip (s). Shape (N,)."""
    elapsed = (self._env.common_step_counter - self.reset_step).to(torch.float32)
    return self.start_time + elapsed * self._env.step_dt

  def get_frames(self, time_offsets: torch.Tensor) -> MotionFrames:
    """Reference frames at ``motion_time + time_offsets`` (offsets (K,) or (N, K), seconds).
    Positions are in the world frame (env origins applied)."""
    times = self.motion_time.unsqueeze(-1) + time_offsets
    frames = self.library.get_frames(self.handles, times)
    frames.body_pos_w = frames.body_pos_w + self._env.scene.env_origins[:, None, None, :]
    return frames

  @property
  def frame(self) -> MotionFrames:
    """Current reference frame, K=1. Cached per env step."""
    if self._frame is None or self._cache_step != self._env.common_step_counter:
      self._frame = self.get_frames(torch.zeros(1, device=self.device))
      self._relative = None
      self._cache_step = self._env.common_step_counter
    return self._frame

  def _invalidate(self) -> None:
    self._frame = None
    self._relative = None

  # current-frame accessors, shapes (N, ...) -------------------------------------

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.frame.joint_pos[:, 0]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.frame.joint_vel[:, 0]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self.frame.body_pos_w[:, 0]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.frame.body_quat_w[:, 0]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.frame.body_lin_vel_w[:, 0]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.frame.body_ang_vel_w[:, 0]

  @property
  def base_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.root_index]

  @property
  def base_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.root_index]

  @property
  def base_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.root_index]

  @property
  def base_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.root_index]

  @property
  def valid(self) -> torch.Tensor:
    """False once the reference time is past the clip end. Shape (N,)."""
    return self.frame.valid[:, 0]

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

  # robot side, same body order as the reference ------------------------------------

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_body_ids]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_body_ids]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_body_ids]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_body_ids]

  def relative_body_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference link poses moved under the robot: base at the robot's xy (reference
    height), rotated by the yaw difference robot vs reference base. engineai's
    ``reference_link_pos_relative_w``/``reference_link_quat_relative_w``."""
    frame = self.frame  # refreshes the cache (and drops a stale _relative)
    if self._relative is None:
      robot_root_pos = self.robot.data.root_link_pos_w
      robot_root_quat = self.robot.data.root_link_quat_w
      anchor_pos = robot_root_pos.clone()
      anchor_pos[:, 2] = self.base_pos_w[:, 2]
      delta_quat = yaw_quat(quat_mul(robot_root_quat, quat_inv(self.base_quat_w)))
      dq = delta_quat[:, None, :].expand(-1, self.num_bodies, -1)
      pos = anchor_pos[:, None, :] + quat_apply(dq, frame.body_pos_w[:, 0] - self.base_pos_w[:, None, :])
      quat = quat_mul(dq, frame.body_quat_w[:, 0])
      self._relative = (pos, quat)
    return self._relative

  # -- CommandTerm hooks --

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.handles[env_ids] = self.library.sample_clips(n)
    lo, hi = self.cfg.start_range
    lengths = self.library.clip_lengths_s(self.handles[env_ids])
    self.start_time[env_ids] = (lo + (hi - lo) * torch.rand(n, device=self.device)) * lengths
    self.reset_step[env_ids] = self._env.common_step_counter
    self._invalidate()
    if self.cfg.reset_robot:
      self._write_reference_state(env_ids)

  def _write_reference_state(self, env_ids: torch.Tensor) -> None:
    """Reset the robot to the reference state at its start time (every reset)."""
    n = len(env_ids)
    frame = self.library.get_frames(self.handles[env_ids], self.start_time[env_ids, None])
    root = self.root_index
    root_pos = frame.body_pos_w[:, 0, root].clone()
    if self.cfg.ensure_links_above_ground:
      min_z = frame.body_pos_w[:, 0, :, 2].min(dim=-1).values
      root_pos[:, 2] += (-min_z).clamp(min=0.0)
    root_pos[:, 2] += self.cfg.start_height_offset
    root_pos += self._env.scene.env_origins[env_ids] + torch.tensor(self.cfg.position_offset, device=self.device)
    root_quat = frame.body_quat_w[:, 0, root]
    root_lin_vel = frame.body_lin_vel_w[:, 0, root] * self.cfg.base_lin_vel_ratio
    root_ang_vel = frame.body_ang_vel_w[:, 0, root] * self.cfg.base_ang_vel_ratio
    joint_pos = frame.joint_pos[:, 0].clone()
    joint_vel = frame.joint_vel[:, 0] * self.cfg.joint_vel_ratio

    # Reference envs only (engineai): position + linear-velocity noise; the reference
    # orientation / angular velocity are kept as they are.
    if self.cfg.pose_range:
      r = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in _AXES[:3]], device=self.device)
      root_pos += sample_uniform(r[:, 0], r[:, 1], (n, 3), device=self.device)
    if self.cfg.velocity_range:
      r = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in _AXES[:3]], device=self.device)
      root_lin_vel += sample_uniform(r[:, 0], r[:, 1], (n, 3), device=self.device)
    lo, hi = self.cfg.joint_position_range
    if (lo, hi) != (0.0, 0.0):
      joint_pos += sample_uniform(lo, hi, joint_pos.shape, device=self.device)

    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self.robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    self.robot.write_root_link_velocity_to_sim(torch.cat([root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self) -> None:
    self.library.update(self._env.step_dt, self.handles)

  def _update_metrics(self) -> None:
    valid = self.valid.float()
    self.metrics["error_base_pos"] = torch.norm(self.base_pos_w - self.robot.data.root_link_pos_w, dim=-1) * valid
    self.metrics["error_base_rot"] = quat_error_magnitude(self.base_quat_w, self.robot.data.root_link_quat_w) * valid
    self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot.data.joint_pos, dim=-1) * valid
    self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot.data.joint_vel, dim=-1) * valid
    rel_pos, _ = self.relative_body_pose_w()
    self.metrics["error_link_pos"] = torch.norm(rel_pos - self.robot_body_pos_w, dim=-1).mean(-1) * valid

  # -- visualization (ghost robot, as mjlab's MotionCommand) --

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    if self._ghost_model is None:
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      self._ghost_model.geom_rgba[:] = np.array(self.cfg.ghost_color, dtype=np.float32)
    indexing = self.robot.indexing
    free_q = indexing.free_joint_q_adr.cpu().numpy()
    joint_q = indexing.joint_q_adr.cpu().numpy()
    offset = np.asarray(self.cfg.ghost_offset)
    for i in env_indices:
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_q[0:3]] = self.base_pos_w[i].cpu().numpy() + offset
      qpos[free_q[3:7]] = self.base_quat_w[i].cpu().numpy()
      qpos[joint_q] = self.joint_pos[i].cpu().numpy()
      visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{i}")


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_lib: MotionLibraryCfg
  body_names: tuple[str, ...]
  """Reference links (engineai ``link_of_interests``), in the order every link term uses."""
  root_body_name: str
  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  """Clips are only resampled on reset (the clip end terminates the episode)."""

  start_range: tuple[float, float] = (0.0, 0.0)
  """Episode start time as a fraction of the clip length (engineai
  ``motion_start_from_middle_range``)."""

  # Reset to the reference state (engineai reset_robot_state_by_reference params).
  reset_robot: bool = True
  position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Only x/y/z are applied (engineai skips orientation noise for reference resets)."""
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Only x/y/z are applied (engineai skips angular noise for reference resets)."""
  joint_position_range: tuple[float, float] = (0.0, 0.0)
  joint_vel_ratio: float = 1.0
  base_lin_vel_ratio: float = 1.0
  base_ang_vel_ratio: float = 1.0
  ensure_links_above_ground: bool = True
  start_height_offset: float = 0.0

  ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
  ghost_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
