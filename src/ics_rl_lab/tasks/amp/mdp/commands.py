"""HullVelocityCommand: the velocity command for the running tasks.

A velocity command bounded by a kinematic "hull": vy and omega's allowed range
are piecewise-linear functions of the sampled vx (faster forward speed -> tighter
lateral/turning bounds), with tracking-error metrics binned by commanded speed.
Structurally mirrors mjlab's own stock `UniformVelocityCommand`
(`mjlab/tasks/velocity/mdp/velocity_command.py`) -- same `CommandTerm` base class,
same metric/resample/update method shapes -- just with the interpolated-bound
sampling logic swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class HullVelocityCommand(CommandTerm):
  """Velocity command in SE(2), with vy/omega bounds that shrink as vx grows."""

  cfg: "HullVelocityCommandCfg"

  def __init__(self, cfg: "HullVelocityCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.heading is None:
      raise ValueError("heading_command=True but heading is set to None.")

    self.robot: Entity = env.scene[cfg.asset_name]

    vy_pts = torch.tensor(self.cfg.vy_bound_points, dtype=torch.float32, device=self.device)
    self.vy_xp, self.vy_yp = vy_pts[:, 0], vy_pts[:, 1]
    omega_pts = torch.tensor(self.cfg.omega_bound_points, dtype=torch.float32, device=self.device)
    self.omega_xp, self.omega_yp = omega_pts[:, 0], omega_pts[:, 1]

    self.min_vx = max(self.vy_xp[0].item(), self.omega_xp[0].item())
    self.max_vx = min(self.vy_xp[-1].item(), self.omega_xp[-1].item())

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    self.vel_bins = self.cfg.velocity_bins
    self.bin_keys: list[tuple[float, float, str]] = []
    for i in range(len(self.vel_bins) - 1):
      low, high = self.vel_bins[i], self.vel_bins[i + 1]
      key_suffix = f"{low:.1f}_to_{high:.1f}"
      self.bin_keys.append((low, high, key_suffix))
      self.metrics[f"error_vel_xy_{key_suffix}"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics[f"error_vel_yaw_{key_suffix}"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def get_policy_metadata(self) -> dict:
    """Command bounds for the exported ONNX policy (picked up by MjlabVecEnv)."""
    max_vy = max(p[1] for p in self.cfg.vy_bound_points)
    max_omega = max(p[1] for p in self.cfg.omega_bound_points)
    return {
      "vy_bound_points": self.cfg.vy_bound_points,
      "omega_bound_points": self.cfg.omega_bound_points,
      "cmd_range_vx": [self.min_vx, self.max_vx],
      "cmd_range_vy": [-max_vy, max_vy],
      "cmd_range_omega": [-max_omega, max_omega],
    }

  def _batched_interp(self, x: torch.Tensor, xp: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    """Vectorized 1D piecewise-linear interpolation."""
    idx = torch.searchsorted(xp, x)
    idx = torch.clamp(idx, 1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = yp[idx - 1], yp[idx]
    t = (x - x0) / (x1 - x0 + 1e-6)
    return y0 + t * (y1 - y0)

  def _update_metrics(self) -> None:
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt

    step_error_xy = torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1)
    step_error_omega = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])

    self.metrics["error_vel_xy"] += step_error_xy / max_command_step
    self.metrics["error_vel_yaw"] += step_error_omega / max_command_step

    vx_cmds = torch.abs(self.vel_command_b[:, 0])
    for low, high, key_suffix in self.bin_keys:
      in_bin_mask = (vx_cmds >= low) & (vx_cmds < high)
      active_count = in_bin_mask.sum()
      scale = self.num_envs / active_count if active_count > 0 else 0.0
      self.metrics[f"error_vel_xy_{key_suffix}"] += (step_error_xy * in_bin_mask * scale) / max_command_step
      self.metrics[f"error_vel_yaw_{key_suffix}"] += (step_error_omega * in_bin_mask * scale) / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)

    self.vel_command_b[env_ids, 0] = r.uniform_(self.min_vx, self.max_vx)

    vx_samples = self.vel_command_b[env_ids, 0]
    max_vy = self._batched_interp(vx_samples, self.vy_xp, self.vy_yp)
    max_omega = self._batched_interp(vx_samples, self.omega_xp, self.omega_yp)

    self.vel_command_b[env_ids, 1] = r.uniform_(-1.0, 1.0) * max_vy
    self.vel_command_b[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(-1.0, 1.0) * max_omega

    if self.cfg.heading_command:
      assert self.cfg.heading is not None
      self.heading_target[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.heading)
      self.is_heading_env[env_ids] = (
        torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
      )

    self.is_standing_env[env_ids] = (
      torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    )

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      heading_error = wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
      proposed_omega = self.cfg.heading_control_stiffness * heading_error

      vx_cmds = self.vel_command_b[env_ids, 0]
      max_omega_for_vx = self._batched_interp(vx_cmds, self.omega_xp, self.omega_yp)
      self.vel_command_b[env_ids, 2] = torch.clamp(proposed_omega, min=-max_omega_for_vx, max=max_omega_for_vx)

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0


@dataclass(kw_only=True)
class HullVelocityCommandCfg(CommandTermCfg):
  """Configuration for the kinematic hull velocity command generator."""

  asset_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0

  vy_bound_points: list[list[float]] = field(
    default_factory=lambda: [[-1.0, 0.3], [0.0, 0.6], [1.5, 0.8], [3.0, 0.0]]
  )
  """[vx, max_vy] control points, vx strictly ascending."""

  omega_bound_points: list[list[float]] = field(
    default_factory=lambda: [[-1.0, 0.5], [0.0, 1.2], [1.0, 0.8], [3.0, 0.0]]
  )
  """[vx, max_omega_z] control points, vx strictly ascending."""

  velocity_bins: list[float] = field(default_factory=lambda: [0.0, 1.5, 3.0, 4.5])
  """Speed thresholds to bucket tracking-error metrics by commanded |vx|."""

  heading: tuple[float, float] | None = (-3.141592653589793, 3.141592653589793)

  def build(self, env: "ManagerBasedRlEnv") -> HullVelocityCommand:
    return HullVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.heading is None:
      raise ValueError("heading_command=True but heading is set to None.")
