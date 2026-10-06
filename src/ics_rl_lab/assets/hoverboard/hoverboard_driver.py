"""Hoverboard firmware balancer, as a zero-dim mjlab `ActionTerm`.

Control law: per-side LQR with gains from Ackermann pole placement, plus yaw bias, pitch
integral and a delay on the sensed state. Implementation notes:

* Runs every physics step through `apply_actions()`.
* The plate state is computed from the board's *generalized coordinates* (qpos/qvel), not
  from `body_link_quat_w` / `body_link_ang_vel_w`. mjlab refreshes derived kinematics
  (xquat, cvel) only once per *policy* step, after the decimation loop; `mj_step` computes
  them before integrating. Inside the loop, where this term runs every physics step, they
  are fresh / duplicated / 1-step stale on substeps 0 / 1 / 2-3. Fed to the high-gain rate
  feedback, that irregular lag made the balancer destabilize the board (a passive rider
  fell faster with the balancer on than off).
* Sensed-state delay: mjlab's `DelayBuffer` with `hold_prob=1.0` (never resamples by
  itself) plus an explicit `set_lags()` on reset, so the lag is fixed per episode.
* `hb.set_joint_effort_target(torque)` goes to the wheels' `<motor>` actuators (see
  `hoverboard_cfg.py`). The plate joints have no actuator, so their (always zero) entries
  are ignored.

`torque` is (N, num_joints) in the *entity's* joint order, which for mjlab is depth-first
(left_plate, left_wheel, right_plate, right_wheel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.buffers.delay_buffer import DelayBuffer
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul

from ics_rl_lab.assets.hoverboard.hoverboard_cfg import (
  AXLE_WIDTH,
  PLATE_BODIES,
  PLATE_JOINTS,
  WHEEL_JOINTS,
  WHEEL_PEAK_TORQUE,
  WHEEL_RADIUS,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# ============================================================================
# Physical model parameters (inverted pendulum on wheels)
# These parameterize the *controller's* plant model, not the simulated board.
# ============================================================================

M_HB = 10.0  # hoverboard body mass [kg] (plant-model value; the sim board is 8.7 kg)
I_WHEEL = 0.00775  # rotational inertia per wheel [kg*m^2] (~0.5*2.35*0.0814^2)
M_HUMAN = 50.0  # humanoid mass [kg]
L_COM = 0.8  # humanoid CoM height above wheel axis [m]
G = 9.81  # [m/s^2]
WHEEL_TORQUE_LIMIT = WHEEL_PEAK_TORQUE  # [Nm] peak torque of YK_B_1972 (30)

__all__ = [
  "HoverboardActionTerm",
  "HoverboardActionTermCfg",
  "M_HB",
  "I_WHEEL",
  "M_HUMAN",
  "L_COM",
  "G",
  "WHEEL_RADIUS",
  "AXLE_WIDTH",
  "WHEEL_TORQUE_LIMIT",
]

# ============================================================================
# utils
# ============================================================================


def quat_to_pitch(quat: torch.Tensor) -> torch.Tensor:
  """Pitch angle from quaternion (w,x,y,z) -- handles any leading batch dims."""
  w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  return torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))


def _plant_matrices(M_c: float, m_p: float, L: float, R_w: float, g: float) -> tuple[np.ndarray, np.ndarray]:
  """Linearised EOM about upright equilibrium for state [theta, omega_wheel, theta_dot]:
      d/dt [theta    ]   [  0      0  1 ] [theta    ]   [     0      ]
           [omega    ] = [ a11     0  0 ] [omega    ] + [  1/(Mc R^2)] * tau
           [theta_dot]   [ a21     0  0 ] [theta_dot]   [ -1/(Mc L R)]

  a11 = -mp*g / (Mc*R)     (cart acceleration coupling, stable)
  a21 = (Mc+mp)*g / (Mc*L) (unstable pendulum pole)
  """
  a11 = -(m_p * g) / (M_c * R_w)
  a21 = (M_c + m_p) * g / (M_c * L)
  b1 = 1.0 / (M_c * R_w**2)
  b2 = -1.0 / (M_c * L * R_w)

  A = np.array([[0.0, 0.0, 1.0], [a11, 0.0, 0.0], [a21, 0.0, 0.0]])
  B = np.array([[0.0], [b1], [b2]])
  return A, B


def _ackermann_placement_vectors(A: np.ndarray, B: np.ndarray) -> torch.Tensor:
  """Precompute 4 constant row-vectors [r0, r1, r2, r3] such that, for any desired monic
  characteristic polynomial s^3 + a2*s^2 + a1*s + a0, the state-feedback gain K (for
  u = -K @ x) placing eig(A - B@K) at the roots of that polynomial is:
      K = a0*r0 + a1*r1 + a2*r2 + r3
  This is Ackermann's formula in closed form, so per-environment gains can be computed
  with one batched linear-algebra expression instead of calling a pole-placement solver
  once per environment. Every resulting K is stabilizing for the nominal plant by
  construction, unlike sampling each gain component independently.
  """
  n = A.shape[0]
  ctrb = np.hstack([B, A @ B, A @ A @ B])
  ctrb_inv = np.linalg.inv(ctrb)
  e_last = np.zeros((1, n))
  e_last[0, -1] = 1.0
  w = e_last @ ctrb_inv
  rows = np.stack(
    [
      (w @ np.eye(n)).flatten(),
      (w @ A).flatten(),
      (w @ (A @ A)).flatten(),
      (w @ (A @ A @ A)).flatten(),
    ]
  )
  return torch.tensor(rows, dtype=torch.float32)  # (4, 3): rows are [r0, r1, r2, r3]


# ============================================================================
# ActionTerm
# ============================================================================


class HoverboardActionTerm(ActionTerm):
  """ActionTerm that runs two independent per-side hoverboard LQR controllers.

  Each side reads its own plate's world-frame pitch and wheel velocity, matching
  the real board's firmware architecture. Steering emerges naturally from the
  differential plate pitch (driven by the robot's ankle joints via passive plate
  joints), with no explicit rotation controller.

  Called automatically by ActionManager.apply_action() before every physics step.
  The RL policy has no direct action inputs here (action_dim=0).
  """

  cfg: HoverboardActionTermCfg
  _entity: Entity

  def __init__(self, cfg: HoverboardActionTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)  # resolves self._entity via cfg.entity_name

    hb = self._entity
    N = self.num_envs
    dev = self.device

    # -- Gains (shared across both sides) --------------------------------
    # Randomization samples desired closed-loop poles (wn, zeta, and a 3rd real pole)
    # and synthesizes K via Ackermann pole placement, so every draw is guaranteed
    # stabilizing for the nominal plant -- unlike sampling each gain component
    # independently, which can silently draw non-stabilizing (or delay-fragile)
    # combinations. wn/zeta are also the two knobs to reach for when reasoning about
    # target dynamics (zeta -> how oscillatory, wn -> how fast).
    A, B = _plant_matrices(M_HB, M_HUMAN, L_COM, WHEEL_RADIUS, G)
    self._ackermann_rows = _ackermann_placement_vectors(A, B).to(dev)  # (4, 3): [r0,r1,r2,r3]

    self._wn_range = cfg.lqr_wn_range
    self._zeta_range = cfg.lqr_zeta_range
    self._third_pole_ratio = cfg.lqr_third_pole_ratio
    self._third_pole_floor = cfg.lqr_third_pole_floor

    self.K_lqr = torch.zeros(N, 3, device=dev)

    # -- Sensed-state input delay (IMU/estimator latency the internal LQR loop acts on) --
    # Delays the *sensed* [pitch, wheel_omega, pitch_dot] fed into the control law, not the
    # true physical state used for logging/observations/rewards -- mirrors a real firmware
    # loop reacting to stale sensor data while the physical board has already moved on.
    # hold_prob=1.0: the lag never resamples on its own; `_resample_input_delay` sets it
    # once per episode.
    self._input_delay_range = cfg.lqr_input_delay_range
    if self._input_delay_range is not None:
      lo, hi = int(self._input_delay_range[0]), int(self._input_delay_range[1])
      self._state_delay_buffer: DelayBuffer | None = DelayBuffer(
        min_lag=lo, max_lag=hi, batch_size=N, device=dev, hold_prob=1.0
      )
    else:
      self._state_delay_buffer = None
    self._all_env_ids = torch.arange(N, device=dev)

    # -- Per-side pitch integral (models PI firmware behavior) -----------
    self._K_int_nominal = torch.full((N,), cfg.nominal_integral_gain, device=dev)
    self.K_int = self._K_int_nominal.clone()
    self._int_noise_range = cfg.gain_noise_range_integral or [1.0, 1.0]
    self._integral_clamp = cfg.integral_clamp
    self._pitch_integral = torch.zeros(N, 2, device=dev)  # (N, [left, right])
    self._dt = env.physics_dt

    # -- Pre-allocated output buffers ------------------------------------
    self._state = torch.zeros(N, 3, device=dev)  # avg state for logging
    self._torque = torch.zeros(N, hb.num_joints, device=dev)

    # -- Per-episode yaw bias (firmware yaw-zero offset sim lacks) --------
    self._yaw_bias_range = cfg.yaw_bias_range or [0.0, 0.0]
    self._yaw_bias = torch.zeros(N, device=dev)
    self._resample_yaw_bias(self._all_env_ids)
    self._resample_input_delay(self._all_env_ids)

    # -- Body/joint index caches (left, right) ----------------------------
    self._plate_idx = hb.find_bodies(PLATE_BODIES, preserve_order=True)[0]
    self._wheel_idx = hb.find_joints(WHEEL_JOINTS, preserve_order=True)[0]
    plate_joint_idx = hb.find_joints(PLATE_JOINTS, preserve_order=True)[0]
    # Raw qpos/qvel addresses (always current, unlike derived xquat/cvel -- see module docstring).
    idx = hb.data.indexing
    self._base_quat_adr = idx.free_joint_q_adr[3:7]
    self._base_angvel_adr = idx.free_joint_v_adr[3:6]  # free-joint angular velocity, *local* frame
    self._plate_q_adr = idx.joint_q_adr[plate_joint_idx]
    self._plate_v_adr = idx.joint_v_adr[plate_joint_idx]
    self._wheel_v_adr = idx.joint_v_adr[self._wheel_idx]
    self._y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev)

    self._empty_action = torch.empty(N, 0, device=dev)

    self.randomize_gains()

  # -- Gain helpers --------------------------------------------------------

  def randomize_gains(self, env_ids: torch.Tensor | None = None) -> None:
    """Re-sample per-instance closed-loop poles (wn, zeta) and synthesize LQR gains via
    Ackermann pole placement (call at each episode reset).

    Only touches `env_ids` (default: all envs) so that resetting one env does not
    change the plant dynamics of every other still-running env mid-episode.
    """
    if env_ids is None:
      env_ids = self._all_env_ids
    n = len(env_ids)
    dev = self.K_lqr.device

    wn = torch.empty(n, device=dev).uniform_(*self._wn_range)
    zeta = torch.empty(n, device=dev).uniform_(*self._zeta_range)
    p3 = -self._third_pole_ratio * torch.clamp(zeta * wn, min=self._third_pole_floor)

    # Coefficients of the desired monic characteristic polynomial s^3 + a2*s^2 + a1*s + a0,
    # for the complex pair -zeta*wn +- j*wn*sqrt(1-zeta^2) and the real pole p3.
    a2 = 2.0 * zeta * wn - p3
    a1 = wn**2 - 2.0 * zeta * wn * p3
    a0 = -(wn**2) * p3

    coeffs = torch.stack([a0, a1, a2, torch.ones_like(a0)], dim=-1)  # (n, 4)
    self.K_lqr[env_ids] = coeffs @ self._ackermann_rows  # (n, 4) @ (4, 3) -> (n, 3)

    int_noise = torch.empty(n, device=self.K_int.device).uniform_(*self._int_noise_range)
    self.K_int[env_ids] = self._K_int_nominal[env_ids] * int_noise

  def _resample_yaw_bias(self, env_ids: torch.Tensor) -> None:
    lo, hi = self._yaw_bias_range
    if hi == 0.0:
      self._yaw_bias[env_ids] = 0.0
      return
    n = len(env_ids)
    mag = torch.empty(n, device=self._yaw_bias.device).uniform_(lo, hi)
    sign = torch.randint(0, 2, (n,), device=self._yaw_bias.device).float() * 2.0 - 1.0
    self._yaw_bias[env_ids] = mag * sign

  def _resample_input_delay(self, env_ids: torch.Tensor) -> None:
    if self._state_delay_buffer is None:
      return
    lo, hi = self._input_delay_range
    n = len(env_ids)
    time_lag = torch.randint(int(lo), int(hi) + 1, (n,), device=env_ids.device, dtype=torch.long)
    self._state_delay_buffer.reset(env_ids)
    self._state_delay_buffer.set_lags(time_lag, batch_ids=env_ids)

  # -- Custom property access (used by reward/obs terms) --------------------

  @property
  def state(self) -> torch.Tensor:
    return self._state

  @property
  def torque(self) -> torch.Tensor:
    return self._torque

  # -- ActionTerm interface -------------------------------------------------

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._empty_action

  def process_actions(self, actions: torch.Tensor) -> None:
    pass

  def apply_actions(self) -> None:
    hb = self._entity
    N = self.num_envs
    qpos, qvel = hb.data.data.qpos, hb.data.data.qvel

    # Plate world orientation = base orientation * rotation about the plate hinge (base y).
    # The plate bodies carry no fixed rotation relative to base_link (hoverboard.xml).
    base_quat = qpos[:, self._base_quat_adr]  # (N, 4)
    plate_q = qpos[:, self._plate_q_adr]  # (N, 2)
    hinge_quat = quat_from_angle_axis(plate_q.reshape(-1), self._y_axis)  # (2N, 4)
    plate_quats = quat_mul(base_quat.repeat_interleave(2, dim=0), hinge_quat).view(N, 2, 4)

    pitches = quat_to_pitch(plate_quats)  # (N, 2)

    # Plate body-frame Y angular velocity: a rotation about y leaves the y component alone,
    # so it is the base's local y rate plus the hinge rate.
    pitch_dot = qvel[:, self._base_angvel_adr][:, 1:2] + qvel[:, self._plate_v_adr]  # (N, 2)

    wheel = qvel[:, self._wheel_v_adr]  # (N, 2)

    # Log average TRUE (undelayed) state for observations / rewards
    self._state[:, 0] = 0.5 * (pitches[:, 0] + pitches[:, 1])
    self._state[:, 1] = 0.5 * (wheel[:, 0] + wheel[:, 1])
    self._state[:, 2] = 0.5 * (pitch_dot[:, 0] + pitch_dot[:, 1])

    # The control law itself acts on the (possibly stale) sensed state -- a real firmware
    # loop reacts to what its IMU/estimator reported, not the instantaneous physical state.
    if self._state_delay_buffer is not None:
      sensed = torch.cat([pitches, wheel, pitch_dot], dim=-1)  # (N, 6)
      self._state_delay_buffer.append(sensed)
      sensed = self._state_delay_buffer.compute()
      pitches_ctrl, wheel_ctrl, pitch_dot_ctrl = sensed[:, 0:2], sensed[:, 2:4], sensed[:, 4:6]
    else:
      pitches_ctrl, wheel_ctrl, pitch_dot_ctrl = pitches, wheel, pitch_dot

    # Accumulate per-side pitch integral with anti-windup, on the same (possibly stale)
    # sensed pitch the rest of the control law uses -- the firmware only ever sees its own
    # sensor stream, so its internal integrator lags along with everything else.
    self._pitch_integral += pitches_ctrl * self._dt
    self._pitch_integral.clamp_(-self._integral_clamp, self._integral_clamp)

    # Two independent PI controllers -- one per plate/wheel
    # tau = -K @ [pitch, omega_wheel, pitch_dot] - K_int * integral(pitch)
    K = self.K_lqr
    tau_L = -(
      pitches_ctrl[:, 0] * K[:, 0]
      + wheel_ctrl[:, 0] * K[:, 1]
      + pitch_dot_ctrl[:, 0] * K[:, 2]
      + self._pitch_integral[:, 0] * self.K_int
    )
    tau_R = -(
      pitches_ctrl[:, 1] * K[:, 0]
      + wheel_ctrl[:, 1] * K[:, 1]
      + pitch_dot_ctrl[:, 1] * K[:, 2]
      + self._pitch_integral[:, 1] * self.K_int
    )

    # Yaw bias: differential offset models real firmware yaw-zero offset
    tau_L = tau_L + self._yaw_bias
    tau_R = tau_R - self._yaw_bias

    self._torque[:, self._wheel_idx[0]] = tau_L.clamp(-WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
    self._torque[:, self._wheel_idx[1]] = tau_R.clamp(-WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
    hb.set_joint_effort_target(self._torque)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = self._all_env_ids[env_ids if isinstance(env_ids, slice) else slice(None)]
    self._state[env_ids] = 0.0
    self._torque[env_ids] = 0.0
    self._pitch_integral[env_ids] = 0.0
    self._resample_yaw_bias(env_ids)
    self._resample_input_delay(env_ids)
    self.randomize_gains(env_ids)


@dataclass(kw_only=True)
class HoverboardActionTermCfg(ActionTermCfg):
  """Configuration for :class:`HoverboardActionTerm`."""

  entity_name: str = "hoverboard"

  lqr_wn_range: list[float] = field(default_factory=lambda: [2.0, 8.0])
  """[min, max] natural frequency (rad/s) of the desired closed-loop pole pair. Governs how
  fast the board reacts to a pitch disturbance. Upper bound is calibrated so that, combined
  with lqr_zeta_range, the resulting gains stay within WHEEL_TORQUE_LIMIT at representative
  operating amplitudes -- high wn *and* high zeta together demands more torque than the motor
  can deliver, at which point the controller just saturates and the pole-placement guarantee
  no longer describes the (now-nonlinear, clipped) closed loop."""

  lqr_zeta_range: list[float] = field(default_factory=lambda: [0.05, 0.5])
  """[min, max] damping ratio of the desired closed-loop pole pair. Low zeta (<~0.2) gives a
  visibly oscillatory board (matches real hardware behavior) but has little margin against
  control-loop delay -- keep the low end of this range only when input delay is disabled, or
  raise the floor once delay is modeled on this loop. Upper bound kept at 0.5 (rather than up
  to 1.0) to stay within torque budget together with lqr_wn_range -- see its docstring."""

  lqr_third_pole_ratio: float = 1.5
  """The 3rd (real) closed-loop pole is placed at -ratio * max(zeta*wn, lqr_third_pole_floor),
  i.e. left of the dominant pole pair so it has limited influence on the dominant response.
  Larger ratios need disproportionately more gain (and thus torque) for little dynamical
  benefit here -- 1.5 was chosen as part of the torque-budget calibration above."""

  lqr_third_pole_floor: float = 1.0
  """Minimum magnitude (rad/s) used for max(zeta*wn, floor) when placing the 3rd pole, so it
  doesn't degenerate to the origin when zeta*wn -> 0."""

  lqr_input_delay_range: list[int] | None = None
  """[min, max] delay (in physics steps) applied to the sensed [pitch, wheel_omega, pitch_dot]
  fed into the LQR control law -- models IMU/estimator latency in the board's own firmware
  loop, separate from any delay on the robot's own joint actuators. Resampled per episode.
  None = no delay (default, matches the delay-free variant). Delay doesn't change the gain,
  it erodes phase margin -- its effect is concentrated on the low-zeta (already lightly
  damped) end of lqr_zeta_range, since that's where the least margin exists to begin with."""

  nominal_integral_gain: float = 0.0
  """Per-side pitch integral gain K_i [Nm / (rad*s)]. Models PI firmware behavior:
  a sustained plate pitch accumulates torque, causing slow yaw buildup if uncorrected."""

  gain_noise_range_integral: list = field(default_factory=list)
  """Multiplicative noise range [low, high] for integral gain. [1.0, 1.0] = deterministic."""

  integral_clamp: float = 0.5
  """Anti-windup clamp on the integral state [rad*s]. Max torque contribution = K_i * clamp."""

  yaw_bias_range: list[float] | None = None
  """[min, max] magnitude of per-episode additive yaw-torque bias [Nm].
  Sign is randomized independently. Resampled each reset. Models the real
  board's firmware yaw-zero offset, forcing the policy to close the loop on
  yaw rather than memorize a fixed ankle posture.
  None or [0, 0] = disabled."""

  def build(self, env: ManagerBasedRlEnv) -> HoverboardActionTerm:
    return HoverboardActionTerm(self, env)
