"""ics_rl PPO configuration for pm01 tracking.

EncoderActorCritic: the actor's 11-frame ``*_hist`` terms go through one Conv1d head
(64), the critic's 20-frame ``future_motion`` lookahead through another (64); both
latents are concatenated with the remaining (current-frame) terms into the MLPs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ics_rl_lab.rl.config import MjlabActorCriticCfg, MjlabPpoAlgorithmCfg, MjlabRunnerCfg
from ics_rl_lab.tasks.tracking.tracking_env_cfg import _policy_terms

# The (T, D) history terms of the policy group -- derived from the env cfg so they stay in sync.
ACTOR_HISTORY_COMPONENTS = [f"{name}_hist" for name in _policy_terms()]


def _conv1d_head(components: list[str], output_size: int = 64) -> dict[str, Any]:
  return {
    "class_name": "Conv1dHeadModel",
    "component_names": components,
    "output_size": output_size,
    "takeout_input_components": True,
    "nonlinearity": "elu",
    "channel_size": 20,
  }


@dataclass(kw_only=True)
class TrackingPolicyCfg(MjlabActorCriticCfg):
  class_name: str = "EncoderActorCritic"
  init_noise_std: float = 1.0
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
  activation: str = "elu"
  encoder_configs: dict[str, Any] = field(
    default_factory=lambda: {"proprio_history": _conv1d_head(ACTOR_HISTORY_COMPONENTS)}
  )
  critic_encoder_configs: dict[str, Any] | None = field(
    default_factory=lambda: {"future_motion": _conv1d_head(["future_motion"])}
  )


@dataclass(kw_only=True)
class TrackingAlgorithmCfg(MjlabPpoAlgorithmCfg):
  class_name: str = "PPO"
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  entropy_coef: float = 0.005
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1e-3
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0


@dataclass(kw_only=True)
class Pm01TrackingRunnerCfg(MjlabRunnerCfg):
  policy: TrackingPolicyCfg = field(default_factory=TrackingPolicyCfg)
  algorithm: TrackingAlgorithmCfg = field(default_factory=TrackingAlgorithmCfg)
  normalizers: dict[str, dict[str, Any]] | None = field(
    default_factory=lambda: {
      "policy": {"class_name": "EmpiricalNormalization"},
      "critic": {"class_name": "EmpiricalNormalization"},
    }
  )
  num_steps_per_env: int = 24
  max_iterations: int = 30_000
  save_interval: int = 1000
  log_interval: int = 10
  experiment_name: str = "pm01_tracking_wb"


def pm01_tracking_rl_cfg() -> MjlabRunnerCfg:
  return Pm01TrackingRunnerCfg()
