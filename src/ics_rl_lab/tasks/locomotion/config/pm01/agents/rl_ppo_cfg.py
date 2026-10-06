"""ics_rl PPO configuration for pm01 locomotion (`MjlabRunnerCfg` dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ics_rl_lab.rl.config import MjlabActorCriticCfg, MjlabPpoAlgorithmCfg, MjlabRunnerCfg


@dataclass(kw_only=True)
class PolicyCfg(MjlabActorCriticCfg):
  init_noise_std: float = 1.0
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
  activation: str = "elu"


@dataclass(kw_only=True)
class AlgorithmCfg(MjlabPpoAlgorithmCfg):
  class_name: str = "PPO"
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  entropy_coef: float = 0.008
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1e-3
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0


@dataclass(kw_only=True)
class Pm01LocomotionFlatWalkRunnerCfg(MjlabRunnerCfg):
  policy: PolicyCfg = field(default_factory=PolicyCfg)
  algorithm: AlgorithmCfg = field(default_factory=AlgorithmCfg)
  num_steps_per_env: int = 24
  max_iterations: int = 20_000
  save_interval: int = 1_000
  log_interval: int = 1
  experiment_name: str = "pm01_locomotion_flat_walk"


def pm01_locomotion_rl_cfg() -> MjlabRunnerCfg:
  return Pm01LocomotionFlatWalkRunnerCfg()
