"""ics_rl AMPAlgoPPO configuration for pm01 AMP.
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
class AmpAlgoCfg(MjlabPpoAlgorithmCfg):
  class_name: str = "AMPAlgoPPO"
  discriminator_kwargs: dict = field(
    default_factory=lambda: {"hidden_sizes": [1024, 512, 256], "nonlinearity": "ReLU", "use_minibatch_std": True}
  )
  discriminator_reward_coef: float = 0.09
  discriminator_reward_type: str = "quad"
  discriminator_loss_func: str = "MSELoss"
  discriminator_gradient_penalty_coef: float = 10.0
  task_reward_lerp: float | None = 0.7
  discriminator_optimizer_class_name: str = "AdamW"
  discriminator_optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 1.0e-4, "betas": [0.9, 0.999]})
  optimizer_class_name: str = "Adam"
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  entropy_coef: float = 0.006
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1e-3
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0


@dataclass(kw_only=True)
class Pm01AmpFlatWalkRunnerCfg(MjlabRunnerCfg):
  policy: PolicyCfg = field(default_factory=PolicyCfg)
  algorithm: AmpAlgoCfg = field(default_factory=AmpAlgoCfg)
  num_steps_per_env: int = 24
  max_iterations: int = 20_000
  save_interval: int = 500
  log_interval: int = 1
  experiment_name: str = "pm01_amp_walk"


def pm01_amp_rl_cfg() -> MjlabRunnerCfg:
  return Pm01AmpFlatWalkRunnerCfg()


@dataclass(kw_only=True)
class Pm01AmpFlatRunRunnerCfg(Pm01AmpFlatWalkRunnerCfg):
  experiment_name: str = "pm01_amp_run"


def pm01_amp_run_rl_cfg() -> MjlabRunnerCfg:
  return Pm01AmpFlatRunRunnerCfg()
