"""Runner-config dataclasses for training mjlab tasks with ics_rl.

These mirror ics_rl's ``OnPolicyRunner`` expected ``train_cfg`` dict shape
(``policy``/``algorithm``/... keys) the same way mjlab's own ``mjlab.rl.config``
mirrors rsl_rl's -- plain, framework-agnostic dataclasses that live with the
project consuming both libraries, not inside either library itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


def _to_plain_dict(obj: Any) -> Any:
  if is_dataclass(obj):
    result = {}
    for f in fields(obj):
      value = getattr(obj, f.name)
      if value is None:
        continue
      result[f.name] = _to_plain_dict(value)
    return result
  if isinstance(obj, dict):
    return {k: _to_plain_dict(v) for k, v in obj.items()}
  if isinstance(obj, (tuple, list)):
    return [_to_plain_dict(v) for v in obj]
  return obj


@dataclass(kw_only=True)
class MjlabActorCriticCfg:
  class_name: str = "ActorCritic"
  init_noise_std: float = 1.0
  actor_hidden_dims: tuple[int, ...] = (256, 128, 128)
  critic_hidden_dims: tuple[int, ...] = (256, 128, 128)
  activation: str = "elu"


@dataclass(kw_only=True)
class MjlabPpoAlgorithmCfg:
  class_name: str = "PPO"
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  entropy_coef: float = 0.005
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1.0e-3
  optimizer_class_name: str = "Adam"
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0


@dataclass(kw_only=True)
class MjlabRunnerCfg:
  seed: int = 42
  device: str = "cuda:0"
  num_steps_per_env: int = 24
  max_iterations: int = 30_000
  policy: MjlabActorCriticCfg = field(default_factory=MjlabActorCriticCfg)
  algorithm: MjlabPpoAlgorithmCfg = field(default_factory=MjlabPpoAlgorithmCfg)
  save_interval: int = 500
  save_interval_onnx: int = -1
  """Also export ONNX every N iterations to <log_dir>/exported/model_<iter>/; -1 disables.
  Independent of this, train.py exports once when training ends (--export-onnx)."""
  log_interval: int = 1
  logger: str = "tensorboard"
  wandb_project: str = "ics_rl_lab"
  experiment_name: str = "ics_rl_lab"
  run_name: str = ""
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  policy_observation_group: str = "policy"
  critic_observation_group: str = "critic"

  def to_dict(self) -> dict[str, Any]:
    return _to_plain_dict(self)
