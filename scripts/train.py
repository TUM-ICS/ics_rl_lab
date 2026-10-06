"""Train ics_rl policies on mjlab environments.

Deliberately minimal for a first pass: no multi-GPU/distributed handling (unlike
InstinctMJ's reference train.py, which this is modeled on) -- just enough to prove
the mjlab <-> ics_rl wiring end to end. Add that back if/when it's actually needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mjlab
import torch
import tyro
from ics_rl.env.mjlab import MjlabVecEnv
from ics_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

import ics_rl_lab.tasks  # noqa: F401  (import registers every task)
from ics_rl_lab.rl import MjlabRunnerCfg
from ics_rl_lab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: MjlabRunnerCfg
  num_envs: int | None = None
  device: str | None = None
  export_onnx: bool = True
  """Export the final policy (+ deployment metadata) to <log_dir>/exported/model_<iter>/
  when training ends, also when stopped with Ctrl+C."""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  configure_torch_backends()

  if cfg.num_envs is not None:
    cfg.env.scene.num_envs = cfg.num_envs
  cfg.agent.device = device

  print(f"[INFO] Task={task_id}, device={device}, num_envs={cfg.env.scene.num_envs}")
  print(f"[INFO] Logging to: {log_dir}")

  env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
  vec_env = MjlabVecEnv(
    env,
    policy_group=cfg.agent.policy_observation_group,
    critic_group=cfg.agent.critic_observation_group,
  )

  runner = OnPolicyRunner(vec_env, cfg.agent.to_dict(), log_dir=str(log_dir), device=device)

  log_dir.mkdir(parents=True, exist_ok=True)
  dump_yaml(log_dir / "params" / "agent.yaml", cfg.agent.to_dict())

  interrupted = False
  try:
    runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
  except KeyboardInterrupt:
    interrupted = True
    print("[INFO] Training interrupted, saving final checkpoint.")
    runner.save(str(log_dir / f"model_{runner.current_learning_iteration}.pt"))
  # learn() already exports on normal completion when save_interval_onnx is set.
  if cfg.export_onnx and (interrupted or runner.save_interval_onnx == -1):
    runner.export_onnx_checkpoint()
  vec_env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    config=mjlab.TYRO_FLAGS,
  )

  log_root = Path("logs") / "ics_rl" / cfg.agent.experiment_name
  log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_train(chosen_task, cfg, log_dir)


if __name__ == "__main__":
  main()
