"""Task registry for ics_rl-trained mjlab tasks.

Entry-point-string based (module.path:ClassName), same pattern InstinctMJ proved
useful: a downstream task can point at its own env/vecenv/runner classes without
this registry importing anything eagerly.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from mjlab.envs import ManagerBasedRlEnvCfg

from ics_rl_lab.rl import MjlabRunnerCfg

DEFAULT_ENV_ENTRY_POINT = "mjlab.envs:ManagerBasedRlEnv"
DEFAULT_VECENV_ENTRY_POINT = "ics_rl.env.mjlab:MjlabVecEnv"


@dataclass
class _TaskCfg:
  env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg]
  play_env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg]
  rl_cfg_factory: Callable[[], MjlabRunnerCfg]
  env_entry_point: str
  vecenv_entry_point: str
  runner_entry_point: str | None


_REGISTRY: dict[str, _TaskCfg] = {}


def _resolve_entry_point(entry_point: str) -> type:
  module_path, _, class_name = entry_point.partition(":")
  return getattr(importlib.import_module(module_path), class_name)


def register_task(
  task_id: str,
  env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg],
  play_env_cfg_factory: Callable[[], ManagerBasedRlEnvCfg],
  rl_cfg_factory: Callable[[], MjlabRunnerCfg],
  env_entry_point: str = DEFAULT_ENV_ENTRY_POINT,
  vecenv_entry_point: str = DEFAULT_VECENV_ENTRY_POINT,
  runner_entry_point: str | None = None,
) -> None:
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered.")
  _REGISTRY[task_id] = _TaskCfg(
    env_cfg_factory,
    play_env_cfg_factory,
    rl_cfg_factory,
    env_entry_point,
    vecenv_entry_point,
    runner_entry_point,
  )


def list_tasks() -> list[str]:
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  task_cfg = _REGISTRY[task_name]
  return task_cfg.play_env_cfg_factory() if play else task_cfg.env_cfg_factory()


def load_rl_cfg(task_name: str) -> MjlabRunnerCfg:
  return _REGISTRY[task_name].rl_cfg_factory()


def load_env_cls(task_name: str) -> type:
  return _resolve_entry_point(_REGISTRY[task_name].env_entry_point)


def load_vecenv_cls(task_name: str) -> type:
  return _resolve_entry_point(_REGISTRY[task_name].vecenv_entry_point)


def load_runner_cls(task_name: str) -> type | None:
  entry_point = _REGISTRY[task_name].runner_entry_point
  if entry_point is None:
    return None
  return _resolve_entry_point(entry_point)
