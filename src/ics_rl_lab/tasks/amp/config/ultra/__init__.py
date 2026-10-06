"""Registers the Ultra AMP (flat-walk and flat-run) tasks."""

from ics_rl_lab.tasks.registry import register_task

from .agents.rl_amp_cfg import ultra_amp_rl_cfg, ultra_amp_run_rl_cfg
from .env_cfgs import ultra_amp_env_cfg, ultra_amp_run_env_cfg

register_task(
  task_id="Ics-AMP-Walk-Ultra",
  env_cfg_factory=lambda: ultra_amp_env_cfg(play=False),
  play_env_cfg_factory=lambda: ultra_amp_env_cfg(play=True),
  rl_cfg_factory=ultra_amp_rl_cfg,
)

register_task(
  task_id="Ics-AMP-Run-Ultra",
  env_cfg_factory=lambda: ultra_amp_run_env_cfg(play=False),
  play_env_cfg_factory=lambda: ultra_amp_run_env_cfg(play=True),
  rl_cfg_factory=ultra_amp_run_rl_cfg,
)
