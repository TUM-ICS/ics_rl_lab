"""Registers the PM01 AMP (flat-walk and flat-run) tasks."""

from ics_rl_lab.tasks.registry import register_task

from .agents.rl_amp_cfg import pm01_amp_rl_cfg, pm01_amp_run_rl_cfg
from .env_cfgs import pm01_amp_env_cfg, pm01_amp_run_env_cfg

register_task(
  task_id="Ics-AMP-Walk-PM01",
  env_cfg_factory=lambda: pm01_amp_env_cfg(play=False),
  play_env_cfg_factory=lambda: pm01_amp_env_cfg(play=True),
  rl_cfg_factory=pm01_amp_rl_cfg,
)

register_task(
  task_id="Ics-AMP-Run-PM01",
  env_cfg_factory=lambda: pm01_amp_run_env_cfg(play=False),
  play_env_cfg_factory=lambda: pm01_amp_run_env_cfg(play=True),
  rl_cfg_factory=pm01_amp_run_rl_cfg,
)
