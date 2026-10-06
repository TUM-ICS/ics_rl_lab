"""Registers the G1 AMP (flat-walk and flat-run) tasks."""

from ics_rl_lab.tasks.registry import register_task

from .agents.rl_amp_cfg import g1_amp_rl_cfg, g1_amp_run_rl_cfg
from .env_cfgs import g1_amp_env_cfg, g1_amp_run_env_cfg

register_task(
  task_id="Ics-AMP-Walk-G1",
  env_cfg_factory=lambda: g1_amp_env_cfg(play=False),
  play_env_cfg_factory=lambda: g1_amp_env_cfg(play=True),
  rl_cfg_factory=g1_amp_rl_cfg,
)

register_task(
  task_id="Ics-AMP-Run-G1",
  env_cfg_factory=lambda: g1_amp_run_env_cfg(play=False),
  play_env_cfg_factory=lambda: g1_amp_run_env_cfg(play=True),
  rl_cfg_factory=g1_amp_run_rl_cfg,
)
