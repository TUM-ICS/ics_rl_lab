"""Registers the S2 AMP (flat-walk and flat-run) tasks."""

from ics_rl_lab.tasks.registry import register_task

from .agents.rl_amp_cfg import s2_amp_rl_cfg, s2_amp_run_rl_cfg
from .env_cfgs import s2_amp_env_cfg, s2_amp_run_env_cfg

register_task(
  task_id="Ics-AMP-Walk-S2",
  env_cfg_factory=lambda: s2_amp_env_cfg(play=False),
  play_env_cfg_factory=lambda: s2_amp_env_cfg(play=True),
  rl_cfg_factory=s2_amp_rl_cfg,
)

register_task(
  task_id="Ics-AMP-Run-S2",
  env_cfg_factory=lambda: s2_amp_run_env_cfg(play=False),
  play_env_cfg_factory=lambda: s2_amp_run_env_cfg(play=True),
  rl_cfg_factory=s2_amp_run_rl_cfg,
)
