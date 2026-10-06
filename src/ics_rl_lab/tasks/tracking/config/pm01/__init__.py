"""Registers the PM01 tracking tasks (resident / windowed motion storage)."""

from ics_rl_lab.tasks.registry import register_task

from .agents.rl_ppo_cfg import pm01_tracking_rl_cfg
from .env_cfgs import pm01_tracking_env_cfg

register_task(
  task_id="Ics-Tracking-Flat-PM01",
  env_cfg_factory=lambda: pm01_tracking_env_cfg(play=False),
  play_env_cfg_factory=lambda: pm01_tracking_env_cfg(play=True),
  rl_cfg_factory=pm01_tracking_rl_cfg,
)

register_task(
  task_id="Ics-Tracking-Flat-PM01-Windowed",
  env_cfg_factory=lambda: pm01_tracking_env_cfg(play=False, windowed=True),
  play_env_cfg_factory=lambda: pm01_tracking_env_cfg(play=True, windowed=True),
  rl_cfg_factory=pm01_tracking_rl_cfg,
)
