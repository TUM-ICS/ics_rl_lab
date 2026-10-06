"""Registers the PM01 hoverboard-driving task."""

from ics_rl_lab.tasks.registry import register_task

from .agents.rl_ppo_cfg import pm01_hoverboard_rl_cfg
from .env_cfgs import pm01_hoverboard_env_cfg

register_task(
  task_id="Ics-Hoverboard-Driving-PM01",
  env_cfg_factory=lambda: pm01_hoverboard_env_cfg(play=False),
  play_env_cfg_factory=lambda: pm01_hoverboard_env_cfg(play=True),
  rl_cfg_factory=pm01_hoverboard_rl_cfg,
)
