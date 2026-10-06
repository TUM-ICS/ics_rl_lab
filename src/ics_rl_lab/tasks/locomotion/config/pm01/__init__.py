"""Registers the PM01 flat-ground velocity-tracking locomotion task."""

from ics_rl_lab.tasks.registry import register_task

from .agents.rl_ppo_cfg import pm01_locomotion_rl_cfg
from .env_cfgs import pm01_locomotion_env_cfg

register_task(
  task_id="Ics-Locomotion-Flat-PM01",
  env_cfg_factory=lambda: pm01_locomotion_env_cfg(play=False),
  play_env_cfg_factory=lambda: pm01_locomotion_env_cfg(play=True),
  rl_cfg_factory=pm01_locomotion_rl_cfg,
)
