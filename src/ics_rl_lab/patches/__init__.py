"""Local fixes to third-party code, applied when ics_rl_lab is imported."""

from . import mujoco_warp_hfield

mujoco_warp_hfield.apply()
