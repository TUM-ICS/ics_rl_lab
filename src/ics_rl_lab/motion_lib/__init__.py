"""Reference-motion data for the tasks, all reading the pre-baked retargeted ``.npz`` clips.

- ``motion_library``: per-env clip playback for tracking (resident or windowed storage).
- ``amp_motion_loader``: (s, s') expert mini-batches for AMP discriminators.
- ``amp_motion_loader_sensor``: exposes an ``AMPMotionLoader`` as ``scene["motionloader"]``,
  where ics_rl's ``OnPolicyRunner`` picks it up.
"""

from .amp_motion_loader import AMPMotionLoader
from .amp_motion_loader_sensor import MotionLoaderSensor, MotionLoaderSensorCfg
from .paths import MOTION_DATA_ENV, motion_data_dir
from .motion_library import (
  MotionFrames,
  MotionLibrary,
  MotionLibraryCfg,
  ResidentMotionLibrary,
  ResidentMotionLibraryCfg,
  WindowedMotionLibrary,
  WindowedMotionLibraryCfg,
  discover_motion_files,
)
