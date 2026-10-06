"""Location of the motion datasets.

Motion data is not part of the repository. It lives under one data root:
``$ICS_MOTION_DATA`` if set, else ``<repo>/data/motions`` (gitignored). Expected layout:

  <root>/pm01/amp/*.npz          AMP expert clips (walk + run)
  <root>/s2/amp/*.npz            S2 AMP expert clips
  <root>/ultra/amp/*.npz         Ultra AMP expert clips
  <root>/g1/amp/*.npz            G1 AMP expert clips
  <root>/pm01/seed_motion/*.npz  retargeted seed set for tracking (48k clips, ~8 GB)

Symlinks are fine (e.g. seed_motion -> an existing checkout).
"""

from __future__ import annotations

import os
from pathlib import Path

MOTION_DATA_ENV = "ICS_MOTION_DATA"
DEFAULT_MOTION_DATA = Path(__file__).resolve().parents[3] / "data" / "motions"


def motion_data_dir(*parts: str) -> Path:
  """Path below the motion data root. Not checked for existence (tasks without motions
  must keep registering on machines without the data)."""
  root = Path(os.environ.get(MOTION_DATA_ENV, DEFAULT_MOTION_DATA)).expanduser()
  return root.joinpath(*parts)
