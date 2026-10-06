"""pm01 foot skin sensors (left and right foot).

Each patch file goes on the body its ``base_frame`` names; p1e/p2e are corrected, see
``skin/README.md``.
"""

from __future__ import annotations

import os

from mjlab.sensor import ObjRef

from ics_rl_lab.sensors import SkinPatchPatternCfg, SkinSensorCfg

SKIN_DIR = os.path.join(os.path.dirname(__file__), "skin")
FOOT_SKIN_PATCHES = {
  "ankle_roll_l_link": ("p1e.json", "p2e.json"),
  "ankle_roll_r_link": ("p3e.json", "p4e.json"),
}
SKIN_OFFSET = 0.02
SKIN_MAX_DISTANCE = 0.5


def pm01_foot_skin_cfgs(
  offset: float = SKIN_OFFSET, max_distance: float = SKIN_MAX_DISTANCE, debug_vis: bool = False
) -> tuple[SkinSensorCfg, SkinSensorCfg]:
  """(left, right) foot skin sensors, named ``skin_foot_l`` / ``skin_foot_r``."""
  cfgs = []
  for body, files in FOOT_SKIN_PATCHES.items():
    side = body.split("_")[2]  # "l" / "r"
    cfgs.append(
      SkinSensorCfg(
        name=f"skin_foot_{side}",
        frame=ObjRef(type="body", name=body, entity="robot"),
        pattern=SkinPatchPatternCfg(
          cfg_files=tuple(os.path.join(SKIN_DIR, f) for f in files), offset=offset, expected_base_frame=body
        ),
        skin_max_distance=max_distance,
        debug_vis=debug_vis,
      )
    )
  return cfgs[0], cfgs[1]
