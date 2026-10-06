"""mjlab `MujocoCfg` with the extra MuJoCo options our tasks need."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
from mjlab.sim import MujocoCfg


@dataclass
class IcsMujocoCfg(MujocoCfg):
  box_box_primitive: bool = True
  """Collide box pairs with the analytic `box_box` routine instead of GJK/EPA.

  mujoco_warp's float32 GJK/EPA drops box-box contacts when the sizes differ a lot:
  pm01's foot boxes (~0.2 m) on mjlab's terrain border boxes (120 x 20 x 1 m) got one
  contact instead of MuJoCo C's six, intermittently none, and the feet sank 10-18 cm
  into the border. mujoco_warp switches box-box to the primitive collider when
  `mjDSBL_NATIVECCD` is set; that is the flag's only effect there. (On the CPU model it
  also selects libccd for convex pairs -- only CPU-side tools see that.)
  """

  def apply(self, model: mujoco.MjModel) -> None:
    super().apply(model)
    if self.box_box_primitive:
      model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
