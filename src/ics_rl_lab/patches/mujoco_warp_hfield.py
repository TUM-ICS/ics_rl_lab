"""Remove mujoco_warp's height-field witness "correction" in EPA (bogus terrain contacts).

`collision_gjk._epa_witness` special-cases height-field prisms: when the EPA face mixes
top and bottom prism vertices it replaces the witness points by the support point of
geom2 in direction ``normalize(x2)`` -- a *position* in the height-field frame, i.e.
roughly "away from the tile centre" -- projects it vertically onto the prism top and
returns ``-|x1 - x2|``. A box corner *above* the surface thus becomes a penetration
as deep as the gap, with the normal pointing down. On the gravel terrain ~0.7% of the
pm01 foot-terrain contacts were such phantoms (up to 210 mm too deep, 30-67 kN spikes).

MuJoCo C has no such branch (each prism is a plain convex polytope) and is correct on
the same states, so the patch compiles the branch out. Present in mujoco_warp 3.6.0
through 3.14.0. Applied at `ics_rl_lab` import, before any kernel is built.
"""

from __future__ import annotations

import inspect
import linecache
import textwrap

import warp as wp
from mujoco_warp._src import collision_gjk

_HFIELD_BRANCH = "if geomtype1 == GeomType.HFIELD and (i1 != i2 or i1 != i3):"


def apply() -> None:
  if getattr(collision_gjk, "_ics_hfield_witness_patched", False):
    return
  fn = collision_gjk._epa_witness
  src = textwrap.dedent(inspect.getsource(fn.func))
  if src.count(_HFIELD_BRANCH) != 1:
    raise RuntimeError(
      "mujoco_warp's _epa_witness changed; re-check the height-field witness bug "
      "before updating ics_rl_lab/patches/mujoco_warp_hfield.py."
    )
  src = src.replace(_HFIELD_BRANCH, "if wp.static(False):  # ics_rl_lab: hfield witness correction removed")
  src = src.replace("@wp.func\n", "", 1)
  # Warp re-reads function source via inspect, so the patched code needs a linecache entry.
  filename = "<ics_rl_lab.patches.mujoco_warp_hfield._epa_witness>"
  linecache.cache[filename] = (len(src), None, src.splitlines(keepends=True), filename)
  namespace = collision_gjk.__dict__
  exec(compile(src, filename, "exec"), namespace)
  collision_gjk._epa_witness = wp.func(namespace["_epa_witness"])
  collision_gjk._ics_hfield_witness_patched = True
