"""Exact surface height of mjlab heightfield terrain, batched in torch.

mjlab's generator lays sub-terrains out as a regular grid of equally sized tiles; the
heightfield ones are one hfield geom each. This reads the compiled hfields back and
evaluates the surface the way MuJoCo collides against it (`mjc_ConvexHField`, and
mujoco_warp's port of it): row -> y, column -> x, each cell split into two triangles
along the (r, c)-(r+1, c+1) diagonal. Checked against `mj_ray` to float precision.

Anything that is not an hfield tile (plane terrain, box-based sub-terrains, the
border) reads as z = 0.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch


class TerrainHeight:
  def __init__(self, model: mujoco.MjModel, device: str):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    tiles = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD]
    self.device = device
    self.empty = not tiles
    if self.empty:
      return

    hids = [int(model.geom_dataid[g]) for g in tiles]
    self.nrow, self.ncol = int(model.hfield_nrow[hids[0]]), int(model.hfield_ncol[hids[0]])
    self.sx, self.sy, self.sz = (float(v) for v in model.hfield_size[hids[0]][:3])
    for g, h in zip(tiles, hids):
      if (model.hfield_nrow[h], model.hfield_ncol[h]) != (self.nrow, self.ncol) or not np.allclose(
        model.hfield_size[h][:3], (self.sx, self.sy, self.sz)
      ):
        raise ValueError("TerrainHeight expects equally sized hfield tiles.")
      if not np.allclose(data.geom_xmat[g], np.eye(3).ravel()):
        raise ValueError("TerrainHeight expects axis-aligned hfield tiles.")

    # Tile (i, j) covers [x0 + 2*sx*i, x0 + 2*sx*(i+1)) x [y0 + 2*sy*j, ...).
    centers = np.array([data.geom_xpos[g] for g in tiles])
    self.x0 = float(centers[:, 0].min()) - self.sx
    self.y0 = float(centers[:, 1].min()) - self.sy
    ti = np.rint((centers[:, 0] - self.sx - self.x0) / (2 * self.sx)).astype(int)
    tj = np.rint((centers[:, 1] - self.sy - self.y0) / (2 * self.sy)).astype(int)
    self.ntx, self.nty = int(ti.max()) + 1, int(tj.max()) + 1

    grid = np.zeros((self.ntx, self.nty, self.nrow, self.ncol), dtype=np.float32)
    present = np.zeros((self.ntx, self.nty), dtype=bool)
    for g, h, i, j in zip(tiles, hids, ti, tj):
      adr = model.hfield_adr[h]
      heights = model.hfield_data[adr : adr + self.nrow * self.ncol].reshape(self.nrow, self.ncol)
      grid[i, j] = heights * self.sz + data.geom_xpos[g][2]
      present[i, j] = True
    self.grid = torch.from_numpy(grid).to(device)
    self.present = torch.from_numpy(present).to(device)
    self.dx = 2 * self.sx / (self.ncol - 1)
    self.dy = 2 * self.sy / (self.nrow - 1)

  def __call__(self, xy: torch.Tensor) -> torch.Tensor:
    """Surface height at world positions ``xy`` (..., 2) -> (...)."""
    if self.empty:
      return torch.zeros_like(xy[..., 0])
    x, y = xy[..., 0], xy[..., 1]
    ti = torch.floor((x - self.x0) / (2 * self.sx)).long()
    tj = torch.floor((y - self.y0) / (2 * self.sy)).long()
    inside = (ti >= 0) & (ti < self.ntx) & (tj >= 0) & (tj < self.nty)
    ti, tj = ti.clamp(0, self.ntx - 1), tj.clamp(0, self.nty - 1)
    inside &= self.present[ti, tj]

    u = (x - self.x0 - ti * 2 * self.sx) / self.dx  # column coordinate
    v = (y - self.y0 - tj * 2 * self.sy) / self.dy  # row coordinate
    c = torch.floor(u).long().clamp(0, self.ncol - 2)
    r = torch.floor(v).long().clamp(0, self.nrow - 2)
    fu, fv = (u - c).clamp(0, 1), (v - r).clamp(0, 1)
    g = self.grid
    h00, h01 = g[ti, tj, r, c], g[ti, tj, r, c + 1]
    h10, h11 = g[ti, tj, r + 1, c], g[ti, tj, r + 1, c + 1]
    tri_a = h00 + fv * (h10 - h00) + fu * (h11 - h10)  # (r,c), (r+1,c), (r+1,c+1)
    tri_b = h00 + fu * (h01 - h00) + fv * (h11 - h01)  # (r,c), (r,c+1), (r+1,c+1)
    h = torch.where(fv >= fu, tri_a, tri_b)
    return torch.where(inside, h, torch.zeros_like(h))


def box_bottom_points(half_size: np.ndarray, spacing: float, device: str) -> torch.Tensor:
  """Grid of points on a box's -z face, in the box frame, at most ``spacing`` apart."""
  nx = int(np.ceil(2 * half_size[0] / spacing)) + 1
  ny = int(np.ceil(2 * half_size[1] / spacing)) + 1
  gx = torch.linspace(-half_size[0], half_size[0], nx, device=device)
  gy = torch.linspace(-half_size[1], half_size[1], ny, device=device)
  X, Y = torch.meshgrid(gx, gy, indexing="ij")
  return torch.stack([X.ravel(), Y.ravel(), torch.full_like(X.ravel(), -float(half_size[2]))], -1)
