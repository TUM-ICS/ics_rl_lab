"""Hoverboard termination terms."""

from __future__ import annotations

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


def feet_outside_plate(
  env,
  nominal_offsets: list[list[float]],
  max_offset_x: float = 0.10,
  max_offset_y: float = 0.08,
  feet_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("hoverboard"),
) -> torch.Tensor:
  """Terminate if any foot slides outside the plate bounds, measured in the plate's frame.

  Foot i is measured relative to plate i, so the body order of `plate_asset_cfg` must
  match `feet_asset_cfg` and `nominal_offsets` (use `preserve_order=True` on both).

  Uses the plate's full orientation, not only its yaw: the foot body sits
  ~8.5 cm above the plate's hinge axis, so in a yaw-only frame a plate pitched by
  0.15-0.3 rad (normal when driving fast) moved a foot that had not slid by 1.5-2.5 cm
  towards the limit. In the full plate frame only real slip on the plate counts.
  """
  hoverboard = env.scene[plate_asset_cfg.name]
  robot = env.scene[feet_asset_cfg.name]

  feet_pos_w = robot.data.body_link_pos_w[:, feet_asset_cfg.body_ids, :3]  # (N, F, 3)
  N, F = feet_pos_w.shape[:2]

  plate_pos_w = hoverboard.data.body_link_pos_w[:, plate_asset_cfg.body_ids, :3]  # (N, F, 3)
  plate_quat_w = hoverboard.data.body_link_quat_w[:, plate_asset_cfg.body_ids, :]  # (N, F, 4)

  feet_rel_hb = quat_apply_inverse(plate_quat_w.reshape(-1, 4), (feet_pos_w - plate_pos_w).reshape(-1, 3)).view(N, F, 3)

  nominal = torch.tensor([[o[0], o[1]] for o in nominal_offsets], device=env.device, dtype=feet_pos_w.dtype)
  err = feet_rel_hb[..., :2] - nominal.unsqueeze(0)  # (N, F, 2)
  out_of_bounds = (err[..., 0].abs() > max_offset_x) | (err[..., 1].abs() > max_offset_y)
  return torch.any(out_of_bounds, dim=-1)
