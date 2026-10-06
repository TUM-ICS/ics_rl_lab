"""Custom locomotion event terms not covered by mjlab's stock `mjlab.envs.mdp`
(reset by offset exists there; the locomotion task resets by *scale*),
plus `place_on_terrain`, which puts reset robots on the actual terrain surface.

Mass/COM/friction/PD-gain randomization all reuse mjlab's stock `mjlab.envs.mdp.dr`
functions directly in `locomotion_env_cfg.py`.
"""

from __future__ import annotations

import mujoco
import numpy as np
import mujoco_warp as mjw
import torch
import warp as wp
from mjlab.managers import EventTermCfg, SceneEntityCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import sample_uniform

from ics_rl_lab.utils.terrain_height import TerrainHeight, box_bottom_points


def reset_joints_by_scale(
    env,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset joint state by scaling default joint positions and velocities."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_pos *= sample_uniform(*position_range, joint_pos.shape, env.device)
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )


class place_on_terrain(ManagerTermBase):
    """Shift reset robots vertically so their lowest foot point is ``clearance`` above
    the terrain directly below it.

    The reset terms place the root at the default (flat-ground) height above the env
    origin, but mjlab's heightfield tiles span ``[0, noise_max - noise_min]`` while the
    origin sits at the middle of ``noise_range`` -- on the gravel terrain the feet
    started 40-90 mm inside the ground on nearly every reset. Must run after every
    term that writes the root/joint state (and after ``randomize_terrain``).

    ``asset_cfg.geom_names`` are the robot's lowest collision geoms: boxes (bottom face
    sampled every ``sample_spacing`` metres), capsules (axis sampled, lowest surface point
    = axis point minus the radius; exact on flat ground, a close bound on the gravel's
    0.1 m grid) or spheres. Runs FK for all
    worlds (cheap, ~1 ms) -- the env's own ``sim.forward()`` follows the reset terms.
    """

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        mj_model = env.sim.mj_model
        local_ids = asset_cfg.geom_ids
        if isinstance(local_ids, slice):
            raise ValueError("place_on_terrain needs explicit asset_cfg.geom_names.")
        self._geom_ids = asset.indexing.geom_ids[torch.as_tensor(local_ids)].long().to(env.device)
        spacing = cfg.params.get("sample_spacing", 0.005)
        points, radii = [], []
        for g in self._geom_ids.tolist():
            gtype, size = mj_model.geom_type[g], mj_model.geom_size[g]
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                points.append(box_bottom_points(size, spacing, env.device))
                radii.append(0.0)
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                n_axis = int(np.ceil(2 * size[1] / spacing)) + 1
                axis = torch.linspace(-size[1], size[1], n_axis, device=env.device)
                points.append(torch.stack([torch.zeros_like(axis), torch.zeros_like(axis), axis], -1))
                radii.append(float(size[0]))
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                points.append(torch.zeros(1, 3, device=env.device))
                radii.append(float(size[0]))
            else:
                raise ValueError(f"place_on_terrain: geom {mj_model.geom(g).name} is not a box, capsule or sphere.")
        self._radii = torch.tensor(radii, device=env.device)  # (G,)
        # Pad to a common count by repeating the first point (a duplicate never changes the min).
        n = max(len(p) for p in points)
        self._points = torch.stack([torch.cat([p, p[:1].expand(n - len(p), 3)]) for p in points])  # (G, P, 3)
        self._height = TerrainHeight(mj_model, env.device)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        clearance: float = 0.003,
        sample_spacing: float = 0.005,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = env_ids.long()
        sim = env.sim
        with wp.ScopedDevice(sim.wp_device):
            mjw.kinematics(sim.wp_model, sim.wp_data)

        pos = wp.to_torch(sim.wp_data.geom_xpos)[env_ids][:, self._geom_ids]  # (N, G, 3)
        rot = wp.to_torch(sim.wp_data.geom_xmat)[env_ids][:, self._geom_ids].reshape(len(env_ids), -1, 3, 3)
        points = pos[:, :, None] + torch.einsum("ngij,gpj->ngpi", rot, self._points)  # (N, G, P, 3)
        bottom = points[..., 2] - self._radii[None, :, None]
        gap = (bottom - self._height(points[..., :2])).flatten(1).amin(-1)

        asset = env.scene[asset_cfg.name]
        root_pose = asset.data.root_link_pose_w[env_ids].clone()
        root_pose[:, 2] += clearance - gap
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
