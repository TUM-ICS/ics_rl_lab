"""mjlab-native `Sensor` wrapper around `AMPMotionLoader`.

A thin, stateless sensor whose only job is making the loader addressable as
`env.scene["motionloader"].data`, which `ics_rl`'s `OnPolicyRunner` already knows how
to pick up for any algorithm declaring a `motion_loader` slot (see
`ics_rl/runners/on_policy_runner.py`). No `ics_rl` changes needed.

Joint/body selection for the *live* AMP observation terms is done via ordinary `SceneEntityCfg(..., preserve_order=True)` in
`tasks/amp/amp_env_cfg.py` rather than sensor-exposed `.joint_ids`/`.body_ids`
properties -- both approaches resolve to the same index order (confirmed: mjlab's
`find_joints`/`find_bodies` preserve the given name-list order when
`preserve_order=True`), so this sensor only needs to build the loader and remap its
reset-state joint order, not track ids used elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
from mjlab.entity import Entity
from mjlab.sensor import Sensor, SensorCfg

from ics_rl_lab.motion_lib.amp_motion_loader import AMPMotionLoader


class MotionLoaderSensor(Sensor[AMPMotionLoader]):
    """Stateless sensor: builds an `AMPMotionLoader` once, exposes it via `.data`."""

    def __init__(self, cfg: "MotionLoaderSensorCfg"):
        super().__init__()
        self.cfg = cfg
        self._robot_entity: Entity | None = None
        self._loader: AMPMotionLoader | None = None

    def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
        # No physics model changes needed -- capture the robot entity for use in
        # initialize(), which (unlike edit_spec) doesn't receive `entities`.
        self._robot_entity = entities[self.cfg.entity_name]

    def initialize(self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str) -> None:
        assert self._robot_entity is not None
        self._loader = AMPMotionLoader(
            device=device,
            motion_directory=self.cfg.motion_directory,
            motion_names=list(self.cfg.motion_names),
            motion_weights=list(self.cfg.motion_weights),
            root_body_name=self.cfg.root_body_name,
            selected_obs_names=list(self.cfg.selected_obs_names),
            selected_body_names=list(self.cfg.selected_body_names) if self.cfg.selected_body_names else None,
            selected_joint_names=list(self.cfg.selected_joint_names) if self.cfg.selected_joint_names else None,
        )
        # reset_states is written positionally into the sim, so it must be in the
        # robot's actual joint order, not the motion files' raw order.
        self._loader.remap_reset_joint_order(list(self._robot_entity.joint_names))

    def _compute_data(self) -> AMPMotionLoader:
        assert self._loader is not None
        return self._loader


@dataclass
class MotionLoaderSensorCfg(SensorCfg):
    motion_directory: str = ""
    motion_names: tuple[str, ...] = ()
    motion_weights: tuple[float, ...] = ()
    root_body_name: str = ""
    selected_obs_names: tuple[str, ...] = ()
    selected_body_names: tuple[str, ...] | None = None
    selected_joint_names: tuple[str, ...] | None = None
    entity_name: str = "robot"

    def build(self) -> MotionLoaderSensor:
        return MotionLoaderSensor(self)
