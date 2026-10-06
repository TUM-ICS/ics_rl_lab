"""Helpers shared by the S2 and Ultra entity configs.

PM01 keeps its own, older copies of these in `assets/pm01/` (pose helpers in `robot.py`,
delay wiring in `pm01_cfg.py`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.delayed_actuator import DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg

#: Maximum position-target delay, as a time (0-20 ms).
MAX_DELAY_S = 0.020


def standing_root_height(xml: Path, joint_pos: dict[str, float], foot_box_geoms: Sequence[str]) -> float:
  """Root height that puts the lowest corner of `foot_box_geoms` on z=0 at `joint_pos`."""
  model = mujoco.MjModel.from_xml_path(str(xml))
  data = mujoco.MjData(model)
  for name, q in joint_pos.items():
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = q
  mujoco.mj_forward(model, data)
  zs = []
  for name in foot_box_geoms:
    g = model.geom(name).id
    half = model.geom_size[g]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    zs.append((data.geom_xpos[g] + corners @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
  return float(data.qpos[2] - min(zs))


def make_articulation(
  actuators: Sequence[BuiltinPositionActuatorCfg], max_delay_steps: int = 0
) -> EntityArticulationInfoCfg:
  """Articulation with every actuator behind a 0..`max_delay_steps` position-target delay.

  `delay_hold_prob=1` keeps mjlab from resampling the lag every physics step; the lag is
  set once per episode by `actuator_delay_event`.
  With `max_delay_steps=0` the actuators are used directly.
  """
  if max_delay_steps > 0:
    actuators = tuple(
      DelayedActuatorCfg(base_cfg=a, delay_min_lag=0, delay_max_lag=max_delay_steps, delay_hold_prob=1.0)
      for a in actuators
    )
  return EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9)


def apply_actuator_delay(cfg, actuators: Sequence[BuiltinPositionActuatorCfg], max_delay_s: float = MAX_DELAY_S) -> None:
  """Put the task's robot (`cfg.scene.entities["robot"]`) behind a 0..`max_delay_s` delay,
  converted with the task's physics step, plus the reset event that samples it."""
  n = round(max_delay_s / cfg.sim.mujoco.timestep)
  cfg.scene.entities["robot"].articulation = make_articulation(actuators, n)
  cfg.events["actuator_delay"] = EventTermCfg(
    func=dr.sync_actuator_delays, mode="reset", params={"lag_range": (0, n)}
  )


def effort_limits(actuators: Sequence[BuiltinPositionActuatorCfg]) -> dict[str, float]:
  """{joint regex: effort limit}, the format `applied_torque_limits_by_ratio` takes."""
  out: dict[str, float] = {}
  for a in actuators:
    assert a.effort_limit is not None
    for expr in a.target_names_expr:
      out[expr] = a.effort_limit
  return out
