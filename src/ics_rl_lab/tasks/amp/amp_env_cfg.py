"""Robot-agnostic AMP task skeleton.

Reuses `locomotion.mdp` reward/observation functions directly rather than duplicating
them, and adds the `amp_policy` discriminator-input
observation group plus the `motionloader` sensor and `reset_expert_state` (RSI)
event that make it an AMP task rather than a plain locomotion one.

Body/joint/geom names are left as placeholders (`()`), filled in by
`config/pm01/env_cfgs.py`.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import HfRandomUniformTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from ics_rl_lab.sim import IcsMujocoCfg
from ics_rl_lab.tasks.amp import mdp

# Same rough terrain
GRAVEL_TERRAIN_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "random_rough": HfRandomUniformTerrainCfg(
      proportion=1.0, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
    ),
  },
)

MOTIONLOADER = "motionloader"


def make_amp_env_cfg() -> ManagerBasedRlEnvCfg:
  """Build the pm01 AMP task. Robot-specific names filled in by the caller."""

  ##
  # Observations
  ##

  policy_terms = {
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.2),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "commands": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
    "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel_rel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **{k: ObservationTermCfg(func=v.func, params=v.params, scale=v.scale) for k, v in policy_terms.items()},
    "base_height": ObservationTermCfg(func=mdp.base_height),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, scale=0.2),
    "feet_rel_pos": ObservationTermCfg(
      func=mdp.link_pos_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=()), "flatten": True}
    ),
    "feet_rel_ori": ObservationTermCfg(
      func=mdp.link_quat_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=()), "flatten": True}
    ),
    "feet_rel_lin_vel": ObservationTermCfg(
      func=mdp.link_lin_vel_b,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=()), "flatten": True},
      scale=0.2,
    ),
    "feet_rel_ang_vel": ObservationTermCfg(
      func=mdp.link_ang_vel_b,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=()), "flatten": True},
      scale=0.2,
    ),
    "feet_contact": ObservationTermCfg(
      func=mdp.contact_force_norm, params={"sensor_name": ""}, scale=0.01
    ),
  }

  amp_policy_terms = {
    "root_pos_z": ObservationTermCfg(func=mdp.base_height),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.amp_joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(), preserve_order=True)}),
    "joint_vel": ObservationTermCfg(func=mdp.amp_joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(), preserve_order=True)}),
    "root_lin_vel_b": ObservationTermCfg(func=mdp.base_lin_vel),
    "root_ang_vel_b": ObservationTermCfg(func=mdp.base_ang_vel),
    "body_pos_b": ObservationTermCfg(
      func=mdp.link_pos_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True), "flatten": True}
    ),
    "body_quat_b": ObservationTermCfg(
      func=mdp.link_quat_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True), "flatten": True}
    ),
    "body_lin_vel_b": ObservationTermCfg(
      func=mdp.link_lin_vel_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True), "flatten": True}
    ),
    "body_ang_vel_b": ObservationTermCfg(
      func=mdp.link_ang_vel_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True), "flatten": True}
    ),
  }

  # nan_policy="sanitize": backstop for a blown-up env (zeroes NaN/Inf and warns) so
  # it can't reach the networks or the AMP replay buffer/normalizer. The
  # `nan_detection` termination below is the primary guard.
  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms, concatenate_terms=True, enable_corruption=True, history_length=10,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False, history_length=10,
      nan_policy="sanitize",
    ),
    "amp_policy": ObservationGroupCfg(
      terms=amp_policy_terms, concatenate_terms=True, enable_corruption=False, nan_policy="sanitize"
    ),
  }

  ##
  # Actions / Commands (identical to locomotion)
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=0.25, use_default_offset=True
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "base_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.2,
      rel_heading_envs=1.0,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.2), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "physics_material": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=()), "operation": "abs", "ranges": (0.3, 1.6)},
    ),
    "add_base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=()), "operation": "add", "ranges": (-1.0, 3.0)},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
    "reset_expert_state": EventTermCfg(
      func=mdp.reset_expert_state,
      mode="reset",
      params={
        "expert_reset_prob": 0.8,
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
        "velocity_range": {
          "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
          "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
        },
        "dof_position_range": (0.5, 1.5),
        "dof_velocity_range": (0.0, 0.0),
      },
    ),
    # Must stay after every reset term that writes the robot state.
    "place_on_terrain": EventTermCfg(
      func=mdp.place_on_terrain,
      mode="reset",
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=()), "clearance": 0.003},
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 5.0),
      params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.0, params={"command_name": "base_velocity", "std": 0.25}
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.track_ang_vel_z_world_exp, weight=5.0, params={"command_name": "base_velocity", "std": 0.25}
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.5,
      params={"command_name": "base_velocity", "vel_threshold": 0.15, "threshold": 0.4, "sensor_name": ""},
    ),
    "dont_wait": RewardTermCfg(func=mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"}),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still, weight=-0.3, params={"command_name": "base_velocity", "offset": 4.0}
    ),
    # stand_still only sees joint positions, so with a zero command the policy could
    # creep/sway slowly for free.
    "stand_still_vel": RewardTermCfg(
      func=mdp.stand_still_vel, weight=-1.0, params={"command_name": "base_velocity"}
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.contact_slide,
      weight=-0.2,
      params={"sensor_name": "", "asset_cfg": SceneEntityCfg("robot", body_names=())},
    ),
    "feet_fly": RewardTermCfg(func=mdp.dont_fly, weight=-1.0, params={"sensor_name": "", "threshold": 1.0}),
    "feet_stumble": RewardTermCfg(func=mdp.feet_stumble, weight=-2.0, params={"sensor_name": ""}),
    "penalty_feet_too_near": RewardTermCfg(
      func=mdp.feet_too_near,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=()), "threshold": 0.2},
    ),
    "penalty_feet_force": RewardTermCfg(
      func=mdp.contact_force_threshold,
      weight=-1.0,
      params={"sensor_name": "", "threshold": 1.25, "max_reward": 1.0},
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts, weight=-5.0, params={"sensor_name": "", "threshold": 1.0}
    ),
    "penalty_flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
    "penalty_ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-1.0),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "dof_torque_limits": RewardTermCfg(
      func=mdp.applied_torque_limits_by_ratio,
      weight=-10.0,
      params={"effort_limits": {}, "limit_ratio": 0.8, "max_reward": 0.5},
    ),
    "penalty_energy": RewardTermCfg(func=mdp.energy, weight=-1e-3),
    "penalty_action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "penalty_dof_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001),
    "penalty_dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_deviation_hip": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.15,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_yaw.*", r".*hip_roll.*", r".*shoulder_pitch.*", r".*elbow.*"))},
    ),
    "joint_deviation_arms": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.2,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*waist.*", r".*shoulder_roll.*", r".*shoulder_yaw.*"))},
    ),
    "joint_deviation_legs": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-1e-3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_pitch.*", r".*knee_pitch.*", r".*ankle.*"))},
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "root_height_below_minimum": TerminationTermCfg(
      func=mdp.root_height_below_minimum, params={"minimum_height": 0.4}
    ),
    "bad_orientation": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 0.8}),
    # Resets an env whose physics state went NaN/Inf before its observations are computed.
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
  }

  ##
  # Assemble
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="generator", terrain_generator=GRAVEL_TERRAIN_CFG),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35, njmax=1500, mujoco=IcsMujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)
    ),
    decimation=4,
    episode_length_s=20.0,
  )


__all__ = ["make_amp_env_cfg", "MOTIONLOADER"]
