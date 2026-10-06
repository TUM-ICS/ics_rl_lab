"""Robot-agnostic whole-body motion-tracking task skeleton.

The reference motion comes from one ``MotionCommand`` ("motion") on top of
``ics_rl_lab.motion_lib`` -- see ``mdp/commands.py``.

Observation layout: the actor gets the
current reference + proprioception plus an 11-frame (T, D) history of the same terms
(compressed by a Conv1d encoder in the agent cfg); the critic gets privileged link
state plus a 20-frame future-reference lookahead (also Conv1d-encoded). Both groups
are unconcatenated so the encoders can pick their components by name.

Robot-specific names are placeholders, filled in by ``config/<robot>/env_cfgs.py``.
"""

from __future__ import annotations

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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from ics_rl_lab.motion_lib import ResidentMotionLibraryCfg
from ics_rl_lab.tasks.tracking import mdp

FEET_CONTACT = "feet_contact"
UNDESIRED_CONTACT = "undesired_contact"
HISTORY_LENGTH = 11

PUSH_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def _policy_terms() -> dict[str, ObservationTermCfg]:
  """Current-frame actor terms (source PolicyObsCfg, first block)."""
  return {
    "joint_pos_ref": ObservationTermCfg(func=mdp.ref_joint_pos),
    "joint_vel_ref": ObservationTermCfg(func=mdp.ref_joint_vel),
    "position_ref": ObservationTermCfg(func=mdp.ref_base_pos_b, noise=Unoise(n_min=-0.25, n_max=0.25)),
    "rotation_ref": ObservationTermCfg(func=mdp.ref_base_rot_tannorm_b, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)),
    # biased=True + the encoder_bias event == engineai's randomize_default_joint_pos
    # (action offset and joint_pos_rel reference shifted by the same calibration error).
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"biased": True}, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Build the tracking task. Robot-specific names/motions filled in by the caller."""

  ##
  # Observations
  ##

  current = _policy_terms()
  history = {
    f"{name}_hist": ObservationTermCfg(
      func=term.func,
      params=dict(term.params),
      noise=term.noise,
      history_length=HISTORY_LENGTH,
      flatten_history_dim=False,
    )
    for name, term in current.items()
  }
  policy_terms = {**current, **history}

  critic_terms = {
    "joint_pos_ref": ObservationTermCfg(func=mdp.ref_joint_pos),
    "joint_vel_ref": ObservationTermCfg(func=mdp.ref_joint_vel),
    "position_ref": ObservationTermCfg(func=mdp.ref_base_pos_b),
    "rotation_ref": ObservationTermCfg(func=mdp.ref_base_rot_tannorm_b),
    "link_pos": ObservationTermCfg(
      func=mdp.link_pos_b,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True)},  # Set per-robot.
    ),
    "link_rot": ObservationTermCfg(
      func=mdp.link_tannorm_b,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True)},  # Set per-robot.
    ),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True}),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "last_action": ObservationTermCfg(func=mdp.last_action),
    # Privileged 20-frame lookahead (0.1 s apart -> 2 s horizon), (T, D) for the critic encoder.
    "future_motion": ObservationTermCfg(func=mdp.future_motion_sequence),
  }

  observations = {
    "policy": ObservationGroupCfg(terms=policy_terms, concatenate_terms=False, enable_corruption=True),
    "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=False, enable_corruption=False),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=1.0, use_default_offset=True
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": mdp.MotionCommandCfg(
      motion_lib=ResidentMotionLibraryCfg(path=""),  # Set per-robot.
      body_names=(),  # Set per-robot (link_of_interests).
      root_body_name="",  # Set per-robot.
      # Source reset_robot_state_by_reference params.
      position_offset=(0.0, 0.0, 0.0),
      joint_vel_ratio=1.0,
      base_lin_vel_ratio=1.0,
      base_ang_vel_ratio=1.0,
      pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
      velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2)},
      joint_position_range=(-0.1, 0.1),
      debug_vis=True,
    )
  }

  ##
  # Events
  ##

  events = {
    "physics_material": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.6),  # source static_friction_range
      },
    ),
    "add_joint_default_pos": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": PUSH_VELOCITY_RANGE},
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "tracking_base_position": RewardTermCfg(func=mdp.tracking_base_position, weight=0.5, params={"std": 0.3}),
    "tracking_base_rot": RewardTermCfg(func=mdp.tracking_base_rot, weight=0.5, params={"std": 0.4}),
    "tracking_link_pos": RewardTermCfg(
      func=mdp.tracking_link_pos,
      weight=1.0,
      params={"combine_method": "mean_prod", "in_relative_world_frame": True, "std": 0.3},
    ),
    "tracking_link_rot": RewardTermCfg(
      func=mdp.tracking_link_rot,
      weight=1.0,
      params={"combine_method": "mean_prod", "in_relative_world_frame": True, "std": 0.4},
    ),
    "tracking_link_lin_vel": RewardTermCfg(
      func=mdp.tracking_link_lin_vel, weight=1.0, params={"combine_method": "mean_prod", "std": 1.0}
    ),
    "tracking_link_ang_vel": RewardTermCfg(
      func=mdp.tracking_link_ang_vel, weight=1.0, params={"combine_method": "mean_prod", "std": 3.14}
    ),
    "tracking_joint_pos": RewardTermCfg(func=mdp.tracking_joint_pos, weight=2.0),
    "tracking_joint_vel": RewardTermCfg(func=mdp.tracking_joint_vel, weight=0.2),
    "tracking_base_lin_vel": RewardTermCfg(func=mdp.tracking_base_velocity, weight=1.0),
    "tracking_base_ang_vel": RewardTermCfg(func=mdp.tracking_base_ang_vel, weight=1.0),
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "dof_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4),
    "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-5e-8),
    "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.01),
    "joint_limit": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "dof_torque_limits": RewardTermCfg(
      func=mdp.applied_torque_limits_by_ratio,
      weight=-1.0,
      params={"effort_limits": {}, "max_reward": 0.5},  # Set per-robot.
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.contact_slide,
      weight=-0.2,
      params={"sensor_name": FEET_CONTACT, "asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "feet_stumble": RewardTermCfg(func=mdp.feet_stumble, weight=-2.0, params={"sensor_name": FEET_CONTACT}),
    "feet_contact_forces": RewardTermCfg(
      func=mdp.contact_force_threshold, weight=-1.0, params={"sensor_name": FEET_CONTACT}
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts, weight=-1.0, params={"sensor_name": UNDESIRED_CONTACT, "threshold": 1.0}
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "base_pos_too_far": TerminationTermCfg(
      func=mdp.pos_far_from_ref, params={"distance_threshold": 0.25, "height_only": True}
    ),
    "base_pg_too_far": TerminationTermCfg(
      func=mdp.projected_gravity_far_from_ref, params={"projected_gravity_threshold": 0.8}
    ),
    "link_pos_too_far": TerminationTermCfg(
      func=mdp.link_pos_far_from_ref,
      params={"body_names": (), "distance_threshold": 0.25, "height_only": True},  # Set per-robot.
    ),
    "dataset_exhausted": TerminationTermCfg(func=mdp.dataset_exhausted, time_out=True),
  }

  ##
  # Assemble
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1, env_spacing=4.0),
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
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=60,
      njmax=1500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=10.0,
  )


__all__ = ["make_tracking_env_cfg", "FEET_CONTACT", "UNDESIRED_CONTACT", "HISTORY_LENGTH"]
