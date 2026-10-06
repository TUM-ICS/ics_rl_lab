"""Robot-agnostic hoverboard-driving task skeleton.

A humanoid stands on a self-balancing hoverboard and steers it by tilting the plates
with its ankles. The board's own firmware balancer runs as a zero-dim action term
(`HoverboardActionTermCfg`, lives with the asset in `assets/hoverboard/`); the policy
only commands the robot's joints.

Standard terms come from mjlab's stock `mdp`; the custom ones live in `mdp/`. Robot
body/geom names are placeholders (`()`), filled in by `config/<robot>/env_cfgs.py` -- same
layout as the locomotion task. The joint-name regexes in the posture terms use pm01 naming.
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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from ics_rl_lab.assets.hoverboard.hoverboard_cfg import PLATE_BODIES, PLATE_JOINTS, get_hoverboard_cfg
from ics_rl_lab.assets.hoverboard.hoverboard_driver import HoverboardActionTermCfg
from ics_rl_lab.tasks.hoverboard import mdp

# Contact sensors, created per robot in config/<robot>/env_cfgs.py.
FEET_CONTACT = "feet_contact"  # per-foot net force against anything (plates), 3-substep history
UNDESIRED_CONTACT = "undesired_contact"  # one slot per non-ankle body, against anything

HOVERBOARD_ACTION = "hoverboard_lqr"

OBS_CLIP = (-100.0, 100.0)
OBS_HISTORY = 10


def make_hoverboard_env_cfg() -> ManagerBasedRlEnvCfg:
  """Build the hoverboard-driving task. Robot-specific names filled in by the caller."""

  ##
  # Observations (source: every term clipped to +-100, history 10; critic noise-free)
  ##

  policy_terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), scale=0.2),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "commands": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
    "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel_rel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.2),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }

  hb = SceneEntityCfg("hoverboard")
  critic_terms = {
    **{k: ObservationTermCfg(func=v.func, params=v.params, scale=v.scale) for k, v in policy_terms.items()},
    # robot
    "base_height": ObservationTermCfg(func=mdp.base_height),
    "feet_contact": ObservationTermCfg(
      func=mdp.contact_force, params={"sensor_name": FEET_CONTACT, "flatten": True}, scale=0.01
    ),
    # robot <-> hoverboard
    "hb_robot_rel_pos": ObservationTermCfg(
      func=mdp.link_relative_pos,
      params={
        "parent_asset_cfg": hb,
        "child_asset_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),  # Set per-robot.
        "flatten": True,
      },
    ),
    "hb_robot_rel_quat": ObservationTermCfg(
      func=mdp.link_relative_quat,
      params={
        "parent_asset_cfg": hb,
        "child_asset_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),  # Set per-robot.
        "flatten": True,
      },
    ),
    # hoverboard. Board defaults are all zero, so *_rel == absolute (source: joint_pos/joint_vel).
    "hb_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, params={"asset_cfg": hb}, scale=0.2),
    "hb_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, params={"asset_cfg": hb}),
    "hb_gravity": ObservationTermCfg(func=mdp.projected_gravity, params={"asset_cfg": hb}),
    "hb_joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("hoverboard", joint_names=PLATE_JOINTS)}
    ),
    "hb_joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": hb}, scale=0.2),
    "hb_state": ObservationTermCfg(func=mdp.hoverboard_state, params={"action_name": HOVERBOARD_ACTION}, scale=0.2),
    "hb_torque": ObservationTermCfg(func=mdp.hoverboard_torque, params={"action_name": HOVERBOARD_ACTION}, scale=0.1),
  }
  for term in (*policy_terms.values(), *critic_terms.values()):
    term.clip = OBS_CLIP

  # nan_policy="sanitize": see the `nan_detection` termination below. The frame right after
  # such a reset still carries the dead step's (NaN) contact-sensor readings.
  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=OBS_HISTORY,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=OBS_HISTORY,
      nan_policy="sanitize",
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Source value; kept for pm01 (see config/pm01/env_cfgs.py).
      use_default_offset=True,
    ),
    HOVERBOARD_ACTION: HoverboardActionTermCfg(
      entity_name="hoverboard",
      lqr_wn_range=[2.0, 8.0],
      lqr_zeta_range=[0.05, 0.5],
      lqr_third_pole_ratio=1.5,
      lqr_third_pole_floor=1.0,
      lqr_input_delay_range=[0, 0],
      nominal_integral_gain=0.0,
      gain_noise_range_integral=[0.0, 0.0],
      integral_clamp=0.0,
      yaw_bias_range=[0.3, 5.0],
    ),
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "base_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 8.0),
      rel_standing_envs=0.2,
      rel_heading_envs=1.0,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=False,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 2.5),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.5, 1.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Rewards
  ##

  rewards = {
    "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # tracking (of the *hoverboard's* velocity)
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=mdp.track_lin_vel_xy_yaw_frame_exp,
      weight=5.0,
      params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": hb},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.track_ang_vel_z_world_exp,
      weight=5.0,
      params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": hb},
    ),
    "dont_drift": RewardTermCfg(
      func=mdp.dont_drift,
      weight=-0.1,
      params={"command_name": "base_velocity", "cmd_threshold": 0.1, "asset_cfg": hb},
    ),
    "feet_contact_loss": RewardTermCfg(
      func=mdp.feet_contact_loss, weight=-1.0, params={"sensor_name": FEET_CONTACT, "threshold": 5.0}
    ),
    # regularization
    "hb_ang_vel_l2": RewardTermCfg(
      func=mdp.hoverboard_ang_vel_l2, weight=-0.1, params={"asset_cfg": hb, "scale": (1.0, 1.0)}
    ),
    "joint_deviation_upper": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.5,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=("knee_pitch_.*", "shoulder_roll_.*", "shoulder_yaw_.*", "shoulder_pitch_.*", "elbow_.*", "head_.*"),
        )
      },
    ),
    "joint_deviation_hip": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-2.0,
      # hip_pitch: punish leaning back and forth too much.
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("hip_pitch_.*", "hip_yaw_.*"))},
    ),
    "joint_deviation_hip_roll": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.5,
      # Allowed to lean into curves.
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("hip_roll_.*",))},
    ),
    "joint_deviation_roll": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-5.0,
      # ankle_roll: punish CoM left/right motion; waist: prevent waist rotations.
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("ankle_roll_.*", "waist_yaw_joint"))},
    ),
    "penalty_joint_deviation_ankle": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-1e-3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("ankle_pitch_.*",))},
    ),
    "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_scaled_l2, weight=-0.5, params={"scale": (0.5, 0.5)}),
    "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "dof_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-4.0e-6),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_scaled_l2, weight=-2.0, params={"scale": (0.5, 0.5)}
    ),
    "energy": RewardTermCfg(
      func=mdp.motors_power_square,
      weight=-5e-5,
      params={"asset_cfg": SceneEntityCfg("robot"), "normalize_by_stiffness": True},
    ),
    # safety
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts, weight=-5.0, params={"sensor_name": UNDESIRED_CONTACT, "threshold": 1.0}
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Under random/early-training actions, mujoco-warp's float32 solver occasionally NaNs one env (~1-3 per 1.6M env-steps).
    # NaN states make every other termination compare False, so the env would stay dead
    # until time-out. Reset it at once; counted as a truncation, not a failure, so the
    # policy isn't charged the termination penalty for a simulator fault.
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection, time_out=True),
    "bad_orientation": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 0.8}),
    "feet_outside_plate": TerminationTermCfg(
      func=mdp.feet_outside_plate,
      params={
        "nominal_offsets": [
          [0.0, 0.03],  # left foot relative to left_plate_link
          [0.0, -0.03],  # right foot relative to right_plate_link
        ],
        "max_offset_x": 0.08,
        "max_offset_y": 0.08,
        "plate_asset_cfg": SceneEntityCfg("hoverboard", body_names=PLATE_BODIES, preserve_order=True),
        "feet_asset_cfg": SceneEntityCfg("robot", body_names=(), preserve_order=True),  # Set per-robot (left, right).
      },
    ),
  }

  ##
  # Events. MuJoCo has one friction coefficient per geom, so only the static friction is
  # randomized (dynamic friction / restitution have no MuJoCo counterpart).
  ##

  events = {
    "physics_material": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.8, 1.2),
      },
    ),
    "hoverboard_plate_material": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("hoverboard", geom_names=(r".*_plate_col",)),
        "operation": "abs",
        "ranges": (1.2, 1.8),
      },
    ),
    "hoverboard_wheel_material": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("hoverboard", geom_names=(r".*_wheel_col",)),
        "operation": "abs",
        "ranges": (0.6, 1.2),
      },
    ),
    "add_base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": (-1.0, 3.0),
      },
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
    "randomize_actuator_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.8, 1.2),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "reset_base": EventTermCfg(
      func=mdp.reset_hoverboard_root_state,  # Spawn hoverboard and robot together.
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "robot_offset_pos": [0.0, 0.0, 0.1],
        "robot_offset_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
        "robot_asset_cfg": SceneEntityCfg("robot"),
        "hoverboard_asset_cfg": hb,
      },
    ),
    "reset_lower_body": EventTermCfg(
      func=mdp.reset_joints_by_scale,
      mode="reset",
      params={
        "position_range": (1.0, 1.0),  # Keep the legs the same.
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("hip_.*", "knee_.*", "ankle_.*")),
      },
    ),
    "reset_upper_body": EventTermCfg(
      func=mdp.reset_joints_by_scale,  # Randomize the upper body.
      mode="reset",
      params={
        "position_range": (0.8, 1.2),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("waist_.*", "shoulder_.*", "elbow_.*")),
      },
    ),
    "reset_hoverboard_joints": EventTermCfg(
      func=mdp.reset_joints_by_scale,
      mode="reset",
      params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": hb},
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(10.0, 15.0),
      params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.5, 0.5)}},
    ),
  }

  ##
  # Assemble
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"hoverboard": get_hoverboard_cfg()},  # Robot added per-robot.
      num_envs=1,
      env_spacing=2.5,
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
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    # 2 ms physics (500 Hz balancer), decimation 10 -> policy stays at 50 Hz. At 5 ms the board's LQR rate feedback drives the (necessarily soft,
    # timeconst >= 2*dt) MuJoCo foot-plate contact into a bounce and the board can't hold a
    # rider it holds at 2 ms; no contact/solver setting fixes it at 5 ms.
    #
    # Elliptic friction cone, impratio 5: with the default pyramidal cone the feet creep
    # backwards on the plates under the sustained tangential load of riding (~15 mm per
    # 20 s), which ends most episodes via feet_outside_plate. More solver iterations don't
    # help; stiffer solref/solimp make it worse. impratio >= 10 NaNs mujoco-warp's float32
    # solver (0.3% of resets at 10, ~all at 20). 5/10 iterations recover most of the cost
    # of the elliptic cone (-24% vs. pyramidal at 4096 envs instead of -40%).
    sim=SimulationCfg(
      nconmax=64,
      njmax=2000,
      mujoco=MujocoCfg(timestep=0.002, cone="elliptic", impratio=5.0, iterations=5, ls_iterations=10),
    ),
    decimation=10,
    episode_length_s=20.0,
  )


__all__ = ["make_hoverboard_env_cfg", "FEET_CONTACT", "UNDESIRED_CONTACT", "HOVERBOARD_ACTION"]
