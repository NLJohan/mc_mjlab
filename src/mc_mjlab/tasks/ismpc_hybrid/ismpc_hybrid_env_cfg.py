"""ISMPC hybrid task: RL learns CoM-height sine parameters for ISMPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.dr import body as dr_body
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains.heightfield_terrains import HfRandomUniformTerrainCfg

from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.robots.robots_registry import get_main_robot_spec, prepare_cfg_for_mc_rtc
from mc_mjlab.tasks.ismpc_hybrid.ismpc_sine_action import IsmpcSineActionCfg
from mc_mjlab.tasks.ismpc_hybrid import mdp as ismpc_mdp

NUM_ENVS = 500
PLAY_NUM_ENVS = 1

EPISODE_LENGTH_S = 6.0

FRAMESKIP = 2

# --- Push disturbances (mjlab.envs.mdp.events.apply_body_impulse). ---
PUSH_SETTLE_TICKS = 20
PUSH_FORCE_TORSO_N = (-50.0, 50.0)
PUSH_FORCE_HAND_N = (-100.0, 100.0)
PUSH_DURATION_S = (0.1, 0.6)
PUSH_COOLDOWN_TORSO_S = (3.0, 15.0)
PUSH_COOLDOWN_HAND_S = (3.0, 15.0)

# --- Mass/inertia domain randomization. ---
BODY_MASS_ALPHA_RANGE = (-0.05, 0.05)
HAND_PAYLOAD_MASS_RANGE_KG = (0.0, 3.0)

# --- Uneven terrain (mjlab.terrains, HfRandomUniformTerrainCfg). ---
ENABLE_UNEVEN_TERRAIN = False
TERRAIN_NUM_ROWS = 1
TERRAIN_NUM_COLS = 1
TERRAIN_NOISE_RANGE_M = (-0.005, 0.005)
TERRAIN_PATCH_SIZE_M = (8.0, 8.0)


def _make_terrain_cfg() -> TerrainEntityCfg:
  """Build the scene's TerrainEntityCfg, gated on ENABLE_UNEVEN_TERRAIN.

  True -> the small-amplitude generator/heightfield terrain (see the block
  above for the full design rationale). False -> the original flat plane,
  byte-for-byte the same as before uneven terrain was introduced.
  """
  if not ENABLE_UNEVEN_TERRAIN:
    return TerrainEntityCfg(terrain_type="plane")
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      curriculum=False,
      size=TERRAIN_PATCH_SIZE_M,
      num_rows=TERRAIN_NUM_ROWS,
      num_cols=TERRAIN_NUM_COLS,
      border_width=1.0,
      sub_terrains={
        "rough": HfRandomUniformTerrainCfg(
          noise_range=TERRAIN_NOISE_RANGE_M,
          noise_step=0.005,
          downsampled_scale=0.4,
          horizontal_scale=0.1,
          vertical_scale=0.005,
          scale_with_difficulty=False,
        ),
      },
    ),
  )

def _make_env_cfg(
  num_envs: int = NUM_ENVS,
  num_workers: int | None = None,
  console_output: str = "none",
  mc_rtc_yaml: Path = MC_RTC_YAML_PATH,
) -> ManagerBasedRlEnvCfg:
  robot_name, robot = get_main_robot_spec(mc_rtc_yaml)
  robot_cfg = prepare_cfg_for_mc_rtc(
    robot.cfg_fn(), names_collision_geoms=robot.names_collision_geoms
  )

  actions: dict[str, ActionTermCfg] = {
    "ismpc_sine": IsmpcSineActionCfg(
      entity_name="robot",
      target_actuator_names=(".*",),
      mc_rtc_config_path=str(mc_rtc_yaml),
      mc_rtc_robot_name=robot_name,
      frameskip=FRAMESKIP,
      num_workers=num_workers,
      pd_gains_path=str(robot.pd_gains_path),
      console_output=console_output,
    )
  }

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "last_sine_params": ObservationTermCfg(
      func=ismpc_mdp.last_sine_params, params={"action_name": "ismpc_sine"}
    ),
    "last_walk_action": ObservationTermCfg(
      func=ismpc_mdp.last_walk_action, params={"action_name": "ismpc_sine"}
    ),
    "last_step_timing_action": ObservationTermCfg(
      func=ismpc_mdp.last_step_timing_action, params={"action_name": "ismpc_sine"}
    ),
    "ismpc_wants_stop": ObservationTermCfg(
      func=ismpc_mdp.ismpc_wants_stop, params={"action_name": "ismpc_sine"}
    ),
    "target_twist": ObservationTermCfg(
      func=ismpc_mdp.target_twist, params={"command_name": "twist"}
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(terms=dict(actor_terms), concatenate_terms=True),
    "critic": ObservationGroupCfg(terms=dict(actor_terms), concatenate_terms=True),
  }

  rewards = {
    "is_alive": RewardTermCfg(
      func=ismpc_mdp.is_alive, 
      weight=6.0),
    "not_walking_penalty": RewardTermCfg(
      func=ismpc_mdp.not_walking_penalty, 
      weight=-2.0, 
      params={"action_name": "ismpc_sine"}
    ),
    # "upright": RewardTermCfg(
    #   func=ismpc_mdp.upright_reward, 
    #   weight=1.0),
    "sine_position_continuity": RewardTermCfg(
      func=ismpc_mdp.sine_position_continuity,
      weight=1.0,
      params={"action_name": "ismpc_sine", "sigma" : 0.003},
    ),
    "sine_velocity_continuity": RewardTermCfg(
      func=ismpc_mdp.sine_velocity_continuity,
      weight=0.1,
      params={"action_name": "ismpc_sine"},
    ),
    "joint_torque": RewardTermCfg(
      func=ismpc_mdp.joint_torque_reward, 
      weight=1.0
    ),
    "target_linear_vel": RewardTermCfg(
        func=ismpc_mdp.target_linear_vel,
        weight=2.0,
        params={"command_name": "twist", "std": 0.1},
    ),
    "target_angular_vel": RewardTermCfg(
        func=ismpc_mdp.target_angular_vel,
        weight=0.5,
        params={"command_name": "twist", "std": 0.05},
    ),
    # --- Plotting-only, weight=0.0
    "debug_target_height": RewardTermCfg(
      func=ismpc_mdp.debug_target_height,
      weight=0.0,
      params={"action_name": "ismpc_sine"},
    ),
    "debug_step_timing": RewardTermCfg(
      func=ismpc_mdp.debug_step_timing,
      weight=0.0,
      params={"action_name": "ismpc_sine"},
    ),
    "debug_is_walking": RewardTermCfg(
      func=ismpc_mdp.debug_is_walking,
      weight=0.0,
      params={"action_name": "ismpc_sine"},
    ),
  }

  terminations = {
    "fell_over": TerminationTermCfg(func=ismpc_mdp.fell_over),
    "collapsed": TerminationTermCfg(func=ismpc_mdp.collapsed),
    "controller_failed": TerminationTermCfg(
      func=ismpc_mdp.controller_failed, 
      params={"action_name": "ismpc_sine"}
    ),
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
  }

  # Domain randomization on reset
  events = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default, 
      mode="reset"
    ),
    "reset_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.02, 0.02),  # rad
        "velocity_range": (-0.01, 0.01),  # rad/s
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "push_torso": EventTermCfg(
      func=ismpc_mdp.settle_gated_apply_body_impulse,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
        "force_range": PUSH_FORCE_TORSO_N,
        "torque_range": (0.0, 0.0),
        "duration_s": PUSH_DURATION_S,
        "cooldown_s": PUSH_COOLDOWN_TORSO_S,
        "settle_ticks": PUSH_SETTLE_TICKS,
      },
    ),
    "push_right_hand": EventTermCfg(
      func=ismpc_mdp.settle_gated_apply_body_impulse,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["Rhand_Link0_Plan2"]),
        "force_range": PUSH_FORCE_HAND_N,
        "torque_range": (0.0, 0.0),
        "duration_s": PUSH_DURATION_S,
        "cooldown_s": PUSH_COOLDOWN_HAND_S,
        "settle_ticks": PUSH_SETTLE_TICKS,
      },
    ),
    "push_left_hand": EventTermCfg(
      func=ismpc_mdp.settle_gated_apply_body_impulse,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["Lhand_Link0_Plan2"]),
        "force_range": PUSH_FORCE_HAND_N,
        "torque_range": (0.0, 0.0),
        "duration_s": PUSH_DURATION_S,
        "cooldown_s": PUSH_COOLDOWN_HAND_S,
        "settle_ticks": PUSH_SETTLE_TICKS,
      },
    ),
    "randomize_body_density": EventTermCfg(
      func=dr_body.pseudo_inertia,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "alpha_range": BODY_MASS_ALPHA_RANGE,
        "distribution": "uniform",
      },
    ),
    "randomize_hand_payload": EventTermCfg(
      func=dr_body.body_mass,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=["Rhand_Link0_Plan2", "Lhand_Link0_Plan2"]
        ),
        "ranges": HAND_PAYLOAD_MASS_RANGE_KG,
        "operation": "add",
        "distribution": "uniform",
      },
    ),
  }

  # made consistent with each other.
  commands = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.4, 0.4),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_z=(-0.5, 0.5),
      ),
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      terrain=_make_terrain_cfg(),
      entities={"robot": robot_cfg},
    ),
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminations=terminations,
    events=events,
    commands=commands,
    decimation=FRAMESKIP,
    episode_length_s=EPISODE_LENGTH_S,
    sim=SimulationCfg(
      njmax=1500,
      nconmax=100,
      mujoco=MujocoCfg(
        timestep=0.001,
        integrator="euler",
        solver="newton",
        iterations=50,
        tolerance=1e-10,
        jacobian="dense",
      ),
    ),
  )


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.scene.num_envs = PLAY_NUM_ENVS
  for name in ("debug_target_height", "debug_step_timing", "debug_is_walking"):
    cfg.rewards[name].weight = 1.0
  return cfg


def ismpc_hybrid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(console_output="all" if play else "none")
  if play:
    _apply_play_overrides(cfg)
  return cfg


def ismpc_hybrid_ppo_cfg(max_iterations: int = 500) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.4,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="mc_rtc_ismpc_hybrid",
    save_interval=100,
    num_steps_per_env=512,
    max_iterations=max_iterations,
    logger="wandb",
  )