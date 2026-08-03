"""ISMPC hybrid task: RL learns CoM-height sine parameters for ISMPC.

SKELETON STAGE: observations, reward, and termination below are placeholders
(FOO), copied from patterns proven elsewhere in this repo (zero_residual's
observation set, residual_balance's termination conditions) just to make
this a well-formed, trainable env. None of them are ISMPC-specific yet --
that's the next design pass, once this skeleton is confirmed to construct
and step (and ideally train for a handful of iterations without crashing).

Rates: FOO placeholder. ISMPC's MPC period is m_delta=0.05s (20Hz); the
action should update once per MPC period, i.e. frameskip should make
`frameskip * sim.mujoco.timestep == 0.05`. Below assumes timestep=0.001,
frameskip=50; revisit if the real controller's mc_rtc.yaml uses a
different delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.robots.robots_registry import get_main_robot_spec, prepare_cfg_for_mc_rtc
from mc_mjlab.tasks.ismpc_hybrid.ismpc_sine_action import IsmpcSineActionCfg
from mc_mjlab.tasks.ismpc_hybrid import mdp as ismpc_mdp  # FOO module, see below

NUM_ENVS = 1  # FOO: matches residual_balance's scale; untuned for this task.
PLAY_NUM_ENVS = 1

# FOO: matches residual_balance's episode length; untuned for this task --
# revisit once ISMPC-specific reward/termination exist and you know how long
# a meaningful episode actually is for this controller/behavior.
EPISODE_LENGTH_S = 16.0

# FOO placeholder: m_delta=0.05s / timestep=0.001s. Confirm against your
# actual mc_rtc.yaml's `ismpc.delta` before trusting this.
FRAMESKIP = 2


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

  # FOO: base state matches zero_residual/residual_balance's set verbatim.
  # last_sine_params is new (see mdp.py docstring): the physical, mapped
  # params from the previous period, given as an observation specifically
  # so the policy can reason about phase continuity across periods. No
  # ISMPC-internal signal (tracking error, footstep/QP state) yet -- next
  # design pass, once this skeleton runs.
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "last_sine_params": ObservationTermCfg(
      func=ismpc_mdp.last_sine_params, params={"action_name": "ismpc_sine"}
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(terms=dict(actor_terms), concatenate_terms=True),
    "critic": ObservationGroupCfg(terms=dict(actor_terms), concatenate_terms=True),
  }

  # FOO: weights are placeholders, entirely untuned. `alive`/`upright` give
  # the base survival/stability pressure (Hypothesis A from the design
  # discussion: outcome-based, no explicit height/wrench/stability-margin
  # term, so any height-modulation strategy has to be *discovered* rather
  # than hinted at). `sine_continuity` and `joint_torque` are regularizers,
  # not task-defining rewards -- see mdp.py for why each is shaped the way
  # it is (continuity: value/slope match at the splice instant, not raw
  # parameter distance; torque: ordinary control-cost shaping, decoupled
  # from the disputed "humans do this to save energy" framing).
  rewards = {
    "alive": RewardTermCfg(func=ismpc_mdp.is_alive, weight=1.0),
    "upright": RewardTermCfg(func=ismpc_mdp.upright_penalty, weight=-1.0),
    "sine_continuity": RewardTermCfg(
      func=ismpc_mdp.sine_continuity_penalty,
      weight=-1.0,
      params={"action_name": "ismpc_sine"},
    ),
    "joint_torque": RewardTermCfg(func=ismpc_mdp.joint_torque_penalty, weight=-1.0e-4),
  }

  # FOO: placeholder termination, copied from residual_balance's pattern
  # (fell over / collapsed / controller failed). Thresholds untuned for
  # this specific task/controller.
  terminations = {
    "fell_over": TerminationTermCfg(func=ismpc_mdp.fell_over),
    "collapsed": TerminationTermCfg(func=ismpc_mdp.collapsed),
    "controller_failed": TerminationTermCfg(
      func=ismpc_mdp.controller_failed, params={"action_name": "ismpc_sine"}
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
    ),
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminations=terminations,
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
  return cfg


def ismpc_hybrid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(console_output="all" if play else "none")
  if play:
    _apply_play_overrides(cfg)
  return cfg


def ismpc_hybrid_ppo_cfg(max_iterations: int = 500) -> RslRlOnPolicyRunnerCfg:
  # FOO: copied from residual_balance's PPO hyperparameters verbatim;
  # untuned for this task's very different action space (4-dim, non-joint).
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.2,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu"),
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
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    logger="wandb",
  )