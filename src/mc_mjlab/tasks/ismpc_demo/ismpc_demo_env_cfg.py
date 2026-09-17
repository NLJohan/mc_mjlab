"""ISMPC CoM-height sine demo task.

Same spirit as zero_residual: no reward, no termination, nothing is being
learned. Unlike zero_residual, the action term is not actually a no-op --
ScriptedIsmpcSineDemoAction drives the ISMPC CoM-height reference through a
fixed constant -> sine -> constant schedule (see that file's docstring for
the exact timings), purely to make the earlier smoke-test-verified plumbing
(RL/scripted params -> ismpc_walking_python bridge -> ISMPC_Solver) visible
in a viewer session. The joint-position side of the action is ordinary
zero-residual behavior: mc_rtc's own q/alpha drives the joints throughout.

This is a *play*-only task, same restriction and same reasons as
zero_residual: no reward to optimise, and no termination to end an episode
a policy could otherwise wreck. `train` on it is not supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.robots.registry import get_main_robot_spec, prepare_cfg_for_mc_rtc
from mc_mjlab.tasks.ismpc_demo.scripted_ismpc_sine_action import (
  ScriptedIsmpcSineDemoActionCfg,
)

# One controller per env (~70 MB, ~570 ms to construct): a viewer wants one.
PLAY_NUM_ENVS = 1
NUM_ENVS = 1

# Covers the full demo schedule (warmup + sine + a settle-back tail) with
# margin; effectively unbounded for a viewer session, same reasoning as
# zero_residual's EPISODE_LENGTH_S (0 breaks rsl_rl's sampling, a huge
# sentinel overflows the episode-length buffer's dtype).
EPISODE_LENGTH_S = 1800.0


def _make_env_cfg(
  num_envs: int = NUM_ENVS,
  num_workers: int | None = None,
  console_output: Literal["none", "single", "all"] = "none",
  mc_rtc_yaml: Path = MC_RTC_YAML_PATH,
) -> ManagerBasedRlEnvCfg:
  """Build the ISMPC sine demo env cfg for the config's ``MainRobot``.

  ``etc/mc_rtc.yaml``'s ``Enabled`` must be the ISMPC walking controller
  (not Posture) for this task to do anything meaningful: the scripted action
  writes sine params that only ``Walking_controller``/``ISMPC_Solver``
  consume -- against Posture, the ismpc_walking_python bridge's dynamic_cast
  simply fails every step (harmlessly: it returns False, logged nowhere by
  default) and the demo just shows an ordinary standing Posture robot.
  """
  robot_name, robot = get_main_robot_spec(mc_rtc_yaml)
  robot_cfg = prepare_cfg_for_mc_rtc(
    robot.cfg_fn(), names_collision_geoms=robot.names_collision_geoms
  )

  actions: dict[str, ActionTermCfg] = {
    "robot_joints": ScriptedIsmpcSineDemoActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      residual_actuator_names=robot.get_residual_joints(),
      mc_rtc_config_path=str(mc_rtc_yaml),
      mc_rtc_robot_name=robot_name,
      frameskip=2,
      num_workers=num_workers,
      pd_gains_path=str(robot.pd_gains_path),
      console_output=console_output,
      # Demo schedule defaults (offset/amplitude_ratio/frequency/warmup/sine
      # duration) are left at ScriptedIsmpcSineDemoActionCfg's own defaults,
      # matching the smoke test's known-working values.
    )
  }

  # Same non-learning insurance as zero_residual: rsl_rl needs a non-empty
  # observation set even when nothing reads it.
  terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
  }
  observations = {
    "actor": ObservationGroupCfg(terms=dict(terms), concatenate_terms=True),
    "critic": ObservationGroupCfg(terms=dict(terms), concatenate_terms=True),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
    ),
    observations=observations,
    actions=actions,
    decimation=2,
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


PLAY_CONSOLE_OUTPUT = "single"


def ismpc_demo_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The ISMPC CoM-height sine demo, scripted (no RL, no reward)."""
  cfg = _make_env_cfg(console_output=PLAY_CONSOLE_OUTPUT if play else "none")
  if play:
    _apply_play_overrides(cfg)
  return cfg


def ismpc_demo_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  """Placeholder runner cfg: registration wants one, no policy is ever built."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(32,),
      activation="elu",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.2,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(32,), activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=1,
      num_mini_batches=1,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="mc_rtc_ismpc_demo",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1,
    logger="tensorboard",
  )
