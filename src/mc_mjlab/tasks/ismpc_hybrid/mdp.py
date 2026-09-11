"""Placeholder reward/termination functions for the ismpc_hybrid skeleton.

FOO STAGE: these exist only to make the task well-formed and trainable end
to end (rsl_rl needs a real reward signal and at least one termination
condition to be a sane RL problem). None of this is ISMPC-specific -- no
CoM-height tracking error, no footstep/QP-quality term, nothing that uses
the bridge's get_com_height_ref/qp_succeeded getters yet.

ASSUMED, NOT VERIFIED: the exact function signature mjlab's
RewardTermCfg/TerminationTermCfg expect (`func(env, **params) -> Tensor`).
Modeled on the pattern implied by residual_balance's usage, not confirmed
against mjlab's manager source directly -- check this against a real
manager_base.py/reward_manager.py read if this errors on first run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Constant +1 per env, per step: reward for simply not having terminated
  yet. FOO: the entire "stay alive" pressure in this placeholder reward
  comes from this term plus the termination conditions below -- there is no
  shaping term at all yet."""
  return torch.ones(env.num_envs, device=env.device)


def not_walking_penalty(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Reward component for "is the robot currently stopped" -- returns 1.0
  when NOT walking, 0.0 when walking (note the negation: this is a penalty
  indicator, meant to be combined with a NEGATIVE weight, not a reward
  indicator combined with a positive one -- see the weight note below).
  Named for what it returns (previously misleadingly named `is_walking`,
  which returned the opposite of what that name implied).

  Penalizes staying alive-but-stopped relative to alive-and-walking (see
  the is_alive/not_walking_penalty split rationale: the policy has full
  authority to stop walking for safety, per
  Walking_controller::policyWantsWalk, but should pay a real (if smaller
  than falling) cost for choosing to, so "stop and stand forever" doesn't
  become a dominant strategy).

  Reads action_term.is_walking_obs -- ground truth (Robot_Walking),
  distinct from ismpc_wants_stop_obs (ISMPC's own advisory opinion) and
  from last_walk_action (only the policy's commanded intent): neither of
  those alone is safe to reward against, since the policy could command
  "walking" without ISMPC actually walking, or vice versa. is_walking_obs
  is populated via the ismpc_walking_python bridge's is_walking readback,
  wired through mc_rtc_controller_io_binding.py/mc_rtc_controller_host.py
  same as ismpc_wants_stop.

  Weights (set in ismpc_hybrid_env_cfg.py's rewards dict, not here): the
  user's stated design is alive+walking=10, alive+not-walking=5, i.e. this
  term's own weight should be NEGATIVE and equal to the *gap* (10-5=5,
  so weight=-5) alongside is_alive's weight=10 -- NOT a second full-size
  reward, since is_alive already fires every step regardless of walking
  state; this term only needs to claw back the difference when not
  walking. Placeholder magnitudes per the user, both explicitly FOO.
  """
  action_term = env.action_manager.get_term(action_name)
  return (~action_term.is_walking_obs.bool()).squeeze(-1).to(dtype=torch.get_default_dtype())


def upright_reward(env: ManagerBasedRlEnv, sigma: float = 0.1) -> torch.Tensor:
  """Gaussian-kernel reward for staying upright, via the entity's projected
  gravity vector (parallels residual_balance's orientation shaping,
  simplified). Projected gravity is (0, 0, -1) when upright, so its
  horizontal component's squared magnitude is 0 when upright and grows
  toward 1 as the robot tips toward horizontal.

  sigma=0.1 (applied to the already-squared, dimensionless quantity, so
  effectively sigma^2=0.1 on the raw sum-of-squares): reward ~0.6 at a
  ~8 degree tilt, decays to ~0 by gravity_xy_sq=0.2 -- well before
  fell_over's termination boundary (gravity_xy_threshold=0.7, i.e.
  gravity_xy_sq=0.49), so this term signals trouble well ahead of actual
  termination rather than staying flat until the cliff. Tunable; revisit
  once you can see logged tilt distributions during training.
  """
  entity = env.scene["robot"]
  gravity_b = entity.data.projected_gravity_b
  gravity_xy_sq = torch.sum(gravity_b[:, :2] ** 2, dim=-1)
  return torch.exp(-gravity_xy_sq / (2.0 * sigma**2))


def fell_over(env: ManagerBasedRlEnv, gravity_xy_threshold: float = 0.3) -> torch.Tensor:
  """True once the entity's projected gravity's horizontal component
  exceeds a threshold -- i.e. the robot has tipped over substantially.
  """
  entity = env.scene["robot"]
  gravity_b = entity.data.projected_gravity_b
  return torch.sum(gravity_b[:, :2] ** 2, dim=-1) > gravity_xy_threshold**2


def collapsed(env: ManagerBasedRlEnv, min_root_height: float = 0.20) -> torch.Tensor:
  """True once the entity's root height drops below a threshold.
  """
  entity = env.scene["robot"]
  root_height = entity.data.root_link_pos_w[:, 2]
  return root_height < min_root_height


def last_sine_params(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Current period's physical sine params (offset, amplitude_ratio,
  frequency, phase), as an observation.

  Exposing this (rather than only the raw pre-mapping action mjlab's
  built-in `last_action` observation would give) lets the policy condition
  on physically meaningful quantities directly, and -- per the phase-
  continuity discussion this design pass -- gives it the information it
  needs to choose a new phase that continues smoothly from where the
  previous period's sine left off, rather than having to reconstruct that
  from raw, pre-sigmoid/pre-clamp numbers.
  """
  action_term = env.action_manager.get_term(action_name)
  return action_term.physical_params

def target_linear_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def target_angular_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def last_walk_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """The policy's current walk/stop decision (1.0 = walk, 0.0 = stop), as
  an observation -- mirrors last_sine_params' role for the sine params, so
  the policy can condition on its own last decision directly.
  """
  action_term = env.action_manager.get_term(action_name)
  return action_term.last_walk_action


def last_step_timing_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """The policy's current step-timing (Ts, seconds between footsteps)
  command, as an observation -- mirrors last_walk_action's/
  last_sine_params' role: lets the policy condition on its own last Ts
  decision directly, rather than having to infer it from downstream
  effects (gait cadence, stability error) alone.
  """
  action_term = env.action_manager.get_term(action_name)
  return action_term.last_step_timing_action


def ismpc_wants_stop(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """ISMPC's own advisory safety opinion from the most recent MPC solve
  (1.0 = ISMPC would have stopped walking on its own), as an observation.

  Independent of what the policy actually commanded via the walk-gate
  action: the policy has full, unconditional authority over walking (see
  Walking_controller::policyWantsWalk), so this does NOT reflect the
  controller's actual Stop state. It exists so the policy can learn to
  react to -- or preemptively avoid -- situations where ISMPC's own safety
  logic disagrees with its walk decision, rather than only discovering
  that disagreement's consequences after the fact (e.g. via a fall).
  """
  action_term = env.action_manager.get_term(action_name)
  return action_term.ismpc_wants_stop_obs


def controller_failed(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """True for envs whose mc_rtc controller's QP gave up last step.

  Reads the named action term's `controller_failed` buffer, same pattern
  IsmpcSineAction (and the residual actions) already maintain.
  """
  action_term = env.action_manager.get_term(action_name)
  return action_term.controller_failed


def _target_height_pos(p: dict[str, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
  """CoM-height reference of the sine defined by physical params p, at time t."""
  omega = 2.0 * torch.pi * p["frequency"]
  phase = omega * t
  return p["offset"] + p["sin_amp"] * torch.sin(phase) + p["cos_amp"] * torch.cos(phase)


def _target_height_vel(p: dict[str, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
  """Time-derivative of _sine_height_at, at time t."""
  omega = 2.0 * torch.pi * p["frequency"]
  phase = omega * t
  return p["sin_amp"] * omega * torch.cos(phase) - p["cos_amp"] * omega * torch.sin(phase)


def sine_position_continuity(
  env: ManagerBasedRlEnv, action_name: str, sigma: float = 0.03
) -> torch.Tensor:
  """Gaussian-kernel reward for CoM-height reference continuity across the
  period splice: exp(-(h_curr - h_prev)^2 / (2*sigma^2)), evaluated at the
  instant the switch took effect (action_term._period_t0).

  Deliberately NOT a penalty on the raw or physical *parameters* changing
  (e.g. ||params_t - params_{t-1}||^2): two different (offset,
  amplitude_ratio, frequency, phase) tuples can still produce a continuous
  trajectory at the splice point (e.g. a phase shift that exactly
  compensates a frequency change), and conversely small parameter changes
  can still produce a visible position jump depending on where in the
  cycle the switch lands. Evaluating both sines at the same instant and
  comparing their value directly targets the physically meaningful
  quantity: does the CoM height reference actually jump, which is what
  would inject a spurious feedforward acceleration kick into
  ISMPC_Solver's zc_ddot term (see ismpc_solver_patch.md).

  sigma=0.03m: reward ~0.95 at a 1cm jump, ~0.25 at 5cm, ~0 by 10cm+ --
  tunable, chosen to sit near the boundary between splice discontinuities
  ISMPC's own tracking can likely absorb vs ones large enough to disrupt
  it; revisit once you can see logged jump magnitudes during training.
  """
  action_term = env.action_manager.get_term(action_name)
  h_prev = _target_height_pos(action_term._physical_prev, action_term._period_t0)
  h_curr = _target_height_pos(action_term._physical_curr, action_term._period_t0)
  return torch.exp(-(h_curr - h_prev) ** 2 / (2.0 * sigma**2))


def sine_velocity_continuity(
  env: ManagerBasedRlEnv, action_name: str, sigma: float = 0.3
) -> torch.Tensor:
  """Gaussian-kernel reward for CoM-height reference SLOPE continuity
  across the period splice: exp(-(v_curr - v_prev)^2 / (2*sigma^2)).

  Companion to sine_position_continuity -- see that docstring for why this
  compares the two sines' derivative at the splice instant directly,
  rather than a raw-parameter distance. Kept as a separate reward term
  (rather than combined into one with a blend weight, per an earlier
  design) specifically because height (m) and vertical velocity (m/s) are
  different units: a Gaussian kernel per term, each with its own
  physically meaningful sigma, makes the two terms' weights directly
  comparable to every other Gaussian-kernel reward in this task, instead
  of requiring a hand-tuned blend constant inside a single mixed-units
  squared-error term (that approach's actual problem: velocity carries an
  extra factor of frequency*2*pi, up to ~50 rad/s at this task's frequency
  ceiling, which dominated any small fixed blend weight regardless of its
  value).

  sigma=0.3 m/s: reward ~0.95 at 0.1 m/s, ~0.6 at 0.3 m/s, ~0 above ~1 m/s
  -- tunable, same rationale as sine_position_continuity's sigma.
  """
  action_term = env.action_manager.get_term(action_name)
  v_prev = _target_height_vel(action_term._physical_prev, action_term._period_t0)
  v_curr = _target_height_vel(action_term._physical_curr, action_term._period_t0)
  return torch.exp(-(v_curr - v_prev) ** 2 / (2.0 * sigma**2))


def joint_torque_reward(env: ManagerBasedRlEnv, sigma: float = 100.0) -> torch.Tensor:
  """Gaussian-kernel reward for low joint effort: exp(-sum(tau^2) / (2*sigma^2)),
  summed over all DOFs (currently equivalent to "the target joint subset",
  since IsmpcSineActionCfg.target_actuator_names=(".*",) already covers
  every actuator in this task -- revisit the DOF scoping only if a future
  task variant narrows the action term's target set).

  Legitimate on ordinary robotics/control grounds (reduces actuator heat,
  peak torque demand, standard RL shaping) -- NOT included on the basis of
  any specific human-biomechanics claim; see the removed penalty version's
  docstring for the literature check that ruled that framing out.

  sigma=100 (in sum-of-squared-Nm units, i.e. sigma^2=10000): a rough,
  UNVALIDATED starting guess -- assumes something on the order of
  ~30 Nm across ~12 actively-loaded joints during normal walking
  (30^2 * 12 ~= 10800), giving reward ~0.6 at that rough "typical effort"
  level and decaying by ~2x that. This is not measured against this
  system's actual torques/PD gains/robot mass; set it properly once you
  can see logged sum(qfrc_actuator**2) values from a real training run,
  ideally picking sigma^2 near the middle of the observed range rather
  than guessing.
  """
  return torch.exp(-torch.sum(env.sim.data.qfrc_actuator**2, dim=-1) / (2.0 * sigma**2))


def target_twist(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
  """Currently sampled target walking velocity (vx, vy, wz), as an
  observation. The policy has no direct authority over velocity tracking
  itself (footstep planning/execution is entirely internal to ISMPC), but
  needs this as context: it explains part of what shows up in
  base_lin_vel, and more importantly, the target speed/turn-rate is
  plausibly informative for what CoM-height sine strategy is safe or
  appropriate (e.g. a fast commanded walk likely needs different height
  modulation than near-stationary standing) -- which is squarely the
  policy's actual job.
  """
  return env.command_manager.get_command(command_name)


# --- Debug-only "reward" shims, weight=0.0 in ismpc_hybrid_env_cfg.py. ---
#
# NativeMujocoViewer's native in-viewer plot panel (mjlab/viewer/native.py,
# toggled with the P key during `uv run play`) is wired specifically to
# reward_manager's active terms -- there is no separate "register an
# arbitrary scalar for plotting" hook. Rather than fork/monkeypatch that
# viewer code (which mc_mjlab doesn't own), these three terms expose
# exactly the signals asked for -- target CoM height, step timing (Ts),
# and the walking/stopped flag -- AS zero-weight reward terms purely so
# they ride the same plotting machinery for free. weight=0.0 means they
# contribute nothing to total reward or to training in any way; they exist
# solely to be visible in the play viewer's plot strip. Do not give these
# a nonzero weight without renaming them out of this block and writing a
# real docstring justifying the shaping choice -- their current docstrings
# describe plotting, not reward design.


def debug_target_height(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- live CoM-height reference (m), for the
  play viewer's native plot panel. See
  IsmpcSineAction.target_height_obs's docstring for what this is (and is
  not) ground truth for."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.target_height_obs.squeeze(-1)


def debug_step_timing(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- current commanded Ts (s), for the play
  viewer's native plot panel."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.last_step_timing_action.squeeze(-1)


def debug_is_walking(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- ground-truth walking/stopped flag (1.0 =
  walking, 0.0 = stopped), for the play viewer's native plot panel.
  Reads is_walking_obs (Robot_Walking ground truth from the controller),
  NOT last_walk_action (the policy's own commanded intent) -- for a
  debugging display you want to see what's actually happening, not just
  what was asked for; see is_walking_obs's docstring for why the two can
  legitimately disagree."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.is_walking_obs.squeeze(-1)