"""Placeholder reward/termination functions for the ismpc_hybrid skeleton.

FOO STAGE: these exist only to make the task well-formed and trainable end
to end (rsl_rl needs a real reward signal and at least one termination
condition to be a sane RL problem). None of this is ISMPC-specific -- no
CoM-height tracking error, no footstep/QP-quality term, nothing that uses
the bridge's get_com_height_ref/qp_succeeded getters yet.

Ported from the old Cython-bridge ismpc_hybrid/mdp.py onto the native
`mc_rtc_interface`-backed `IsmpcSineAction` (mc_mjlab/actions/ismpc_sine_action.py).
Every function here is unchanged except `ismpc_wants_stop` and
`debug_is_walking`, which now read `action_term.ismpc_wants_stop_obs` /
`action_term.is_walking_obs` -- both properties already exist on the new
action term (backed by the generic datastore-output machinery inherited
from `McRtcActionBase`) with the same names/shapes as before, so nothing
else in this file changes. `sine_position_continuity`/`sine_velocity_continuity`
keep reaching directly into `action_term._physical_prev`/`_physical_curr`/
`_period_t0`, which exist on `IsmpcSineAction` with the same names.

ASSUMED, NOT VERIFIED: the exact function signature mjlab's
RewardTermCfg/TerminationTermCfg expect (`func(env, **params) -> Tensor`).
Modeled on the pattern implied by residual_balance's usage, not confirmed
against mjlab's manager source directly -- check this against a real
manager_base.py/reward_manager.py read if this errors on first run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import apply_body_impulse
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
  is populated via the mc_rtc_interface datastore output
  `ismpc_walking::robot_walking_d`, collected each control period the same
  way every other datastore getter is (McRtcActionBase._collect_controller_output).

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
  """Current period's physical sine params (offset, frequency, sin_amp,
  cos_amp), as an observation.

  Exposing this (rather than only the raw pre-mapping action mjlab's
  built-in `last_action` observation would give) lets the policy condition
  on physically meaningful quantities directly, and -- per the phase-
  continuity discussion this design pass -- gives it the information it
  needs to choose new sine params that continue smoothly from where the
  previous period's sine left off, rather than having to reconstruct that
  from raw, pre-clamp numbers.
  """
  action_term = env.action_manager.get_term(action_name)
  return action_term.physical_params


def last_twist_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """The policy's current (vx, vy, omega) twist command, as an observation
  -- mirrors last_sine_params'/last_step_timing_action's role, just for the
  reference-velocity channel. This is the rate-limited value actually
  written to ismpc_walking::set_rl_ref_vel this period (see
  IsmpcSineAction.last_twist_action), not the raw pre-mapping action."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.last_twist_action


def estimated_com_lin_vel(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """CoM linear velocity (vx, vy, vz) in base axes, as ESTIMATED by the
  controller (ismpc_walking::get_com_lin_vel ->
  Walking_controller::estimatedComLinVel). Replaces MuJoCo ground-truth
  envs_mdp.base_lin_vel so training and deployment read the same signal."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.com_lin_vel_est_obs


class target_vel_window:
  """Boxcar-window velocity tracking for ONE axis (x or y).

  Averages the achieved root-link body-frame velocity on `axis` over the
  last window_s seconds and rewards it matching the command on that axis:
  exp(-(command - avg)^2 / std^2). Instantiate once per axis (two
  RewardTermCfg entries) so each axis gets its own std and weight.

  ASSUMED, not verified: the reward manager constructs a class passed as
  `func` as cls(cfg, env), reading axis from cfg.params. It does for the
  event terms (settle_gated_apply_body_impulse). If it errors at startup,
  tell me and I'll switch to a plain function with state stored on env.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._axis = int(cfg.params["axis"])
    self._angular = bool(cfg.params.get("angular", False))
    self._buf = None  # allocated on first call (needs window_s)
    self._idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._rows = torch.arange(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    axis: int,
    std: float = 0.05,
    window_s: float = 1.1,
    angular: bool = False, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    del axis  # consumed in __init__ via cfg.params
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, self._axis]
    vel = (
      asset.data.root_link_ang_vel_b[:, 2]
      if self._angular
      else asset.data.root_link_lin_vel_b[:, self._axis]
    )

    if self._buf is None:
      n = max(1, round(window_s / env.step_dt))
      self._buf = torch.zeros(env.num_envs, n, device=env.device)
    n = self._buf.shape[1]

    fresh = env.episode_length_buf <= 1
    self._idx = torch.where(fresh, torch.zeros_like(self._idx), self._idx)
    self._count = torch.where(fresh, torch.zeros_like(self._count), self._count)

    self._buf[self._rows, self._idx] = vel
    self._idx = (self._idx + 1) % n
    self._count = torch.clamp(self._count + 1, max=n)

    slot = torch.arange(n, device=env.device).unsqueeze(0)
    valid = slot < self._count.unsqueeze(-1)
    avg = (self._buf * valid).sum(dim=1) / self._count

    return torch.exp(-torch.square(cmd - avg) / std**2)


def twist_tutor(
  env: ManagerBasedRlEnv,
  action_name: str,
  command_name: str,
  axis: int,
  std: float = 0.1,
) -> torch.Tensor:
  """Dense guidance ("tutor") term: exp(-(policy twist output - command)^2 / std^2)
  on one axis (0=vx, 1=vy, 2=omega).

  Reads action_term.last_twist_action -- the physical, rate-limited twist
  actually written to ismpc_walking::set_rl_ref_vel -- so it rewards the
  policy's own twist output (raw[6:9]) directly, with no gait lag and no
  gait ripple, unlike target_vel_window, which scores the ACHIEVED velocity
  averaged over a window. It is meant to carry a DECAYING weight (see the
  tutor_* reward_curriculum entries in ismpc_hybrid_env_cfg.py): strong
  early, to give the network a dense signal linking the target_twist
  observation to raw[6:9]; weak late, so the policy is free to deviate from
  the command when that pays off (e.g. to recover from a push). Keep the
  achieved-velocity terms in the reward: a policy can satisfy this term
  perfectly while the robot does not actually achieve the velocity.
  """
  action_term = env.action_manager.get_term(action_name)
  out = action_term.last_twist_action[:, axis]
  cmd = env.command_manager.get_command(command_name)[:, axis]
  return torch.exp(-torch.square(out - cmd) / std**2)


class target_linear_vel_avg:
  """Windowed-average version of target_linear_vel.

  Compares the COMMAND against an exponential moving average of the
  achieved root-link body-frame linear velocity (time constant window_s,
  default 1.2 s ~ one gait cycle), instead of the instantaneous velocity.
  Rationale: ISMPC's CoM velocity oscillates within each step, so an
  instantaneous kernel can never reach ~1 even for a perfect gait.

  Stateful callable class (same pattern as settle_gated_apply_body_impulse):
  the manager constructs it with (cfg, env) and then calls it each step.
  ASSUMED, not verified against mjlab's reward manager: that a class passed
  as `func` is instantiated this way for reward terms too. It is for event
  terms (your settle_gated_apply_body_impulse works), but rewards may
  differ. If it errors at startup, tell me and I'll convert it to a plain
  function with the state stored on `env`.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._avg = torch.zeros(env.num_envs, 2, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.3,
    window_s: float = 1.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    vel = asset.data.root_link_lin_vel_b[:, :2]

    # First step(s) of an episode: seed the average with the current
    # velocity so no state carries over from the previous episode.
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    self._avg = torch.where(fresh, vel, self._avg)

    alpha = min(1.0, env.step_dt / window_s)
    self._avg = (1.0 - alpha) * self._avg + alpha * vel

    err = torch.sum(torch.square(command[:, :2] - self._avg), dim=1)
    return torch.exp(-err / std**2)

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

def debug_target_omega(
  env: ManagerBasedRlEnv, command_name: str = "twist"
) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- commanded yaw rate (rad/s), column 2 of
  the `twist` command. Raw input the policy observes; see
  debug_target_vel_x."""
  return env.command_manager.get_command(command_name)[:, 2]


def debug_output_omega(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- the POLICY's own commanded yaw rate
  (rad/s), rate-limited and mapped to physical units. Column 2 of
  action_term.last_twist_action; see debug_output_vel_x."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.last_twist_action[:, 2]


def debug_base_omega(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- ACHIEVED root-link yaw rate (rad/s), body
  frame. Same quantity target_angular_vel compares against the command
  (root_link_ang_vel_b[:, 2]). Root link, not true CoM/heading."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b[:, 2]


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

  Reads action_term.ismpc_wants_stop_obs, backed by the
  `ismpc_walking::wants_stop_d` datastore output collected through
  McRtcActionBase's generic datastore machinery -- replaces the old
  Cython-bridge file's ControllerIoBinding.read_ismpc_wants_stop readback.
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
  (e.g. ||params_t - params_{t-1}||^2): two different (offset, frequency,
  sin_amp, cos_amp) tuples can still produce a continuous trajectory at the
  splice point (e.g. an amplitude split that exactly compensates a
  frequency change), and conversely small parameter changes can still
  produce a visible position jump depending on where in the cycle the
  switch lands. Evaluating both sines at the same instant and comparing
  their value directly targets the physically meaningful quantity: does
  the CoM height reference actually jump, which is what would inject a
  spurious feedforward acceleration kick into ISMPC_Solver's zc_ddot term
  (see ismpc_solver_patch.md).

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


# Lower body only: both legs + waist, per the robot's own _ref_joint_order
# naming (RCY/RCR/RCP/RKP/RAP/RAR = right hip yaw/roll/pitch, knee pitch,
# ankle pitch/roll; L-prefixed = left leg mirror; WP/WR/WY = waist
# pitch/roll/yaw). Deliberately excludes HY/HP (head/neck) and every
# arm/hand joint (RS*/LS*/RE*/LE*/RW*/LW*/RH*/LH*, plus the finger joints
# RT*/RI*/RM*/LT*/LI*/LM*) -- confirmed against the robot's actual joint
# list, not guessed from a general humanoid naming convention.
LOWER_BODY_JOINT_NAMES = (
  "RCY", "RCR", "RCP", "RKP", "RAP", "RAR",
  "LCY", "LCR", "LCP", "LKP", "LAP", "LAR",
  "WP", "WR", "WY",
)


def _resolve_lower_body_joint_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> None:
  """Resolve asset_cfg.joint_ids in place, once, on first call.

  SceneEntityCfg.joint_ids defaults to slice(None) (= "every joint") until
  .resolve(scene) is called -- confirmed empirically: a bare
  SceneEntityCfg("robot", joint_names=["RCY"]) still reports
  joint_ids == slice(None) before resolve() runs. Nothing in the reward
  manager path calls this automatically for a plain function parameter (unlike
  the action term's own _target_ids, which IS resolved by BaseAction.__init__
  from cfg.actuator_names). So this function must call resolve() itself,
  exactly once -- resolve() mutates asset_cfg.joint_ids in place from a list
  to... still a list (or slice(None) if it happened to match every joint,
  which it won't here since this is a strict subset), so checking
  `isinstance(asset_cfg.joint_ids, list)` is a safe, idempotent "already
  resolved" guard for every call after the first.
  """
  if isinstance(asset_cfg.joint_ids, list):
    return
  asset_cfg.resolve(env.scene)


def joint_torque_reward(
  env: ManagerBasedRlEnv,
  sigma: float = 100,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_NAMES),
) -> torch.Tensor:
  """Gaussian-kernel reward for low joint effort: exp(-sum(tau^2) / (2*sigma^2)),
  summed over LOWER BODY JOINTS ONLY (both legs + waist, 15 DOF -- see
  LOWER_BODY_JOINT_NAMES). Previously summed every actuated DOF, arms/hands/
  head included, via the raw env.sim.data.qfrc_actuator with no joint
  filtering at all -- that meant an idle arm sitting at zero torque was
  diluting the signal from legs, which is where effort actually matters for
  a walking task, and made sigma impossible to tune meaningfully (a robot
  holding its arms still contributes ~0 regardless of how the legs behave).

  IMPORTANT -- asset_cfg default is a module-level SceneEntityCfg instance,
  shared across every env that doesn't pass its own asset_cfg (standard
  Python mutable-default-argument sharing, same as any dataclass instance
  used as a default). This is intentional here, not a bug: joint topology
  is identical across all parallel envs of a single run, so resolving once
  and caching on that shared instance is exactly the right behavior, not a
  cross-env leak (no per-env state is stored on it, only the fixed
  name->id mapping). Do NOT give this function's asset_cfg per-instance
  mutable state beyond joint_ids/joint_names resolution if it's ever
  extended.

  sigma=100 (in sum-of-squared-Nm units, i.e. sigma^2=10000): a rough,
  UNVALIDATED starting guess -- STILL not measured against this system's
  actual lower-body torques/PD gains/robot mass now that the DOF scope has
  changed (previously it was an all-DOF guess, which is now a different,
  larger-population quantity, so if a real distribution was fit against the
  old all-joint sum, it doesn't carry over -- 12 idle arm/hand DOFs no
  longer pad the sum toward zero, so the real 15-DOF sum(tau^2) is likely
  SMALLER and more sharply varying than the guess this sigma was based on).
  Use debug_joint_torque_raw (weight=0.0, wired into uv run play like the
  other debug_* terms) to see actual logged sum(tau^2) values before
  trusting this sigma -- pick sigma^2 near the middle of the observed
  range, not by inspection of this docstring's arithmetic.
  """
  _resolve_lower_body_joint_ids(env, asset_cfg)
  asset = env.scene[asset_cfg.name]
  tau_sq_sum = torch.sum(asset.data.qfrc_actuator[:, asset_cfg.joint_ids] ** 2, dim=-1)
  return torch.exp(-tau_sq_sum / (2.0 * sigma**2))


def debug_joint_torque_raw(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_NAMES),
) -> torch.Tensor:
  """PLOTTING-ONLY (weight=0.0 in ismpc_hybrid_env_cfg.py, same convention
  as debug_target_height/debug_step_timing/debug_is_walking) -- the raw,
  unscaled sum(tau^2) over the same lower-body joint set joint_torque_reward
  uses, in sum-of-squared-Nm units, with NO Gaussian kernel applied. Exists
  purely so `uv run play` (console_output="all" in play mode, per
  ismpc_hybrid_env_cfg.ismpc_hybrid_env_cfg) shows this in the reward-terms
  readout, letting you read off real typical/peak values for sigma tuning
  instead of guessing -- see joint_torque_reward's docstring. Because this
  shares the SAME module-level default asset_cfg object as
  joint_torque_reward (both reference the identical LOWER_BODY_JOINT_NAMES
  tuple, but construct their OWN separate SceneEntityCfg default per
  function signature -- Python evaluates each default expression once per
  def, not shared between the two functions), it resolves independently;
  harmless, just a second one-time resolve() call rather than a shared
  cache across both functions.
  """
  _resolve_lower_body_joint_ids(env, asset_cfg)
  asset = env.scene[asset_cfg.name]
  return torch.sum(asset.data.qfrc_actuator[:, asset_cfg.joint_ids] ** 2, dim=-1)


def target_twist(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
  """Currently sampled target walking velocity (vx, vy, wz), as an
  observation.

  The policy now has direct authority over velocity tracking via the twist
  action (IsmpcSineAction._map_twist -> ismpc_walking::set_rl_ref_vel,
  effective only while the controller-side rlVelocityControl toggle is on
  -- see that action's module docstring). This observation remains useful
  independent of that: it is the *target* the target_linear_vel/
  target_angular_vel reward terms measure tracking against, and -- when
  rlVelocityControl happens to be off, or during the settle window right
  after reset before the twist action has taken effect -- it still explains
  part of what shows up in base_lin_vel and is plausibly informative for
  what CoM-height sine strategy is safe or appropriate (e.g. a fast
  commanded walk likely needs different height modulation than
  near-stationary standing).
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

def debug_base_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    """PLOTTING ONLY (weight=0.0) -- achieved root-link height (m)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2]

def debug_step_timing(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- current commanded Ts (s), for the play
  viewer's native plot panel."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.last_step_timing_action.squeeze(-1)


def debug_is_walking(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- ground-truth walking/stopped flag (1.0 =
  walking, 0.0 = stopped), for the play viewer's native plot panel.
  Reads is_walking_obs (Robot_Walking ground truth from the controller,
  via the `ismpc_walking::robot_walking_d` datastore output), NOT
  last_walk_action (the policy's own commanded intent) -- for a debugging
  display you want to see what's actually happening, not just what was
  asked for; see is_walking_obs's docstring for why the two can
  legitimately disagree."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.is_walking_obs.squeeze(-1)


def debug_target_vel_x(
  env: ManagerBasedRlEnv, command_name: str = "twist"
) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- commanded reference velocity, x
  component (m/s), for the play viewer's native plot panel. This is the
  RAW INPUT the policy receives as an observation (target_twist / the
  `twist` command sampled by UniformVelocityCommandCfg), not anything the
  policy itself produced -- plotted alongside debug_output_vel_x so the two
  can be visually compared to check both that the command is actually
  reaching the robot/policy, and that the policy is learning to track it."""
  return env.command_manager.get_command(command_name)[:, 0]


def debug_target_vel_y(
  env: ManagerBasedRlEnv, command_name: str = "twist"
) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- commanded reference velocity, y
  component (m/s). See debug_target_vel_x's docstring; same source
  (env.command_manager.get_command), just column 1 instead of column 0."""
  return env.command_manager.get_command(command_name)[:, 1]


def debug_output_vel_x(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- the POLICY's own commanded twist, x
  component (m/s), after mapping to physical units. Reads
  action_term.last_twist_action (IsmpcSineAction._map_twist's output) --
  the same rate-limited, already-physical value actually written to
  ismpc_walking::set_rl_ref_vel this period, NOT the raw pre-mapping
  action. Compare against debug_target_vel_x: if rlVelocityControl is on
  and the policy has learned to track the command, these two should
  converge; if they diverge, either the command isn't reaching the policy
  as an observation, or the policy hasn't learned to track it yet -- this
  pair of plots is what distinguishes those two failure modes visually."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.last_twist_action[:, 0]


def debug_output_vel_y(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- the POLICY's own commanded twist, y
  component (m/s), after mapping to physical units. See
  debug_output_vel_x's docstring; same source
  (action_term.last_twist_action), just column 1 instead of column 0."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.last_twist_action[:, 1]

def debug_base_vel_x(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- ACHIEVED root-link linear velocity, x
  component (m/s), in the body frame. Same quantity target_linear_vel
  compares against the command (root_link_lin_vel_b), so this is directly
  comparable to debug_target_vel_x / debug_output_vel_x. Root link, not
  true CoM."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, 0]


def debug_base_vel_y(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- achieved root-link linear velocity, y
  component (m/s), body frame. See debug_base_vel_x."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, 1]


def debug_est_com_vel_x(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- controller-ESTIMATED CoM linear velocity,
  x (m/s, base axes). Compare with debug_base_vel_x (root link, ground truth):
  same sign and scale on average, not stuck at zero. Not identical: the CoM
  oscillates around the base velocity while walking."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.com_lin_vel_est_obs[:, 0]


def debug_est_com_vel_y(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """PLOTTING ONLY (weight=0.0) -- estimated CoM linear velocity, y (m/s,
  base axes). See debug_est_com_vel_x."""
  action_term = env.action_manager.get_term(action_name)
  return action_term.com_lin_vel_est_obs[:, 1]


class settle_gated_apply_body_impulse:
  """Wraps mjlab.envs.mdp.events.apply_body_impulse so it cannot trigger a
  push during the first ``settle_ticks`` steps of an episode.

  Root cause this fixes: apply_body_impulse's OWN reset() already resamples
  a fresh cooldown at every episode reset (so a push can't fire at literally
  tick 0), but that cooldown is drawn from the ordinary cooldown_s RANGE --
  nothing prevents an unlucky low draw from landing in the first few ticks
  of a fresh episode. mode="step" means this can happen on ANY tick,
  including the very first ones, independent of IsmpcSineAction's own
  one-tick settle guard (_dispatch_ticks_since_sine_update's reset-time
  seed to 1) for the POLICY's own actions -- that guard only covers the
  policy's sine/walk-gate/Ts outputs, it has no knowledge of, and provides
  no protection against, this separate push event. An early push landing
  before the robot/policy has had any chance to settle can trigger an
  immediate fell_over/collapsed, which resets the env again, which can
  itself draw another unlucky early cooldown -- for a small, unlucky subset
  of envs this compounds into the exact "resets every single tick, forever"
  pattern seen in the reset_race_guard spam this was written to fix (see
  the training-investigation thread: envs 74/295/87 discarding stale
  collected results on essentially every collect() cycle, while ~297 other
  envs were fine -- consistent with only a small unlucky subset ever
  drawing a bad early cooldown).

  Mechanism: delegates every call to a real, internally-held
  apply_body_impulse instance (constructed with the SAME cfg/params this
  wrapper receives, minus settle_ticks itself), but before delegating,
  forces _interval_time_left back up to at least settle_ticks worth of
  seconds for any env still inside its settle window -- so the wrapped
  instance's own cooldown-expiry check can never see a <=0 value for those
  envs, and its trigger branch is simply never reached for them. This is
  NOT "let it trigger then zero the force": the wrapped instance's internal
  state is corrected BEFORE its trigger logic runs, so no impulse is ever
  computed or written for a settling env in the first place.

  cfg.params must include everything apply_body_impulse itself needs
  (asset_cfg, force_range, torque_range, duration_s, cooldown_s), PLUS
  settle_ticks (int, number of env steps after reset during which pushes
  are suppressed for that env).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._env = env
    self._settle_ticks = int(cfg.params["settle_ticks"])
    # The wrapped instance must NOT see settle_ticks in its own params --
    # apply_body_impulse.__call__ has no such kwarg and would raise on an
    # unexpected argument.
    inner_params = {k: v for k, v in cfg.params.items() if k != "settle_ticks"}

    class _InnerCfg:
      params = inner_params

    self._inner = apply_body_impulse(_InnerCfg(), env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    settle_ticks: int,
    **inner_kwargs,
  ) -> None:
    del env_ids  # apply_body_impulse itself always operates on all envs.
    still_settling = env.episode_length_buf < self._settle_ticks
    if bool(still_settling.any()):
      # Push the wrapped instance's cooldown timer back out for settling
      # envs, every step, so it never counts down to a trigger for them.
      # settle_ticks is a step count; convert to the same seconds unit
      # _interval_time_left is tracked in.
      floor_s = self._settle_ticks * env.step_dt
      current = self._inner._interval_time_left
      self._inner._interval_time_left = torch.where(
        still_settling, torch.clamp(current, min=floor_s), current
      )
    self._inner(env, None, **inner_kwargs)

  def debug_vis(self, visualizer) -> None:
    if hasattr(self._inner, "debug_vis"):
      self._inner.debug_vis(visualizer)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if hasattr(self._inner, "reset"):
      self._inner.reset(env_ids)

class stage_curriculum:
  """Staged curriculum over command ranges and push force.

  Each stage is a dict:
    {"step": int,                      # env steps (env.common_step_counter) at which it starts
     "lin_vel_x": (lo, hi), "lin_vel_y": (lo, hi), "ang_vel_z": (lo, hi),
     "push_scale": float}              # multiplier on the base force ranges
  The active stage is the last one whose "step" <= common_step_counter,
  unless params["fixed_stage"] is not None, in which case that index is
  forced (used by play to pin a stage).

  Mutates live cfgs each call, like mjlab's reward_curriculum:
    * command ranges: env.command_manager.get_term(command_name).cfg.ranges
    * push force: the event terms' params["force_range"]
  ASSUMED, not verified: get_term(...).cfg.ranges is the live object the
  command sampler reads on resample (mjlab's own commands_vel presumably
  does this, but I haven't read its source), and that Ranges is mutable.
  Ranges only take effect at the NEXT command resample (once per episode,
  resampling_time_range = EPISODE_LENGTH_S), so a stage change reaches
  each env at its next reset, not instantly.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._stages = p["stages"]
    self._fixed = p.get("fixed_stage")
    self._cmd_cfg = env.command_manager.get_term(p["command_name"]).cfg
    self._push_cfgs = [env.event_manager.get_term_cfg(n) for n in p["push_event_names"]]
    # Base (full-strength) force ranges, captured once from the cfg.
    self._base_force = [c.params["force_range"] for c in self._push_cfgs]
    self._last = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids,
    command_name: str,
    push_event_names: list[str],
    stages: list[dict],
    fixed_stage: int | None = None,
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, push_event_names, stages, fixed_stage
    if self._fixed is not None:
      idx = int(self._fixed)
    else:
      idx = 0
      for i, s in enumerate(self._stages):
        if env.common_step_counter >= s["step"]:
          idx = i
    if idx != self._last:
      s = self._stages[idx]
      r = self._cmd_cfg.ranges
      r.lin_vel_x = s["lin_vel_x"]
      r.lin_vel_y = s["lin_vel_y"]
      r.ang_vel_z = s["ang_vel_z"]
      k = float(s["push_scale"])
      for c, (lo, hi) in zip(self._push_cfgs, self._base_force):
        c.params["force_range"] = (lo * k, hi * k)
      self._last = idx
    return {"stage": torch.tensor(float(idx))}