"""Learned CoM-height sine parameter action for ISMPC.

Unlike ScriptedIsmpcSineDemoAction (fixed time-based schedule, proof of the
plumbing), this action's 4 sine parameters -- offset, amplitude_ratio,
frequency, phase -- come from the RL policy's own output each period, mapped
from raw (unconstrained) actions into physical units via the scale/clip/
sigmoid rules worked out earlier in this project:

  offset            = clip(offset_scale * raw[0] + offset_bias, OFFSET_MIN, OFFSET_MAX)   (m)
  amplitude_ratio    = sigmoid(raw[1])                                       (0, 1)
  amplitude          = amplitude_ratio * offset   -- derived, not learned directly;
                        guarantees offset - amplitude >= 0 structurally, no
                        clamp needed anywhere downstream (mc_mjlab or C++)
  frequency          = clip(freq_scale * raw[2] + freq_bias, 0.1, 8.0)       (Hz)
  phase              = wrap_to_pi(raw[3])                                    (rad)

Like the scripted demo, joint actuation itself is unchanged: mc_rtc's own
q/alpha output drives the joints (no joint-space residual in this task --
the whole point is that RL only ever touches the ISMPC parameter, never
touches joint commands directly, per the user's stated design).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from mc_mjlab.actions.mc_rtc_controller_io_binding import ControllerIoBinding
from mc_mjlab.actions.mc_rtc_controller_pool import ControllerPool

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# --- Physical ranges, locked in earlier in this project. ---
OFFSET_MIN, OFFSET_MAX = 0.4, 1.05  # m
offset_bias = 0.9

FREQUENCY_MIN, FREQUENCY_MAX = 0.1, 8.0  # Hz

AMPLITUDE_SCALE = 0.10  # new constant, tune this: raw~O(1) -> physical amplitude ~O(0.05)

# Step timing (Ts, seconds between footsteps). Matches
# Walking_controller::kDefaultTSteps (1.1) and the controller's own
# ts_range clamp (0.4-2.0, from mc_rtc.yaml's ismpc.ts_range) -- this range
# is a soft/exploration-shaping bound on top of that hard controller-side
# clamp, not a replacement for it.
TS_MIN, TS_MAX = 0.4, 2.0  # s
TS_DEFAULT = 1.1  # s -- must match Walking_controller::kDefaultTSteps


@dataclass(kw_only=True)
class IsmpcSineActionCfg(ActionTermCfg):
  """Configuration for the learned ISMPC CoM-height sine parameter action.

  Subclasses ActionTermCfg directly, NOT BaseActionCfg: this action controls
  no per-joint actuators at all (its action space is a fixed 6 -- offset,
  amplitude_ratio, frequency, phase, walk-gate, step timing -- unrelated to
  joint count), so BaseActionCfg's actuator_names/scale/offset machinery
  (sized off matched joints) does not apply here. Joint actuation is still
  driven by mc_rtc's own q/alpha output, same as the residual actions, but
  that's wiring internal to this action, not something ActionTermCfg needs
  to know about.
  """

  target_actuator_names: tuple[str, ...] = (".*",)
  """Actuator names (regex) mc_rtc drives on the entity -- NOT the RL
  action's targets (which are the 4 fixed sine params); this only selects
  which of the entity's joints receive mc_rtc's q/alpha output."""

  mc_rtc_config_path: str
  """Path to the mc_rtc configuration file."""

  mc_rtc_robot_name: str = "jvrc1"
  """Name of the robot in mc_rtc."""

  frameskip: int = 1
  """Physics substeps between controller `run()` calls. Must match the
  controller's own configured Timestep (e.g. mc_rtc.yaml Timestep=0.002s,
  physics timestep=0.001s -> frameskip=2): the FSM, footstep planner, and
  stabilizer inside mc_rtc are stateful and integrate every call, so
  skipping calls starves them regardless of anything ISMPC-specific. This
  is NOT the same cadence the sine parameters should update at -- see
  sine_param_frequency_hz."""

  sine_param_frequency_hz: float = 20.0
  """How often (Hz) the RL-set sine params are pushed into the controller,
  matching ISMPC_Solver's own MPC solve period (m_delta=0.05s -> 20Hz by
  default). The controller itself still gets stepped every `frameskip`
  physics substeps regardless -- only the sine-parameter *write* (and the
  continuity-penalty bookkeeping tied to it) happens on this slower
  cadence. Conflating the two (using one frameskip value for both) starves
  the controller's own internal state if frameskip is set slow enough to
  match the MPC period, which is what originally broke the pendulum
  feasibility solver in this task."""

  num_workers: int | None = None
  """Worker process count; None = min(num_envs, cpu_count - 2)."""

  use_worker_processes: bool = True

  pd_gains_path: str | None = None
  """Optional mc_mujoco PDgains_sim.dat overriding the entity's PD gains."""

  use_controller_reset: bool = True

  console_output: str = "none"

  offset_scale: float = 0.075
  offset_bias: float = 0.9

  frequency_scale: float = torch.log(torch.tensor(FREQUENCY_MAX / FREQUENCY_MIN)).item() / 2
  frequency_bias: float = torch.log(torch.tensor(FREQUENCY_MIN * FREQUENCY_MAX)).item() / 2

  """Scales raw_sin_amp/raw_cos_amp into physical sin_amp/cos_amp (m) before
    the offset-based radius clamp. Combined reachable oscillation radius at
    raw values of magnitude r (per-axis) is roughly amplitude_scale * r *
    sqrt(2) if both axes are excited equally; tune alongside the policy's
    action-distribution std to target a specific typical-exploration ceiling
    rather than only the structural (offset-clamped) maximum."""
  amplitude_scale: float = 0.10

  """Shifts the walk/stop decision threshold away from 0. Positive values
  make 'walk' more likely at policy init (raw actions near 0); negative
  values make 'stop' more likely. Leave at 0.0 for the original unbiased
  50/50 behavior."""
  walk_gate_bias: float = 0.0

  """Scales raw_ts into physical Ts (s) before the TS_MIN/TS_MAX clamp:
  ts = clip(ts_scale * raw_ts + ts_bias, TS_MIN, TS_MAX). ts_bias = 1.1
  centers the mapping on the controller's own default/reset value, so
  raw_ts=0 (a freshly-initialized policy's typical early output) reproduces
  today's fixed-Ts behavior exactly. ts_scale = 0.35 keeps raw~O(1)
  excursions (+-1) comfortably inside the range (Ts in [0.75, 1.45]) --
  reaching the actual TS_MIN/TS_MAX clamp edges needs |raw_ts| > ~2.6, so
  clipping is a rare tail event early in training, not routine, matching
  the "extreme values unlikely at first" requirement this was tuned for."""
  ts_scale: float = 0.35
  ts_bias: float = TS_DEFAULT

  def build(self, env: ManagerBasedRlEnv) -> "IsmpcSineAction":
    return IsmpcSineAction(self, env)


class IsmpcSineAction(ActionTerm):
  """Maps a 4-dim raw RL action to ISMPC CoM-height sine parameters.

  Subclasses ActionTerm directly (not BaseAction): implements the plain
  4-method contract (action_dim, process_actions, apply_actions, raw_action)
  itself, since BaseAction's per-joint-actuator machinery has no meaning for
  a 4-dim, non-joint action space.
  """

  cfg: IsmpcSineActionCfg

  def __init__(self, cfg: IsmpcSineActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self.cfg = cfg

    # mc_rtc's own joint targets (q/alpha) still need somewhere on the
    # entity to land -- resolved the same way BaseAction would, but done
    # explicitly here since we are not subclassing it.
    self._target_ids, self._target_names = self._entity.find_joints_by_actuator_names(
      cfg.target_actuator_names
    )
    self._target_ids = torch.tensor(
      self._target_ids, device=self.device, dtype=torch.long
    )

    self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    # Sine-param update cadence, in units of controller-dispatch ticks (not
    # physics substeps): e.g. frameskip=2 (500Hz controller) with
    # sine_param_frequency_hz=20 means the controller dispatches every
    # tick, but only every 25th dispatch actually carries new sine params.
    # See IsmpcSineActionCfg.sine_param_frequency_hz for why this is kept
    # separate from frameskip.
    #
    # UNVERIFIED: env.step_dt is confirmed elsewhere in this file/the
    # scripted demo action to be the RL-decision-step dt (physics_timestep
    # * decimation), not the raw physics timestep -- so dividing it by
    # cfg.frameskip here is only correct if decimation == frameskip for
    # this task (true in both ismpc_hybrid and ismpc_demo's cfgs as of this
    # writing, but not something this action can see or enforce). If you
    # ever decouple decimation from frameskip, this derivation breaks
    # silently. Confirm against mjlab's actual SimulationCfg/env attributes
    # (e.g. whether env.sim.mujoco.timestep or similar exists) before
    # relying on this in a task where decimation != frameskip.
    controller_hz = 1.0 / (self._env.step_dt / cfg.frameskip)
    self._sine_param_period_ticks = max(
      1, round(controller_hz / cfg.sine_param_frequency_hz)
    )
    self._dispatch_ticks_since_sine_update = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Same transport/host machinery the residual actions use: one real
    # mc_rtc controller per env, worker processes, shared-memory I/O.
    self._pool = ControllerPool(
      cfg.mc_rtc_config_path,
      self.num_envs,
      self._target_names,
      num_workers=cfg.num_workers,
      use_worker_processes=cfg.use_worker_processes,
      console_output=cfg.console_output,
    )
    metadata = self._pool.await_ready()

    self._io = ControllerIoBinding(
      self._env,
      self._entity,
      self._target_names,
      self._target_ids,
      metadata,
      cfg.use_controller_reset,
      output_channels=("q", "alpha"),
      has_ismpc_sine=True,
      has_ismpc_velocity=True,
      has_ismpc_walk_gate=True,
      has_ismpc_ts=True,
    )
    if cfg.pd_gains_path is not None:
      from mc_mjlab.actions.mc_rtc_controller_io_binding import (
        apply_reference_pd_gains,
      )

      apply_reference_pd_gains(
        self._entity, metadata.ref_joint_order, self._target_names, cfg.pd_gains_path
      )

    self._pool.configure(self._io.layout)
    self._in_np = self._pool.in_np
    self._out_np = self._pool.out_np

    self._previous_control = {
      c: torch.zeros(self.num_envs, len(self._target_names), device=self.device)
      for c in ("q", "alpha")
    }
    self._next_control = {
      c: torch.zeros(self.num_envs, len(self._target_names), device=self.device)
      for c in ("q", "alpha")
    }
    self._staged_control = {
      c: torch.zeros(self.num_envs, len(self._target_names), device=self.device)
      for c in ("q", "alpha")
    }
    self._has_staged_control = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._steps_since_run = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Plain-Python mirror of the counter above, used ONLY to compute
    # run_indices in apply_actions() without a GPU nonzero()/tolist() sync
    # (measured at ~11s/12800 steps, the single largest .nonzero() cost in
    # training). Must be kept in lockstep with the GPU tensor at every
    # mutation site: the reset zeroing below and the += 1 in apply_actions.
    self._steps_since_run_cpu = [0] * self.num_envs

    self.controller_failed = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # --- Reset/dispatch race guard. ---
    # A controller step can be dispatched to a worker, then that env gets
    # reset (episode end) BEFORE the worker's result is collected. Without
    # this guard, apply_actions()'s next collect() would stage/apply that
    # stale, pre-reset result into joint_pos_target/joint_vel_target right
    # after reset() had just set them correctly to the fresh stance -- a
    # confirmed root cause of the post-reset "hop"/huge-qacc bug (stale
    # target vs freshly-teleported pose -> huge PD error -> huge torque),
    # and plausibly also of the mc-rtc-side "ZMP cannot be computed"
    # spam (mc-rtc's own internal state mid-transition when asked for its
    # first post-reset solve).
    #
    # _reset_generation increments every time reset() is called for an
    # env. _dispatch_generation records, per env, which generation was
    # current at the moment its most recent dispatch was SENT. At collect
    # time, only envs whose dispatch generation still matches their
    # CURRENT reset generation get their result staged/applied; anything
    # older is a stale, pre-reset result and is discarded instead.
    self._reset_generation = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._dispatch_generation = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # --- Continuity bookkeeping. ---
    # Physical params actually pushed to the controller last period vs this
    # period, kept as named tensors (not just the raw 4-vector) so both the
    # "last_action" observation and the continuity reward term can read
    # them without recomputing _map_to_physical themselves. `_period_t0`
    # additionally records *when* (in seconds, per-env) the current period's
    # params took effect -- needed to evaluate each sine at the exact
    # splice instant for the continuity penalty (see continuity_penalty()).
    zeros = torch.zeros(self.num_envs, device=self.device)
    self._physical_prev = {
      "offset": zeros.clone(),
      "frequency": zeros.clone(),
      "sin_amp": zeros.clone(),
      "cos_amp": zeros.clone(),
    }
    self._physical_curr = {k: v.clone() for k, v in self._physical_prev.items()}
    self._period_t0 = zeros.clone()

    # --- Walk/stop gate. ---
    # The policy's current walk decision (thresholded bool, see
    # _map_walk_gate) and ISMPC's own advisory safety opinion from the most
    # recent MPC solve, read back after each dispatch. Both kept as
    # per-env tensors so the last_walk_action/ismpc_wants_stop observation
    # terms (mdp.py) can read them without recomputing anything. Defaults
    # to False (not walking) -- matches Walking_controller::reset()'s own
    # policyWantsWalk default, so a freshly-constructed action and a
    # freshly-reset controller start in agreement.
    self._walk_enabled = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # --- Step timing (Ts). ---
    # Same "current value, updated only on the slow sine cadence, written
    # every dispatch tick regardless" pattern as _physical_curr above --
    # kept as its own tensor (not folded into _physical_curr/_physical_prev)
    # since Ts isn't part of the CoM-height sine and has no continuity-
    # penalty use, only a plain "current commanded value" need. Initialized
    # to TS_DEFAULT so the very first dispatch tick (before any
    # due_for_sine_update batch has run) writes the same value
    # Walking_controller::reset() itself defaults to, rather than 0.0 --
    # a step timing of 0 would be nonsensical even transiently.
    self._ts_curr = torch.full(
      (self.num_envs,), TS_DEFAULT, device=self.device
    )
    self._ismpc_wants_stop = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Ground truth (Walking_controller::Robot_Walking) -- distinct from both
    # _ismpc_wants_stop (ISMPC's advisory opinion) and _walk_enabled (the
    # policy's own commanded intent): neither of those alone reflects what
    # actually happened. Same default/reset rationale as _ismpc_wants_stop
    # above -- False until the first post-reset solve has actually run.
    self._is_walking = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  # ---- Required ActionTerm properties/methods. ----

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    """Store this period's raw policy output. Called once per policy step,
    before apply_actions runs (possibly several times, once per physics
    substep within the period)."""
    self._raw_actions[:] = actions
    self._processed_actions[:] = actions

  # ---- Raw action -> physical units. ----

  def _map_to_physical(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
      """First 4 of the 5-wide raw action -> {offset, frequency, alpha,
      beta}, all structurally within their safe/valid ranges. The 5th
      (walk-gate) dimension is handled separately by _map_walk_gate."""
      raw_offset, raw_frequency, raw_sin_amp, raw_cos_amp = raw[..., :4].unbind(dim=-1)

      offset = torch.clamp(
          self.cfg.offset_scale * raw_offset + self.cfg.offset_bias,
          min=OFFSET_MIN,
          max=OFFSET_MAX,
      )

      frequency = torch.clamp(
          torch.exp(self.cfg.frequency_scale * raw_frequency + self.cfg.frequency_bias),
          min=FREQUENCY_MIN,
          max=FREQUENCY_MAX,
      )

      sin_amp_raw = raw_sin_amp * self.cfg.amplitude_scale
      cos_amp_raw = raw_cos_amp * self.cfg.amplitude_scale
      radius = torch.sqrt(sin_amp_raw * sin_amp_raw + cos_amp_raw * cos_amp_raw)
      amplitude_ratio = offset / torch.maximum(radius, offset)
      sin_amp = sin_amp_raw * amplitude_ratio
      cos_amp = cos_amp_raw * amplitude_ratio

      return {
          "offset": offset,
          "frequency": frequency,
          "sin_amp": sin_amp,
          "cos_amp": cos_amp,
      }

  def _map_walk_gate(self, raw: torch.Tensor) -> torch.Tensor:
    """5th raw action -> bool walk/stop decision.

    Plain threshold at 0 on the raw (pre-squash) action: unlike the sine
    params, there's no "safe near-zero default" concern here the way
    AMPLITUDE_RATIO_ZERO_BIAS addresses for amplitude -- Walking_controller
    already starts every episode with policyWantsWalk=False regardless of
    what this action does on step 1 (see Walking_controller::reset()), and
    the policy has full authority either way, so there is no failure mode
    a bias here would protect against. raw > 0 -> walk; raw <= 0 -> stop,
    matching a standard zero-centered Gaussian policy's natural symmetry
    (no reason to bias the initial exploration toward either side).
    """
    return raw > self.cfg.walk_gate_bias

  def _map_step_timing(self, raw: torch.Tensor) -> torch.Tensor:
    """6th raw action -> physical Ts (s), linearly scaled and clamped.

    Same shape as the offset mapping above: ts = clip(ts_scale * raw +
    ts_bias, TS_MIN, TS_MAX). See IsmpcSineActionCfg.ts_scale/ts_bias's
    docstring for how the constants were chosen (centered on
    TS_DEFAULT=1.1, extreme values rare at raw~O(1)). The controller
    applies its own independent ts_range clamp on top of this one (see
    Walking_controller::ts(double)) -- this clamp only shapes exploration,
    it is not the safety mechanism.
    """
    return torch.clamp(
      self.cfg.ts_scale * raw + self.cfg.ts_bias,
      min=TS_MIN,
      max=TS_MAX,
    )

  # ---- Accessors for observation/reward terms. ----

  @property
  def physical_params(self) -> torch.Tensor:
    """Current period's physical params, stacked (num_envs, 4) in
    [offset, amplitude_ratio, frequency, phase] order. Intended for use as
    a `last_action`-style observation -- unlike `raw_action`, this is in
    physical units, which is more directly meaningful for the policy to
    condition on (e.g. "what CoM height am I currently riding on" rather
    than an arbitrary pre-sigmoid/pre-clamp number)."""
    return torch.stack(
      [
        self._physical_curr["offset"],
        self._physical_curr["frequency"],
        self._physical_curr["sin_amp"],
        self._physical_curr["cos_amp"],
      ],
      dim=-1,
    )

  @property
  def last_walk_action(self) -> torch.Tensor:
    """The policy's current walk/stop decision, as a (num_envs, 1) float
    observation (1.0 = walk, 0.0 = stop) -- mirrors physical_params'
    role for the sine params: lets the policy condition on its own last
    decision directly, same rationale as last_sine_params in mdp.py."""
    return self._walk_enabled.to(dtype=torch.get_default_dtype()).unsqueeze(-1)

  @property
  def last_step_timing_action(self) -> torch.Tensor:
    """The policy's current step-timing (Ts) command, as a (num_envs, 1)
    float observation (s) -- mirrors physical_params'/last_walk_action's
    role: lets the policy condition on its own last decision directly."""
    return self._ts_curr.unsqueeze(-1)

  @property
  def ismpc_wants_stop_obs(self) -> torch.Tensor:
    """ISMPC's own advisory safety opinion from the most recent MPC solve,
    as a (num_envs, 1) float observation (1.0 = ISMPC would have stopped).
    Independent of what the policy actually commanded -- see
    ControllerIoBinding.read_ismpc_wants_stop for the full rationale."""
    return self._ismpc_wants_stop.to(dtype=torch.get_default_dtype()).unsqueeze(-1)

  @property
  def is_walking_obs(self) -> torch.Tensor:
    """The controller's ACTUAL current walking state
    (Walking_controller::Robot_Walking), as a (num_envs, 1) float
    observation (1.0 = walking). Ground truth, distinct from both
    ismpc_wants_stop_obs (ISMPC's advisory opinion) and last_walk_action
    (the policy's own commanded intent) -- see
    ControllerIoBinding.read_is_walking for the full rationale. Intended
    for reward terms (e.g. mdp.is_walking) that need to know what actually
    happened, not what was requested or advised."""
    return self._is_walking.to(dtype=torch.get_default_dtype()).unsqueeze(-1)

  # ---- ActionTerm API. ----

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)

    env_indices = (
      list(range(self.num_envs))[env_ids]
      if isinstance(env_ids, slice) or env_ids is None
      else env_ids.tolist()
    )
    if env_ids is None:
      env_indices = list(range(self.num_envs))

    # --- DEBUG: deliberately NOT clearing _debug_pending_dispatch_ids
    # here. If a reset happens while env 0 still has an in-flight
    # dispatch pending collection, that is itself diagnostic information
    # (e.g. a worker respawn discarding a dispatch mid-flight rather than
    # waiting for/cancelling it cleanly) -- clearing the FIFO on reset
    # would hide exactly the race this debug pass is trying to catch.
    # if 0 in env_indices and self._debug_pending_dispatch_ids[0]:
      # print(
      #   f"[DEBUG reset] env=0 reset() called with "
      #   f"{len(self._debug_pending_dispatch_ids[0])} dispatch(es) still "
      #   f"pending collection: {self._debug_pending_dispatch_ids[0]}",
      #   flush=True,
      # )
    # --- END DEBUG ---
    self._io.reset_controller_input(self._in_np)
    self._pool.reset_envs(env_indices)
    self._steps_since_run[env_indices] = 0
    for _i in env_indices:
      self._steps_since_run_cpu[_i] = 0
    self._dispatch_ticks_since_sine_update[env_indices] = 1

    env_indices_t = torch.tensor(env_indices, device=self.device, dtype=torch.long)

    # Bump these envs' reset generation FIRST, before anything else below.
    # Any dispatch still in flight for these envs was tagged with the OLD
    # generation at send time (see the dispatch-time write near the bottom
    # of apply_actions()), so it will now read as stale at collect time and
    # get discarded there instead of clobbering the fresh state we're about
    # to write. See the constructor's comment on _reset_generation for the
    # full rationale.
    self._reset_generation[env_indices_t] += 1

    stance = self._entity.data.joint_pos[:, self._target_ids]
    self._previous_control["q"][env_indices_t] = stance[env_indices_t]
    self._next_control["q"][env_indices_t] = stance[env_indices_t]
    self._previous_control["alpha"][env_indices_t] = 0.0
    self._next_control["alpha"][env_indices_t] = 0.0
    # Pass env_indices_t with shape [N, 1] using unsqueeze(-1) or [:, None]
    self._entity.set_joint_position_target(
        stance[env_indices_t],
        joint_ids=self._target_ids,
        env_ids=env_indices_t.unsqueeze(-1),
    )

    self._entity.set_joint_velocity_target(
        torch.zeros_like(stance[env_indices_t]),
        joint_ids=self._target_ids,
        env_ids=env_indices_t.unsqueeze(-1),
    )
    self._has_staged_control[env_indices_t] = False
    self.controller_failed[env_indices_t] = False
    self._out_np[env_indices, self._io.layout.status_off] = 0.0

    # Fresh episode: no meaningful "previous period" to compare continuity
    # against yet. Seed both prev and curr to the same flat (zero-
    # amplitude) reference at the new stance offset, so the first real
    # period's continuity penalty compares against a sane baseline rather
    # than leftover values from whichever env used to occupy this slot.
    for k in self._physical_prev:
      self._physical_prev[k][env_indices_t] = 0.0
      self._physical_curr[k][env_indices_t] = 0.0
    self._physical_prev["offset"][env_indices_t] = self.cfg.offset_bias
    self._physical_curr["offset"][env_indices_t] = self.cfg.offset_bias
    self._period_t0[env_indices_t] = 0.0

    # Matches Walking_controller::reset()'s own T_Steps default
    # (kDefaultTSteps, see the C++ side) -- every episode is independent,
    # so the Python-side mirror must snap back to the same default the
    # controller itself just reset to, rather than riding on whatever the
    # previous episode's policy (or, in manual mode, a GUI edit) last left
    # it at until the next slow-cadence sine_update_indices_t batch
    # overwrites it (which, per _dispatch_ticks_since_sine_update's reset
    # to 1 just above, is not necessarily the very next dispatch tick).
    self._ts_curr[env_indices_t] = TS_DEFAULT

    # Matches Walking_controller::reset()'s own policyWantsWalk default
    # (False) -- see the constructor's comment on _walk_enabled. ISMPC has
    # no opinion yet either (no solve has happened this episode), and
    # Robot_Walking genuinely is false at this point (Walking_controller::
    # reset() explicitly sets it false -- see this project's reset
    # investigation), so this matches ground truth, not just a placeholder.
    self._walk_enabled[env_indices_t] = False
    self._ismpc_wants_stop[env_indices_t] = False
    self._is_walking[env_indices_t] = False

  def apply_actions(self) -> None:
    substep_in_period = self._steps_since_run % self.cfg.frameskip
    run_indices = [i for i, s in enumerate(self._steps_since_run_cpu) if s % self.cfg.frameskip == 0]

    if isinstance(run_indices, int):
      run_indices = [run_indices]

    if run_indices:
      env_indices = self._pool.collect()
      if env_indices is not None:
        env_indices_t = torch.tensor(env_indices, device=self.device, dtype=torch.long)

        # --- Reset/dispatch race guard (see constructor comment on
        # _reset_generation). A collected result is only valid if it was
        # dispatched under the env's CURRENT reset generation -- i.e. no
        # reset() has happened for that env since the dispatch was sent.
        # Stale results (dispatched pre-reset, collected post-reset) are
        # dropped here rather than staged: applying them would clobber the
        # fresh joint_pos_target/joint_vel_target reset() just wrote with a
        # q/alpha output computed from the OLD episode's state.
        stale = self._dispatch_generation[env_indices_t] != self._reset_generation[env_indices_t]
        if bool(stale.any()):
          stale_env_indices = env_indices_t[stale].tolist()
          print(
            f"[reset_race_guard] discarding {len(stale_env_indices)} stale "
            f"collected result(s) for envs {stale_env_indices} "
            "(dispatched before their most recent reset)",
            flush=True,
          )
          breakpoint_variable = 0
        fresh_mask = ~stale
        env_indices_t = env_indices_t[fresh_mask]
        env_indices = env_indices_t.tolist()
        # --- END reset/dispatch race guard ---

        if env_indices:
          new_output = self._io.read_controller_output(self._out_np, env_indices)
          for c in ("q", "alpha"):
            self._staged_control[c][env_indices_t] = new_output[c]
          self._has_staged_control[env_indices_t] = True

          newly_failed = self._io.read_controller_failed(self._out_np, env_indices)
          # Read alongside controller_failed: same collect() cycle, same
          # "this reflects the dispatch that just completed" timing. Overwrite
          # (not OR-accumulate like controller_failed) -- this is ISMPC's
          # opinion as of the MOST RECENT solve, not a latched "ever true"
          # flag, matching Walking_controller::ismpc_wants_stop's own
          # per-solve-cleared semantics.
          self._ismpc_wants_stop[env_indices_t] = self._io.read_ismpc_wants_stop(
            self._out_np, env_indices
          )
          # Same collect() cycle, same "reflects the solve that just
          # completed" timing as ismpc_wants_stop above -- ground truth
          # (Robot_Walking), not an opinion or a command.
          self._is_walking[env_indices_t] = self._io.read_is_walking(
            self._out_np, env_indices
          )

          self.controller_failed[env_indices_t] |= newly_failed

      run_indices_t = torch.tensor(run_indices, device=self.device, dtype=torch.long)
      fresh = self._has_staged_control[run_indices_t]
      if bool(fresh.any()):
        fresh_indices_t = run_indices_t[fresh]
        for c in ("q", "alpha"):
          self._previous_control[c][fresh_indices_t] = self._next_control[c][
            fresh_indices_t
          ]
          self._next_control[c][fresh_indices_t] = self._staged_control[c][
            fresh_indices_t
          ]
        self._has_staged_control[fresh_indices_t] = False

      # The policy's action for THIS period is what gets pushed to the
      # controller for the step about to be dispatched -- but only
      # *recomputed* on the slower sine-param cadence (sine_param_frequency_hz),
      # not every dispatch tick. Envs not due for an update this tick simply
      # keep riding on self._physical_curr's existing values (already
      # written into self._in_np last time they were computed) -- the
      # write below still happens every dispatch tick for every run_index,
      # since ControllerIoBinding.write_ismpc_sine_params's signature only
      # takes the full per-env arrays (no partial-env write verified to
      # exist), but the *values* themselves only change on the slow clock.
      due_for_sine_update = (
        self._dispatch_ticks_since_sine_update[run_indices_t] == 0
      )
      sine_update_indices_t = run_indices_t[due_for_sine_update]

      if sine_update_indices_t.numel() > 0:
        physical = self._map_to_physical(self._processed_actions)

        # Continuity bookkeeping: only meaningful for envs actually
        # starting a new sine period this call -- shift curr -> prev,
        # record the new curr and the sim time it took effect.
        now = self._env.episode_length_buf[sine_update_indices_t].to(
          dtype=torch.get_default_dtype()
        ) * self._env.step_dt
        for k in self._physical_prev:
          self._physical_prev[k][sine_update_indices_t] = self._physical_curr[k][
            sine_update_indices_t
          ]
          self._physical_curr[k][sine_update_indices_t] = physical[k][
            sine_update_indices_t
          ]
        self._period_t0[sine_update_indices_t] = now

        # Walk-gate decision updates on the SAME cadence as the sine params
        # (sine_param_frequency_hz) -- "maintain the same inference
        # frequency for our NN" (same rationale documented for the
        # reference-velocity channel). Uses the 5th raw-action dim,
        # unaffected by _map_to_physical only reading raw[..., :4].
        self._walk_enabled[sine_update_indices_t] = self._map_walk_gate(
          self._processed_actions[sine_update_indices_t, 4]
        )

        # Step timing (Ts) updates on the SAME cadence as the sine params/
        # walk gate above, same rationale (consistent NN inference
        # frequency across all of this action's channels). Uses the 6th
        # (last) raw-action dim.
        self._ts_curr[sine_update_indices_t] = self._map_step_timing(
          self._processed_actions[sine_update_indices_t, 5]
        )

      self._dispatch_ticks_since_sine_update[run_indices_t] = (
        self._dispatch_ticks_since_sine_update[run_indices_t] + 1
      ) % self._sine_param_period_ticks

      # Written every dispatch tick (all run_indices_t), using
      # self._physical_curr's current values -- unchanged since the last
      # sine-param update for envs not due this tick, freshly updated above
      # for envs that were due. This keeps the write call's signature
      # exactly as it was (full per-env arrays), rather than assuming a
      # partial-env write path exists.
      self._io.write_ismpc_sine_params(
        self._in_np,
        self._physical_curr["offset"],
        self._physical_curr["frequency"],
        self._physical_curr["sin_amp"],
        self._physical_curr["cos_amp"],
      )

      # Same "write current value every dispatch tick, update only on the
      # slow clock" convention as the sine params above.
      self._io.write_ismpc_walk_gate(self._in_np, self._walk_enabled)

      # Same "write current value every dispatch tick, update only on the
      # slow clock" convention as the sine params/walk gate above.
      self._io.write_ismpc_ts(self._in_np, self._ts_curr)

      # Push the per-env sampled twist command into the shared input row so
      # ismpc_walking's reference velocity actually varies per-env (see the
      # env cfg's "twist" UniformVelocityCommandCfg) instead of every
      # worker following the static auto_start.speed in mc_rtc.yaml.
      # Written every dispatch tick alongside the sine params, same
      # all-run_indices_t pattern -- get_command returns the currently
      # sampled value for every env regardless of whether it was just
      # resampled this tick, so writing it every time is correct (matches
      # write_ismpc_sine_params's own "write current value every tick"
      # convention above, not just on change).
      twist = self._env.command_manager.get_command("twist")
      self._io.write_ismpc_velocity(
        self._in_np, twist[:, 0], twist[:, 1], twist[:, 2]
      )

      # Reset/dispatch race guard (see constructor comment on
      # _reset_generation): stamp each dispatched env with the reset
      # generation current RIGHT NOW, at send time. If reset() bumps the
      # generation for this env before the result is collected, the
      # mismatch at collect time marks the result stale and it gets
      # discarded there instead of clobbering the fresh post-reset state.
      self._dispatch_generation[run_indices_t] = self._reset_generation[run_indices_t]

      self._io.fill_controller_input(self._in_np)
      self._pool.dispatch_controller_step(run_indices)

    interpolation_coef = (
      (substep_in_period + 1).float() / self.cfg.frameskip
    ).unsqueeze(-1)
    interpolated_control = {
      c: self._previous_control[c]
      + interpolation_coef * (self._next_control[c] - self._previous_control[c])
      for c in ("q", "alpha")
    }
    self._steps_since_run += 1
    for _i in range(self.num_envs):
        self._steps_since_run_cpu[_i] += 1
    # No residual: mc_rtc's own q/alpha drive the joints directly, unmodified.
    self._entity.set_joint_position_target(
      interpolated_control["q"], joint_ids=self._target_ids
    )
    self._entity.set_joint_velocity_target(
      interpolated_control["alpha"], joint_ids=self._target_ids
    )