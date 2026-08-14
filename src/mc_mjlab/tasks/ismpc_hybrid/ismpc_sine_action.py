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
OFFSET_MIN, OFFSET_MAX = 0.4, 1.0  # m
FREQUENCY_MIN, FREQUENCY_MAX = 0.1, 8.0  # Hz

# Shifts the amplitude-ratio sigmoid so raw_action=0 -> amplitude_ratio~=0
# (flat reference), not sigmoid(0)=0.5 (a large, constant-amplitude
# oscillation from the very first step). This was the actual cause of the
# "Control Horizon reached" / NaN-in-solver crash seen with --agent zero and
# again at the start of training: sigmoid(0)=0.5 combined with the default
# frequency_bias=4.05 Hz produced ~0.35m swings at ~4Hz immediately, which
# is numerically violent enough to break ISMPC's feasibility QP on the
# first few control periods, before the policy has had any chance to learn
# otherwise. sigmoid(-4.0) ~= 0.018, close enough to zero to be safe as a
# default while still leaving the policy free to push it up when useful.
AMPLITUDE_RATIO_ZERO_BIAS = -4.0


@dataclass(kw_only=True)
class IsmpcSineActionCfg(ActionTermCfg):
  """Configuration for the learned ISMPC CoM-height sine parameter action.

  Subclasses ActionTermCfg directly, NOT BaseActionCfg: this action controls
  no per-joint actuators at all (its action space is a fixed 4 -- offset,
  amplitude_ratio, frequency, phase -- unrelated to joint count), so
  BaseActionCfg's actuator_names/scale/offset machinery (sized off matched
  joints) does not apply here. Joint actuation is still driven by mc_rtc's
  own q/alpha output, same as the residual actions, but that's wiring
  internal to this action, not something ActionTermCfg needs to know about.
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

  # Set to explore full OFFSET_MIN OFFSET_MAX range in natural range from 
  # arithmetic mean
  offset_scale: float = (OFFSET_MAX-OFFSET_MIN)/2
  offset_bias: float = (OFFSET_MAX+OFFSET_MIN)/2
  # Set to explore full FREQUENCY_MIN FREQUENCY_MAX in log range from 
  # geometric mean 
  frequency_scale: float = torch.log(torch.tensor(FREQUENCY_MAX / FREQUENCY_MIN)).item() / 2
  frequency_bias: float = torch.log(torch.tensor(FREQUENCY_MIN * FREQUENCY_MAX)).item() / 2

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

    self._raw_actions = torch.zeros(self.num_envs, 5, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    # --- DEBUG: monotonic per-env dispatch counter + a record of what was
    # sent on each dispatch, so a failure observed later at collect() time
    # can be matched back to the exact params that caused it, even across
    # the async dispatch/collect gap and even if a reset happens to this
    # env in between. Remove once the root cause is found.
    self._debug_dispatch_id = [0] * self.num_envs
    self._debug_dispatch_log: dict[int, dict] = {}
    self._debug_pending_dispatch_ids: list[list[int]] = [[] for _ in range(self.num_envs)]
    # --- END DEBUG ---

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
      "amplitude_ratio": zeros.clone(),
      "amplitude": zeros.clone(),
      "frequency": zeros.clone(),
      "phase": zeros.clone(),
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
    self._ismpc_wants_stop = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  # ---- Required ActionTerm properties/methods. ----

  @property
  def action_dim(self) -> int:
    return 5

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
    """First 4 of the 5-wide raw action -> {offset, amplitude, frequency,
    phase}, all structurally within their safe/valid ranges. The 5th
    (walk-gate) dimension is handled separately by _map_walk_gate."""
    raw_offset, raw_ratio, raw_freq, raw_phase = raw[..., :4].unbind(dim=-1)

    offset = torch.clamp(
      self.cfg.offset_scale * raw_offset + self.cfg.offset_bias,
      min=OFFSET_MIN,
      max=OFFSET_MAX,
    )
    # Sigmoid, shifted by AMPLITUDE_RATIO_ZERO_BIAS so raw_ratio=0 maps to
    # amplitude_ratio near 0 (flat), not sigmoid(0)=0.5 (a large constant
    # oscillation) -- see the module-level comment on
    # AMPLITUDE_RATIO_ZERO_BIAS for why this matters. The sigmoid itself
    # (rather than a raw clamp) still keeps the policy's own action
    # distribution unconstrained (better-behaved for PPO's Gaussian) while
    # guaranteeing amplitude_ratio in (0, 1) *by construction* -- combined
    # with deriving amplitude = ratio * offset (not learned directly), this
    # is what makes "the CoM height trajectory can never go negative" a
    # structural property rather than a clamp bolted on after the fact (see
    # the ismpc_solver_patch.md notes on this).
    amplitude_ratio = torch.sigmoid(raw_ratio + AMPLITUDE_RATIO_ZERO_BIAS)
    amplitude = amplitude_ratio * offset

    frequency = torch.clamp(
      torch.exp(self.cfg.frequency_scale * raw_freq + self.cfg.frequency_bias),
      min=FREQUENCY_MIN,
      max=FREQUENCY_MAX,
    )

    # Wrap to [-pi, pi] via atan2(sin, cos) rather than a raw clamp/modulo:
    # smooth and well-defined everywhere, no discontinuous jump at the
    # wrap boundary the way a naive `% (2*pi)` would produce.
    phase = torch.atan2(torch.sin(raw_phase), torch.cos(raw_phase))

    return {
      "offset": offset,
      "amplitude_ratio": amplitude_ratio,
      "amplitude": amplitude,
      "frequency": frequency,
      "phase": phase,
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
    return raw > 0.0

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
        self._physical_curr["amplitude_ratio"],
        self._physical_curr["frequency"],
        self._physical_curr["phase"],
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
  def ismpc_wants_stop_obs(self) -> torch.Tensor:
    """ISMPC's own advisory safety opinion from the most recent MPC solve,
    as a (num_envs, 1) float observation (1.0 = ISMPC would have stopped).
    Independent of what the policy actually commanded -- see
    ControllerIoBinding.read_ismpc_wants_stop for the full rationale."""
    return self._ismpc_wants_stop.to(dtype=torch.get_default_dtype()).unsqueeze(-1)

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
    self._dispatch_ticks_since_sine_update[env_indices] = 0

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
    self._entity.set_joint_position_target(
      stance[env_indices_t], joint_ids=self._target_ids, env_ids=env_indices_t
    )
    self._entity.set_joint_velocity_target(
      torch.zeros_like(stance[env_indices_t]),
      joint_ids=self._target_ids,
      env_ids=env_indices_t,
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
    self._physical_prev["offset"][env_indices_t] = OFFSET_MIN
    self._physical_curr["offset"][env_indices_t] = OFFSET_MIN
    self._period_t0[env_indices_t] = 0.0

    # Matches Walking_controller::reset()'s own policyWantsWalk default
    # (False) -- see the constructor's comment on _walk_enabled. ISMPC has
    # no opinion yet either (no solve has happened this episode).
    self._walk_enabled[env_indices_t] = False
    self._ismpc_wants_stop[env_indices_t] = False

  def apply_actions(self) -> None:
    substep_in_period = self._steps_since_run % self.cfg.frameskip
    run_envs = substep_in_period == 0
    run_indices = run_envs.nonzero(as_tuple=False).squeeze(-1).tolist()
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

          # --- DEBUG: pop the oldest pending dispatch id for env 0 (FIFO --
          # collect() results arrive in the same order they were dispatched,
          # assuming the pool preserves per-env ordering, which every other
          # part of this action already assumes via _steps_since_run/
          # _has_staged_control's sequencing) and print its recorded params
          # if this collect reports a failure. This is the corrected version
          # of the earlier debug attempt: that one read self._physical_curr
          # AFTER collect(), which a reset() interleaved between the failing
          # dispatch and this collect() could have already zeroed out. This
          # version logs at dispatch time instead, immune to that. Remove
          # once the root cause is found.
          if 0 in env_indices:
            local_idx = env_indices.index(0)
            pending = self._debug_pending_dispatch_ids[0]
            popped_id = pending.pop(0) if pending else None
            if bool(newly_failed[local_idx]):
              record = (
                self._debug_dispatch_log.get(popped_id)
                if popped_id is not None
                else None
              )
              print(
                f"[DEBUG controller_failed] env=0 CONFIRMED failure for "
                f"dispatch_id={popped_id} record={record}",
                flush=True,
              )
          # --- END DEBUG ---

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
        self._physical_curr["amplitude_ratio"],
        self._physical_curr["frequency"],
        self._physical_curr["phase"],
      )

      # Same "write current value every dispatch tick, update only on the
      # slow clock" convention as the sine params above.
      self._io.write_ismpc_walk_gate(self._in_np, self._walk_enabled)


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

      # --- DEBUG: record what's actually being dispatched for env 0 this
      # tick, tagged with a monotonic id, BEFORE dispatch -- uncontaminated
      # by any reset that might happen to this env before we later collect
      # the result. Remove once the root cause is found.
      if 0 in run_indices:
        did = self._debug_dispatch_id[0]
        self._debug_dispatch_id[0] += 1
        t_now = float(self._env.episode_length_buf[0]) * float(self._env.step_dt)
        record = {
          "sim_t": t_now,
          "offset": float(self._physical_curr["offset"][0]),
          "amplitude_ratio": float(self._physical_curr["amplitude_ratio"][0]),
          "amplitude": float(self._physical_curr["amplitude"][0]),
          "frequency": float(self._physical_curr["frequency"][0]),
          "phase": float(self._physical_curr["phase"][0]),
          "period_t0": float(self._period_t0[0]),
          "steps_since_run": int(self._steps_since_run[0]),
          "dispatch_ticks_since_sine_update": int(
            self._dispatch_ticks_since_sine_update[0]
          ),
        }
        self._debug_dispatch_log[did] = record
        self._debug_pending_dispatch_ids[0].append(did)
        # Bound memory: keep only the last 200 dispatch records.
        if len(self._debug_dispatch_log) > 200:
          oldest = min(self._debug_dispatch_log)
          del self._debug_dispatch_log[oldest]
        # print(f"[DEBUG dispatch] env=0 dispatch_id={did} {record}", flush=True) 
      # --- END DEBUG ---

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

    # No residual: mc_rtc's own q/alpha drive the joints directly, unmodified.
    self._entity.set_joint_position_target(
      interpolated_control["q"], joint_ids=self._target_ids
    )
    self._entity.set_joint_velocity_target(
      interpolated_control["alpha"], joint_ids=self._target_ids
    )