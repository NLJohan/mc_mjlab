"""Learned CoM-height sine parameter action for ISMPC.

This is the native-`mc_rtc_interface` counterpart of the old Cython-bridge
`ismpc_sine_action.py`. The RL policy's 9 raw actions are mapped into
physical ISMPC parameters each slow sine-param tick:

  offset            = clip(offset_scale * raw[0] + offset_bias, OFFSET_MIN, OFFSET_MAX)   (m)
  frequency         = clip(exp(frequency_scale * raw[1] + frequency_bias), FREQUENCY_MIN, FREQUENCY_MAX)  (Hz)
  sin_amp, cos_amp  = raw[2], raw[3] scaled by amplitude_scale, then radius-clamped
                       against offset so offset - amplitude >= 0 structurally
  walk_gate         = raw[4] > walk_gate_bias                               (bool)
  ts                = clip(ts_scale * raw[5] + ts_bias, TS_MIN, TS_MAX)     (s)
  twist (vx,vy,w)   = rate_limit(clip(raw[6:9], -1, 1) * twist_scale)       (m/s, m/s, rad/s)
                       -- see _map_twist/twist_max_delta for the per-axis rate limit

The twist (vx, vy, omega) drives ismpc_walking's footstep reference velocity
via a new datastore setter (SET_RL_REF_VEL), ONLY when the controller-side
rlVelocityControl toggle is on (YAML `walking_controller.rl_velocity_control`
or the "RL Reference Velocity" GUI checkbox in Walking_controller -- this
toggle is NOT exposed to or writable from Python; see
Walking_controller.h/.cpp). When that toggle is off, this class's writes to
SET_RL_REF_VEL are harmless but inert: the controller's own mux
(Walking_controller::updateReferenceVelocity()) ignores rl_reference_velocity
and follows user_reference_velocity (GUI/joystick) instead. The controller
exposes its estimated CoM velocity (base axes) via ismpc_walking::get_com_lin_vel,
read as a datastore_vectors_outputs entry -- see com_lin_vel_est_obs -- which is
the policy's "com_lin_vel" observation.

Joint actuation itself is unchanged from the old file: mc_rtc's own q/alpha
output drives the joints directly. There is no joint-space residual in this
task -- RL only ever touches the ISMPC parameters, never joint commands.

Subclasses `McRtcActionBase` (not `McRtcResidualActionBase`): this action has
no residual concept whatsoever -- its action space is a fixed 9, unrelated to
joint count, and mc_rtc_residual_action.py is neither imported from nor
modified by this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mc_mjlab.actions.mc_rtc_action_base import McRtcActionBase, McRtcActionCfg

# --- Physical ranges, ported verbatim from the Cython-bridge action. ---
OFFSET_MIN, OFFSET_MAX = 0.4, 1.05  # m

FREQUENCY_MIN, FREQUENCY_MAX = 0.1, 8.0  # Hz

# Step timing (Ts, seconds between footsteps). Matches
# Walking_controller::kDefaultTSteps (1.1) and the controller's own
# ts_range clamp (0.4-2.0, from mc_rtc.yaml's ismpc.ts_range) -- this range
# is a soft/exploration-shaping bound on top of that hard controller-side
# clamp, not a replacement for it.
TS_MIN, TS_MAX = 0.4, 2.0  # s
TS_DEFAULT = 1.1  # s -- must match Walking_controller::kDefaultTSteps

# --- Datastore keys (docs/coupling.md "Datastore callbacks": a controller's
# own getters/setters are named as constants in the task that uses them, not
# in mc_mjlab/bridge/controller_datastore.py). See migration guidelines §1.3. ---
SET_OFFSET = "ismpc_walking::set_com_height_sine_offset"
SET_FREQUENCY = "ismpc_walking::set_com_height_sine_frequency"
SET_SIN_AMP = "ismpc_walking::set_com_height_sine_sin_amp"
SET_COS_AMP = "ismpc_walking::set_com_height_sine_cos_amp"
SET_POLICY_WANTS_WALK = "ismpc_walking::set_policy_wants_walk"
SET_TS = "ismpc_walking::set_ts"
GET_WANTS_STOP = "ismpc_walking::wants_stop_d"
GET_ROBOT_WALKING = "ismpc_walking::robot_walking_d"
# Vector3d (datastore_vectors_inputs/outputs), unlike the scalars above.
# set_rl_ref_vel writes Walking_controller::rl_reference_velocity only --
# inert unless the controller-side rlVelocityControl toggle is on (see the
# module docstring). get_com_lin_vel reads Walking_controller::
# estimatedComLinVel() (realRobot CoM velocity rotated into base axes).
SET_RL_REF_VEL = "ismpc_walking::set_rl_ref_vel"
GET_COM_LIN_VEL = "ismpc_walking::get_com_lin_vel"


@dataclass(kw_only=True)
class IsmpcSineActionCfg(McRtcActionCfg):
  """Configuration for the learned ISMPC CoM-height sine parameter action.

  Subclasses `McRtcActionCfg`, not `McRtcResidualActionCfg`: no residual
  fields apply here -- this action controls no per-joint actuators via RL at
  all (its action space is a fixed 6, unrelated to joint count). Joint
  actuation is still driven by mc_rtc's own q/alpha output, same as the
  residual actions, but that's internal wiring, not something the cfg needs
  residual-specific fields for.

  `actuator_names` (inherited from `BaseActionCfg`) selects which of the
  entity's joints mc_rtc drives via q/alpha -- NOT the RL action's own
  targets (the 6 fixed sine/gate/timing params), matching the old Cython-
  bridge file's `target_actuator_names` field, now folded into the inherited
  field since this class routes through `BaseAction.__init__` (see
  `IsmpcSineAction.__init__`). `scale`/`offset`/`clip` (also inherited) are
  not used by this action -- leave them at `BaseActionCfg`'s defaults; the
  raw-to-physical mapping is entirely the explicit `_map_to_physical`/
  `_map_walk_gate`/`_map_step_timing` methods below, not an affine transform.
  """

  actuator_names: tuple[str, ...] | list[str] = (".*",)
  """Which of the entity's joints receive mc_rtc's q/alpha output."""

  required_controller: str = "ismpc_walking"
  """Matches `Enabled: ismpc_walking` in the target mc_rtc.yaml."""

  sine_param_frequency_hz: float = 20.0
  """How often (Hz) the RL-set sine/gate/timing params are pushed into the
  controller, matching ISMPC_Solver's own MPC solve period (m_delta=0.05s ->
  20Hz by default). The controller itself is still stepped every `frameskip`
  physics substeps regardless -- only the *write* of these params (and the
  continuity-penalty bookkeeping tied to it) happens on this slower cadence.
  Conflating the two starves the controller's own internal state if
  frameskip is set slow enough to match the MPC period."""

  offset_scale: float = 0.075
  offset_bias: float = 0.9

  frequency_scale: float = torch.log(torch.tensor(FREQUENCY_MAX / FREQUENCY_MIN)).item() / 2
  frequency_bias: float = torch.log(torch.tensor(FREQUENCY_MIN * FREQUENCY_MAX)).item() / 2

  amplitude_scale: float = 0.10
  """Scales raw_sin_amp/raw_cos_amp into physical sin_amp/cos_amp (m) before
  the offset-based radius clamp. Combined reachable oscillation radius at raw
  values of magnitude r (per-axis) is roughly amplitude_scale * r * sqrt(2)
  if both axes are excited equally."""

  walk_gate_bias: float = 0.0
  """Shifts the walk/stop decision threshold away from 0. Positive values
  make 'walk' more likely at policy init (raw actions near 0); negative
  values make 'stop' more likely. Leave at 0.0 for the original unbiased
  50/50 behavior."""

  ts_scale: float = 0.35
  ts_bias: float = TS_DEFAULT
  """ts = clip(ts_scale * raw_ts + ts_bias, TS_MIN, TS_MAX). ts_bias =
  TS_DEFAULT centers the mapping on the controller's own default/reset
  value, so raw_ts=0 (a freshly-initialized policy's typical early output)
  reproduces the fixed-Ts behavior exactly."""

  twist_scale: tuple[float, float, float] = (0.5, 0.1, 0.2)
  """Per-axis (vx, vy, omega) magnitude limit of the policy's reference
  twist: physical = clamp(raw, -1, 1) * twist_scale, in (m/s, m/s, rad/s).

  Set to bound the twist the policy can request to (vx ±0.5, vy ±0.1,
  omega ±0.2). vx and vy match the `twist` command ranges in
  ismpc_hybrid_env_cfg.py; keep them in sync if either changes. omega's
  command range is currently (0, 0) there, so its limit only bounds what
  the policy may request."""

  twist_max_delta: tuple[float, float, float] = (0.3, 0.1, 0.125)
  """Per-axis rate limit on the commanded twist, in physical units per slow
  tick (one tick = 1 / sine_param_frequency_hz s, 0.05 s at 20 Hz), so
  the twist can move by at most this much every 0.05 s.

  Too loose risks large QP breaks from abrupt reference-velocity changes
  (see Walking_controller::reset()'s comments); too tight slows the
  policy's response to a changing command. Unvalidated: tune against
  controller_failed telemetry."""

  def build(self, env) -> "IsmpcSineAction":
    return IsmpcSineAction(self, env)


class IsmpcSineAction(McRtcActionBase):
  """Maps a 9-dim raw RL action to ISMPC CoM-height sine + walk-gate + Ts +
  twist (reference velocity).

  Subclasses `McRtcActionBase` directly, not `McRtcResidualActionBase`: no
  residual authority/gating/projection/printer concept applies -- the action
  vector maps entirely to controller-side ISMPC parameters, and joint
  actuation is pure mc_rtc q/alpha tracking.
  """

  cfg: IsmpcSineActionCfg

  output_channels: tuple[str, ...] = ("q", "alpha")

  def __init__(self, cfg: IsmpcSineActionCfg, env) -> None:
    # Base's `_build_bridge` reads cfg.datastore_scalar_inputs/outputs, so
    # the six datastore keys are declared here before `McRtcActionBase.__init__`
    # runs -- the generic `_setup_datastore_inputs`/`_setup_datastore_outputs`
    # machinery then handles them exactly like any other declared column,
    # no special-casing needed anywhere in the base.
    cfg.datastore_scalar_inputs = tuple(
      dict.fromkeys(
        (*cfg.datastore_scalar_inputs, SET_OFFSET, SET_FREQUENCY, SET_SIN_AMP,
         SET_COS_AMP, SET_POLICY_WANTS_WALK, SET_TS)
      )
    )
    cfg.datastore_scalar_outputs = tuple(
      dict.fromkeys((*cfg.datastore_scalar_outputs, GET_WANTS_STOP, GET_ROBOT_WALKING))
    )
    # Vector3d counterparts of the above, same declare-before-super() pattern
    # and same (num_envs, 3)-shaped transport (set_datastore_vector_input /
    # datastore_vector_output -- see mc_rtc_action_base.py).
    cfg.datastore_vectors_inputs = tuple(
      dict.fromkeys((*cfg.datastore_vectors_inputs, SET_RL_REF_VEL))
    )
    cfg.datastore_vectors_outputs = tuple(
      dict.fromkeys((*cfg.datastore_vectors_outputs, GET_COM_LIN_VEL))
    )

    # `BaseAction.__init__` (via McRtcActionBase's own super().__init__())
    # resolves `_entity`/`_target_ids`/`_target_names` from `cfg.actuator_names`
    # the same way it does for the residual actions -- no explicit resolution
    # needed here, since this class subclasses McRtcActionBase -> BaseAction,
    # unlike the old Cython-bridge file which subclassed ActionTerm directly.
    #
    # One mismatch this creates: BaseAction.__init__ also sizes _action_dim,
    # _raw_actions, _processed_actions, _scale, _offset, _clip off the
    # matched actuator count (len(target_ids)), since that's the only shape
    # BaseActionCfg's own fields (scale/offset/clip) know about. None of that
    # has any meaning for this action's fixed 9-dim space, so _action_dim/
    # _raw_actions/_processed_actions are unconditionally replaced right
    # below, and _scale/_offset/_clip are simply never read anywhere in this
    # class (cfg.scale/cfg.offset/cfg.clip are left at BaseActionCfg's inert
    # defaults of 1.0/0.0/None and should not be set by task configs using
    # this action -- there is no affine mapping here, only the explicit
    # _map_to_physical/_map_walk_gate/_map_step_timing/_map_twist methods
    # below).
    super().__init__(cfg, env)

    # Action dim 6 -> 9: +3 for the twist (vx, vy, omega). Old 6-dim
    # checkpoints will not load against this -- accepted, per-plan.
    self._action_dim = 9
    self._raw_actions = torch.zeros(self.num_envs, 9, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    # Sine-param update cadence, in units of controller-dispatch ticks (not
    # physics substeps). See cfg.sine_param_frequency_hz's docstring for why
    # this is kept separate from frameskip.
    # Real controller period = frameskip physics substeps = step_dt * frameskip
    # / decimation (the old step_dt / frameskip gave the physics rate, not the
    # controller rate, so the latch was frameskip-dependent and too slow).
    controller_dt = self._env.step_dt * cfg.frameskip / self._env.cfg.decimation
    controller_hz = 1.0 / controller_dt
    self._sine_param_period_ticks = max(
      1, round(controller_hz / cfg.sine_param_frequency_hz)
    )
    self._dispatch_ticks_since_sine_update = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    print(
      f"[IsmpcSineAction] controller_dt={controller_dt * 1e3:.3f} ms, "
      f"latch every {self._sine_param_period_ticks} controller steps "
      f"({self._sine_param_period_ticks * controller_dt * 1e3:.1f} ms)"
    )

    # --- Continuity bookkeeping. ---
    # Physical params actually pushed to the controller last period vs this
    # period, kept as named tensors so both the last_action observation and
    # the continuity reward term (mdp.py) can read them without recomputing
    # `_map_to_physical` themselves. `_period_t0` records when (in seconds,
    # per-env) the current period's params took effect.
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
    # Defaults to False (not walking) -- matches Walking_controller::reset()'s
    # own policyWantsWalk default.
    self._walk_enabled = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # --- Step timing (Ts). ---
    self._ts_curr = torch.full((self.num_envs,), TS_DEFAULT, device=self.device)

    # --- Twist (reference velocity: vx, vy, omega). ---
    # Defaults to zero -- matches Walking_controller::reset()'s own
    # rl_reference_velocity.setZero(). Rate-limited toward the mapped target
    # in _advance_sine_period via cfg.twist_max_delta.
    self._twist_curr = torch.zeros(self.num_envs, 3, device=self.device)

  # ---- Required ActionTerm properties/methods. ----

  @property
  def action_dim(self) -> int:
    return 9

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    """Store this period's raw policy output; no scale/offset/clip here --
    the seven physical mappings (sine x4, walk gate, Ts, twist) happen in
    apply_actions/the slow-cadence block, same split the old Cython-bridge
    file used."""
    self._raw_actions[:] = actions
    self._processed_actions[:] = actions

  # ---- Raw action -> physical units. ----

  def _map_to_physical(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """First 4 of the 6-wide raw action -> {offset, frequency, sin_amp,
    cos_amp}, all structurally within their safe/valid ranges."""
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
    # Cap the radius so both offset-radius and offset+radius stay within
    # [OFFSET_MIN, OFFSET_MAX], not just the trough >= 0 as before.
    max_radius = torch.minimum(offset - OFFSET_MIN, OFFSET_MAX - offset).clamp(min=0.0)
    amplitude_ratio = torch.where(
      radius > max_radius, max_radius / torch.clamp(radius, min=1e-8), torch.ones_like(radius)
    )
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

    Plain threshold at `walk_gate_bias` on the raw (pre-squash) action.
    """
    return raw > self.cfg.walk_gate_bias

  def _map_step_timing(self, raw: torch.Tensor) -> torch.Tensor:
    """6th raw action -> physical Ts (s), linearly scaled and clamped.

    The controller applies its own independent ts_range clamp on top of
    this one -- this clamp only shapes exploration, it is not the safety
    mechanism.
    """
    return torch.clamp(
      self.cfg.ts_scale * raw + self.cfg.ts_bias,
      min=TS_MIN,
      max=TS_MAX,
    )

  def _map_twist(self, raw: torch.Tensor) -> torch.Tensor:
    """Last 3 of the 9-wide raw action -> physical (vx, vy, omega), m/s and
    rad/s.

    Simple clamp-and-scale (not tanh): keeps raw=0 mapping exactly to
    physical=0, matching the convention used by _map_step_timing, so a
    freshly-initialized policy's near-zero output reproduces "no commanded
    motion" rather than some arbitrary offset. Rate limiting toward this
    target happens separately in _advance_sine_period (cfg.twist_max_delta)
    -- this method only computes the instantaneous target, not the
    rate-limited value actually written to the datastore.
    """
    scale = torch.tensor(self.cfg.twist_scale, device=raw.device, dtype=raw.dtype)
    return torch.clamp(raw, -1.0, 1.0) * scale

  # ---- Accessors for observation/reward terms (ismpc_mdp.py). ----

  @property
  def physical_params(self) -> torch.Tensor:
    """Current period's physical params, stacked (num_envs, 4) in
    [offset, frequency, sin_amp, cos_amp] order."""
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
    """The policy's current walk/stop decision, as a (num_envs, 1) float."""
    return self._walk_enabled.to(dtype=torch.get_default_dtype()).unsqueeze(-1)

  @property
  def last_step_timing_action(self) -> torch.Tensor:
    """The policy's current step-timing (Ts) command, (num_envs, 1), seconds."""
    return self._ts_curr.unsqueeze(-1)

  @property
  def last_twist_action(self) -> torch.Tensor:
    """The policy's current (vx, vy, omega) twist command, (num_envs, 3),
    m/s and rad/s -- the rate-limited value actually written to
    SET_RL_REF_VEL this period, not the raw per-tick target."""
    return self._twist_curr

  @property
  def com_lin_vel_est_obs(self) -> torch.Tensor:
    """Controller-estimated CoM linear velocity (vx, vy, vz), (num_envs, 3),
    m/s, in base axes, read via the GET_COM_LIN_VEL datastore output
    (Walking_controller::estimatedComLinVel). Collected every control period
    by McRtcActionBase's generic datastore-output mechanism."""
    return self.datastore_vector_output(GET_COM_LIN_VEL)

  @property
  def target_height_obs(self) -> torch.Tensor:
    """Live CoM-height reference (m), evaluated now, as a (num_envs, 1)
    tensor. Debugging/visualization only -- not the C++-side ground truth."""
    now = self._env.episode_length_buf.to(
      dtype=torch.get_default_dtype()
    ) * self._env.step_dt
    t_in_period = now
    omega = 2.0 * torch.pi * self._physical_curr["frequency"]
    phase = omega * t_in_period
    h = (
      self._physical_curr["offset"]
      + self._physical_curr["sin_amp"] * torch.sin(phase)
      + self._physical_curr["cos_amp"] * torch.cos(phase)
    )
    return h.unsqueeze(-1)

  @property
  def ismpc_wants_stop_obs(self) -> torch.Tensor:
    """ISMPC's own advisory safety opinion from the most recent MPC solve,
    (num_envs, 1) float (1.0 = ISMPC would have stopped). Read through the
    base's generic datastore-output machinery -- replaces the old file's
    manual ControllerIoBinding readback."""
    return self.datastore_scalar_output(GET_WANTS_STOP).unsqueeze(-1)

  @property
  def is_walking_obs(self) -> torch.Tensor:
    """The controller's actual current walking state
    (Walking_controller::Robot_Walking), (num_envs, 1) float (1.0 =
    walking). Ground truth, distinct from both ismpc_wants_stop_obs and
    last_walk_action."""
    return self.datastore_scalar_output(GET_ROBOT_WALKING).unsqueeze(-1)

  # ---- McRtcActionBase abstract methods. ----

  def _seed_interpolation(self, env_ids: torch.Tensor) -> None:
    """Seed q from current biased joint position, alpha from 0 -- identical
    to McRtcResidualJointPositionAction._seed_interpolation."""
    stance = self._entity.data.joint_pos_biased[:, self._target_ids]
    self._previous_control["q"][env_ids] = stance[env_ids]
    self._next_control["q"][env_ids] = stance[env_ids]
    self._previous_control["alpha"][env_ids] = 0.0
    self._next_control["alpha"][env_ids] = 0.0

  def _apply_control(self, interpolated_control: dict[str, torch.Tensor]) -> None:
    """Write q/alpha targets. No residual branch -- mc_rtc's own q/alpha
    output drives the joints directly and unmodified."""
    self._entity.set_joint_position_target(
      interpolated_control["q"], joint_ids=self._target_ids
    )
    self._entity.set_joint_velocity_target(
      interpolated_control["alpha"], joint_ids=self._target_ids
    )

  # ---- reset()/apply_actions() overrides. ----

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)

    if env_ids is None:
      env_indices = list(range(self.num_envs))
    elif isinstance(env_ids, slice):
      env_indices = list(range(self.num_envs))[env_ids]
    else:
      env_indices = env_ids.tolist()
    rows = torch.tensor(env_indices, device=self.device, dtype=torch.long)

    self._raw_actions[rows] = 0.0
    self._processed_actions[rows] = 0.0

    # Fresh episode: no meaningful "previous period" to compare continuity
    # against yet. Seed both prev and curr to the same flat (zero-amplitude)
    # reference at the reset stance offset, so the first real period's
    # continuity penalty compares against a sane baseline.
    for k in self._physical_prev:
      self._physical_prev[k][rows] = 0.0
      self._physical_curr[k][rows] = 0.0
    self._physical_prev["offset"][rows] = self.cfg.offset_bias
    self._physical_curr["offset"][rows] = self.cfg.offset_bias
    self._period_t0[rows] = 0.0

    # Matches Walking_controller::reset()'s own T_Steps/policyWantsWalk
    # defaults -- every episode is independent, so the Python-side mirror
    # must snap back to the same defaults the controller itself resets to.
    self._ts_curr[rows] = TS_DEFAULT
    self._walk_enabled[rows] = False
    # Same reasoning, extended to the twist: matches
    # Walking_controller::reset()'s rl_reference_velocity.setZero().
    self._twist_curr[rows] = 0.0

    # So the very next control period is treated as "due" for a sine
    # update, same as the old Cython-bridge file's own convention.
    self._dispatch_ticks_since_sine_update[rows] = 1

  def apply_actions(self) -> None:
    """Advance the control period on its first substep, then write
    interpolated q/alpha targets. Ported directly from
    McRtcResidualJointPositionAction-style structure, minus the residual."""
    substep_in_period = self._substep % self.cfg.frameskip
    self._substep += 1
    if substep_in_period == 0:
      self._advance_sine_period()

    interpolation_coef = (substep_in_period + 1) / self.cfg.frameskip
    interpolated_control = {}
    for channel in self.output_channels:
      torch.lerp(
        self._previous_control[channel],
        self._next_control[channel],
        interpolation_coef,
        out=self._interpolated[channel],
      )
      interpolated_control[channel] = self._interpolated[channel]

    torch.maximum(
      self._torque_peak,
      self._entity.data.qfrc_actuator[:, self._target_ids].abs(),
      out=self._torque_peak,
    )

    self._apply_control(interpolated_control)

  def _advance_sine_period(self) -> None:
    """This class's analogue of `_advance_control_period`: run the generic
    dispatch/collect first, then do the sine-specific slow-cadence work.

    Does NOT port the old file's manual `_reset_generation`/
    `_dispatch_generation` staleness guard -- `McRtcActionBase`'s
    `_pending_reset`/`_dispatch_resets` + WORKER_FAILED handling already
    solves the same problem generically.
    """
    super()._advance_control_period()

    due_for_sine_update = self._dispatch_ticks_since_sine_update == 0

    if bool(due_for_sine_update.any()):
      physical = self._map_to_physical(self._processed_actions)

      now = (
        self._env.episode_length_buf.to(dtype=torch.get_default_dtype())
        * self._env.step_dt
      )
      for k in self._physical_prev:
        self._physical_prev[k] = torch.where(
          due_for_sine_update, self._physical_curr[k], self._physical_prev[k]
        )
        self._physical_curr[k] = torch.where(
          due_for_sine_update, physical[k], self._physical_curr[k]
        )
      self._period_t0 = torch.where(due_for_sine_update, now, self._period_t0)

      # Walk-gate and step-timing update on the SAME cadence as the sine
      # params, for consistent NN inference frequency across every channel
      # of this action.
      self._walk_enabled = torch.where(
        due_for_sine_update,
        self._map_walk_gate(self._processed_actions[:, 4]),
        self._walk_enabled,
      )
      self._ts_curr = torch.where(
        due_for_sine_update,
        self._map_step_timing(self._processed_actions[:, 5]),
        self._ts_curr,
      )

      # Twist: same cadence and same direct snap to the mapped target as the
      # other channels (no rate limit). due_for_sine_update is (num_envs,);
      # unsqueeze to broadcast against the (num_envs, 3) twist tensors.
      target_twist = self._map_twist(self._processed_actions[:, 6:9])
      self._twist_curr = torch.where(
        due_for_sine_update.unsqueeze(-1), target_twist, self._twist_curr
      )

    self._dispatch_ticks_since_sine_update = (
      self._dispatch_ticks_since_sine_update + 1
    ) % self._sine_param_period_ticks

    # Written every control period regardless of which envs were due this
    # tick, using self._physical_curr's current values -- matches
    # datastore_scalar_inputs' own "unconditional every-period write"
    # contract (docs/coupling.md) and the old file's own "write current
    # value every dispatch tick, update only on the slow clock" convention.
    self.set_datastore_scalar_input(SET_OFFSET, self._physical_curr["offset"])
    self.set_datastore_scalar_input(SET_FREQUENCY, self._physical_curr["frequency"])
    self.set_datastore_scalar_input(SET_SIN_AMP, self._physical_curr["sin_amp"])
    self.set_datastore_scalar_input(SET_COS_AMP, self._physical_curr["cos_amp"])
    self.set_datastore_scalar_input(
      SET_POLICY_WANTS_WALK,
      self._walk_enabled.to(dtype=torch.get_default_dtype()),
    )
    self.set_datastore_scalar_input(SET_TS, self._ts_curr)
    # Vector3d counterpart of the scalar writes above -- same unconditional
    # every-period contract (datastore_vectors_inputs, not just
    # datastore_scalar_inputs). Inert on the controller side whenever
    # rlVelocityControl is off; see the module docstring.
    self.set_datastore_vector_input(SET_RL_REF_VEL, self._twist_curr)