"""Learned CoM-height sine parameter action for ISMPC.

This is the native-`mc_rtc_interface` counterpart of the old Cython-bridge
`ismpc_sine_action.py`. The RL policy's 6 raw actions are mapped into
physical ISMPC parameters each slow sine-param tick:

  offset            = clip(offset_scale * raw[0] + offset_bias, OFFSET_MIN, OFFSET_MAX)   (m)
  frequency         = clip(exp(frequency_scale * raw[1] + frequency_bias), FREQUENCY_MIN, FREQUENCY_MAX)  (Hz)
  sin_amp, cos_amp  = raw[2], raw[3] scaled by amplitude_scale, then radius-clamped
                       against offset so offset - amplitude >= 0 structurally
  walk_gate         = raw[4] > walk_gate_bias                               (bool)
  ts                = clip(ts_scale * raw[5] + ts_bias, TS_MIN, TS_MAX)     (s)

Joint actuation itself is unchanged from the old file: mc_rtc's own q/alpha
output drives the joints directly. There is no joint-space residual in this
task -- RL only ever touches the ISMPC parameters, never joint commands.

Subclasses `McRtcActionBase` (not `McRtcResidualActionBase`): this action has
no residual concept whatsoever -- its action space is a fixed 6, unrelated to
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

  def build(self, env) -> "IsmpcSineAction":
    return IsmpcSineAction(self, env)


class IsmpcSineAction(McRtcActionBase):
  """Maps a 6-dim raw RL action to ISMPC CoM-height sine + walk-gate + Ts.

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
    # has any meaning for this action's fixed 6-dim space, so _action_dim/
    # _raw_actions/_processed_actions are unconditionally replaced right
    # below, and _scale/_offset/_clip are simply never read anywhere in this
    # class (cfg.scale/cfg.offset/cfg.clip are left at BaseActionCfg's inert
    # defaults of 1.0/0.0/None and should not be set by task configs using
    # this action -- there is no affine mapping here, only the explicit
    # _map_to_physical/_map_walk_gate/_map_step_timing methods below).
    super().__init__(cfg, env)

    self._action_dim = 6
    self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    # Sine-param update cadence, in units of controller-dispatch ticks (not
    # physics substeps). See cfg.sine_param_frequency_hz's docstring for why
    # this is kept separate from frameskip.
    controller_hz = 1.0 / (self._env.step_dt / cfg.frameskip)
    self._sine_param_period_ticks = max(
      1, round(controller_hz / cfg.sine_param_frequency_hz)
    )
    self._dispatch_ticks_since_sine_update = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
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

  # ---- Required ActionTerm properties/methods. ----

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    """Store this period's raw policy output; no scale/offset/clip here --
    the six physical mappings happen in apply_actions/the slow-cadence
    block, same split the old Cython-bridge file used."""
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
  def target_height_obs(self) -> torch.Tensor:
    """Live CoM-height reference (m), evaluated now, as a (num_envs, 1)
    tensor. Debugging/visualization only -- not the C++-side ground truth."""
    now = self._env.episode_length_buf.to(
      dtype=torch.get_default_dtype()
    ) * self._env.step_dt
    t_in_period = now - self._period_t0
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