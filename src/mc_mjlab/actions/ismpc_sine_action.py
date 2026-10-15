"""Learned CoM-height sine parameter action for ISMPC.

The RL policy's 9 raw actions are mapped into physical ISMPC parameters each
slow sine-param tick:

  offset            = clip(offset_scale * raw[0] + offset_bias, OFFSET_MIN, OFFSET_MAX)   (m)
  frequency         = clip(exp(frequency_scale * raw[1] + frequency_bias), FREQUENCY_MIN, FREQUENCY_MAX)  (Hz)
  sin_amp, cos_amp  = raw[2], raw[3] scaled by amplitude_scale, then radius-clamped
                       against offset so offset - amplitude >= 0 structurally
  walk_gate         = raw[4] > walk_gate_bias                               (bool)
  ts                = clip(ts_scale * raw[5] + ts_bias, TS_MIN, TS_MAX)     (s)
  twist (vx,vy,w)   = clip(raw[6:9], -1, 1) * twist_scale                   (m/s, m/s, rad/s)

The twist (vx, vy, omega) drives ismpc_walking's footstep reference velocity
via the datastore setters (SET_RL_REF_VEL_X/Y/Z), ONLY when the controller-side
rlVelocityControl toggle is on (YAML `walking_controller.rl_velocity_control`
or the "RL Reference Velocity" GUI checkbox in Walking_controller -- this
toggle is NOT exposed to or writable from Python; see
Walking_controller.h/.cpp). When that toggle is off, this class's writes to
SET_RL_REF_VEL_X/Y/Z are harmless but inert: the controller's own mux
(Walking_controller::updateReferenceVelocity()) ignores rl_reference_velocity
and follows user_reference_velocity (GUI/joystick) instead. The controller
exposes its estimated CoM velocity (base axes) via ismpc_walking::get_com_lin_vel,
read as a datastore_vectors_outputs entry -- see com_lin_vel_est_obs -- which is
the policy's "com_lin_vel" observation.

Joint actuation: mc_rtc's own q/alpha output drives the joints directly.
There is no joint-space residual in this
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
from mc_mjlab.robots import robot_module as robots

# --- Physical ranges. ---
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
# in mc_mjlab/bridge/controller_datastore.py). ---
SET_OFFSET = "ismpc_walking::set_com_height_sine_offset"
SET_FREQUENCY = "ismpc_walking::set_com_height_sine_frequency"
SET_SIN_AMP = "ismpc_walking::set_com_height_sine_sin_amp"
SET_COS_AMP = "ismpc_walking::set_com_height_sine_cos_amp"
SET_POLICY_WANTS_WALK = "ismpc_walking::set_policy_wants_walk"
SET_TS = "ismpc_walking::set_ts"
GET_WANTS_STOP = "ismpc_walking::wants_stop_d"
GET_ROBOT_WALKING = "ismpc_walking::robot_walking_d"
# The policy's twist, one scalar setter per component (datastore_scalar_inputs), like the contract's twist outputs.
# They write Walking_controller::rl_reference_velocity only -- inert unless the controller-side rlVelocityControl
# toggle is on (see the module docstring).
SET_RL_REF_VEL_X = "ismpc_walking::set_rl_ref_vel_x"
SET_RL_REF_VEL_Y = "ismpc_walking::set_rl_ref_vel_y"
SET_RL_REF_VEL_Z = "ismpc_walking::set_rl_ref_vel_z"
# Vector3d (datastore_vectors_inputs/outputs), unlike the scalars above. get_com_lin_vel reads Walking_controller::
# estimatedComLinVel() (realRobot CoM velocity rotated into base axes).
# The commanded target twist (vx, vy, wz): training writes the sampled command here every control period and the
# actor reads it back through GET_USER_REF_VEL, the same key the deployed policy reads for target_twist. The
# controller ignores it for motion while its rl_velocity_control mux is on (the policy's own twist drives then).
SET_USER_REF_VEL = "ismpc_walking::set_user_ref_vel"
GET_USER_REF_VEL = "ismpc_walking::get_user_ref_vel"
GET_COM_LIN_VEL = "ismpc_walking::get_com_lin_vel"
# Scalar getters (datastore_scalar_outputs): the three low-passed ISMPC signals (norms, metres) as left by the
# controller's last run(), and the cutoff period (s) of that filter, which the exporter records in the contract.
GET_FILT_PERTURBATION = "ismpc_walking::get_filt_perturbation"
GET_FILT_ZMP_ERROR = "ismpc_walking::get_filt_zmp_error"
GET_FILT_DCM_MEAS_PRED_ERROR = "ismpc_walking::get_filt_dcm_meas_pred_error"
GET_OBS_FILTER_CUTOFF_T = "ismpc_walking::get_obs_filter_cutoff_T"
# Controller-estimated CoM height above the support foot (m, raw, ~0.93).
GET_COM_HEIGHT_EST = "ismpc_walking::get_com_height_est"
# What the controller currently holds of the policy's own commands, and the phase of the RlSine CoM-height reference
# it is playing (MoveCoM's m_rl_phase_applied_). These are the keys the deployed policy reads for last_sine_params,
# rl_phase_sincos, last_walk_action, last_step_timing_action and last_twist_action.
GET_COM_HEIGHT_SINE_PARAMS = "ismpc_walking::get_com_height_sine_params"  # VectorXd 4: offset, frequency, sin_amp, cos_amp (after the radius cap)
GET_RL_PHASE_SINCOS = "ismpc_walking::get_rl_phase_sincos"  # VectorXd 2: sin, cos
GET_POLICY_WANTS_WALK = "ismpc_walking::get_policy_wants_walk"  # double, 1.0 = walk
GET_TS_TARGET = "ismpc_walking::get_ts_target"  # double, s
GET_RL_REF_VEL = "ismpc_walking::get_rl_ref_vel"  # Vector3d, (vx, vy, omega)
# Joint encoder positions / velocities of the controller's realRobot(), one value per reference joint in the robot's
# reference joint order (VectorXd). These are the keys the deployed policy reads for joint_pos / joint_vel.
GET_JOINT_POS = "ismpc_walking::get_joint_pos"
GET_JOINT_VEL = "ismpc_walking::get_joint_vel"
# Base angular velocity (rad/s) and gravity direction (unit vector, down = (0, 0, -1) when upright), both in the base
# frame (Vector3d): the controller's own state estimate (E * realRobot().velW().angular(), E * (0, 0, -1)).
# These are the keys the deployed policy reads for base_ang_vel / projected_gravity.
GET_BASE_ANG_VEL = "ismpc_walking::get_base_ang_vel"
GET_PROJECTED_GRAVITY = "ismpc_walking::get_projected_gravity"


@dataclass(kw_only=True)
class IsmpcSineActionCfg(McRtcActionCfg):
  """Configuration for the learned ISMPC CoM-height sine parameter action.

  Subclasses `McRtcActionCfg`, not `McRtcResidualActionCfg`: no residual
  fields apply here -- this action controls no per-joint actuators via RL at
  all (its action space is a fixed 9, unrelated to joint count). Joint
  actuation is still driven by mc_rtc's own q/alpha output, same as the
  residual actions, but that's internal wiring, not something the cfg needs
  residual-specific fields for.

  `actuator_names` (inherited from `BaseActionCfg`) selects which of the
  entity's joints mc_rtc drives via q/alpha -- NOT the RL action's own
  targets (the fixed sine/gate/timing/twist parameters). `scale`/`offset`/`clip`
  (also inherited) are
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

  command_name: str = "twist"
  """Name of the velocity command (vx, vy, wz) that is fed to the controller's user reference velocity."""

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
         SET_COS_AMP, SET_POLICY_WANTS_WALK, SET_TS, SET_RL_REF_VEL_X, SET_RL_REF_VEL_Y, SET_RL_REF_VEL_Z)
      )
    )
    cfg.datastore_scalar_outputs = tuple(
      dict.fromkeys(
        (*cfg.datastore_scalar_outputs, GET_WANTS_STOP, GET_ROBOT_WALKING,
         GET_FILT_PERTURBATION, GET_FILT_ZMP_ERROR, GET_FILT_DCM_MEAS_PRED_ERROR, GET_OBS_FILTER_CUTOFF_T,
         GET_COM_HEIGHT_EST, GET_POLICY_WANTS_WALK, GET_TS_TARGET)
      )
    )
    # Vector3d counterparts of the above, same declare-before-super() pattern
    # and same (num_envs, 3)-shaped transport (set_datastore_vector_input /
    # datastore_vector_output -- see mc_rtc_action_base.py).
    cfg.datastore_vectors_inputs = tuple(
      dict.fromkeys((*cfg.datastore_vectors_inputs, SET_USER_REF_VEL))
    )
    cfg.datastore_vectors_outputs = tuple(
      dict.fromkeys(
        (*cfg.datastore_vectors_outputs, GET_COM_LIN_VEL, GET_BASE_ANG_VEL, GET_PROJECTED_GRAVITY,
         GET_RL_REF_VEL, GET_USER_REF_VEL)
      )
    )
    # VectorXd getters: (name, size) pairs. The size is the number of reference joints; the native layer checks it
    # against the controller's getter when it initializes.
    n_ref_joints = len(robots.get_ref_joint_order(cfg.mc_rtc_robot_name))
    cfg.datastore_vectorx_outputs = tuple(
      dict.fromkeys(
        (*cfg.datastore_vectorx_outputs, (GET_JOINT_POS, n_ref_joints), (GET_JOINT_VEL, n_ref_joints),
         (GET_COM_HEIGHT_SINE_PARAMS, 4), (GET_RL_PHASE_SINCOS, 2))
      )
    )

    # `BaseAction.__init__` (via McRtcActionBase's own super().__init__())
    # resolves `_entity`/`_target_ids`/`_target_names` from `cfg.actuator_names`
    # the same way it does for the residual actions -- no explicit resolution
    # needed here, since this class subclasses McRtcActionBase -> BaseAction.
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
    # Controller phase (rad) of the CoM-height reference at the instant the current period's params took effect.
    # Both sines of the continuity rewards are evaluated at this SAME phase: the controller integrates the phase,
    # so a frequency change only changes its rate and cannot move the reference.
    self._period_phi = zeros.clone()

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
    # rl_reference_velocity.setZero().
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
    the slow-cadence block of _advance_sine_period."""
    self._raw_actions[:] = actions
    self._processed_actions[:] = actions

  # ---- Raw action -> physical units. ----

  def _map_to_physical(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """First 4 of the 9-wide raw action -> {offset, frequency, sin_amp,
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
    motion" rather than some arbitrary offset.
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
    m/s and rad/s"""
    return self._twist_curr

  @property
  def com_lin_vel_est_obs(self) -> torch.Tensor:
    """Controller-estimated CoM linear velocity (vx, vy, vz), (num_envs, 3),
    m/s, in base axes, read via the GET_COM_LIN_VEL datastore output
    (Walking_controller::estimatedComLinVel). Collected every control period
    by McRtcActionBase's generic datastore-output mechanism."""
    return self.datastore_vector_output(GET_COM_LIN_VEL)

  @property
  def base_ang_vel_est_obs(self) -> torch.Tensor:
    """Controller-estimated base angular velocity (rad/s), (num_envs, 3), base frame
    (ismpc_walking::get_base_ang_vel)."""
    return self.datastore_vector_output(GET_BASE_ANG_VEL)

  @property
  def projected_gravity_est_obs(self) -> torch.Tensor:
    """Controller-estimated gravity direction in the base frame, (num_envs, 3)
    (ismpc_walking::get_projected_gravity)."""
    return self.datastore_vector_output(GET_PROJECTED_GRAVITY)

  @property
  def target_height_obs(self) -> torch.Tensor:
    """Live CoM-height reference (m), evaluated now, as a (num_envs, 1)
    tensor. Debugging/visualization only -- not the C++-side ground truth."""
    phase = self.datastore_vectorx_output(GET_RL_PHASE_SINCOS)
    sin_phase, cos_phase = phase[:, 0], phase[:, 1]
    h = (
      self._physical_curr["offset"]
      + self._physical_curr["sin_amp"] * sin_phase
      + self._physical_curr["cos_amp"] * cos_phase
    )
    return h.unsqueeze(-1)

  @property
  def ismpc_wants_stop_obs(self) -> torch.Tensor:
    """ISMPC's own advisory safety opinion from the most recent MPC solve,
    (num_envs, 1) float (1.0 = ISMPC would have stopped). Read through the
    base's generic datastore-output machinery."""
    return self.datastore_scalar_output(GET_WANTS_STOP).unsqueeze(-1)

  @property
  def rl_phase_sincos_obs(self) -> torch.Tensor:
    """(sin, cos) of the controller's integrated RlSine phase, (num_envs, 2). Phase of the reference the
    controller is playing (ismpc_walking::get_rl_phase_sincos), collected by the generic datastore-output
    machinery."""
    return self.datastore_vectorx_output(GET_RL_PHASE_SINCOS)

  @property
  def sine_params_obs(self) -> torch.Tensor:
    """(offset, frequency, sin_amp, cos_amp) the controller currently uses, (num_envs, 4)
    (ismpc_walking::get_com_height_sine_params, after the controller's radius cap). The reward terms keep reading
    the Python mirror `physical_params`."""
    return self.datastore_vectorx_output(GET_COM_HEIGHT_SINE_PARAMS)

  @property
  def walk_command_obs(self) -> torch.Tensor:
    """Walk/stop command the controller holds (1.0 = walk), (num_envs, 1) (ismpc_walking::get_policy_wants_walk).
    The reward terms keep reading the Python mirror `last_walk_action`."""
    return self.datastore_scalar_output(GET_POLICY_WANTS_WALK).unsqueeze(-1)

  @property
  def step_timing_obs(self) -> torch.Tensor:
    """Step timing Ts (s) the controller holds, (num_envs, 1) (ismpc_walking::get_ts_target, after its own
    ts_range clamp). The reward terms keep reading the Python mirror `last_step_timing_action`."""
    return self.datastore_scalar_output(GET_TS_TARGET).unsqueeze(-1)

  @property
  def twist_command_obs(self) -> torch.Tensor:
    """Reference twist (vx, vy, omega) the controller holds, (num_envs, 3) (ismpc_walking::get_rl_ref_vel). The
    reward terms keep reading the Python mirror `last_twist_action`."""
    return self.datastore_vector_output(GET_RL_REF_VEL)

  @property
  def user_ref_vel_obs(self) -> torch.Tensor:
    """Target twist (vx, vy, wz) the controller holds, (num_envs, 3) (ismpc_walking::get_user_ref_vel). Training
    feeds it from the sampled command every control period."""
    return self.datastore_vector_output(GET_USER_REF_VEL)

  @property
  def com_height_est_obs(self) -> torch.Tensor:
    """Controller-estimated CoM height above the support foot (m), (num_envs, 1)
    (ismpc_walking::get_com_height_est -> Walking_controller::estimatedComHeight)."""
    return self.datastore_scalar_output(GET_COM_HEIGHT_EST).unsqueeze(-1)

  @property
  def joint_pos_est_obs(self) -> torch.Tensor:
    """Controller joint positions (rad), (num_envs, n_reference_joints), in the robot's reference joint order
    (ismpc_walking::get_joint_pos: realRobot().encoderValues()). Select joints with ref_columns."""
    return self.datastore_vectorx_output(GET_JOINT_POS)

  @property
  def joint_vel_est_obs(self) -> torch.Tensor:
    """Controller joint velocities (rad/s), (num_envs, n_reference_joints), reference joint order
    (ismpc_walking::get_joint_vel: realRobot().encoderVelocities())."""
    return self.datastore_vectorx_output(GET_JOINT_VEL)

  def ref_columns(self, joint_names) -> torch.Tensor:
    """Column of each named joint in the reference-order joint vectors above, as a long tensor (cached)."""
    key = tuple(joint_names)
    cache = self.__dict__.setdefault("_ref_columns_cache", {})
    if key not in cache:
      order = list(self._bridge.layout.input.joint_order)
      missing = [name for name in key if name not in order]
      if missing:
        raise ValueError(f"joints {missing} are not in the controller's reference joint order")
      cache[key] = torch.tensor(
        [order.index(name) for name in key], dtype=torch.long, device=self.device
      )
    return cache[key]

  @property
  def filt_perturbation_obs(self) -> torch.Tensor:
    """Low-passed ISMPC perturbation |w_.xy| (m), (num_envs, 1). Filtered inside the controller (cutoff period
    walking_controller.obs_filter_cutoff_T), read through the generic datastore-output machinery."""
    return self.datastore_scalar_output(GET_FILT_PERTURBATION).unsqueeze(-1)

  @property
  def filt_zmp_error_obs(self) -> torch.Tensor:
    """Low-passed |measured ZMP - MPC planned ZMP|.xy (m), (num_envs, 1). See filt_perturbation_obs."""
    return self.datastore_scalar_output(GET_FILT_ZMP_ERROR).unsqueeze(-1)

  @property
  def filt_dcm_meas_pred_error_obs(self) -> torch.Tensor:
    """Low-passed |DCM bias| (m), (num_envs, 1). See filt_perturbation_obs."""
    return self.datastore_scalar_output(GET_FILT_DCM_MEAS_PRED_ERROR).unsqueeze(-1)

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
    self._period_phi[rows] = 0.0

    # Matches Walking_controller::reset()'s own T_Steps/policyWantsWalk
    # defaults -- every episode is independent, so the Python-side mirror
    # must snap back to the same defaults the controller itself resets to.
    self._ts_curr[rows] = TS_DEFAULT
    self._walk_enabled[rows] = False
    # Same reasoning, extended to the twist: matches
    # Walking_controller::reset()'s rl_reference_velocity.setZero().
    self._twist_curr[rows] = 0.0

    # Counter 0 = "due". With decimation = 10 * frameskip the latch fires on
    # the first controller tick of every env step, so the policy output
    # sampled from an observation is latched with no delay. The first env
    # step after a reset is skipped by the episode_length_buf gate in
    # _advance_sine_period (settle guard: the controller reset is still
    # being serviced).
    self._dispatch_ticks_since_sine_update[rows] = 0

    # The datastore input feed is only refreshed at the END of _advance_sine_period, i.e. AFTER the
    # dispatch of that period has already read it. So the first dispatch of a new episode (the one
    # carrying the native reset flag) would send the PREVIOUS episode's last values (CoM-height sine
    # params, walk gate, Ts, twist) and the controller's first solve would consume them. Overwrite the
    # reset rows of the feed with the same neutral values the mirrors above were just reset to.
    scalar_feed = self._datastore_scalar_input_feed
    scalar_cols = self._datastore_scalar_input_columns
    for setter, values in (
      (SET_OFFSET, self._physical_curr["offset"]),
      (SET_FREQUENCY, self._physical_curr["frequency"]),
      (SET_SIN_AMP, self._physical_curr["sin_amp"]),
      (SET_COS_AMP, self._physical_curr["cos_amp"]),
      (SET_POLICY_WANTS_WALK, self._walk_enabled.to(dtype=scalar_feed.dtype)),
      (SET_TS, self._ts_curr),
      (SET_RL_REF_VEL_X, self._twist_curr[:, 0]),
      (SET_RL_REF_VEL_Y, self._twist_curr[:, 1]),
      (SET_RL_REF_VEL_Z, self._twist_curr[:, 2]),
    ):
      index = self._datastore_input_index(scalar_cols, setter, "scalar")
      scalar_feed[rows, index] = values[rows].to(dtype=scalar_feed.dtype)

    # The base zeroes the collected datastore outputs of the reset rows, and they stay zero until the first fresh
    # collect. For the phase zero is off-distribution (sin = cos = 0 is not on the unit circle): seed what the
    # controller itself holds right after its reset (phase 0). Every other collected value (CoM height, joints,
    # base velocity, gravity, sine params, walk, Ts, twist) is NOT seeded and reads 0 until the first collect.
    phase = self._datastore_vectorx_outputs[GET_RL_PHASE_SINCOS]
    phase[rows] = torch.tensor([0.0, 1.0], device=phase.device, dtype=phase.dtype)

  def apply_actions(self) -> None:
    """Advance the control period on its first substep, then write
    interpolated q/alpha targets."""
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
    """
    # The sampled command goes into the feed BEFORE the dispatch, so even the first dispatch of an episode (the one
    # carrying the native reset flag) already carries the command of the new episode. The worker writes the
    # datastore inputs after its reset, so the controller's own zeroing of the user velocity cannot overwrite it.
    self.set_datastore_vector_input(
      SET_USER_REF_VEL, self._env.command_manager.get_command(self.cfg.command_name)
    )
    super()._advance_control_period()

    # episode_length_buf is incremented after the physics loop, so it is 0
    # during the first env step after a reset: no latch in that window.
    due_for_sine_update = (self._dispatch_ticks_since_sine_update == 0) & (
      self._env.episode_length_buf > 0
    )

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
      # Phase of the reference at the splice, from the latest collected controller phase.
      phase = self.datastore_vectorx_output(GET_RL_PHASE_SINCOS)
      phi_now = torch.atan2(phase[:, 0], phase[:, 1])
      self._period_phi = torch.where(due_for_sine_update, phi_now, self._period_phi)

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
    # contract (docs/coupling.md): the current value is written every dispatch
    # tick and updated only on the slow clock.
    self.set_datastore_scalar_input(SET_OFFSET, self._physical_curr["offset"])
    self.set_datastore_scalar_input(SET_FREQUENCY, self._physical_curr["frequency"])
    self.set_datastore_scalar_input(SET_SIN_AMP, self._physical_curr["sin_amp"])
    self.set_datastore_scalar_input(SET_COS_AMP, self._physical_curr["cos_amp"])
    self.set_datastore_scalar_input(
      SET_POLICY_WANTS_WALK,
      self._walk_enabled.to(dtype=torch.get_default_dtype()),
    )
    self.set_datastore_scalar_input(SET_TS, self._ts_curr)
    # The twist, one scalar per component, same unconditional every-period contract. Inert on the controller side
    # whenever rlVelocityControl is off; see the module docstring.
    self.set_datastore_scalar_input(SET_RL_REF_VEL_X, self._twist_curr[:, 0])
    self.set_datastore_scalar_input(SET_RL_REF_VEL_Y, self._twist_curr[:, 1])
    self.set_datastore_scalar_input(SET_RL_REF_VEL_Z, self._twist_curr[:, 2])