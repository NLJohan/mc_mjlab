"""Worker-side mc_rtc controller host.

Owns everything that touches the mc_rtc bindings; imports only numpy, the
stdlib and the bindings -- never torch or mjlab, so it stays light in worker
processes. I/O flows through two ``IoLayout``-shaped shared-memory blocks;
commands travel over a pipe per worker, whose send/recv also orders the
shared-memory writes. The same ``ControllerHost`` serves the in-process path
(``use_worker_processes=False``).
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# Out-of-tree bindings; the host raises ImportError on construction if absent.
try:
  import eigen
  import mc_control
  import sva
except ImportError:
  mc_control = None
  sva = None
  eigen = None

# Separately optional: only needed by tasks that set IoLayout.has_ismpc_sine.
# Unlike the group above, its absence must NOT break tasks that don't use it
# (e.g. residual_balance) -- checked lazily in step_env instead of gating
# ControllerHost construction.
try:
  import ismpc_walking_python
except ImportError:
  ismpc_walking_python = None


@contextlib.contextmanager
def suppress_mc_rtc_output() -> Iterator[None]:
  """Silence mc_rtc's terminal logging for the duration of the block."""
  sys.stdout.flush()
  sys.stderr.flush()
  saved_out, saved_err = os.dup(1), os.dup(2)
  capture = tempfile.TemporaryFile()
  try:
    os.dup2(capture.fileno(), 1)
    os.dup2(capture.fileno(), 2)
    try:
      yield
    finally:
      # Let spdlog's async flush thread drain before restoring the fds.
      time.sleep(0.05)
      sys.stdout.flush()
      sys.stderr.flush()
      os.dup2(saved_out, 1)
      os.dup2(saved_err, 2)
  except BaseException:
    capture.seek(0)
    os.write(saved_err, capture.read())
    raise
  finally:
    os.close(saved_out)
    os.close(saved_err)
    capture.close()


@contextlib.contextmanager
def redirect_output_to_devnull() -> Iterator[None]:
  """Discard fds 1/2 for the duration of the block, cheaply."""
  sys.stdout.flush()
  sys.stderr.flush()
  saved_out, saved_err = os.dup(1), os.dup(2)
  devnull = os.open(os.devnull, os.O_WRONLY)
  try:
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
      yield
    finally:
      sys.stdout.flush()
      sys.stderr.flush()
      os.dup2(saved_out, 1)
      os.dup2(saved_err, 2)
  finally:
    os.close(devnull)
    os.close(saved_out)
    os.close(saved_err)


class _IterItemsDict(dict):
  """``setWrenches`` iterates with Python-2 ``.iteritems()``; alias it."""

  def iteritems(self):
    return iter(self.items())


# Controller output channel -> the rbdyn MultiBodyConfig member holding it.
# All are per-joint vectors indexed like mbc.q, so one write loop serves all.
MBC_ATTR_BY_CHANNEL = {"q": "q", "alpha": "alpha", "tau": "jointTorque"}


@dataclass(frozen=True)
class HostMetadata:
  """What the trainer needs to know about the mc_rtc robot, probed worker-side."""

  ref_joint_order: tuple[str, ...]
  body_sensor_names: tuple[str, ...]
  force_sensor_names: tuple[str, ...]
  has_named_setters: bool
  has_reset: bool


@dataclass(frozen=True)
class IoLayout:
  """Column layout of the shared input/output blocks (one row per env).

  Input row::

    [0, T)          target-joint positions (encoders)
    [T, 2T)         target-joint velocities
    [2T, 3T)        target-joint torques (qfrc_actuator)
    [3T, 3T+16)     root block; the first 7 are always pos(3) + quat wxyz(4):
                      named routing:   qpos7, qvel6, qacc3
                      singular routing: pos3, quat4, linvel3, omega_body3, accel3
    [imu_off, ...)  6 per IMU body sensor: gyro(3), accel(3)
    [wrench_off, ..) 6 per force sensor: force(3), torque(3) as MuJoCo reads them

    [ismpc_sine_off, +4)  present only when ``has_ismpc_sine``: the RL-set
                    CoM-height sine reference, written by the action term and
                    consumed worker-side (via the mc_rtc datastore) right
                    before ``controller.run()``: offset (m), amplitude (m,
                    already ratio*offset so it can never drive the height
                    trajectory negative), frequency (Hz), phase (rad).

    [ismpc_velocity_off, +3)  present only when ``has_ismpc_velocity``: the
                    sampled reference walking velocity (vx, vy, wz; m/s,
                    m/s, rad/s), written by the action term from
                    ``env.command_manager``'s sampled command and consumed
                    worker-side right before ``controller.run()``, same
                    timing as the sine reference above.

    [ismpc_walk_off, +1)  present only when ``has_ismpc_walk_gate``: the
                    RL policy's walk/stop decision (1.0 = walk, else stop),
                    written by the action term and consumed worker-side via
                    the ismpc_walking_python bridge's set_policy_wants_walk
                    right before ``controller.run()``, same timing as the
                    sine/velocity channels above. The policy has full,
                    unconditional authority over walking -- this overrides
                    whatever ISMPC's own autonomous safety-stop logic would
                    otherwise have wanted (see ismpc_wants_stop_off below,
                    the observation the policy can react to instead).

    [ismpc_ts_off, +1)  present only when ``has_ismpc_ts``: the RL-set step
                    timing Ts (s between footsteps), written by the action
                    term and consumed worker-side via the
                    ismpc_walking_python bridge's set_step_timing right
                    before ``controller.run()``, same timing as the other
                    ISMPC channels above. Clamped controller-side to
                    ``controller_config_.ts_range`` regardless of what's
                    written here (see ``Walking_controller::ts(double)``).
                    Independent of has_ismpc_walk_gate/has_ismpc_sine/
                    has_ismpc_velocity, appended after all three so existing
                    layouts' widths are unaffected by this field existing.
                    Manual GUI control of Ts remains available (see
                    ``Walking_controller::policyControlsTs``); this channel
                    is only consulted here when the task sets
                    ``has_ismpc_ts=True``.

  Output row: one T-wide block per entry of ``output_channels``, in order --
  e.g. the default ``("q", "alpha")`` gives q in ``[0, T)`` and alpha in
  ``[T, 2T)``, for the target joints -- followed by a single status column at
  ``status_off`` carrying 1.0 once the controller has failed (see
  ``ControllerHost.step_env``), followed by ``ismpc_wants_stop_off``
  (present only when ``has_ismpc_walk_gate``): ISMPC's own advisory safety
  opinion from the most recent MPC solve (1.0 = ISMPC would have stopped),
  independent of what the policy actually commanded -- read back via the
  bridge's get_ismpc_wants_stop so the policy can observe and learn from it.
  Followed by ``is_walking_off`` (present only when ``has_ismpc_walk_gate``,
  same gate as ismpc_wants_stop_off since both are read back together):
  the controller's ACTUAL current walking state
  (``Walking_controller::Robot_Walking``, 1.0 = walking), ground truth
  distinct from both ismpc_wants_stop_off (ISMPC's advisory opinion) and
  ismpc_walk_off (the policy's own commanded intent) -- read back via the
  bridge's get_is_walking.
  """

  num_targets: int
  named_routing: bool
  has_floating_base_sensor: bool
  use_reset: bool
  # Singular routing only: the accel slot carries a real accelerometer reading.
  feed_accel_fallback: bool
  # (body sensor name, has_gyro, has_accel) per IMU, in input-block order.
  imu: tuple[tuple[str, bool, bool], ...] = ()
  # Force sensor names, in input-block order.
  wrenches: tuple[str, ...] = ()
  # Controller outputs written per env, in output-block order. Keys of
  # MBC_ATTR_BY_CHANNEL; the action term's `output_channels` must match.
  output_channels: tuple[str, ...] = ("q", "alpha")
  # Whether this layout carries the ISMPC CoM-height sine input columns.
  # False (default) reproduces the pre-existing in_width exactly, so any task
  # that does not use this channel (e.g. residual_balance) is unaffected.
  has_ismpc_sine: bool = False
  # Whether this layout carries the ISMPC reference-velocity input columns.
  # Independent of has_ismpc_sine (a task could in principle want one
  # without the other), appended after it so has_ismpc_sine-only layouts'
  # widths are completely unaffected by this field existing.
  has_ismpc_velocity: bool = False
  # Whether this layout carries the RL-policy walk/stop gate: one input
  # column (the policy's decision) and one output column (ismpc_wants_stop,
  # ISMPC's own advisory opinion). Independent of has_ismpc_sine/
  # has_ismpc_velocity, appended after both so existing layouts' widths are
  # unaffected by this field existing. See policyWantsWalk/ismpc_wants_stop
  # in Walking_controller.h for the full design rationale.
  has_ismpc_walk_gate: bool = False
  # Whether this layout carries the RL-policy step-timing (Ts) input
  # column. Independent of has_ismpc_sine/has_ismpc_velocity/
  # has_ismpc_walk_gate, appended after all three so existing layouts'
  # widths are unaffected by this field existing. See
  # policyControlsTs/SetPolicyStepTiming in Walking_controller.h for the
  # full design rationale (manual-vs-RL mode split).
  has_ismpc_ts: bool = False

  @property
  def root_off(self) -> int:
    return 3 * self.num_targets

  @property
  def imu_off(self) -> int:
    return self.root_off + 16

  @property
  def wrench_off(self) -> int:
    return self.imu_off + 6 * len(self.imu)

  @property
  def ismpc_sine_off(self) -> int:
    return self.wrench_off + 6 * len(self.wrenches)

  @property
  def ismpc_velocity_off(self) -> int:
    base = self.ismpc_sine_off
    return base + 4 if self.has_ismpc_sine else base

  @property
  def ismpc_walk_off(self) -> int:
    base = self.ismpc_velocity_off
    return base + 3 if self.has_ismpc_velocity else base

  @property
  def ismpc_ts_off(self) -> int:
    base = self.ismpc_walk_off
    return base + 1 if self.has_ismpc_walk_gate else base

  @property
  def in_width(self) -> int:
    base = self.ismpc_ts_off
    return base + 1 if self.has_ismpc_ts else base

  @property
  def status_off(self) -> int:
    return len(self.output_channels) * self.num_targets

  @property
  def ismpc_wants_stop_off(self) -> int:
    return self.status_off + 1

  @property
  def is_walking_off(self) -> int:
    return self.ismpc_wants_stop_off + 1

  @property
  def out_width(self) -> int:
    base = self.status_off + 1  # +1 for the existing failure-status column
    return base + 2 if self.has_ismpc_walk_gate else base


@dataclass
class _ShmHandle:
  """A shared-memory block and its numpy view, kept alive together."""

  shm: SharedMemory
  arr: np.ndarray = field(repr=False)

  def close(self) -> None:
    self.arr = None  # type: ignore[assignment]
    self.shm.close()


def attach_shm(name: str, shape: tuple[int, int]) -> _ShmHandle:
  """Attach to an existing shared block, untracked: the trainer unlinks it.

  `track=False` requires Python 3.13+; on older interpreters we fall back
  to the default (tracked) behavior, which just means this worker's own
  resource tracker may also attempt cleanup on exit — harmless here since
  the trainer is the one that actually unlinks the block.
  """
  if sys.version_info >= (3, 13):
    shm = SharedMemory(name=name, track=False)
  else:
    shm = SharedMemory(name=name)
  arr = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
  return _ShmHandle(shm, arr)


class ControllerHost:
  """Owns a slice of MCGlobalControllers and steps/resets them from I/O rows.

  ``env_ids`` maps global env indices to local controllers; all I/O goes
  through ``IoLayout``-shaped arrays. ``allowed_output_envs`` lists the global
  env ids whose controllers may write to the console; ``None`` (the worker
  path, where silencing is fd-level for the whole process) suppresses nothing.
  """

  def __init__(
    self,
    config_path: str,
    env_ids: Sequence[int],
    target_names: Sequence[str],
    allowed_output_envs: Sequence[int] | None = None,
  ):
    if mc_control is None:
      raise ImportError(
        "mc_control, mc_rbdyn, sva, eigen modules are required for mc_rtc integration."
      )
    self._env_ids = list(env_ids)
    self._local_of = {env_id: k for k, env_id in enumerate(self._env_ids)}
    self._target_names = list(target_names)
    self._allowed_output = (
      None if allowed_output_envs is None else frozenset(allowed_output_envs)
    )

    self._controllers = []
    for env_id in self._env_ids:
      with self._output_guard(env_id):
        self._controllers.append(mc_control.MCGlobalController(config_path))
    # mc_mujoco calls init() exactly once per controller; later resets go
    # through MCGlobalController::reset().
    self._initialized = [False] * len(self._controllers)
    # Set when run() reports failure (the QP gives up once the robot is far
    # enough gone). Latches until the env is reset; see step_env.
    self._failed = [False] * len(self._controllers)
    # DIAGNOSTIC (temporary): consecutive-tick counter for "QP solve did not
    # succeed this tick" (qp_succeeded()==False), independent of controller.run()
    # itself returning True/False -- the pathological training stall observed
    # (worker "went unresponsive for 60s") correlates with the controller
    # logging many repeated "MPC result is too far from stability condition" /
    # "QP Failed" lines for the SAME env while run() keeps returning True (the
    # solver "ignores" the failure and carries on rather than hard-failing).
    # Remove this whole block (and its use in step_env) once the stall's root
    # cause is found and fixed -- this is not meant to be permanent.
    self._consecutive_qp_failures = [0] * len(self._controllers)

    robot = self._controllers[0].robot()
    rn = robot.name()
    self._robot_key: bytes = rn if isinstance(rn, bytes) else rn.encode()

    # jointIndexByName on a missing joint throws a C++ std::out_of_range that
    # terminates the process (it cannot be caught from Python), so always
    # probe with hasJoint first.
    def joint_index(name: str) -> int:
      return robot.jointIndexByName(name) if robot.hasJoint(name) else -1

    # refJointOrder includes joints mjlab does not simulate; those keep their
    # default-stance values.
    self._ref_joint_order = [
      j.decode() if isinstance(j, bytes) else j
      for j in robot.module().ref_joint_order()
    ]
    stance_q = robot.mbc.q
    self._default_encoders = np.array(
      [
        stance_q[joint_index(name)][0]
        if joint_index(name) != -1 and len(stance_q[joint_index(name)]) > 0
        else 0.0
        for name in self._ref_joint_order
      ],
      dtype=np.float64,
    )

    target_to_ref = [
      self._ref_joint_order.index(n) if n in self._ref_joint_order else -1
      for n in self._target_names
    ]
    self._valid_k = np.array(
      [k for k, r in enumerate(target_to_ref) if r != -1], dtype=np.intp
    )
    self._valid_ref = np.array([r for r in target_to_ref if r != -1], dtype=np.intp)
    self._target_mbc_indices = [joint_index(name) for name in self._target_names]
    if all(i == -1 for i in self._target_mbc_indices):
      robot_name = self._robot_key.decode()
      raise RuntimeError(
        f"none of the mjlab entity's joints exist on the mc_rtc robot "
        f"'{robot_name}': the entity does not match the config's MainRobot."
      )

    body_sensor_names: list[str] = []
    try:
      body_sensor_names = [
        s.name().decode() if isinstance(s.name(), bytes) else s.name()
        for s in robot.bodySensors()
      ]
    except AttributeError:
      # Older binding without bodySensors(): probe conventional names.
      for probe in ("FloatingBase", "Accelerometer"):
        if robot.hasBodySensor(probe):
          body_sensor_names.append(probe)

    # The binding exposes no forceSensors() enumeration; probe known names.
    force_sensor_names = [
      name
      for name in (
        "RightFootForceSensor",
        "LeftFootForceSensor",
        "RightHandForceSensor",
        "LeftHandForceSensor",
      )
      if robot.hasForceSensor(name)
    ]

    self._metadata = HostMetadata(
      ref_joint_order=tuple(self._ref_joint_order),
      body_sensor_names=tuple(body_sensor_names),
      force_sensor_names=tuple(force_sensor_names),
      has_named_setters=hasattr(self._controllers[0], "setSensorPositions"),
      has_reset=hasattr(self._controllers[0], "reset"),
    )
    self._layout: IoLayout | None = None
    self._zero_base = np.zeros(len(self._ref_joint_order), dtype=np.float64)

  def metadata(self) -> HostMetadata:
    return self._metadata

  def configure(self, layout: IoLayout) -> None:
    """Fix the I/O layout and precompute the per-step name keys."""
    unknown = set(layout.output_channels) - set(MBC_ATTR_BY_CHANNEL)
    if unknown:
      raise ValueError(
        f"unknown controller output channel(s) {sorted(unknown)}; "
        f"known: {sorted(MBC_ATTR_BY_CHANNEL)}"
      )
    self._layout = layout
    self._output_attrs = [MBC_ATTR_BY_CHANNEL[c] for c in layout.output_channels]
    self._imu_keys = [name.encode() for name, _, _ in layout.imu]
    self._wrench_keys = [name.encode() for name in layout.wrenches]

  def _output_guard(
    self, env_id: int, hot: bool = False
  ) -> contextlib.AbstractContextManager[None]:
    """Suppression wrapper for one env's controller call."""
    if self._allowed_output is None or env_id in self._allowed_output:
      return contextlib.nullcontext()
    return redirect_output_to_devnull() if hot else suppress_mc_rtc_output()

  def _expand(self, values: np.ndarray, base: np.ndarray) -> list[float]:
    """Expand target-joint values into refJointOrder on top of ``base``."""
    full = base.copy()
    full[self._valid_ref] = values[self._valid_k]
    return full.tolist()

  def reset_envs(self, env_ids: Sequence[int], in_arr: np.ndarray) -> None:
    """init() (first time) or reset() the controllers for the given envs."""
    layout = self._layout
    assert layout is not None
    T = layout.num_targets
    ro = layout.root_off
    for env_id in env_ids:
      with self._output_guard(env_id):
        local = self._local_of[env_id]
        controller = self._controllers[local]
        row = in_arr[env_id]
        encoders = self._expand(row[0:T], self._default_encoders)
        pos = row[ro : ro + 3]
        quat = row[ro + 3 : ro + 7]

        if layout.use_reset and self._initialized[local]:
          # reset() takes the inverse of the MuJoCo world<-body quaternion
          # (the same convention init() applies internally to its 7-array).
          q = eigen.Quaterniond(
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
          )
          pose = sva.PTransformd(
            q.inverse(), eigen.Vector3d(float(pos[0]), float(pos[1]), float(pos[2]))
          )
          controller.reset({self._robot_key: encoders}, {self._robot_key: pose})
        else:
          # AttitudeObserver's EKF orientation state is correctly re-seeded from
          # the control robot by Walking_controller::reset() (called internally
          # by controller.init(), confirmed empirically -- this binding has no
          # Python-visible reset() method at all, hasattr(MCGlobalController,
          # 'reset') is False, so EVERY reset_envs() call takes this init()
          # branch, not the controller.reset(...) branch above).
          #
          # FOUND: FloatingBase's position/orientation were never set here --
          # only velocity/acceleration were zeroed. step_env()'s normal
          # per-tick path always sets Position+Orientation+LinearVelocity+
          # AngularVelocity+LinearAcceleration together (see the
          # layout.named_routing branch above); leaving position/orientation
          # untouched at reset meant FloatingBase's position/orientation
          # sensor buffers still held the PREVIOUS episode's last-written
          # pre-fall pose at the moment init() (and the observer pipeline it
          # drives) ran, while everything else (encoders, init_attitude,
          # velocity/acceleration) already reflected the fresh reset state.
          # That position/velocity inconsistency is consistent with what was
          # observed C++-side: realRobot().com() (position) came out sane
          # post-reset, but realRobot().comVelocity() was garbage (tens of
          # m/s) and decayed slowly rather than resetting cleanly -- the
          # signature of an EKF/observer computing an innovation against a
          # stale position. Setting Position+Orientation here too, from the
          # same fresh pos/quat used for init_attitude below, closes that gap.
          zero = eigen.Vector3d(0.0, 0.0, 0.0)
          if layout.has_floating_base_sensor:
            fb = b"FloatingBase"
            fb_quat = eigen.Quaterniond(
              float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            )
            controller.setSensorPosition(
              eigen.Vector3d(float(pos[0]), float(pos[1]), float(pos[2]))
            )
            controller.setSensorOrientation(fb_quat.inverse())
            controller.setSensorLinearVelocity(zero)
            controller.setSensorAngularVelocity(zero)
            controller.setSensorLinearAcceleration(zero)
          
          controller.setSensorAngularVelocity(zero)
          controller.setSensorLinearAcceleration(zero)
          controller.setEncoderValues(encoders)
          # init() attitude is [qw, qx, qy, qz, tx, ty, tz].
          init_attitude = [
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
          ]
          controller.init(encoders, init_attitude)
          self._initialized[local] = True

        controller.running = True
        # Whatever made the QP give up is gone with the new state.
        self._failed[local] = False

  def step_env(self, env_id: int, in_arr: np.ndarray, out_arr: np.ndarray) -> None:
    """Feed one env's input row to its controller, run it, write q/alpha out."""
    layout = self._layout
    assert layout is not None
    T = layout.num_targets
    ro = layout.root_off
    local = self._local_of[env_id]
    controller = self._controllers[local]
    row = in_arr[env_id]

    # Uninitialized covers a respawned worker's rebuilt controllers: their
    # envs read as failed until their next reset takes the init() path.
    if self._failed[local] or not self._initialized[local]:
      out_arr[env_id][layout.status_off] = 1.0
      return

    # DIAGNOSTIC (temporary): phase-timing start. See the matching block
    # after run() for what this measures and why. Remove together.
    _t_start = time.perf_counter()

    controller.setEncoderValues(self._expand(row[0:T], self._default_encoders))
    controller.setEncoderVelocities(self._expand(row[T : 2 * T], self._zero_base))

    if layout.named_routing:
      # mc_mujoco routing: raw free-joint state to "FloatingBase", MuJoCo
      # gyro/accelerometer readings to each IMU body sensor.
      if layout.has_floating_base_sensor:
        fb = b"FloatingBase"
        q = eigen.Quaterniond(
          float(row[ro + 3]), float(row[ro + 4]), float(row[ro + 5]), float(row[ro + 6])
        )
        controller.setSensorPositions(
          {fb: eigen.Vector3d(float(row[ro]), float(row[ro + 1]), float(row[ro + 2]))}
        )
        controller.setSensorOrientations({fb: q.inverse()})
        controller.setSensorLinearVelocities(
          {
            fb: eigen.Vector3d(
              float(row[ro + 7]), float(row[ro + 8]), float(row[ro + 9])
            )
          }
        )
        controller.setSensorAngularVelocities(
          {
            fb: eigen.Vector3d(
              float(row[ro + 10]), float(row[ro + 11]), float(row[ro + 12])
            )
          }
        )
        controller.setSensorLinearAccelerations(
          {
            fb: eigen.Vector3d(
              float(row[ro + 13]), float(row[ro + 14]), float(row[ro + 15])
            )
          }
        )
      gyros = {}
      accels = {}
      for i, (key, (_, has_gyro, has_accel)) in enumerate(
        zip(self._imu_keys, layout.imu, strict=True)
      ):
        off = layout.imu_off + 6 * i
        if has_gyro:
          gyros[key] = eigen.Vector3d(
            float(row[off]), float(row[off + 1]), float(row[off + 2])
          )
        if has_accel:
          accels[key] = eigen.Vector3d(
            float(row[off + 3]), float(row[off + 4]), float(row[off + 5])
          )
      if gyros:
        controller.setSensorAngularVelocities(gyros)
      if accels:
        controller.setSensorLinearAccelerations(accels)
    else:
      # Singular-setter fallback (older binding): everything lands on
      # bodySensors[0]. The trainer already computed omega in the base frame.
      q = eigen.Quaterniond(
        float(row[ro + 3]), float(row[ro + 4]), float(row[ro + 5]), float(row[ro + 6])
      )
      controller.setSensorPosition(
        eigen.Vector3d(float(row[ro]), float(row[ro + 1]), float(row[ro + 2]))
      )
      controller.setSensorOrientation(q.inverse())
      controller.setSensorLinearVelocity(
        eigen.Vector3d(float(row[ro + 7]), float(row[ro + 8]), float(row[ro + 9]))
      )
      controller.setSensorAngularVelocity(
        eigen.Vector3d(float(row[ro + 10]), float(row[ro + 11]), float(row[ro + 12]))
      )
      if layout.feed_accel_fallback:
        controller.setSensorLinearAcceleration(
          eigen.Vector3d(float(row[ro + 13]), float(row[ro + 14]), float(row[ro + 15]))
        )

    # MuJoCo reports the force on the sensor site; mc_rtc wants the reaction
    # on the robot, hence the negation (mc_mujoco's `fs *= -1`).
    if self._wrench_keys:
      wrenches = _IterItemsDict()
      for i, key in enumerate(self._wrench_keys):
        off = layout.wrench_off + 6 * i
        wrenches[key] = sva.ForceVecd(
          [-float(row[off + 3]), -float(row[off + 4]), -float(row[off + 5])],
          [-float(row[off]), -float(row[off + 1]), -float(row[off + 2])],
        )
      controller.setWrenches(wrenches)

    controller.setJointTorques(self._expand(row[2 * T : 3 * T], self._zero_base))

    # DIAGNOSTIC (temporary): phase boundary, sensors done / bridge writes
    # starting -- isolates the ISMPC-specific bridge calls below (absent in
    # standalone mc_mujoco) from the sensor-setting block above (present in
    # both), to find where the ~430us vs ~200us gap vs standalone actually
    # lives. Remove alongside the other diagnostic blocks once resolved.
    _t_sensors_done = time.perf_counter()

    if layout.has_ismpc_sine:
      if ismpc_walking_python is None:
        raise ImportError(
          "IoLayout.has_ismpc_sine is set but the ismpc_walking_python "
          "bridge module could not be imported; build it as part of "
          "ismpc_walking's CMake (see ismpc_walking_python/CMakeLists "
          "entry) and ensure it's on PYTHONPATH."
        )
      off = layout.ismpc_sine_off
      com_height_offset = float(row[off])
      frequency = float(row[off + 1])
      sin_amp = float(row[off + 2])
      cos_amp = float(row[off + 3])
      # Routed through the ismpc_walking_python bridge module, NOT the mc_rtc
      # datastore: the Python bindings for MCController/MCGlobalController do
      # not expose datastore() at all (confirmed empirically -- see the
      # session notes), so this small separate Cython extension (built as
      # part of ismpc_walking's own CMake) is what actually reaches the live
      # Walking_controller/ISMPC_Solver instance from Python. It safely
      # no-ops (returns False) if the loaded controller isn't a
      # Walking_controller, e.g. during any task that doesn't use this
      # channel -- but has_ismpc_sine being True should only ever be paired
      # with a Walking_controller config, so a False here would indicate a
      # config/task mismatch worth investigating, not routine behavior.\

      ismpc_walking_python.set_com_height_sine_params(
        controller.controller(), com_height_offset, frequency, sin_amp, cos_amp
      )

    if layout.has_ismpc_velocity:
      if ismpc_walking_python is None:
        raise ImportError(
          "IoLayout.has_ismpc_velocity is set but the ismpc_walking_python "
          "bridge module could not be imported; build it as part of "
          "ismpc_walking's CMake (see ismpc_walking_python/CMakeLists "
          "entry) and ensure it's on PYTHONPATH."
        )
      off = layout.ismpc_velocity_off
      vx = float(row[off])
      vy = float(row[off + 1])
      wz = float(row[off + 2])
      # Same routing rationale as the sine params above (no datastore
      # access from Python; goes through the ismpc_walking_python bridge
      # instead). Safe to call every step_env call regardless of whether
      # this particular period actually changed the sampled command --
      # Walking_controller::SetReferenceVelocity is a plain assignment, so
      # writing the same value repeatedly is harmless.
      ismpc_walking_python.set_reference_velocity(controller.controller(), vx, vy, wz)

    if layout.has_ismpc_walk_gate:
      if ismpc_walking_python is None:
        raise ImportError(
          "IoLayout.has_ismpc_walk_gate is set but the ismpc_walking_python "
          "bridge module could not be imported; build it as part of "
          "ismpc_walking's CMake (see ismpc_walking_python/CMakeLists "
          "entry) and ensure it's on PYTHONPATH."
        )
      # Same routing rationale as the sine/velocity channels above. The
      # policy has full, unconditional authority: this call directly sets
      # the controller's Stop flag, overriding whatever ISMPC's own
      # autonomous safety logic would otherwise have wanted this tick (see
      # the ismpc_wants_stop readback below, and
      # Walking_controller::SetPolicyWantsWalk for the C++ side). Threshold
      # at 0.5 -- the action term is expected to write a clean 0.0/1.0
      # (e.g. from a sigmoid squash thresholded on its own side), but this
      # keeps step_env robust to any intermediate float value regardless.
      walk_enabled = float(row[layout.ismpc_walk_off]) > 0.5
      ismpc_walking_python.set_policy_wants_walk(controller.controller(), walk_enabled)

    if layout.has_ismpc_ts:
      if ismpc_walking_python is None:
        raise ImportError(
          "IoLayout.has_ismpc_ts is set but the ismpc_walking_python "
          "bridge module could not be imported; build it as part of "
          "ismpc_walking's CMake (see ismpc_walking_python/CMakeLists "
          "entry) and ensure it's on PYTHONPATH."
        )
      # Same routing rationale as the other ISMPC channels above. Clamped
      # controller-side to ts_range regardless of what's written here (see
      # Walking_controller::ts(double)); manual GUI control of Ts remains
      # available and independent of this write (see
      # Walking_controller::policyControlsTs) -- this call always lands
      # when has_ismpc_ts is set, the GUI checkbox only gates whether the
      # GUI's OWN NumberInput is also allowed to write.
      ts = float(row[layout.ismpc_ts_off])
      ismpc_walking_python.set_step_timing(controller.controller(), ts)

    # mc_mujoco stops the whole sim when run() reports failure. A trainer
    # cannot: the QP giving up is the normal end of a fall, and it must cost
    # one episode, not the run. Latch it and let the trainer terminate the env
    # (the last good outputs stay in the block for the substeps still to come).

    # DIAGNOSTIC (temporary): measure real controller.run() wall time inside
    # the worker, split into four phases: sensor setup (present in
    # standalone mc_mujoco too), ISMPC bridge writes (mc_mjlab-only),
    # run() itself, and post-run readback (mc_mjlab-only). Writes directly
    # to a per-worker file, bypassing suppress_output/console_output
    # fd-redirection entirely. Remove once the ~430us vs ~200us gap vs
    # standalone mc_mujoco is understood.
    _t_bridge_write_done = time.perf_counter()
    ok = controller.run()
    _t_run_done = time.perf_counter()
    self._failed[local] = not ok

    if ismpc_walking_python is not None:
      qp_ok = ismpc_walking_python.qp_succeeded(controller.controller())
      if qp_ok is False:
        self._consecutive_qp_failures[local] += 1
      elif qp_ok is True:
        self._consecutive_qp_failures[local] = 0
    mbc = controller.robot().mbc
    out_row = out_arr[env_id]
    out_row[layout.status_off] = 0.0 if ok else 1.0

    if layout.has_ismpc_walk_gate:
      wants_stop = ismpc_walking_python.get_ismpc_wants_stop(controller.controller())
      out_row[layout.ismpc_wants_stop_off] = 1.0 if wants_stop else 0.0
      is_walking = ismpc_walking_python.get_is_walking(controller.controller())
      out_row[layout.is_walking_off] = 1.0 if is_walking else 0.0

    _t_readback_done = time.perf_counter()

    # DIAGNOSTIC (temporary): four-phase breakdown, to find where the
    # ~430us (mc_mjlab, single worker) vs ~200us (standalone mc_mujoco) gap
    # actually lives. sensors: present in both mc_mujoco and mc_mjlab.
    # bridge_write / readback: mc_mjlab-only additions (ISMPC sine/velocity/
    # walk-gate channel). run: the QP solve itself. Remove all four
    # diagnostic blocks in this function together once resolved.
    _dt_sensors = _t_sensors_done - _t_start
    _dt_bridge_write = _t_bridge_write_done - _t_sensors_done
    _dt_run = _t_run_done - _t_bridge_write_done
    _dt_readback = _t_readback_done - _t_run_done
    self._diag_run_total = getattr(self, "_diag_run_total", 0.0) + _dt_run
    self._diag_sensors_total = getattr(self, "_diag_sensors_total", 0.0) + _dt_sensors
    self._diag_bridge_write_total = (
      getattr(self, "_diag_bridge_write_total", 0.0) + _dt_bridge_write
    )
    self._diag_readback_total = (
      getattr(self, "_diag_readback_total", 0.0) + _dt_readback
    )
    self._diag_run_count = getattr(self, "_diag_run_count", 0) + 1
    if self._diag_run_count % 200 == 0:
      n = self._diag_run_count
      diag_dir = os.environ.get("MC_MJLAB_DIAG_DIR", "/tmp/mc_mjlab_diag")
      try:
        os.makedirs(diag_dir, exist_ok=True)
        with open(
          os.path.join(diag_dir, f"worker-{os.getpid()}.diag.log"), "a"
        ) as f:
          f.write(
            f"n={n} envs_hosted={len(self._env_ids)} env={env_id} "
            f"sensors_avg_us={self._diag_sensors_total / n * 1e6:.1f} "
            f"bridge_write_avg_us={self._diag_bridge_write_total / n * 1e6:.1f} "
            f"run_avg_us={self._diag_run_total / n * 1e6:.1f} "
            f"readback_avg_us={self._diag_readback_total / n * 1e6:.1f}\n"
          )
      except OSError:
        pass

    for c, attr in enumerate(self._output_attrs):
      values = getattr(mbc, attr)
      base = c * T
      for k, j in enumerate(self._target_mbc_indices):
        # A joint the controller does not drive (or, for tau, one the QP left
        # unset) reads 0.0; consumers treat that as "no command".
        out_row[base + k] = values[j][0] if j != -1 and len(values[j]) > 0 else 0.0

  def step_envs(
    self, env_ids: Sequence[int], in_arr: np.ndarray, out_arr: np.ndarray
  ) -> None:
    if self._allowed_output is None:
      for env_id in env_ids:
        self.step_env(env_id, in_arr, out_arr)
      return
    for env_id in env_ids:
      with self._output_guard(env_id, hot=True):
        self.step_env(env_id, in_arr, out_arr)


def _start_orphan_watchdog(trainer_pid: int, poll_s: float = 2.0) -> None:
  """Exit the worker once the trainer process is gone."""

  def watch() -> None:
    while True:
      time.sleep(poll_s)
      try:
        os.kill(trainer_pid, 0)
      except ProcessLookupError:
        os._exit(1)
      except OSError:
        pass  # e.g. EPERM: the pid exists, so the trainer is alive

  threading.Thread(target=watch, name="mc_rtc_orphan_watchdog", daemon=True).start()


def worker_main(
  conn: Connection,
  config_path: str,
  env_ids: Sequence[int],
  target_names: Sequence[str],
  suppress_output: bool = False,
  trainer_pid: int | None = None,
) -> None:
  """Entry point of a controller worker process."""
  # Ctrl+C hits the whole foreground process group; shutdown is coordinated by
  # the trainer instead ("stop", pipe EOF, the watchdog, or the daemon flag).
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  if trainer_pid is not None:
    _start_orphan_watchdog(trainer_pid)
  capture = None
  log_dir = os.environ.get("MC_MJLAB_WORKER_LOG_DIR")
  if log_dir:
    fd = os.open(
      os.path.join(log_dir, f"worker-{os.getpid()}.log"),
      os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
      0o644,
    )
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    import faulthandler

    faulthandler.enable()
  elif suppress_output:
    # spdlog writes from C++, so only an fd-level redirect silences it. A
    # capture file rather than /dev/null so error replies can attach mc_rtc's
    # own error text; reply_ok truncates it to keep it from growing.
    capture = tempfile.TemporaryFile()
    os.dup2(capture.fileno(), 1)
    os.dup2(capture.fileno(), 2)

  def error_payload() -> str:
    tb = traceback.format_exc()
    if capture is None:
      return tb
    try:
      capture.seek(0)
      tail = capture.read()[-8192:].decode(errors="replace")
      capture.seek(0)
      capture.truncate()
    except OSError:
      return tb
    if not tail.strip():
      return tb
    return f"{tb}\n--- captured mc_rtc output (tail) ---\n{tail}"

  def reply_ok() -> None:
    conn.send(("ok", None))
    if capture is not None:
      try:
        capture.seek(0)
        capture.truncate()
      except OSError:
        pass

  host: ControllerHost | None = None
  in_h: _ShmHandle | None = None
  out_h: _ShmHandle | None = None
  try:
    try:
      host = ControllerHost(config_path, env_ids, target_names)
    except BaseException:
      conn.send(("error", error_payload()))
      return
    conn.send(("meta", host.metadata()))

    while True:
      cmd, payload = conn.recv()
      try:
        if cmd == "configure":
          layout, in_name, out_name, in_shape, out_shape = payload
          in_h = attach_shm(in_name, in_shape)
          out_h = attach_shm(out_name, out_shape)
          host.configure(layout)
          reply_ok()
        elif cmd == "step":
          assert in_h is not None and out_h is not None
          host.step_envs(payload, in_h.arr, out_h.arr)
          reply_ok()
        elif cmd == "reset":
          assert in_h is not None
          host.reset_envs(payload, in_h.arr)
          reply_ok()
        elif cmd == "stop":
          conn.send(("ok", None))
          return
        else:
          conn.send(("error", f"unknown command {cmd!r}"))
      except BaseException:
        conn.send(("error", error_payload()))
  except (EOFError, BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
    pass  # trainer went away: exit quietly
  finally:
    if in_h is not None:
      in_h.close()
    if out_h is not None:
      out_h.close()
    if capture is not None:
      capture.close()