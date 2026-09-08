"""Sim <-> mc_rtc input wiring.

Translates mjlab sim state into the mc_rtc input block and owns the ``IoLayout``
that describes the shared blocks. At construction it introspects the MuJoCo model
(target->dof addresses, the root free-joint, IMU/force sensor addresses), decides
the sensor routing and builds the layout; per step it fills the input block from
``entity.data``/``sim.data``. This is the mjlab-coupled counterpart to the
torch-free ``ControllerPool`` (transport) and ``ControllerHost`` (worker compute).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mc_mjlab.actions.mc_rtc_controller_host import HostMetadata, IoLayout

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def apply_reference_pd_gains(
  entity,
  ref_joint_order: Sequence[str],
  target_names: Sequence[str],
  path: str,
) -> None:
  """Set the entity's PD gains from an mc_mujoco ``PDgains_sim.dat``."""
  with open(path) as f:
    rows = [line.split() for line in f if line.strip()]
  gains = {
    name: (float(row[0]), float(row[1]))
    for name, row in zip(ref_joint_order, rows, strict=True)
    if len(row) >= 2
  }

  matched = 0
  for act in entity.actuators:
    stiffness = getattr(act, "stiffness", None)
    damping = getattr(act, "damping", None)
    if stiffness is None or damping is None:
      continue
    for j, name in enumerate(act.target_names):
      if name in gains:
        kp, kd = gains[name]
        stiffness[:, j] = kp
        damping[:, j] = kd
        matched += 1
        print(
          f"[mc_rtc] applied reference PD gains {kp=} and {kd=} from {path.split('/')[-1]} to {name}"
        )


def _actuator_gain_columns(
  entity, target_names: Sequence[str]
) -> list[tuple[int, torch.Tensor, torch.Tensor, int]]:
  """``(target index, stiffness, damping, column)`` per PD-driven target joint."""
  index_of = {name: i for i, name in enumerate(target_names)}
  columns: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
  for act in entity.actuators:
    stiffness = getattr(act, "stiffness", None)
    damping = getattr(act, "damping", None)
    if stiffness is None or damping is None:
      continue
    for j, name in enumerate(act.target_names):
      i = index_of.get(name)
      if i is not None:
        columns.append((i, stiffness, damping, j))
  return columns


def read_pd_gains(
  entity, target_names: Sequence[str], num_envs: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
  """``(kp, kd)`` copies, each ``(num_envs, num_targets)`` in target order."""
  kp = torch.zeros(num_envs, len(target_names), device=device)
  kd = torch.zeros_like(kp)
  for i, stiffness, damping, j in _actuator_gain_columns(entity, target_names):
    kp[:, i] = stiffness[:, j]
    kd[:, i] = damping[:, j]
  return kp, kd


def zero_pd_gains(entity, target_names: Sequence[str]) -> int:
  """Zero the entity's PD gains for the target joints; returns the count."""
  columns = _actuator_gain_columns(entity, target_names)
  for _, stiffness, damping, j in columns:
    stiffness[:, j] = 0.0
    damping[:, j] = 0.0
  return len(columns)


class ControllerIoBinding:
  """Owns the ``IoLayout`` and fills the mc_rtc input block from the sim.

  Resolves all model addresses and the sensor routing once at construction;
  ``fill_controller_input``/``reset_controller_input`` write the shared input
  array each step/reset.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    entity,
    target_names: Sequence[str],
    target_ids: torch.Tensor,
    metadata: HostMetadata,
    use_controller_reset: bool,
    output_channels: Sequence[str],
    has_ismpc_sine: bool = False,
    has_ismpc_velocity: bool = False,
    has_ismpc_walk_gate: bool = False,
    has_ismpc_ts: bool = False,
  ):
    self._env = env
    self._entity = entity
    self._target_names = list(target_names)
    self._target_ids_np = target_ids.cpu().numpy()
    self._device = target_ids.device
    self._output_channels = tuple(output_channels)
    self._num_targets = len(self._target_names)

    mj_model = env.sim.mj_model

    # Model joint names carry the entity prefix (e.g. "robot/RCY"); resolve each
    # target to its dof address and recover that prefix.
    dof_by_name = {
      mj_model.joint(j).name: int(mj_model.jnt_dofadr[j]) for j in range(mj_model.njnt)
    }
    self._target_dof_adr: list[int] = []
    entity_prefix = ""
    for name in self._target_names:
      match = next(
        ((n, d) for n, d in dof_by_name.items() if n == name or n.endswith("/" + name)),
        None,
      )
      if match is None:
        raise ValueError(f"Target joint '{name}' not found in the MuJoCo model.")
      self._target_dof_adr.append(match[1])
      entity_prefix = match[0].removesuffix(name)

    # Controllers expect env-local coordinates (mc_mujoco robots live at the
    # world origin); feeding offset positions confuses the observers.
    self._env_origins_np = env.scene.env_origins.cpu().numpy()

    # Free (root) joint addresses; -1 when fixed-base.
    self._root_qpos_adr = -1
    self._root_dof_adr = -1
    for j in range(mj_model.njnt):
      jnt = mj_model.joint(j)
      if jnt.type[0] == mujoco.mjtJoint.mjJNT_FREE and jnt.name.startswith(
        entity_prefix
      ):
        self._root_qpos_adr = int(mj_model.jnt_qposadr[j])
        self._root_dof_adr = int(mj_model.jnt_dofadr[j])
        break

    # Named routing (mc_mujoco parity: raw base state to "FloatingBase", IMU
    # readings to the other body sensors) needs the extended binding's
    # name-keyed setters; the singular fallback only reaches bodySensors[0].
    self._use_named_routing = metadata.has_named_setters and self._root_qpos_adr >= 0
    use_reset = use_controller_reset and metadata.has_reset

    # Sensor names follow mc_mujoco ("<sensor>_gyro") with mjlab ("imu_gyro")
    # as fallback.
    imu_sensors: list[tuple[str, int, int]] = []
    for bs_name in metadata.body_sensor_names:
      if not bs_name or bs_name == "FloatingBase":
        continue
      g_adr = self._sensor_adr(f"{bs_name}_gyro")
      if g_adr < 0:
        g_adr = self._sensor_adr("imu_gyro")
      a_adr = self._sensor_adr(f"{bs_name}_accelerometer")
      if a_adr < 0:
        a_adr = self._sensor_adr("imu_accelerometer")
      if g_adr >= 0 or a_adr >= 0:
        imu_sensors.append((bs_name, g_adr, a_adr))

    if imu_sensors:
      self._gyro_adr = imu_sensors[0][1]
      self._accel_adr = imu_sensors[0][2]
    else:
      self._gyro_adr = self._sensor_adr("imu_gyro")
      self._accel_adr = self._sensor_adr("imu_accelerometer")

    # mc_rtc force sensor "X" maps to model sensors "<prefix>X_fsensor"/"_tsensor".
    wrench_sensors: list[tuple[str, int, int]] = []
    adr_by_name = {
      mj_model.sensor(i).name: int(mj_model.sensor(i).adr[0])
      for i in range(mj_model.nsensor)
    }
    for name in metadata.force_sensor_names:
      f_adr = next(
        (a for n, a in adr_by_name.items() if n.endswith(name + "_fsensor")), None
      )
      t_adr = next(
        (a for n, a in adr_by_name.items() if n.endswith(name + "_tsensor")), None
      )
      if f_adr is not None and t_adr is not None:
        wrench_sensors.append((name, f_adr, t_adr))
    print(
      f"[mc_rtc] feeding {len(wrench_sensors)} force sensor(s): "
      f"{[n for n, _, _ in wrench_sensors]}"
    )

    self.layout = IoLayout(
      num_targets=len(self._target_names),
      named_routing=self._use_named_routing,
      has_floating_base_sensor="FloatingBase" in metadata.body_sensor_names,
      use_reset=use_reset,
      feed_accel_fallback=self._accel_adr >= 0,
      imu=tuple((n, g >= 0, a >= 0) for n, g, a in imu_sensors),
      wrenches=tuple(n for n, _, _ in wrench_sensors),
      output_channels=self._output_channels,
      has_ismpc_sine=has_ismpc_sine,
      has_ismpc_velocity=has_ismpc_velocity,
      has_ismpc_walk_gate=has_ismpc_walk_gate,
      has_ismpc_ts=has_ismpc_ts,
    )

    # Gather columns for a single fancy-indexed sensordata copy per step;
    # missing readings (adr < 0) keep their zeroed columns.
    src_cols: list[int] = []
    dst_cols: list[int] = []
    for i, (_, g_adr, a_adr) in enumerate(imu_sensors):
      off = self.layout.imu_off + 6 * i
      if g_adr >= 0:
        src_cols += [g_adr, g_adr + 1, g_adr + 2]
        dst_cols += [off, off + 1, off + 2]
      if a_adr >= 0:
        src_cols += [a_adr, a_adr + 1, a_adr + 2]
        dst_cols += [off + 3, off + 4, off + 5]
    for i, (_, f_adr, t_adr) in enumerate(wrench_sensors):
      off = self.layout.wrench_off + 6 * i
      src_cols += [f_adr, f_adr + 1, f_adr + 2, t_adr, t_adr + 1, t_adr + 2]
      dst_cols += [off, off + 1, off + 2, off + 3, off + 4, off + 5]
    self._sens_src_cols = np.array(src_cols, dtype=np.intp)
    self._sens_dst_cols = np.array(dst_cols, dtype=np.intp)

  def _sensor_adr(self, suffix: str) -> int:
    """sensordata offset of the model sensor whose name ends with ``suffix``."""
    mj_model = self._env.sim.mj_model
    for i in range(mj_model.nsensor):
      if mj_model.sensor(i).name.endswith(suffix):
        return int(mj_model.sensor(i).adr[0])
    return -1

  @staticmethod
  def _ang_vel_world_to_body(quat_wxyz: np.ndarray, omega_w: np.ndarray) -> np.ndarray:
    """Rotate world-frame angular velocities into the base (gyro) frame, batched."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    rt = np.stack(
      [
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
      ],
      axis=-1,
    ).reshape(-1, 3, 3)
    return np.einsum("nij,nj->ni", rt, omega_w)

  def fill_controller_input(self, in_np: np.ndarray) -> None:
    """Write every input column (encoders/velocities/torques, root, sensors)."""
    self._fill_joint_columns(in_np)
    self._fill_root_and_sensor_columns(in_np)

  def reset_controller_input(self, in_np: np.ndarray) -> None:
    """Write the encoder and root columns the hosts read on reset/init."""
    T = self.layout.num_targets
    ro = self.layout.root_off
    in_np[:, 0:T] = self._entity.data.joint_pos_biased.cpu().numpy()[
      :, self._target_ids_np
    ]
    if self._root_qpos_adr >= 0:
      adr = self._root_qpos_adr
      in_np[:, ro : ro + 7] = self._env.sim.data.qpos.cpu().numpy()[:, adr : adr + 7]
    else:
      in_np[:, ro : ro + 3] = self._entity.data.root_link_pos_w.cpu().numpy()
      in_np[:, ro + 3 : ro + 7] = self._entity.data.root_link_quat_w.cpu().numpy()
    in_np[:, ro : ro + 3] -= self._env_origins_np
    # Zero root velocity/acceleration on reset. This function only wrote
    # position/orientation above; the velocity columns (ro+7:ro+16, and the
    # full joint velocity block written elsewhere) are otherwise left
    # untouched -- since in_np is a persistent, reused shared-memory buffer,
    # NOT zeroing this leaves whatever fill_controller_input's
    # _fill_root_and_sensor_columns last wrote there during the PREVIOUS
    # episode's final live step (i.e. the robot's real velocity at the
    # moment it fell), carried forward verbatim into the freshly-reset
    # episode. Confirmed empirically: root_qvel read back nonzero on every
    # reset even with reset_joints_by_offset disabled and
    # default_root_state's velocity components confirmed zero at
    # construction -- this write (or lack thereof) is the actual source.
    in_np[:, ro + 7 : ro + 16] = 0.0
    # Joint velocities (T:2T) have the identical staleness bug: this
    # function writes joint POSITIONS (0:T, see above) but never touched
    # joint VELOCITIES -- _fill_joint_columns is the only other writer of
    # T:2T, and it only runs during live steps, so this block also carried
    # over the previous episode's real, pre-reset joint velocities through
    # the same reused-buffer mechanism. Confirmed by the same debug trace
    # that found the root-velocity bug: joint_qvel_maxabs was consistently
    # 0.7-2.3 rad/s at reset, far outside the declared
    # reset_joints_by_offset velocity_range of +-0.001 rad/s.
    in_np[:, T : 2 * T] = 0.0

  def read_controller_output(
    self, out_np: np.ndarray, env_indices: list[int]
  ) -> dict[str, torch.Tensor]:
    """Unpack the controller outputs for ``env_indices`` into sim-ready tensors."""
    T = self._num_targets
    rows = out_np[env_indices]
    dtype = torch.get_default_dtype()
    return {
      c: torch.tensor(rows[:, i * T : (i + 1) * T], dtype=dtype, device=self._device)
      for i, c in enumerate(self._output_channels)
    }

  def read_controller_failed(
    self, out_np: np.ndarray, env_indices: list[int]
  ) -> torch.Tensor:
    """Bool per env: did that controller's QP give up on the last step?"""
    failed = out_np[env_indices, self.layout.status_off] != 0.0
    return torch.tensor(failed, dtype=torch.bool, device=self._device)

  def write_ismpc_sine_params(
    self,
    in_np: np.ndarray,
    offset: torch.Tensor,
    frequency: torch.Tensor,
    sin_amp: torch.Tensor,
    cos_amp: torch.Tensor,
  ) -> None:
    """Write the CoM-height sine reference for every env, this step.

    ``amplitude_ratio`` is a fraction in (0, 1); the actual physical
    amplitude (``amplitude_ratio * offset``) is derived worker-side
    (``mc_rtc_controller_host.step_env``), not here -- this keeps the
    "amplitude <= offset, so height never goes negative" guarantee
    structural regardless of which caller (a scripted demo or a trained
    policy) is writing these columns.

    Requires ``self.layout.has_ismpc_sine``; callers that don't set that on
    their action's ``IoLayout`` should never reach this method.
    """
    assert self.layout.has_ismpc_sine, (
      "write_ismpc_sine_params called but this IoLayout was built with "
      "has_ismpc_sine=False -- the input row has no columns reserved for "
      "these values."
    )
    off = self.layout.ismpc_sine_off
    in_np[:, off] = offset.cpu().numpy()
    in_np[:, off + 1] = frequency.cpu().numpy()
    in_np[:, off + 2] = sin_amp.cpu().numpy()
    in_np[:, off + 3] = cos_amp.cpu().numpy()

  def write_ismpc_velocity(
    self,
    in_np: np.ndarray,
    vx: torch.Tensor,
    vy: torch.Tensor,
    wz: torch.Tensor,
  ) -> None:
    """Write the reference walking velocity for every env, this step.

    Requires ``self.layout.has_ismpc_velocity``; callers that don't set
    that on their action's ``IoLayout`` should never reach this method.
    """
    assert self.layout.has_ismpc_velocity, (
      "write_ismpc_velocity called but this IoLayout was built with "
      "has_ismpc_velocity=False -- the input row has no columns reserved "
      "for these values."
    )
    off = self.layout.ismpc_velocity_off
    in_np[:, off] = vx.cpu().numpy()
    in_np[:, off + 1] = vy.cpu().numpy()
    in_np[:, off + 2] = wz.cpu().numpy()

  def write_ismpc_walk_gate(self, in_np: np.ndarray, walk_enabled: torch.Tensor) -> None:
    """Write the RL policy's walk/stop decision for every env, this step.

    ``walk_enabled`` is a bool tensor (num_envs,): True requests walking,
    False requests stopping. Written as 1.0/0.0 -- ``step_env`` worker-side
    (``mc_rtc_controller_host.py``) thresholds at 0.5. The policy has full,
    unconditional authority: this overrides whatever ISMPC's own autonomous
    safety-stop logic would otherwise have wanted (see
    ``read_ismpc_wants_stop`` for that advisory, non-gating signal).

    Requires ``self.layout.has_ismpc_walk_gate``; callers that don't set
    that on their action's ``IoLayout`` should never reach this method.
    """
    assert self.layout.has_ismpc_walk_gate, (
      "write_ismpc_walk_gate called but this IoLayout was built with "
      "has_ismpc_walk_gate=False -- the input row has no column reserved "
      "for this value."
    )
    off = self.layout.ismpc_walk_off
    in_np[:, off] = walk_enabled.to(dtype=torch.float32).cpu().numpy()

  def write_ismpc_ts(self, in_np: np.ndarray, ts: torch.Tensor) -> None:
    """Write the RL policy's commanded step timing (Ts, s between
    footsteps) for every env, this step.

    Clamped controller-side to ``controller_config_.ts_range`` regardless
    of what's written here -- see ``Walking_controller::ts(double)``.
    Manual GUI control of Ts remains available and independent of this
    channel (see ``Walking_controller::policyControlsTs``): this write
    always lands worker-side when the layout has ``has_ismpc_ts`` set, the
    GUI checkbox only gates whether the GUI's own NumberInput may also
    write.

    Requires ``self.layout.has_ismpc_ts``; callers that don't set that on
    their action's ``IoLayout`` should never reach this method.
    """
    assert self.layout.has_ismpc_ts, (
      "write_ismpc_ts called but this IoLayout was built with "
      "has_ismpc_ts=False -- the input row has no column reserved for "
      "this value."
    )
    off = self.layout.ismpc_ts_off
    in_np[:, off] = ts.cpu().numpy()

  def read_ismpc_wants_stop(
    self, out_np: np.ndarray, env_indices: list[int]
  ) -> torch.Tensor:
    """Bool per env: did ISMPC's own safety logic want to stop walking on
    the most recent MPC solve, independent of what the policy actually
    commanded via ``write_ismpc_walk_gate``?

    This is advisory/observational only -- it does NOT reflect the
    controller's actual Stop state, which the policy has full authority
    over (see ``Walking_controller::ismpc_wants_stop`` for the full
    rationale). Intended as an observation term so the policy can learn to
    react to -- or preemptively avoid -- situations where ISMPC disagrees
    with its walk decision.

    Requires ``self.layout.has_ismpc_walk_gate``; callers that don't set
    that on their action's ``IoLayout`` should never reach this method.
    """
    assert self.layout.has_ismpc_walk_gate, (
      "read_ismpc_wants_stop called but this IoLayout was built with "
      "has_ismpc_walk_gate=False -- the output row has no column reserved "
      "for this value."
    )
    wants_stop = out_np[env_indices, self.layout.ismpc_wants_stop_off] != 0.0
    return torch.tensor(wants_stop, dtype=torch.bool, device=self._device)

  def read_is_walking(
    self, out_np: np.ndarray, env_indices: list[int]
  ) -> torch.Tensor:
    """Bool per env: is the controller ACTUALLY walking right now
    (``Walking_controller::Robot_Walking``)?

    Ground truth, distinct from both ``read_ismpc_wants_stop`` (ISMPC's own
    advisory opinion) and whatever the policy last commanded via
    ``write_ismpc_walk_gate`` (the policy's own intent) -- neither alone is
    safe to reward against, since the policy could report "walking" without
    the controller actually walking, or vice versa. Intended for reward
    terms (e.g. ``is_walking`` in mdp.py) that need to know what actually
    happened, not what was requested or advised.

    Requires ``self.layout.has_ismpc_walk_gate``; callers that don't set
    that on their action's ``IoLayout`` should never reach this method --
    same requirement as ``read_ismpc_wants_stop``, since both columns are
    only reserved together (see ``IoLayout``/``mc_rtc_controller_host.py``).
    """
    assert self.layout.has_ismpc_walk_gate, (
      "read_is_walking called but this IoLayout was built with "
      "has_ismpc_walk_gate=False -- the output row has no column reserved "
      "for this value."
    )
    is_walking = out_np[env_indices, self.layout.is_walking_off] != 0.0
    return torch.tensor(is_walking, dtype=torch.bool, device=self._device)

  def _fill_joint_columns(self, in_np: np.ndarray) -> None:
    """Write encoder/velocity/torque columns of the input block (all envs)."""
    T = self.layout.num_targets
    # Biased: these are the robot's *encoders*, so the controller's own state
    # estimate should carry the same calibration error the policy observes,
    # rather than being handed the ground truth the real one never sees.
    current_pos = self._entity.data.joint_pos_biased.cpu().numpy()
    current_vel = self._entity.data.joint_vel.cpu().numpy()
    in_np[:, 0:T] = current_pos[:, self._target_ids_np]
    in_np[:, T : 2 * T] = current_vel[:, self._target_ids_np]
    in_np[:, 2 * T : 3 * T] = self._env.sim.data.qfrc_actuator.cpu().numpy()[
      :, self._target_dof_adr
    ]

  def _fill_root_and_sensor_columns(self, in_np: np.ndarray) -> None:
    """Write the root block and the IMU/wrench columns of the input block."""
    ro = self.layout.root_off

    if self._use_named_routing:
      qadr, dadr = self._root_qpos_adr, self._root_dof_adr
      in_np[:, ro : ro + 7] = self._env.sim.data.qpos.cpu().numpy()[:, qadr : qadr + 7]
      in_np[:, ro : ro + 3] -= self._env_origins_np
      in_np[:, ro + 7 : ro + 13] = self._env.sim.data.qvel.cpu().numpy()[
        :, dadr : dadr + 6
      ]
      in_np[:, ro + 13 : ro + 16] = self._env.sim.data.qacc.cpu().numpy()[
        :, dadr : dadr + 3
      ]
    else:
      root_quat = self._entity.data.root_link_quat_w.cpu().numpy()
      in_np[:, ro : ro + 3] = (
        self._entity.data.root_link_pos_w.cpu().numpy() - self._env_origins_np
      )
      in_np[:, ro + 3 : ro + 7] = root_quat
      in_np[:, ro + 7 : ro + 10] = self._entity.data.root_link_lin_vel_w.cpu().numpy()
      # Prefer the real gyro (filled below); fall back to rotating world omega.
      if self._gyro_adr < 0:
        in_np[:, ro + 10 : ro + 13] = self._ang_vel_world_to_body(
          root_quat, self._entity.data.root_link_ang_vel_w.cpu().numpy()
        )

    if len(self._sens_src_cols) or (
      not self._use_named_routing and (self._gyro_adr >= 0 or self._accel_adr >= 0)
    ):
      sensordata = self._env.sim.data.sensordata.cpu().numpy()
      if len(self._sens_src_cols):
        in_np[:, self._sens_dst_cols] = sensordata[:, self._sens_src_cols]
      if not self._use_named_routing:
        if self._gyro_adr >= 0:
          g = self._gyro_adr
          in_np[:, ro + 10 : ro + 13] = sensordata[:, g : g + 3]
        if self._accel_adr >= 0:
          a = self._accel_adr
          in_np[:, ro + 13 : ro + 16] = sensordata[:, a : a + 3]