"""Generic base for action terms backed by per-env mc_rtc controllers.

This is deliberately independent of ``mc_rtc_residual_action.py``. It is not
an extraction from ``McRtcResidualActionBase`` and does not import from it;
the two files own separate copies of the same plumbing pattern so that
residual behavior can never be affected by anything written here. Subclasses
of ``McRtcActionBase`` have no residual concept at all (no scale/offset/clip,
no hardware-bound projection, no recovery authority, no printer) — the
subclass owns the entire meaning of its action vector via
``process_actions``/``apply_actions``, which stay abstract-in-spirit here
(not declared abstract, since even their signatures are subclass-specific).
"""

from __future__ import annotations

import abc
import os
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

import mc_rtc_interface as native
from mc_mjlab.bridge.config import get_controller_name
from mc_mjlab.bridge.controller_datastore import (
  input_columns,
  output_columns,
  read_outputs,
  write_inputs,
)
from mc_mjlab.bridge.shared_memory import ShmHandle, create_shm, row_window
from mc_mjlab.bridge.sim_controller_bridge import SimControllerBridge
from mc_mjlab.robots.pd_gains import apply_reference_pd_gains

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class McRtcActionCfg(BaseActionCfg):
  """Shared configuration for mc_rtc action terms with no residual concept."""

  mc_rtc_config_path: str
  """Path to the mc_rtc configuration file."""

  mc_rtc_robot_name: str = "jvrc1"
  """Name of the robot in mc_rtc."""

  required_controller: str | None = None
  """Enabled controller this term needs; ``None`` runs on any. docs/coupling.md"""

  frameskip: int = 1
  """Physics substeps per controller step (e.g. 5ms control / 1ms physics -> 5)."""

  num_workers: int | None = None
  """Worker process count; ``None`` = ``min(num_envs, cpu_count - 2)``."""

  pd_gains_path: str | None = None
  """Optional reference-order mc_mujoco PD gains overriding entity defaults."""

  datastore_scalar_inputs: tuple[str, ...] = ()
  """Native double setters written every control period. docs/coupling.md"""

  datastore_scalar_outputs: tuple[str, ...] = ()
  """Native double getter callbacks collected after each controller step."""

  datastore_vectors_inputs: tuple[str, ...] = ()
  """Native Vector3d setters written every control period. docs/coupling.md"""

  datastore_vectors_outputs: tuple[str, ...] = ()
  """Native Vector3d getters collected each period, without interpolation."""

  controller_timeout_ms: int = 60000
  """Native collection timeout, in milliseconds."""

  console_output: Literal["none", "single", "all"] = "none"
  """Native row logging: none, environment zero (single), or all."""


class McRtcActionBase(BaseAction):
  """mc_rtc action base: steps per-env controllers via a native manager pool.

  Owns only mc_rtc-controller plumbing: manager lifecycle, dispatch/collect,
  datastore transport, reset-race handling and teardown. Carries no residual
  concept whatsoever — subclasses decide entirely what their action vector
  means and how it drives the controller.
  """

  cfg: McRtcActionCfg

  output_channels: tuple[str, ...] = ()
  """Public joint output channels; native qd is exposed as alpha."""

  def __init__(self, cfg: McRtcActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg=cfg, env=env)

    self._mc_rtc_robot_name = cfg.mc_rtc_robot_name
    self._num_targets = len(self._target_names)
    # Every env shares one dispatch phase (nothing resets this per env), so a
    # host int keeps the substep loop free of device synchronisation.
    self._substep = 0

    self._validate_cfg(cfg)

    self._build_bridge(cfg)

    self._manager = None
    self._finalizer = None
    self._input_memory = self._output_memory = None
    self._in_np = self._out_np = None
    self._pending_dispatch = False
    self._pending_reset = np.zeros(self.num_envs, dtype=bool)
    self._dispatch_resets = np.zeros(self.num_envs, dtype=bool)

    try:
      self._start_controllers(cfg)
      self._finish_initialization(env, cfg)
    except BaseException:
      self.close()
      raise

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear generic per-episode state and queue a native controller reset."""
    super().reset(env_ids=env_ids)

    self._collect_controller_output()

    if env_ids is None:
      env_ids = slice(None)

    self._torque_peak[env_ids] = 0.0

    if isinstance(env_ids, slice):
      env_indices = list(range(self.num_envs))[env_ids]
    else:
      env_indices = env_ids.tolist()

    self._manager.respawn(env_indices)
    self._pending_reset[env_indices] = True

    # Seed interpolation (subclass-specific rest value per channel) and discard
    # any staged output for the reset envs; they restart from that seed.
    rows = torch.tensor(env_indices, device=self.device, dtype=torch.long)
    self._seed_interpolation(rows)
    self._has_staged_control[rows] = False

    for readouts in (self._datastore_vector_outputs, self._datastore_scalar_outputs):
      for values in readouts.values():
        values[rows] = 0.0
    self._datastore_output_fresh[rows] = False

    # Episode latches clear now; the queued native reset still has to complete.
    self.controller_failed[rows] = False
    self.controller_worker_failed[rows] = False

  def close(self) -> None:
    """Stop the manager before releasing either shared-memory block."""
    if self._finalizer is not None:
      self._finalizer()
      self._finalizer = None
      self._input_memory = self._output_memory = None
    elif self._manager is not None:
      self._manager.close()

    self._manager = None
    self._bridge.release_views()
    self._in_np = self._out_np = None

    for name in ("_input_memory", "_output_memory"):
      memory = getattr(self, name, None)
      if memory is not None:
        memory.unlink()
        setattr(self, name, None)

  def controller_reference(self, channel: str) -> torch.Tensor:
    """Latest raw controller output for ``channel``."""
    return self._next_control[channel]

  def datastore_scalar_output(self, getter: str) -> torch.Tensor:
    """Latest scalar datastore getter value collected from each controller."""
    try:
      return self._datastore_scalar_outputs[getter]
    except KeyError:
      raise KeyError(
        f"datastore scalar output {getter!r} is not configured; add it to the "
        f"action term's `datastore_scalar_outputs` "
        f"(have: {sorted(self._datastore_scalar_outputs)})"
      ) from None

  def datastore_vector_output(self, getter: str) -> torch.Tensor:
    """Latest ``(num_envs, 3)`` value of one collected vector datastore getter."""
    try:
      return self._datastore_vector_outputs[getter]
    except KeyError:
      raise KeyError(
        f"datastore vector output {getter!r} is not collected; add it to the "
        f"action term's `datastore_vectors_outputs` "
        f"(have: {sorted(self._datastore_vector_outputs)})"
      ) from None

  def datastore_scalar_input(self, setter: str) -> torch.Tensor:
    """Value currently fed to one configured scalar datastore setter."""
    index = self._datastore_input_index(
      self._datastore_scalar_input_columns, setter, "scalar"
    )
    return self._datastore_scalar_input_feed[:, index]

  def datastore_vector_input(self, setter: str) -> torch.Tensor:
    """Value currently fed to one configured vector datastore setter."""
    index = self._datastore_input_index(
      self._datastore_vector_input_columns, setter, "vectors"
    )
    return self._datastore_vector_input_feed[:, index]

  def set_datastore_scalar_input(self, setter: str, values: torch.Tensor) -> None:
    """Feed one scalar datastore setter; the value holds until set again."""
    index = self._datastore_input_index(
      self._datastore_scalar_input_columns, setter, "scalar"
    )
    if tuple(values.shape) != (self.num_envs,):
      raise ValueError(
        f"datastore scalar input shape {tuple(values.shape)}, "
        f"expected {(self.num_envs,)}"
      )
    self._datastore_scalar_input_feed[:, index].copy_(values)

  def set_datastore_vector_input(self, setter: str, values: torch.Tensor) -> None:
    """Feed one vector datastore setter; the value holds until set again."""
    index = self._datastore_input_index(
      self._datastore_vector_input_columns, setter, "vectors"
    )
    if tuple(values.shape) != (self.num_envs, 3):
      raise ValueError(
        f"datastore vector input shape {tuple(values.shape)}, "
        f"expected {(self.num_envs, 3)}"
      )
    self._datastore_vector_input_feed[:, index].copy_(values)

  def consume_torque_peak(self) -> torch.Tensor:
    """Peak |joint torque| since the last call, over the target joints; resets it."""
    peak = self._torque_peak.clone()
    self._torque_peak.zero_()
    return peak

  def _advance_control_period(self) -> None:
    """Collect the finished solve, roll the ramp endpoints and dispatch the next."""
    assert self._manager is not None
    assert self._in_np is not None and self._out_np is not None

    # Collect the previous period's dispatch (it solved while the intervening
    # sim substeps ran) before reusing the shared I/O blocks.
    self._collect_controller_output()

    # Masked, not indexed: a boolean gather's data-dependent shape would sync.
    fresh = self._has_staged_control.unsqueeze(-1)
    for channel in self.output_channels:
      scratch = self._swap_scratch[channel]
      torch.where(
        fresh, self._next_control[channel], self._previous_control[channel], out=scratch
      )
      self._previous_control[channel].copy_(scratch)
      torch.where(
        fresh, self._staged_control[channel], self._next_control[channel], out=scratch
      )
      self._next_control[channel].copy_(scratch)
    self._has_staged_control.zero_()

    # Sample the current state and dispatch this period's solve without
    # blocking; it overlaps the next `frameskip` substeps of sim.
    self._bridge.fill_controller_input(self._in_np)
    write_inputs(
      self._in_np,
      self._datastore_scalar_input_columns,
      self._datastore_scalar_input_feed,
      "scalar",
    )
    write_inputs(
      self._in_np,
      self._datastore_vector_input_columns,
      self._datastore_vector_input_feed,
      "vector3",
    )

    self._dispatch_resets[:] = self._pending_reset
    self._in_np[:, self._bridge.layout.input.reset_offset()] = self._dispatch_resets
    # An unserviced row must never look like a fresh successful result.
    self._out_np[:, self._bridge.layout.output.status_offset()] = int(
      native.OutputLayout.Status.WORKER_FAILED
    )
    self._manager.dispatch(native.Command.Step)
    self._pending_dispatch = True

  def _collect_controller_output(self) -> None:
    """Await the outstanding async step (if any) and stage its outputs."""
    if not self._pending_dispatch:
      return
    assert self._manager is not None and self._out_np is not None

    failed = self._manager.collect()
    self._pending_dispatch = False

    # Merge the worker failures into the block itself, so the single upload
    # below carries the final status and no mask has to cross separately.
    status_column = self._out_np[:, self._bridge.layout.output.status_offset()]
    status_column[failed] = int(native.OutputLayout.Status.WORKER_FAILED)
    status = status_column.copy()

    worker_failed = status == int(native.OutputLayout.Status.WORKER_FAILED)
    self._pending_reset[self._dispatch_resets & ~worker_failed] = False
    self._pending_reset[worker_failed] = True

    block = self._bridge.upload_controller_output(self._out_np)
    status_t = block[:, self._bridge.layout.output.status_offset()]
    ok = status_t == int(native.OutputLayout.Status.OK)
    self.controller_failed |= status_t == int(native.OutputLayout.Status.QP_FAILED)
    self.controller_worker_failed |= status_t == int(
      native.OutputLayout.Status.WORKER_FAILED
    )

    # Every status is OK, QP_FAILED or WORKER_FAILED: fresh output means OK.
    self._has_staged_control.copy_(ok)
    fresh = ok.unsqueeze(-1)
    for channel, values in self._bridge.read_controller_output(block).items():
      staged = self._staged_control[channel]
      staged.copy_(torch.where(fresh, values, staged))

    self._latch_datastore_outputs(block, ok)

  @abc.abstractmethod
  def _seed_interpolation(self, env_ids: torch.Tensor) -> None:
    """Seed the interpolation endpoints for the given (reset) envs."""
    raise NotImplementedError

  @abc.abstractmethod
  def _apply_control(
    self, interpolated_control: dict[str, torch.Tensor]
  ) -> None:
    """Write targets to the controller for the current interpolated command."""
    raise NotImplementedError

  def _validate_cfg(self, cfg: McRtcActionCfg) -> None:
    """Reject configurations the shared-memory pipeline cannot honour."""
    if cfg.frameskip <= 0 or self._env.cfg.decimation % cfg.frameskip:
      raise ValueError("environment decimation must be divisible by positive frameskip")

    if cfg.controller_timeout_ms < 0:
      raise ValueError("controller_timeout_ms must be nonnegative")

    if cfg.console_output not in ("none", "single", "all"):
      raise ValueError(f"invalid console_output: {cfg.console_output!r}")

    # Before any worker starts: a term built on another controller's calls would
    # otherwise surface as a native init failure. docs/coupling.md
    if cfg.required_controller is not None:
      enabled = get_controller_name(Path(cfg.mc_rtc_config_path))
      if enabled != cfg.required_controller:
        raise ValueError(
          f"{type(self).__name__} requires the {cfg.required_controller!r} "
          f"controller, but {cfg.mc_rtc_config_path} enables {enabled!r}. "
          f"If {enabled!r} is compatible with the current configuration, set the action cfg's "
          "`required_controller` to it (or None to accept any controller)."
        )

  def _build_bridge(self, cfg: McRtcActionCfg) -> None:
    """Bind the simulation bridge, declare its datastore columns and load PD gains."""
    self._bridge = SimControllerBridge(
      self._env,
      self._entity,
      self._target_names,
      self._target_ids,
      cfg.mc_rtc_robot_name,
      self.output_channels,
      cfg.entity_name,
    )

    self._bridge.layout.output.datastore_scalar = list(
      dict.fromkeys(cfg.datastore_scalar_outputs)
    )

    self._bridge.layout.output.datastore_vector3 = list(
      dict.fromkeys(cfg.datastore_vectors_outputs)
    )

    # Declared before the command pairs, which append their setters after these.
    self._bridge.layout.input.datastore_scalar = list(
      dict.fromkeys(cfg.datastore_scalar_inputs)
    )

    self._bridge.layout.input.datastore_vector3 = list(
      dict.fromkeys(cfg.datastore_vectors_inputs)
    )

    self._setup_datastore_outputs(cfg)
    self._setup_datastore_inputs(cfg)

    if cfg.pd_gains_path is not None:
      apply_reference_pd_gains(
        self._entity,
        self._bridge.layout.input.joint_order,
        self._target_names,
        cfg.pd_gains_path,
      )

  def _start_controllers(self, cfg: McRtcActionCfg) -> None:
    """Map the shared-memory blocks, start the worker pool and initialize every row."""
    # Both sizes count the datastore columns declared above; keep this after them.
    self._input_memory = create_shm((self.num_envs, self._bridge.layout.input_size))
    self._output_memory = create_shm((self.num_envs, self._bridge.layout.output_size))
    self._in_np = self._input_memory.arr
    self._out_np = self._output_memory.arr
    self._bridge.fill_controller_input(self._in_np)

    configuration = native.WorkerStartMessage(
      self._bridge.layout,
      native.SharedMemoryDescription(*row_window(self._input_memory, 0, self.num_envs)),
      native.SharedMemoryDescription(
        *row_window(self._output_memory, 0, self.num_envs)
      ),
    )

    workers = (
      cfg.num_workers
      if cfg.num_workers is not None
      else min(self.num_envs, max(1, (os.cpu_count() or 1) - 2))
    )

    self._manager = native.ControllersManager(
      cfg.mc_rtc_config_path,
      self.num_envs,
      workers,
      configuration,
      cfg.controller_timeout_ms,
    )

    self._finalizer = weakref.finalize(
      self,
      self._release_controller,
      self._manager,
      self._input_memory,
      self._output_memory,
    )

    self._manager.dispatch(native.Command.Initialize)
    failed = self._manager.collect()
    status = self._out_np[:, self._bridge.layout.output.status_offset()]

    if failed or np.any(status != int(native.OutputLayout.Status.OK)):
      raise RuntimeError(
        "native controller initialization failed; verify configured numeric "
        "datastore callbacks and the mc_mjlab controller adapter "
        f"(vectors={list(self._bridge.layout.output.datastore_vector3)}, "
        f"scalars={list(self._bridge.layout.output.datastore_scalar)})"
      )

    self._latch_datastore_outputs(
      self._bridge.upload_controller_output(self._out_np),
      torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
    )

  def _finish_initialization(self, env: ManagerBasedRlEnv, cfg: McRtcActionCfg) -> None:
    """Allocate simulation buffers while native resources remain guarded."""
    self._alloc_interpolation_buffers()
    self._alloc_failure_latches()
    self._alloc_torque_peak()

  def _alloc_torque_peak(self) -> None:
    """Allocate the per-episode peak |joint torque| buffer over target joints."""
    self._torque_peak = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )

  def _setup_datastore_outputs(self, cfg: McRtcActionCfg) -> None:
    """Resolve the collected getter columns and their latched readouts."""
    self._datastore_vector_output_columns = output_columns(
      self._bridge.layout, cfg.datastore_vectors_outputs, "vector3"
    )
    self._datastore_scalar_output_columns = output_columns(
      self._bridge.layout, cfg.datastore_scalar_outputs, "scalar"
    )

    # Whole-controller vectors: latched as collected, no ramp (see the cfg).
    self._datastore_vector_outputs = {
      getter: torch.zeros(self.num_envs, 3, device=self.device)
      for getter in self._datastore_vector_output_columns
    }
    self._datastore_scalar_outputs = {
      getter: torch.zeros(self.num_envs, device=self.device)
      for getter in self._datastore_scalar_output_columns
    }
    # Read by a term feeding a setter relative to its getter: a reset zeroes the
    # readouts, so the value only means something once this says so.
    self._datastore_output_fresh = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _setup_datastore_inputs(self, cfg: McRtcActionCfg) -> None:
    """Resolve the unconditionally fed setter columns and their value buffers."""
    self._datastore_scalar_input_columns = input_columns(
      self._bridge.layout, dict.fromkeys(cfg.datastore_scalar_inputs), "scalar"
    )
    self._datastore_vector_input_columns = input_columns(
      self._bridge.layout, dict.fromkeys(cfg.datastore_vectors_inputs), "vector3"
    )

    # Every declared setter is written each period from the first step on, so a
    # task that declares one owns its value from then on. docs/coupling.md
    self._datastore_scalar_input_feed = torch.zeros(
      self.num_envs, len(self._datastore_scalar_input_columns), device=self.device
    )
    self._datastore_vector_input_feed = torch.zeros(
      self.num_envs, len(self._datastore_vector_input_columns), 3, device=self.device
    )

  def _alloc_interpolation_buffers(self) -> None:
    """Per-channel ramp endpoints plus the one-period-behind staging buffer."""
    self._previous_control = self._zero_channels()
    self._next_control = self._zero_channels()
    self._staged_control = self._zero_channels()
    # Reused every substep: allocating the blend and the masked swap in the loop
    # would churn the caching allocator for no reason.
    self._interpolated = self._zero_channels()
    self._swap_scratch = self._zero_channels()

    self._has_staged_control = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _zero_channels(self) -> dict[str, torch.Tensor]:
    """One zeroed ``(num_envs, num_targets)`` buffer per output channel."""
    return {
      channel: torch.zeros(self.num_envs, self._num_targets, device=self.device)
      for channel in self.output_channels
    }

  def _alloc_failure_latches(self) -> None:
    """Allocate the per-episode controller and worker failure flags."""
    # Latched per env until reset; read by the `controller_failed` termination
    # term so a QP giving up ends that episode instead of the whole run.
    self.controller_failed = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Apart from the QP latch: losing a worker is exogenous, so the task truncates.
    self.controller_worker_failed = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _latch_datastore_outputs(self, block: torch.Tensor, ok: torch.Tensor) -> None:
    """Latch the collected getters of every environment the block is fresh for."""
    self._datastore_output_fresh.copy_(ok)
    for kind, columns, destination in (
      (
        "vector3",
        self._datastore_vector_output_columns,
        self._datastore_vector_outputs,
      ),
      ("scalar", self._datastore_scalar_output_columns, self._datastore_scalar_outputs),
    ):
      mask = ok.unsqueeze(-1) if kind == "vector3" else ok
      for name, value in read_outputs(block, columns, kind).items():
        destination[name].copy_(torch.where(mask, value, destination[name]))

  def _datastore_input_index(
    self, columns: dict[str, int], setter: str, kind: str
  ) -> int:
    """Position of one configured setter in its input value buffer."""
    try:
      return list(columns).index(setter)
    except ValueError:
      raise KeyError(
        f"datastore {kind} input {setter!r} is not configured; add it to the "
        f"action term's `datastore_{kind}_inputs` (have: {sorted(columns)})"
      ) from None

  @staticmethod
  def _release_controller(
    manager: native.ControllersManager,
    input_memory: ShmHandle,
    output_memory: ShmHandle,
  ) -> None:
    """Release workers before the memory they may still be accessing."""
    manager.close()
    input_memory.unlink()
    output_memory.unlink()