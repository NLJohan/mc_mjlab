"""Scripted CoM-height sine demo action.

Not a learned action: the RL/agent input is ignored entirely (this is meant
to be driven by ``play --agent zero``, exactly like the zero_residual demo).
Instead, each step computes a fixed, time-based schedule --

  0s  - warmup_s:            flat (amplitude_ratio = 0, CoM_height = offset)
  warmup_s - warmup_s+sine_s: oscillating (amplitude_ratio = demo_amplitude_ratio)
  warmup_s+sine_s - episode end: flat again

-- and writes it into the shared input row's ismpc_sine columns via
``ControllerIoBinding.write_ismpc_sine_params``, so you can watch the CoM
height reference visibly change in the viewer without training anything.

Joint actuation itself is unchanged from the ordinary position-control
residual action: mc_rtc's own q/alpha output drives the joints, with the RL
residual (here always zero) added on top. This class only adds the sine-
schedule side effect on top of that existing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
  McRtcResidualJointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ScriptedIsmpcSineDemoActionCfg(McRtcResidualJointPositionActionCfg):
  """Configuration for the scripted CoM-height sine demo action."""

  demo_offset: float = 0.75
  """CoM height (m) the sine rides on, throughout the whole demo."""

  demo_amplitude_ratio: float = 0.15/0.75
  """Amplitude as a fraction of demo_offset during the oscillating phase
  (amplitude = demo_amplitude_ratio * demo_offset); 0 during the flat
  phases. Matches the smoke test's values by default."""

  demo_frequency: float = 0.5
  """Sine frequency (Hz) during the oscillating phase."""

  demo_warmup_s: float = 5.0
  """Duration (s) of the initial flat phase."""

  demo_sine_s: float = 10.0
  """Duration (s) of the oscillating phase, starting right after warmup."""

  def build(self, env: ManagerBasedRlEnv) -> "ScriptedIsmpcSineDemoAction":
    return ScriptedIsmpcSineDemoAction(self, env)


class ScriptedIsmpcSineDemoAction(McRtcResidualJointPositionAction):
  """Joint-position residual action (residual always zero in this demo) that
  additionally scripts the ISMPC CoM-height sine reference on a fixed
  schedule, so the effect is visible in a viewer session with no training.
  """

  cfg: ScriptedIsmpcSineDemoActionCfg

  # Reserves the ismpc_sine input columns on this action's IoLayout; every
  # other existing action (residual_balance, zero_residual) leaves this at
  # the base class default of False and is unaffected.
  has_ismpc_sine = True

  def apply_actions(self) -> None:
    # Schedule is purely time-based, per env, independent of any RL action
    # -- deliberately ignores self._raw_actions/self._processed_actions
    # entirely for the sine-parameter side (the joint-position residual
    # itself, applied by the base class below, still comes from whatever
    # action `play` supplies -- zero, for this demo).
    t = self._env.episode_length_buf.to(dtype=torch.get_default_dtype())
    t = t * self._env.step_dt

    warmup = self.cfg.demo_warmup_s
    sine_end = warmup + self.cfg.demo_sine_s
    in_sine_phase = (t >= warmup) & (t < sine_end)

    offset = torch.full_like(t, self.cfg.demo_offset)
    amplitude_ratio = torch.where(
      in_sine_phase,
      torch.full_like(t, self.cfg.demo_amplitude_ratio),
      torch.zeros_like(t),
    )
    frequency = torch.full_like(t, self.cfg.demo_frequency)
    # Phase left at zero: the sine's own time origin (t=0 at the start of
    # the oscillating phase) already gives a clean start; nothing in this
    # demo needs a nonzero phase offset.
    phase = torch.zeros_like(t)

    self._io.write_ismpc_sine_params(
      self._in_np, offset, amplitude_ratio, frequency, phase
    )

    # Ordinary zero-residual joint-position behavior: mc_rtc's own q/alpha
    # output drives the joints, RL residual (zero here) added on top. This
    # also owns dispatching the controller step / collecting output / the
    # interpolation ramp -- the sine params written above ride along on the
    # very same dispatched step, since it's the same IoLayout/shared block.
    super().apply_actions()
