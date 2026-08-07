"""Placeholder reward/termination functions for the ismpc_hybrid skeleton.

FOO STAGE: these exist only to make the task well-formed and trainable end
to end (rsl_rl needs a real reward signal and at least one termination
condition to be a sane RL problem). None of this is ISMPC-specific -- no
CoM-height tracking error, no footstep/QP-quality term, nothing that uses
the bridge's get_com_height_ref/qp_succeeded getters yet.

ASSUMED, NOT VERIFIED: the exact function signature mjlab's
RewardTermCfg/TerminationTermCfg expect (`func(env, **params) -> Tensor`).
Modeled on the pattern implied by residual_balance's usage, not confirmed
against mjlab's manager source directly -- check this against a real
manager_base.py/reward_manager.py read if this errors on first run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Constant +1 per env, per step: reward for simply not having terminated
  yet. FOO: the entire "stay alive" pressure in this placeholder reward
  comes from this term plus the termination conditions below -- there is no
  shaping term at all yet."""
  return torch.ones(env.num_envs, device=env.device)


def upright_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize deviation from upright, via the entity's projected gravity
  vector (parallels residual_balance's orientation shaping, simplified).

  FOO: entity name "robot" hardcoded; no configurable params dict yet.
  """
  entity = env.scene["robot"]
  gravity_b = entity.data.projected_gravity_b
  # Projected gravity is (0, 0, -1) when upright; penalize the horizontal
  # component's magnitude (any tilt shows up there).
  return torch.sum(gravity_b[:, :2] ** 2, dim=-1)


def fell_over(env: ManagerBasedRlEnv, gravity_xy_threshold: float = 0.7) -> torch.Tensor:
  """True once the entity's projected gravity's horizontal component
  exceeds a threshold -- i.e. the robot has tipped over substantially.

  FOO: threshold untuned; copied in spirit from residual_balance's
  fell_over term without seeing its exact numeric threshold.
  """
  entity = env.scene["robot"]
  gravity_b = entity.data.projected_gravity_b
  return torch.sum(gravity_b[:, :2] ** 2, dim=-1) > gravity_xy_threshold**2


def collapsed(env: ManagerBasedRlEnv, min_root_height: float = 0.5) -> torch.Tensor:
  """True once the entity's root height drops below a threshold.

  FOO: threshold untuned (JVRC1-specific standing height is ~0.8m per this
  session's earlier smoke test baseline of 0.75-0.82m CoM height; 0.5m as a
  collapse threshold is a rough guess, not derived from the robot's actual
  geometry).
  """
  entity = env.scene["robot"]
  root_height = entity.data.root_link_pos_w[:, 2]
  return root_height < min_root_height


def last_sine_params(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Current period's physical sine params (offset, amplitude_ratio,
  frequency, phase), as an observation.

  Exposing this (rather than only the raw pre-mapping action mjlab's
  built-in `last_action` observation would give) lets the policy condition
  on physically meaningful quantities directly, and -- per the phase-
  continuity discussion this design pass -- gives it the information it
  needs to choose a new phase that continues smoothly from where the
  previous period's sine left off, rather than having to reconstruct that
  from raw, pre-sigmoid/pre-clamp numbers.
  """
  action_term = env.action_manager.get_term(action_name)
  return action_term.physical_params


def last_walk_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """The policy's current walk/stop decision (1.0 = walk, 0.0 = stop), as
  an observation -- mirrors last_sine_params' role for the sine params, so
  the policy can condition on its own last decision directly.
  """
  action_term = env.action_manager.get_term(action_name)
  return action_term.last_walk_action


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


def sine_continuity_penalty(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Penalize a discontinuous CoM-height reference at period boundaries.

  Delegates to IsmpcSineAction.continuity_penalty() -- see that method's
  docstring for why this compares the two sines' value/slope at the splice
  instant, rather than penalizing raw parameter change directly.
  """
  action_term = env.action_manager.get_term(action_name)
  return action_term.continuity_penalty()


def joint_torque_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sum of squared joint torques (actuator effort), summed over joints.

  Legitimate on ordinary robotics/control grounds (reduces actuator heat,
  peak torque demand, and is a standard RL shaping term) -- NOT included on
  the basis of any specific human-biomechanics claim. A literature check
  during this design pass (Gard et al., "Metabolic and Mechanical Energy
  Costs of Reducing Vertical Center of Mass Movement During Gait") found
  the opposite of the initially-proposed justification: deviating from the
  natural range of CoM vertical displacement, in either direction,
  *increases* metabolic cost in humans; it is not a strategy the body uses
  to *reduce* energy expenditure. Kept here purely as a standard control
  shaping term.

  FOO: uses ALL dofs via sim.data.qfrc_actuator, not scoped to the specific
  target joint subset the action drives; untuned weight. Confirmed source
  (not guessed): ControllerIoBinding._fill_joint_columns already reads
  env.sim.data.qfrc_actuator for this exact purpose (feeding mc_rtc's torque
  input channel), so this is known to be the right attribute name -- only
  the dof-subset scoping here is a placeholder.
  """
  return torch.sum(env.sim.data.qfrc_actuator**2, dim=-1)