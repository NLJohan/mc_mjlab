"""Actor joint observations read the controller's encoder getters (reference joint order), not MuJoCo."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab.actions import ismpc_sine_action as act
from mc_mjlab.actions.ismpc_sine_action import IsmpcSineAction
from mc_mjlab.tasks.ismpc_hybrid import mdp

# The controller lists its joints as c, x, a, b; the entity as a, b, c (x is not simulated).
REF_ORDER = ("c", "x", "a", "b")
ENTITY_JOINTS = ("a", "b", "c")


def _scene(joint_ids):
  q = torch.tensor([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]])
  term = object.__new__(IsmpcSineAction)
  term._env = NS(device="cpu", num_envs=2)  # ty: ignore[invalid-assignment]
  term._bridge = NS(layout=NS(input=NS(joint_order=list(REF_ORDER))))  # ty: ignore[invalid-assignment]
  term._datastore_vectorx_outputs = {act.GET_JOINT_POS: q, act.GET_JOINT_VEL: 0.1 * q}
  robot = NS(name="robot", joint_names=ENTITY_JOINTS)
  env = NS(scene={"robot": robot}, action_manager=NS(get_term=lambda name: term))
  return env, NS(name="robot", joint_ids=joint_ids)


def test_selection_follows_the_observation_order_and_the_reference_columns() -> None:
  env, cfg = _scene([2, 0])  # observed: c, then a -> reference columns 0 and 2
  pos = mdp.estimated_joint_pos(env, "ismpc_sine", cfg)  # ty: ignore[invalid-argument-type]
  vel = mdp.estimated_joint_vel(env, "ismpc_sine", cfg)  # ty: ignore[invalid-argument-type]
  # position and velocity: raw controller values of the observed joints (c, then a), no offset
  torch.testing.assert_close(pos, torch.tensor([[10.0, 12.0], [20.0, 22.0]]))
  torch.testing.assert_close(vel, torch.tensor([[1.0, 1.2], [2.0, 2.2]]))


def test_every_joint_selection_is_read_in_entity_order() -> None:
  env, cfg = _scene(slice(None))  # a, b, c -> reference columns 2, 3, 0
  vel = mdp.estimated_joint_vel(env, "ismpc_sine", cfg)  # ty: ignore[invalid-argument-type]
  torch.testing.assert_close(vel, torch.tensor([[1.2, 1.3, 1.0], [2.2, 2.3, 2.0]]))


def test_a_joint_the_controller_does_not_list_is_refused() -> None:
  env, cfg = _scene([0])
  env.scene["robot"].joint_names = ("zz", "b", "c")
  with pytest.raises(ValueError, match="zz"):
    mdp.estimated_joint_pos(env, "ismpc_sine", cfg)  # ty: ignore[invalid-argument-type]
