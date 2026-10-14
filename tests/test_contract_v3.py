"""Golden test of the contract writer (contract version 3).

`to_v3` is fed the contract the previous exporter wrote for model_900 (fixtures/contract_v1_model900.json, a real
output of the version 1 code) and must give the controller repo's golden file (fixtures/contract_v3_model0.json,
a copy of ismpc_walking/tests/fixtures/contract_v3_model0.json: keep the two equal). Checkpoint info and the
normalizer values legitimately differ between the two models, so only their shape is compared.

No env, no GPU: contract.py is loaded from the tracked file `*/contract.py`.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"


def _load_contract_module():
  root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()
  tracked = subprocess.check_output(["git", "ls-files", "*contract.py"], cwd=root, text=True).split()
  found = [p for p in tracked if Path(p).name == "contract.py"]
  assert len(found) == 1, f"expected exactly one tracked contract.py, found {found}"
  spec = importlib.util.spec_from_file_location("contract_under_test", Path(root) / found[0])
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


contract = _load_contract_module()


def _load(name: str) -> dict:
  return json.loads((FIX / name).read_text())


def _assert_equal(got, want, path="contract"):
  if isinstance(want, dict):
    assert isinstance(got, dict), f"{path}: expected an object"
    assert set(got) == set(want), f"{path}: keys differ: {sorted(set(got) ^ set(want))}"
    for k in want:
      _assert_equal(got[k], want[k], f"{path}.{k}")
  elif isinstance(want, list):
    assert isinstance(got, list) and len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
    for i, (g, w) in enumerate(zip(got, want)):
      _assert_equal(g, w, f"{path}[{i}]")
  elif isinstance(want, (int, float)) and not isinstance(want, bool):
    assert got == pytest.approx(want, rel=1e-9, abs=1e-12), f"{path}: {got} != {want}"
  else:
    assert got == want, f"{path}: {got!r} != {want!r}"


def _strip_run_specific(c: dict) -> dict:
  c = copy.deepcopy(c)
  c.pop("checkpoint")
  n = c["obs"].pop("normalizer")
  assert len(n["mean"]) == len(n["std"]) == sum(t["dim"] for t in c["obs"]["terms"])
  assert n["eps"] > 0
  return c


def test_v1_output_of_current_exporter_converts_to_the_golden_v3():
  got = contract.to_v3(_load("contract_v1_model900.json"))
  _assert_equal(_strip_run_specific(got), _strip_run_specific(_load("contract_v3_model0.json")))


def test_checkpoint_and_normalizer_are_carried_over():
  v1 = _load("contract_v1_model900.json")
  got = contract.to_v3(v1)
  assert got["checkpoint"] == v1["checkpoint"]
  assert got["obs"]["normalizer"] == {
    "mean": v1["obs"]["normalizer"]["mean"],
    "std": v1["obs"]["normalizer"]["std"],
    "eps": v1["obs"]["normalizer"]["eps"],
  }


def test_serialises_to_plain_json():
  text = contract.to_json(contract.to_v3(_load("contract_v1_model900.json")))
  assert json.loads(text)["contract_version"] == 3


def test_unbound_actor_term_is_refused():
  v1 = _load("contract_v1_model900.json")
  v1["obs"]["terms"].append({"name": "mystery_term", "dim": 2})
  with pytest.raises(contract.ContractError, match="mystery_term"):
    contract.to_v3(v1)


def test_wrong_dim_for_a_fixed_getter_is_refused():
  v1 = _load("contract_v1_model900.json")
  v1["obs"]["terms"][0]["dim"] = 4  # com_lin_vel is a vec3
  with pytest.raises(contract.ContractError, match="com_lin_vel"):
    contract.to_v3(v1)


def test_changed_action_layout_is_refused():
  v1 = _load("contract_v1_model900.json")
  v1["action"]["layout"]["ts"] = 6
  with pytest.raises(contract.ContractError, match="layout"):
    contract.to_v3(v1)


def test_missing_filter_cutoff_is_refused():
  v1 = _load("contract_v1_model900.json")
  del v1["filters"]
  with pytest.raises(contract.ContractError, match="filters"):
    contract.to_v3(v1)


def test_period_must_exceed_the_observation_delay():
  v1 = _load("contract_v1_model900.json")
  v1["latch_ticks"] = contract.PIPELINE_OBS_DELAY_TICKS
  with pytest.raises(contract.ContractError, match="period_ticks"):
    contract.to_v3(v1)


def test_normalizer_of_the_wrong_size_is_left_out_not_fatal():
  v1 = _load("contract_v1_model900.json")
  v1["obs"]["normalizer"]["mean"] = v1["obs"]["normalizer"]["mean"][:-1]
  assert "normalizer" not in contract.to_v3(v1)["obs"]
