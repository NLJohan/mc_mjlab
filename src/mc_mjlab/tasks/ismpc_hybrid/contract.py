"""Builds the `ismpc_contract` embedded in every exported `model_N.onnx`.

The C++ `PolicyRunner` reads ONLY this contract (never this repo's Python), so
everything the deployment side must reproduce is read here from the LIVE
environment at save time -- nothing is hardcoded unless it cannot be read from
the env (see TWIST_RATE_LIMITED).

Raises ContractError when the actor observation uses anything the C++
ObservationBuilder would silently disagree with (history, delay, noise, scale
!= 1, clipping). The runner then skips the export and training continues.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mc_mjlab.actions import ismpc_sine_action as act
from mc_mjlab.tasks.ismpc_hybrid import mdp

CONTRACT_VERSION = 1
ACTOR_GROUP = "actor"
ACTION_NAME = "ismpc_sine"
COMMAND_NAME = "twist"

# Observation terms whose value is low-passed INSIDE the controller. When any is in the actor group the contract
# carries a `filters` block with the cutoff period they were trained with.
FILTERED_TERMS = ("filt_perturbation", "filt_zmp_error", "filt_dcm_bias")

# IsmpcSineAction._advance_sine_period currently snaps the twist straight to
# its mapped target (no rate limit), even though cfg.twist_max_delta exists.
# That cannot be detected from the env, so it is recorded here by hand: if the
# action term ever starts applying twist_max_delta, flip this to True.
TWIST_RATE_LIMITED = False


class ContractError(RuntimeError):
  """The env cannot be described faithfully by contract version 1."""


def _plain(x: Any) -> Any:
  """Recursively convert tensors/arrays/tuples to JSON-safe python values."""
  if isinstance(x, torch.Tensor):
    return _plain(x.detach().cpu().tolist())
  if isinstance(x, np.ndarray):
    return _plain(x.tolist())
  if isinstance(x, dict):
    return {str(k): _plain(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_plain(v) for v in x]
  if isinstance(x, (np.floating, np.integer)):
    return _plain(x.item())
  if isinstance(x, float) and not math.isfinite(x):
    return None  # JSON has no inf/nan
  return x


def to_json(contract: dict, *, indent: int | None = None) -> str:
  separators = None if indent else (",", ":")
  return json.dumps(_plain(contract), indent=indent, separators=separators)


def _is_unit_scale(scale: Any) -> bool:
  if scale is None:
    return True
  if isinstance(scale, torch.Tensor):
    scale = scale.detach().cpu().numpy()
  return bool(np.allclose(np.asarray(scale, dtype=float), 1.0))


def _obs_section(env: Any) -> dict:
  om = env.observation_manager
  names = list(om.active_terms[ACTOR_GROUP])

  dims_by_group = getattr(om, "group_obs_term_dim", None)
  if dims_by_group is None or ACTOR_GROUP not in dims_by_group:
    raise ContractError(
      "observation_manager has no group_obs_term_dim for the actor group; "
      f"available dim-like attributes: {[a for a in dir(om) if 'dim' in a]}"
    )
  dims = [int(np.prod(d)) for d in dims_by_group[ACTOR_GROUP]]
  if len(dims) != len(names):
    raise ContractError(f"{len(names)} actor terms but {len(dims)} dims")

  gcfg = env.cfg.observations[ACTOR_GROUP]
  if not getattr(gcfg, "concatenate_terms", True):
    raise ContractError("actor group must have concatenate_terms=True")

  for name in names:
    cfg = om.get_term_cfg(ACTOR_GROUP, name)
    problems = []
    if not _is_unit_scale(getattr(cfg, "scale", None)):
      problems.append("scale != 1")
    if getattr(cfg, "clip", None) is not None:
      problems.append("clip")
    if getattr(cfg, "noise", None) is not None:
      problems.append("noise")
    if getattr(cfg, "history_length", 0) not in (None, 0):
      problems.append("history")
    for lag in ("delay_min_lag", "delay_max_lag"):
      if getattr(cfg, lag, 0) not in (None, 0):
        problems.append("delay")
        break
    if problems:
      raise ContractError(
        f"actor term '{name}' uses {', '.join(problems)}: the C++ builder "
        "would not reproduce it. Remove it or extend the contract."
      )

  return {
    "group": ACTOR_GROUP,
    "dim": int(sum(dims)),
    "terms": [{"name": n, "dim": d} for n, d in zip(names, dims)],
  }


def _normalizer_section(policy: Any) -> dict | None:
  """Diagnostics only (z-score monitor in C++). The ONNX already contains the
  normalizer; a failure here never blocks the export."""
  try:
    for _, mod in policy.named_modules():
      mean = getattr(mod, "_mean", getattr(mod, "mean", None))
      std = getattr(mod, "_std", getattr(mod, "std", None))
      if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        eps = getattr(mod, "eps", None)
        return {
          "mean": mean.detach().cpu().flatten().tolist(),
          "std": std.detach().cpu().flatten().tolist(),
          "eps": float(eps) if eps is not None else 0.01,
          "eps_source": "module" if eps is not None else "assumed 0.01",
        }
  except Exception as e:  # noqa: BLE001
    return {"error": f"{type(e).__name__}: {e}"}
  return None


def _action_section(env: Any, term: Any) -> dict:
  cfg = term.cfg
  dim = int(term.action_dim)
  return {
    "name": ACTION_NAME,
    "dim": dim,
    "layout": {
      "sine": [0, 4],
      "walk_gate": 4,
      "ts": 5,
      "twist": [6, 9],
    },
    "constants": {
      "offset_scale": cfg.offset_scale,
      "offset_bias": cfg.offset_bias,
      "offset_min": act.OFFSET_MIN,
      "offset_max": act.OFFSET_MAX,
      "frequency_scale": cfg.frequency_scale,
      "frequency_bias": cfg.frequency_bias,
      "frequency_min": act.FREQUENCY_MIN,
      "frequency_max": act.FREQUENCY_MAX,
      "amplitude_scale": cfg.amplitude_scale,
      "walk_gate_bias": cfg.walk_gate_bias,
      "ts_scale": cfg.ts_scale,
      "ts_bias": cfg.ts_bias,
      "ts_min": act.TS_MIN,
      "ts_max": act.TS_MAX,
      "ts_default": act.TS_DEFAULT,
      "twist_scale": list(cfg.twist_scale),
      "twist_raw_clamp": [-1.0, 1.0],
      # None = the trained action snaps to its target every latch.
      "twist_max_delta_per_latch": (
        list(cfg.twist_max_delta) if TWIST_RATE_LIMITED else None
      ),
      "twist_max_delta_cfg_unused": list(cfg.twist_max_delta),
    },
    "reset_defaults": {
      "offset": cfg.offset_bias,
      "frequency": 0.0,
      "sin_amp": 0.0,
      "cos_amp": 0.0,
      "walk": False,
      "ts": act.TS_DEFAULT,
      "twist": [0.0, 0.0, 0.0],
    },
  }


def _timing_section(env: Any, term: Any) -> tuple[float, int]:
  cfg = term.cfg
  controller_dt = float(env.step_dt * cfg.frameskip / env.cfg.decimation)
  latch = getattr(term, "_sine_param_period_ticks", None)
  if latch is None:
    latch = max(1, round(1.0 / controller_dt / cfg.sine_param_frequency_hz))
  return controller_dt, int(latch)


def _observed_joint_ids(env: Any) -> list[int]:
  """Robot joint indices of joint_pos / joint_vel, in observation order, read from the RESOLVED actor terms.

  This is the single source of truth: whatever the env cfg selected is what the contract lists. Raises when the
  two terms disagree, or when an ACTOR_EXCLUDED_JOINT_NAMES entry is not a joint of the robot (a typo would
  silently keep that joint in the observation).
  """
  om = env.observation_manager
  robot = env.scene["robot"]
  n = len(robot.joint_names)
  ids: dict[str, list[int]] = {}
  for term in ("joint_pos", "joint_vel"):
    asset_cfg = om.get_term_cfg(ACTOR_GROUP, term).params.get("asset_cfg")
    if asset_cfg is None:
      raise ContractError(f"actor term '{term}' has no asset_cfg: its joints cannot be read from the env")
    sel = asset_cfg.joint_ids
    ids[term] = list(range(n)) if isinstance(sel, slice) else [int(i) for i in sel]
  if ids["joint_pos"] != ids["joint_vel"]:
    raise ContractError("actor terms joint_pos and joint_vel observe different joints (or in a different order)")
  missing = [j for j in mdp.ACTOR_EXCLUDED_JOINT_NAMES if j not in robot.joint_names]
  if missing:
    raise ContractError(f"ACTOR_EXCLUDED_JOINT_NAMES not found on the robot: {missing}")
  leaked = [robot.joint_names[i] for i in ids["joint_pos"] if robot.joint_names[i] in mdp.ACTOR_EXCLUDED_JOINT_NAMES]
  if leaked:
    raise ContractError(f"excluded joints are still observed: {leaked}")
  return ids["joint_pos"]


def _joints_section(env: Any) -> dict:
  robot = env.scene["robot"]
  ids = _observed_joint_ids(env)
  default = robot.data.default_joint_pos[0].detach().cpu()
  return {
    "names": [robot.joint_names[i] for i in ids],
    "default_pos": default[ids].tolist(),
  }


def _command_section(env: Any) -> dict:
  ranges = {}
  cmd_cfg = env.command_manager.get_term(COMMAND_NAME).cfg.ranges
  for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    ranges[key] = list(getattr(cmd_cfg, key))
  return {"names": list(env.command_manager.active_terms), "ranges": ranges}


def _filters_section(term: Any, obs: dict) -> dict | None:
  """Cutoff period (s) of the controller-side observation filters, read from the LIVE controllers.

  None when no filtered term is observed. Envs that were just reset read 0 until their first collect, so only
  the positive entries count; they must all agree (the cutoff is a controller constant, one YAML key)."""
  if not any(t["name"] in FILTERED_TERMS for t in obs["terms"]):
    return None
  vals = term.datastore_scalar_output(act.GET_OBS_FILTER_CUTOFF_T).detach().cpu().flatten()
  pos = vals[vals > 0]
  if pos.numel() == 0:
    raise ContractError(
      "filtered observation terms are present but no controller reports its obs_filter_cutoff_T yet "
      "(all readouts are 0); the next save will retry"
    )
  if not torch.allclose(pos, pos[0].expand_as(pos)):
    raise ContractError(f"controllers disagree on obs_filter_cutoff_T: {sorted(set(pos.tolist()))}")
  return {"obs_filter_cutoff_T": float(pos[0])}


def _git_info() -> dict:
  here = str(Path(__file__).resolve().parent)

  def run(*args: str) -> str | None:
    try:
      out = subprocess.run(
        ["git", "-C", here, *args],
        capture_output=True, text=True, timeout=5, check=True,
      )
      return out.stdout.strip()
    except Exception:  # noqa: BLE001
      return None

  status = run("status", "--porcelain")
  return {"hash": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def build_contract(
  env: Any, policy: Any, *, checkpoint_path: str, iteration: int
) -> dict:
  """`env` is the unwrapped ManagerBasedRlEnv, `policy` is alg.get_policy()."""
  term = env.action_manager.get_term(ACTION_NAME)
  obs = _obs_section(env)
  action = _action_section(env, term)
  controller_dt, latch_ticks = _timing_section(env, term)
  filters = _filters_section(term, obs)
  norm = _normalizer_section(policy)
  if norm is not None and "mean" in norm:
    obs["normalizer"] = norm
  elif norm is not None:
    obs["normalizer_error"] = norm

  return {
    "contract_version": CONTRACT_VERSION,
    "checkpoint": {
      "stem": Path(checkpoint_path).stem,
      "iteration": int(iteration),
      "git": _git_info(),
    },
    "controller_dt": controller_dt,
    "latch_ticks": latch_ticks,
    # Mirrors IsmpcSineAction: counter seeded to 1 at reset; each controller
    # step checks `counter == 0` FIRST, then counter = (counter + 1) % ticks.
    # So the first latch is the ticks-th step after reset.
    "latch": {
      "counter_seed": 1,
      "due_when": "counter == 0, checked before the increment",
      "actions_used": "only the action computed on a latch step",
    },
    "obs": obs,
    **({"filters": filters} if filters is not None else {}),
    "action": action,
    "joints": _joints_section(env),
    "command": _command_section(env),
  }
