"""Builds the `ismpc_contract` embedded in every exported `model_N.onnx` (contract version 3).

The C++ `PolicyRunner` reads ONLY this contract (never this repo's Python), so everything the deployment side
must reproduce is read here from the LIVE environment at save time, except what cannot be read from the env:
the controller datastore key of each observation term (OBS_BINDINGS) and the observation delay
(PIPELINE_OBS_DELAY_TICKS).

Two stages: `_collect_parts` reads the env into a plain dict (the layout contract version 1 used), `to_v3`
turns that dict into the version 3 contract. `to_v3` is pure, so it also converts an old contract file
(restamp tool) and is what the golden test exercises without an env.

Raises ContractError when the actor observation uses anything the C++ side would silently disagree with
(history, delay, noise, scale != 1, clipping), or when the env does not match what the tables below describe.
The runner then skips the export and training continues.
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

CONTRACT_VERSION = 3
ACTOR_GROUP = "actor"
ACTION_NAME = "ismpc_sine"
COMMAND_NAME = "twist"

KEY_PREFIX = "ismpc_walking::"

# Observation delay of the training pipeline, in controller ticks (the deployed runner reads a snapshot taken
# this many end-of-step samples ago). Derived by hand from the async dispatch in mc_rtc_action_base
# (plan F1); NOT measured yet: the P1 replay decides it.
PIPELINE_OBS_DELAY_TICKS = 2

# Actor term name -> (controller datastore getter without prefix, type, dim). dim None = taken from the env
# (joint terms: the number of observed joints). Until observations are read generically through the datastore,
# this table IS the link between a training term and a controller key.
OBS_BINDINGS: dict[str, tuple[str, str, int | None]] = {
  "com_lin_vel": ("get_com_lin_vel", "vec3", 3),
  "base_ang_vel": ("get_base_ang_vel", "vec3", 3),
  "projected_gravity": ("get_projected_gravity", "vec3", 3),
  "joint_pos": ("get_joint_pos", "vecx", None),
  "joint_vel": ("get_joint_vel", "vecx", None),
  "last_sine_params": ("get_com_height_sine_params", "vecx", 4),
  "rl_phase_sincos": ("get_rl_phase_sincos", "vecx", 2),
  "com_height": ("get_com_height_est", "double", 1),
  "last_walk_action": ("get_policy_wants_walk", "double", 1),
  "last_step_timing_action": ("get_ts_target", "double", 1),
  "last_twist_action": ("get_rl_ref_vel", "vec3", 3),
  "ismpc_wants_stop": ("wants_stop_d", "double", 1),
  "filt_perturbation": ("get_filt_perturbation", "double", 1),
  "filt_zmp_error": ("get_filt_zmp_error", "double", 1),
  "filt_dcm_bias": ("get_filt_dcm_bias", "double", 1),
  "target_twist": ("get_user_ref_vel", "vec3", 3),
}
JOINT_NAMES_KEY = "get_joint_names"

# Raw action layout the output table below assumes (IsmpcSineAction).
EXPECTED_ACTION_DIM = 9
EXPECTED_LAYOUT = {"sine": [0, 4], "walk_gate": 4, "ts": 5, "twist": [6, 9]}


class ContractError(RuntimeError):
  """The env cannot be described faithfully by contract version 3."""


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


def _filters_section(term: Any) -> dict:
  """Cutoff period (s) of the controller-side observation filters, read from the LIVE controllers.

  Always required by contract v3 (the controller swaps its filter cutoff at attach). Envs that were just reset
  read 0 until their first collect, so only the positive entries count; they must all agree (the cutoff is a
  controller constant, one YAML key)."""
  vals = term.datastore_scalar_output(act.GET_OBS_FILTER_CUTOFF_T).detach().cpu().flatten()
  pos = vals[vals > 0]
  if pos.numel() == 0:
    raise ContractError(
      "no controller reports its obs_filter_cutoff_T yet (all readouts are 0); the next save will retry"
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


def _collect_parts(env: Any, policy: Any, *, checkpoint_path: str, iteration: int) -> dict:
  """Reads the live env into the plain dict `to_v3` consumes (the contract version 1 layout, without the
  latch block)."""
  term = env.action_manager.get_term(ACTION_NAME)
  obs = _obs_section(env)
  action = _action_section(env, term)
  controller_dt, period_ticks = _timing_section(env, term)
  norm = _normalizer_section(policy)
  if norm is not None and "mean" in norm:
    obs["normalizer"] = norm
  elif norm is not None:
    obs["normalizer_error"] = norm
  return {
    "checkpoint": {
      "stem": Path(checkpoint_path).stem,
      "iteration": int(iteration),
      "git": _git_info(),
    },
    "controller_dt": controller_dt,
    "latch_ticks": period_ticks,
    "obs": obs,
    "filters": _filters_section(term),
    "action": action,
    "joints": _joints_section(env),
    "command": _command_section(env),
  }


def _need(d: dict, *path: str) -> Any:
  cur: Any = d
  for i, k in enumerate(path):
    if not isinstance(cur, dict) or k not in cur:
      raise ContractError(f"missing '{'.'.join(path[: i + 1])}' in the contract parts")
    cur = cur[k]
  return cur


def _term_specs(parts: dict) -> list[dict]:
  joints = _need(parts, "joints")
  cmd = _need(parts, "command", "ranges")
  specs: list[dict] = []
  for t in _need(parts, "obs", "terms"):
    name, dim = t["name"], int(t["dim"])
    if name not in OBS_BINDINGS:
      raise ContractError(
        f"actor term '{name}' has no controller datastore key: add it to OBS_BINDINGS (or remove the term)"
      )
    getter, typ, fixed = OBS_BINDINGS[name]
    if fixed is not None and dim != fixed:
      raise ContractError(f"actor term '{name}' has dim {dim}, the controller getter {getter} gives {fixed}")
    spec: dict = {"name": name, "key": KEY_PREFIX + getter, "type": typ, "dim": dim}
    if name in ("joint_pos", "joint_vel"):
      names = list(joints["names"])
      if len(names) != dim:
        raise ContractError(f"actor term '{name}' has dim {dim} but {len(names)} observed joints")
      spec["select"] = names
      spec["labels_key"] = KEY_PREFIX + JOINT_NAMES_KEY
      if name == "joint_pos":
        spec["offset"] = [float(v) for v in joints["default_pos"]]
    elif name == "target_twist":
      spec["clamp"] = [list(cmd[k]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]
    specs.append(spec)
  return specs


def _output_specs(action: dict) -> list[dict]:
  """One entry per raw network output: y = clamp(fn(scale * raw + bias), range), written to the setter `key`."""
  if int(action["dim"]) != EXPECTED_ACTION_DIM or action["layout"] != EXPECTED_LAYOUT:
    raise ContractError(
      f"action layout {action['layout']} (dim {action['dim']}) differs from the one the output table "
      f"assumes: {EXPECTED_LAYOUT} (dim {EXPECTED_ACTION_DIM})"
    )
  c, d = action["constants"], action["reset_defaults"]
  clamp = c["twist_raw_clamp"]

  def out(name: str, raw: int, setter: str, fn: str, scale: float, bias: float, rng: list | None, init: float) -> dict:
    o: dict = {"name": name, "raw": raw, "key": KEY_PREFIX + setter, "fn": fn, "scale": scale, "bias": bias}
    if rng is not None:
      o["range"] = rng
    o["init"] = init
    return o

  outs = [
    out("com_height_offset", 0, "set_com_height_sine_offset", "id",
        c["offset_scale"], c["offset_bias"], [c["offset_min"], c["offset_max"]], d["offset"]),
    out("com_height_frequency", 1, "set_com_height_sine_frequency", "exp",
        c["frequency_scale"], c["frequency_bias"], [c["frequency_min"], c["frequency_max"]], d["frequency"]),
    out("com_height_sin_amp", 2, "set_com_height_sine_sin_amp", "id", c["amplitude_scale"], 0.0, None, d["sin_amp"]),
    out("com_height_cos_amp", 3, "set_com_height_sine_cos_amp", "id", c["amplitude_scale"], 0.0, None, d["cos_amp"]),
    # The walk decode is `raw > walk_gate_bias`, i.e. step(1 * raw + bias) with bias = -gate.
    out("walk", 4, "set_policy_wants_walk", "step", 1.0, -c["walk_gate_bias"], None, float(d["walk"])),
    out("step_timing", 5, "set_ts", "id", c["ts_scale"], c["ts_bias"], [c["ts_min"], c["ts_max"]], d["ts"]),
  ]
  for i, (axis, setter) in enumerate((("twist_x", "set_rl_ref_vel_x"), ("twist_y", "set_rl_ref_vel_y"),
                                      ("twist_yaw", "set_rl_ref_vel_z"))):
    s = c["twist_scale"][i]
    # clamp(raw, -1, 1) * s == clamp(s * raw, s * -1, s * 1) for s > 0
    if not s > 0:
      raise ContractError(f"twist_scale[{i}] must be > 0 (got {s})")
    outs.append(out(axis, 6 + i, setter, "id", s, 0.0, [s * clamp[0], s * clamp[1]], d["twist"][i]))
  return outs


def to_v3(parts: dict) -> dict:
  """Pure: the dict `_collect_parts` builds (or an old contract version 1 file) -> contract version 3."""
  terms = _term_specs(parts)
  outputs = _output_specs(_need(parts, "action"))
  c = parts["action"]["constants"]
  period_ticks = int(_need(parts, "latch_ticks"))
  if period_ticks < PIPELINE_OBS_DELAY_TICKS + 1:
    raise ContractError(f"period_ticks {period_ticks} must exceed obs_delay_ticks {PIPELINE_OBS_DELAY_TICKS}")

  obs: dict = {"terms": terms}
  norm = parts["obs"].get("normalizer")
  if norm is not None:
    obs_dim = sum(t["dim"] for t in terms)
    if len(norm["mean"]) == obs_dim and len(norm["std"]) == obs_dim:
      obs["normalizer"] = {"mean": norm["mean"], "std": norm["std"], "eps": norm["eps"]}
    # else: diagnostics only, the ONNX has the real normalizer; leave it out rather than block the export

  return {
    "contract_version": CONTRACT_VERSION,
    "checkpoint": parts["checkpoint"],
    "controller_dt": _need(parts, "controller_dt"),
    "period_ticks": period_ticks,
    "obs_delay_ticks": PIPELINE_OBS_DELAY_TICKS,
    "controller": {
      "obs_filter_cutoff_T": _need(parts, "filters", "obs_filter_cutoff_T"),
      "com_height_range": [c["offset_min"], c["offset_max"]],
    },
    "obs": obs,
    "action": {"outputs": outputs},
  }


def build_contract(
  env: Any, policy: Any, *, checkpoint_path: str, iteration: int
) -> dict:
  """`env` is the unwrapped ManagerBasedRlEnv, `policy` is alg.get_policy()."""
  parts = _collect_parts(env, policy, checkpoint_path=checkpoint_path, iteration=iteration)
  return to_v3(parts)
