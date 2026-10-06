"""Reset audit for the ismpc_hybrid env: is an env after reset() the same as a fresh one?

Run (from the repo root, next to push_benchmark.py in scripts/):
  uv run python scripts/reset_audit.py
Edit the constants below, no CLI args. Reuses build_env / load_policy of push_benchmark.py
(no pushes, no curriculum, command pinned, auto_reset=False, constant policy by default).

Three phases, each can be switched off:

 1. DISCOVERY / INVENTORY. Lists every value reachable from Python in the env and in the
    action term (tensors, arrays, scalars, nested objects, warp arrays) and what the native
    controller binding exposes. Written to logs/reset_audit/inventory_<stamp>.txt.
    This is the "what can we even look at" list; C++ members that the binding does not expose
    do not appear here.

 2. FAILURE STATS. Runs FAIL_EPISODES episodes per env with per-env resets (exactly what
    push_benchmark does) and records, per episode: ticks alive, outcome, which termination
    term fired, and the OUTCOME OF THE PREVIOUS EPISODE of the same env. Tells whether the
    early failures are (a) only the first episode of each env (startup), (b) any episode
    right after a reset, (c) correlated with how the previous episode ended (stale state), or
    (d) simply the policy falling at random times.

 3. REPLAY DIFF. Resets with the same torch seed REPLAY_EPISODES times and runs the same
    constant/checkpoint policy. If reset were clean the episodes would be identical. It prints
      - the keys whose value right after reset() differs between episode 0 / 1 / 2
        (anything that is not supposed to differ is a lead),
      - the first tick and the first key at which the per-tick traces diverge.
    Use NUM_ENVS = 1 for this phase (it stops at the first termination of ANY env).
"""
from __future__ import annotations

import csv
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

import push_benchmark as pb  # same folder; importing only defines things

# =============================== EDIT ME ===============================
POLICY = "/home/noahluc/workspace/mc_mjlab/logs/rsl_rl/mc_rtc_ismpc_hybrid/2026-10-04_21-05-27/model_500.pt"                 # "" -> constant policy (pb.CONST_* values), or "x/model_N.pt"
NUM_ENVS = 200               # phase 2 wants many envs, phase 3 wants 1
SEED = 42
LATCH_TICKS = 10            # controller latch period in env steps (contract latch_ticks)
RESET_AT_START = True      # True: call a full env.reset() once before the first step

RUN_INVENTORY = True
RUN_FAILURE_STATS = True
RUN_REPLAY = False          # set True (and NUM_ENVS = 1) for the replay diff

# phase 2
FAIL_EPISODES = 5          # episodes per env
FAIL_MAX_TICKS = 1200       # 6 s: a failure after this tick is not counted (push window ends ~here)
STAGGER_TICKS = 100         # each episode is cut at FAIL_MAX_TICKS + randint(0, STAGGER_TICKS) so envs
                            # are reset at DIFFERENT global dispatch phases and in small batches, like
                            # push_benchmark does. 0 = lockstep: every env resets at the same step,
                            # always at the same phase, in one batch (the first audit runs were this).

# phase 3
REPLAY_EPISODES = 3
REPLAY_TICKS = 600
REPLAY_TOL = 1e-6           # abs difference regarded as "different"
IGNORE = (                  # substrings of keys that legitimately change between episodes
  "common_step_counter", "_sim_step_counter", "recorder", "_rng", "generator", "extras",
  "episode_length_buf", "time_out",
)

OUT_DIR = "logs/reset_audit"
MAX_ELEMS = 20000           # skip arrays bigger than this (per env slice)
MAX_DEPTH = 3
# ========================================================================


# ----------------------------------------------------------------------------
# Generic state collector
# ----------------------------------------------------------------------------
_SKIP_MODULES = ("torch", "warp", "numpy", "mujoco", "builtins", "typing", "enum", "functools")
_SCALARS = (bool, int, float, np.integer, np.floating, np.bool_)


def _array(v, n: int, env: int):
  """tensor / ndarray / warp array -> numpy slice of one env (or the whole thing)."""
  try:
    if isinstance(v, torch.Tensor):
      x = v.detach()
      if x.ndim >= 1 and x.shape[0] == n:
        x = x[env]
      if x.numel() > MAX_ELEMS:
        return None
      return x.cpu().numpy().copy()
    if isinstance(v, np.ndarray):
      x = v[env] if v.ndim >= 1 and v.shape[0] == n else v
      return x.copy() if x.size <= MAX_ELEMS else None
    if hasattr(v, "numpy") and hasattr(v, "shape") and hasattr(v, "dtype"):  # warp array
      if int(np.prod(v.shape)) > 50 * MAX_ELEMS:
        return None
      x = np.asarray(v.numpy())
      x = x[env] if x.ndim >= 1 and x.shape[0] == n else x
      return x.copy() if x.size <= MAX_ELEMS else None
  except Exception:  # noqa: BLE001
    return None
  return None


def _public_props(obj):
  """For objects without __dict__ (pybind classes): non-callable public attributes."""
  out = []
  for name in dir(obj):
    if name.startswith("_"):
      continue
    try:
      v = getattr(obj, name)
    except Exception:  # noqa: BLE001
      continue
    if callable(v):
      continue
    out.append((name, v))
  return out


def _visit(v, key, out, n, env, depth, max_depth, seen):
  if v is None or isinstance(v, str):
    return
  if isinstance(v, _SCALARS):
    out[key] = np.asarray(v)
    return
  arr = _array(v, n, env)
  if arr is not None:
    out[key] = arr
    return
  if isinstance(v, dict):
    for i, (kk, vv) in enumerate(v.items()):
      if i >= 64:
        break
      _visit(vv, f"{key}[{kk}]", out, n, env, depth, max_depth, seen)
    return
  if isinstance(v, (list, tuple)):
    for i, vv in enumerate(v[:64]):
      _visit(vv, f"{key}[{i}]", out, n, env, depth, max_depth, seen)
    return
  if depth >= max_depth or id(v) in seen:
    return
  mod = type(v).__module__ or ""
  if mod.startswith(_SKIP_MODULES) or callable(v) and not hasattr(v, "__dict__"):
    return
  _walk(v, key, out, n, env, depth + 1, max_depth, seen)


def _walk(obj, prefix, out, n, env, depth, max_depth, seen):
  seen.add(id(obj))
  try:
    items = list(vars(obj).items())
  except TypeError:
    items = _public_props(obj)
  for name, v in items:
    if name.startswith("__") or name.endswith("cfg") or name == "_env":
      continue
    _visit(v, f"{prefix}.{name}", out, n, env, depth, max_depth, seen)


def collect(roots: dict, n: int, env: int = 0) -> dict[str, np.ndarray]:
  """roots: name -> (object, max_depth). Returns flat key -> numpy array of env `env`."""
  out: dict[str, np.ndarray] = {}
  seen: set[int] = set()
  for name, (obj, depth) in roots.items():
    if obj is None:
      continue
    _walk(obj, name, out, n, env, 0, depth, seen)
  return out


def roots_full(uw, term):
  rob = uw.scene["robot"]
  return {
    "term": (term, MAX_DEPTH),
    "actionmgr": (uw.action_manager, 1),
    "obsmgr": (uw.observation_manager, MAX_DEPTH),
    "eventmgr": (uw.event_manager, 2),
    "termmgr": (uw.termination_manager, 2),
    "cmdmgr": (uw.command_manager, 2),
    "rewmgr": (uw.reward_manager, 1),
    "robot_data": (getattr(rob, "data", None), 2),
    "sim_data": (getattr(uw.sim, "data", None), 1),
    "env": (uw, 0),
  }


def roots_small(uw, term):
  return {
    "term": (term, 2),
    "robot_data": (getattr(uw.scene["robot"], "data", None), 1),
    "sim_data": (getattr(uw.sim, "data", None), 1),
  }


def diff_snaps(a: dict, b: dict, tol: float) -> list[tuple[float, str, str]]:
  rows = []
  for k in sorted(set(a) | set(b)):
    if any(s in k for s in IGNORE):
      continue
    if k not in a or k not in b:
      rows.append((float("inf"), k, "only in " + ("A" if k in a else "B")))
      continue
    x, y = np.asarray(a[k], dtype=np.float64), np.asarray(b[k], dtype=np.float64)
    if x.shape != y.shape:
      rows.append((float("inf"), k, f"shape {x.shape} vs {y.shape}"))
      continue
    both_nan = np.isnan(x) & np.isnan(y)
    d = np.where(both_nan, 0.0, np.abs(x - y))
    m = float(np.nanmax(d)) if d.size else 0.0
    if m > tol or np.isnan(m):
      rows.append((m, k, f"shape {x.shape}  A={_short(x)}  B={_short(y)}"))
  rows.sort(key=lambda r: -r[0])
  return rows


def _short(x: np.ndarray) -> str:
  f = x.reshape(-1)
  s = np.array2string(f[:4], precision=5, suppress_small=True)
  return s + ("..." if f.size > 4 else "")


# ----------------------------------------------------------------------------
# Phase 1: inventory / discovery
# ----------------------------------------------------------------------------
def inventory(uw, term, stamp: str):
  Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
  path = Path(OUT_DIR) / f"inventory_{stamp}.txt"
  snap = collect(roots_full(uw, term), uw.num_envs, 0)
  lines = [f"# {len(snap)} numeric values reachable from Python (env 0 slice)"]
  for k in sorted(snap):
    a = snap[k]
    lines.append(f"{k}\tshape={tuple(a.shape)}\t{a.dtype}\t{_short(a)}")

  lines += ["", "# action term: type and public attributes"]
  lines.append(f"type(term) = {type(term)}")
  for name in sorted(n for n in dir(term) if not n.startswith("__")):
    try:
      v = getattr(term, name)
    except Exception as e:  # noqa: BLE001
      v = f"<error {e}>"
    kind = "method" if callable(v) else type(v).__name__
    lines.append(f"  {name}\t{kind}")

  man = getattr(term, "_manager", None)
  lines += ["", f"# native manager object term._manager: {type(man)}"]
  if man is not None:
    for name in sorted(n for n in dir(man) if not n.startswith("__")):
      lines.append(f"  {name}")

  lines += ["", "# loaded modules that look like the mc_rtc binding"]
  for modname in sorted(sys.modules):
    if "mc_rtc" in modname.lower() and not modname.startswith("mc_mjlab.tasks"):
      mod = sys.modules[modname]
      pub = [n for n in dir(mod) if not n.startswith("_")]
      lines.append(f"  {modname}: {pub[:60]}")
  path.write_text("\n".join(lines) + "\n")
  print(f"[inventory] {len(snap)} Python-visible numeric values, written to {path}")
  groups = Counter(k.split(".")[0] for k in snap)
  print("[inventory] by root:", dict(groups))
  return snap


# ----------------------------------------------------------------------------
# Phase 2: failure statistics per episode index
# ----------------------------------------------------------------------------
def failure_stats(venv, uw, policy, term, stamp: str):
  n = uw.num_envs
  dev = uw.device
  ep = torch.zeros(n, dtype=torch.long, device=dev)
  prev_outcome = ["start"] * n
  prev_ticks = [0] * n
  # term._substep is ONE host int shared by all envs and not reset per env: the phase (mod the
  # latch period) at which an env is reset decides how long its controller waits for the first
  # dispatch. Recorded per episode (-1: attribute not found).
  phase_now = lambda: int(getattr(term, "_substep", -1)) % LATCH_TICKS if hasattr(term, "_substep") else -1  # noqa: E731
  start_phase = [phase_now()] * n
  start_batch = [n] * n
  cap_t = FAIL_MAX_TICKS + torch.randint(0, STAGGER_TICKS + 1, (n,), device=dev)
  rows = []
  obs = venv.get_observations()
  tm = uw.termination_manager
  step_cap = (FAIL_EPISODES + 2) * FAIL_MAX_TICKS * 2
  steps = 0
  with torch.no_grad():
    while bool((ep < FAIL_EPISODES).any()):
      steps += 1
      if steps > step_cap:
        print("[failure] step cap reached, stopping")
        break
      obs, _rew, dones, _ex = venv.step(policy(obs))
      t = uw.episode_length_buf.clone()
      terminated = tm.terminated.clone()
      flags = {nm: tm.get_term(nm).clone() for nm in pb.FAULT_TERMS}
      wf = term.controller_worker_failed.clone()
      cap = t >= cap_t
      to_reset = dones.bool() | cap | wf
      if not bool(to_reset.any()):
        continue
      ids = to_reset.nonzero(as_tuple=False).squeeze(-1)
      for e in ids.tolist():
        if int(ep[e]) >= FAIL_EPISODES:
          continue
        if bool(wf[e]):
          outcome, cause = "worker_failed", "worker_failed"
        elif bool(terminated[e]) and int(t[e]) > FAIL_MAX_TICKS:
          outcome, cause = "survived", "late_fail"  # fell after the window of interest
        elif bool(terminated[e]):
          outcome = "fail"
          cause = "+".join(nm for nm in pb.FAULT_TERMS if bool(flags[nm][e])) or "?"
        else:
          outcome, cause = "survived", ""
        rows.append(dict(
          env=e, episode=int(ep[e]), ticks=int(t[e]), outcome=outcome, cause=cause,
          prev_outcome=prev_outcome[e], prev_ticks=prev_ticks[e], reset_phase=start_phase[e],
          reset_batch=start_batch[e],
        ))
        prev_outcome[e], prev_ticks[e] = outcome, int(t[e])
        ep[e] += 1
      ph = phase_now()
      for e in ids.tolist():
        start_phase[e] = ph
        start_batch[e] = len(ids)
      cap_t[ids] = FAIL_MAX_TICKS + torch.randint(0, STAGGER_TICKS + 1, (len(ids),), device=dev)
      obs_dict, _ = uw.reset(env_ids=ids)
      obs = TensorDict(obs_dict, batch_size=[n])

  Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
  path = Path(OUT_DIR) / f"episodes_{stamp}.csv"
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  print(f"[failure] {len(rows)} episodes, per-episode rows in {path}")
  _print_failure_tables(rows)


def _print_failure_tables(rows):
  def is_fail(r):
    return r["outcome"] in ("fail", "worker_failed")

  def rate(rs):
    k = sum(is_fail(r) for r in rs)
    return f"{k:4d}/{len(rs):<4d} = {100 * k / max(len(rs), 1):5.1f} % fail"

  print(f"\n-- failure within {FAIL_MAX_TICKS} ticks ({FAIL_MAX_TICKS * 0.005:.1f} s) by episode index")
  by_ep = defaultdict(list)
  for r in rows:
    by_ep[r["episode"]].append(r)
  for k in sorted(by_ep):
    fails = [r["ticks"] for r in by_ep[k] if is_fail(r)]
    med = f", median fail tick {statistics.median(fails):.0f}" if fails else ""
    print(f"  episode {k:2d}: {rate(by_ep[k])}{med}")

  print("\n-- by how the PREVIOUS episode of the same env ended")
  by_prev = defaultdict(list)
  for r in rows:
    by_prev[r["prev_outcome"]].append(r)
  for k in sorted(by_prev):
    print(f"  after {k:<14}: {rate(by_prev[k])}")

  ep0 = Counter(r["cause"] or r["outcome"] for r in rows if r["episode"] == 0)
  print(f"\n-- how episode 0 ended: {dict(ep0)}")
  print("   median ticks of episode-0 failures:",
        statistics.median([r["ticks"] for r in rows if r["episode"] == 0 and is_fail(r)] or [0]))

  print("\n-- episodes >= 1 by dispatch phase (term._substep % latch) at the reset that started them")
  by_ph = defaultdict(list)
  for r in rows:
    if r["episode"] >= 1:
      by_ph[r["reset_phase"]].append(r)
  for k in sorted(by_ph):
    print(f"  phase {k:3d}: {rate(by_ph[k])}")

  print("\n-- episodes >= 1 by size of the reset batch that started them (envs reset together)")
  buckets = [(1, 1), (2, 4), (5, 16), (17, 10**9)]
  for lo, hi in buckets:
    rs = [r for r in rows if r["episode"] >= 1 and lo <= r["reset_batch"] <= hi]
    if rs:
      print(f"  batch {lo}-{hi if hi < 10**9 else 'inf'}: {rate(rs)}")

  print("\n-- failure tick histogram (episodes >= 1 only), by cause")
  edges = [0, 5, 20, 100, 400, 800, FAIL_MAX_TICKS + 1]
  hist = defaultdict(Counter)
  for r in rows:
    if r["episode"] >= 1 and is_fail(r):
      b = max(i for i, e in enumerate(edges[:-1]) if r["ticks"] >= e)
      hist[r["cause"]][b] += 1
  header = "  ".join(f"[{edges[i]},{edges[i + 1]})" for i in range(len(edges) - 1))
  print(f"  {'cause':<28}{header}")
  for cause, c in sorted(hist.items(), key=lambda kv: -sum(kv[1].values())):
    print(f"  {cause:<28}" + "  ".join(f"{c[i]:>{len(f'[{edges[i]},{edges[i + 1]})')}}" for i in range(len(edges) - 1)))
  print(
    "\nReading guide: failures only in episode 0 -> startup transient. Failures at tiny ticks in\n"
    "every episode -> reset/initialization problem. Failure rate depending on the previous\n"
    "outcome -> stale state carried across the reset. Failures spread over hundreds of ticks\n"
    "and independent of the above -> the policy genuinely falls."
  )


# ----------------------------------------------------------------------------
# Phase 3: replay diff
# ----------------------------------------------------------------------------
def replay(venv, uw, policy, term):
  n = uw.num_envs
  snaps, traces = [], []
  with torch.no_grad():
    for r in range(REPLAY_EPISODES):
      torch.manual_seed(SEED)  # identical samples in the reset events
      obs_dict, _ = uw.reset()
      obs = TensorDict(obs_dict, batch_size=[n])
      snaps.append(collect(roots_full(uw, term), n, 0))
      trace = []
      for _k in range(REPLAY_TICKS):
        obs, _rew, dones, _ex = venv.step(policy(obs))
        snap = collect(roots_small(uw, term), n, 0)
        snap["obs.actor"] = obs["actor"][0].detach().cpu().numpy().copy()
        trace.append(snap)
        if bool(dones.any()):
          break
      traces.append(trace)
      print(f"[replay] episode {r}: ran {len(trace)} ticks"
            f"{' (terminated)' if len(trace) < REPLAY_TICKS else ''}")

  print(f"\n== values right after reset() that differ between episodes (tol {REPLAY_TOL:g})")
  for a, b in [(0, 1), (1, 2)]:
    if b >= len(snaps):
      continue
    rows = diff_snaps(snaps[a], snaps[b], REPLAY_TOL)
    print(f"\n-- episode {a} vs {b}: {len(rows)} differing keys (top 40)")
    for m, k, info in rows[:40]:
      print(f"  {m:12.4g}  {k}  {info}")

  print("\n== first tick at which the per-tick traces diverge")
  for a, b in [(0, 1), (1, 2)]:
    if b >= len(traces):
      continue
    ta, tb = traces[a], traces[b]
    first: dict[str, tuple[int, float]] = {}
    for k in range(min(len(ta), len(tb))):
      for key in set(ta[k]) & set(tb[k]):
        if key in first or any(s in key for s in IGNORE):
          continue
        x, y = np.asarray(ta[k][key], np.float64), np.asarray(tb[k][key], np.float64)
        if x.shape != y.shape:
          continue
        d = float(np.nanmax(np.abs(x - y))) if x.size else 0.0
        if d > REPLAY_TOL:
          first[key] = (k + 1, d)
    print(f"\n-- episode {a} vs {b} (lengths {len(ta)} / {len(tb)}): {len(first)} keys diverge")
    for key, (k, d) in sorted(first.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[:30]:
      print(f"  tick {k:4d}  max|diff| {d:10.4g}  {key}")
  print(
    "\nIf nothing differs and the lengths match, reset() is clean for this seed/policy and the\n"
    "failures come from elsewhere. The earliest keys in the divergence list are the leads."
  )


# ----------------------------------------------------------------------------
def main():
  pb.NUM_ENVS = NUM_ENVS
  pb.SEED = SEED
  pb.SHOW_VIEWER = False
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  torch.manual_seed(SEED)
  venv, _cfg, agent_cfg, device = pb.build_env()
  try:
    uw = venv.unwrapped
    term = uw.action_manager.get_term(pb.ACTION_NAME)
    policy, label, _path, _it, _extra = pb.load_policy(venv, agent_cfg, device, POLICY)
    print(f"[audit] policy '{label}', {NUM_ENVS} env(s), RESET_AT_START={RESET_AT_START}")
    if RESET_AT_START:
      uw.reset()
    t0 = time.perf_counter()
    if RUN_INVENTORY:
      inventory(uw, term, stamp)
    if RUN_FAILURE_STATS:
      failure_stats(venv, uw, policy, term, stamp)
    if RUN_REPLAY:
      replay(venv, uw, policy, term)
    print(f"[audit] done in {time.perf_counter() - t0:.0f} s")
  finally:
    venv.close()


if __name__ == "__main__":
  main()