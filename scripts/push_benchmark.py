"""Push-robustness benchmark: random torso pushes on a walking robot, one row per trial.

Run: uv run python scripts/push_benchmark.py
Edit the constants below, no CLI args.

POLICIES is a queue: the script runs one full benchmark per entry, in list order, and writes
one CSV per entry (fresh env per policy, same SEED, so every policy sees the same push draws).

Works for any NUM_ENVS (1 to hundreds). Each env runs its own trial lifecycle:

    spawn -> walk -> push appears at a random time in PUSH_START_RANGE_S
          -> robot recovers (survives RECOVERY_TIMEOUT_S after the push start) or falls
          -> trial is appended to a timestamped CSV, one recap line is printed, env is reset
             and gets a fresh random trial until N_TRIALS have been issued and resolved.

SHOW_VIEWER=True opens the MuJoCo viewer (meant for NUM_ENVS=1 debugging, it pulls
the benchmark along in real time). The push is drawn as a pink arrow (same look as the
training push events; press the viewer's debug-vis toggle key to hide it).
"""
from __future__ import annotations

import copy
import json
import csv
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401  (populate registry)
import mc_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

# =============================== EDIT ME ===============================
TASK_ID = "Mc-Mjlab-Ismpc-Hybrid-Ismpc-Walking-Hrp5P"
ACTION_NAME = "ismpc_sine"
TORSO_BODY = "Body"

# Policies to benchmark, run one after the other in this order, one CSV each.
# Each entry is a path string or a (path, label) tuple:
#   ""              -> constant (zero) policy built from the CONST_* physical values below
#   "x/model_N.pt"  -> that checkpoint
#   "x/model_N.onnx"-> the sibling .pt with the same stem (the benchmark runs the training-side
#                      policy, not the deployed ONNX graph)
#   (path, label)   -> same, with an explicit name stored in the CSV and its file name
# Default label: "constant" or the checkpoint stem. All paths are checked before the queue starts.
POLICIES = [
  # "",
  ("/home/noahluc/workspace/mc_mjlab/logs/rsl_rl/mc_rtc_ismpc_hybrid/2026-10-04_21-05-27/model_500.pt", "model_500"),
  ("logs/rsl_rl/mc_rtc_ismpc_hybrid/2026-10-03_17-40-52/model_800.pt", "model_800"),
]
STOP_ON_ERROR = False     # False: a failing policy is reported and the queue goes on with the next one

NUM_ENVS = 400
N_TRIALS = 5000              # total trials issued; the run ends when all of them are resolved
SEED = 42
SHOW_VIEWER = False       # MuJoCo viewer (real-time pacing); for NUM_ENVS=1 debugging

TARGET_TWIST = (0.0, 0.0, 0.0)   # vx, vy, omega pinned as the command
PUSH_DURATION_S = 0.8            # fixed per run
F_MIN, F_MAX = 0., 140.0         # N, push sampled uniformly by AREA in the annulus F_MIN <= |F| <= F_MAX
                                 # (disk point picking; F_MIN=0 gives the full disk, F_MIN=F_MAX a ring)
PUSH_START_RANGE_S = (3.0, 6.0)  # push start, seconds after episode start, uniform
RECOVERY_TIMEOUT_S = 8.0         # survive this long after the push START to count as recovered
DR_TRAIN = False                 # True keeps the training mass / payload randomization

# Constant-policy physical values (what reaches the controller). Used for "" entries of POLICIES.
CONST_WALK = True
CONST_TS = 1.1
CONST_COM_OFFSET = 0.9
CONST_COM_FREQ = 1.0
CONST_COM_SIN_AMP = 0.0
CONST_COM_COS_AMP = 0.0

CSV_DIR = "logs/push_benchmark"  # one file per run: push_benchmark_<YYYYmmdd_HHMMSS>_<label>.csv

# Push arrow in the viewer (same defaults as mjlab's apply_body_impulse.VizCfg).
PUSH_ARROW_RGBA = (0.9, 0.2, 0.8, 0.9)
PUSH_ARROW_SCALE = 0.005   # meters of arrow per Newton
PUSH_ARROW_WIDTH = 0.015
PRINT_EACH_TRIAL = False   # one recap line per resolved trial (turn off for big runs)
PROGRESS_EVERY_S = 30.0   # progress line period (wall seconds); 0 disables
MAX_PREPUSH_PRINTS = 10   # print the first few pre-push failures with their cause
# ========================================================================

FAULT_TERMS = ("fell_over", "collapsed", "controller_failed")  # failure = any of these
REMOVED_EVENTS = ("push_torso", "push_right_hand", "push_left_hand")
DR_EVENTS = ("randomize_body_density", "randomize_hand_payload")


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
def _patch_mappings(term) -> None:
  """Constant policy: the action term's raw->physical maps return CONST_* values.

  Same trick as scripts/manual_play.py: only the mapping is replaced; latch cadence,
  interpolation and datastore writes run exactly as in training.
  """
  from mc_mjlab.actions import ismpc_sine_action as m

  def col(x, ref):
    return torch.full_like(ref[..., 0], float(x))

  def map_to_physical(raw):
    offset = min(max(CONST_COM_OFFSET, m.OFFSET_MIN), m.OFFSET_MAX)
    freq = min(max(CONST_COM_FREQ, m.FREQUENCY_MIN), m.FREQUENCY_MAX)
    r = (CONST_COM_SIN_AMP**2 + CONST_COM_COS_AMP**2) ** 0.5
    ratio = offset / max(r, offset)
    return {
      "offset": col(offset, raw),
      "frequency": col(freq, raw),
      "sin_amp": col(CONST_COM_SIN_AMP * ratio, raw),
      "cos_amp": col(CONST_COM_COS_AMP * ratio, raw),
    }

  def map_walk_gate(raw):
    return torch.full_like(raw, CONST_WALK, dtype=torch.bool)

  def map_step_timing(raw):
    return torch.full_like(raw, min(max(CONST_TS, m.TS_MIN), m.TS_MAX))

  def map_twist(raw):
    return (
      torch.tensor(TARGET_TWIST, device=raw.device, dtype=raw.dtype)
      .expand_as(raw)
      .clone()
    )

  term._map_to_physical = map_to_physical
  term._map_walk_gate = map_walk_gate
  term._map_step_timing = map_step_timing
  term._map_twist = map_twist


class ConstantPolicy:
  """Raw action is irrelevant (mappings are patched); zeros of the right shape."""

  def __init__(self, action_shape, device):
    self._a = torch.zeros(action_shape, device=device)

  def __call__(self, obs):
    del obs
    return self._a


def _check_contract(contract_path: Path, term, venv) -> dict | None:
  """Abort on checkpoint/env mismatch. Warn when the contract is missing."""
  if not contract_path.exists():
    print(f"[warn] no contract json at {contract_path}: skipping the consistency check")
    return None
  c = json.loads(contract_path.read_text())
  uw = venv.unwrapped
  cfg = term.cfg
  problems = []

  live_dt = uw.step_dt * cfg.frameskip / uw.cfg.decimation
  if not math.isclose(c["controller_dt"], live_dt, rel_tol=1e-9):
    problems.append(f"controller_dt: contract {c['controller_dt']} vs live {live_dt}")
  if c["latch_ticks"] != term._sine_param_period_ticks:
    problems.append(
      f"latch_ticks: contract {c['latch_ticks']} vs live {term._sine_param_period_ticks}"
    )
  live_terms = list(uw.observation_manager.active_terms["actor"])
  contract_terms = [t["name"] for t in c["obs"]["terms"]]
  if live_terms != contract_terms:
    problems.append(f"actor obs terms differ:\n  contract {contract_terms}\n  live     {live_terms}")
  live_dim = int(venv.get_observations()["actor"].shape[-1])
  if c["obs"]["dim"] != live_dim:
    problems.append(f"actor obs dim: contract {c['obs']['dim']} vs live {live_dim}")
  const = c["action"]["constants"]
  for k in (
    "offset_scale", "offset_bias", "frequency_scale", "frequency_bias",
    "amplitude_scale", "walk_gate_bias", "ts_scale", "ts_bias",
  ):
    if not math.isclose(const[k], float(getattr(cfg, k)), rel_tol=1e-6, abs_tol=1e-9):
      problems.append(f"action.{k}: contract {const[k]} vs live {getattr(cfg, k)}")
  if [float(x) for x in const["twist_scale"]] != [float(x) for x in cfg.twist_scale]:
    problems.append(f"action.twist_scale: contract {const['twist_scale']} vs live {cfg.twist_scale}")
  if problems:
    raise RuntimeError("checkpoint contract does not match the live env:\n- " + "\n- ".join(problems))

  rng = c.get("command", {}).get("ranges", {})
  for key, v in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), TARGET_TWIST):
    if key in rng and not (rng[key][0] <= v <= rng[key][1]):
      print(f"[warn] target {key}={v} is outside the checkpoint's command range {rng[key]}")
  print(f"[ok] contract check passed ({contract_path.name})")
  return c


def _parse_spec(spec) -> tuple[str, str]:
  """POLICIES entry -> (path string, user label). "" path means the constant policy."""
  if isinstance(spec, (tuple, list)):
    assert len(spec) == 2, f"POLICIES tuple entries are (path, label), got {spec!r}"
    return str(spec[0]), str(spec[1])
  return str(spec), ""


def _checkpoint_path(raw_path: str) -> Path:
  path = Path(raw_path).expanduser()
  return path.with_suffix(".pt") if path.suffix == ".onnx" else path


def validate_policies() -> None:
  """Fail fast, before hours of running: every non-constant entry must point to a file."""
  assert len(POLICIES) >= 1, "POLICIES is empty"
  missing = []
  for spec in POLICIES:
    raw, _ = _parse_spec(spec)
    if raw and not _checkpoint_path(raw).exists():
      missing.append(str(_checkpoint_path(raw)))
  if missing:
    raise FileNotFoundError("checkpoint(s) not found:\n- " + "\n- ".join(missing))


def load_policy(venv, agent_cfg, device, spec):
  """Returns (policy, label, path_str, iteration, extra_info_dict) for one POLICIES entry."""
  term = venv.unwrapped.action_manager.get_term(ACTION_NAME)
  raw_path, user_label = _parse_spec(spec)
  if not raw_path:
    _patch_mappings(term)
    policy = ConstantPolicy(venv.unwrapped.action_space.shape, venv.unwrapped.device)
    params = dict(
      walk=CONST_WALK, ts=CONST_TS, com_offset=CONST_COM_OFFSET, com_freq=CONST_COM_FREQ,
      com_sin_amp=CONST_COM_SIN_AMP, com_cos_amp=CONST_COM_COS_AMP, twist=list(TARGET_TWIST),
    )
    return policy, (user_label or "constant"), "", None, params

  path = Path(raw_path).expanduser()
  if path.suffix == ".onnx":
    pt = path.with_suffix(".pt")
    print(f"[info] {path.name} is ONNX: benchmarking the training-side checkpoint {pt.name}")
    path = pt
  if not path.exists():
    raise FileNotFoundError(f"checkpoint not found: {path}")
  contract = _check_contract(path.with_name(path.stem + ".contract.json"), term, venv)

  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(venv, asdict(agent_cfg), device=device)
  runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  iteration = None
  if contract is not None:
    iteration = contract.get("checkpoint", {}).get("iteration")
  if iteration is None:
    try:
      iteration = int(path.stem.split("_")[1])
    except (IndexError, ValueError):
      pass
  return policy, (user_label or path.stem), str(path), iteration, {}


# ----------------------------------------------------------------------------
# CSV log
# ----------------------------------------------------------------------------
TRIAL_COLS = (
  "trial_id", "env_id", "fx", "fy", "magnitude", "angle_rad",
  "push_start_s", "push_duration_s",
  "outcome",         # 'fell' | 'recovered' | 'worker_failed'
  "cause",           # termination term(s) that fired, '' if recovered
  "time_to_fall_s",  # from push start; empty unless fell (substitution is a plot-time choice)
  "recovered",       # 1 / 0, empty for worker_failed
)


class CsvLog:
  """One timestamped CSV per run. Run-level settings are repeated on every row so the
  file is self-contained. Rows are flushed per batch: Ctrl+C loses nothing."""

  def __init__(self, directory: str, label: str, run_cols: dict):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    self.path = d / f"push_benchmark_{stamp}_{safe}.csv"
    k = 2
    while self.path.exists():  # two runs in the same second with the same label
      self.path = d / f"push_benchmark_{stamp}_{safe}_{k}.csv"
      k += 1
    self._run_vals = list(run_cols.values())
    self._f = open(self.path, "w", newline="")
    self._w = csv.writer(self._f)
    self._w.writerow([*run_cols.keys(), *TRIAL_COLS])
    self._f.flush()

  def add_trials(self, rows: list[tuple]) -> None:
    for r in rows:
      self._w.writerow([*self._run_vals, *("" if v is None else v for v in r)])
    self._f.flush()

  def close(self) -> None:
    self._f.close()


# ----------------------------------------------------------------------------
# Env construction
# ----------------------------------------------------------------------------
def build_env():
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  cfg = copy.deepcopy(load_env_cfg(TASK_ID, play=True))  # registry cfgs are shared: copy
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = SEED
  cfg.auto_reset = False  # we resolve and reset envs ourselves (flags/ticks stay readable)
  cfg.actions[ACTION_NAME].console_output = "none"
  for name in REMOVED_EVENTS + (() if DR_TRAIN else DR_EVENTS):
    if name not in cfg.events:
      print(f"[warn] event {name!r} not in cfg.events")
    cfg.events.pop(name, None)
  cfg.curriculum = {}  # would overwrite command ranges at the first reset
  cmd = cfg.commands["twist"]
  cmd.ranges.lin_vel_x = (TARGET_TWIST[0], TARGET_TWIST[0])
  cmd.ranges.lin_vel_y = (TARGET_TWIST[1], TARGET_TWIST[1])
  cmd.ranges.ang_vel_z = (TARGET_TWIST[2], TARGET_TWIST[2])

  needed_s = PUSH_START_RANGE_S[1] + RECOVERY_TIMEOUT_S + 2.0
  if cfg.episode_length_s < needed_s:
    print(f"[info] episode_length_s {cfg.episode_length_s} -> {needed_s} (longest trial + margin)")
    cfg.episode_length_s = needed_s

  agent_cfg = load_rl_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  expected_events = {"reset_scene_to_default", "reset_joints"} | (set(DR_EVENTS) if DR_TRAIN else set())
  assert set(cfg.events) == expected_events, f"unexpected events: {list(cfg.events)}"
  active = set(env.termination_manager.active_terms)
  assert active == set(FAULT_TERMS) | {"time_out"}, (
    f"termination terms changed: {sorted(active)}. Decide explicitly whether the new term "
    "counts as a failure, then update FAULT_TERMS."
  )
  return venv, cfg, agent_cfg, device


# ----------------------------------------------------------------------------
# Benchmark driver
# ----------------------------------------------------------------------------
class PushBenchmark:
  """Per-env trial lifecycle on top of the wrapped env. All state is per-env tensors.

  Tick convention: t = episode_length_buf = steps completed since this env's last reset.
  Before the step taken at t == start the push wrench is written, so the push acts from
  sim time start*dt for exactly dur_ticks steps (zeroed before the step at t == end).
  """

  def __init__(self, venv, log: CsvLog, term, body_id: int):
    self.venv = venv
    self.uw = venv.unwrapped
    self.log = log
    self.term = term
    self.body_id = body_id
    self.n = self.uw.num_envs
    self.dev = self.uw.device
    self.dt = self.uw.step_dt
    self.robot = self.uw.scene["robot"]
    self.on_finished = None

    self.dur_ticks = max(1, round(PUSH_DURATION_S / self.dt))
    self.timeout_ticks = round(RECOVERY_TIMEOUT_S / self.dt)
    self.start_lo = round(PUSH_START_RANGE_S[0] / self.dt)
    self.start_hi = round(PUSH_START_RANGE_S[1] / self.dt)
    assert self.dur_ticks < self.timeout_ticks, "push must end before the recovery timeout"
    assert self.start_hi >= self.start_lo >= 1
    assert self.uw.max_episode_length > self.start_hi + self.timeout_ticks, (
      "episode length is shorter than the longest trial"
    )
    self.max_trial_ticks = self.start_hi + self.timeout_ticks
    print(
      f"[bench] step_dt {self.dt*1e3:.1f} ms | push {self.dur_ticks} ticks, start "
      f"[{self.start_lo}, {self.start_hi}] ticks, timeout {self.timeout_ticks} ticks"
    )

    self.gen = torch.Generator(device=self.dev).manual_seed(SEED)
    z = lambda dtype: torch.zeros(self.n, dtype=dtype, device=self.dev)  # noqa: E731
    self.active = z(torch.bool)
    self.trial_id = z(torch.long)
    self.fx, self.fy = z(torch.float32), z(torch.float32)
    self.start, self.end, self.deadline = z(torch.long), z(torch.long), z(torch.long)

    self.issued = 0
    self.resolved = 0
    self.n_fell = self.n_recovered = self.n_worker_failed = 0
    self.pre_push = 0
    self.anomalies = 0
    self.total_steps = 0
    self.finished = False
    self._t0 = time.perf_counter()
    self._last_progress = self._t0

    all_ids = torch.arange(self.n, device=self.dev)
    self._zero_wrench(all_ids)
    self._assign(all_ids)
    if self.issued < self.n:
      print(f"[note] N_TRIALS={N_TRIALS} < NUM_ENVS={self.n}: {self.n - self.issued} envs stay idle")

  # --- trial bookkeeping ---------------------------------------------------
  def _rand(self, n):
    return torch.rand(n, generator=self.gen, device=self.dev)

  def _draw(self, ids):
    """Fresh random push parameters for these envs (keeps their trial_id)."""
    n = len(ids)
    # Disk point picking: uniform density over the annulus F_MIN..F_MAX (full disk when
    # F_MIN = 0). The radius is the sqrt of a uniform draw over the squared radii, because
    # the area element grows with r; a uniform magnitude would crowd samples near the center.
    mag = torch.sqrt(F_MIN**2 + (F_MAX**2 - F_MIN**2) * self._rand(n))
    ang = 2.0 * math.pi * self._rand(n)
    self.fx[ids] = mag * torch.cos(ang)
    self.fy[ids] = mag * torch.sin(ang)
    start = torch.randint(self.start_lo, self.start_hi + 1, (n,), generator=self.gen, device=self.dev)
    self.start[ids] = start
    self.end[ids] = start + self.dur_ticks
    self.deadline[ids] = start + self.timeout_ticks

  def _assign(self, ids):
    """Give these envs a new trial while any remain to be issued, else idle them."""
    k = min(len(ids), N_TRIALS - self.issued)
    if k > 0:
      act = ids[:k]
      self.trial_id[act] = torch.arange(self.issued, self.issued + k, device=self.dev)
      self.active[act] = True
      self._draw(act)
      self.issued += k
    if k < len(ids):
      self.active[ids[max(k, 0):]] = False

  # --- wrench --------------------------------------------------------------
  def _write_wrench(self, ids, fx, fy):
    f = torch.zeros(len(ids), 1, 3, device=self.dev)
    f[:, 0, 0] = fx
    f[:, 0, 1] = fy
    self.robot.write_external_wrench_to_sim(
      f, torch.zeros_like(f), env_ids=ids, body_ids=[self.body_id]
    )

  def _zero_wrench(self, ids):
    z = torch.zeros(len(ids), device=self.dev)
    self._write_wrench(ids, z, z)

  # --- stepping ------------------------------------------------------------
  def step(self, actions):
    uw = self.uw
    t = uw.episode_length_buf
    on = self.active & (t == self.start)
    off = self.active & (t == self.end)
    if bool(on.any()):
      ids = on.nonzero(as_tuple=False).squeeze(-1)
      self._write_wrench(ids, self.fx[ids], self.fy[ids])
    if bool(off.any()):
      self._zero_wrench(off.nonzero(as_tuple=False).squeeze(-1))

    obs, rew, dones, extras = self.venv.step(actions)
    self.total_steps += 1

    # auto_reset=False: these buffers are still valid. Clone, the managers rewrite them.
    tm = uw.termination_manager
    t_after = uw.episode_length_buf.clone()
    done = dones.bool()
    terminated = tm.terminated.clone()
    time_outs = tm.time_outs.clone()
    flags = {nm: tm.get_term(nm).clone() for nm in FAULT_TERMS}
    worker_failed = self.term.controller_worker_failed.clone()

    a = self.active
    wfail = a & worker_failed
    rest = a & ~wfail
    anom = rest & time_outs & ~terminated                     # unexpected: time_out mid-trial
    pre = rest & terminated & (t_after <= self.start)         # fell before the push acted
    fell = rest & terminated & (t_after > self.start)
    recov = rest & ~done & (t_after >= self.deadline)
    to_reset = done | recov | wfail

    if bool(to_reset.any()):
      resolved_mask = fell | recov | wfail
      if bool(resolved_mask.any()):
        self._record(resolved_mask, fell, wfail, t_after, flags)
      redo = pre | anom
      if bool(redo.any()):
        self._note_redo(redo, pre, t_after, flags)

      ids = to_reset.nonzero(as_tuple=False).squeeze(-1)
      obs_dict, _ = uw.reset(env_ids=ids)
      self._zero_wrench(ids)  # reset should clear xfrc_applied; be explicit anyway
      obs = TensorDict(obs_dict, batch_size=[self.n])

      if bool(resolved_mask.any()):
        self._assign(resolved_mask.nonzero(as_tuple=False).squeeze(-1))
      if bool(redo.any()):
        self._draw(redo.nonzero(as_tuple=False).squeeze(-1))

    self._maybe_progress()
    if not self.finished and self.resolved >= N_TRIALS:
      self.finished = True
      if self.on_finished is not None:
        self.on_finished()
    return obs, rew, dones, extras

  def reset_all(self):
    """Full reset (viewer R key): in-flight trials are redrawn, not recorded."""
    obs_dict, _ = self.uw.reset()
    self._zero_wrench(torch.arange(self.n, device=self.dev))
    ids = self.active.nonzero(as_tuple=False).squeeze(-1)
    if len(ids):
      self._draw(ids)
    return TensorDict(obs_dict, batch_size=[self.n])

  # --- viewer: push arrow ----------------------------------------------------
  def debug_vis(self, visualizer) -> None:
    """Called by env.update_visualizers (registered in manager_visualizers in main()).

    Draws the same pink arrow as mjlab's apply_body_impulse for envs whose push is
    acting: the wrench is written before the step at t == start and zeroed before the
    step at t == end, so it is in effect for episode_length_buf in (start, end].
    """
    t = self.uw.episode_length_buf
    pushing = self.active & (t > self.start) & (t <= self.end)
    if not bool(pushing.any()):
      return
    com = self.robot.data.body_com_pos_w  # (nworld, nbody, 3)
    for env in visualizer.get_env_indices(self.n):
      if not bool(pushing[env]):
        continue
      start = com[env, self.body_id].cpu().numpy()
      force = torch.stack(
        [self.fx[env], self.fy[env], torch.zeros((), device=self.dev)]
      ).cpu().numpy()
      visualizer.add_arrow(
        start=start, end=start + force * PUSH_ARROW_SCALE,
        color=PUSH_ARROW_RGBA, width=PUSH_ARROW_WIDTH,
      )

  # --- recording / printing ------------------------------------------------
  def _record(self, mask, fell, wfail, t_after, flags):
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    g = lambda x: x[ids].cpu().tolist()  # noqa: E731
    env_l, tid, fx, fy, start = ids.tolist(), g(self.trial_id), g(self.fx), g(self.fy), g(self.start)
    ta, fell_l, wf_l = g(t_after), g(fell), g(wfail)
    flag_l = {nm: g(v) for nm, v in flags.items()}
    rows = []
    for j, env in enumerate(env_l):
      mag = math.hypot(fx[j], fy[j])
      ang = math.atan2(fy[j], fx[j])
      t_push = start[j] * self.dt
      if wf_l[j]:
        outcome, cause, ttf, rec = "worker_failed", "worker_failed", None, None
        self.n_worker_failed += 1
        tail = f"-> WORKER FAILED (trial excluded from statistics)"
      elif fell_l[j]:
        cause = "+".join(nm for nm in FAULT_TERMS if flag_l[nm][j]) or "unknown"
        ttf = (ta[j] - start[j]) * self.dt
        outcome, rec = "fell", 0
        self.n_fell += 1
        tail = f"-> FELL ({cause}) after {ttf:.2f} s"
      else:
        outcome, cause, ttf, rec = "recovered", "", None, 1
        self.n_recovered += 1
        tail = f"-> RECOVERED (survived {self.timeout_ticks * self.dt:.2f} s)"
      rows.append((
        tid[j], env, fx[j], fy[j], mag, ang, t_push, self.dur_ticks * self.dt,
        outcome, cause, ttf, rec,
      ))
      self.resolved += 1
      if PRINT_EACH_TRIAL:
        print(
          f"[{self.resolved:>4}/{N_TRIALS}] env {env:<3} push Fx={fx[j]:+6.1f} N Fy={fy[j]:+6.1f} N "
          f"(|F|={mag:5.1f} N, {math.degrees(ang):+7.1f} deg) at t={t_push:5.2f} s "
          f"for {self.dur_ticks * self.dt:.2f} s {tail}"
        )
    self.log.add_trials(rows)

  def _note_redo(self, redo, pre, t_after, flags):
    ids = redo.nonzero(as_tuple=False).squeeze(-1).tolist()
    for env in ids:
      if bool(pre[env]):
        self.pre_push += 1
        if self.pre_push <= MAX_PREPUSH_PRINTS:
          cause = "+".join(nm for nm in FAULT_TERMS if bool(flags[nm][env])) or "unknown"
          print(
            f"[warn] env {env}: terminated BEFORE its push ({cause}) at tick "
            f"{int(t_after[env])} / push tick {int(self.start[env])}; trial redrawn "
            f"(pre-push failures so far: {self.pre_push})"
          )
      else:
        self.anomalies += 1
        print(f"[warn] env {env}: time_out mid-trial (anomaly); trial redrawn")

  def _maybe_progress(self):
    if PROGRESS_EVERY_S <= 0:
      return
    now = time.perf_counter()
    if now - self._last_progress < PROGRESS_EVERY_S:
      return
    self._last_progress = now
    el = now - self._t0
    print(
      f"[progress] {self.resolved}/{N_TRIALS} resolved (fell {self.n_fell}, recovered "
      f"{self.n_recovered}) | pre-push failures {self.pre_push} | "
      f"{self.total_steps / el:.0f} steps/s | {el:.0f} s elapsed"
    )

  def summary(self) -> str:
    valid = self.n_fell + self.n_recovered
    rate = f"{100.0 * self.n_recovered / valid:.1f} %" if valid else "n/a"
    return (
      f"[summary] {self.resolved}/{N_TRIALS} trials: recovered {self.n_recovered}, fell "
      f"{self.n_fell}, worker-failed {self.n_worker_failed} | recovery rate {rate} | "
      f"pre-push failures {self.pre_push}, anomalies {self.anomalies} | "
      f"{self.total_steps} steps in {time.perf_counter() - self._t0:.0f} s"
    )


class _EnvProxy:
  """What the viewer sees as its env: step()/reset() go through the benchmark, the rest
  (cfg, unwrapped, num_envs, device, get_observations, close) is the wrapped env."""

  def __init__(self, bench: PushBenchmark, venv):
    self._bench = bench
    self._venv = venv

  def step(self, actions):
    return self._bench.step(actions)

  def reset(self):
    return self._bench.reset_all(), {}

  def __getattr__(self, name):
    return getattr(self._venv, name)


# ----------------------------------------------------------------------------
def run_one(spec) -> dict:
  """One full benchmark (fresh env, N_TRIALS trials, one CSV) for one POLICIES entry."""
  venv, cfg, agent_cfg, device = build_env()
  try:
    return _run_with_env(venv, agent_cfg, device, spec)
  finally:
    venv.close()  # also when setup (policy load, contract check) failed


def _run_with_env(venv, agent_cfg, device, spec) -> dict:
  uw = venv.unwrapped
  term = uw.action_manager.get_term(ACTION_NAME)
  body_ids, body_names = uw.scene["robot"].find_bodies(TORSO_BODY)
  assert len(body_ids) == 1, f"{TORSO_BODY!r} matched {body_names}"

  policy, label, ckpt_path, iteration, const_params = load_policy(venv, agent_cfg, device, spec)

  run_cols = dict(
    started_local=datetime.now().isoformat(timespec="seconds"),
    policy_label=label, policy_path=ckpt_path, policy_iteration=iteration,
    constant_params=json.dumps(const_params) if const_params else "",
    run_push_duration_s=PUSH_DURATION_S, twist_x=TARGET_TWIST[0], twist_y=TARGET_TWIST[1],
    twist_w=TARGET_TWIST[2], f_min=F_MIN, f_max=F_MAX, start_min_s=PUSH_START_RANGE_S[0],
    start_max_s=PUSH_START_RANGE_S[1], timeout_s=RECOVERY_TIMEOUT_S, step_dt=uw.step_dt,
    n_trials=N_TRIALS, num_envs=NUM_ENVS, seed=SEED, dr="train" if DR_TRAIN else "off",
  )
  log = CsvLog(CSV_DIR, label, run_cols)
  print(
    f"[bench] policy '{label}' | {NUM_ENVS} env(s) | {N_TRIALS} trials | "
    f"push {F_MIN:g}-{F_MAX:g} N for {PUSH_DURATION_S:g} s, start {PUSH_START_RANGE_S[0]:g}-"
    f"{PUSH_START_RANGE_S[1]:g} s | twist {TARGET_TWIST} | CSV {log.path}"
  )

  bench = PushBenchmark(venv, log, term, body_ids[0])
  # The training push events are removed, so mjlab no longer draws their arrow:
  # register the benchmark as a debug visualizer (env.update_visualizers iterates this dict).
  uw.manager_visualizers["push_benchmark"] = bench
  t_start = time.perf_counter()
  interrupted = False
  try:
    with torch.no_grad():
      if SHOW_VIEWER:
        from mjlab.viewer import NativeMujocoViewer

        proxy = _EnvProxy(bench, venv)
        viewer = NativeMujocoViewer(proxy, policy)
        bench.on_finished = lambda: setattr(viewer, "_interrupted", True)
        viewer.run()
      else:
        waves = math.ceil(N_TRIALS / NUM_ENVS) + 1
        step_cap = 4 * waves * bench.max_trial_ticks
        obs = venv.get_observations()
        while not bench.finished:
          if bench.total_steps >= step_cap:
            raise RuntimeError(
              f"step cap {step_cap} reached with {bench.resolved}/{N_TRIALS} trials resolved"
            )
          actions = policy(obs)
          obs = bench.step(actions)[0]
  except KeyboardInterrupt:
    interrupted = True
    print("\n[bench] interrupted")
  finally:
    print(bench.summary())
    print(f"[bench] finished={bench.finished}, {time.perf_counter() - t_start:.0f} s wall")
    print(f"[bench] results in {log.path}")
    log.close()
  valid = bench.n_fell + bench.n_recovered
  return dict(
    label=label, csv=str(log.path), finished=bench.finished, interrupted=interrupted,
    resolved=bench.resolved, recovery=(bench.n_recovered / valid if valid else None),
    pre_push=bench.pre_push, wall_s=time.perf_counter() - t_start,
  )


def _free_gpu() -> None:
  import gc

  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


def main():
  assert NUM_ENVS >= 1 and N_TRIALS >= 1
  validate_policies()
  if SHOW_VIEWER and len(POLICIES) > 1:
    print("[note] SHOW_VIEWER=True with several policies: each run waits for its viewer to close.")
  results = []
  t_all = time.perf_counter()
  for k, spec in enumerate(POLICIES):
    raw, user_label = _parse_spec(spec)
    name = user_label or (Path(raw).stem if raw else "constant")
    print(f"\n{'=' * 78}\n[queue] {k + 1}/{len(POLICIES)}: {name}\n{'=' * 78}")
    try:
      res = run_one(spec)
    except Exception as e:  # noqa: BLE001  (one bad policy must not kill the whole night)
      import traceback

      traceback.print_exc()
      print(f"[queue] policy '{name}' FAILED: {type(e).__name__}: {e}")
      results.append(dict(label=name, csv="", finished=False, interrupted=False, error=str(e)))
      _free_gpu()
      if STOP_ON_ERROR:
        break
      continue
    results.append(res)
    _free_gpu()
    if res["interrupted"]:
      print("[queue] Ctrl+C: the remaining policies are not run")
      break

  print(f"\n{'=' * 78}\n[queue] summary ({time.perf_counter() - t_all:.0f} s total)\n{'=' * 78}")
  for r in results:
    if "error" in r:
      status = f"FAILED ({r['error']})"
    else:
      rate = f"{100 * r['recovery']:.1f} %" if r["recovery"] is not None else "n/a"
      status = (
        f"{'done' if r['finished'] else 'INCOMPLETE'}, {r['resolved']}/{N_TRIALS} trials, "
        f"recovery {rate}, pre-push failures {r['pre_push']}, {r['wall_s']:.0f} s"
      )
    print(f"  {r['label']:<24} {status}\n    {r['csv']}")
  skipped = len(POLICIES) - len(results)
  if skipped:
    print(f"  ({skipped} policy/policies not run)")


if __name__ == "__main__":
  main()