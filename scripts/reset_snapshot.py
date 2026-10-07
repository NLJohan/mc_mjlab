"""Record what the sim and the native controller look like around every episode reset.

Run exactly like `uv run play`, but through this file:

    uv run python scripts/reset_snapshot.py

Apply your torso force with the mouse as usual. Every reset of ENV_IDX writes one
file `reset_NNNN.npz` under OUT_DIR, plus `timeline.npz` (one row per env step)
when the viewer closes. Note which resets looked weird, then compare them with
`scripts/reset_snapshot_diff.py`.

What is recorded, per env step of ENV_IDX (all read after the step returns):
  sim_*   mjlab/MuJoCo-Warp data row (root pose/vel/acc, applied force, ...)
  ctl_*   native controller input/output rows, status, reset flags
  ds_*    controller datastore outputs read back by the action term, with the
          `fresh` flag (False right after a reset until the new controller answers)
  act_*   Python-side state of the action term (sine params, Ts, twist, ...)

Per reset file:
  pre_step__*   the state just BEFORE mjlab resets the env (end of the old episode)
  post_reset__* the state right AFTER mjlab's reset events + action-manager reset
  before__*     the last N_STEPS_BEFORE rows up to and including the terminating step
  after__*      the next N_STEPS_AFTER rows (cut short if another reset comes first)

Timing: for a reset that happens inside env.step, before__[-1] is the terminating
step but its sim_* values are already POST-reset (mjlab resets inside the step); the
true end-of-episode state is pre_step__*. For a reset triggered outside a step (the viewer's own start-up/reset key),
before__[-1] is the first step after that reset. `rec_dispatch_reset_ordinal` counts
controller dispatches that carried the reset flag; the first one takes the
`m_controller->reset` path in ControllerInstance::reset, later ones the soft init() path.
Recording ~40 small GPU reads per step can make play slower than real time.
"""

from __future__ import annotations

import atexit
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

# --- What to run (same as your `uv run play` command line). ---
TASK_ID = "Mc-Mjlab-Ismpc-Hybrid-Ismpc-Walking-Hrp5P"
CHECKPOINT_FILE = "wandb/run-20261006_195721-eljotywc/files/model_500.pt"
VIEWER = "native"
NUM_ENVS = 1  # None keeps the task's play default; ENV_IDX must be the viewer's env.

# --- What to record. ---
ENV_IDX = 0
ACTION_NAME = "ismpc_sine"
N_STEPS_BEFORE = 60  # env steps kept before each reset (5 ms each)
N_STEPS_AFTER = 200  # env steps recorded after each reset
OUT_ROOT = Path("reset_snapshots")
TIMELINE_FLUSH_EVERY = 5000  # env steps between timeline.npz rewrites

# Fields of the MuJoCo-Warp data row stored whole; the root slices are stored every step.
SIM_FULL_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "qacc_warmstart",
  "ctrl",
  "act",
  "qfrc_applied",
  "xfrc_applied",
  "qfrc_actuator",
  "time",
)
ROOT_QPOS = slice(0, 7)  # free joint: xyz + wxyz
ROOT_DOF = slice(0, 6)


def _np(x) -> np.ndarray:
  """Copy a torch tensor / numpy array / scalar into a fresh numpy array."""
  if hasattr(x, "detach"):
    x = x.detach().cpu().numpy()
  return np.array(x, copy=True)


def _pitch(quat_wxyz: np.ndarray) -> float:
  """Rotation about body-frame y of the root; + means leaning towards body +x."""
  w, x, y, z = quat_wxyz
  return float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))


class Recorder:
  def __init__(self) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.out_dir = OUT_ROOT / stamp
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.step_count = 0
    self.reset_count = 0  # mjlab resets of ENV_IDX seen
    self.dispatch_reset_count = 0  # steps whose controller dispatch carried reset=1
    self.ring: deque[dict] = deque(maxlen=N_STEPS_BEFORE)
    self.timeline: list[dict] = []
    self.pending_reset_flag = False
    self.pre_step_snap: dict | None = None
    self.post_reset_snap: dict | None = None
    self.window: dict | None = None
    self.missing: set[str] = set()
    self.warned_decimation = False
    atexit.register(self.close)
    print(f"[reset_snapshot] writing to {self.out_dir}")

  # ---- snapshot helpers. ----

  def _sim_field(self, env, name: str):
    try:
      return _np(getattr(env.sim.data, name)[ENV_IDX])
    except Exception:
      if name not in self.missing:
        self.missing.add(name)
        print(f"[reset_snapshot] sim.data.{name} unavailable; skipped")
      return None

  def snapshot(self, env, term, full: bool) -> dict[str, np.ndarray]:
    """One row of everything we record. `full` adds the whole-vector fields."""
    row: dict[str, np.ndarray] = {}
    sim = {n: self._sim_field(env, n) for n in SIM_FULL_FIELDS}

    qpos, qvel, qacc, qfa = sim["qpos"], sim["qvel"], sim["qacc"], sim["qfrc_applied"]
    if qpos is not None:
      row["sim_root_qpos"] = qpos[ROOT_QPOS]
      row["sim_root_pitch"] = np.array([_pitch(qpos[3:7])])
    if qvel is not None:
      row["sim_root_qvel"] = qvel[ROOT_DOF]
    if qacc is not None:
      row["sim_root_qacc"] = qacc[ROOT_DOF]
    if qfa is not None:
      row["sim_qfrc_applied_norm"] = np.array([np.linalg.norm(qfa)])
    xfa = sim["xfrc_applied"]
    if xfa is not None:
      row["sim_xfrc_applied_absmax"] = np.array([np.abs(xfa).max()])
    if sim["time"] is not None:
      row["sim_time"] = np.atleast_1d(sim["time"])
    if full:
      for name, value in sim.items():
        if value is not None:
          row[f"sim_full_{name}"] = value

    row["env_episode_length_buf"] = np.array([int(env.episode_length_buf[ENV_IDX])])

    # Controller side: shared-memory rows and the action term's own state.
    layout = term._bridge.layout
    # Only the INPUT row is read: the output row can be mid-write by a worker while
    # a dispatch is in flight, so collected outputs are read from the staged buffers.
    in_row = term._in_np[ENV_IDX]
    row["ctl_in_reset_flag"] = np.array([in_row[layout.input.reset_offset()]])
    row["ctl_pending_reset"] = np.array([bool(term._pending_reset[ENV_IDX])])
    row["ctl_dispatch_resets"] = np.array([bool(term._dispatch_resets[ENV_IDX])])
    row["ctl_failed"] = np.array([bool(term.controller_failed[ENV_IDX])])
    row["ctl_worker_failed"] = np.array([bool(term.controller_worker_failed[ENV_IDX])])
    row["ctl_pending_dispatch"] = np.array([bool(term._pending_dispatch)])
    row["ctl_has_staged"] = np.array([bool(term._has_staged_control[ENV_IDX])])

    row["ds_fresh"] = np.array([bool(term._datastore_output_fresh[ENV_IDX])])
    for name, values in term._datastore_scalar_outputs.items():
      row[f"ds_s__{name}"] = np.array([float(values[ENV_IDX])])
    for name, values in term._datastore_vector_outputs.items():
      row[f"ds_v__{name}"] = _np(values[ENV_IDX])

    for name, values in term._physical_curr.items():
      row[f"act_physical_{name}"] = np.array([float(values[ENV_IDX])])
    row["act_ts_curr"] = np.array([float(term._ts_curr[ENV_IDX])])
    row["act_walk_enabled"] = np.array([bool(term._walk_enabled[ENV_IDX])])
    row["act_twist_curr"] = _np(term._twist_curr[ENV_IDX])
    row["act_sine_ticks"] = np.array([int(term._dispatch_ticks_since_sine_update[ENV_IDX])])
    row["act_substep"] = np.array([int(term._substep)])

    if full:
      row["ctl_in_row"] = _np(in_row)
      for channel in term.output_channels:
        row[f"act_staged_{channel}"] = _np(term._staged_control[channel][ENV_IDX])
        row[f"act_prev_{channel}"] = _np(term._previous_control[channel][ENV_IDX])
        row[f"act_next_{channel}"] = _np(term._next_control[channel][ENV_IDX])
    return row

  # ---- hook bodies. ----

  def term(self, env):
    return env.action_manager.get_term(ACTION_NAME)

  def before_reset_idx(self, env, env_ids) -> bool:
    if env_ids is None:
      hit = True
    else:
      hit = ENV_IDX in _np(env_ids).reshape(-1).tolist()
    if hit:
      self.pre_step_snap = self.snapshot(env, self.term(env), full=True)
    return hit

  def after_reset_idx(self, env, hit: bool) -> None:
    if not hit:
      return
    self.reset_count += 1
    self.pending_reset_flag = True
    self.post_reset_snap = self.snapshot(env, self.term(env), full=True)

  def after_step(self, env) -> None:
    term = self.term(env)
    if not self.warned_decimation and env.cfg.decimation != term.cfg.frameskip:
      self.warned_decimation = True
      print("[reset_snapshot] WARNING: decimation != frameskip; `ctl_*` rows are "
            "the last dispatch of each env step only")
    row = self.snapshot(env, term, full=True)
    row["rec_step"] = np.array([self.step_count])
    row["rec_reset_ordinal"] = np.array([self.reset_count])
    row["rec_reset_this_step"] = np.array([self.pending_reset_flag])
    if bool(row["ctl_in_reset_flag"][0]):
      self.dispatch_reset_count += 1
    row["rec_dispatch_reset_ordinal"] = np.array([self.dispatch_reset_count])
    self.step_count += 1

    # Timeline keeps only the light fields.
    self.timeline.append({k: v for k, v in row.items() if not k.startswith(("sim_full_", "ctl_in_row", "act_prev_", "act_next_", "act_staged_"))})
    if self.step_count % TIMELINE_FLUSH_EVERY == 0:
      self.flush_timeline()

    if self.window is not None:
      self.window["rows"].append(row)
      if len(self.window["rows"]) >= N_STEPS_AFTER:
        self.finish_window(truncated=False)

    self.ring.append(row)

    if self.pending_reset_flag:
      self.pending_reset_flag = False
      if self.window is not None:  # next reset came before the window filled
        self.finish_window(truncated=True)
      self.window = {
        "ordinal": self.reset_count,
        "step": self.step_count - 1,
        "before": list(self.ring),
        "pre_step": self.pre_step_snap,
        "post_reset": self.post_reset_snap,
        "rows": [],
      }

  # ---- output. ----

  @staticmethod
  def _stack(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not rows:
      return out
    for key in rows[0]:
      try:
        out[key] = np.stack([r[key] for r in rows])
      except (KeyError, ValueError):
        pass
    return out

  def finish_window(self, truncated: bool) -> None:
    win, self.window = self.window, None
    if win is None:
      return
    data: dict[str, np.ndarray] = {}
    for prefix, rows in (("before", win["before"]), ("after", win["rows"])):
      for key, value in self._stack(rows).items():
        data[f"{prefix}__{key}"] = value
    for prefix in ("pre_step", "post_reset"):
      for key, value in (win[prefix] or {}).items():
        data[f"{prefix}__{key}"] = value
    meta = {
      "reset_ordinal": win["ordinal"],
      "step": win["step"],
      "after_steps": len(win["rows"]),
      "truncated_by_next_reset": truncated,
      "n_steps_before": len(win["before"]),
    }
    data["meta_json"] = np.array(json.dumps(meta))
    path = self.out_dir / f"reset_{win['ordinal']:04d}.npz"
    np.savez_compressed(path, **data)
    print(f"[reset_snapshot] wrote {path.name} (after={len(win['rows'])}"
          f"{', cut by next reset' if truncated else ''})")

  def flush_timeline(self) -> None:
    arrays = self._stack(self.timeline)
    if arrays:
      np.savez_compressed(self.out_dir / "timeline.npz", **arrays)

  def close(self) -> None:
    if self.window is not None:
      self.finish_window(truncated=True)
    self.flush_timeline()


def install_hooks(recorder: Recorder) -> None:
  from mjlab.envs import ManagerBasedRlEnv

  original_reset_idx = ManagerBasedRlEnv._reset_idx
  original_step = ManagerBasedRlEnv.step

  def reset_idx(self, env_ids=None):
    hit = recorder.before_reset_idx(self, env_ids)
    result = original_reset_idx(self, env_ids)
    recorder.after_reset_idx(self, hit)
    return result

  def step(self, action):
    result = original_step(self, action)
    recorder.after_step(self)
    return result

  ManagerBasedRlEnv._reset_idx = reset_idx
  ManagerBasedRlEnv.step = step


def main() -> None:
  import mjlab.tasks  # noqa: F401  (registers tasks, incl. entry-point packages)
  from mjlab.scripts.play import PlayConfig, run_play

  recorder = Recorder()
  install_hooks(recorder)
  run_play(
    TASK_ID,
    PlayConfig(checkpoint_file=CHECKPOINT_FILE, viewer=VIEWER, num_envs=NUM_ENVS),
  )


if __name__ == "__main__":
  main()
