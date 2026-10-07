"""Inspect and compare the reset files written by scripts/reset_snapshot.py.

Edit the constants, then:  uv run python scripts/reset_snapshot_diff.py

1. Always prints an index of every reset in the session: ordinal, episode length at
   termination, applied-force norm before/after, root height and pitch at the end of the
   episode (from `pre_step__*`), the controller failure latches at that moment, and the
   pitch 0.2 s into the next episode. Row k describes the episode that ENDED at reset k.
2. Prints physics invariants of `post_reset`: fields MuJoCo-Warp's reset_data zeroes
   (qvel, qacc, qacc_warmstart, ctrl) -- anything non-zero here is state mjlab's own
   reset events wrote or left.
3. If A_ORDINAL and B_ORDINAL are set, compares their `after__` rows tick by tick and
   prints, per recorded key, the max |A-B| and the first tick it exceeds TOL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_ROOT = Path("reset_snapshots")
SESSION_DIR: str | None = None  # None = newest session under OUT_ROOT
A_ORDINAL: int | None = None  # e.g. a reset that looked fine
B_ORDINAL: int | None = None  # e.g. a reset that looked weird
N_COMPARE = 40  # first env steps after the reset to compare
TOL = 1e-6
TOP = 25  # keys listed in the comparison
SHOW_KEYS = (  # printed tick by tick for A and B
  "sim_root_pitch",
  "sim_qfrc_applied_norm",
  "ds_fresh",
  "ds_s__*",  # every scalar datastore output
)
SHOW_TICKS = 12
INVARIANT_FIELDS = (
  "sim_full_qvel",
  "sim_full_qacc",
  "sim_full_qacc_warmstart",
  "sim_full_ctrl",
  "sim_full_time",
)


def load(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as f:
    return {k: f[k] for k in f.files}


def meta(d: dict) -> dict:
  return json.loads(str(d["meta_json"]))


def session() -> Path:
  if SESSION_DIR is not None:
    return Path(SESSION_DIR)
  return sorted(p for p in OUT_ROOT.iterdir() if p.is_dir())[-1]


def index(files: list[Path]) -> None:
  # pre_step__* is the state just before mjlab reset the env (the real end of the
  # episode); before__[-1] is already post-reset, so it is NOT used for end-state values.
  print(f"{'ord':>4} {'dispatch#':>9} {'ep_len':>6} {'F_before':>9} {'F_after0':>9} "
        f"{'z_end':>7} {'pitch_end':>9} {'ctl_fail':>8} {'wrk_fail':>8} {'pitch+0.2s':>10} {'after':>5}")
  for path in files:
    d = load(path)
    m = meta(d)
    f_before = float(d["before__sim_qfrc_applied_norm"][:-1][-5:].max()) if len(d["before__sim_qfrc_applied_norm"]) > 1 else float("nan")
    f_after = float(d["after__sim_qfrc_applied_norm"][:5].max()) if len(d["after__sim_qfrc_applied_norm"]) else float("nan")
    pitch = d["after__sim_root_pitch"][:, 0]
    first_disp = int(d["after__rec_dispatch_reset_ordinal"][0, 0]) if len(pitch) else -1
    print(f"{m['reset_ordinal']:>4} {first_disp:>9} {int(d['pre_step__env_episode_length_buf'][0]):>6} "
          f"{f_before:>9.1f} {f_after:>9.1f} {float(d['pre_step__sim_root_qpos'][2]):>7.3f} "
          f"{float(d['pre_step__sim_root_pitch'][0]):>9.3f} {int(d['pre_step__ctl_failed'][0]):>8} "
          f"{int(d['pre_step__ctl_worker_failed'][0]):>8} "
          f"{(float(pitch[min(40, len(pitch) - 1)]) if len(pitch) else float('nan')):>10.3f} "
          f"{m['after_steps']:>5}")


def invariants(files: list[Path]) -> None:
  print("\nphysics right after mjlab's reset (max |value|; reset_data zeroes these):")
  print(f"{'ord':>4} " + " ".join(f"{f.replace('sim_full_', ''):>14}" for f in INVARIANT_FIELDS))
  for path in files:
    d = load(path)
    cells = []
    for field in INVARIANT_FIELDS:
      key = f"post_reset__{field}"
      cells.append(f"{np.abs(d[key]).max():>14.3e}" if key in d and d[key].size else f"{'n/a':>14}")
    print(f"{meta(d)['reset_ordinal']:>4} " + " ".join(cells))


def expand(keys: tuple[str, ...], available: list[str]) -> list[str]:
  out: list[str] = []
  for k in keys:
    if k.endswith("*"):
      out += [a for a in available if a.startswith("after__" + k[:-1])]
    elif "after__" + k in available:
      out.append("after__" + k)
  return out


def compare(a: dict, b: dict) -> None:
  n = min(N_COMPARE, a["after__rec_step"].shape[0], b["after__rec_step"].shape[0])
  print(f"\nA={meta(a)['reset_ordinal']} vs B={meta(b)['reset_ordinal']}, first {n} steps after reset")
  rows = []
  for key in sorted(k for k in a if k.startswith("after__") and k in b and not k.startswith("after__rec_")):
    x, y = a[key][:n], b[key][:n]
    if x.dtype.kind not in "fiub" or x.shape != y.shape or x.size == 0:
      continue
    diff = np.abs(x.astype(float) - y.astype(float)).reshape(n, -1).max(axis=1)
    over = np.nonzero(diff > TOL)[0]
    rows.append((float(diff.max()), key[len("after__"):], int(over[0]) if len(over) else -1))
  rows.sort(reverse=True)
  print(f"{'max|A-B|':>12}  {'first tick>tol':>14}  key")
  for dmax, key, first in rows[:TOP]:
    print(f"{dmax:>12.4e}  {first:>14}  {key}")
  print("(expected non-zero: joint/root qpos from reset randomization, force while the mouse differs)")

  avail = list(a)
  for key in expand(SHOW_KEYS, avail):
    print(f"\n{key[len('after__'):]}  tick: A | B")
    for t in range(min(SHOW_TICKS, n)):
      print(f"  {t:>3}: {np.round(a[key][t].astype(float), 5)} | {np.round(b[key][t].astype(float), 5)}")


def main() -> None:
  folder = session()
  files = sorted(folder.glob("reset_*.npz"))
  print(f"session {folder} ({len(files)} resets)")
  if not files:
    return
  index(files)
  invariants(files)
  if A_ORDINAL is not None and B_ORDINAL is not None:
    by_ord = {meta(load(p))["reset_ordinal"]: p for p in files}
    compare(load(by_ord[A_ORDINAL]), load(by_ord[B_ORDINAL]))


if __name__ == "__main__":
  main()
