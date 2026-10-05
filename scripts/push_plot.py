"""Plot push-benchmark CSVs in ONE interactive window.

Run: uv run python scripts/push_plot.py
Edit the constants below, no CLI args.

Reads the CSVs written by push_benchmark.py (one file = one run = one policy, one push
duration, one target twist). Binning happens here, never in the benchmark.

Any number of CSVs can be given (e.g. one per policy from a push_benchmark.py queue):
direction and magnitude curves overlay all of them. The 2D maps and the 3D surface show
one dataset per row with 1 or 2 CSVs (plus the difference panels for 2), and ONE dataset
picked by the radio box with 3 or more.

Panels (one window, laid out on a 4-column grid):
  per dataset   recovery rate, mean time-to-fall (recovered trial = recovery timeout, both
                from the push start) and trial count, over the (Fx, Fy) plane
  difference    only with exactly 2 datasets with matching run settings: A minus B
  surface       3D mean time-to-fall (radio buttons pick the dataset, or A - B)
  direction     recovery rate vs push direction sector (Wilson 95 % bands)
  magnitude     recovery rate vs push magnitude (Wilson 95 % bands)

Sliders (live): force-plane bin size, MIN_COUNT (bins with fewer valid trials are blanked),
number of direction sectors, magnitude bin size. The constants below are their start values.

Push forces are world frame: angle 0 is world +x, which is the robot's forward direction
only at spawn (the robot may have turned a little by the time of the push).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
import numpy as np

# =============================== EDIT ME ===============================
# CSV files to plot. [] -> the most recent CSV in CSV_DIR. With two entries the
# difference figure is A (first) minus B (second), e.g. [trained_csv, constant_csv].
DATASETS: list[str] = []
CSV_DIR = "logs/push_benchmark"
LAST_N = 5          # when DATASETS is empty: plot the LAST_N most recent CSVs of CSV_DIR
                    # (names sort chronologically, so a night's queue of 4 policies -> LAST_N = 4)

# Start values of the live sliders:
BIN_N = 10.0        # N, side of a force-plane bin
MIN_COUNT = 5       # bins with fewer valid trials are blanked in the value maps
N_SECTORS = 8       # push-direction sectors for the direction plot
MAG_BIN_N = 10.0    # N, magnitude bin of the "recovery vs magnitude" curve

SHOW_CELL_COUNTS = False  # True: print the trial count in the middle of each 2D cell (<= 16 bins per side)
SHOW = True         # open the interactive window (False: no window)
SAVE_PNG = False    # True: also save the initial view as <first csv stem>_view.png at startup
                    # (the window always has a "save PNG" button for the current slider state)
DPI = 130
# ========================================================================

# slider ranges: (min, max, step)
BIN_RANGE = (2.5, 30.0, 2.5)
MIN_COUNT_RANGE = (1, 40, 1)
SECTOR_RANGE = (4, 16, 1)
MAG_BIN_RANGE = (2.5, 20.0, 2.5)

if not SHOW:
  matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

RUN_KEYS = ("run_push_duration_s", "twist_x", "twist_y", "twist_w", "timeout_s")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
class Data:
  """Valid trials of one CSV (worker_failed rows dropped) as numpy arrays."""

  def __init__(self, path: Path):
    with open(path, newline="") as f:
      rows = list(csv.DictReader(f))
    if not rows:
      raise ValueError(f"{path} has no trials")
    self.path = path
    self.meta = {k: rows[0][k] for k in rows[0] if k not in _TRIAL_COLS}
    self.label = self.meta.get("policy_label", path.stem)
    self.timeout_s = float(self.meta["timeout_s"])
    self.f_min = float(self.meta["f_min"])
    self.f_max = float(self.meta["f_max"])

    self.n_total = len(rows)
    valid = [r for r in rows if r["outcome"] in ("fell", "recovered")]
    self.n_worker_failed = self.n_total - len(valid)
    f = lambda key: np.array([float(r[key]) for r in valid])  # noqa: E731
    self.fx, self.fy = f("fx"), f("fy")
    self.mag, self.ang = f("magnitude"), f("angle_rad")
    self.recovered = np.array([r["outcome"] == "recovered" for r in valid])
    # time to fall from the push start; a recovered trial counts as the timeout
    self.ttf = np.array(
      [self.timeout_s if r["outcome"] == "recovered" else float(r["time_to_fall_s"]) for r in valid]
    )
    self.n = len(valid)
    n_rec = int(self.recovered.sum())
    print(
      f"[load] {path.name}: '{self.label}' {self.n} valid trials "
      f"(recovered {n_rec}, fell {self.n - n_rec}, worker-failed dropped {self.n_worker_failed}), "
      f"push {self.meta['run_push_duration_s']} s, twist "
      f"({self.meta['twist_x']}, {self.meta['twist_y']}, {self.meta['twist_w']}), "
      f"|F| {self.f_min:g}-{self.f_max:g} N"
    )


_TRIAL_COLS = {
  "trial_id", "env_id", "fx", "fy", "magnitude", "angle_rad", "push_start_s",
  "push_duration_s", "outcome", "cause", "time_to_fall_s", "recovered",
}


def resolve_paths() -> list[Path]:
  if DATASETS:
    paths = [Path(p).expanduser() for p in DATASETS]
  else:
    files = sorted(Path(CSV_DIR).glob("push_benchmark_*.csv"))
    if not files:
      raise FileNotFoundError(f"no push_benchmark_*.csv in {CSV_DIR}")
    paths = files[-max(LAST_N, 1):]  # timestamped names sort chronologically
  for p in paths:
    if not p.exists():
      raise FileNotFoundError(p)
  return paths


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def wilson(k, n, z: float = 1.96):
  """95 % Wilson interval for a proportion k/n (arrays; NaN where n == 0)."""
  k, n = np.asarray(k, float), np.asarray(n, float)
  with np.errstate(divide="ignore", invalid="ignore"):
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return centre - half, centre + half


class Plane:
  """2D binned statistics over (Fx, Fy) for one dataset."""

  def __init__(self, d: Data, lim: float, bin_n: float, min_count: int):
    nb = max(1, int(round(2 * lim / bin_n)))
    self.edges = np.linspace(-lim, lim, nb + 1)
    self.centres = 0.5 * (self.edges[:-1] + self.edges[1:])
    h = lambda w=None: np.histogram2d(d.fx, d.fy, bins=[self.edges, self.edges], weights=w)[0]  # noqa: E731
    self.count = h()
    n_rec = h(d.recovered.astype(float))
    ttf_sum = h(d.ttf)
    with np.errstate(divide="ignore", invalid="ignore"):
      self.rate = n_rec / self.count
      self.ttf = ttf_sum / self.count
    thin = self.count < max(min_count, 1)  # empty or below MIN_COUNT: blanked
    self.rate[thin] = np.nan
    self.ttf[thin] = np.nan


def _rings(ax, d: Data):
  for r in {d.f_min, d.f_max}:
    if r > 0:
      ax.add_patch(Circle((0, 0), r, fill=False, ec="k", lw=0.6, ls="--"))


def _sector_stats(d: Data, n_sectors: int):
  """Sectors centred on 0, 360/n, ... deg, all magnitudes pooled."""
  w = 360.0 / n_sectors
  deg = (np.degrees(d.ang) + w / 2) % 360.0
  sec = np.minimum((deg // w).astype(int), n_sectors - 1)
  n = np.bincount(sec, minlength=n_sectors)
  r = np.bincount(sec, weights=d.recovered.astype(float), minlength=n_sectors)
  return w, n, r


# ----------------------------------------------------------------------------
# The single interactive window
# ----------------------------------------------------------------------------
class Viewer:
  N_COLS = 4

  def __init__(self, datasets: list[Data], out_prefix: Path):
    from matplotlib.widgets import Button, RadioButtons, Slider
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    self.ds = datasets
    self.out_prefix = out_prefix
    self.multi = len(datasets) > 2   # 3+ CSVs: maps/3D show one dataset chosen by the radio box
    self.focus = 0
    self.lim_of = lambda bin_n: math.ceil(max(d.f_max for d in datasets) / bin_n) * bin_n

    # difference only when run settings match
    self.diff = False
    if len(datasets) == 2:
      a, b = datasets
      mism = [k for k in RUN_KEYS if a.meta[k] != b.meta[k]]
      if mism:
        print(f"[warn] difference panels skipped: run settings differ in {mism}")
      else:
        self.diff = True
    elif len(datasets) > 2:
      print("[info] 3+ datasets: no difference panels; maps and 3D show the dataset picked by the radio box")

    # panel list, filled row by row on a N_COLS grid
    panels = []
    for i in range(1 if self.multi else len(datasets)):
      panels += [("rate", i), ("ttf", i), ("count", i)]
    if self.diff:
      panels += [("drate", None), ("dttf", None)]
    panels += [("surface", None), ("direction", None), ("magnitude", None)]
    nrows = math.ceil(len(panels) / self.N_COLS)

    self.fig = plt.figure(figsize=(18, 3.0 * nrows + 1.8))
    gs = self.fig.add_gridspec(
      nrows, self.N_COLS, left=0.045, right=0.985, top=0.95, bottom=0.2, wspace=0.28, hspace=0.42
    )
    self.ax = {}    # (kind, i) -> axes
    self.cax = {}   # (kind, i) -> colorbar axes
    for k, (kind, i) in enumerate(panels):
      r, c = divmod(k, self.N_COLS)
      if kind == "surface":
        ax = self.fig.add_subplot(gs[r, c], projection="3d")
      else:
        ax = self.fig.add_subplot(gs[r, c])
        if kind in ("rate", "ttf", "count", "drate", "dttf"):
          self.cax[(kind, i)] = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.06)
      self.ax[(kind, i)] = ax

    # --- widgets ------------------------------------------------------------
    f = self.fig
    mk = lambda rect, label, rng, init: Slider(  # noqa: E731
      f.add_axes(rect), label, rng[0], rng[1], valinit=init, valstep=rng[2]
    )
    self.s_bin = mk([0.09, 0.115, 0.26, 0.022], "bin size [N]", BIN_RANGE, BIN_N)
    self.s_min = mk([0.09, 0.075, 0.26, 0.022], "min count", MIN_COUNT_RANGE, MIN_COUNT)
    self.s_sec = mk([0.50, 0.115, 0.22, 0.022], "sectors", SECTOR_RANGE, N_SECTORS)
    self.s_mag = mk([0.50, 0.075, 0.22, 0.022], "mag bin [N]", MAG_BIN_RANGE, MAG_BIN_N)

    self.surf_options = [f"{i + 1}: {d.label}" for i, d in enumerate(datasets)]
    if self.diff:
      self.surf_options.append("A - B")
    self.surf_sel = self.surf_options[0]
    self.radio = None
    if len(self.surf_options) > 1:
      rax = f.add_axes([0.775, 0.03, 0.13, 0.12])
      rax.set_title("maps + 3D show" if self.multi else "3D shows", fontsize=8)
      self.radio = RadioButtons(rax, self.surf_options)
      self.radio.on_clicked(self._on_radio)
    bax = f.add_axes([0.92, 0.06, 0.06, 0.04])
    self.btn = Button(bax, "save PNG")
    self.btn.on_clicked(lambda _e: self.save("view"))

    self.s_bin.on_changed(lambda _v: self._after(maps=True, surface=True))
    self.s_min.on_changed(lambda _v: self._after(maps=True, surface=True, magnitude=True))
    self.s_sec.on_changed(lambda _v: self._after(direction=True))
    self.s_mag.on_changed(lambda _v: self._after(magnitude=True))

    self._after(maps=True, surface=True, direction=True, magnitude=True)

  # --- callbacks -------------------------------------------------------------
  def _on_radio(self, label):
    self.surf_sel = label
    if self.multi:
      self.focus = self.surf_options.index(label)
      self._after(maps=True, surface=True)
    else:
      self._after(surface=True)

  def _after(self, maps=False, surface=False, direction=False, magnitude=False):
    if maps or surface:
      self._planes_cache = None
    if maps:
      self.draw_maps()
    if surface:
      self.draw_surface()
    if direction:
      self.draw_direction()
    if magnitude:
      self.draw_magnitude()
    self.fig.canvas.draw_idle()

  def save(self, name: str):
    out = self.out_prefix.with_name(f"{self.out_prefix.name}_{name}.png")
    self.fig.savefig(out, dpi=DPI)
    print(f"[plot] saved {out}")

  # --- planes ----------------------------------------------------------------
  def shown(self) -> list[Data]:
    """Datasets that get map panels: all of them with 1-2 CSVs, the radio's pick with 3+."""
    return [self.ds[self.focus]] if self.multi else self.ds

  def planes(self):
    if getattr(self, "_planes_cache", None) is None:
      bin_n, mc = float(self.s_bin.val), int(self.s_min.val)
      lim = self.lim_of(bin_n)
      self._planes_cache = (lim, [Plane(d, lim, bin_n, mc) for d in self.shown()])
    return self._planes_cache

  # --- 2D maps ---------------------------------------------------------------
  def _map(self, key, grid, lim, title, cmap, vmin, vmax, cbar_label, counts=None, d=None, centres=None):
    ax, cax = self.ax[key], self.cax[key]
    ax.clear()
    cax.clear()
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#e8e8e8")
    im = ax.imshow(
      grid.T, origin="lower", extent=(-lim, lim, -lim, lim), cmap=cm, vmin=vmin, vmax=vmax,
      interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Fx [N] (world +x)", fontsize=8)
    ax.set_ylabel("Fy [N]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    if d is not None:
      _rings(ax, d)
    if SHOW_CELL_COUNTS and counts is not None and grid.shape[0] <= 16:
      for ii, cx in enumerate(centres):
        for jj, cy in enumerate(centres):
          if counts[ii, jj] > 0:
            ax.text(cx, cy, f"{int(counts[ii, jj])}", ha="center", va="center", fontsize=5.5)
    cb = self.fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=7)

  def draw_maps(self):
    lim, planes = self.planes()
    mc = int(self.s_min.val)
    for i, (d, p) in enumerate(zip(self.shown(), planes)):
      self._map(("rate", i), p.rate, lim, f"{d.label}: recovery rate (n>={mc})", "RdYlGn", 0, 1,
                "recovered / trials", p.count, d, p.centres)
      self._map(("ttf", i), p.ttf, lim,
                f"{d.label}: mean time-to-fall (recovered = {d.timeout_s:g} s)", "viridis", 0,
                d.timeout_s, "s after push start", p.count, d, p.centres)
      self._map(("count", i), np.where(p.count > 0, p.count, np.nan), lim,
                f"{d.label}: trials per bin (total {d.n})", "Blues", 0, None, "trials",
                p.count, d, p.centres)
    if self.diff:
      a, b = self.ds
      pa, pb = planes
      self._map(("drate", None), pa.rate - pb.rate, lim,
                f"recovery rate: {a.label} - {b.label}", "RdBu", -1, 1, "delta rate", None, a)
      self._map(("dttf", None), pa.ttf - pb.ttf, lim,
                f"mean time-to-fall: {a.label} - {b.label} [blue = first better]", "RdBu",
                -a.timeout_s, a.timeout_s, "delta s", None, a)

  # --- 3D surface ------------------------------------------------------------
  def draw_surface(self):
    lim, planes = self.planes()
    ax = self.ax[("surface", None)]
    elev, azim = ax.elev, ax.azim
    ax.clear()
    ax.view_init(elev, azim)
    k = self.surf_options.index(self.surf_sel)
    if self.multi:
      k = 0  # planes() holds only the focused dataset
    shown = self.shown()
    t0 = self.ds[0].timeout_s
    if k < len(shown):
      Z, cmap, lo, hi, zl = planes[k].ttf, "viridis", 0.0, shown[k].timeout_s, (0.0, shown[k].timeout_s)
      centres = planes[k].centres
      title = f"{shown[k].label}: mean time-to-fall"
    else:
      Z = planes[0].ttf - planes[1].ttf
      cmap, lo, hi, zl = "RdBu", -t0, t0, (-t0, t0)
      centres = planes[0].centres
      title = f"time-to-fall: {self.ds[0].label} - {self.ds[1].label}"
    X, Y = np.meshgrid(centres, centres, indexing="ij")
    ax.plot_surface(X, Y, Z, cmap=cmap, vmin=lo, vmax=hi, edgecolor="none")
    ax.set_zlim(*zl)
    ax.set_xlabel("Fx [N]", fontsize=7)
    ax.set_ylabel("Fy [N]", fontsize=7)
    ax.set_zlabel("s", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)

  # --- direction / magnitude -------------------------------------------------
  def draw_direction(self):
    ax = self.ax[("direction", None)]
    ax.clear()
    ns = int(self.s_sec.val)
    for k, d in enumerate(self.ds):
      w, n, r = _sector_stats(d, ns)
      rate = np.where(n > 0, r / np.maximum(n, 1), np.nan)
      ctr = np.arange(ns) * w
      ax.plot(ctr, rate, "-o", ms=4, color=f"C{k % 10}", label=f"{d.label} (n={d.n})")
      for x, y, c in zip(ctr, rate, n):
        if c > 0 and ns <= 12 and len(self.ds) <= 2:
          ax.annotate(str(c), (x, y), textcoords="offset points", xytext=(0, 6),
                      ha="center", fontsize=5.5, color=f"C{k % 10}")
    w = 360.0 / ns
    ax.set_xticks(np.arange(ns) * w)
    ax.set_xticklabels([f"{x:g}" for x in np.arange(ns) * w], fontsize=6, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("push direction [deg from world +x]", fontsize=8)
    ax.set_ylabel("recovery rate", fontsize=8)
    ax.set_ylim(-0.05, 1.12)
    ax.set_title(f"recovery vs direction ({ns} sectors, magnitudes pooled)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

  def draw_magnitude(self):
    ax = self.ax[("magnitude", None)]
    ax.clear()
    mb, mc = float(self.s_mag.val), max(int(self.s_min.val), 1)
    drawn = False
    for k, d in enumerate(self.ds):
      edges = np.arange(math.floor(d.f_min / mb) * mb, d.f_max + mb, mb)
      if len(edges) < 2:
        continue
      idx = np.clip(np.digitize(d.mag, edges) - 1, 0, len(edges) - 2)
      n = np.bincount(idx, minlength=len(edges) - 1)
      r = np.bincount(idx, weights=d.recovered.astype(float), minlength=len(edges) - 1)
      ok = n >= mc
      if ok.sum() >= 2:
        drawn = True
        x = 0.5 * (edges[:-1] + edges[1:])
        lo, hi = wilson(r, n)
        ax.plot(x[ok], (r / np.maximum(n, 1))[ok], "-o", ms=4, color=f"C{k % 10}", label=d.label)
        ax.fill_between(x[ok], lo[ok], hi[ok], color=f"C{k % 10}", alpha=0.2)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("push magnitude [N]", fontsize=8)
    ax.set_ylabel("recovery rate", fontsize=8)
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.3)
    if drawn:
      ax.set_title(f"recovery vs magnitude (directions pooled, bins n>={mc})", fontsize=9)
      ax.legend(fontsize=7)
    else:
      ax.set_title("recovery vs magnitude: needs >= 2 bins; widen F_MIN..F_MAX in the run", fontsize=8)


# ----------------------------------------------------------------------------
def main():
  paths = resolve_paths()
  datasets = [Data(p) for p in paths]
  labels = [d.label for d in datasets]
  for i, d in enumerate(datasets):  # identical labels would make the legends unreadable
    if labels.count(d.label) > 1:
      d.label = f"{d.label} [{i + 1}]"
  if len(datasets) >= 2:
    for key in RUN_KEYS:
      if len({d.meta[key] for d in datasets}) > 1:
        print(f"[warn] the datasets differ in {key}: {[d.meta[key] for d in datasets]} "
              "(curves of different conditions are overlaid; the plan keeps them separate)")
  viewer = Viewer(datasets, paths[0].with_suffix(""))
  if SAVE_PNG:
    viewer.save("view")
  if SHOW:
    plt.show()


if __name__ == "__main__":
  main()