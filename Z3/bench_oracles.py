"""
Isolated runtime of the two max-search oracles: z3 against brute force.

Both answer the same question at a dual point ``x`` -- the largest violation
of the dual constraints over every latent stratum, and the strata attaining
it -- through the oracle contract ``x -> (h, idx, rhs)``:

  brute force  ``prob.separate``: the closed-form max over the reduced index
               arithmetic (for IV_cont the same computation as ``max_oracle``
               in Data/IV_cont/reduced_ipm_solver.py, ``O(k^3)``)
  z3           :class:`~Z3.z3_oracle.Z3Oracle`: one ``Optimize.check()`` per
               configuration key, maximising over the remaining choices

The interior-point method only supplies the query points.  It is run once per
bound with the brute-force oracle while every point it queries is recorded;
both oracles are then timed on the same recorded points, outside the solver,
so nothing but the max search is measured.

    python -m Z3.bench_oracles                          # every problem
    python -m Z3.bench_oracles --problems iv --sizes 2 3 4 5
    python -m Z3.bench_oracles --report-only            # table + plots from CSVs
"""

import argparse
import csv
import glob
import importlib.util
import os
import platform
import time

import numpy as np

from . import main, problems, z3_oracle
from .run_iv import load_iv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Z3", "bench_results")

# where along an IPM trajectory to query, coarse to fine, so that a budget cut
# still leaves the points spread out
POSITIONS = (0.5, 0.25, 0.75, 0.125)


# ---------------------------------------------------------------------------
# the problems, one size parameter each
# ---------------------------------------------------------------------------

def iv_problem(k):
    P, edges, _ = load_iv(k)
    yv = (edges[:-1] + edges[1:]) / 2
    return lambda sign: problems.IVCont(P, k, sign=sign, y_values=yv)


def ev_problem(k):
    from Data.Edu_vs_Voting.LP_construction import (empirical_distribution_EV,
                                                    generate_data_EV)
    data, _, _ = generate_data_EV(10000, tau=0.5, seed=2020)
    P, _, y_bins = empirical_distribution_EV(data, k, k)
    return lambda sign: problems.EduVsVoting(P, k, k, y_bins, sign=sign)


def med_problem(kY, kM=2):
    # the directory name has a space, so it cannot be imported as a package
    path = os.path.join(ROOT, "Data", "Mediate Analysis",
                        "reduced_dual_solver.py")
    spec = importlib.util.spec_from_file_location("mediation_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p0, p1, vY, _ = mod.make_data(kM, kY)
    return lambda sign: problems.Mediation(p0, p1, vY, sign=sign)


PROBLEMS = {
    "iv": dict(title="IV (continuous)", axis="k  (Z and Y bins)",
               sizes=range(2, 9), build=iv_problem,
               z3="k^2 checks, each over 2^k treatment maps",
               bf="O(k^3)"),
    "ev": dict(title="Education vs voting", axis="k  (X and Y bins)",
               sizes=range(2, 9), build=ev_problem,
               z3="1 check over all 2^k k^(2k) strata", bf="O(k^3)"),
    "med": dict(title="Mediation, kM = 2", axis="kY  (Y bins)",
                sizes=range(2, 11), build=med_problem,
                z3="kM^2 checks, each over kY^(2kM) outcome maps",
                bf="O(kM^2 kY)"),
}


def n_strata(prob):
    """Rows of the full dual, i.e. every latent stratum (exact integer)."""
    if isinstance(prob, problems.IVCont):
        return 2 ** prob.k * prob.k ** 2
    if isinstance(prob, problems.EduVsVoting):
        return 2 ** prob.nx * prob.ky ** (2 * prob.nx)
    return prob.kM ** 2 * prob.kY ** (2 * prob.kM)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def ipm_points(prob):
    """Every dual point the IPM queries during one bound solve."""
    seen = []

    def record(x):
        seen.append(np.array(x, float))
        return prob.separate(x)

    main.solve(prob, oracle=record)
    return seen


def time_call(fn, target_s=0.02, repeat=5):
    """Median per-call time of ``fn()`` over ``repeat`` timed loops."""
    fn()
    t0 = time.perf_counter()
    fn()
    number = max(1, int(target_s / max(time.perf_counter() - t0, 1e-7)))
    runs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        runs.append((time.perf_counter() - t0) / number)
    return float(np.median(runs))


def timed_z3(prob):
    """A :class:`Z3Oracle` whose ``Optimize.check()`` calls are timed."""
    t0 = time.perf_counter()
    # top_up=False: the default tops a one-row answer up with prob.separate,
    # which would put brute-force time inside the z3 measurement
    orc = z3_oracle.Z3Oracle(prob, top_up=False)
    setup = time.perf_counter() - t0
    checks = []
    check = orc.opt.check

    def timed_check(*args):
        t = time.perf_counter()
        out = check(*args)
        checks.append(time.perf_counter() - t)
        return out

    orc.opt.check = timed_check
    return orc, setup, checks


def bench_size(name, size, budget_s):
    """Both oracles on the same recorded IPM points at one size."""
    make = PROBLEMS[name]["build"](size)
    runs = {}
    for sign in (+1, -1):
        prob = make(sign)
        runs[sign] = (prob, ipm_points(prob)) + timed_z3(prob)

    rows, spent = [], 0.0
    for pos in POSITIONS:
        # a position queries both bounds, so at least one point per bound runs
        if rows and spent >= budget_s:
            break
        for sign in (+1, -1):
            prob, traj, orc, setup, checks = runs[sign]
            # round 0 is skipped: lam = 0 there, every stratum ties and z3
            # answers instantly -- one call in ~50, not representative
            rnd = 1 + int(round(pos * (len(traj) - 2)))
            x = traj[rnd]
            checks.clear()
            t0 = time.perf_counter()
            h, sel, srhs = orc(x)
            z3_s = time.perf_counter() - t0
            spent += z3_s
            bf_s = time_call(lambda: prob.separate(x))
            h_bf = prob.separate(x)[0]
            # z3's value and the value its returned rows really attain must
            # both match the brute-force maximum
            attained = float((x[sel].sum(axis=1) + x[-1]
                              - np.ravel(srhs)).max())
            diff = max(abs(h - h_bf), abs(attained - h_bf))
            rows.append(dict(problem=name, size=size, sign=sign, round=rnd,
                             ipm_calls=len(traj), n_vars=prob.n,
                             n_strata=n_strata(prob), z3_checks=len(checks),
                             z3_setup_s=setup, z3_call_s=z3_s,
                             z3_in_check_s=sum(checks), bf_call_s=bf_s,
                             abs_diff=diff))
            print(f"  [{name} size={size} sign={sign:+d} round={rnd:2d}] "
                  f"z3 {fmt_time(z3_s):>9s} ({len(checks)} checks)  "
                  f"brute force {fmt_time(bf_s):>9s}  "
                  f"ratio {fmt_count(z3_s / bf_s):>6s}  |dh| {diff:.1e}",
                  flush=True)
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

FIELDS = ("problem", "size", "sign", "round", "ipm_calls", "n_vars",
          "n_strata", "z3_checks", "z3_setup_s", "z3_call_s", "z3_in_check_s",
          "bf_call_s", "abs_diff")
INTS = ("size", "sign", "round", "ipm_calls", "n_vars", "n_strata",
        "z3_checks")


def save_points(rows, path):
    with open(path, "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=FIELDS)
        out.writeheader()
        out.writerows(rows)


def load_points(outdir):
    rows = []
    for path in sorted(glob.glob(os.path.join(outdir, "points_*.csv"))):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({k: (int(v) if k in INTS else
                                 v if k == "problem" else float(v))
                             for k, v in r.items()})
    return rows


def summarise(points):
    """``{problem: [row per size]}``, medians and ranges over the points."""
    out = {}
    for name in PROBLEMS:
        sizes = sorted({p["size"] for p in points if p["problem"] == name})
        for size in sizes:
            g = [p for p in points
                 if p["problem"] == name and p["size"] == size]
            z3 = np.array([p["z3_call_s"] for p in g])
            bf = np.array([p["bf_call_s"] for p in g])
            out.setdefault(name, []).append(dict(
                size=size, points=len(g), n_vars=g[0]["n_vars"],
                n_strata=g[0]["n_strata"], z3_checks=g[0]["z3_checks"],
                ipm_calls=float(np.mean([p["ipm_calls"] for p in g])),
                z3_setup=sorted({p["z3_setup_s"] for p in g}),
                z3_med=float(np.median(z3)), z3_min=float(z3.min()),
                z3_max=float(z3.max()),
                z3_check=float(np.median([p["z3_in_check_s"] / p["z3_checks"]
                                          for p in g])),
                z3_in_check=float(sum(p["z3_in_check_s"] for p in g)
                                  / z3.sum()),
                bf_med=float(np.median(bf)), bf_min=float(bf.min()),
                bf_max=float(bf.max()),
                ratio=float(np.median(z3) / np.median(bf)),
                abs_diff=max(p["abs_diff"] for p in g)))
    return out


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def fmt_time(s):
    if s >= 1e3:
        return f"{s / 60:.3g} min"
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if s >= scale:
            return f"{s / scale:.3g} {unit}"
    return f"{s * 1e9:.3g} ns"


def fmt_count(v):
    for suffix, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= scale:
            return f"{v / scale:.3g}{suffix}"
    return f"{v:.3g}"


def fmt_strata(n):
    return f"{n:,}" if n < 1e6 else f"{n:.1e}".replace("e+", "e")


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

def write_table(summary, path):
    import z3
    lines = ["| problem | size | dual vars | strata | points "
             "| Z3 checks / call | Z3 / check | Z3 / call "
             "| brute force / call | Z3 ÷ brute force | max abs Δh |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    setups, shares = [], []
    for name, rows in summary.items():
        size_name = PROBLEMS[name]["axis"].split()[0]
        for r in rows:
            setups += r["z3_setup"]
            shares.append(r["z3_in_check"])
            lines.append(
                f"| {PROBLEMS[name]['title']} | {size_name}={r['size']} "
                f"| {r['n_vars']} | {fmt_strata(r['n_strata'])} "
                f"| {r['points']} | {r['z3_checks']} "
                f"| {fmt_time(r['z3_check'])} | {fmt_time(r['z3_med'])} "
                f"| {fmt_time(r['bf_med'])} | {fmt_count(r['ratio'])}× "
                f"| {r['abs_diff']:.0e} |")
    lines += [
        "",
        "Medians over the query points of each row.  Both oracles are timed "
        "on the same points: the dual iterates one IPM bound solve queried "
        "(lower and upper bound, rounds spread along each trajectory; round 0 "
        "is skipped because every stratum ties there).  Z3 runs once per "
        "point; brute force is the median of 5 timed loops.",
        "",
        f"* Z3 model construction, once per bound and not counted above: "
        f"{min(setups) * 1e3:.1f}–{max(setups) * 1e3:.1f} ms.",
        f"* Share of Z3's per-call time spent inside `Optimize.check()`: "
        f"{100 * min(shares):.1f}–{100 * max(shares):.1f}%.",
        "* max abs Δh: worst disagreement with the brute-force maximum, "
        "of both z3's value and the value its returned rows attain.",
    ]
    for name in summary:
        spec = PROBLEMS[name]
        lines.append(f"* {spec['title']}: Z3 = {spec['z3']}; "
                     f"brute force = {spec['bf']}.")
    lines.append(f"* z3 {z3.get_version_string()}, numpy {np.__version__}, "
                 f"Python {platform.python_version()}, "
                 f"{os.cpu_count()} CPUs.")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return text


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
Z3_COLOR, BF_COLOR = "#2a78d6", "#eb6834"         # categorical slots 1, 2


def _axes(n, height, sharey=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, height), sharey=sharey,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return plt, fig, axes[0]


def _style(ax, x, title, xlabel):
    from matplotlib.ticker import LogLocator, NullLocator
    ax.set_facecolor(SURFACE)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=30))
    ax.yaxis.set_minor_locator(NullLocator())
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=9, labelcolor=MUTED, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xlim(x[0] - 0.4, x[-1] + 1.6)          # room for the end labels
    ax.set_title(title, loc="left", color=INK, fontsize=11,
                 fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def _line(ax, x, y, color, label=None, marker="o"):
    ax.plot(x, y, color=color, linewidth=1.6, marker=marker, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=label,
            solid_joinstyle="round", solid_capstyle="round", zorder=3)


def _end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points",
                va="center", ha="left", color=INK2, fontsize=9)


def _titles(fig, title, subtitle):
    fig.text(0.012, 0.975, title, color=INK, fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.012, 0.915, subtitle, color=INK2, fontsize=9.5, va="top")


def plot_runtime(summary, path):
    from matplotlib.ticker import FuncFormatter
    plt, fig, axes = _axes(len(summary), 4.6)
    for ax, (name, rows) in zip(axes, summary.items()):
        x = [r["size"] for r in rows]
        for key, color, label in (("z3", Z3_COLOR, "Z3 oracle"),
                                  ("bf", BF_COLOR, "Brute-force oracle")):
            med = [r[f"{key}_med"] for r in rows]
            ax.fill_between(x, [r[f"{key}_min"] for r in rows],
                            [r[f"{key}_max"] for r in rows], color=color,
                            alpha=0.10, linewidth=0, zorder=2)
            _line(ax, x, med, color, label)
            _end_label(ax, x[-1], med[-1], fmt_time(med[-1]))
        _style(ax, x, PROBLEMS[name]["title"], PROBLEMS[name]["axis"])
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: fmt_time(v).replace(".0 ", " ")))
    axes[0].set_ylabel("time per oracle call", color=INK2, fontsize=9)
    _titles(fig, "Max-search time per oracle call: Z3 vs brute force",
            "Median over the same recorded IPM query points; band shows "
            "min–max.  Log scale, lower is faster.")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False,
               fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.995, 0.985),
               handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.0)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_speedup(summary, path):
    from matplotlib.ticker import FuncFormatter
    plt, fig, axes = _axes(len(summary), 4.2)
    for ax, (name, rows) in zip(axes, summary.items()):
        x = [r["size"] for r in rows]
        ratio = [r["ratio"] for r in rows]
        # a derived pair-wise quantity, so neutral ink rather than either
        # oracle's identity colour
        _line(ax, x, ratio, INK2)
        _end_label(ax, x[-1], ratio[-1], f"{fmt_count(ratio[-1])}×")
        _style(ax, x, PROBLEMS[name]["title"], PROBLEMS[name]["axis"])
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{fmt_count(v)}×"))
    axes[0].set_ylabel("Z3 time ÷ brute-force time", color=INK2, fontsize=9)
    _titles(fig, "How many times slower Z3 is than brute force",
            "Ratio of median per-call times on the same query points.  "
            "Log scale.")
    fig.tight_layout(rect=(0, 0, 1, 0.85), w_pad=2.0)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def report(outdir):
    summary = summarise(load_points(outdir))
    if not summary:
        raise SystemExit(f"no points_*.csv in {outdir}")
    print()
    print(write_table(summary, os.path.join(outdir, "oracle_table.md")))
    plot_runtime(summary, os.path.join(outdir, "oracle_runtime.png"))
    plot_speedup(summary, os.path.join(outdir, "oracle_speedup.png"))
    print(f"wrote oracle_table.md, oracle_runtime.png, oracle_speedup.png "
          f"to {outdir}")


def cli():
    p = argparse.ArgumentParser(
        description="Isolated runtime of the z3 and brute-force max-search "
                    "oracles on the same dual points.")
    p.add_argument("--problems", nargs="+", choices=list(PROBLEMS),
                   default=list(PROBLEMS))
    p.add_argument("--sizes", nargs="+", type=int, default=None,
                   help="override every chosen problem's size grid")
    p.add_argument("--budget", type=float, default=60.0,
                   help="z3 seconds per size after which no new query "
                        "points start (one point per bound always runs)")
    p.add_argument("--out", default=OUT)
    p.add_argument("--report-only", action="store_true",
                   help="rebuild the table and plots from saved CSVs")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.report_only:
        # the first z3 object in a process pays for loading the library;
        # keep that out of the first size's setup time
        z3_oracle.Z3Oracle(problems.IVCont(np.full((2, 2, 2), 0.25), 2))
        for name in args.problems:
            rows = []
            for size in args.sizes or PROBLEMS[name]["sizes"]:
                rows += bench_size(name, size, args.budget)
                # saved after every size so a long run can be watched or cut
                save_points(rows, os.path.join(args.out,
                                               f"points_{name}.csv"))
    report(args.out)


if __name__ == "__main__":
    cli()
