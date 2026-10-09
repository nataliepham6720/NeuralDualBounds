"""
What makes the z3 oracle slow.

``Z3Oracle.__call__`` (z3_oracle.py:51-86) is replayed step by step; the step
that dominates -- ``Optimize.check()`` on z3_oracle.py:65 -- is then timed on
its own, and the same max is handed to z3 in other encodings to find what
inside that check costs the time.

    python -m Z3.profile_z3_oracle --part steps    # every step of a call
    python -m Z3.profile_z3_oracle --part check    # the slow step alone (IV)
    python -m Z3.profile_z3_oracle --report-only   # tables + plots from CSVs

steps  replays the call line for line, with a timer around each step, at the
       mid-trajectory IPM query point bench_oracles also measures
check  times ``Optimize.check()`` alone, one configuration at a time, with
       z3's own counters (they are per check), and solves each of those
       configurations again as
         array + Bool  the Select into the x array of problems.py:102, at an
                       index a Bool picks instead of the functions t, y
         direct        one Bool per z and an If over the two cells it can
                       reach: no array, no uninterpreted function
         find / prove  the repo encoding as two plain satisfiability checks,
                       cost >= v* (sat) and cost > v* (unsat)
       and by brute force, for scale
"""

import argparse
import csv
import os
import time

import numpy as np

from . import problems, z3_oracle
from .bench_oracles import (BF_COLOR, MUTED, OUT, PROBLEMS, SURFACE,
                            Z3_COLOR, _axes, _end_label, _line, _style,
                            _titles, fmt_count, fmt_time, ipm_points,
                            time_call)

STEP_SIZES = {"iv": range(2, 8), "ev": range(2, 7), "med": range(2, 9)}
CHECK_SIZES = range(2, 11)
ARRAY_BOOL_MAX_K = 8        # as slow as the repo encoding from k = 6 on
FIND_PROVE_MAX_K = 7

CHECK, ARRAY, READ, PUSH = ("Optimize.check()", "x → z3 array + assert",
                            "read model", "push / pin / pop")
# the lines of z3_oracle.py each step covers
STEP_LINES = {CHECK: "65", ARRAY: "57-58", READ: "71-73",
              PUSH: "56, 62-64, 76-77"}
COUNTERS = ("decisions", "conflicts", "mk bool var")
AQUA, YELLOW = "#1baf7a", "#eda100"                 # categorical slots 3, 4


def mid_point(prob):
    """The mid-trajectory IPM query point (bench_oracles' first position)."""
    traj = ipm_points(prob)
    rnd = 1 + int(round(0.5 * (len(traj) - 2)))
    return rnd, traj[rnd]


def counters(solver):
    """z3's search counters for the solver's last check."""
    st = solver.statistics()
    have = set(st.keys())
    return {c.replace(" ", "_"): st.get_key_value(c) if c in have else 0
            for c in COUNTERS}


def timed_check(solver):
    t0 = time.perf_counter()
    solver.check()
    return time.perf_counter() - t0, counters(solver)


def warm_up():
    """Load z3 and run one tiny call, so no measured size pays for that."""
    prob = problems.IVCont(np.full((2, 2, 2), 0.25), 2)
    z3_oracle.Z3Oracle(prob, top_up=False)(np.zeros(prob.n))


# ---------------------------------------------------------------------------
# 1. every step of one call
# ---------------------------------------------------------------------------

def replay_call(orc, x):
    """
    ``orc(x)`` replayed step by step -- the same z3 calls in the same order as
    z3_oracle.py:56-77 -- with a timer around each step.
    """
    z3, prob, opt = orc._z3, orc.prob, orc.opt
    spent = dict.fromkeys(STEP_LINES, 0.0)
    n_checks = decisions = 0

    def step(name, fn):
        t0 = time.perf_counter()
        out = fn()
        spent[name] += time.perf_counter() - t0
        return out

    step(PUSH, opt.push)                                              # :56
    arr = step(ARRAY, lambda: orc._container(x[:prob.n_obs]))         # :57
    step(ARRAY, lambda: opt.add(orc.lam_arr == arr,                   # :57-58
                                orc.cost_0 == z3.RealVal(repr(float(x[-1])))))
    h = -np.inf
    for key in orc.keys:
        step(PUSH, opt.push)                                          # :62
        step(PUSH, lambda: [opt.add(c) for c in prob.z3_pin(z3, key)])
        step(CHECK, opt.check)                                        # :65
        n_checks += 1
        decisions += counters(opt)["decisions"]
        model = step(READ, opt.model)                                 # :71
        h = max(h, step(READ, lambda: orc._value(model)))             # :72
        step(READ, lambda: orc.read(model, key))                      # :73
        step(PUSH, opt.pop)                                           # :76
    step(PUSH, opt.pop)                                               # :77
    return h, spent, n_checks, decisions


def run_steps(names, path):
    rows = []
    for name in names:
        for size in STEP_SIZES[name]:
            make = PROBLEMS[name]["build"](size)
            for sign in (+1, -1):
                prob = make(sign)
                rnd, x = mid_point(prob)
                orc = z3_oracle.Z3Oracle(prob, top_up=False)
                t0 = time.perf_counter()
                h, spent, n_checks, decisions = replay_call(orc, x)
                total = time.perf_counter() - t0
                rows.append(dict(
                    problem=name, size=size, sign=sign, round=rnd,
                    total_s=total, check_s=spent[CHECK],
                    array_s=spent[ARRAY], read_s=spent[READ],
                    push_s=spent[PUSH], n_checks=n_checks,
                    decisions=decisions,
                    abs_diff=abs(h - prob.separate(x)[0])))
                print(f"  [steps {name} size={size} sign={sign:+d}] "
                      f"call {fmt_time(total)}: " + ", ".join(
                          f"{s} {fmt_time(v)}" for s, v in spent.items()),
                      flush=True)
            save_csv(rows, path)
    return rows


# ---------------------------------------------------------------------------
# 2. the slow step alone, and 3. the same max in other encodings (IV)
# ---------------------------------------------------------------------------

def _real(z3, v):
    # numbers enter z3 as text, as the oracle passes them (z3_oracle.py:40)
    return z3.RealVal(repr(float(v)))


def _value(opt, cost):
    return float(opt.model().eval(cost, model_completion=True).as_fraction())


def direct_model(orc, x, key):
    """The same max: one Bool per z, an If over the two cells it can reach."""
    z3, prob = orc._z3, orc.prob
    k, (y0, y1) = prob.k, key
    lam = x[:prob.n_obs].reshape(k, 2, k)
    t = [z3.Bool(f"t{z}") for z in range(k)]
    cost = _real(z3, x[-1] - prob.rhs[y0, y1]) + z3.Sum(
        [z3.If(t[z], _real(z3, lam[z, 1, y1]), _real(z3, lam[z, 0, y0]))
         for z in range(k)])
    opt = z3.Optimize()
    opt.maximize(cost)
    return opt, cost


def array_bool_model(orc, x, key):
    """problems.py:102's Select into the x array, at an index a Bool picks."""
    z3, prob = orc._z3, orc.prob
    k, (y0, y1) = prob.k, key
    arr = z3.Array("matrix", z3.IntSort(), z3.RealSort())
    t = [z3.Bool(f"t{z}") for z in range(k)]
    cost = _real(z3, x[-1] - prob.rhs[y0, y1]) + z3.Sum(
        [z3.Select(arr, z3.If(t[z], z * 2 * k + k + y1, z * 2 * k + y0))
         for z in range(k)])
    opt = z3.Optimize()
    opt.add(arr == orc._container(x[:prob.n_obs]))
    opt.maximize(cost)
    return opt, cost


def find_prove(orc, x, key, v_star):
    """The repo encoding as two satisfiability checks around its optimum."""
    z3, prob = orc._z3, orc.prob
    s = z3.Solver()
    lam_arr = z3.Array("matrix", z3.IntSort(), z3.RealSort())
    cost_0 = z3.Real("cost0")
    cost, _ = prob.z3_model(z3, s, lam_arr, cost_0)
    s.add(lam_arr == orc._container(x[:prob.n_obs]),
          cost_0 == z3.RealVal(repr(float(x[-1]))))
    for con in prob.z3_pin(z3, key):
        s.add(con)
    out = {}
    for label, goal, want in (("find", cost >= v_star, z3.sat),
                              ("prove", cost > v_star, z3.unsat)):
        s.push()
        s.add(goal)
        t0 = time.perf_counter()
        got = s.check()
        out[f"{label}_s"] = time.perf_counter() - t0
        s.pop()
        if got != want:
            raise RuntimeError(f"{label}: z3 returned {got}, expected {want}")
    return out


def check_key(orc, rnd, x, key):
    """One configuration: the oracle's own check, then the other encodings."""
    z3, prob = orc._z3, orc.prob
    k, (y0, y1) = prob.k, key
    lam = x[:prob.n_obs].reshape(k, 2, k)
    a0, a1 = lam[:, 0, y0], lam[:, 1, y1]
    const = float(x[-1] - prob.rhs[y0, y1])
    best = float(np.maximum(a0, a1).sum() + const)
    row = dict(k=k, sign=prob.sign, round=rnd, y0=y0, y1=y1,
               bf_s=time_call(lambda: np.maximum(a0, a1).sum() + const))

    # the slow step alone: z3_oracle.py:62-65, then read off the optimum
    orc.opt.push()
    for con in prob.z3_pin(z3, key):
        orc.opt.add(con)
    row["repo_s"], stats = timed_check(orc.opt)
    row.update({f"repo_{c}": v for c, v in stats.items()})
    v_star = orc.opt.model().eval(orc.cost, model_completion=True)
    orc.opt.pop()
    diffs = [abs(float(v_star.as_fraction()) - best)]

    builds = [("direct", direct_model)]
    if k <= ARRAY_BOOL_MAX_K:
        builds.append(("array_bool", array_bool_model))
    for label, build in builds:
        opt, cost = build(orc, x, key)
        row[f"{label}_s"], stats = timed_check(opt)
        row.update({f"{label}_{c}": v for c, v in stats.items()})
        diffs.append(abs(_value(opt, cost) - best))
    if k <= FIND_PROVE_MAX_K:
        row.update(find_prove(orc, x, key, v_star))
    row["abs_diff"] = max(diffs)
    return row


def sample_keys(k, n=4):
    """``n`` configurations ``(y0, y1)`` spread evenly over the k^2."""
    keys = [(y0, y1) for y0 in range(k) for y1 in range(k)]
    pos = ((np.arange(n) + 0.5) * len(keys) / n).astype(int)
    return [keys[i] for i in sorted(set(pos))]


def run_check(sizes, budget_s, path):
    import z3
    rows = []
    for k in sizes:
        make = PROBLEMS["iv"]["build"](k)
        points = []
        for sign in (+1, -1):
            prob = make(sign)
            rnd, x = mid_point(prob)
            orc = z3_oracle.Z3Oracle(prob, top_up=False)
            # z3_oracle.py:56-58, once per point as in a real call
            orc.opt.push()
            orc.opt.add(orc.lam_arr == orc._container(x[:prob.n_obs]),
                        orc.cost_0 == z3.RealVal(repr(float(x[-1]))))
            points.append((orc, rnd, x))
        t_size = time.perf_counter()
        for j, key in enumerate(sample_keys(k)):
            # the first configuration always runs, for both bounds
            if j and time.perf_counter() - t_size >= budget_s:
                break
            for orc, rnd, x in points:
                row = check_key(orc, rnd, x, key)
                rows.append(row)
                print(f"  [check k={k} sign={row['sign']:+d} key={key}] "
                      f"repo {fmt_time(row['repo_s'])} "
                      f"({row['repo_decisions']} decisions)  direct "
                      f"{fmt_time(row['direct_s'])}  brute force "
                      f"{fmt_time(row['bf_s'])}  |dv| {row['abs_diff']:.1e}",
                      flush=True)
        save_csv(rows, path)
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(rows, path):
    fields = list(dict.fromkeys(f for r in rows for f in r))
    with open(path, "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=fields, restval="")
        out.writeheader()
        out.writerows(rows)


def _num(v):
    if v == "":
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(vals)) if vals else None


def _t(v):
    return "–" if v is None else fmt_time(v)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def steps_table(steps, outdir):
    bench = {}
    for name in PROBLEMS:
        for r in load_csv(os.path.join(outdir, f"points_{name}.csv")):
            bench[(name, r["size"], r["sign"], r["round"])] = r["z3_call_s"]
    lines = [
        "| problem | size | per call | `check()` :65 "
        "| x → z3 array + assert :57-58 | read model :71-73 "
        "| push / pin / pop | `check()` share | benchmark, same point |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in PROBLEMS:
        size_name = PROBLEMS[name]["axis"].split()[0]
        for size in sorted({r["size"] for r in steps if r["problem"] == name}):
            g = [r for r in steps if r["problem"] == name and r["size"] == size]
            share = sum(r["check_s"] for r in g) / sum(r["total_s"] for r in g)
            same = [bench[key] for r in g
                    if (key := (name, size, r["sign"], r["round"])) in bench]
            lines.append(
                f"| {PROBLEMS[name]['title']} | {size_name}={size} "
                f"| {_t(_med(g, 'total_s'))} | {_t(_med(g, 'check_s'))} "
                f"| {_t(_med(g, 'array_s'))} | {_t(_med(g, 'read_s'))} "
                f"| {_t(_med(g, 'push_s'))} | {100 * share:.1f}% "
                f"| {_t(float(np.median(same)) if same else None)} |")
    return lines


def check_table(checks, outdir):
    bench = {}
    for r in load_csv(os.path.join(outdir, "points_iv.csv")):
        bench.setdefault(r["size"], []).append(r["z3_call_s"])
    lines = [
        "| k | checks timed | `check()` median | range | decisions "
        "| conflicts | Boolean vars | × k² checks / call "
        "| benchmark per call |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in sorted({r["k"] for r in checks}):
        g = [r for r in checks if r["k"] == k]
        t = [r["repo_s"] for r in g]
        lines.append(
            f"| {k} | {len(g)} | {fmt_time(np.median(t))} "
            f"| {fmt_time(min(t))} – {fmt_time(max(t))} "
            f"| {fmt_count(_med(g, 'repo_decisions'))} "
            f"| {fmt_count(_med(g, 'repo_conflicts'))} "
            f"| {fmt_count(_med(g, 'repo_mk_bool_var'))} "
            f"| {fmt_time(k * k * np.median(t))} "
            f"| {_t(float(np.median(bench[k])) if k in bench else None)} |")
    return lines


def encodings_table(checks):
    lines = [
        "| k | repo: Select at a solver-chosen index | array + Bool index "
        "| direct If | brute force | repo as find (sat) "
        "| repo as prove (unsat) | max abs Δ vs brute force |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in sorted({r["k"] for r in checks}):
        g = [r for r in checks if r["k"] == k]
        lines.append(
            f"| {k} | {_t(_med(g, 'repo_s'))} | {_t(_med(g, 'array_bool_s'))} "
            f"| {_t(_med(g, 'direct_s'))} | {_t(_med(g, 'bf_s'))} "
            f"| {_t(_med(g, 'find_s'))} | {_t(_med(g, 'prove_s'))} "
            f"| {max(r['abs_diff'] for r in g):.0e} |")
    return lines


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _time_axis(ax):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: fmt_time(v).replace(".0 ", " ")))


def _count_axis(ax):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_count(v)))


def plot_steps(steps, path):
    names = [n for n in PROBLEMS if any(r["problem"] == n for r in steps)]
    # check() is the story, so it alone is direct-labelled; the gray context
    # lines sit too close to label without collisions and go by the legend
    series = ((CHECK, "check_s", Z3_COLOR, "o"), (ARRAY, "array_s", MUTED, "s"),
              (READ, "read_s", MUTED, "^"), (PUSH, "push_s", MUTED, "D"))
    plt, fig, axes = _axes(len(names), 4.8)
    for ax, name in zip(axes, names):
        rows = [r for r in steps if r["problem"] == name]
        x = sorted({r["size"] for r in rows})
        for step, key, color, marker in series:
            y = [_med([r for r in rows if r["size"] == s], key) for s in x]
            _line(ax, x, y, color, f"{step}  (z3_oracle.py:{STEP_LINES[step]})",
                  marker)
            if step == CHECK:
                _end_label(ax, x[-1], y[-1], f"check() {fmt_time(y[-1])}")
        _style(ax, x, PROBLEMS[name]["title"], PROBLEMS[name]["axis"])
    _time_axis(axes[0])
    axes[0].set_ylabel("seconds per call in the step", color=MUTED, fontsize=9)
    _titles(fig, "Where one Z3 oracle call spends its time",
            "Z3Oracle.__call__ replayed line by line at the mid-trajectory "
            "IPM query point; median of the two bounds.  Log scale.")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=MUTED, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 0.86), w_pad=2.0)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_check(checks, path):
    ks = sorted({r["k"] for r in checks})
    panels = (("Time per check()", "repo_s", _time_axis, fmt_time),
              ("SAT decisions per check()", "repo_decisions", _count_axis,
               fmt_count),
              ("Boolean variables per check()", "repo_mk_bool_var",
               _count_axis, fmt_count))
    plt, fig, axes = _axes(len(panels), 4.4, sharey=False)
    for ax, (title, key, axis, fmt) in zip(axes, panels):
        groups = [[r[key] for r in checks if r["k"] == k] for k in ks]
        med = [float(np.median(g)) for g in groups]
        ax.fill_between(ks, [min(g) for g in groups], [max(g) for g in groups],
                        color=Z3_COLOR, alpha=0.10, linewidth=0, zorder=2)
        _line(ax, ks, med, Z3_COLOR)
        _end_label(ax, ks[-1], med[-1], fmt(med[-1]))
        _style(ax, ks, title, "k  (Z and Y bins)")
        axis(ax)
    _titles(fig, "Optimize.check() on its own  (IV, z3_oracle.py:65)",
            "One configuration (y0, y1) per check at the mid-trajectory IPM "
            "point of each bound; z3's own per-check counters.  Median, band "
            "min–max.  Log scale.")
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.5)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_encodings(checks, path):
    ks = sorted({r["k"] for r in checks})
    series = (("repo_s", Z3_COLOR, "Z3, repo: Select at a solver-chosen index",
               "repo"),
              ("array_bool_s", AQUA, "Z3, array + Bool-chosen index",
               "array + Bool"),
              ("direct_s", YELLOW, "Z3, direct: If over the two cells",
               "direct If"),
              ("bf_s", BF_COLOR, "Brute force", "brute force"))
    plt, fig, axes = _axes(1, 5.0)
    ax = axes[0]
    fig.set_size_inches(9.0, 5.0)
    for key, color, label, short in series:
        x = [k for k in ks if _med([r for r in checks if r["k"] == k], key)
             is not None]
        y = [_med([r for r in checks if r["k"] == k], key) for k in x]
        _line(ax, x, y, color, label)
        # a series that stops early would label on top of the longer lines
        _end_label(ax, x[-1], y[-1], f"{short} {fmt_time(y[-1])}",
                   dy=-12 if x[-1] < ks[-1] else 0)
    _style(ax, ks, "IV: one configuration (y0, y1), solved four ways",
           "k  (Z and Y bins)")
    ax.set_xlim(ks[0] - 0.4, ks[-1] + 2.6)
    _time_axis(ax)
    ax.set_ylabel("time per configuration", color=MUTED, fontsize=9)
    _titles(fig, "Inside check(): the array lookup is what costs",
            "Median over the timed configurations; every encoding returns "
            "the brute-force value.  Log scale.")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, fontsize=9, labelcolor=MUTED,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 0.84))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def report(outdir):
    steps = load_csv(os.path.join(outdir, "profile_steps.csv"))
    checks = load_csv(os.path.join(outdir, "profile_check.csv"))
    out = ["# Where the z3 oracle spends its time", ""]
    if steps:
        out += ["## 1. Every step of one call", ""] + steps_table(steps, outdir)
        plot_steps(steps, os.path.join(outdir, "z3_steps.png"))
    if checks:
        out += ["", "## 2. `Optimize.check()` on its own (IV)", ""]
        out += check_table(checks, outdir)
        out += ["", "## 3. The same configurations in other encodings (IV)",
                ""] + encodings_table(checks)
        plot_check(checks, os.path.join(outdir, "z3_check_isolated.png"))
        plot_encodings(checks, os.path.join(outdir, "z3_encodings.png"))
    text = "\n".join(out) + "\n"
    with open(os.path.join(outdir, "z3_profile.md"), "w") as fh:
        fh.write(text)
    print(text)


def cli():
    p = argparse.ArgumentParser(
        description="Where the z3 oracle spends its time.")
    p.add_argument("--part", choices=("steps", "check", "all"), default="all")
    p.add_argument("--problems", nargs="+", choices=list(PROBLEMS),
                   default=list(PROBLEMS), help="problems for the step replay")
    p.add_argument("--sizes", nargs="+", type=int, default=None,
                   help="IV sizes for the isolated check (default 2..10)")
    p.add_argument("--budget", type=float, default=90.0,
                   help="seconds per IV size after which no new "
                        "configuration starts (one always runs)")
    p.add_argument("--out", default=OUT)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.report_only:
        warm_up()
        if args.part in ("steps", "all"):
            run_steps(args.problems, os.path.join(args.out,
                                                  "profile_steps.csv"))
        if args.part in ("check", "all"):
            run_check(args.sizes or CHECK_SIZES, args.budget,
                      os.path.join(args.out, "profile_check.csv"))
    report(args.out)


if __name__ == "__main__":
    cli()
