"""
app_hook_robots.py -- application hook: multi-robot task assignment.

Where the DISCRETE PLAN IS THE DELIVERABLE: n robots must each be assigned
exactly one task (a fractional assignment is physically meaningless).  The
standard practice in multi-robot task allocation (MAMR / warehouse picking,
drone teaming) is exactly the Hungarian/auction family of exact solvers.

Per round: n new tasks appear; robots are assigned to them; robots move to
their tasks (position := task) before the next round.  Policies compared:
  exact   : LAPJV (Hungarian family) -- the industry default
  greedy  : cheapest-first (the common fast baseline)
  sghard  : entropic Sinkhorn (f=0.05, tol 1e-7) -> row-wise argmax hardening
            (the approximate-OT pipeline from the main study, Part A)
Metrics: episode total travel distance, cumulative gap vs exact, and the
wall time of the ASSIGNMENT STEP itself (the part a better solver controls).
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import otlib as ot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)

NS = (50, 200, 800)
SEEDS = (7, 8)
T = 15          # rounds per episode
F = 0.05        # sinkhorn regularization fraction (for sghard)
TOL = 1e-7      # marginal tol for sghard (the plan is hardened anyway)


def gen_cfg(dist, seed, n):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * 100, rng.random((n, 2)) * 100


def episode(n, seed):
    """Run one episode under all three policies. Returns a dict.

    Fairness: all policies share the SAME start positions and the SAME task
    stream (one task set per round); only the per-round assignment -- and
    hence the positions that result -- differ.
    """
    rng0 = np.random.default_rng(seed + 77)
    pos0 = rng0.random((n, 2)) * 100
    trng = np.random.default_rng(seed + 999)
    tasks_all = [trng.random((n, 2)) * 100 for _ in range(T)]
    pos = {p: pos0.copy() for p in ("exact", "greedy", "sghard")}
    acc = {p: 0.0 for p in pos}
    gap_rounds = {"greedy": [], "sghard": []}
    t_policy = {p: 0.0 for p in pos}
    for t in range(T):
        tasks = tasks_all[t]
        C = {p: ot.eucl2(pos[p], tasks) for p in pos}
        perms, costs = {}, {}
        # exact (reference for the round)
        t0 = time.perf_counter()
        ce, pe, _ = ot.exact_lapjv(C["exact"])
        t_policy["exact"] += time.perf_counter() - t0
        costs["exact"], perms["exact"] = ce, pe
        # greedy
        t0 = time.perf_counter()
        order = np.argsort(C["greedy"].ravel())
        perm = np.full(n, -1)
        used = np.zeros(n, bool)
        for flat in order:
            i, j = divmod(flat, n)
            if perm[i] == -1 and not used[j]:
                perm[i] = j
                used[j] = True
        costs["greedy"] = float(C["greedy"][np.arange(n), perm].sum())
        perms["greedy"] = perm
        t_policy["greedy"] += time.perf_counter() - t0
        # sghard
        t0 = time.perf_counter()
        s = ot.sinkhorn(C["sghard"], F * C["sghard"].mean(), tol=TOL,
                        maxit=200000, time_cap=300.0)
        cs, ps = ot.greedy_hard(s["P"], C["sghard"])
        costs["sghard"], perms["sghard"] = cs, ps
        t_policy["sghard"] += time.perf_counter() - t0
        for p in pos:
            acc[p] += costs[p]
            pos[p] = tasks[perms[p]]
        gap_rounds["greedy"].append(costs["greedy"] / ce - 1)
        gap_rounds["sghard"].append(costs["sghard"] / ce - 1)
    return {"n": n, "seed": seed, "acc": acc,
            "gap_greedy": acc["greedy"] / acc["exact"] - 1,
            "gap_sghard": acc["sghard"] / acc["exact"] - 1,
            "gap_round_greedy": gap_rounds["greedy"],
            "gap_round_sghard": gap_rounds["sghard"],
            "t_policy": t_policy}


def main():
    t_start = time.perf_counter()
    jpath = os.path.join(OUT, "app_hook.json")
    if os.path.exists(jpath):
        R = json.load(open(jpath))
        print("reusing results/app_hook.json (cached)")
        run = False
    else:
        R = []
        run = True
    if run:
        for n in NS:
            for seed in SEEDS:
                e = episode(n, seed)
                R.append(e)
                print("n=%3d seed=%d  exact=%9.1f  greedy=%9.1f (%+.2f%%)  "
                      "sghard=%9.1f (%+.2f%%)  t: exact %.2fs greedy %.3fs "
                      "sghard %.1fs"
                      % (n, seed, e["acc"]["exact"], e["acc"]["greedy"],
                         100 * e["gap_greedy"], e["acc"]["sghard"],
                         100 * e["gap_sghard"], e["t_policy"]["exact"],
                         e["t_policy"]["greedy"], e["t_policy"]["sghard"]),
                      flush=True)
                json.dump(R, open(jpath, "w"))

    # ------------------------------------------------------------------
    md = []
    md.append("# Application hook: multi-robot task assignment\n")
    md.append("The deliverable here is the *plan itself*: each of n robots must\n"
              "be matched to exactly one of n tasks; a fractional transport plan\n"
              "is physically meaningless (you cannot send half a robot).  This is\n"
              "the multi-robot task-allocation / warehouse-picking setting, whose\n"
              "standard practice is the Hungarian/auction family of *exact*\n"
              "solvers.\n")
    md.append("Episode: open field [0,100)^2, %d rounds; each round n fresh tasks\n"
              % T)
    md.append("appear and robots are (re)assigned, then move to their tasks.  "
              "Policies:\n")
    md.append("- **exact**: LAPJV (Hungarian family) -- the industry default;\n")
    md.append("- **greedy**: cheapest-first, the common fast baseline;\n")
    md.append("- **sghard**: entropic Sinkhorn (f=0.05, tol 1e-7) + row-wise\n"
              "  argmax hardening -- the approximate-OT pipeline of the main\n"
              "  study, end-to-end.\n")
    md.append("")
    md.append("| n | seed | travel exact | greedy | sghard | gap greedy | gap sghard "
              "| t exact | t greedy | t sghard |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for e in R:
        md.append("| %d | %d | %.1f | %.1f | %.1f | %+.2f%% | %+.2f%% | "
                  "%.2fs | %.3fs | %.1fs |"
                  % (e["n"], e["seed"], e["acc"]["exact"], e["acc"]["greedy"],
                     e["acc"]["sghard"], 100 * e["gap_greedy"],
                     100 * e["gap_sghard"], e["t_policy"]["exact"],
                     e["t_policy"]["greedy"], e["t_policy"]["sghard"]))
    md.append("")
    gg = [e["gap_greedy"] for e in R]
    sg = [e["gap_sghard"] for e in R]
    ex800 = [e for e in R if e["n"] == 800]
    te = sum(e["t_policy"]["exact"] for e in ex800) / (len(ex800) * T)
    tg = sum(e["t_policy"]["sghard"] for e in ex800) / (len(ex800) * T)
    md.append("Findings:\n")
    md.append("1. **The quality tax accumulates.** Per-round gaps compound\n"
              "   through robot positions: over %d rounds, greedy ends "
              "%+.1f to\n"
              % (T, 100 * min(gg)))
    md.append("   %+.1f%% and Sinkhorn-hardened %+.1f to %+.1f%% above the "
              "exact-assignment\n"
              % (100 * max(gg), 100 * min(sg), 100 * max(sg)))
    md.append("   policy, which is also the *best achievable* (optimal every "
              "round).\n")
    md.append("2. **Approximate is not even faster here.** The assignment step\n"
              "   at n=800 takes exact ~%.0f ms/round vs Sinkhorn+hardening "
              "~%.0f ms/round:\n"
              % (te * 1000, tg * 1000))
    md.append("   the approximate pipeline costs ~%.0fx the wall time of the "
              "exact one *and*\n"
              % (tg / te))
    md.append("   loses quality -- it is a tax on both time and distance in the\n"
              "   n <= 10^3 regime.\n")
    md.append("3. **The hook.** Where the discrete plan is the deliverable, "
              "the\n"
              "   matching-based exact solver is the right default up to the\n"
              "   memory wall: optimality is free (milliseconds per round at "
              "n=800), and\n"
              "   the approximate alternatives only win beyond it (n >~ 10^4), "
              "where a\n"
              "   fractional plan is also usually what the application can "
              "consume.\n")
    md.append("")
    md.append("Scope note: synthetic open field, no obstacles/kinematics; the "
              "point is the\n")
    md.append("structural one (discrete plan = deliverable), not a robotics "
              "benchmark.\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for pol, lab, c in (("exact", "exact (LAPJV)", "tab:blue"),
                        ("greedy", "greedy", "tab:orange"),
                        ("sghard", "Sinkhorn+hardening", "tab:red")):
        xs = [e["n"] for e in R]
        ys = [e["acc"][pol] for e in R]
        ax.scatter(xs, ys, label=lab, color=c)
        fit = np.polyfit(np.log(xs), np.log(ys), 1)
        xx = np.logspace(np.log10(40), np.log10(900), 50)
        ax.loglog(xx, np.exp(np.polyval(fit, np.log(xx))), color=c, lw=1)
    ax.set_xlabel("n robots / tasks")
    ax.set_ylabel("total travel distance (%d rounds)" % T)
    ax.set_title("episode cost: exact is the floor, the tax compounds")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ns = sorted(set(e["n"] for e in R))
    w = 0.27
    xpos = np.arange(len(ns))
    bars = (("exact", "exact (LAPJV)", "tab:blue"),
            ("greedy", "greedy", "tab:orange"),
            ("sghard", "Sinkhorn+hardening", "tab:red"))
    for i, (pol, lab, c) in enumerate(bars):
        vals = []
        for n in ns:
            sel = [e for e in R if e["n"] == n]
            vals.append(sum(e["t_policy"][pol] for e in sel) /
                        (len(sel) * T))
        ax.bar(xpos + (i - 1) * w, [max(v, 1e-4) for v in vals], w,
               label=lab, color=c)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["n=%d" % n for n in ns])
    ax.set_ylabel("assignment step (s/round, log)")
    ax.set_title("exact matching is milliseconds -- the tax is in quality, "
                 "not time")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "app_hook.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(OUT, "app_hook.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\nwrote %s" % os.path.join(OUT, "app_hook.md"))
    print("total wall time: %.1f min" % ((time.perf_counter() - t_start) / 60))


if __name__ == "__main__":
    main()
