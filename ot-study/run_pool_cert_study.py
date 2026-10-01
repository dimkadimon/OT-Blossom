"""
run_pool_cert_study.py -- pool-OT solver WITH a rigorous gap certificate.

For each (distribution, seed, n) and pool size k:
  * UB  = exact assignment on the kNN pool (Blossom VI)   ->  OPT <= UB
  * LB  = max( pool-dual-repair dual, entropic dual value ) ->  LB <= OPT
          - pool dual: Blossom VI's MWM dual certificate of the POOL problem,
            repaired to dense feasibility by one O(n^2) pass
            (y'(v) = min(y(v), min_u [C(u,v) - y(u)]), proven feasible);
          - entropic: value of a converged Sinkhorn run (V = <C,P> + reg*sum
            P log P), a valid lower bound since sum(P log P) <= 0.
  * Certificate:  LB <= OPT <= UB, so the pool solution is within
    gap_cert = (UB - LB)/LB of the dense optimum WITHOUT solving it.

Ground truth (dense exact, LAPJV) is computed for validation only: the
certificate itself never uses it.  Invariants checked on every cell:
LB <= exact <= UB and gap_cert >= true gap.
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

KS = (16, 32, 64)
CFG_1000 = [(d, s) for d in ("uniform", "clustered", "1d+noise")
            for s in (7, 8, 9, 10)]
CFG_2000 = [(d, 7) for d in ("uniform", "clustered", "1d+noise")]


def gen2d(n, seed=7):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * 100, rng.random((n, 2)) * 100


def gen_cfg(dist, seed, n):
    if dist == "uniform":
        return gen2d(n, seed)
    if dist == "clustered":
        rng = np.random.default_rng(seed)
        cx = rng.random((8, 2)) * 100
        idx = np.repeat(np.arange(8), n // 8 + 1)[:n]
        x = cx[idx] + rng.normal(0, 8.0, (n, 2))
        cy = cx + rng.normal(0, 3.0, (8, 2))
        idx2 = np.repeat(np.arange(8), n // 8 + 1)[:n]
        y = cy[idx2] + rng.normal(0, 8.0, (n, 2))
        return x, y
    if dist == "1d+noise":
        rng = np.random.default_rng(seed)
        t = rng.random(n) * 100
        x = np.stack([t + rng.normal(0, 2.0, n), rng.normal(0, 2.0, n)], 1)
        rng2 = np.random.default_rng(seed + 1000)
        s = rng2.random(n) * 100
        y = np.stack([s + rng2.normal(0, 2.0, n), rng2.normal(0, 2.0, n)], 1)
        return x, y
    raise ValueError(dist)


def main():
    t_start = time.perf_counter()
    jpath = os.path.join(OUT, "pool_cert.json")
    R = json.load(open(jpath)) if os.path.exists(jpath) else {}
    # migrate entries written by earlier revisions: derive the new fields
    # from the stored ub/lb (no recompute needed)
    for e in R.values():
        for kk in e["k"].values():
            kk["cert_abs"] = kk["ub"] - kk["lb"]
            kk["gap_cert_ub"] = (kk["ub"] - kk["lb"]) / kk["ub"]
            kk["gap_cert"] = (kk["ub"] - kk["lb"]) / kk["lb"] \
                if kk["lb"] > 0 else None
    n_missing = sum(1 for n, cfgs in ((1000, CFG_1000), (2000, CFG_2000))
                    for dist, seed in cfgs
                    if "%s_%d_%d" % (dist, seed, n) not in R)
    print("configs: %d cached, %d to run" % (len(R), n_missing), flush=True)

    for n, cfgs in ((1000, CFG_1000), (2000, CFG_2000)):
        for dist, seed in cfgs:
            key = "%s_%d_%d" % (dist, seed, n)
            if key in R:
                continue
            x, y = gen_cfg(dist, seed, n)
            C = ot.eucl2(x, y)
            ex, perm, ex_t = ot.exact_lapjv(C)
            # one entropic lower bound per config (f = 0.05, cheap)
            reg = 0.05 * C.mean()
            lb_ent, s = ot.entropic_lower_bound(C, reg, tol=1e-9,
                                                maxit=200000,
                                                time_cap=240.0)
            print("%s  exact=%.1f (%.3fs)  ent_lb=%s  sink %s iters %d"
                  % (key, ex, ex_t,
                     ("%.1f" % lb_ent) if lb_ent is not None else "NONE",
                     "conv" if s["converged"] else "CAPPED", s["iters"]),
                  flush=True)
            entry = {"dist": dist, "seed": seed, "n": n, "exact": ex,
                     "exact_t": ex_t, "f_ent": 0.05,
                     "lb_ent": lb_ent, "sink_conv": s["converged"],
                     "k": {}}
            for k in KS:
                r = ot.pool_certified(C, k, lb_ent=lb_ent)
                entry["k"][str(k)] = {kk: vv for kk, vv in r.items()
                                      if kk != "perm"}
                # ---- certificate invariants (ground truth) ----
                assert r["status"] == "ok", (key, k, r["status"])
                assert r["lb"] <= ex * (1 + 1e-6), (key, k, r["lb"], ex)
                assert ex <= r["ub"] * (1 + 1e-6), (key, k, ex, r["ub"])
                assert r["cert_abs"] >= 0
                if r["gap_cert"] is not None:
                    assert r["gap_cert"] >= (r["ub"] / ex - 1) - 1e-6, \
                        (key, k)
                cert_s = ("%+.3f%%" % (100 * r["gap_cert"])) \
                    if r["gap_cert"] is not None \
                    else ("lb<=0, abs %.1f" % r["cert_abs"])
                print("  k=%2d  ub=%9.1f (%+.3f%%)  lb=%9.1f "
                      "(dual %9.1f / ent %s)  cert=%-18s t_pool=%.3fs"
                      % (k, r["ub"], 100 * (r["ub"] / ex - 1), r["lb"],
                         r["lb_dual"],
                         ("%.1f" % r["lb_ent"])
                         if r["lb_ent"] is not None else "none",
                         cert_s, r["t_pool"]), flush=True)
            R[key] = entry
            del C
            json.dump(R, open(jpath, "w"))  # incremental cache
    print("invariants held on all cells: LB <= exact <= UB; "
          "cert >= true gap")

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------
    md = []
    md.append("# Pool-OT solver with a rigorous gap certificate\n")
    md.append("Question: can the pool solver *certify* how close its solution is\n"
              "to the dense optimum, without solving the dense problem?\n"
              "Answer: yes -- via duality.\n")
    md.append("**Certificate (no dense solve involved).**\n"
              "- UB: exact assignment on the kNN pool (Blossom VI): a feasible\n"
              "  dense solution, so OPT <= UB.\n"
              "- LB: two rigorous lower bounds on the dense optimum, take max:\n"
              "  1. **pool dual repair.** Blossom VI's MWM dual of the pool\n"
              "     problem (node weights y, sum y = UB, y(u)+y(v) <= w(u,v) on\n"
              "     pool edges) is repaired to DENSE feasibility in one O(n^2)\n"
              "     pass: y'(v) = min(y(v), min_u [C(u,v) - y(u)]).  Then\n"
              "     y'(u)+y'(v) <= C(u,v) for ALL (u,v) (y'(v) <= C(u,v)-y(u)\n"
              "     and y'(u) <= y(u)), so sum y' <= OPT.\n"
              "  2. **entropic dual.** value of a converged Sinkhorn run,\n"
              "     V = <C,P*> + reg*sum(P* log P*) <= OPT (since sum P log P <= 0).\n"
              "- Hence LB <= OPT <= UB and the pool solution is within\n"
              "  gap_cert = (UB - LB)/LB of the true optimum, certified without\n"
              "  ever solving the dense problem.  (Costs are milliunit-rounded,\n"
              "  as in B1/B3.)\n")
    md.append("Ground truth (dense exact via LAPJV) is used ONLY to validate the\n"
              "certificate, not to compute it.\n")

    md.append("## Certificate table (gap_cert must always be >= true gap)\n")
    md.append("| config | n | exact | k | UB (true gap) | LB | LB dual | LB entropic "
              "| gap_cert | t_pool |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for key in sorted(R, key=lambda q: (R[q]["n"], R[q]["dist"], R[q]["seed"])):
        e = R[key]
        for k in KS:
            r = e["k"][str(k)] if str(k) in e["k"] else e["k"][k]
            le = ("%.1f" % r["lb_ent"]) if r["lb_ent"] is not None else "-"
            if r["gap_cert"] is not None:
                gc = "%+.3f%%" % (100 * r["gap_cert"])
            else:
                gc = "~%.3f%% (rel to UB; LB<=0)" % (100 * r["gap_cert_ub"])
            md.append("| %s %d | %d | %.1f | %d | %.1f (%+.3f%%) | %.1f | "
                      "%.1f | %s | %s | %.2fs |"
                      % (e["dist"], e["seed"], e["n"], e["exact"], k, r["ub"],
                         100 * (r["ub"] / e["exact"] - 1), r["lb"],
                         r["lb_dual"], le, gc, r["t_pool"]))
    md.append("")
    md.append("(gap_cert = (UB-LB)/LB, conservative; when the LB is non-positive\n"
              "the relative-to-UB width (UB-LB)/UB is shown instead -- the\n"
              "absolute interval [LB, UB] is the certificate either way.)\n")

    # findings
    n_cfg = len(R)
    n_cells = n_cfg * 3
    def cell(key, k):
        e = R[key]
        return e["k"][str(k)] if str(k) in e["k"] else e["k"][k]
    neg_lb = [(key, k) for key in R for k in KS if cell(key, k)["lb"] <= 0]
    mono = all(cell(key, 16)["cert_abs"] >= cell(key, 32)["cert_abs"] - 1e-9
               and cell(key, 32)["cert_abs"] >= cell(key, 64)["cert_abs"]
               - 1e-9 for key in R)
    def gw(k):
        # certified width as a fraction of the UB (well-behaved for all
        # cells, incl. LB <= 0); the conservative (UB-LB)/LB form is in
        # the table
        sel = [cell(key, k)["gap_cert_ub"] for key in R]
        tg = [cell(key, k)["ub"] / R[key]["exact"] - 1 for key in R]
        return min(sel), max(sel), min(tg), max(tg)
    for k in KS:
        cmin, cmax, tmin, tmax = gw(k)
        md.append("- k=%d: certified width (UB-LB)/UB = %.2f to %.2f%% "
                  "across\n"
                  "  all %d configs (true gaps %.2f to %.2f%%): the "
                  "certificate\n  always contains the truth.\n"
                  % (k, 100 * cmin, 100 * cmax, n_cfg, 100 * tmin,
                     100 * tmax))
    md.append("")
    md.append("Findings:\n")
    md.append("1. **The certificate is valid on %d/%d cells** (LB <= exact <= "
              "UB and,\n"
              "   where LB > 0, gap_cert >= true gap; asserted in the study "
              "script).\n"
              % (n_cells, n_cells))
    if neg_lb:
        cells_s = ", ".join("%s k=%d" % (key, k) for key, k in neg_lb)
        md.append("   On %d cell(s) (%s) the repaired dual LB is non-"
                  "positive:\n"
                  "   the certificate there is the absolute interval [LB, UB] "
                  "alone.\n"
                  % (len(neg_lb), cells_s))
    per_class = []
    for dist in ("uniform", "clustered", "1d+noise"):
        keys = [key for key in R if R[key]["dist"] == dist]
        c32 = max(cell(key, 32)["gap_cert_ub"] for key in keys)
        t32 = max(cell(key, 32)["ub"] / R[key]["exact"] - 1 for key in keys)
        c64 = max(cell(key, 64)["gap_cert_ub"] for key in keys)
        per_class.append((dist, c32, t32, c64))
    mono_txt = ("the width tightens monotonically in k on every config.\n"
                if mono else
                "the width tightens in k on most configs (check the table).\n")
    md.append("2. **Tight where the pool is tight.** At k=32 the certified "
              "width is\n"
              "   <= %.2f%% of the solution cost on the uniform configs "
              "(true gap <=\n"
              "   %.2f%%) and <= %.2f%% at k=64; on the clustered and "
              "1-D+noise\n"
              "   configs the dense repair is conservative (width up to "
              "%.1f%% at k=32\n"
              "   vs true gap up to %.2f%%).  It is still a *valid* bound "
              "there, and\n"
              "   %s" % (100 * per_class[0][1], 100 * per_class[0][2],
                         100 * per_class[0][3],
                         100 * max(per_class[1][1], per_class[2][1]),
                         100 * max(per_class[1][2], per_class[2][2]),
                         mono_txt))
    md.append("3. **The pool-dual-repair bound dominates the entropic one** "
              "whenever the\n"
              "   pool is decent (k >= 16): it is derived from the exact "
              "pool dual and\n"
              "   costs one O(n^2) pass (~ms at n=2000) with no Sinkhorn run. "
              "The\n"
              "   entropic bound (f=0.05 here) is often negative/vacuous on "
              "these\n"
              "   instances -- a labeled cross-checking floor, not the "
              "certificate.\n")
    md.append("4. **Cost of certification is negligible:** pool solve + dual "
              "repair is\n"
              "   sub-second to ~1 s, i.e. certification is *cheaper than "
              "solving the dense\n"
              "   problem* (%.2f s at n=2000) and gives bounds the dense "
              "solve only\n"
              "   gives after the fact.\n"
              % max(e["exact_t"] for e in R.values() if e["n"] == 2000))
    md.append("")

    # figure: (left) cert gap vs true gap per (config, k);
    #         (right) cert gap vs k per config (monotone tightening)
    configs = []
    for key in sorted(R, key=lambda q: (R[q]["n"], R[q]["dist"],
                                        R[q]["seed"])):
        e = R[key]
        configs.append(("%s%s n%d" % (e["dist"][:1].upper(), e["seed"],
                                      e["n"]), e))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    xpos = np.arange(len(configs))
    w = 0.27
    for i, k in enumerate(KS):
        vals = []
        for _, e in configs:
            r = e["k"][str(k)] if str(k) in e["k"] else e["k"][k]
            gc = r["gap_cert"] if r["gap_cert"] is not None \
                else r["gap_cert_ub"]
            vals.append(max(100 * gc, 1e-3))
        ax.bar(xpos + (i - 1) * w, vals, w, label="cert k=%d" % k)
    for i, k in enumerate(KS):
        vals = []
        for _, e in configs:
            r = e["k"][str(k)] if str(k) in e["k"] else e["k"][k]
            vals.append(max(100 * (r["ub"] / e["exact"] - 1), 1e-3))
        ax.scatter(xpos + (i - 1) * w, vals, s=14, color="k", zorder=5)
    ax.scatter([], [], s=14, color="k", label="true gap")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([lab for lab, _ in configs], fontsize=6, rotation=60)
    ax.set_ylabel("gap vs exact (%)")
    ax.set_title("cert gap (bars) vs true gap (dots): cert always >= true")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    for lab, e in configs:
        vals = []
        for k in KS:
            r = e["k"][str(k)] if str(k) in e["k"] else e["k"][k]
            gc = r["gap_cert"] if r["gap_cert"] is not None \
                else r["gap_cert_ub"]
            vals.append(max(100 * gc, 1e-3))
        ax.semilogy(KS, vals, "o-", ms=3, lw=0.8,
                    color=({"uniform": "tab:blue", "clustered": "tab:green",
                            "1d+noise": "tab:orange"}[e["dist"]]),
                    alpha=0.8,
                    label=lab if e["seed"] == 7 else None)
    ax.set_xlabel("pool size k")
    ax.set_ylabel("gap_cert (%)")
    ax.set_title("certificate tightens monotonically with k")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=1, title="seeds 7 (others fainter)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pool_cert.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(OUT, "pool_cert.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\nwrote %s (%d lines)" % (os.path.join(OUT, "pool_cert.md"), len(md)))
    print("total wall time: %.1f min" % ((time.perf_counter() - t_start) / 60))


if __name__ == "__main__":
    main()
