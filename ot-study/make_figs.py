"""Regenerate paper figures from cached JSON at high (300+ dpi) resolution."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "paper", "figs")
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(os.path.dirname(__file__), "results")

# ---------------------------------------------------------------------------
# Figure 1: Exact vs Sinkhorn vs Greenkhorn on dense uniform grid, with rough
# extrapolations to 1% gap target.
# ---------------------------------------------------------------------------
A = json.load(open(os.path.join(RES, "ot_study.json")))["A"]
fit = json.load(open(os.path.join(RES, "ot_study.json")))["fit"]
GA = json.load(open(os.path.join(RES, "ot-study", "results", "greenkhorn_A.json"))) if False else json.load(open(os.path.join(RES, "ot_study.json")))["GA"]

ns = sorted(int(k) for k in A.keys() if k.isdigit())
exact_t = [A[str(n)]["exact_t"] for n in ns]
# Use f=0.01 (largest converged) Sinkhorn/Greenkhorn times
sink_t = []
green_t = []
for n in ns:
    a = A[str(n)]
    sink_t.append(a.get("f0.0100", {}).get("t", np.nan))
    g = GA[str(n)].get("f0.0100", {}) if str(n) in GA else {}
    green_t.append(g.get("t", np.nan))

# Rough extrapolation to 1% gap. Values are reported explicitly in
# Table 3 of the paper (rounded to one or two significant figures). For
# visual consistency the dashed extrapolation lines continue the
# empirical log-log slope observed for f=0.01 on the measured points
# (Sinkhorn ~n^1.9, Greenkhorn ~n^1.6), and star markers at each n
# indicate the f* points from Table 3 (which additionally account for
# extra iterations needed at smaller regularization). Convert to
# seconds to match the measured-series units.
ns_arr = np.array(ns, dtype=float)
mask_s = ~np.isnan(sink_t_arr := np.array(sink_t, dtype=float))
mask_g = ~np.isnan(green_t_arr := np.array(green_t, dtype=float))
slope_V = np.polyfit(np.log(ns_arr[mask_s]), np.log(sink_t_arr[mask_s]), 1)[0]
slope_G = np.polyfit(np.log(ns_arr[mask_g]), np.log(green_t_arr[mask_g]), 1)[0]

# Anchor the slope-continuation lines at the largest-n f=0.01 measured
# point for each method and extend with the empirical slope.
n_anchor_V, t_anchor_V = ns_arr[mask_s][-1], sink_t_arr[mask_s][-1]
n_anchor_G, t_anchor_G = ns_arr[mask_g][-1], green_t_arr[mask_g][-1]
ns_slope = np.array([n_anchor_V, 8000], dtype=float)
tV_slope = t_anchor_V * (ns_slope/n_anchor_V)**slope_V
ns_slope_G = np.array([n_anchor_G, 8000], dtype=float)
tG_slope = t_anchor_G * (ns_slope_G/n_anchor_G)**slope_G

# f* points (Table 3)
ns_ext = np.array([500,1000,2000,4000,8000], dtype=float)
tG_ext = np.array([10, 12, 20, 40, 80], dtype=float) * 60.0       # seconds
tV_ext = np.array([200, 700, 3000, 10000, 50000], dtype=float) * 60.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

ax1.loglog(ns, exact_t, "o-", color="C2", lw=2, label="Exact (LAPJV)")
ax1.loglog(ns, sink_t, "s-", color="C0", lw=2, label="Sinkhorn (f=0.01, gap 5–9%)")
ax1.loglog(ns, green_t, "^-", color="C3", lw=2, label="Greenkhorn (f=0.01, gap 5–9%)")
ax1.loglog(ns_slope, tV_slope, "C0--", lw=1.2, alpha=0.6)
ax1.loglog(ns_slope_G, tG_slope, "C3--", lw=1.2, alpha=0.6)
ax1.loglog(ns_ext, tV_ext, "C0*", ms=10, label="Sinkhorn extrap. to 1%")
ax1.loglog(ns_ext, tG_ext, "C3*", ms=10, label="Greenkhorn extrap. to 1%")
ax1.set_xlabel("n"); ax1.set_ylabel("wall time (s)")
ax1.set_title("Time (log-log)"); ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=7.5, loc="upper left")

# Quality: gap vs n at f=0.01; include exact (gap=0)
sink_gap = [A[str(n)].get("f0.0100", {}).get("gap", np.nan) for n in ns]
green_gap = []
for n in ns:
    g = GA[str(n)].get("f0.0100", {}) if str(n) in GA else {}
    green_gap.append(g.get("gap", np.nan))
ax2.semilogx(ns, [0]*len(ns), "o-", color="C2", lw=2, label="Exact (0%)")
ax2.semilogx(ns, sink_gap, "s-", color="C0", lw=2, label="Sinkhorn (f=0.01)")
ax2.semilogx(ns, green_gap, "^-", color="C3", lw=2, label="Greenkhorn (f=0.01)")
ax2.axhline(1.0, ls=":", color="k", alpha=0.5, label="1% target")
ax2.set_xlabel("n"); ax2.set_ylabel("gap to exact (%)")
ax2.set_title("Quality (semi-log)"); ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "crossover.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUT, "crossover.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved crossover.pdf/png")

# ---------------------------------------------------------------------------
# Figure 2: certificate widths per configuration.
# ---------------------------------------------------------------------------
pc = json.load(open(os.path.join(RES, "pool_cert.json")))
# pc: dict of config_name -> {k: {LB, UB, OPT, width, true_gap, ...}}
cfgs = sorted(pc.keys())
ks = [16, 32, 64]
families = ["uniform", "clustered", "1d+noise"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# Left: bar chart per config (grouped by family) with dot for true gap
import numpy as np
bar_w = 0.25
colors = ["#4C72B0", "#DD8452", "#55A868"]
xpos = np.arange(len(cfgs))
for ki, k in enumerate(ks):
    widths = []
    trues = []
    for c in cfgs:
        kd = pc[c]["k"][str(k)]
        widths.append(100*kd["gap_cert_ub"])
        trues.append(100*(kd["ub"]/pc[c]["exact"] - 1.0))
    ax1.bar(xpos + (ki-1)*bar_w, widths, width=bar_w, color=colors[ki], label=f"k={k}")
    ax1.plot(xpos + (ki-1)*bar_w, trues, "k.", ms=3)
ax1.set_xticks(xpos)
ax1.set_xticklabels([c.split("_")[0][:3]+"_"+c.split("_")[1] for c in cfgs], rotation=60, fontsize=7)
ax1.set_ylabel("certified width (%) / true gap (dots)")
ax1.set_title("Certificate width per configuration")
ax1.legend(fontsize=8); ax1.grid(True, axis="y", alpha=0.3)

# Right: width vs k averaged by family
for fi, fam in enumerate(families):
    fam_cfgs = [c for c in cfgs if c.startswith(fam)]
    w_mean = []
    for k in ks:
        ws = [100*pc[c]["k"][str(k)]["gap_cert_ub"] for c in fam_cfgs if str(k) in pc[c]["k"]]
        w_mean.append(np.nanmean(ws))
    ax2.semilogy(ks, w_mean, "o-", color=colors[fi], lw=2, label=fam)
ax2.set_xlabel("k"); ax2.set_ylabel("mean certified width (%)")
ax2.set_title("Width vs k (monotone on every config)")
ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "pool_cert.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUT, "pool_cert.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved pool_cert.pdf/png")

# ---------------------------------------------------------------------------
# Figure 3: multi-robot episode travel and per-round assignment time.
# ---------------------------------------------------------------------------
app = json.load(open(os.path.join(RES, "app_hook.json")))
# app is a list; inspect structure
print("app type:", type(app), "len:", len(app) if hasattr(app,'__len__') else '?')
if isinstance(app, dict):
    print("app keys:", list(app.keys()))
    policies = app.get("policies", app)
else:
    print("app[0] keys:", list(app[0].keys()) if app else "empty")

# ---------------------------------------------------------------------------
# Figure 3: app hook (episode travel and per-round times)
# ---------------------------------------------------------------------------
app = json.load(open(os.path.join(RES, "app_hook.json")))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ns = sorted(set(r["n"] for r in app))
seeds = sorted(set(r["seed"] for r in app))
for pol, color, marker in [("exact","C2","o"),("greedy","C1","s"),("sghard","C3","^")]:
    # average over seeds
    means = []
    for n in ns:
        vals = [r["acc"][pol] for r in app if r["n"]==n]
        means.append(np.mean(vals))
    ax1.plot(ns, means, marker=marker, color=color, lw=2, label=pol)
ax1.set_xlabel("n"); ax1.set_ylabel("episode travel distance over 15 rounds")
ax1.set_title("Cumulative travel distance")
ax1.set_xscale("log"); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=9)

for pol, color, marker in [("exact","C2","o"),("greedy","C1","s"),("sghard","C3","^")]:
    means = []
    for n in ns:
        vals = [r["t_policy"][pol] for r in app if r["n"]==n]
        means.append(np.mean(vals))
    ax2.loglog(ns, means, marker=marker, color=color, lw=2, label=pol)
ax2.set_xlabel("n"); ax2.set_ylabel("per-round assignment time (s)")
ax2.set_title("Per-round assignment cost (log scale)")
ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "app_hook.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUT, "app_hook.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved app_hook.pdf/png")
