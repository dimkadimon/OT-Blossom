import json, sys, os, time
import numpy as np
sys.path.insert(0, '/home/user/OT-Blossom/ot-study')
import otlib as ot

R = json.load(open('/home/user/OT-Blossom/ot-study/results/pool_cert.json'))

KS = (16, 32, 64)
CFG_1000 = [(d, s) for d in ("uniform", "clustered", "1d+noise") for s in (7,8,9,10)]
CFG_2000 = [(d, 7) for d in ("uniform", "clustered", "1d+noise")]

def gen_cfg(dist, seed, n):
    if dist == "uniform":
        rng = np.random.default_rng(seed)
        return rng.random((n,2))*100, rng.random((n,2))*100
    if dist == "clustered":
        rng = np.random.default_rng(seed)
        cx = rng.random((8,2))*100
        idx = np.repeat(np.arange(8), n//8+1)[:n]
        x = cx[idx] + rng.normal(0,8.0,(n,2))
        cy = cx + rng.normal(0,3.0,(8,2))
        idx2 = np.repeat(np.arange(8), n//8+1)[:n]
        y = cy[idx2] + rng.normal(0,8.0,(n,2))
        return x, y
    if dist == "1d+noise":
        rng = np.random.default_rng(seed)
        t = rng.random(n)*100
        x = np.stack([t + rng.normal(0,2.0,n), rng.normal(0,2.0,n)], 1)
        rng2 = np.random.default_rng(seed+1000)
        s = rng2.random(n)*100
        y = np.stack([s + rng2.normal(0,2.0,n), rng2.normal(0,2.0,n)], 1)
        return x, y

from scipy.optimize import linear_sum_assignment

worst_rel = 0.0
worst_cell = None
ok = 0
bad = 0
for n, cfgs in ((1000, CFG_1000), (2000, CFG_2000)):
    for dist, seed in cfgs:
        key = f"{dist}_{seed}_{n}"
        e = R[key]
        x, y = gen_cfg(dist, seed, n)
        C = ot.eucl2(x, y)
        # Original driver used k+1 per side (off-by-one). Reproduce it.
        for k in KS:
            keep = np.zeros((n,n), dtype=bool)
            for i in range(n):
                keep[i, np.argpartition(C[i], k+1)[:k+1]] = True
            keep = keep | keep.T
            ne = keep.sum()
            # Build submatrix restricted to pool edges, with large cost elsewhere
            BIG = C.max() * n * 10
            Csub = np.where(keep, C, BIG)
            t0 = time.perf_counter()
            ri, ci = linear_sum_assignment(Csub)
            t_scipy = time.perf_counter()-t0
            cost = float(C[ri, ci].sum())
            cached = e["k"][str(k)]["ub"]
            rel = abs(cost - cached)/max(abs(cached), 1)
            if rel > worst_rel:
                worst_rel = rel; worst_cell = (key, k, cached, cost, ne)
            if rel < 1e-9:
                ok += 1
            else:
                bad += 1
                if bad <= 5:
                    print(f"MISMATCH {key} k={k} cached={cached:.2f} scipy={cost:.2f} rel={rel:.3e} edges={ne}")
print(f"\nChecked cells: {ok+bad}; perfect matches: {ok}; mismatches: {bad}")
print(f"Worst rel error: {worst_rel:.3e} at {worst_cell}")
