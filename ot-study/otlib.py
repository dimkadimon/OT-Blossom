"""
otlib.py -- exact (matching-based) optimal transport vs Sinkhorn.

Exact solvers:
  exact_lapjv          balanced unit-mass assignment (scipy LAPJV)
  sparse_assign        assignment on a sparse pool via Blossom VI (edge list)
  monotone_1d          1-D OT closed form (quantile map), O(n log n)
Approximate solver:
  sinkhorn             log-domain entropic Sinkhorn, proper marginal criterion
"""
import os
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

# Path to a tspkit Python shim that wraps the Blossom VI binary.
# Override with the TSPKIT environment variable, or install Blossom VI
# from https://github.com/Paul566/blossom-vi and point this at its src tree.
TSPKIT = os.environ.get(
    "TSPKIT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tsp-blossom", "src"),
)


def eucl2(x, y):
    """Cost matrix for 2-D points."""
    dx = x[:, None, 0] - y[None, :, 0]
    dy = x[:, None, 1] - y[None, :, 1]
    return np.sqrt(dx * dx + dy * dy)


def eucl2_mem(x, y):
    """Memory-frugal eucl2: one n^2 array + O(n) temps (norm identity)."""
    C = -2.0 * (x @ y.T)                    # n^2 float64
    C += (x ** 2).sum(1)[:, None]
    C += (y ** 2).sum(1)[None, :]
    np.maximum(C, 0.0, out=C)
    np.sqrt(C, out=C)
    return C


def eucl1(x, y):
    """Cost matrix for 1-D points."""
    return np.abs(x[:, None] - y[None, :])


# ----------------------------------------------------------------------
# exact
# ----------------------------------------------------------------------
def exact_lapjv(C):
    """Balanced unit-mass OT = min-cost assignment. Returns (cost, perm, secs)."""
    t0 = time.perf_counter()
    r, c = linear_sum_assignment(C)
    cost = float(C[r, c].sum())
    return cost, c, time.perf_counter() - t0


def knn_pool(C, k):
    """Symmetric k-NN pool as a boolean mask (kNN of both sides)."""
    n = C.shape[0]
    k = min(k, n - 1)
    keep = np.zeros((n, n), dtype=bool)
    for i in range(n):
        keep[i, np.argpartition(C[i], k + 1)[:k + 1]] = True
    return keep | keep.T


def sparse_assign(C, keep, timings=None):
    """
    Exact assignment restricted to the pool `keep` (non-pool edges forbidden),
    solved by Blossom VI on the bipartite edge list.
    Returns (cost, perm, secs, n_edges) or (None, None, None, n_edges) if
    the restricted problem has no perfect matching.
    """
    import sys
    if TSPKIT not in sys.path:
        sys.path.insert(0, TSPKIT)
    import tspkit as tk
    n = C.shape[0]
    big = int(np.max(C)) * 1000 * n + 1000
    edges = []
    for i in range(n):
        for j in np.flatnonzero(keep[i]):
            w = int(round(C[i, j] * 1000))
            if w > 2 ** 29 - 1:
                w = 2 ** 29 - 1
            edges.append((i, n + int(j), w))
    t0 = time.perf_counter()
    try:
        total, pairs = tk.solve_matching_edges(2 * n, edges, solver="vi",
                                               timings=timings)
    except RuntimeError:
        return None, None, time.perf_counter() - t0, len(edges)
    perm = np.full(n, -1)
    for u, v in pairs:
        if u < n and v >= n:
            perm[u] = v - n
    assert (perm >= 0).all() and sorted(perm) == list(range(n))
    cost = float(C[np.arange(n), perm].sum())
    return cost, perm, time.perf_counter() - t0, len(edges)


BLOSSOM_VI = os.path.join(os.path.dirname(TSPKIT), "bin", "blossom_vi")


def blossom_pool_duals(C, keep, timeout=120.0):
    """
    Exact assignment restricted to the pool `keep`, solved by Blossom VI,
    WITH the per-node dual certificate (MWM dual weights) of the pool problem.
    Returns (cost, perm, y, secs, n_edges); cost is None if infeasible or the
    solver timed out.  y has length 2n and satisfies y(u)+y(v) <= w(u,v) on
    every pool edge with sum(y) == cost (strong duality).
    """
    import subprocess
    n = C.shape[0]
    edges = []
    for i in range(n):
        for j in np.flatnonzero(keep[i]):
            w = int(round(C[i, j] * 1000))
            edges.append((i, n + int(j), min(w, 2 ** 29 - 1)))
    gfile = "/tmp/bv_cert_g.txt"
    mfile = "/tmp/bv_cert_m.txt"
    dfile = "/tmp/bv_cert_d.txt"
    with open(gfile, "w") as f:
        f.write("%d %d\n" % (2 * n, len(edges)))
        f.write("\n".join("%d %d %d" % e for e in edges))
        f.write("\n")
    t0 = time.perf_counter()
    try:
        r = subprocess.run([BLOSSOM_VI, "--duals", dfile, gfile, mfile],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None, None, time.perf_counter() - t0, len(edges)
    if r.returncode != 0:
        return None, None, None, time.perf_counter() - t0, len(edges)
    # The solver works on milliunit (1e-3-scaled, rounded) weights, so its
    # dual weights are in the same units; rescale to the float cost units.
    y = np.zeros(2 * n)
    with open(dfile) as f:
        for line in f:
            p = line.split()
            if p and p[0] != "dual_objective_quadrupled":
                y[int(p[0])] = int(p[1]) / 4.0 / 1000.0
    perm = np.full(n, -1)
    with open(mfile) as f:
        for line in f:
            u, v = line.split()
            u, v = int(u), int(v)
            if u < n and v >= n:
                perm[u] = v - n
            elif v < n and u >= n:
                perm[v] = u - n
    if not ((perm >= 0).all() and sorted(perm) == list(range(n))):
        return None, None, None, time.perf_counter() - t0, len(edges)
    cost = float(C[np.arange(n), perm].sum())
    return cost, perm, y, time.perf_counter() - t0, len(edges)


def dual_lower_bound(C, y):
    """
    Rigorous LOWER BOUND on the dense (full-graph) assignment optimum, from
    the POOL dual weights y (length 2n: source potentials ys, target
    potentials yt; ys_i + yt_j <= C_ij on POOL edges).

    One-sided repair (Proposition in the paper): clipping EITHER side alone
    against the unclipped opposite side yields a DENSE-feasible dual, since
        yt'(j) = min( yt(j), min_i [C(i,j) - ys(i)] )   =>
        ys(i) + yt'(j) <= C(i,j)  for all (i,j),
    and symmetrically for clipping ys.  Take the better of the two
    one-sided repairs; this dominates any sequential two-pass variant
    (the sequential one is one-sided B after an extra, objective-lowering
    clip).  Any dense-feasible dual is a lower bound on the dense optimum
    (weak duality), so the returned sum <= OPT_dense.
    """
    n = C.shape[0]
    ys, yt = y[:n], y[n:]
    # option A: clip target side only (against original source dual)
    ytp_A = np.minimum(yt, (C - ys[:, None]).min(axis=0))
    objA = float(ys.sum() + ytp_A.sum())
    # option B: clip source side only (against original target dual)
    ysp_B = np.minimum(ys, (C - yt[None, :]).min(axis=1))
    objB = float(ysp_B.sum() + yt.sum())
    if objA >= objB:
        return objA, ys.copy(), ytp_A
    return objB, ysp_B, yt.copy()


def entropic_lower_bound(C, reg, tol=1e-9, maxit=200000, time_cap=120.0):
    """
    LOWER BOUND on the dense assignment optimum from a CONVERGED entropic
    Sinkhorn run.  With P* the converged plan (row/col sums 1),
        V = <C,P*> + reg * sum(P* log P*)   <=   OPT
    because reg*sum(P log P) <= 0.  Valid only when converged; returns
    (lb, info) with lb = None otherwise.
    """
    s = sinkhorn(C, reg, tol=tol, maxit=maxit, time_cap=time_cap)
    if not s["converged"]:
        return None, s
    P = s["P"]
    lb = s["cost"] + reg * float((P * np.log(P)).sum())
    return lb, s


def pool_certified(C, k, reg=None, lb_ent=None, sink_tol=1e-9,
                   sink_maxit=200000, sink_cap=120.0, dual_timeout=120.0):
    """
    Pool-restricted exact assignment WITH a gap certificate.

    Returns a dict:
      ub, perm, t_pool   pool matching cost / permutation / time (feasible
                         dense solution  =>  OPT <= ub)
      lb                 max of the rigorous lower bounds (=>  lb <= OPT)
      lb_dual, lb_ent    the pool-dual-repair and entropic lower bounds
      gap_cert           (ub - lb) / lb   (guaranteed >= the true gap)
      n_pool_edges, status   status in {'ok','infeasible','timeout'}
    Certificate: lb <= OPT_dense <= ub,  so the pool solution is within
    gap_cert (relative) of the true optimum WITHOUT solving the dense problem.
    """
    n = C.shape[0]
    keep = knn_pool(C, k)
    ub, perm, y, t_pool, ne = blossom_pool_duals(C, keep,
                                                 timeout=dual_timeout)
    out = {"k": k, "n_pool_edges": ne, "t_pool": t_pool}
    if ub is None:
        out.update({"ub": None, "perm": None, "lb": None, "lb_dual": None,
                    "lb_ent": None, "gap_cert": None, "status": "infeasible"})
        return out
    # 1) pool-dual-repair lower bound (rigorous, always available)
    lb_dual, ysp, ytp = dual_lower_bound(C, y)
    out["lb_dual"] = lb_dual
    # 2) entropic lower bound (rigorous, if the Sinkhorn converges in time).
    #    Either pass a precomputed lb_ent, or a reg to run Sinkhorn now.
    if lb_ent is None and reg is not None:
        lb_ent, s = entropic_lower_bound(C, reg, tol=sink_tol,
                                         maxit=sink_maxit, time_cap=sink_cap)
        out["t_sink"] = s["secs"]
        out["sink_iters"] = s["iters"]
        out["sink_conv"] = s["converged"]
    out["lb_ent"] = lb_ent
    cands = [c for c in (lb_dual, lb_ent) if c is not None]
    lb = max(cands)
    # Absolute certificate (always meaningful): the solution is within
    # (ub - lb) of the optimum.  Relative versions: (ub-lb)/lb is the
    # conservative one (valid only when lb > 0); (ub-lb)/ub is always
    # meaningful (interval width as a fraction of the pool cost).
    gap_cert = (ub - lb) / lb if lb > 0 else None
    out.update({"ub": ub, "perm": perm, "lb": lb, "gap_cert": gap_cert,
                "gap_cert_ub": (ub - lb) / ub, "cert_abs": ub - lb,
                "status": "ok"})
    return out


def monotone_1d(x, y):
    """Exact 1-D OT: match sorted quantiles. Returns (cost, perm, secs)."""
    t0 = time.perf_counter()
    ox = np.argsort(x)
    oy = np.argsort(y)
    perm = oy[np.argsort(ox)]  # source rank i -> destination of same rank
    cost = float(np.abs(np.sort(x) - np.sort(y)).sum())
    return cost, perm, time.perf_counter() - t0


def greedy_hard(P, C):
    """Greedily turn a fractional plan into a permutation (per-row argmax)."""
    n = C.shape[0]
    Q = P.copy()
    perm = np.empty(n, dtype=int)
    for i in range(n):
        j = int(np.argmax(Q[i]))
        perm[i] = j
        Q[i, j] = -1
        Q[:, j] = -1
    return float(C[np.arange(n), perm].sum()), perm


# ----------------------------------------------------------------------
# sinkhorn (log domain, proper convergence test)
# ----------------------------------------------------------------------
def sinkhorn(C, reg, tol=1e-12, maxit=200000, check_every=4, time_cap=600.0):
    """
    Entropic Sinkhorn on cost C.  Converges when the implied transport plan
    has |marginals - 1| < tol (both sides).  Stops early at maxit or time_cap.
    Returns dict with iters, secs, cost = <P, C> of the resulting plan.
    Note: the final plan is feasible (marginals ~ 1) only if converged;
    otherwise cost is an upper bound from a partially-scaled plan and 'ok'=False.
    """
    n = C.shape[0]
    L = -C / reg
    u = np.zeros(n)
    v = np.zeros(n)
    it = 0
    t0 = time.perf_counter()
    ok = False
    while it < maxit and time.perf_counter() - t0 < time_cap:
        it += 1
        u = -logsumexp(L + v[None, :], axis=1)
        v = -logsumexp(L + u[:, None], axis=0)
        if it % check_every == 0:
            colsum = np.exp(v) * np.exp(logsumexp(L + u[:, None], axis=0))
            rowsum = np.exp(u) * np.exp(logsumexp(L + v[None, :], axis=1))
            if max(abs(colsum - 1).max(), abs(rowsum - 1).max()) < tol:
                ok = True
                break
    secs = time.perf_counter() - t0
    P = np.exp(L + u[:, None] + v[None, :])
    return {"iters": it, "secs": secs, "cost": float(P.ravel() @ C.ravel()),
            "converged": ok, "P": P}


def greenkhorn(C, reg, tol=1e-12, maxit=5000000, time_cap=600.0,
               cost_tol=1e-9, check_every=20000, feas_tol=0.02):
    """
    Greenkhorn (Altschuler-Weed-Rigollet, NeurIPS 2017, Algorithm 4),
    implemented in the log domain for numerical robustness at small reg.

    Greedy coordinate-descent Sinkhorn projection: at each step, pick the row
    or column with the largest rho(a,b) = b - a + a*log(a/b) marginal violation
    and update that single coordinate to its exact closed-form optimum
    (re-normalize that row/column to its target marginal). Marginal sums are
    maintained incrementally -> O(n) work per step (vs O(n^2) per
    row/column pass of vanilla Sinkhorn).

    Log-domain details: A0 = exp(logA) with logA fixed; scaling vectors x, y;
    log-marginals lr, lc (refreshed from scratch every check_every steps to
    reset incremental drift).  A coordinate update by f = exp(delta) updates
    each opposite-side marginal M via  M' = M + (f-1) A:  for f >= 1 that is a
    log-sum,  log M' = logaddexp(log M, logexpm1(delta) + log A)  (no ratio,
    overflow-proof);  for f < 1 it is  log M' = log M + log1p(expm1(delta) w),
    w = A/M in (0,1], with a drift guard.  Selection uses
    phi(u) = expm1(u) - u, u = log(n*M/M_target), which equals rho/n up to a
    per-side constant, so it is the same argmax as the paper.

    Same limit object as vanilla `sinkhorn` (Sinkhorn projection of
    A0 = exp(-(C - maxC)/reg)), hence the same unregularized OT plan as
    reg -> 0; differs only in how fast it gets there.

    Stopping: (a) marginal tol: max(|rowsums - 1|, |colsums - 1|) < tol
    (same units as sinkhorn()); OR (b) cost plateau: the plan cost is flat
    (rel change < cost_tol) for two consecutive checks AND the plan is
    essentially feasible (margdev < feas_tol) -> the entropic gap has
    converged even if marginals plateau at a coarser level (observed in the
    ill-conditioned small-reg regime).  The feas_tol gate matters: an
    infeasible plan's cost can sit BELOW the exact optimum, so a flat cost
    on a still-infeasible plan is not a converged gap.

    Returns dict: iters (= coordinate steps), secs, cost = <P, C>,
    converged (either stop rule), margdev (achieved max marginal deviation),
    stop ('tol' | 'plateau' | 'cap' | 'stall'), P.
    """
    n = C.shape[0]
    Csh = C - C.max()                            # shift invariance: plan unchanged
    K = -Csh / reg                               # log-kernel entries (<= 0)
    lA = K - _logsumexp2d(K)                     # logA0, sum(A0) = 1
    x = np.zeros(n)                              # row log-scaling
    y = np.zeros(n)                              # col log-scaling
    # initial log-marginals of A = exp(lA + x + y):
    #   lr = x + logsumexp_rows(lA, y);  lc = y + logsumexp_cols(lA, x)
    lr = x + _logsumexp_rows(lA, y)              # log row sums
    lc = y + _logsumexp_cols(lA, x)              # log col sums
    lnt = np.log(1.0 / n)                        # log of target marginal
    it = 0
    stop = "cap"
    last_cost = None
    flat = 0
    t0 = time.perf_counter()
    while it < maxit and time.perf_counter() - t0 < time_cap:
        it += 1
        # u_i = log(n * r_i(A));  violation phi(u) = expm1(u) - u (== rho/n).
        # phi is convex, min at u=0, decreasing on (-inf,0), increasing on
        # (0,inf)  =>  argmax phi over a set = the more-violated of {min, max}.
        u_r = lr - lnt
        i1, i2 = int(u_r.argmin()), int(u_r.argmax())
        p1, p2 = _phi1(u_r[i1]), _phi1(u_r[i2])
        I = i1 if p1 >= p2 else i2
        u_c = lc - lnt
        j1, j2 = int(u_c.argmin()), int(u_c.argmax())
        q1, q2 = _phi1(u_c[j1]), _phi1(u_c[j2])
        J = j1 if q1 >= q2 else j2
        if max(p1, p2, q1, q2) < 1e-14:          # float-floor stagnation
            stop = "stall"
            break
        if (p1 if p1 >= p2 else p2) > (q1 if q1 >= q2 else q2):
            # make row I exactly feasible; column j: c_j' = c_j + (f-1) A[I,j]
            delta = lnt - lr[I]                    # f = exp(delta) = (1/n)/rI
            if delta >= 0:                         # mass added: log of a sum
                lc = np.logaddexp(lc, _logexpm1(delta) + lA[I, :] + x[I] + y)
            else:                                  # mass removed: ratio form
                E = lA[I, :] + x[I] + y - lc       # log(A[I,j]/c_j), <= 0
                if E.max() > 1.0:                  # drift guard
                    lc = y + _logsumexp_cols(lA, x)
                    E = lA[I, :] + x[I] + y - lc
                lc = lc + np.log1p(np.expm1(delta) * np.exp(np.minimum(E, 0.0)))
            lr[I] = lnt
            x[I] += delta
        else:                                     # make column J exactly feasible
            delta = lnt - lc[J]                    # f = (1/n)/cJ
            if delta >= 0:                         # mass added: log of a sum
                lr = np.logaddexp(lr, _logexpm1(delta) + lA[:, J] + x + y[J])
            else:                                  # mass removed: ratio form
                E = lA[:, J] + x + y[J] - lr       # log(A[i,J]/r_i), <= 0
                if E.max() > 1.0:                  # drift guard
                    lr = x + _logsumexp_rows(lA, y)
                    E = lA[:, J] + x + y[J] - lr
                lr = lr + np.log1p(np.expm1(delta) * np.exp(np.minimum(E, 0.0)))
            lc[J] = lnt
            y[J] += delta
        if it % check_every == 0:
            # refresh log-marginals from scratch (resets incremental drift)
            lr = x + _logsumexp_rows(lA, y)
            lc = y + _logsumexp_cols(lA, x)
            md = max(np.abs(np.expm1(lr - lnt)).max(),
                     np.abs(np.expm1(lc - lnt)).max())
            if md < tol:
                stop = "tol"
                break
            # cost-plateau check (O(n^2) every check_every steps).  Only accept
            # the plateau once the plan is essentially feasible: an infeasible
            # plan's cost can sit BELOW the exact optimum, so a flat cost there
            # is not a converged gap.
            if md < feas_tol:
                Pc = np.exp(lA + x[:, None] + y[None, :])
                cc = float(Pc.ravel() @ C.ravel())
                if last_cost is not None:
                    if abs(cc - last_cost) <= cost_tol * max(1.0, abs(last_cost)):
                        flat += 1
                        if flat >= 2:
                            stop = "plateau"
                            break
                    else:
                        flat = 0
                last_cost = cc
    secs = time.perf_counter() - t0
    margdev = float(max(np.abs(np.expm1(lr - lnt)).max(),
                        np.abs(np.expm1(lc - lnt)).max()))
    # NOTE: scale to match sinkhorn()/exact convention: each marginal = 1,
    # total mass n, cost = <P, C> directly comparable to exact_lapjv's sum.
    P = n * np.exp(lA + x[:, None] + y[None, :])
    return {"iters": it, "secs": secs, "cost": float(P.ravel() @ C.ravel()),
            "converged": stop in ("tol", "plateau"), "margdev": margdev,
            "stop": stop, "P": P}


def _logexpm1(delta):
    """
    log(exp(delta) - 1) for delta >= 0, stable in both regimes:
      delta > log(2):  delta + log1p(-exp(-delta))
      delta <= log(2): log(expm1(delta))
    (delta = 0 gives -inf, which logaddexp treats as 'no mass added'.)
    """
    if delta > 0.6931471805599453:               # log(2)
        return float(delta + np.log1p(-np.exp(-delta)))
    return float(np.log(np.expm1(delta)))


def _phi1(u):
    """Scalar phi(u) = expm1(u) - u.  u is bounded here (marginals ~ O(1/n),
    total mass ~ n), so no overflow; for very negative u this is ~ -1 - u."""
    if u < -700.0:
        return -1.0 - u
    return np.expm1(u) - u


def _logsumexp2d(K):
    m = K.max()
    return float(m + np.log(np.exp(K - m).sum()))


def _logsumexp_rows(lA, y):
    """log_i sum_j exp(lA_ij + y_j)."""
    M = (lA + y[None, :]).max(axis=1)
    return M + np.log(np.exp(lA + y[None, :] - M[:, None]).sum(axis=1))


def _logsumexp_cols(lA, x):
    """log_j sum_i exp(lA_ij + x_i)."""
    M = (lA + x[:, None]).max(axis=0)
    return M + np.log(np.exp(lA + x[:, None] - M[None, :]).sum(axis=0))



