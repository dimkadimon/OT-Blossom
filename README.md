# Exact Optimal Transport by Matching

This repository contains the code, data, and paper source for

> **Exact Optimal Transport by Matching**
> Dmitry Kamenetsky (with an Arena.ai Agent Mode autonomous software agent
> directing the algorithm design, coding, experiments, analysis, and writing).
> arXiv:2610.04085, October 2026.

- **arXiv preprint:** <https://arxiv.org/abs/2610.04085>
- **Compiled PDF in this repo:** [`paper/exact_ot_blossom.pdf`](paper/exact_ot_blossom.pdf)
- **Source code repository:** <https://github.com/dimkadimon/OT-Blossom>

## What it does

Balanced discrete optimal transport between $n$ sources and $n$ targets of
unit mass is exactly the minimum-cost assignment problem (a bipartite
perfect matching). This repo benchmarks three ways of solving it:

1. **Exact, dense** -- SciPy's Jonker--Volgenant `linear_sum_assignment`
   (shortest-augmenting path; roughly cubic scaling empirically, $O(n^2)$ memory).
2. **Exact, sparse** -- a $k$NN-pool solver using Blossom VI
   (Arkhipov & Kolmogorov, 2026) as the matching engine, plus a cheap
   dual-repair *gap certificate* that rigorously bounds distance from the
   dense optimum in one $O(n^2)$ pass.
3. **Approximate** -- vanilla log-domain Sinkhorn and Greenkhorn (Altschuler,
   Weed & Rigollet, 2017; Lin, Ho & Jordan, 2022).

It also contains a **multi-robot task allocation** demonstration where the
discrete transport plan itself is the deliverable.

The headline empirical finding is that for $n \le 10^4$ on 2-D Euclidean
instances the exact solver is both faster and strictly more accurate than
either Sinkhorn variant at the $1\%$ quality level, while the sparse $k$NN
pool solver with the dual-repair certificate delivers sub-percent gaps in
well under a second.

## Layout

```
ot-study/
  otlib.py                  -- core library: exact (LAPJV, Blossom pool, 1-D
                              quantile map), Sinkhorn, Greenkhorn, dual repair,
                              entropic floor, combined certificate
  run_ot_study.py           -- Parts A/B benchmarks -> results/ot_study.json
  run_greenkhorn_study.py   -- Greenkhorn-vs-Sinkhorn sweep -> results/
  run_pool_cert_study.py    -- 15-config x 3-k certificate study ->
                              results/pool_cert.json (asserts invariants)
  app_hook_robots.py        -- multi-robot task-allocation experiment ->
                              results/app_hook.json
  make_figs.py              -- regenerates paper/figs/*.pdf from cached JSON
  verify_pool_ub.py         -- independent SciPy verification of pool UB costs
  results/                  -- cached JSON outputs for all runs (all seeds,
                              all families, all configurations reported
                              in the paper); re-rendering scripts load these
paper/
  exact_ot_blossom.tex      -- LaTeX source of the paper
  exact_ot_blossom.pdf      -- compiled PDF
  figs/                     -- figures (vector PDF)
```

## Requirements

- Python $\ge$ 3.10 with `numpy`, `scipy`, `matplotlib`.
- For the sparse $k$NN-pool solves you need a working **Blossom VI** binary
  (P. Arkhipov & V. Kolmogorov, <https://github.com/Paul566/blossom-vi>,
  arXiv:2604.20351). Our driver in `otlib.py` talks to it via a `tspkit`
  Python shim that loads the edge-list interface with the `--duals` flag;
  point the `TSPKIT` constant in `otlib.py` at your build. Dense exact,
  Sinkhorn, Greenkhorn, and the 1-D solver do not require Blossom.

## Running

From `ot-study/`:

```bash
python run_ot_study.py           # Parts A/B (dense exact + Sinkhorn grid)
python run_greenkhorn_study.py   # Greenkhorn head-to-head
python run_pool_cert_study.py    # 15x3 certificate study (needs Blossom VI)
python app_hook_robots.py        # multi-robot application
python make_figs.py              # regenerate paper/figs/ from cached JSON
```

Each script caches results to JSON under `results/` and re-renders in
seconds on a re-run. A cold re-run takes roughly 25 min (Greenkhorn grid),
20 min (certificate study), and 6 min (application hook) on the 2-core /
2-GB sandbox used in the paper.

## Data

All measured data reported in the paper is cached as JSON under
[`ot-study/results/`](ot-study/results/):

- `ot_study.json` -- dense exact times and vanilla Sinkhorn runs (Parts A/B).
- `greenkhorn_A.json`, partial files -- Greenkhorn-vs-Sinkhorn head-to-head.
- `pool_cert.json` -- 15 configurations $\times$ 3 pool sizes for the
  certificate study (pool UB, repaired dual LB, entropic LB, true exact).
- `app_hook.json` -- multi-robot episode totals and per-policy timings.

Running the scripts with the cached JSON present reproduces every table
and figure in seconds without needing Blossom VI.

## Certificate

Given a pool matching with upper bound cost UB and Blossom duals
$(\alpha, \beta)$, the lower bound is

$$
\hat\beta_j = \min(\beta_j, \min_i (C_{ij} - \alpha_i)), \quad
\hat\alpha_i = \min(\alpha_i, \min_j (C_{ij} - \beta_j)),
$$
$$
\mathrm{LB}_1 = \max\Big(\sum_i\alpha_i + \sum_j\hat\beta_j,\;
                        \sum_i\hat\alpha_i + \sum_j\beta_j\Big),
$$

which satisfies $\mathrm{LB}_1 \le \mathrm{OPT} \le \mathrm{UB}$ without
solving the dense problem (Proposition 3 / Theorem 1 of the paper). An
entropic lower bound from a Sinkhorn run is combined as
$\max(\mathrm{LB}_1, W(\varepsilon))$, though $\mathrm{LB}_1$ dominates
everywhere we measured.

## Reproducing the paper

Run the four scripts above to regenerate all JSON outputs under
`results/`; then `python make_figs.py` regenerates the figures. Compile
`paper/exact_ot_blossom.tex` with a standard LaTeX toolchain
(`pdflatex`; BibTeX is not needed -- the paper uses a `thebibliography`
environment).

## License

MIT. See `LICENSE`.

## Citing

If you use this code, the empirical findings, or the dual-repair
certificate, please cite the arXiv preprint:

```bibtex
@misc{kamenetsky2026exact,
  title         = {Exact Optimal Transport by Matching},
  author        = {Kamenetsky, Dmitry},
  year          = {2026},
  eprint        = {2610.04085},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DS},
  url           = {https://arxiv.org/abs/2610.04085},
  note          = {Algorithm design, implementation, experiments, analysis,
                   and writing carried out by an Arena.ai Agent Mode
                   autonomous software agent under the author's direction;
                   source code: https://github.com/dimkadimon/OT-Blossom}
}
```

A plain-text reference for acknowledgements/footnote use:

> Dmitry Kamenetsky. *Exact Optimal Transport by Matching.*
> arXiv:2610.04085 [cs.DS], October 2026.
> <https://arxiv.org/abs/2610.04085>
