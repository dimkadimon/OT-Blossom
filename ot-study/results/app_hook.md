# Application hook: multi-robot task assignment

The deliverable here is the *plan itself*: each of n robots must
be matched to exactly one of n tasks; a fractional transport plan
is physically meaningless (you cannot send half a robot).  This is
the multi-robot task-allocation / warehouse-picking setting, whose
standard practice is the Hungarian/auction family of *exact*
solvers.

Episode: open field [0,100)^2, 15 rounds; each round n fresh tasks

appear and robots are (re)assigned, then move to their tasks.  Policies:

- **exact**: LAPJV (Hungarian family) -- the industry default;

- **greedy**: cheapest-first, the common fast baseline;

- **sghard**: entropic Sinkhorn (f=0.05, tol 1e-7) + row-wise
  argmax hardening -- the approximate-OT pipeline of the main
  study, end-to-end.


| n | seed | travel exact | greedy | sghard | gap greedy | gap sghard | t exact | t greedy | t sghard |

|---|---|---|---|---|---|---|---|---|---|

| 50 | 7 | 10232.8 | 12353.3 | 10930.4 | +20.72% | +6.82% | 0.00s | 0.015s | 1.8s |
| 50 | 8 | 10531.8 | 12357.2 | 11134.9 | +17.33% | +5.73% | 0.00s | 0.014s | 2.3s |
| 200 | 7 | 21835.6 | 28718.1 | 24955.1 | +31.52% | +14.29% | 0.03s | 0.300s | 10.4s |
| 200 | 8 | 23804.5 | 30051.8 | 26802.5 | +26.24% | +12.59% | 0.03s | 0.302s | 9.0s |
| 800 | 7 | 47327.7 | 64986.7 | 59959.2 | +37.31% | +26.69% | 1.02s | 3.222s | 237.9s |
| 800 | 8 | 49050.9 | 66519.9 | 61229.6 | +35.61% | +24.83% | 0.91s | 3.473s | 222.1s |

Findings:

1. **The quality tax accumulates.** Per-round gaps compound
   through robot positions: over 15 rounds, greedy ends +17.3 to

   +37.3% and Sinkhorn-hardened +5.7 to +26.7% above the exact-assignment

   policy, which is also the *best achievable* (optimal every round).

2. **Approximate is not even faster here.** The assignment step
   at n=800 takes exact ~64 ms/round vs Sinkhorn+hardening ~15336 ms/round:

   the approximate pipeline costs ~239x the wall time of the exact one *and*

   loses quality -- it is a tax on both time and distance in the
   n <= 10^3 regime.

3. **The hook.** Where the discrete plan is the deliverable, the
   matching-based exact solver is the right default up to the
   memory wall: optimality is free (milliseconds per round at n=800), and
   the approximate alternatives only win beyond it (n >~ 10^4), where a
   fractional plan is also usually what the application can consume.


Scope note: synthetic open field, no obstacles/kinematics; the point is the

structural one (discrete plan = deliverable), not a robotics benchmark.
