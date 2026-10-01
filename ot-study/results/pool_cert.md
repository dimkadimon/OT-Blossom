# Pool-OT solver with a rigorous gap certificate

Question: can the pool solver *certify* how close its solution is
to the dense optimum, without solving the dense problem?
Answer: yes -- via duality.

**Certificate (no dense solve involved).**
- UB: exact assignment on the kNN pool (Blossom VI): a feasible
  dense solution, so OPT <= UB.
- LB: two rigorous lower bounds on the dense optimum, take max:
  1. **pool dual repair.** Blossom VI's MWM dual of the pool
     problem (node weights y, sum y = UB, y(u)+y(v) <= w(u,v) on
     pool edges) is repaired to DENSE feasibility in one O(n^2)
     pass: y'(v) = min(y(v), min_u [C(u,v) - y(u)]).  Then
     y'(u)+y'(v) <= C(u,v) for ALL (u,v) (y'(v) <= C(u,v)-y(u)
     and y'(u) <= y(u)), so sum y' <= OPT.
  2. **entropic dual.** value of a converged Sinkhorn run,
     V = <C,P*> + reg*sum(P* log P*) <= OPT (since sum P log P <= 0).
- Hence LB <= OPT <= UB and the pool solution is within
  gap_cert = (UB - LB)/LB of the true optimum, certified without
  ever solving the dense problem.  (Costs are milliunit-rounded,
  as in B1/B3.)

Ground truth (dense exact via LAPJV) is used ONLY to validate the
certificate, not to compute it.

## Certificate table (gap_cert must always be >= true gap)

| config | n | exact | k | UB (true gap) | LB | LB dual | LB entropic | gap_cert | t_pool |

|---|---|---|---|---|---|---|---|---|---|

| 1d+noise 7 | 1000 | 1534.9 | 16 | 1570.8 (+2.336%) | 1381.0 | 1381.0 | -4462.3 | +13.737% | 0.07s |
| 1d+noise 7 | 1000 | 1534.9 | 32 | 1541.5 (+0.432%) | 1512.5 | 1512.5 | -4462.3 | +1.922% | 0.15s |
| 1d+noise 7 | 1000 | 1534.9 | 64 | 1536.7 (+0.115%) | 1531.0 | 1531.0 | -4462.3 | +0.368% | 0.23s |
| 1d+noise 8 | 1000 | 1260.5 | 16 | 1291.6 (+2.466%) | 1091.7 | 1091.7 | -4367.7 | +18.307% | 0.06s |
| 1d+noise 8 | 1000 | 1260.5 | 32 | 1268.7 (+0.647%) | 1223.2 | 1223.2 | -4367.7 | +3.721% | 0.13s |
| 1d+noise 8 | 1000 | 1260.5 | 64 | 1261.7 (+0.090%) | 1257.7 | 1257.7 | -4367.7 | +0.315% | 0.26s |
| 1d+noise 9 | 1000 | 1134.7 | 16 | 1165.8 (+2.739%) | 963.5 | 963.5 | -4546.3 | +20.991% | 0.05s |
| 1d+noise 9 | 1000 | 1134.7 | 32 | 1140.2 (+0.480%) | 1117.2 | 1117.2 | -4546.3 | +2.059% | 0.10s |
| 1d+noise 9 | 1000 | 1134.7 | 64 | 1135.3 (+0.054%) | 1132.6 | 1132.6 | -4546.3 | +0.237% | 0.20s |
| 1d+noise 10 | 1000 | 2039.3 | 16 | 2080.3 (+2.013%) | 1832.0 | 1832.0 | -4260.7 | +13.555% | 0.11s |
| 1d+noise 10 | 1000 | 2039.3 | 32 | 2052.8 (+0.662%) | 1986.4 | 1986.4 | -4260.7 | +3.343% | 0.19s |
| 1d+noise 10 | 1000 | 2039.3 | 64 | 2043.1 (+0.187%) | 2026.9 | 2026.9 | -4260.7 | +0.797% | 0.42s |
| clustered 7 | 1000 | 4381.0 | 16 | 4440.0 (+1.346%) | 4166.0 | 4166.0 | -2840.6 | +6.577% | 0.06s |
| clustered 7 | 1000 | 4381.0 | 32 | 4393.4 (+0.284%) | 4348.3 | 4348.3 | -2840.6 | +1.038% | 0.10s |
| clustered 7 | 1000 | 4381.0 | 64 | 4383.2 (+0.051%) | 4375.8 | 4375.8 | -2840.6 | +0.170% | 0.17s |
| clustered 8 | 1000 | 4451.4 | 16 | 4577.5 (+2.831%) | 4086.6 | 4086.6 | -2977.1 | +12.011% | 0.08s |
| clustered 8 | 1000 | 4451.4 | 32 | 4503.5 (+1.170%) | 4331.1 | 4331.1 | -2977.1 | +3.982% | 0.13s |
| clustered 8 | 1000 | 4451.4 | 64 | 4459.7 (+0.186%) | 4430.8 | 4430.8 | -2977.1 | +0.653% | 0.25s |
| clustered 9 | 1000 | 4661.6 | 16 | 4727.5 (+1.413%) | 4462.0 | 4462.0 | - | +5.949% | 0.07s |
| clustered 9 | 1000 | 4661.6 | 32 | 4683.4 (+0.468%) | 4578.4 | 4578.4 | - | +2.294% | 0.10s |
| clustered 9 | 1000 | 4661.6 | 64 | 4670.4 (+0.188%) | 4626.7 | 4626.7 | - | +0.945% | 0.26s |
| clustered 10 | 1000 | 3592.3 | 16 | 3730.3 (+3.841%) | 3090.8 | 3090.8 | -3818.8 | +20.690% | 0.06s |
| clustered 10 | 1000 | 3592.3 | 32 | 3685.7 (+2.599%) | 3246.2 | 3246.2 | -3818.8 | +13.538% | 0.12s |
| clustered 10 | 1000 | 3592.3 | 64 | 3658.8 (+1.850%) | 3442.8 | 3442.8 | -3818.8 | +6.274% | 0.23s |
| uniform 7 | 1000 | 3107.0 | 16 | 3123.3 (+0.524%) | 3043.0 | 3043.0 | -3125.6 | +2.637% | 0.04s |
| uniform 7 | 1000 | 3107.0 | 32 | 3107.3 (+0.010%) | 3105.1 | 3105.1 | -3125.6 | +0.072% | 0.07s |
| uniform 7 | 1000 | 3107.0 | 64 | 3107.0 (+0.000%) | 3106.8 | 3106.8 | -3125.6 | +0.007% | 0.14s |
| uniform 8 | 1000 | 3289.1 | 16 | 3316.8 (+0.840%) | 3142.1 | 3142.1 | -2869.4 | +5.557% | 0.06s |
| uniform 8 | 1000 | 3289.1 | 32 | 3290.7 (+0.048%) | 3280.1 | 3280.1 | -2869.4 | +0.323% | 0.09s |
| uniform 8 | 1000 | 3289.1 | 64 | 3289.2 (+0.001%) | 3288.9 | 3288.9 | -2869.4 | +0.009% | 0.17s |
| uniform 9 | 1000 | 3223.7 | 16 | 3250.9 (+0.844%) | 3095.0 | 3095.0 | -2969.3 | +5.037% | 0.05s |
| uniform 9 | 1000 | 3223.7 | 32 | 3225.9 (+0.068%) | 3215.4 | 3215.4 | -2969.3 | +0.328% | 0.09s |
| uniform 9 | 1000 | 3223.7 | 64 | 3223.8 (+0.002%) | 3223.4 | 3223.4 | -2969.3 | +0.011% | 0.18s |
| uniform 10 | 1000 | 3360.8 | 16 | 3397.6 (+1.095%) | 3184.8 | 3184.8 | -3026.0 | +6.681% | 0.05s |
| uniform 10 | 1000 | 3360.8 | 32 | 3365.0 (+0.126%) | 3341.5 | 3341.5 | -3026.0 | +0.704% | 0.07s |
| uniform 10 | 1000 | 3360.8 | 64 | 3360.8 (+0.001%) | 3360.6 | 3360.6 | -3026.0 | +0.007% | 0.15s |
| 1d+noise 7 | 2000 | 1633.8 | 16 | 1671.8 (+2.327%) | 1470.9 | 1470.9 | - | +13.659% | 0.16s |
| 1d+noise 7 | 2000 | 1633.8 | 32 | 1642.1 (+0.511%) | 1602.5 | 1602.5 | - | +2.470% | 0.27s |
| 1d+noise 7 | 2000 | 1633.8 | 64 | 1635.0 (+0.075%) | 1629.1 | 1629.1 | - | +0.363% | 0.73s |
| clustered 7 | 2000 | 5620.8 | 16 | 5818.5 (+3.517%) | 3593.0 | 3593.0 | - | +61.942% | 0.19s |
| clustered 7 | 2000 | 5620.8 | 32 | 5718.0 (+1.729%) | 4467.0 | 4467.0 | - | +28.006% | 0.32s |
| clustered 7 | 2000 | 5620.8 | 64 | 5642.7 (+0.389%) | 5342.6 | 5342.6 | - | +5.617% | 0.65s |
| uniform 7 | 2000 | 4146.4 | 16 | 4170.5 (+0.581%) | 4011.1 | 4011.1 | -10523.4 | +3.975% | 0.08s |
| uniform 7 | 2000 | 4146.4 | 32 | 4147.7 (+0.032%) | 4142.8 | 4142.8 | -10523.4 | +0.118% | 0.15s |
| uniform 7 | 2000 | 4146.4 | 64 | 4146.5 (+0.003%) | 4145.7 | 4145.7 | -10523.4 | +0.019% | 0.37s |

(gap_cert = (UB-LB)/LB, conservative; when the LB is non-positive
the relative-to-UB width (UB-LB)/UB is shown instead -- the
absolute interval [LB, UB] is the certificate either way.)

- k=16: certified width (UB-LB)/UB = 2.57 to 38.25% across
  all 15 configs (true gaps 0.52 to 3.84%): the certificate
  always contains the truth.

- k=32: certified width (UB-LB)/UB = 0.07 to 21.88% across
  all 15 configs (true gaps 0.01 to 2.60%): the certificate
  always contains the truth.

- k=64: certified width (UB-LB)/UB = 0.01 to 5.90% across
  all 15 configs (true gaps 0.00 to 1.85%): the certificate
  always contains the truth.


Findings:

1. **The certificate is valid on 45/45 cells** (LB <= exact <= UB and,
   where LB > 0, gap_cert >= true gap; asserted in the study script).

2. **Tight where the pool is tight.** At k=32 the certified width is
   <= 0.70% of the solution cost on the uniform configs (true gap <=
   0.13%) and <= 0.02% at k=64; on the clustered and 1-D+noise
   configs the dense repair is conservative (width up to 21.9% at k=32
   vs true gap up to 2.60%).  It is still a *valid* bound there, and
   the width tightens monotonically in k on every config.

3. **The pool-dual-repair bound dominates the entropic one** whenever the
   pool is decent (k >= 16): it is derived from the exact pool dual and
   costs one O(n^2) pass (~ms at n=2000) with no Sinkhorn run. The
   entropic bound (f=0.05 here) is often negative/vacuous on these
   instances -- a labeled cross-checking floor, not the certificate.

4. **Cost of certification is negligible:** pool solve + dual repair is
   sub-second to ~1 s, i.e. certification is *cheaper than solving the dense
   problem* (0.71 s at n=2000) and gives bounds the dense solve only
   gives after the fact.

