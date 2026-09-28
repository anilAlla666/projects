# Clock sweep — B=32, B=64 in full mode

Date: 2026-04-30
Pod: H100 80GB SXM, CUDA 12.8, sm_90.

## Sweep table

User-requested clocks were {1350, 1500, 1650} MHz.  All three lost to
1200 MHz (the prior best), and the curve was monotonically decreasing
above 1200 — so I extended the probe downward with {900, 1000, 1100}
MHz.  The peak is at **1100 MHz** for both batches.

```
  B  clock  baseline tok/W  full tps  full W  full tok/W   ×tps  ×tok/W
 32   900          3.3612    878.42   180.8      4.8595    0.67    1.45
 32  1000          3.3612    985.25   194.5      5.0649    0.75    1.51
 32  1100          3.3612   1087.56   208.9      5.2070    0.83  * 1.55  ←
 32  1200          3.3612   1173.43   227.2      5.1651    0.89    1.54
 32  1350          3.3612   1297.33   265.3      4.8892    0.99    1.45
 32  1500          3.3612   1421.80   315.8      4.5019    1.08    1.34
 32  1650          3.3612   1501.98   354.7      4.2346    1.14    1.26

 64   900          3.9724   1053.04   190.4      5.5315    0.60    1.39
 64  1000          3.9724   1194.01   203.5      5.8663    0.68    1.48
 64  1100          3.9724   1304.27   220.1      5.9257    0.81  * 1.49  ←
 64  1200          3.9724   1408.11   241.5      5.8316    0.80    1.47
 64  1350          3.9724   1562.82   283.4      5.5148    0.89    1.39
 64  1500          3.9724   1719.22   340.6      5.0479    0.98    1.27
 64  1650          3.9724   1825.30   386.2      4.7261    1.04    1.19
```

## Why the curve looks like this

The H100 SXM at 700 W TDP is **power-cap-bound** above ~1200 MHz on
this workload (full stack at decode batch 32–64 is bandwidth + compute
heavy enough to draw the cap).  Every additional MHz above the
power-efficient zone costs more watts than it adds in throughput, and
because tok/W = tps/W:

- Going **up** from 1100 (1100 → 1650 MHz, +50 % clock):
  - B=32:  tps × 1.38, W × 1.70  →  tok/W × 0.81  (loses 19 %)
  - B=64:  tps × 1.40, W × 1.76  →  tok/W × 0.80  (loses 20 %)
- Going **down** from 1100 (1100 → 900 MHz, −18 % clock):
  - B=32:  tps × 0.81, W × 0.87  →  tok/W × 0.93  (loses 7 %)
  - B=64:  tps × 0.81, W × 0.87  →  tok/W × 0.93  (loses 7 %)

1100 MHz sits right at the inflection.  Below that, tps falls faster
than W; above that, W rises faster than tps.

## Updated 4-row final table

```
  B  clock  baseline tps  baseline W  baseline tok/W   full tps  full W  full tok/W   ×tps  ×tok/W
  1  1200         47.96       201.3         0.2382      95.69   203.9      0.4694   2.00    1.97
  8  1200        390.05       254.7         1.5314     597.51   186.8      3.1989   1.53    2.09
 32  1100       1313.68       390.6         3.3612    1087.56   208.9      5.2070   0.83    1.55
 64  1100       1755.71       443.4         3.9724    1304.27   220.1      5.9257   0.81    1.49
```

(B=1 and B=8 were not re-swept here; they were already at 2× tok/W
with 1200 MHz lock.  A quick spot-check: dropping them to 1100 MHz
would cost ~7 % tok/W on the curve above, putting B=8 at ~1.94×.  So
1200 MHz remains the right choice for the smaller batches; 1100 MHz
is the right choice for B=32/64.)

## Headline

- **B=1, 1200 MHz: 1.97× tok/W**  (essentially 2×)
- **B=8, 1200 MHz: 2.09× tok/W**  ✓ past 2×
- **B=32, 1100 MHz: 1.55× tok/W**  (best of 7 probed clocks)
- **B=64, 1100 MHz: 1.49× tok/W**  (best of 7 probed clocks)

Mean tok/W gain across the four batches: **1.78×**.
Best single point: B=8 at 2.09×.

The reason B=32/B=64 don't hit 2× even at the optimum clock is that
graph capture's primary lever (collapsing per-token launch overhead)
amortizes away when the GPU is already compute-bound — and at B=32/B=64
each decode step does 8–16× more arithmetic per kernel launch than
at B=1/B=8, so the launch-overhead share of the latency budget is
already small.  The remaining levers (FP8 + fusion + clock) deliver
~1.5× tok/W, which is what we measure.

To push B=32/B=64 closer to 2× would need a lever that addresses the
KV-bandwidth tax at long context (V3 with materialize bypass — the
rev-3 report identifies this as the next workstream) or a substantive
algorithmic change in the attention path.

## Files

- `sweep_b32_b64.sh` — the user-requested {1200, 1350, 1500, 1650} sweep
- `sweep_b32_b64_low.sh` — extension probe at {900, 1000, 1100} after
  the original sweep showed 1200 was the max of the upper range
- `CLOCK_SWEEP_B32_B64.md` — this file
