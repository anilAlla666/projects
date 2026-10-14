# Stage 0 report (8 Oct 2026, provisional)

**The architecture trains correctly on this Mac, and bfloat16 is usable. The speed numbers are provisional:
they were measured while another session's test run was loading the machine (load average 4 to 7), and the
same configuration measured up to 1.7x apart between runs. The config decision below waits on a clean
re-measurement.**

## 1. Numeric check

`chunked_delta` against mlx-lm's step-by-step `gated_delta_ops` at the production head shape (T 512, 12 heads,
head dim 64):

| Gate floor | Max output difference | Max state difference |
|---:|---:|---:|
| −0.1 | 1.0e-7 | 5.4e-7 |
| −1.0 | 7.8e-8 | 8.6e-7 |

The unit tests also match gradients for all five inputs, and a 10% change in beta breaks the comparison.

## 2. Training throughput (contended)

From the first version of `scripts/stage0.py` (report `experiments/stage0/20261008T042613Z/stage0.json`),
printed verbatim:

| config | dtype | compiled | batch x seq | s/step | tokens/s | peak GB | days per 1B tokens |
|---|---|---|---|---:|---:|---:|---:|
| dev | float32 | yes | 8 x 2048 | 21.685 | 756 | 16.42 | 15.32 |
| dev | bfloat16 | yes | 8 x 2048 | 10.18 | 1,609 | 12.42 | 7.19 |
| dev | bfloat16 | no | 8 x 2048 | 9.912 | 1,653 | 9.28 | 7.0 |
| dev | bfloat16 | no | 32 x 512 | 8.694 | 1,885 | 9.02 | 6.14 |
| main | float32 | yes | 2 x 2048 | 13.128 | 312 | 12.92 | 37.1 |
| main | bfloat16 | yes | 2 x 2048 | 9.196 | 445 | 9.42 | 25.99 |
| main | bfloat16 | no | 2 x 2048 | 12.265 | 334 | 8.13 | 34.66 |
| main | bfloat16 | no | 8 x 512 | 10.192 | 402 | 7.4 | 28.8 |

The middle config (d_model 512, 8 layers, 44.5M parameters; `experiments/stage0/mid_config.json`): 1,971 tokens/s
compiled at 4 x 2048 and 2,486 tokens/s eager at 16 x 512. These rows used different tokens per step from the
DEV and MAIN rows, so the configs are not directly comparable here.

**Evidence of contention.** The same configuration measured differently between runs:
- DEV bfloat16 compiled at 8 x 2048: 8.709 s per step in the first run (its report file was overwritten by
  the rerun; the figure is from that run's printed output) against 10.18 s in the saved table.
- DEV bfloat16 eager at 8 x 2048: 5.70 s per step in an unsaved diagnostic run against 9.912 s in the saved
  table.

`uptime` showed load averages of 3.0 before and 4 to 7 during the runs.

**Compilation:** mixed. Compiled was slower than eager in the unsaved diagnostic run and faster for MAIN in the
saved table (445 against 334 tokens/s). There is no default until a clean run.

**Verdict forward time** at 8,192 tokens, batch 1, model only: DEV 0.52 to 1.89 s and MAIN 2.18 to 2.27 s across
the two runs.

## 3. bfloat16 is usable

DEV, one fixed batch (8 x 512), 50 eager steps (`experiments/stage0/bf16_check.json`):

| dtype | step 0 | step 10 | step 25 | step 50 |
|---|---:|---:|---:|---:|
| float32 | 10.896 | 0.056 | 0.018 | 0.011 |
| bfloat16 | 10.875 | 0.056 | 0.019 | 0.015 |

bfloat16 tracks float32 and roughly doubles throughput. This shows it fits a batch; whether it generalises as
well is checked during pretraining.

## 4. DECISION NEEDED: which config to pretrain (provisional)

The pretraining window is not fixed by the spec, so budgets are given for 3, 5 and 7 days at the saved bfloat16
rates above:

| Option | Parameters | Tokens/s (saved bfloat16 rows) | 3 days | 5 days | 7 days | Tokens per parameter at 7 days |
|---|---:|---:|---:|---:|---:|---:|
| A. Keep MAIN | 116M | 334 to 445 | 87M to 115M | 144M to 192M | 202M to 269M | 1.7 to 2.3 |
| B. Middle config | 44.5M | 1,971 to 2,486 | 511M to 644M | 851M to 1.07B | 1.19B to 1.50B | 27 to 34 |
| C. DEV only | 24M | 1,609 to 1,885 | 417M to 489M | 695M to 814M | 973M to 1.14B | 40 to 47 |

A common rule of thumb is about 20 tokens per parameter for a well-trained model. Option A stays far below it
in any of these windows; B and C reach it within 5 to 7 days.

## 5. The clean re-measurement

`scripts/stage0.py` now trains every row on 8,192 tokens per step (4 x 2048 or 16 x 512) for DEV, MID and MAIN,
times three repetitions per row and reports the median with its spread, and records the load average before and
after each row. Run it with nothing else using the Mac:

```bash
.venv/bin/python -u scripts/stage0.py --steps 3
```
