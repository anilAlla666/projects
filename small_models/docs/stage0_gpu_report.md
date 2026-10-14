# Stage 0 on the A100 (8 Oct 2026)

**On one A100 40GB, every config trains fast enough to reach 20 tokens per parameter in hours, for a few
dollars: about 0.9 h for DEV, 2.3 h for MID and 11.8 h for MAIN, compiled. Compiling helps by 25 to 35%. The
config decision is no longer forced by speed.**

Measured with `scripts/stage0_gpu.py --steps 5` (bfloat16 autocast, fla's fused KDA kernel, equal 65,536
tokens per step unless noted, median of 3 repetitions). DEV and MID come from the first run
(`experiments/stage0_gpu/run1_crashed_on_main.log`), which then ran out of memory on MAIN eager at 32 x 2048;
MAIN comes from the rerun after the script learned to halve the batch
(`experiments/stage0_gpu/20261008T055328Z/stage0_gpu.json`). Dollars use $1.29 per hour, pending the
console price.

| config | params | compiled | batch x seq | tokens/s (median) | min to max | peak GB | hours per 1B | $ per 1B |
|---|---:|---|---|---:|---|---:|---:|---:|
| dev | 24,324,959 | no | 32 x 2048 | 119,489 | 119,480 to 119,508 | 19.25 | 2.32 | 3.00 |
| dev | 24,324,959 | yes | 32 x 2048 | 149,783 | 149,778 to 149,795 | 12.14 | 1.85 | 2.39 |
| dev | 24,324,959 | no | 128 x 512 | 119,484 | 119,383 to 119,488 | 19.25 | 2.32 | 3.00 |
| dev | 24,324,959 | yes | 128 x 512 | 149,198 | 149,007 to 149,224 | 12.14 | 1.86 | 2.40 |
| mid | 44,509,873 | no | 32 x 2048 | 81,504 | 81,478 to 81,508 | 31.39 | 3.41 | 4.40 |
| mid | 44,509,873 | yes | 32 x 2048 | 105,543 | 105,518 to 105,547 | 19.18 | 2.63 | 3.40 |
| mid | 44,509,873 | no | 128 x 512 | 82,064 | 82,052 to 82,064 | 31.39 | 3.38 | 4.37 |
| mid | 44,509,873 | yes | 128 x 512 | 106,178 | 106,158 to 106,203 | 19.18 | 2.62 | 3.37 |
| main | 116,490,541 | no | 16 x 2048 (32 ran out of memory) | 40,493 | 40,493 to 40,493 | 34.33 | 6.86 | 8.85 |
| main | 116,490,541 | yes | 16 x 2048 (32 ran out of memory) | 54,594 | 54,586 to 54,620 | 20.65 | 5.09 | 6.56 |
| main | 116,490,541 | no | 64 x 512 (128 ran out of memory) | 40,806 | 40,801 to 40,811 | 34.34 | 6.81 | 8.78 |
| main | 116,490,541 | yes | 64 x 512 (128 ran out of memory) | 54,819 | 54,808 to 54,860 | 20.67 | 5.07 | 6.54 |

Verdict forward pass, MAIN, 8,192 tokens, batch 1: 0.042 s (the Mac took 2.2 s). DEV and MID verdict times
were not measured: the first run crashed before its verdict rows.

Spreads are under 0.2%, so unlike the Mac runs these numbers are stable. Sequence length (512 or 2,048) makes
no difference to throughput.

## DECISION NEEDED: which config to pretrain

At the compiled rate, for 20 tokens per parameter:

| Option | Parameters | Tokens | Hours | Dollars |
|---|---:|---:|---:|---:|
| A. MAIN | 116.5M | 2.33B | 11.8 | 15.2 |
| B. MID | 44.5M | 0.89B | 2.3 | 3.0 |
| C. DEV | 24.3M | 0.49B | 0.9 | 1.2 |

The cap is $100 and about $1.10 is spent (`docs/cost_ledger.md`). Data generation, fine-tuning, the
adversarial rounds and evaluation also use instance hours; together with MAIN's pretraining the whole
project stays well inside the cap.
