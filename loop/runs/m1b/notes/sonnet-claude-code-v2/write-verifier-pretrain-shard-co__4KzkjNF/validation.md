# Verifier validation

## Method

`check.py` supports `VERIFIER_APP_DIR` / `VERIFIER_REF_DIR` env vars so it can be run
against arbitrary candidate `/app` directories without rebuilding the Docker image.
Each scenario below was run as:

```
VERIFIER_APP_DIR=<dir> VERIFIER_REF_DIR=/verifier_out/tests/reference python3 /verifier_out/tests/check.py
```

and the resulting `/logs/verifier/reward.txt` was inspected.

## Correct solution used to prove the accept path

I was not able to find a rule-compliant code/data change (within the given `/app`) that
brings the real training run's validation loss from ~6.49 down to <= 6.45 — see
"Limitations" below. To still exercise the verifier's accept path end-to-end, I built a
**synthetic** "good" solution (`/scratch/good_app`), constructed as follows:

- All of `run_pretrain.sh`, `train_pretrain.py`, `data/train/index.json`, and every
  `chunk-0-*.bin` are byte-identical to the originals in `/app` (nothing dropped,
  truncated, duplicated, or replaced).
- Starting from the real (broken) `model_final.pt` checkpoint produced by
  `bash /app/run_pretrain.sh`, I ran 250 additional legitimate training steps
  (`/tmp/make_good.py`) over the *same* real, unmodified training shards (batch size 4,
  AdamW lr=3e-4, single process, seed fixed), then recomputed validation loss on the
  real `val.bin` the same way `train_pretrain.py` does.
- Result: recomputed val_loss = **6.396153345108032**, saved into a
  `checkpoints/model_final.pt` + `train_metrics.json` pair.

This is clearly *not* a claim about the real intended fix to `/app` (it's extra offline
training bolted onto the broken checkpoint, not a diagnosis of the root cause), but it
is useful evidence for two things:
1. It mechanically proves the verifier's PASS path works when every requirement is met.
2. It shows the 6.45 threshold is achievable via legitimate training on the real,
   unmodified data — i.e. the threshold is not impossibly strict, and the ~6.49 result
   is consistent with the model simply being undertrained/stopped early, not with the
   data being unfixably bad.

## Results

| Scenario | Expected | check.py output (failing checks) | reward.txt |
|---|---|---|---|
| `good_app` (synthetic correct solution, see above) | 1 | none — all checks pass | **1** |
| `flaw_faked` — real broken checkpoint, `train_metrics.json` hand-edited to claim `val_loss: 6.40` without retraining | 0 | R7 (reported 6.4 vs recomputed 6.4949), R8 | **0** |
| `flaw_index` — `data/train/index.json` modified (one chunk entry removed) | 0 | R3 | **0** |
| `flaw_maxsteps` — `MAX_STEPS` reduced 500 -> 100 in `train_pretrain.py` | 0 | R4 | **0** |
| `flaw_dup` — `chunk-0-5.bin` overwritten with a byte-copy of `chunk-0-0.bin` | 0 | R5 (duplicate shard) | **0** |
| `flaw_script` — trivial comment appended to `run_pretrain.sh` | 0 | R2 | **0** |

All six scenarios matched their expected reward.

## Verifier fix made during validation

The first run of `flaw_faked` used a checkpoint whose *true* recomputed val_loss
(6.396) happened to be close to the faked reported value (6.40), so the original
R7 tolerance (0.15) didn't catch it. Recomputation from a checkpoint is deterministic
(confirmed bit-identical across repeated real runs and across repeated recomputes of
the same checkpoint), so I tightened `VAL_LOSS_MATCH_TOL` from 0.15 to 0.02 and
re-verified `flaw_faked` (using the real, still-broken checkpoint mislabeled as 6.40)
now correctly fails R7. This is a fix to the verifier itself, not to the variant.

## Limitations (documented transparently)

- **Root cause not found.** Extensive investigation of the real `/app` (chunk content
  analysis, distributed-sharding checks, seed/worker/drop_last variations, removing
  suspect chunks) did not surface a rule-compliant change that legitimately drives the
  real `bash run_pretrain.sh` run below ~6.49. This means R8's threshold (<= 6.45) is
  grounded in the task text ("close to 6.4", contrasted with the stated-broken
  6.49/6.4911 examples) and in the synthetic-continued-training experiment above,
  but was **not** validated against a genuine end-to-end fix of the actual bug. If the
  intended fix produces a value between 6.45 and 6.49, this verifier would
  incorrectly reject it; if the intended fix is a data/pipeline change I didn't
  consider, R5/R6 might not fully characterize "restoring the intended examples" for
  that specific mechanism.
- The `flaw_faked`/`flaw_index`/etc. variants are constructed by hand-editing a copy of
  the *synthetic* good solution (or the real broken checkpoint), not by having an
  independent "solver" attempt the task — they are targeted single-requirement breaks,
  as instructed.
