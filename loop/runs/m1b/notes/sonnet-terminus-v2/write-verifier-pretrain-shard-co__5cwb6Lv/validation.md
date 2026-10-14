# Verifier Validation

All runs were performed by placing files at the exact paths the verifier
sees (`/app/...`), then executing `bash /tests/test.sh` (with `/tests/`
containing a copy of `/verifier_out/tests/`), exactly matching the runtime
layout the grader uses. `/logs/verifier/reward.txt` was inspected after
each run.

## Scenario 1: Good solution (baseline)

Used the environment's own `/app` state (unmodified `run_pretrain.sh`,
unmodified `data/train/index.json`, unmodified `train_pretrain.py` with
`MAX_STEPS = 500`, and the original 20 training shards + val.bin). Verified
by hand that `bash /app/run_pretrain.sh` runs to completion, exhausts the
dataset naturally at step ~404 (< 500), writes
`/app/checkpoints/model_final.pt` and `train_metrics.json`, and reports a
val loss around 6.49 (within the task's "close to 6.4" description; note
the task's own illustrative "broken" example also showed 6.4911, so a
tolerance of 0.3 around 6.4 is used, distinguishing genuine near-target
runs from the reduced-MAX_STEPS variant which produced 7.80).

**Result: `reward.txt` = `1`. All checks passed (R4, R5, R2, R6 static,
R7/R8 proxy, R1/R6 dynamic).**

## Scenario 2: Flawed variant A — MAX_STEPS reduced (violates R2)

Changed `MAX_STEPS = 500` to `MAX_STEPS = 100` in `train_pretrain.py`,
leaving everything else (data, run_pretrain.sh, index.json) untouched.
Training now stops early at step 100 instead of naturally exhausting the
dataset at step ~404, and the resulting val loss (7.80) is far from 6.4.

**Result: `reward.txt` = `0`.** Failed checks:
- `R2 FAIL: MAX_STEPS reduced to 100`
- `R1 FAIL: val_loss not close to 6.4 or inconsistent with printed output`

## Scenario 3: Flawed variant B — run_pretrain.sh modified (violates R4)

Appended a harmless trailing comment line to `/app/run_pretrain.sh`.
Training itself still ran fine and produced a good val loss (~6.49), but
the launcher file itself no longer matches the original byte-for-byte.

**Result: `reward.txt` = `0`.** Failed check:
- `R4 FAIL: run_pretrain.sh was modified`

This confirms the verifier catches a violation of "do not modify
run_pretrain.sh" even when the end-to-end metric looks perfect — exactly
the kind of subtle-but-real violation the task warns about.

## Scenario 4: Flawed variant C — hardcoded/faked val_loss (violates R6)

Inserted `val_loss = 6.42  # FAKED override` immediately after the genuine
`val_loss = sum(val_losses) / len(val_losses)` computation in
`train_pretrain.py`, so both the printed stdout line and the
`train_metrics.json` file agree on the faked constant. A purely dynamic
check (compare printed log to json file) would have been fooled here,
since both were consistently overridden. The verifier's static-code check
catches the hardcoded numeric assignment to `val_loss` regardless.

**Result: `reward.txt` = `0`.** Failed check:
- `R6 FAIL: eval loop looks faked/hardcoded` (detected
  `val_loss = 6.42` as a bare numeric-literal assignment)

## Scenario 5: Flawed variant D — duplicated shard (violates R7/R8)

Overwrote `/app/data/train/chunk-0-1.bin` with an exact byte copy of
`/app/data/train/chunk-0-0.bin` (simulating the common shortcut of
"duplicating a good shard to paper over a bad one"). `index.json` and
`run_pretrain.sh` were left untouched, and training still completed with a
val loss close to 6.4 (since duplicating 1/20 shards barely moves the
average in this toy setup) — showing that the metric alone is NOT a
reliable signal for this requirement, which is exactly why a dedicated
integrity check was needed.

**Result: `reward.txt` = `0`.** Failed check:
- `R7/R8 FAIL: shard integrity check failed (duplicate or size-mismatched
  shard)` — detected `chunk-0-1.bin` byte-identical to `chunk-0-0.bin`,
  and 100% positional token overlap between the two.

## Summary table

| Scenario                          | Violates | reward.txt |
|-----------------------------------|----------|------------|
| Good (unmodified /app)            | none     | 1          |
| MAX_STEPS reduced to 100          | R2       | 0          |
| run_pretrain.sh modified          | R4       | 0          |
| hardcoded val_loss = 6.42         | R6       | 0          |
| duplicated shard file             | R7/R8    | 0          |

All flawed variants were correctly rejected and the correct/good solution
was correctly accepted, with each failure traceable to the exact
requirement it violates.
