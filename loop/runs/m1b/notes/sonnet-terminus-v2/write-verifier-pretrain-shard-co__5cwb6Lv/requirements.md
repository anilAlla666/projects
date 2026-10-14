# Requirements extracted from task instructions

- R1: "modify `/app` so that `bash /app/run_pretrain.sh` runs to completion, writes a checkpoint, and produces a validation loss close to **6.4**." (three sub-parts: completes successfully, writes a checkpoint file, val loss close to 6.4)

- R2: "Do not reduce `MAX_STEPS` in `/app/train_pretrain.py`; it is an upper bound."

- R3: "Natural dataset exhaustion before 500 steps is expected, and do not change iteration behavior just to force 500 steps."

- R4: "Do not modify `/app/run_pretrain.sh`."

- R5: "Do not modify `/app/data/train/index.json`."

- R6: "Do not fake the results by printing out a fake loss value, a new training run must legitimately produce a checkpoint with the specified loss."

- R7: "Restore the intended training run rather than changing the training recipe, dropping examples, or substituting duplicate examples to hit the target metric."

- R8: "statistically similar, generated, or otherwise approximate replacement inputs are not acceptable even if they produce a validation loss near the target."

Notes: R7 and R8 describe requirements about provenance of the training data that are inherently hard for a black-box verifier to fully confirm without ground-truth reference data. The verifier approximates enforcement of R7/R8 via: (a) requiring no two shard files be byte-identical (rules out the common 'duplicate a good shard to replace a bad one' shortcut), and (b) requiring the shard count/index to be unchanged (rules out dropping examples).
