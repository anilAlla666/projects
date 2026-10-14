# Requirements extracted from task instructions

- **R1**: "modify `/app` so that `bash /app/run_pretrain.sh` runs to completion, writes a checkpoint, and produces a validation loss close to **6.4**." — a checkpoint file and metrics must exist after the fix, produced by a real run of the launcher.

- **R2**: "Do not modify `/app/run_pretrain.sh`" — the launcher script must be byte-identical to the original.

- **R3**: "Do not modify `/app/data/train/index.json`" — the shard index must be byte-identical to the original.

- **R4**: "Do not reduce `MAX_STEPS` in `/app/train_pretrain.py`; it is an upper bound. Natural dataset exhaustion before 500 steps is expected, and do not change iteration behavior just to force 500 steps." — `MAX_STEPS` must remain `500`, and the loop must still stop naturally on `StopIteration` (not be rewritten to cycle/repeat data to reach step 500).

- **R5**: "Restore the intended training run rather than changing the training recipe, dropping examples, or substituting duplicate examples to hit the target metric." — all originally-named training shard files must still be present (none dropped), and no shard file may be a byte-for-byte duplicate of another shard (guards against "substituting duplicate examples").

- **R6**: "statistically similar, generated, or otherwise approximate replacement inputs are not acceptable even if they produce a validation loss near the target" — shard files must match the sizes declared in `index.json` (`chunk_bytes`), so a shard can't be silently truncated/replaced while keeping its filename.

- **R7**: "Do not fake the results by printing out a fake loss value, a new training run must legitimately produce a checkpoint with the specified loss." — the `val_loss` recorded in `train_metrics.json` must be reproducible: recomputing the validation loss directly from `model_final.pt` on the held-out `val.bin` must match the reported value.

- **R8**: "produces a validation loss close to **6.4**" — the (recomputed, verified-genuine) validation loss must be close to 6.4. The task's own example of the *broken* state reports 6.4911/6.4949 (see below), so "close to 6.4" must be interpreted as meaningfully better than that; the verifier uses a threshold of **≤ 6.45**.

- **R9** (derived from R4): natural exhaustion before 500 steps is expected — `train_metrics.json`'s `steps` field must be `< 500`, confirming the run stopped via natural dataset exhaustion rather than being forced to hit the step cap.

Not turned into checks (not stated as requirements): which specific file(s) must change to fix the bug, the exact numeric value of a "correct" fixed run beyond "close to 6.4", or any particular data-pipeline mechanism — the task deliberately leaves the diagnosis/fix approach open ("Diagnose the root cause and modify `/app`").
