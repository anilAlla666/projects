# Verifier validation

`bash /verifier_out/tests/test.sh` was run against scratch copies of the carried paths. The positive fixture used the task's supplied shard/index/validation files and a checkpoint produced by running the provided training script in `/scratch/good` to natural exhaustion at step 404. The verifier independently recomputed its held-out loss as 6.49556; the saved metric was 6.49486. Both are within the verifier's 6.5 ceiling, which is based on the measured scratch reference run and the task's approximate 6.4 target.

| Fixture | Single requirement broken | Reward | Failed check |
|---|---|---:|---|
| `/scratch/good` | none | `1` | none |
| `/scratch/flawed_r5` | R5: lowered `MAX_STEPS` to 499 | `0` | `MAX_STEPS has not been reduced below 500` |
| `/scratch/flawed_r6` | R6: changed natural exhaustion `break` to `continue` | `0` | `iteration stops naturally when StopIteration is raised` |
| `/scratch/flawed_r9` | R9: hard-coded `6.4` in the printed validation result while retaining the genuine checkpoint and metric | `0` | `printed validation loss is not a hard-coded value` |

The shard checks verify the complete indexed inventory, LitData block structure, token range, all 3,240 examples, and absence of duplicate 129-token examples. The provided environment contains no independent clean copy or checksum manifest for the intended original token contents, so exact provenance of a unique, valid replacement sequence cannot be established from the carried paths alone. R4's exact-original-content clause is therefore not fully enforceable by these checks; the verifier does not claim that statistical or structural checks prove provenance.
