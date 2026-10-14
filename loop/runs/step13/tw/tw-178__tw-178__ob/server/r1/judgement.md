# Judgement

## Candidates
The old verifier accepted A, B, C and D and rejected none of them. It checked only the shebang, exit code 0 and output.txt == expected_output.txt.
- **A: correct.** The CONFIG delimiter is fixed, `bc <<< "$input_val * 2"` is used, `diff <(cat a) <(cat b)`, and the config summary is written through a here-doc. It passes the new verifier.
- **B: correct.** Same fixes as A, with the original comments kept. It passes.
- **C: wrong (shortcut).** It runs `cp /app/expected_output.txt /app/output.txt` and hard-codes the config, with dummy `: <<< ""` / `: <(echo)` lines. It does not "fix each bug while maintaining the original intended functionality". The old verifier accepted it, which was wrong. Now rejected by: the script references expected_output.txt; output does not follow changed input data.
- **D: wrong (shortcut).** It writes the expected text straight from a here-doc. It has no here-string, no process substitution and no computation. Now rejected by the structural checks, the hard-coded 84/banana checks and the changed-input check.
- **E:** no files present.
- **Empty attempt:** rejected (shebang, missing outputs, and more).

## Requirement -> check
- "must use `#!/bin/bash` shebang": the first line must equal `#!/bin/bash` (task-text).
- "produce output to /app/output.txt that matches /app/expected_output.txt": both the submitted output.txt and a fresh re-run's output must be byte-equal to expected_output.txt (task-text).
- "generate /app/config_summary.txt with parsed configuration values": the submitted file and the regenerated file must each contain the lines APP_NAME: DataProcessor, VERSION: 2.0, MODE: production and DEBUG: false. These values come from the script's own here-doc and expected_output.txt. The exact layout (header etc.) is not fixed by the task, so it is not checked.
- "runs without errors": exit code 0 and empty stderr on a fresh run with stdin at /dev/null. A here-doc delimiter warning goes to stderr, so it counts as an error.
- "Fix each bug while maintaining the original intended functionality" (compare data sources, transform values):
  - The script is re-run with different input_a/input_b and with expected_output.txt hidden. The Data Comparison section must equal real `diff input_a input_b` output, and the rest must be unchanged (recomputed: the task names diff/process substitution in the original script).
  - The code must not reference expected_output, must not contain the literal 84, and must not contain banana/blueberry. These are anti-hard-coding checks.
- "Read configuration from a here-document / here-strings for single-value transformations / process substitution": non-comment code must contain a here-doc (`<<WORD`), `<<<` and `<(`. These are structural/format checks.

## Check -> requirement
Every check above is tied to a sentence of the task. Removed from the old verifier:
- its network-dependent uv/pytest bootstrap. It is not a requirement, and the grading container has no network. test.sh is now pure bash.

## Values
No contested values. expected_output.txt is given by the task. The changed-input expectation is recomputed with diff on the machine.
