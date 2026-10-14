# Judgement

## State of the original verifier
The evidence shows the harness failed before running anything (missing grader config), so no candidate results were recorded.
Reading the original `test.sh` shows it would also fail in the grader: it runs `apt-get install curl` and downloads uv/pytest,
but the grading container has no network. So every attempt would score 0. That is a format/infrastructure defect.
I replaced it with an offline bash verifier (`test.sh` -> `check.sh`, plus a copy of `expected_output.txt`). It doesn't use pytest
or python (python3 isn't on this machine either, so a bash verifier is the only kind I could test here).

## Candidates (A–D; there is no /candidates/E)
- **A: correct.** Fixed heredoc delimiter, `bc <<<`, `diff <(..) <(..)`, `#!/bin/bash`, writes config_summary through a heredoc. Output matches exactly and follows changed inputs.
- **B: correct.** Same fixes as A, with the original comments kept.
- **C: wrong (shortcut).** `cp /app/expected_output.txt /app/output.txt` plus decoy `: <<< ""` and `: <(echo)` lines. It doesn't fix or keep the
  original functionality. Rejected by: "diff via process substitution", "does not use expected_output.txt" and the behavioural check (the output does not follow changed inputs).
- **D: wrong (shortcut).** Hard-codes the whole output in a heredoc. Rejected by: the here-string check, process substitution, hard-coded 84, hard-coded banana/blueberry, and the behavioural check.
- The empty attempt (the untouched original script is carried over, or no script at all) is rejected.

## Trace: requirement -> check
| Requirement (task text) | Check |
|---|---|
| "The script must use `#!/bin/bash` shebang (not sh)" | first line == `#!/bin/bash` |
| "Fix the script ... ensure it runs without errors" | `bash -n` clean (this catches a mismatched-heredoc warning), exit code 0, no error/warning text on stderr |
| "produce output to /app/output.txt that matches /app/expected_output.txt" | outputs are deleted, the script is re-run, and output.txt is byte-compared with expected (the expected text is copied from the Dockerfile, so it is task-text) |
| "should also generate /app/config_summary.txt with parsed configuration values" | config_summary.txt is deleted, the script is re-run, and the file must contain the 4 config lines (the exact format is not specified, so only the values are checked) |
| "Process input data using here-strings for single-value transformations"; "Fix each bug while maintaining the original intended functionality" | a non-comment `<<<` is present |
| "Compare outputs from multiple data sources using process substitution" | a non-comment `diff ... <(` is present |
| "maintaining the original intended functionality" (the comparison reads input_a/input_b) | behavioural: change the input files, re-run, and the comparison section must equal real `diff` of the new inputs |
| Anti-shortcut (fixing the script, not faking output) | no literal 84, banana/blueberry or `expected_output` in code lines |

Every check follows from one of the sentences above. The original check "output == expected" is kept (task-text).
The original check "exit code 0" is kept. The original check "shebang" is kept.

## Contested values
None. The expected output is given verbatim by the task (expected_output.txt). For config_summary, only the values are checked,
because the exact layout is not stated.

## Near misses (all rejected; A and B accepted)
nm1 shebang `#!/bin/sh`; nm2 config_summary not generated; nm3 pipe instead of process substitution (output still identical);
nm4 `echo | bc` instead of a here-string; nm5 hard-coded diff text (obfuscated) with a decoy `<(true)`; nm6 blank line before
"Processing Complete" missing; nm7 the untouched original script.
