# Grader report: tw-178

Rounds: 1 · cost $0.00 · candidates judged correct: A, B · judged wrong: C, D

| basis | checks |
|---|---:|
| task-text | 7 |
| recomputed | 0 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 0 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| Script uses #!/bin/bash shebang | The script must use #!/bin/bash shebang (not sh) for process substitution support | task-text |
| Script executes without error (exit code 0) | The fixed script should run without errors | task-text |
| Output file /app/output.txt matches /app/expected_output.txt exactly | The fixed script should produce output to /app/output.txt that matches /app/expected_output.txt | task-text |
| Config summary file /app/config_summary.txt exists with correct parsed configuration values | The script should also generate /app/config_summary.txt with parsed configuration values | task-text |
| Script uses here-document (<< DELIMITER) with matching delimiter for configuration | Read configuration from a here-document | task-text |
| Script uses here-string (<<<) with bc for single-value transformation | Process input data using here-strings for single-value transformations | task-text |
| Script uses process substitution (<()) with diff for data comparison | Compare outputs from multiple data sources using process substitution | task-text |
