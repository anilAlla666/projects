# Grader report: tw-178

Rounds: 1 · cost $0.54 · candidates judged correct: A, B · judged wrong: C, D

| basis | checks |
|---|---:|
| task-text | 4 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 4 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| first line is #!/bin/bash | The script must use #!/bin/bash shebang (not sh) | task-text |
| bash -n clean, exit 0, no error/warning on stderr | Fix the script in place ... and ensure it runs without errors | task-text |
| output.txt (regenerated) byte-equals expected_output.txt | The fixed script should produce output to /app/output.txt that matches /app/expected_output.txt | task-text |
| config_summary.txt regenerated and contains the 4 config key: value lines | The script should also generate /app/config_summary.txt with parsed configuration values | task-text |
| non-comment here-string <<< present | Process input data using here-strings for single-value transformations; maintaining the original intended functionality | format |
| diff fed via process substitution <(...) | Compare outputs from multiple data sources using process substitution | format |
| comparison section equals real diff after input files are changed | maintaining the original intended functionality (compare input_a/input_b) | recomputed |
| no literal 84 / banana / blueberry / expected_output in code | Fix each bug while maintaining the original intended functionality | format |
| offline test.sh (no curl/uv/pytest) | grader has no network | format |
