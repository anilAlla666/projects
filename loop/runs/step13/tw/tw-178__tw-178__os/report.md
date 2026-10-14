# Grader report: tw-178

Rounds: 1 · cost $0.63 · candidates judged correct: A, B · judged wrong: C, D

| basis | checks |
|---|---:|
| task-text | 4 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 2 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| first line is exactly #!/bin/bash | The script must use #!/bin/bash shebang (not sh) | task-text |
| bash /app/process_data.sh exits 0 (twice, with original and changed inputs) | ensure it runs without errors | task-text |
| regenerated output.txt byte-identical to the Dockerfile's expected_output.txt | The fixed script should produce output to /app/output.txt that matches /app/expected_output.txt | task-text |
| regenerated config_summary.txt contains header and APP_NAME/VERSION/MODE/DEBUG lines | The script should also generate /app/config_summary.txt with parsed configuration values | task-text |
| script code contains <<<, a << WORD heredoc, and <( | Read configuration from a here-document / here-strings / process substitution; fix each bug maintaining intended functionality | format |
| with changed inputs and decoy expected file, Data Comparison section equals diff of the new inputs; config/transform sections unchanged | Compare outputs from multiple data sources ...; maintaining the original intended functionality | recomputed |
| bc is invoked with the input value 42 on stdin (PATH shim) or by absolute path | Process input data using here-strings for single-value transformations; maintaining the original intended functionality (original uses bc; Dockerfile installs bc for arithmetic) | format |
