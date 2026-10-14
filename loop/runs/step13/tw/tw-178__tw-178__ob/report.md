# Grader report: tw-178

Rounds: 1 · cost $0.53 · candidates judged correct: A, B · judged wrong: C, D

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
| first line == #!/bin/bash | The script must use #!/bin/bash shebang (not sh) | task-text |
| submitted and regenerated /app/output.txt byte-equal expected_output.txt | The fixed script should produce output to /app/output.txt that matches /app/expected_output.txt | task-text |
| config_summary.txt (submitted and regenerated) contains the 4 config key: value lines | The script should also generate /app/config_summary.txt with parsed configuration values | task-text |
| fresh run exits 0 with empty stderr | ensure it runs without errors | task-text |
| with changed input_a/input_b and expected_output.txt hidden, the comparison section equals diff of new inputs | Fix each bug while maintaining the original intended functionality (compare outputs from multiple data sources) | recomputed |
| non-comment code contains here-doc, <<< and <( | Read configuration from a here-document; here-strings for single-value transformations; process substitution | format |
| code does not reference expected_output, literal 84, banana/blueberry | maintaining the original intended functionality (values computed, not hard-coded) | format |
