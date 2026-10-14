# Grader report: tw-44

Rounds: 1 · cost $0.59 · candidates judged correct: A, B, C, E · judged wrong: D

| basis | checks |
|---|---:|
| task-text | 2 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 1 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| /app/shell_report.txt exists and is non-empty | Your report should be saved as `shell_report.txt` in the /app directory | format |
| current-shell entry names bash (first shell token near 'current shell'/'$0') | The current shell being used (identified by $0 or similar methods) | recomputed |
| every executable non-comment /etc/shells entry appears as exact absolute-path token | A list of all installed shells found on the system (from /etc/shells); The absolute path to each shell executable | task-text |
| report states shells exist and are executable; no real shell marked missing/not executable | Verification that these shells actually exist as executable files | task-text |
