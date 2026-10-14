# Grader report: tw-44

Rounds: 1 · cost $0.52 · candidates judged correct: A, B, C, E · judged wrong: D

| basis | checks |
|---|---:|
| task-text | 0 |
| recomputed | 2 |
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
| /app/shell_report.txt exists and is non-empty | Your report should be saved as `shell_report.txt` in the /app directory | format |
| a 'current shell'/$0 entry whose value is a shell name (any shell) | The current shell being used (identified by $0 or similar methods) | format |
| all 13 distinct /etc/shells absolute paths of the image appear as whole paths | A list of all installed shells found on the system (from /etc/shells); The absolute path to each shell executable | recomputed |
| report mentions 'exist' and 'executable'; no listed image shell marked missing/not executable | Verification that these shells actually exist as executable files | recomputed |
