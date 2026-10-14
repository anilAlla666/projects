# Grader report: tw-44

Rounds: 1 · cost $0.00 · candidates judged correct: none · judged wrong: A, B, C, D, E

| basis | checks |
|---|---:|
| task-text | 5 |
| recomputed | 1 |
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
| Report file exists at /app/shell_report.txt | report should be saved as shell_report.txt in the /app directory | task-text |
| Current shell identified as bash ("bash" or "/bin/bash") not a script path | The current shell being used (identified by $0 or similar methods) | task-text |
| All shells from /etc/shells listed in report (including duplicate /usr/bin/zsh) | A list of all installed shells found on the system (from /etc/shells) | recomputed |
| No fabricated shells (e.g., /usr/bin/tmux) in report | A list of all installed shells found on the system (from /etc/shells) | task-text |
| Each shell verified as existing and executable | Verification that these shells actually exist as executable files | task-text |
| Absolute path to each shell executable present | The absolute path to each shell executable | task-text |
