# Grader report: tw-44

Rounds: 1 · cost $0.53 · candidates judged correct: A, B, C, E · judged wrong: D

| basis | checks |
|---|---:|
| task-text | 3 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 1 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** which shell is the current shell: the verifier expects any real shell name (old: any 'sh' substring), the candidates give bash (A,B,C,E), /replay.sh (D); depends on how the solver's harness runs commands; the task fixes no value

## Every check

| check | requirement | basis |
|---|---|---|
| test.sh runs offline (no apt/curl/uv) and writes 1/0 to /logs/verifier/reward.txt | grader runs bash /tests/test.sh; no network | format |
| /app/shell_report.txt exists and is non-empty | Your report should be saved as shell_report.txt in the /app directory | task-text |
| a current-shell/$0 line names a real shell (token basename is a shell name) | The current shell being used (identified by $0 or similar methods) | task-text |
| every unique /etc/shells entry (excluding harness-added tmux/screen) appears as a standalone absolute path | A list of all installed shells found on the system (from /etc/shells); The absolute path to each shell executable | recomputed |
| the report states existence/executability and does not call an existing executable shell missing | Verification that these shells actually exist as executable files | task-text |
