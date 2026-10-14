# Grader report: tw-1086

Rounds: 1 · cost $0.64 · candidates judged correct: A, B · judged wrong: C, D, E

| basis | checks |
|---|---:|
| task-text | 8 |
| recomputed | 0 |
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
| .deploy_state has exact lines DEPLOY_VERSION=2.0 and DEPLOY_ENV=production (carried-over, fresh run, after main) | /app/.deploy_state should contain DEPLOY_VERSION=2.0 and DEPLOY_ENV=production | task-text |
| frontend/.built and backend/.built exist | /app/frontend/.built and /app/backend/.built should exist | task-text |
| components.list non-blank lines are exactly frontend, backend, config | /app/config/components.list should contain all three components | task-text |
| fresh bash /app/deploy.sh exits 0 | After fixing the script, run it with /app/deploy.sh | format |
| set_deploy_config sets DEPLOY_VERSION/DEPLOY_ENV in caller | Set and preserve deployment configuration variables (DEPLOY_VERSION and DEPLOY_ENV) | task-text |
| build_component leaves cwd unchanged and creates .built in target | Temporarily change directories for component builds without affecting the main script's working directory | task-text |
| load_components appends parsed component names to COMPONENTS in caller | Preserve array modifications made inside functions | task-text |
| IFS default after parse_deploy_info; parse still yields 3 parts | Isolate IFS changes to only the parsing section that needs them | task-text |
| after main in the current shell: pwd=/app, IFS default, vars and array set | Script should end in /app directory | task-text |
