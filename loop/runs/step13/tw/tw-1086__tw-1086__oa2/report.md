# Grader report: tw-1086

Rounds: 1 · cost $0.66 · candidates judged correct: A, B · judged wrong: C, D, E

| basis | checks |
|---|---:|
| task-text | 8 |
| recomputed | 0 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 3 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| .deploy_state contains lines DEPLOY_VERSION=2.0 and DEPLOY_ENV=production (solver output, fresh run, extra-component run) | /app/.deploy_state should contain DEPLOY_VERSION=2.0 and DEPLOY_ENV=production | task-text |
| frontend/.built and backend/.built exist | /app/frontend/.built and /app/backend/.built should exist | task-text |
| components.list has lines frontend, backend, config; no commas | /app/config/components.list should contain all three components | task-text |
| script sourced from /app ends with pwd /app and default IFS | Script should end in /app directory; Isolate IFS changes | task-text |
| set_deploy_config sets DEPLOY_VERSION/DEPLOY_ENV in caller | Variable Persistence Bug / Set and preserve deployment configuration variables | task-text |
| build_component leaves caller cwd unchanged and creates .built in component dir | Directory Isolation Bug | task-text |
| load_components leaves COMPONENTS=(frontend backend config) in caller | Array Scope Bug / Preserve array modifications made inside functions | task-text |
| parse_deploy_info leaves IFS unchanged and still prints 'Parsed 3 parts' | IFS Isolation Bug | task-text |
| the four functions exist in deploy.sh | Fix each bug by correctly choosing between () and {} | format |
| deploy.sh parses, exits 0, regenerates outputs, no 'IFS is corrupted'/fallback warning | After fixing the script, run it with /app/deploy.sh | format |
| with an extra component line in components.conf, components.list includes it | components.conf - Configuration file listing components to deploy; Preserve array modifications | format |
