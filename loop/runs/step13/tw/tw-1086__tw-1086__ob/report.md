# Grader report: tw-1086

Rounds: 1 · cost $0.82 · candidates judged correct: A, B · judged wrong: C, D, E

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
| .deploy_state has lines DEPLOY_VERSION=2.0 and DEPLOY_ENV=production (solver output and fresh run) | /app/.deploy_state should contain DEPLOY_VERSION=2.0 and DEPLOY_ENV=production | task-text |
| frontend/.built and backend/.built exist (solver output and fresh run) | /app/frontend/.built and /app/backend/.built should exist | task-text |
| components.list has lines frontend, backend, config | /app/config/components.list should contain all three components | task-text |
| after sourcing deploy.sh, PWD is /app at exit | Script should end in /app directory | task-text |
| unit: set_deploy_config sets DEPLOY_VERSION/DEPLOY_ENV in caller; full run DEPLOY_VERSION=2.0 at exit | Set and preserve deployment configuration variables (DEPLOY_VERSION and DEPLOY_ENV) | task-text |
| unit: build_component does not change caller PWD and creates .built in target dir | Temporarily change directories for component builds without affecting the main script's working directory | task-text |
| unit: load_components populates COMPONENTS in caller (test conf and real conf) | Preserve array modifications made inside functions | task-text |
| unit: IFS default after parse_deploy_info and it still splits on ','; full run IFS default at exit, no 'IFS is corrupted' | Isolate IFS changes to only the parsing section that needs them | task-text |
| the four functions remain defined; bash -n ok; script exits 0; fallback list not used | Fix each bug by correctly choosing between () and {} | format |
