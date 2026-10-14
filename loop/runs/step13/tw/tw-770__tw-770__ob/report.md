# Grader report: tw-770

Rounds: 1 · cost $1.04 · candidates judged correct: A, B, D · judged wrong: C, E

| basis | checks |
|---|---:|
| task-text | 2 |
| recomputed | 6 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 2 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** scope of 'the nginx package' (nginx only vs nginx + nginx-common) for FILES and CONFFILES: the verifier expects original verifier: only loose substring checks; new verifier accepts either scope, the candidates give B,D: nginx only; A,C: nginx + nginx-common; task says 'the nginx package' but its Dockerfile comments attribute nginx-common's configs/services to it

## Every check

| check | requirement | basis |
|---|---|---|
| report exists | The report file must exist at exactly: /home/user/package_audit_report.txt | format |
| header line for each of the 5 sections | well-formatted and clearly organized with section headers | format |
| every file from dpkg -L nginx listed in FILES | A categorized list of all files installed by the nginx package | recomputed |
| /usr/sbin/nginx under Binaries; nginx doc/man files under Documentation | grouped by type: Binaries ... Documentation | task-text |
| no FILES path outside nginx/nginx-common ownership | all files installed by the nginx package | recomputed |
| every Depends package of nginx named | All direct dependencies of the nginx package | recomputed |
| stated version constraints equal dpkg's | Version requirements for each dependency | recomputed |
| nginx.postinst named and summary present | List any postinst scripts ... Provide a brief summary | task-text |
| CONFFILES equals conffiles of nginx (none) or of nginx+nginx-common (17) | A list of configuration files managed by dpkg | recomputed |
| nginx.service named in SERVICES | List any systemd service units associated with nginx | recomputed |
