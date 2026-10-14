# Grader report: tw-770

Rounds: 1 · cost $1.26 · candidates judged correct: B, D · judged wrong: A, C, E

| basis | checks |
|---|---:|
| task-text | 3 |
| recomputed | 6 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 3 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** scope of 'the nginx package' (files, conffiles, scripts): the verifier expects original: any mention of /etc/nginx or 'conffile'; now: nginx only, or nginx+nginx-common complete, the candidates give B, D: nginx only (no conffiles); C: nginx+nginx-common; A: nginx + truncated nginx-common; the task does not say whether the dependency nginx-common (which holds config/unit/conffiles) counts as part of the nginx package

## Every check

| check | requirement | basis |
|---|---|---|
| report exists and non-empty | The report file must exist at exactly: /home/user/package_audit_report.txt | format |
| header line for each of the 5 sections | The report should be well-formatted and clearly organized with section headers | format |
| 5 file categories present | grouped by type: Binaries, Configuration files, Libraries, Documentation, Other files | format |
| every non-directory leaf of dpkg -L nginx listed in FILES | A categorized list of all files installed by the nginx package | recomputed |
| /usr/sbin/nginx under Binaries, nginx doc/man under Documentation, category location rules | Binaries (executables in /usr/sbin, /usr/bin, etc.) ... Documentation (in /usr/share/doc/, /usr/share/man/) | task-text |
| listed file paths are dpkg-owned (nginx/nginx-common) or exist | all files installed by the nginx package | recomputed |
| nginx-common files listed either none or all | all files installed by the nginx package | recomputed |
| every dpkg Depends name present | All direct dependencies of the nginx package | recomputed |
| each versioned dep has a constraint equal to dpkg's (nginx-common: = and same upstream) | Version requirements for each dependency | recomputed |
| nginx postinst named and summarised | List any postinst scripts associated with nginx; Provide a brief summary | task-text |
| conffiles listed are real; set is none (stated) / nginx's / nginx+nginx-common's | A list of configuration files managed by dpkg | recomputed |
| nginx.service mentioned | List any systemd service units associated with nginx | task-text |
