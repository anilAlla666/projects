# Grader report: tw-770

Rounds: 1 · cost $1.41 · candidates judged correct: B · judged wrong: A, C, D, E

| basis | checks |
|---|---:|
| task-text | 1 |
| recomputed | 4 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 5 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| report file exists and is non-empty | The report file must exist at exactly: /home/user/package_audit_report.txt | format |
| five section headers present | clearly organized with section headers (sections 1-5) | format |
| all non-directory files of dpkg -L nginx listed; file set equals nginx or nginx+nginx-common; no foreign paths | A categorized list of all files installed by the nginx package | recomputed |
| nginx binary under Binaries, doc/man under Documentation | grouped by type: Binaries (/usr/sbin...), Documentation (/usr/share/doc/, /usr/share/man/) | task-text |
| every direct dependency with its exact version requirement | All direct dependencies of the nginx package; Version requirements for each dependency | recomputed |
| postinst listed | List any postinst (post-installation) scripts associated with nginx | format |
| summary keyword in scripts section | Provide a brief summary of what these scripts do | format |
| conffiles set equals dpkg conffiles of nginx (empty) or of nginx+nginx-common | A list of configuration files managed by dpkg ... marked as conffiles | recomputed |
| nginx.service mentioned in services section | List any systemd service units associated with nginx | recomputed |
| services section addresses status | Include the service status if possible | format |
