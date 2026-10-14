# Judgement

## Evidence run
evidence.md has no per-candidate results: the probe crashed (missing grader config.json). The original test.sh
also could not have worked in the grader: it runs `apt-get install curl` and downloads uv over the network, and the
grading container has no network. It was replaced by a self-contained `perl` checker (perl-base is essential in
ubuntu:24.04; this machine has only perl-base, so the checker was tested here against the same modules). There is no python3 on this machine.

## Ground truth (recomputed with dpkg on this machine; the checker recomputes it in the grading container)
- `dpkg -L nginx`: /usr/sbin/nginx, /usr/share/doc/nginx/{changelog.Debian.gz,copyright}, /usr/share/man/man8/nginx.8.gz (+dirs).
  The man page does not exist on disk (minimised image path-exclude) but dpkg lists it as installed.
- `dpkg-query -W -f='${Depends}' nginx`: libc6 (>= 2.34), libcrypt1 (>= 1:4.1.0), libpcre2-8-0 (>= 10.22),
  libssl3t64 (>= 3.0.0), zlib1g (>= 1:1.1.4), iproute2, nginx-common (= <exact build version>).
  The build version moves (candidates saw 7.6 and 7.18; 1.24.0-2ubuntu7 also has libcrypt1 >= 1:4.1.0), so for
  nginx-common only "=" plus the same upstream version "1.24.0-" is required.
- Conffiles: nginx has none; nginx-common has 17.
- postinst: /var/lib/dpkg/info/nginx.postinst (and nginx-common.postinst).

## Contested: scope of "the nginx package"
Strict reading: only the `nginx` binary package (no /etc files, no conffiles, one postinst). Broad reading: nginx
plus nginx-common, which holds its config, unit and conffiles (the Dockerfile comments suggest the author thought this way).
Data cannot settle this; it is a choice of meaning. The checker accepts both. It rejects only what is wrong under
**both** readings: a *partial* set of nginx-common files or conffiles. Under the strict reading that wrongly says they belong to nginx,
and under the broad reading it leaves files out. The original conffiles check (mention of /etc/nginx, nginx.conf, mime.types or
"conffile") is still covered, since every passing report has a CONFFILES header.

## Candidates
- A: WRONG. FILES lists nginx-common files cut off at 10 per category (config stops at fastcgi_params; nginx.conf,
  mime.types, index.html, vim files missing). Rejected by the "nginx-common files none-or-all" check. Also stderr
  `/replay.sh: line 73: systemctl: command not found` was pasted into the report (not penalised by itself).
- B: CORRECT (strict reading, consistent; all files, deps with versions, postinst + summary, none conffiles, nginx.service).
- C: WRONG. "libcrypt1: >= 1:4.4.0". dpkg's Depends says >= 1:4.1.0 (the same in 7, 7.18, and per D, 7.6). The task
  requires "Version requirements for each dependency", and the named tool fixes them. A copy of C with only this value fixed
  passes (the broad-reading control).
- D: CORRECT (strict reading).
- E: WRONG. Fabricated: libc6 (>= 2.14), nginx-common (= 1.18.0), /usr/lib/nginx (does not exist), missing deps,
  missing doc/man files, no postinst summary, partial conffiles.

## Trace: requirement -> check
| Requirement (task) | Check |
|---|---|
| "report file must exist at exactly /home/user/package_audit_report.txt" | file exists, non-empty |
| "clearly organized with section headers" + 5 named sections | a header line for each of FILES/DEPENDENCIES/SCRIPTS/CONFFILES/SERVICES (numbering, markdown and "SECTION" suffix allowed) |
| "categorized list of all files installed by the nginx package, grouped by type" (5 types) | all 5 category sub-headers present; every non-directory leaf of `dpkg -L nginx` listed; no listed path that is neither dpkg-owned nor existing; nginx-common files none or all |
| "Binaries (executables in /usr/sbin, /usr/bin, etc.)" etc. | /usr/sbin/nginx under Binaries; nginx doc/man under Documentation; config entries in /etc, doc entries in /usr/share/doc|man|info, lib entries in lib dirs/modules, bin entries in bin dirs or executable |
| "All direct dependencies" | every Depends name from dpkg present |
| "Version requirements for each dependency" | for each versioned dep, at least one stated constraint equals dpkg's (nginx-common: "=" + same upstream) |
| "List any postinst scripts" | nginx(.)postinst named |
| "brief summary of what these scripts do" | summary keyword (start/reload/upgrade/invoke-rc.d/port 80/trigger) |
| "list of configuration files managed by dpkg ... conffiles" | listed /etc paths are real conffiles; set is empty (and says none) or equals nginx's or nginx+nginx-common's |
| "List any systemd service units" | nginx.service mentioned |
| "Include the service status if possible" | not checked: systemctl is absent and the solver could have started nginx by hand, so a stated status cannot be decided |

Original checks: test_report_exists_and_has_sections (words anywhere) → strengthened into header detection;
test_installed_files_accuracy, test_dependencies_completeness and test_conffiles_identification were trivially weak (e.g. "depends" anywhere)
and are replaced by the stronger checks above, which still imply them. No original value check rejected a correct candidate.

## Near misses (all rejected; B and D accepted; empty rejected)
nm1 libssl3t64 wrong version (B); nm2 man page omitted (D); nm3 partial conffiles (D); nm4 postinst summary removed (D);
nm5 /usr/sbin/nginx moved to Other (B); nm6 iproute2 dropped (D); nm7 no service unit (D); nm8 shortcut template report.
