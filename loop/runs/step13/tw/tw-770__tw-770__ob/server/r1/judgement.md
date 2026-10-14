# Judgement

## Machine facts (recomputed with the tools the task names: dpkg, dpkg-query, apt-cache)
- `dpkg -L nginx` (1.24.0-2ubuntu7.18): /usr/sbin/nginx, /usr/share/doc/nginx/{changelog.Debian.gz,copyright}, /usr/share/man/man8/nginx.8.gz (+ dirs).
- Config, conffiles (17), nginx.service, apport hook, vim files belong to `nginx-common`.
- `dpkg-query -W -f='${Depends}' nginx`: libc6 (>= 2.34), libcrypt1 (>= 1:4.1.0), libpcre2-8-0 (>= 10.22), libssl3t64 (>= 3.0.0), zlib1g (>= 1:1.1.4), iproute2, nginx-common (= 1.24.0-2ubuntu7.18).
- Postinst scripts: /var/lib/dpkg/info/nginx.postinst (and nginx-common.postinst). systemctl is not installed.

## Original verifier
Accepted all five candidates including E, which is fabricated (libc6 >= 2.14, nginx-common = 1.18.0, /usr/lib/nginx, missing 5 deps,
no postinst path). Its checks were bare substring tests ("files" anywhere in text, "depends" anywhere, "/etc/nginx" anywhere).
It also depended on network (apt-get, curl astral.sh, uv) which the grader does not have. Rewritten as a perl-base script
(perl-base is Essential in ubuntu:24.04; python3 is not on this machine so a perl checker could actually be tested here).

## Interpretation: what "the nginx package" covers (contested -> accept both)
Reading 1: strictly binary package `nginx` (B, D): no config/conffiles. Reading 2: nginx + its own nginx-common split (A, C):
the Dockerfile comment ("Configuration in /etc/nginx/", "Systemd service units") suggests the author thought of it this way.
The task does not choose; the verifier accepts either: FILES must contain every nginx file and may contain only paths owned by
nginx/nginx-common; CONFFILES must be exactly the conffile set of nginx (empty) or of nginx+nginx-common (17).

## Candidate decisions
- A: correct (reading 2). Accepted.
- B: correct (reading 1). Accepted.
- C: WRONG. Rejected by DEPENDENCIES version check: reports `libcrypt1: >= 1:4.4.0`; dpkg says `libcrypt1 (>= 1:4.1.0)`
  (D, built on the same 7.6 version C claims, also shows 1:4.1.0, so not a version-drift artefact). The task requires
  "Version requirements for each dependency"; a wrong one is a wrong report. Its quoted postinst is also edited (missing the `*)` case).
  Value basis: recomputed with dpkg-query (task-named tool).
- D: correct (reading 1). Its nginx-common (= 1.24.0-2ubuntu7.6) differs from this machine's 7.18 because its build was older;
  the check therefore accepts any `= <upstream>-*` for nginx-common (build-time drift is not the solver's error).
- E: WRONG. Fabricated: missing docs/man page in FILES, non-existent /usr/lib/nginx, 5 deps missing, wrong versions, no postinst path,
  partial conffiles.
- empty attempt: rejected (report missing).

## Trace: requirement -> check
- "report file must exist at exactly /home/user/package_audit_report.txt" -> existence check (format).
- "well-formatted ... with section headers"; sections 1-5 -> a header line for each of FILES/DEPENDENCIES/SCRIPTS/CONFFILES/SERVICES (format).
- FILES "all files installed by the nginx package" -> every leaf path of `dpkg -L nginx` listed (recomputed); "grouped by type: Binaries"
  -> /usr/sbin/nginx under a Binaries heading; Documentation -> nginx doc/man files not under another heading; no paths not owned
  by nginx/nginx-common (anti-fabrication; usrmerge /lib aliases allowed).
- DEPENDENCIES "All direct dependencies" -> each Depends name present; "Version requirements for each dependency" -> every
  operator/version written next to a versioned dep equals dpkg's (nginx-common: operator `=` and upstream prefix) (recomputed).
- SCRIPTS "List any postinst scripts" -> nginx.postinst named; "brief summary" -> mentions start/reload/upgrade/invoke-rc.d/systemd.
- CONFFILES -> exact set equality under either reading (recomputed, contested scope).
- SERVICES "List any systemd service units" -> nginx.service named. "status if possible": systemctl absent; no check (unverifiable).
- Removed: old checks (substring "files", "depends", "/etc/nginx") - subsumed / too weak; old network-dependent test.sh.

## Near misses (all rejected)
nm1 iproute2 dropped; nm2 one conffile dropped; nm3 man page missing; nm4 libssl3t64 version altered; nm5 SERVICES header renamed;
nm6 /usr/sbin/nginx moved to Other files; nm7 postinst replaced with "none found"; nm8 header-only shortcut; nm9 fabricated .so path.

No data-fit readings were needed: all values are read directly from dpkg on the grading image.
