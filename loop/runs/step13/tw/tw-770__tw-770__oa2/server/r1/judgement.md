# Judgement

## Machine facts (same apt snapshot as the grader: curl candidate 8.5.0-2ubuntu10.15 here and in the evidence logs)
- `dpkg -s nginx`: Version 1.24.0-2ubuntu7.18. Depends: libc6 (>= 2.34), libcrypt1 (>= 1:4.1.0), libpcre2-8-0 (>= 10.22),
  libssl3t64 (>= 3.0.0), zlib1g (>= 1:1.1.4), iproute2, nginx-common (= 1.24.0-2ubuntu7.18).
  The apt lists only have 1.24.0-2ubuntu7 and 7.18. **1.24.0-2ubuntu7.6 does not exist on this machine/image.**
- `dpkg -L nginx`: /usr/sbin/nginx, /usr/share/doc/nginx/{changelog.Debian.gz,copyright}, /usr/share/man/man8/nginx.8.gz (+dirs).
- nginx has no conffiles; nginx-common has 17. nginx.postinst exists (nginx-common.postinst too). nginx.service is shipped
  by nginx-common. No systemctl, no python3 on this machine.

## Original verifier
Only substring checks ("files", "libc", "/etc/nginx"...). It accepted all five candidates, including E, which is
fabricated (libc6 >= 2.14, nginx-common = 1.18.0, /usr/lib/nginx which does not exist). It also depended on
downloading uv from the network (failed in evidence). It was rewritten as pure bash/awk/dpkg (test.sh only).

## Interpretation (both readings accepted)
"files installed by the nginx package" / conffiles: the strict reading is the binary package `nginx` (dpkg -L nginx);
the broad reading includes nginx-common (the Dockerfile comments say "Configuration in /etc/nginx/", "Systemd service
units", which live in nginx-common). Both are accepted; what is rejected is a list that matches neither complete set
(partial / truncated / fabricated).

## Candidates
- A: WRONG. Rejected by "FILES section file set is neither all nginx files nor all nginx+nginx-common files". A lists
  nginx-common files but exactly the first 10 of each category (head-truncated): Configuration lists fastcgi.conf but
  not nginx.conf, mime.types, sites-available/default ...; Other omits html/index.html and the vim files. Neither the
  strict reading (contains foreign files) nor the broad one (incomplete) is satisfied. "A categorized list of all files".
- B: CORRECT (strict reading throughout; exact Depends; postinst listed and summarised; conffiles none; nginx.service).
- C: WRONG. DEPENDENCIES states nginx-common (= 1.24.0-2ubuntu7.6) and libcrypt1 >= 1:4.4.0; the installed package
  requires (= 1.24.0-2ubuntu7.18) and (>= 1:4.1.0). Also its quoted postinst is not the real file (missing `*)` branch);
  values were not read from the machine. Check: recomputed from `dpkg-query -W -f '${Depends}' nginx`.
- D: WRONG. Everything else is fine but nginx-common (= 1.24.0-2ubuntu7.6) is a version that is not installed and not
  even available in this image's apt lists. Task: "Version requirements for each dependency" -> recomputed.
  (With 7.6 replaced by 7.18 the report passes — tested.)
- E: WRONG. Fabricated deps/versions, missing doc/man files, /usr/lib/nginx not installed, partial conffiles, no
  script summary.
- empty attempt: rejected (no report).

## Trace: requirement -> check
| Task requirement | Check |
|---|---|
| "The report file must exist at exactly: /home/user/package_audit_report.txt" | file exists, non-empty |
| "well-formatted ... with section headers" / sections 1-5 | header line for each of files/dependencies/scripts/conffiles/services (keyword at line start, decorations/numbering allowed) |
| FILES: "list of all files installed by the nginx package, grouped by type" | every non-directory file from `dpkg -L nginx` listed; non-dir file set equals nginx's or nginx+nginx-common's (symlinks optional); no path not owned by nginx/nginx-common |
| "Binaries (executables in /usr/sbin...)", "Documentation (in /usr/share/doc/, /usr/share/man/)" | /usr/sbin/nginx under a Binaries sub-heading, doc and man files under Documentation sub-heading (only nginx's own files; nginx-common placement e.g. nginx.service is not graded) |
| DEPENDENCIES: "All direct dependencies" + "Version requirements for each" | each item in live Pre-Depends/Depends of nginx named in section; versioned ones with op+version on same line |
| SCRIPTS: "List any postinst scripts" | section mentions postinst and nginx |
| SCRIPTS: "brief summary of what these scripts do" | section has a description keyword (start/reload/upgrade/invoke-rc/port 80/systemd/service/daemon/enable) |
| CONFFILES | set of non-directory paths in section == conffiles(nginx) [empty] or conffiles(nginx)+conffiles(nginx-common) |
| SERVICES: "List any systemd service units" | nginx.service mentioned |
| "Include the service status if possible" | section addresses status (status/active/running/enabled/disabled/dead) — lenient |
| "Use standard Linux package management tools" | not checkable from the output; not checked |

Every check above follows from the cited sentence; checks of the old verifier ("depends" word, "libc" substring) were
replaced by the precise versions. No contested values: all expected values are read live from dpkg on the grader.

## Near misses (all rejected; B and fixed variants of C/D/A pass)
nm1 wrong nginx-common version; nm2 man page under "Other files"; nm3 one dependency dropped; nm4 broad conffiles
minus mime.types; nm5 postinst listed without summary; nm6 broad file list missing one vim file; nm7 no nginx.service.
