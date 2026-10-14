# Judgement

## Environment facts (this machine = solver's starting image)
- `/etc/shells` (non-comment, distinct): /bin/sh /usr/bin/sh /bin/bash /usr/bin/bash /bin/rbash /usr/bin/rbash
  /usr/bin/dash /bin/ksh93 /usr/bin/ksh93 /bin/rksh93 /usr/bin/rksh93 /bin/zsh /usr/bin/zsh (zsh appears twice).
  All exist and are executable (`/bin` -> `usr/bin`; rbash/rksh93/sh are symlinks to executables).
- Candidates A,B,C,E also list `/usr/bin/tmux`, i.e. their harness installed tmux which registers itself in
  /etc/shells. That is extra, not wrong; the verifier therefore uses the fixed image list above (basis: recomputed
  from this machine's /etc/shells, which comes from the Dockerfile packages) and allows extra entries.
- `$0` in this machine's command shell is `bash`.

## Original verifier problems
- test.sh needed network (apt/curl/uv/pytest); the grading container has no network. Replaced by a pure bash script.
- Checks were far too weak: "any of 'sh' in text" for current shell (a report saying `/replay.sh` passes; nearly
  any text contains "sh"), "at least 2 shells from /etc/shells mentioned by name" (no absolute paths, no
  completeness), no check of the existence/executable verification at all. All 5 candidates were accepted.

## Candidate decisions
- A: accepted. Current shell bash, all 13 paths, statement that all exist and are executable (terse but present).
- B, C, E: accepted. Complete, structured.
- D: REJECTED by check "current shell entry names a shell". D's "Current Shell:" value is `/replay.sh`, the name
  of the script that ran its commands, not a shell. Task: "The current shell being used (identified by $0 or
  similar methods)" — the report must identify a shell; `/replay.sh` identifies none. The attempt is wrong, not the
  check (the check accepts any shell name - bash, sh, dash, zsh, ksh... - so it does not impose "bash").
- Empty attempt: rejected (report missing).

## Trace: requirement -> check
- "saved as shell_report.txt in the /app directory" -> file exists and is non-empty (format).
- "The current shell being used (identified by $0 or similar methods)" -> a line labelled "current shell" or "$0"
  whose value (same line or next content line) is a shell name. The specific value (bash) is not asserted: what
  shell the solver's harness used is not something this grader can see; weaker property used.
- "A list of all installed shells found on the system (from /etc/shells)" + "The absolute path to each shell
  executable" -> every one of the 13 distinct /etc/shells paths appears as a whole absolute path (recomputed).
- "Verification that these shells actually exist as executable files" -> report mentions "exist" and
  "executable", and no line containing one of the 13 paths says not found / missing / not executable / fail
  (all 13 really exist and are executable on the image).
- "Verify which shells are listed as valid login shells" -> covered by the /etc/shells path list.
- "clear, well-organized" -> not checkable objectively; no check.

## Trace: check -> requirement
All checks above map to the sentences quoted. Removed: the "'sh' substring" and ">=2 shell names" checks (subsumed
and too weak), and the pytest/uv/network dependency.

## Near misses (all rejected, A/B/C/E accepted)
nm_missing_rbash, nm_dash_wrong_path, nm_no_current_shell, nm_script_as_current_shell, nm_no_verification,
nm_zsh_marked_missing, shortcut_names_only (shell names without paths).

## Contested
None. No numeric/model values involved; the current-shell value is checked only structurally.
