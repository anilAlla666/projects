# Judgement

## Original verifier
It accepted all five candidates and only rejected a missing report. Its checks were very weak: the "current shell"
check passed if the substring `sh` appeared anywhere in the text, and the /etc/shells check needed only 2 shell
*names*. It also depended on uv/pytest (curl to astral.sh fails offline; it only worked from a cache). I rewrote it
as a single bash script, since bash is the one thing certain to be in the grading image. python3 is not present on this
machine, so I could not run a python version here.

## Candidates
- **A: correct.** It says "Current shell: bash", lists every /etc/shells path by absolute path, and says "All these
  shells exist and are executable." That is minimal but meets every stated requirement. Its extra `/usr/bin/tmux`
  entry came from the solver's own /etc/shells (tmux was installed in its harness), so it is not a fabrication.
- **B, C, E: correct.** Each has a current shell of /bin/bash, the full absolute-path list and an
  existence/executable verification.
- **D: wrong.** Its current shell is `/replay.sh`. That is the `$0` of a wrapper script, not a shell, so the attempt
  did not do "Determine which shell is currently being used". The new current-shell check rejects it. The old
  verifier accepted it only because "sh" is a substring of "replay.sh".

## Settled values
- **Current shell = bash.** Recomputed by the methods the task names ("identified by $0 or similar methods"). On this
  machine (the solver's starting environment), `echo $0` gives `bash`, `$SHELL` gives `/bin/bash`, `ps -p $$` gives
  `bash` and `/proc/$$/exe` gives `/usr/bin/bash`. All of these methods agree, so this is basis "recomputed". Only
  the first shell named near the "current shell"/"$0" line is used, and `/bin/bash`, `-bash` and `bash` are all
  accepted.
- **Shell list = non-comment, executable entries of /etc/shells.** Task-text: "A list of all installed shells found
  on the system (from /etc/shells)" and "The absolute path to each shell". The test reads the grading container's
  /etc/shells, which is built from the same image. Each entry must appear as an exact path token, so `/bin/sh` is not
  matched inside `/usr/bin/sh`. Extra entries are allowed (the solvers' machines had /usr/bin/tmux), and so are
  duplicates.

## Trace: requirement -> check
| Requirement (task text) | Check |
|---|---|
| "Your report should be saved as `shell_report.txt` in the /app directory" | file exists and is non-empty |
| "The current shell being used (identified by $0 or similar methods)" | a "current shell"/"$0" line exists, and the first shell named there or in the following block is bash |
| "A list of all installed shells found on the system (from /etc/shells)" / "Verify which shells are listed as valid login shells" | every /etc/shells entry appears |
| "The absolute path to each shell executable" | each entry appears as an exact absolute-path token (names alone fail) |
| "Verification that these shells actually exist as executable files" | report contains "exist" and "executable", and no line marks a real /etc/shells shell as missing or not executable |
| "clear, well-organized" | not mechanically checkable; no check (format is free) |

Check -> requirement: every check above comes from the sentence quoted next to it. The original checks for "any of
bash/sh/dash" and "≥2 shell names" are replaced by the stricter versions above. The task asks for "all" shells, so
listing only 2 names is not enough.

## Near misses (all rejected by the new verifier; all accepted rows above pass)
- nm_missing_dash (C without /usr/bin/dash)
- nm_wrong_current (E with current shell /bin/zsh)
- nm_no_verification (B without the verification section)
- nm_names_only (shortcut: shell names with no absolute paths)
- nm_false_missing (E saying /usr/bin/zsh does not exist)
- nm_no_current (A without the current-shell line)
- empty attempt: rejected.

## Contested
None.

## Unverified
None. One assumption: the grading container's /etc/shells matches this machine's apart from extra harness entries.
Both come from the same Dockerfile.
