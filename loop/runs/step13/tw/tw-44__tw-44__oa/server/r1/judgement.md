# Judgement

## State of the original verifier
- Evidence: the probe never ran the verifier (grader config missing), so there is no run output.
- `test.sh` ran `apt-get`, downloaded `uv` with `curl` and installed pytest. The grading container has **no network**,
  so every attempt (correct ones too) would get 0. That alone made the verifier reject everything.
- `test_outputs.py` was also far too weak: "current shell" passed if the substring `sh` appeared anywhere; "/etc/shells"
  passed if 2 shell *basenames* appeared; there was no check of absolute paths or of the executable verification.
  So a names-only report would pass.
- Replaced with `tests/check_report.sh` (bash + GNU grep/awk/sed only; python3 is not even present on this machine,
  so a bash checker can be tested here and works in the grading image) and a new `test.sh` that writes 1/0.

## Candidates
| cand | decision | reason |
|---|---|---|
| A | correct | "Current shell: bash"; every /etc/shells path listed; "All these shells exist and are executable." Minimal but meets every listed requirement. |
| B | correct | all sections present. |
| C | correct | all sections present. |
| D | **wrong** | "Current Shell: /replay.sh". That is the script name `$0` gave inside the harness's replay script, not a shell. The task asks to "Determine which shell is currently being used"; `/replay.sh` is not a shell, so requirement 1 is not met. Rejected by the current-shell check. Its shell list and verification are fine. |
| E | correct | all sections present. |

Candidates A, B, C, E list `/usr/bin/tmux` (their machine's /etc/shells had tmux, added by the harness); D and this
machine show a duplicate `/usr/bin/zsh` instead. So /etc/shells differs between machines. The checker therefore requires
the unique entries of the grading container's /etc/shells, minus harness-only entries (tmux/screen), and does not
penalise extra entries.

## Trace: requirement -> check
| requirement (task text) | check |
|---|---|
| "Your report should be saved as `shell_report.txt` in the /app directory" | file exists and isn't empty |
| "The current shell being used (identified by $0 or similar methods)" | a line mentioning current shell / $0 / $SHELL names a real shell (basename of a token in {sh,bash,dash,zsh,ksh,...}), either on that line or on the next non-blank line. Which shell it is isn't fixed by the task (it depends on the harness), so any real shell is accepted. |
| "A list of all installed shells found on the system (from /etc/shells)" + "The absolute path to each shell executable" + "Verify which shells are listed as valid login shells" | every unique /etc/shells entry (except tmux/screen) appears as a standalone absolute path |
| "Verification that these shells actually exist as executable files" | the report mentions "execut…" and exist/verif/found/present/ok; no line naming an actually-executable shell says it is missing/not found/not executable |
| "clear, well-organized" | not checked mechanically (subjective); no check |

## Trace: check -> requirement
All checks above map to the quoted sentences; the old checks (substring `sh`, ≥2 basenames) were replaced by stricter
versions of the same requirements. Network installation of pytest was removed (format/infrastructure).

## Expected values
- The /etc/shells entries are recomputed in the grading container from /etc/shells (as the task says, "from /etc/shells").
- The identity of the current shell is contested (bash vs sh depends on the solver's harness); the check accepts any real shell, and the old verifier also accepted any shell name.

## Near misses (all rejected; A,B,C,E accepted; empty rejected; tested here with a simulated /tests and /logs)
- nm1_missing_one_shell (B without /bin/zsh)
- nm2_no_current_shell (C with the current shell unknown, `$0` = /replay.sh)
- nm3_no_verification (E without the verification section)
- nm4_wrong_verification (E saying /usr/bin/dash NOT FOUND)
- nm5_names_only_shortcut (shell names, no paths; the old verifier would have passed this)
- nm6_relative_paths (B with bare names instead of absolute paths)
