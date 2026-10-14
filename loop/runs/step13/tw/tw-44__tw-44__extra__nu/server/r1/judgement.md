# Judgement: Verifier Repair for Shell Report Task

## Task Requirements Traceability

| Task Requirement | Verifier Check | Basis |
|------------------|----------------|-------|
| "report should be saved as shell_report.txt in the /app directory" | File existence check at /app/shell_report.txt | task-text |
| "The current shell being used (identified by $0 or similar methods)" | Checks report mentions "current shell" with "bash" or "/bin/bash"; rejects script paths | task-text |
| "A list of all installed shells found on the system (from /etc/shells)" | Compares report shell list against /etc/shells at runtime; requires all entries present | task-text + recomputed |
| "The absolute path to each shell executable" | Verifies each /etc/shells path appears in report | task-text |
| "Verification that these shells actually exist as executable files" | Checks for verification language (exist, executable, verified) per shell or general statement | task-text |
| Implied: No fabricated shells | Rejects paths not in /etc/shells (especially /usr/bin/tmux) | task-text |

## Candidate Analysis

All five candidates were rejected by the repaired verifier:

| Candidate | Verdict | Reason |
|-----------|---------|--------|
| A | WRONG | Fabricates /usr/bin/tmux (not in /etc/shells); minimal verification |
| B | WRONG | Fabricates /usr/bin/tmux; otherwise well-structured |
| C | WRONG | Fabricates /usr/bin/tmux; otherwise well-structured |
| D | WRONG | Current shell identified as "/replay.sh" (script path, not shell) |
| E | WRONG | Fabricates /usr/bin/tmux; mislabels $SHELL as $0 |

**No candidate is fully correct.** All have fatal errors.

## Near Misses Created

Four near misses, each breaking exactly one requirement:

1. **nm1_wrong_current_shell**: Correct in all respects except identifies current shell as `/bin/zsh` instead of bash.
2. **nm2_missing_shell**: Correct except omits `/usr/bin/dash` from the shell list.
3. **nm3_fabricated_shell**: Correct except adds fabricated `/usr/bin/tmux` to shell list.
4. **nm4_no_verification**: Correct except lacks executable verification for shells.

All near misses are correctly rejected by the verifier.

## Verifier Design Decisions

1. **Current shell identification**: Accepts "bash" (from $0) or "/bin/bash" (from $SHELL). Rejects script paths (e.g., "/replay.sh", "/home/user/script.sh") but allows mentioning script path alongside correct shell.

2. **Shell list from /etc/shells**: Reads /etc/shells at verification time (in grading container). Requires ALL entries (including duplicate `/usr/bin/zsh`). Rejects any extra paths not in /etc/shells.

3. **Duplicate /usr/bin/zsh**: /etc/shells contains `/usr/bin/zsh` twice. Verifier warns if report mentions it < 2 times but doesn't fail (some reports may deduplicate with note).

4. **Executable verification**: Accepts either per-shell verification statements or a general "all shells verified" statement.

5. **Fabricated shells**: Explicitly rejects `/usr/bin/tmux` (common error). Generally flags any `/bin/*` or `/usr/bin/*` path not in /etc/shells.

## Contested Values

None. All expected values are settled by:
- task-text (file location, required sections)
- recomputed (reading /etc/shells at runtime in the grading container)

The grading container is built from the same Dockerfile, so /etc/shells content is deterministic.

## Unverified Checks

None. All checks trace directly to task requirements or recomputed ground truth.
