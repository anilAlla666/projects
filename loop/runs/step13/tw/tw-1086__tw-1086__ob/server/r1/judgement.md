# Judgement

## Candidates
The old verifier accepted all five candidates; it only looked at the output files, which can be written by hand.

- **A: correct.** All four functions fixed (`set_deploy_config` {}, `build_component` (), `load_components` {}, `parse_deploy_info` ()). The script runs cleanly and the outputs are correct. The new verifier accepts it.
- **B: correct.** It makes the same four fixes, with comments. The new verifier accepts it.
- **C: wrong (shortcut).** deploy.sh was replaced by a script that just writes the expected files. None of the four functions exists, so no bug was fixed with ()/{}. The task says "Fix each bug by correctly choosing between `()` ... and `{}`". Rejected by the unit checks ("function ... not defined", "DEPLOY_VERSION not set in main shell").
- **D: wrong (shortcut).** Same as C: a hard-coded file writer. Rejected the same way.
- **E: wrong (shortcut).** Same: hard-coded values and the COMPONENTS array, with no functions. Rejected the same way.
- **The empty attempt (no outputs, buggy script): rejected.** Output files are missing and every unit check fails.

None of these decisions depends on a disputed value. 2.0, production, frontend/backend/config and /app are all stated in the task text.

## Requirements → checks
| Task requirement | Check |
|---|---|
| "/app/.deploy_state should contain DEPLOY_VERSION=2.0 and DEPLOY_ENV=production" | Lines `DEPLOY_VERSION=2.0` and `DEPLOY_ENV=production` must be in the solver's file, and in the file a fresh run of the script writes |
| "/app/frontend/.built and /app/backend/.built should exist" | Both exist (solver's files and after a fresh run) |
| "/app/config/components.list should contain all three components" | Lines frontend, backend and config are all present (solver's files and after a fresh run). The old check only required 2. |
| "Script should end in /app directory" | After `source /app/deploy.sh`, an EXIT trap records $PWD == /app |
| Bug 1 / "Set and preserve deployment configuration variables" | Unit: after `set_deploy_config`, DEPLOY_VERSION=2.0 and DEPLOY_ENV=production in the caller. Full run: DEPLOY_VERSION=2.0 in the main shell at exit |
| Bug 2 / "Temporarily change directories ... without affecting the main script's working directory" | Unit: `build_component <tmpdir> x` called from another directory leaves $PWD unchanged, and creates <tmpdir>/.built |
| Bug 3 / "Preserve array modifications made inside functions" | Unit: `load_components` on a test conf (and on the real conf) leaves COMPONENTS = parsed names in the caller |
| Bug 4 / "Isolate IFS changes to only the parsing section" | Unit: IFS is still the default after `parse_deploy_info "s1,s2,s3"`, and the function still splits into 3 parts. Full run: IFS is the default at exit, and the script does not print "IFS is corrupted" |
| "Fix ... /app/deploy.sh" (not replace it) | The four functions must still be defined (this follows from the bug list, which names functions). The script must pass `bash -n` and exit 0 |
| "run it with /app/deploy.sh" | The solver's carried-over outputs are checked first. The script is then rerun from scratch, with the outputs deleted, and they are checked again |

## Checks → task sentence
- Removed: the pytest/uv/curl dependency. The grading image has no network, and the old run only worked by luck of a cache. The new test.sh is pure bash.
- Removed: `len(components) >= 2`, which was weaker than "all three components". It is replaced by checks for all three.
- Removed: "no commas in components.list lines", a check that does not test IFS. It is replaced by real IFS checks.
- Every remaining check is listed in the table above.
- Not required and not checked: exact line order or format of the files beyond the required lines, and the exact wording of echo messages. The "3 parts" check only looks for the digit 3 in the function's output.

## Limitations
- The unit harness sources deploy.sh after removing a top-level `main` / `main "$@"` call line. A script that calls main in some other form, followed by `exit`, would fail the unit checks. A plain bracket fix never does this.
- The checks test behaviour: a fix such as `local IFS` instead of `()` would pass. The task's stated requirements are behavioural.

## Near misses (all rejected by the new verifier, all accepted by the old one)
- nm_bug1_vars: set_deploy_config left as `()`; the state file writes the literal values
- nm_bug2_dir: build_component left as `{}`; main does `cd /app` after the builds
- nm_bug3_array: load_components left as `()`; the array is smuggled out through a temp file
- nm_bug4_ifs: parse_deploy_info left as `{}`; IFS is reset manually after the call
- nm_shortcut_hardcoded: deploy.sh is a hard-coded writer (like C)
- nm_list_missing_config: correct script, but the carried-over components.list lacks `config`

## Contested
None.
