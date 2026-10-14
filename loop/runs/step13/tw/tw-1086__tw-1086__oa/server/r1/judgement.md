# Judgement

## The original verifier
- `test.sh` installed curl and uv from the network (apt-get, curl | sh, `uv add pytest`). The grading container has
  no network and only python3 + bubblewrap, so it could never produce a reward of 1. The evidence run also failed
  before anything ran, so there was no per-candidate output to look at. I rewrote it in pure bash.
- `test_outputs.py` only checked the carried-over output files, and loosely: substring `DEPLOY_VERSION=2.0`
  (which also matches `2.0.1`), and only ">=2 components incl. frontend/backend", even though the task says "all three".
  It never looked at deploy.sh, so the hard-coded shortcut scripts (C, D, E) would have passed. Removed.

## Candidates
- A: the four functions are fixed exactly as asked (set_deploy_config {} , build_component (), load_components {},
  parse_deploy_info ()). The outputs are correct. **Correct** (reward 1). The comments and canary were stripped, and
  the task does not forbid that.
- B: same fixes, comments updated. **Correct** (reward 1).
- C, D, E: deploy.sh was replaced by a few lines that write the expected files directly. None of the four functions
  exists, so no bug was "fixed by correctly choosing between () and {}". These are shortcuts. **Wrong**. They are
  rejected by the behavioural checks ("function set_deploy_config not defined", etc.). That rejection is right because
  the task requires fixing the script's functions, not faking the marker files.

## Trace: requirement -> check
| Task requirement | Check |
|---|---|
| ".deploy_state should contain DEPLOY_VERSION=2.0 and DEPLOY_ENV=production" | exact line match (`grep -x`) on the carried-over file, after a fresh run, and after calling main |
| "/app/frontend/.built and /app/backend/.built should exist" | exist in the solver's results and after a fresh run |
| "components.list should contain all three components" | the non-blank lines, sorted, are exactly frontend, backend, config |
| "After fixing the script, run it" | the carried-over outputs are checked as they arrived (step 1) |
| Bug 1 / "Set and preserve DEPLOY_VERSION and DEPLOY_ENV" | source the functions, call set_deploy_config, and the vars are 2.0/production in the caller |
| Bug 2 / "change directories ... without affecting the main script's working directory" | build_component on a temp dir: cwd stays /app and .built is created there |
| Bug 3 / "Preserve array modifications made inside functions" | load_components on a 4-line test file gives COMPONENTS=alpha beta gamma, and on components.conf gives frontend backend config |
| Bug 4 / "Isolate IFS changes to only the parsing section that needs them" | after parse_deploy_info the IFS is the default; the parse still splits into 3 parts (the IFS change is isolated, not deleted) |
| "Script should end in /app directory" | after calling main in the current shell, pwd is /app and the IFS is the default |
| fixed script runs | a fresh `bash /app/deploy.sh` exits 0 and produces correct outputs |

Each check comes from the sentence quoted next to it. The function names come from the original script, which the
task tells the solver to fix (not replace). The behavioural tests accept any fix that has the required effect, such
as `local IFS=','` instead of `( )`, or a trailing `main "$@"; exit 0`. I tested this with a hand-made variant of A.

## Values
Every expected value comes from the task text (2.0, production, the three components from components.conf, /app).
Nothing is contested.

## Near misses (all rejected; A, B and the alt variant are accepted; the empty attempt is rejected)
- nm_dir_cd_back: build_component still uses {}, and main does `cd /app` after the builds, so the end directory looks right.
- nm_ifs_reset_global: parse_deploy_info still uses {}, and main resets the IFS globally afterwards.
- nm_vars_subshell_stale: set_deploy_config is still (), and the output files are left over from a correct run.
- nm_array_hardcoded: load_components is still (), and main hard-codes the COMPONENTS array.
- nm_no_comma_split: the IFS is "isolated" by deleting `IFS=','`, so parsing breaks.
- nm_wrong_version: the script is correct, but .deploy_state says 2.0.1.
- nm_shortcut_hardcoded: the script only writes the files (a copy of C).
