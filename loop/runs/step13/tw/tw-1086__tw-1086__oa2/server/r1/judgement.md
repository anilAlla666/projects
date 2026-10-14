# Judgement

## Candidates
The old verifier accepted all five, because it checked only the output files, which can be written by hand.
- A: correct. It makes the four fixes. The new verifier accepts it.
- B: correct. It makes the same four fixes and only changes comments otherwise. Accepted.
- C, D, E: wrong. Each replaced deploy.sh with a script that writes the expected files directly
  (hard-coded `printf` / `COMPONENTS=(frontend backend config)`). None of the four functions is left, so no bug was
  "fixed by correctly choosing between () and {}". These are shortcuts. Rejected by the "required functions missing"
  check and the per-function behaviour checks.

## Requirements -> checks
- "/app/.deploy_state should contain DEPLOY_VERSION=2.0 and DEPLOY_ENV=production" -> lines checked in the solver's file,
  after a fresh run and after a run with an extra component (task-text).
- "/app/frontend/.built and /app/backend/.built should exist" -> checked in the solver's output, again after
  build_component is called directly, and after a fresh run (task-text).
- "/app/config/components.list should contain all three components" -> one line each for frontend, backend and config.
  The old check left out config; it is added now (task-text).
- "Script should end in /app directory" -> the script is sourced from /app and pwd is then /app (task-text).
- Bug 1 / "Set and preserve deployment configuration variables" -> after calling set_deploy_config, the caller sees
  DEPLOY_VERSION=2.0 and DEPLOY_ENV=production.
- Bug 2 / "Temporarily change directories ... without affecting the main script's working directory" -> the caller's
  pwd is unchanged after build_component, and .built is created in the component directory.
- Bug 3 / "Preserve array modifications made inside functions" -> after calling load_components, COMPONENTS is
  (frontend backend config).
- Bug 4 / "Isolate IFS changes to only the parsing section" -> IFS is unchanged after parse_deploy_info, the function
  still prints "Parsed 3 parts", and IFS is the default after the whole script is sourced. The check tests behaviour,
  so `local IFS` is accepted too.
- "run it with /app/deploy.sh" -> deploy.sh exits 0 and regenerates all the outputs after they are deleted. The log
  must not contain "IFS is corrupted" or the fallback warning. These strings come from the given script.
- Outputs are produced by the script from components.conf, not fabricated ("Configuration file listing components to
  deploy"). A run with an extra conf line must list that component. This catches a hard-coded components.list.
  It is a structural/anti-shortcut check, not a new expected value.

## Checks -> requirements; removed or changed
- Removed: the dependency on uv/pytest/curl in test.sh. The grading image has no network, and the task does not
  require them. test.sh is now pure bash.
- The old "deploy_state has >=2 lines" and "no commas in components.list" checks are kept as part of the stricter
  line checks.
- No expected value is contested. Every value is stated in the task text.

## Near misses (all rejected; A and B accepted)
nm_ifs_leak (parse_deploy_info back to {}), nm_dir_leak (build_component back to {}),
nm_var_subshell (set_deploy_config back to (), output files left from A),
nm_array_subshell (load_components back to (), output files left from B),
nm_hardcoded_list (fixed functions but components.list printf hard-coded), nm_list_missing_config (A's script, solver
components.list missing config). Empty attempt: rejected.
Also checked: A with `local IFS=","` and `main "$@"` is accepted.
No reading-comparison data analysis was needed, because no value was contested.
