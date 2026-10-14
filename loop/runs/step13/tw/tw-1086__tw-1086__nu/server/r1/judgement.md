# Judgement: Verifier Analysis and Repair

## Task Requirements Analysis

The task requires fixing FOUR specific bugs in `/app/deploy.sh` by correctly choosing between `()` (subshell) and `{}` (command grouping):

1. **Variable Persistence Bug**: `set_deploy_config` uses `()` but should use `{}` so DEPLOY_VERSION and DEPLOY_ENV persist
2. **Directory Isolation Bug**: `build_component` uses `{}` but should use `()` so directory changes don't leak
3. **Array Scope Bug**: `load_components` uses `()` but should use `{}` so COMPONENTS array modifications persist
4. **IFS Isolation Bug**: `parse_deploy_info` uses `{}` but should use `()` so IFS changes are isolated

The script must produce these outputs after running:
- `/app/.deploy_state` contains `DEPLOY_VERSION=2.0` and `DEPLOY_ENV=production`
- `/app/frontend/.built` and `/app/backend/.built` exist
- `/app/config/components.list` contains all three components (frontend, backend, config)
- Script ends in `/app` directory

## Candidate Analysis

### Candidate A: CORRECT
- Fixes all four bugs correctly:
  - `set_deploy_config() { ... }` - uses `{}` ✓
  - `build_component() { ( ... ) }` - uses `()` ✓
  - `load_components() { ... }` - uses `{}` ✓
  - `parse_deploy_info() { ( ... ) }` - uses `()` ✓
- Produces all expected outputs
- **VERDICT: Should be ACCEPTED**

### Candidate B: CORRECT
- Same fixes as Candidate A, with explanatory comments
- All four bugs fixed correctly
- Produces all expected outputs
- **VERDICT: Should be ACCEPTED**

### Candidate C: WRONG (Shortcut/Hardcoded)
- Completely bypasses the function logic
- Just hardcodes output files directly in `main()`
- Does NOT fix the bugs in the functions - the functions don't even exist in the expected form
- The script doesn't use subshells or command grouping at all for the buggy functions
- **VERDICT: Should be REJECTED** - This is a shortcut that fabricates results without fixing the actual bugs

### Candidate D: WRONG (Shortcut/Hardcoded)
- Same as Candidate C, just with mkdir calls
- Hardcodes output files, doesn't fix the functions
- **VERDICT: Should be REJECTED** - Shortcut

### Candidate E: WRONG (Partial/Workaround)
- Sets variables directly in `main()` and hardcodes COMPONENTS array
- Bypasses the buggy functions instead of fixing them
- The functions `set_deploy_config`, `load_components`, `parse_deploy_info` are not even called properly
- `build_component` is not used at all
- **VERDICT: Should be REJECTED** - Doesn't fix the bugs, works around them

## Current Verifier Problems

The current verifier (`test_outputs.py`) ONLY checks output files:
1. `.deploy_state` content
2. `.built` marker files existence
3. `components.list` content and format
4. IFS isolation via file format checks

It does NOT verify that the script actually fixes the four bugs by using correct `()` vs `{}` constructs. This allows shortcuts (C, D, E) to pass.

## Required Verifier Changes

The verifier must ADD checks that verify the SOURCE CODE of `/app/deploy.sh` contains the correct fixes:

1. **Check `set_deploy_config` uses `{}` (command grouping), not `()`**
2. **Check `build_component` uses `()` (subshell) for directory isolation**
3. **Check `load_components` uses `{}` (command grouping), not `()`**
4. **Check `parse_deploy_info` uses `()` (subshell) for IFS isolation**

These are format/structure checks that the task explicitly requires - the task says "Fix each bug by correctly choosing between `()` (subshell - isolates side effects) and `{}` (command group - preserves side effects)".

## Near Misses to Create

I need to create at least 3 near misses, each breaking exactly one requirement:

1. **NM1: Variable Persistence Bug Not Fixed** - `set_deploy_config` still uses `()` instead of `{}`
2. **NM2: Directory Isolation Bug Not Fixed** - `build_component` still uses `{}` instead of `()`
3. **NM3: Array Scope Bug Not Fixed** - `load_components` still uses `()` instead of `{}`
4. **NM4: IFS Isolation Bug Not Fixed** - `parse_deploy_info` still uses `{}` instead of `()`

I'll create 4 near misses (one per bug) plus potentially a shortcut near miss.

## Basis for Checks

All checks are **task-text** basis - the task explicitly states:
- "Fix each bug by correctly choosing between `()` (subshell - isolates side effects) and `{}` (command group - preserves side effects)"
- Lists the four specific bugs with which construct should be used
- The expected results are explicitly stated in the task

No "contested" or "unverified" values - everything is specified in the task text.
