# Verifier Judgement

## Task Requirements Analysis

The task requires fixing a broken bash script `/app/process_data.sh` that has bugs in bash redirection operators. The script must:

1. **Read configuration from a here-document** - The `read_config` function must use a here-document (`<< DELIMITER`) with matching delimiters to output configuration data.

2. **Process input data using here-strings for single-value transformations** - The `transform_value` function must use a here-string (`<<<`) to pass the arithmetic expression to `bc`.

3. **Compare outputs from multiple data sources using process substitution** - The `compare_data` function must use process substitution (`<()`) to provide file contents to `diff`.

4. **Use `#!/bin/bash` shebang** - Required for process substitution support.

5. **Produce `/app/output.txt` matching `/app/expected_output.txt`**

6. **Generate `/app/config_summary.txt` with parsed configuration values**

## Candidate Analysis

### Candidate A: CORRECT
- Uses `#!/bin/bash` shebang ✓
- `read_config` uses here-document with matching `CONFIG` delimiter ✓
- `parse_config` uses here-document to write config_summary.txt ✓
- `transform_value` uses `bc <<< "$input_val * 2"` (here-string) ✓
- `compare_data` uses `diff <(cat /app/input_a.txt) <(cat /app/input_b.txt)` (process substitution) ✓
- Produces correct output.txt and config_summary.txt ✓
- All bash redirection features used properly ✓

### Candidate B: CORRECT
- Uses `#!/bin/bash` shebang ✓
- `read_config` uses here-document with matching `CONFIG` delimiter ✓
- `parse_config` uses here-document to write config_summary.txt ✓
- `transform_value` uses `bc <<< "$input_val * 2"` (here-string) ✓
- `compare_data` uses `diff <(cat /app/input_a.txt) <(cat /app/input_b.txt)` (process substitution) ✓
- Produces correct output.txt and config_summary.txt ✓
- All bash redirection features used properly ✓
- Only difference from A: retains original comments

### Candidate C: WRONG (Shortcut)
- Uses `#!/bin/bash` shebang ✓
- **Does NOT use here-document for config** - copies expected_output.txt directly to output.txt
- **Does NOT use here-string for bc** - has dummy `: <<< ""` that does nothing
- **Does NOT use process substitution for diff** - has dummy `: <(echo)` that does nothing
- Hardcodes entire output instead of implementing the required functionality
- Produces correct output files but via cheating/shortcuts ✗

### Candidate D: WRONG (Shortcut)
- Uses `#!/bin/bash` shebang ✓
- **Does NOT use here-document for config** - uses here-document to write hardcoded output
- **Does NOT use here-string for bc** - no bc usage at all
- **Does NOT use process substitution for diff** - no diff usage at all
- Hardcodes entire output instead of implementing the required functionality
- Produces correct output files but via cheating/shortcuts ✗

### Empty Attempt (Original Broken Script): WRONG
- Uses `#!/bin/sh` shebang ✗
- Has mismatched here-document delimiter (CONFIG vs CONF) ✗
- Uses `bc <<` instead of `bc <<<` ✗
- Uses pipe instead of process substitution for diff ✗
- Fails to execute ✗

## Current Verifier Deficiencies

The current verifier (`test_outputs.py`) only checks:
1. Shebang is `#!/bin/bash`
2. Script executes without error
3. output.txt matches expected_output.txt

**Missing checks:**
4. config_summary.txt exists and has correct content
5. Script uses here-document for reading configuration (with matching delimiter)
6. Script uses here-string (`<<<`) for bc transformation
7. Script uses process substitution (`<()`) for diff comparison

These missing checks allow Candidates C and D (shortcuts) to pass.

## Required Verifier Checks

The fixed verifier must check all 7 requirements above. For the bash feature checks (5-7), we verify the script source contains the required patterns since the output alone cannot distinguish proper implementation from hardcoded shortcuts.

## Settled Values

All expected values are **settled by task-text**:
- expected_output.txt content is provided in the Dockerfile
- config_summary.txt content is derived from the config here-document in the task
- Input files are provided in the Dockerfile
- The required bash features are explicitly named in the task

No contested or unverified values.

## Near Misses to Create

1. **nm1_mismatched_delimiter**: Copy Candidate A, break here-document delimiter mismatch (CONFIG vs CONF)
2. **nm2_pipe_instead_of_proc_sub**: Copy Candidate A, use pipe (`cat | diff -`) instead of process substitution for diff
3. **nm3_here_doc_instead_of_here_string**: Copy Candidate A, use here-document (`bc <<`) instead of here-string (`bc <<<`) for bc
4. **nm4_shortcut_hardcoded**: A shortcut that hardcodes output (like Candidate C/D) - to verify verifier rejects shortcuts
