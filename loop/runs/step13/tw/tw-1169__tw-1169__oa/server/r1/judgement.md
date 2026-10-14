# Judgement

## The original verifier was broken
`evidence.md` has no per-candidate runs (the probe harness crashed on a missing config.json). Reading the original
`test.sh`: it runs `apt-get install curl`, then `curl ... uv install`, then `uv add pytest`. The grading container
has **no network**, so uv/pytest never install, `uv run pytest` fails, and the reward is always 0. That rejects every
attempt, correct ones included. I rewrote the verifier in plain bash (no pytest, no network) and removed
`test_outputs.py`. Its value checks are carried over unchanged (headless with empty vars, CONTAINER=docker, the
allowed value sets).

## Candidates
- **A**: the script is just 4 hard-coded `echo` lines (VIRTUALIZATION=none, although this host is a KVM guest:
  the cpuinfo hypervisor flag is set and the vendor is Hetzner vServer). Detects nothing. **Wrong** (shortcut). Rejected by the x11 and wayland scenario
  checks: it still prints `headless` when DISPLAY/XDG_SESSION_TYPE=x11 are set and an Xorg process is running.
- **C**: hard-coded echoes, with a comment saying "Hardcoded bypass" and keywords pasted in to fool static checks. **Wrong**
  (shortcut). Rejected by the same scenario checks.
- **E**: hard-coded echoes. **Wrong** (shortcut). Rejected by the same scenario checks.
- **B**: real detection of all four items, with process validation and fallbacks. Passes every check. **Correct**.
- **D**: real detection. It reports INIT_SYSTEM=sysvinit (because /etc/init.d exists) and VIRTUALIZATION=unknown
  (because it reads only product_name and sys_vendor, which are "vServer"/"Hetzner", and the hypervisor flag is set). Neither
  value is settled by the task (see "contested"), and the grader's host may differ from this one, so neither is checked.
  Passes every check. **Correct**.
- **empty**: no script. Rejected by the existence check.

## Requirement -> check
| Requirement (task text) | Check |
|---|---|
| "Build a shell script called `system_profiler.sh` in the /app directory" | file exists |
| "The script should be executable" | `test -x` |
| "must exit with status code 0" | rc==0 in every run |
| "exact format (one line per detection, in this exact order)" + value sets | regex: exactly 4 lines, in order, only the allowed values (stdout only; stderr ignored) |
| "All values must be lowercase" | covered by the value sets in the regex |
| Display server: "Check $XDG_SESSION_TYPE, $DISPLAY, $WAYLAND_DISPLAY ... validate by checking for running processes" | headless with the vars unset and with the vars empty (original); x11 when DISPLAY=:0, XDG_SESSION_TYPE=x11, Xorg and X processes running, /tmp/.X11-unix/X0 socket present; wayland when WAYLAND_DISPLAY=wayland-0, XDG_SESSION_TYPE=wayland, a socket in XDG_RUNTIME_DIR, weston/sway/gnome-shell/kwin_wayland/mutter/Hyprland processes running. Every signal the task names points the same way, so the expected value follows from the task text. |
| Container: "Check for /.dockerenv ..." | CONTAINER=docker (original check; /.dockerenv exists in the docker-built grading image, as it does on this machine) |
| "must work without requiring root privileges" | script copied and run as uid 65534 via setpriv: rc 0, valid format |
| Init system detection | format only (value contested) |
| Virtualization detection | format only (value depends on the grader's host; not checkable) |
| "gracefully handle missing commands and fall back" | partly covered: the grading image has no systemd-detect-virt or systemctl, so a script with no fallback must still exit 0 with valid output |

Check -> requirement: every check above comes from the sentence beside it. The original's install steps are removed
(not required, and they break the verifier).

## Contested
- INIT_SYSTEM in a container where PID 1 is not an init (here PID 1 is `sleep`). Under the "PID 1" reading the answer
  is `unknown`. Under the "init system indicators" reading (/etc/init.d exists) it is `sysvinit`. The task does not pick one. The original had no check. None added.
- VIRTUALIZATION: depends on the grader's host hardware and on how a hypervisor flag without a vendor match is mapped
  (unknown or kvm). The original had no check. None added.
No data-fitting applies here (no numeric data).

## Near misses (all rejected; B and D accepted)
nm_order (CONTAINER and INIT lines swapped), nm_upper (`Docker`), nm_exit (exit 1), nm_noexec (mode 644),
nm_nowayland (the wayland branch prints x11), nm_root (refuses to run as non-root), nm_nodocker (ignores /.dockerenv),
nm_extraline (an extra stdout line), shortcut_hardcoded (copy of A).
