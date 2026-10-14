# Judgement

## The original verifier
- `test.sh` installed curl/uv from the network and ran pytest. The grading container has no network. It only worked
  because uv happened to be cached. I replaced it with a self-contained bash runner (`tests/checks.sh`) that needs
  only bash, coreutils and util-linux (`setpriv`), plus python3 or perl-base to create a unix socket.
- It accepted all five candidates, including three (A, C, E) that just `echo` four fixed lines. E even says
  "Hardcoded bypass for the specific environment". Every check it had passes for any script printing
  `headless/unknown/docker/<anything>`, so it accepted shortcuts.
- It did not check the line order ("in this exact order"), the execute bit ("should be executable"), running
  without root, or the fallback when commands are missing.

## Candidates
| cand | decision | reason |
|---|---|---|
| A | wrong | Hard-coded echo. With DISPLAY=:0, XDG_SESSION_TYPE=x11 and an `Xorg` process running it still prints `headless`. Same for wayland. Fails the task's requirement to "detect". |
| B | correct | Real detection. Passes every check. |
| C | wrong | Hard-coded echo, same as A. |
| D | correct | Real detection. It reports INIT_SYSTEM=sysvinit here (PID 1 is `sleep`, /etc/init.d exists) and VIRTUALIZATION=unknown. Neither value can be fixed from the task text (see contested), so neither value is held against it. |
| E | wrong | Hard-coded echo, with comments listing keywords "to satisfy static analysis". |
| empty | wrong | Script missing. |

The original verifier rejected no candidate, so no rejection had to be traced. The new checks reject A, C and E for
the reason in the table: the x11_detected and wayland_detected checks.

## Trace: requirement -> check
| requirement (task text) | check |
|---|---|
| "script called system_profiler.sh in the /app directory" | exists |
| "The script should be executable" | executable (any x bit). The script is run via its shebang (bash if it has none). |
| "must exit with status code 0" | exit status 0 on every run |
| "exact format (one line per detection, in this exact order)" | exactly 4 lines, `KEY=value`, keys in order DISPLAY_SERVER, INIT_SYSTEM, CONTAINER, VIRTUALIZATION |
| allowed value sets + "All values must be lowercase" | each value must be in its listed set (all lowercase) |
| Display: "Detect whether ... X11, Wayland, or headless. Check $XDG_SESSION_TYPE, $DISPLAY, $WAYLAND_DISPLAY ... validate by checking for running processes" | headless_unset / headless_empty (no vars, no server processes -> headless; this was the original check). x11_detected: XDG_SESSION_TYPE=x11, DISPLAY=:0, processes `Xorg` and `X` running, /tmp/.X11-unix/X0 socket -> x11. wayland_detected: XDG_SESSION_TYPE=wayland, WAYLAND_DISPLAY=wayland-0, compositor processes (weston, sway, gnome-shell, kwin_wayland, mutter, Hyprland) running, $XDG_RUNTIME_DIR/wayland-0 socket -> wayland. Every signal the task names agrees in each case, so the answer is fixed by the task text. |
| Container: "Check for /.dockerenv ..." | container_docker. The grading container is Docker (/.dockerenv exists), so the value is docker. This was the original check; kept. |
| "must work without requiring root privileges" | nonroot: a world-readable copy is run as uid 65534 via setpriv. It must exit 0 with valid format and docker/headless. |
| "gracefully handle missing commands and fall back" | missing_cmds: PATH with ps, pgrep, pidof, pkill, systemctl, systemd-detect-virt, dmidecode, virt-what, lscpu, loginctl and hostnamectl removed. It must exit 0 with valid format and docker/headless. |
| Init system detection | Format/value-set only. A positive test is impossible here: unshare/namespaces are not permitted (tried `unshare --pid`: EPERM), and bwrap is absent on this machine, so I could not test it. |
| Virtualization detection | Format/value-set only. The true value depends on the grading host (here DMI says "Hetzner vServer"), and kvm vs unknown cannot be fixed. |

## Trace: check -> requirement
Every check above follows from the sentence quoted next to it. Nothing the task does not require is checked. Each run
strips surrounding blank lines and per-line whitespace, so a trailing newline is tolerated.

## Contested values (no check added)
- INIT_SYSTEM in a container whose PID 1 is not an init system: `unknown` (B) vs `sysvinit` (D, from /etc/init.d).
  The task names "checking PID 1, presence of systemctl, and init system indicators" but gives no rule for when they
  disagree. There was no existing check, and none is added.
- VIRTUALIZATION on this host: kvm (B, from cpuinfo/DMI) vs unknown (D). It depends on the grading host and on which
  DMI/cpuinfo heuristics count. There was no existing check, and none is added.
- DISPLAY set but no X process running: B and D both say x11. The phrase "validate by checking processes" could mean
  headless here, but it does not say which wins. No check.
No data-fitting readings apply: these are categorical detections, not fitted values.

## Known limitation
A script that detects the display server properly but hard-codes the other three lines cannot be told apart in this
container. Detecting that would need a non-Docker / systemd / VM environment, which a test cannot create without
namespace privileges.

## Near misses (all rejected by the repaired verifier)
nm_wrong_order (B, INIT/CONTAINER swapped), nm_not_executable (B, mode 644), nm_uppercase_x11 (D, prints X11),
nm_exit_status (D, exits 1 when not kvm), nm_requires_root (B, aborts if /proc/1/environ is unreadable),
nm_needs_ps (D, `set -e` + unguarded `ps`), nm_extra_header (B, extra header line),
nm_env_only_display_ignores_session (D, ignores XDG_SESSION_TYPE=wayland), shortcut_hardcoded (copy of A).
