# Grader report: tw-1169

Rounds: 1 · cost $0.85 · candidates judged correct: B, D · judged wrong: A, C, E

| basis | checks |
|---|---:|
| task-text | 6 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 3 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** INIT_SYSTEM inside the grading container (PID 1 not an init): the verifier expects any allowed value, the candidates give B: unknown, D: sysvinit; task gives no rule for containers whose PID 1 is an arbitrary process
- **contested:** DISPLAY_SERVER when DISPLAY is set but no X process runs: the verifier expects not tested, the candidates give B and D: x11; task says validate by processes but not what to report when validation fails

## Every check

| check | requirement | basis |
|---|---|---|
| /app/system_profiler.sh exists and is executable | Build a shell script called system_profiler.sh in the /app directory; The script should be executable | format |
| exit status 0 in every scenario | The script must exit with status code 0 on successful completion | task-text |
| exactly 4 lines in the order DISPLAY_SERVER, INIT_SYSTEM, CONTAINER, VIRTUALIZATION with allowed lowercase values | The output must be in the following exact format (one line per detection, in this exact order); All values must be lowercase | format |
| display vars unset or empty -> DISPLAY_SERVER=headless | Detect whether the system is using X11, Wayland, or running headless | task-text |
| XDG_SESSION_TYPE=x11, DISPLAY=:0, running Xorg process -> x11 | Check $XDG_SESSION_TYPE, $DISPLAY, $WAYLAND_DISPLAY ... and validate by checking for running processes | task-text |
| XDG_SESSION_TYPE=wayland, WAYLAND_DISPLAY=wayland-0, running weston process -> wayland | same as above | task-text |
| CONTAINER=docker in the grading container | Detect if the system is running inside a container and identify the type (Docker...). Check for /.dockerenv | task-text |
| fake systemd-detect-virt kvm/vmware/oracle/microsoft -> kvm/vmware/virtualbox/hyperv | Use systemd-detect-virt if available | recomputed |
| script run as uid 65534 exits 0 with valid output, headless, docker | The script must work without requiring root privileges | task-text |
| INIT_SYSTEM within enum only | INIT_SYSTEM=<systemd|sysvinit|openrc|unknown> | format |
