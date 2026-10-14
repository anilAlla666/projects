# Grader report: tw-1169

Rounds: 1 · cost $0.97 · candidates judged correct: B, D · judged wrong: A, C, E

| basis | checks |
|---|---:|
| task-text | 7 |
| recomputed | 0 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 3 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** INIT_SYSTEM when PID 1 is not an init (container): the verifier expects any listed value (no check), the candidates give unknown (B), sysvinit (D); task gives no precedence between PID 1 and /etc/init.d-style indicators
- **contested:** VIRTUALIZATION on the grading host: the verifier expects any listed value (no check), the candidates give kvm (B), unknown (D); depends on host and on which DMI/cpuinfo heuristics count; task names no rule
- **contested:** DISPLAY_SERVER when DISPLAY is set but no X process runs: the verifier expects not tested, the candidates give x11 (B, D); task says to validate by processes but not which wins on conflict

## Every check

| check | requirement | basis |
|---|---|---|
| script exists at /app/system_profiler.sh | Build a shell script called `system_profiler.sh` in the /app directory | format |
| script has an execute bit | The script should be executable | format |
| exit status 0 on every run | The script must exit with status code 0 on successful completion | task-text |
| exactly 4 KEY=value lines in order DISPLAY_SERVER, INIT_SYSTEM, CONTAINER, VIRTUALIZATION, values from listed sets | The output must be in the following exact format (one line per detection, in this exact order); All values must be lowercase | format |
| CONTAINER=docker in the grading container | Detect if the system is running inside a container ... Check for /.dockerenv | task-text |
| DISPLAY_SERVER=headless with display vars empty or unset | Detect whether the system is using X11, Wayland, or running headless | task-text |
| DISPLAY_SERVER=x11 when XDG_SESSION_TYPE=x11, DISPLAY=:0, Xorg/X processes and X0 socket exist | Check $XDG_SESSION_TYPE, $DISPLAY, $WAYLAND_DISPLAY environment variables and validate by checking for running processes | task-text |
| DISPLAY_SERVER=wayland when XDG_SESSION_TYPE=wayland, WAYLAND_DISPLAY=wayland-0, compositor processes and socket exist | Check $XDG_SESSION_TYPE, $DISPLAY, $WAYLAND_DISPLAY environment variables and validate by checking for running processes | task-text |
| runs as uid 65534 with exit 0, valid format, docker/headless | The script must work without requiring root privileges | task-text |
| runs with ps/pgrep/systemctl/systemd-detect-virt/etc. removed from PATH, exit 0, valid format, docker/headless | The script should gracefully handle missing commands and fall back to alternative detection methods | task-text |
