# Grader report: tw-1169

Rounds: 1 · cost $0.68 · candidates judged correct: B, D · judged wrong: A, C, E

| basis | checks |
|---|---:|
| task-text | 3 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 5 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** INIT_SYSTEM inside the container: the verifier expects any allowed value (no check), the candidates give unknown (A,B,C,E), sysvinit (D); the task does not say whether /etc/init.d counts as a sysvinit indicator when PID 1 is not an init
- **contested:** VIRTUALIZATION: the verifier expects any allowed value (no check), the candidates give kvm (B,C,E), none (A), unknown (D); depends on the grader's host and on how a bare hypervisor flag is mapped

## Every check

| check | requirement | basis |
|---|---|---|
| script exists at /app/system_profiler.sh | Build a shell script called system_profiler.sh in the /app directory | format |
| script has exec bit | The script should be executable | format |
| exit status 0 in every run | The script must exit with status code 0 on successful completion | format |
| exactly 4 lines, exact order, allowed lowercase values | The output must be in the following exact format (one line per detection, in this exact order); All values must be lowercase | format |
| DISPLAY_SERVER=headless with display vars unset or empty | Detect whether the system is using X11, Wayland, or running headless | task-text |
| DISPLAY_SERVER=x11 with DISPLAY=:0, XDG_SESSION_TYPE=x11, Xorg/X processes and X0 socket | Check $XDG_SESSION_TYPE, $DISPLAY, $WAYLAND_DISPLAY environment variables and validate by checking for running processes | task-text |
| DISPLAY_SERVER=wayland with WAYLAND_DISPLAY, XDG_SESSION_TYPE=wayland, compositor processes and socket | Check $XDG_SESSION_TYPE, $DISPLAY, $WAYLAND_DISPLAY environment variables and validate by checking for running processes | task-text |
| CONTAINER=docker | Detect if the system is running inside a container and identify the type ... Check for /.dockerenv | recomputed |
| runs as uid 65534 with rc 0 and valid format | The script must work without requiring root privileges | format |
