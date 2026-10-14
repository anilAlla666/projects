# Judgement

## Candidates
The original verifier accepted all five. It only checked: 4 `KEY=value` lines with allowed values (order not checked),
exit 0, CONTAINER=docker, DISPLAY_SERVER=headless with display vars emptied. A script that always prints the same
four lines passes it.

- **A**: hard-coded `echo` of four constant lines. Does not detect anything. When a display session or a hypervisor is
  simulated, it still prints headless / none. **Wrong** (shortcut).
- **C**: hard-coded constants (VIRTUALIZATION=kvm). **Wrong** (shortcut).
- **E**: hard-coded constants, and the detection methods appear only in comments ("Hardcoded bypass for the specific environment"). **Wrong** (shortcut).
- **B**: really detects. It checks env vars plus processes, PID 1/systemctl, /.dockerenv, /run/.containerenv, cgroup,
  `container=`, `systemd-detect-virt --vm` with the documented ids mapped, and falls back to DMI. Passes every scenario. **Correct**.
- **D**: really detects. It trusts env vars for the display server and does not validate them against processes. It
  calls plain `systemd-detect-virt` with the ids mapped, falls back to DMI and cpuinfo. Its INIT_SYSTEM here is `sysvinit`
  (because /etc/init.d exists while PID 1 is `sleep`), where B gives `unknown`. The task fixes neither value for a container whose
  PID 1 is not an init, so the verifier checks neither. Its outputs are right in every scenario whose answer is settled. **Correct**.
  It skips the process-validation *method*, but the only case where that changes the output (DISPLAY set but no X process) is
  contested, see below.

## Requirement -> check
| Requirement (task text) | Check |
|---|---|
| "Build a shell script called `system_profiler.sh` in the /app directory" | file exists |
| "The script should be executable" | `-x /app/system_profiler.sh` (new) |
| "must exit with status code 0" | rc==0 in every scenario |
| "exact format (one line per detection, in this exact order)" + allowed values + "All values must be lowercase" | exactly 4 lines, in the fixed order, each matching `KEY=(allowed values)` (order is new; the old verifier parsed into a dict) |
| Display: headless when no display server | vars unset, and vars empty -> headless (original kept, unset variant added) |
| Display: X11 detection via $XDG_SESSION_TYPE/$DISPLAY + running process | XDG_SESSION_TYPE=x11, DISPLAY=:0, a running process named `Xorg`, /tmp/.X11-unix/X0 present -> x11 (new) |
| Display: Wayland | XDG_SESSION_TYPE=wayland, WAYLAND_DISPLAY=wayland-0, a socket in XDG_RUNTIME_DIR, a running process `weston` -> wayland (new) |
| Container: docker detection | real grading container (a Docker container, /.dockerenv) -> CONTAINER=docker (original kept) |
| Virtualization: "Use systemd-detect-virt if available" | fake `systemd-detect-virt` on PATH reporting kvm / vmware / oracle / microsoft -> kvm / vmware / virtualbox / hyperv (new). The ids come from systemd-detect-virt(1), the tool the task names: `oracle` = Oracle VM VirtualBox, `microsoft` = Hyper-V. With `-c` the fake reports `docker` |
| "must work without root privileges" | script copy run as uid/gid 65534 via setpriv: exit 0, valid format, headless, docker (new; runs only if the verifier is root) |
| "gracefully handle missing commands" | covered implicitly. The grading image has no systemd-detect-virt, no X tools and no systemctl, and every run must still exit 0 with valid output |
| Init system | enum check only (see contested) |

Every check maps to a sentence above, and no check exists that the task does not require. The old file also needed
uv/pytest from the network. It now runs in pure bash and always writes reward.txt.

## Contested / not checked
- INIT_SYSTEM value in the grading container, where PID 1 is not an init (`sleep` here): B gives unknown, D gives sysvinit. The task gives no
  rule, so only the enum is checked. This matches the original.
- DISPLAY set but no X process running: x11 or headless? "validate by checking for running processes" leaves
  the fallback open. Not tested.
- systemd-detect-virt reports none but DMI shows a hypervisor: whether to fall back is open. Not tested. qemu->kvm: not tested.
- Faking podman/lxc/none containers by removing /.dockerenv would still leave real Docker traces (mountinfo), so a correct
  script could legitimately say docker. Not tested. CONTAINER=docker in the real grader container is kept from the original.

## Near misses (all copies of B unless noted; all rejected, B and D accepted)
nm_order (CONTAINER/VIRTUALIZATION lines swapped), nm_no_wayland (wayland never reported), nm_virtualbox_unmapped
(`oracle` not mapped), nm_exit1, nm_needs_root (reads root-only /proc/1/environ under set -e), nm_not_executable (mode 644),
nm_shortcut_env_only (shortcut: display from env vars only, the rest hard-coded).
