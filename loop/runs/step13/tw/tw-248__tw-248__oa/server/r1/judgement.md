# Judgement

## State of the original verifier
The evidence shows the original verifier was never run. Its `test.sh` runs `apt-get install curl` and
downloads `uv` from the internet, then installs pytest. The grading container has no network, so that
script fails, and pytest never runs or runs on nothing. That means the original verifier
rejects **every** attempt, correct ones included. Its `test_outputs.py` was also very weak (one interface name
plus any of the words "driver/module/version/path" was enough to pass). It compared against the
*grading container's* `ip link` and `/sys/class/net/*/device/driver`, which is a different machine from the solver's (the
candidates' own reports show different kernels, MACs and veth peer indexes). I replaced it with
`check_report.pl`. It uses only perl-base, which is Essential in ubuntu:24.04. I could test it here because this machine has no python3.
`test.sh` writes 1 or 0 to /logs/verifier/reward.txt.

## Candidates
- A: wrong. The report says eth0 (shown as `eth0@if1371`, a veth) has "No driver loaded". It never gives
  the module path, dependencies, version or parameters for any interface (task items 3-6). Rejected by the
  check that the report covers all six items.
- B: wrong. It names driver veth for eth0 but gives "Version: Unknown", although `ethtool -i` gives 1.0, as C and D show. It also calls
  veth "Built-in", but veth is a loaded module. Item 5 (driver version) is not met. Rejected by the per-interface check that a concrete version is given.
- C: correct. eth0 → veth, version 1.0. It says honestly that the module file and parameters are unavailable, and gives
  dependencies none. lo has no driver. It gives a hardware description for both. Accepted.
- D: wrong (fabricated content). The header says `Kernel: 6.8.0-1046-nvidia`, as `uname` gives, but the body says
  "Kernel: 6.1.0-44-cloud-amd64" twice. It also gives veth size 36864, where C's lsmod on the same 6.8.0-1046-nvidia
  kernel shows 45056. It adds a "gve/gvnic" device that is absent from that kernel's module list. At least one of these values
  was not gathered from the system. Rejected by the kernel-release consistency check.
- E: wrong. It is a template with "[placeholder]" for dependencies, version and parameters, and it claims eth0 has no driver. Rejected by
  the placeholder check.

## Trace: requirement → check
| Task requirement | Check |
|---|---|
| "saved to a file called network_driver_report.txt in the working directory" | file /app/network_driver_report.txt exists and is not empty |
| "for each network interface: 1. interface name and hardware description" | loopback `lo` is reported. The task says the system has loopback, and every Linux has it. The report covers a hardware description (hardware/description/MAC/link/type) |
| "2. currently loaded kernel driver name" | report covers "driver" |
| "3. full filesystem path", "4. dependencies", "6. parameters" | covered globally. For every interface section that names a driver, the section must state path, dependencies and parameters (a value or N/A) |
| "5. Driver version information" | for every interface section that names a driver, a version line with a concrete value (contains a digit) |
| "gather this information programmatically" (not fabricated) | no placeholder/TODO text. The kernel releases stated in the report must all be the same (one machine) |
| edge cases (aliases, shared drivers, no-driver interfaces) | not checkable for a specific machine, because the solver's machine is unknown to the grader. Interfaces with no driver need no version |

## Trace: check → requirement
- Original `ip link` cross-check against the grading container: removed. It is not the solver's machine (task-format note: only the
  report file is carried over).
- Original "version" soft check via modinfo in the grading container: removed for the same reason. Replaced by the per-section version check.
- All new checks are listed above with their task sentences.

## Contested / unverified
No expected numeric value is compared. Driver names and versions depend on the solver's machine, which the grader
cannot see, so no value check was added. The version check needs only *some* concrete value, not a specific one.
