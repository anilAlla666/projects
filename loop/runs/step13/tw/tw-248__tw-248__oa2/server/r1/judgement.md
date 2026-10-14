# Judgement

## Environment facts
- Solver environment (as seen in every candidate): `lo` plus `eth0` (a veth pair, `eth0@ifNNN`, link-netnsid 0);
  the `veth` module is loaded (lsmod in C, /proc/modules in D), `ethtool -i eth0` reports `driver: veth`, version 1.0.
  `/lib/modules` is absent, so `modinfo` fails; a module path cannot be resolved, which reports may state.
- The grading container is a different machine; the only thing carried over is the report. Its interfaces/drivers
  need not match the solver's, so the report cannot be cross-checked against the grader's live system values.
  Exact values (MACs, sizes, versions) are environment-specific -> not asserted (unverified/not settleable).
- This review machine has no python3; the grader guarantees python3 but test.sh is now written in perl
  (perl-base is Essential in ubuntu:24.04), so it can be run and checked here. The old test.sh depended on curl+uv
  downloads (no network) and only worked because uv happened to be present.

## Old verifier
Accepted all five candidates. Its checks: file exists; any `ip link` interface name of the *grading* machine appears
(always satisfied by "lo"); any of the words driver/module/version/path/loopback/virtual appear; a soft cross-check
against the grading machine's sysfs drivers. It accepted E, a report of literal "[placeholder]" values, and A,
which never names any driver. Too weak in the accept direction.

## Candidate decisions
- A: WRONG. Reports eth0 (veth pair) as "No driver loaded"; veth is the loaded kernel driver for it (ethtool -i
  gives driver veth; module loaded). Consequently no driver name, path, dependencies, version or parameters
  appear anywhere ("identifies all network device drivers currently in use" unmet). Rejected by check
  "at least one non-loopback interface reported with a loaded driver".
- B: CORRECT. eth0 -> veth, path "Built-in or unavailable via modinfo" (true: no /lib/modules), deps None,
  version "Unknown", params None; lo -> no driver. Version "Unknown" is defensible: veth has no MODULE_VERSION,
  so under the modinfo reading there is no version; ethtool's "1.0" is another source. Which source counts is
  contested, so no version value is asserted.
- C: CORRECT. Full sections for eth0 (veth, version 1.0, path explanation, deps, params) and lo.
- D: CORRECT. Same plus extra PCI gvnic/gve alias section.
- E: WRONG. Shortcut: all fields "[placeholder]", claims eth0 has no driver. Rejected by placeholder check and
  driver check.

## Trace: requirement -> check
1. "saved to a file called network_driver_report.txt in the working directory" -> file /app/network_driver_report.txt exists, non-empty.
2. "for each network interface" (system has loopback) -> per-interface sections found; `lo` reported.
3. "interface name and hardware description" -> each section has a description (hardware/description/type/MAC/link/loopback).
4. "currently loaded kernel driver name" -> each section names a driver or states explicitly that there is none;
   "identifies all network device drivers currently in use" + "multiple network interfaces ... physical ethernet"
   -> at least one non-lo interface reported with a driver name.
5-8. "full filesystem path", "dependencies", "version", "parameters (if any)" -> for every interface with a driver,
   its section mentions path, dependencies, version, parameters (values may be N/A/unknown/none since the module
   file is absent on that system).
9. "Some interfaces may not have drivers loaded" -> no-driver interfaces need only the explicit statement.
10. Shortcut/fabrication -> no "placeholder/TODO/TBD/FIXME" markers.
- "aliases differ from module filenames", "multiple interfaces same driver": cannot be checked without the solver's
  system (unverified); no check.
## Trace: check -> requirement
- Removed: match against grading machine's `ip link` names and sysfs drivers (different machine; not a requirement
  about the report's machine). Replaced by "lo reported" (every Linux system has lo; task names loopback).
- Removed: bare keyword-anywhere test (subsumed by per-section checks).

## Near misses (all rejected; all candidates B,C,D accepted; A,E and empty attempt rejected)
nm1 C minus eth0 driver version; nm2 D minus eth0 dependencies; nm3 B minus lo section; nm4 C minus eth0
parameters; nm5 B minus module path; nm6 B with a "[placeholder]" value; nm7 D with eth0 driver claimed "none"
(shortcut); nm8 empty file.

## Contested / not data-settleable
- Exact driver version (ethtool "1.0" vs modinfo none) - contested; no value check (old verifier had none either).
- No regression/fit applies here; values are environment observations not reproducible on the grader.
