# Judgement

## What was wrong with the original verifier
* It only checked **types and shapes**. It never compared a value with the machine it ran on. That let hard-coded
  output through: C and E print a made-up "AMD EPYC 9B45 / 48 cores / nvme0n1 2000 GB" with no USB. A has USB
  hard-coded to "none". Even the task's own example JSON passes.
* `test.sh` ran `chmod +x /app/hardware_audit.sh` before testing. That hid the requirement "Be executable".
* `test.sh` needed network and uv (apt-get, curl astral.sh, uv add pytest). This machine has no python3 at all. The
  new verifier uses only bash, coreutils, util-linux, usbutils and jq, which are all in the task Dockerfile.
* Nothing checked graceful handling or that the timestamp was current.

## New verifier (tests/test.sh + tests/checks.sh)
test.sh first checks that the script is executable, without chmod. It then runs the script once with a failing
`lsusb` shim first on PATH (graceful-handling run) and once normally. checks.sh compares the normal output with what
the grading machine reports **at grading time**: sysfs, lsusb, lscpu, /proc/cpuinfo, lsblk, nproc and date.

## Candidates (run on this machine; each rejection traced to its check)
| cand | result | check that rejects | decision |
|---|---|---|---|
| A | reject | usb: root_hubs `{}` but the system has usb1 (2.00) and usb2 (3.00); lsusb shows 1d6b:0002 and 1d6b:0003 | attempt wrong: USB is hard-coded to zeros/false, so there is no detection at all. |
| B | reject | usb: root_hubs `{}` | attempt wrong: it matches the lsusb text `N.N root hub`. The Ubuntu 24.04 usbutils in the task image prints "xHCI Host Controller", so it never finds the two root hubs that exist. |
| C | reject | usb, cpu.model ("AMD EPYC 9B45" vs lscpu "AMD EPYC-Rome Processor"), storage (nvme0n1 does not exist) | attempt wrong: the whole JSON is hard-coded. |
| D | reject | usb: root_hubs `{"2.0":1,"3.0":2}` but the system has one 3.0 root hub | attempt wrong: it adds a "3.0" hub from a grep on lsusb ("xhci") and then counts usb2 from sysfs again. The task says "Count how many root hubs exist for each USB version". |
| E | reject | same as C | attempt wrong: hard-coded. |
| empty | reject | script missing | correct. |

None of the five is correct. To show that a correct result is accepted, I wrote my own solution
(/verifier_out/reference_solution) and a variant built on the other readings: GB=1e9 with decimals, `rom` listed,
only a subset of features, a `+00:00` timestamp. Both get reward 1.
Caveat: the verdicts above come from this machine, which has two root hubs. If the grading container shows no USB
root hubs, the USB check expects null/"unknown"/false, and A and B would pass that check. The verifier always judges
against the live system.

## Trace: requirement -> check
| requirement (task text) | check |
|---|---|
| "Be executable and placed in /app/hardware_audit.sh" | file exists and is executable (no chmod by the test) |
| "Output a valid JSON file to /tmp/hardware_audit.json" + the format block | JSON object with usb/cpu/storage/audit_timestamp |
| "Detect all USB root hubs and their versions" / "Count how many root hubs exist for each USB version" | non-zero root_hubs entries must equal the system map: sysfs usbN/version, or lsusb 1d6b:000N (either one is accepted). Keys must be one of 1.0..3.2 and values non-negative integers. Zero entries are optional. |
| "Report the highest USB version" | max_version = highest key in that map; with no hubs, null or "unknown" |
| "Determine if the system has USB 3.0 or higher" | has_usb3 boolean, equal to max >= 3.0 |
| "Report CPU model name" | equals lscpu "Model name" (or /proc/cpuinfo "model name"), whitespace normalised |
| "number of CPU cores" | integer in {Core(s) per socket x Socket(s), unique (core, socket) pairs from lscpu -p} |
| "number of threads" | integer in {lscpu CPU(s), lscpu -p rows, nproc, nproc --all, getconf}; threads >= cores |
| "Detect CPU feature flags (e.g., aes, avx, avx2, sse4_1, etc.)" | list of strings; every entry is a real flag of this CPU; non-empty; contains whichever of aes/avx/avx2/sse4_1 the CPU has |
| "List all block devices with their type (HDD, SSD, NVMe)" | every top-level lsblk TYPE=disk device (not loop/ram/zram) is listed. Every listed name exists in lsblk. A disk's type (case-insensitive) is nvme for an nvme name or transport, otherwise hdd if ROTA=1 and ssd if ROTA=0 |
| "total storage capacity in GB" | each size_gb lies between bytes/2^30-1 and bytes/1e9+1; total = sum of the listed size_gb, within max(1, n devices) |
| "summary of storage device types" | type_summary has integer ssd/hdd/nvme equal to the counts in devices |
| "Include an ISO 8601 formatted timestamp" (audit_timestamp) | extended ISO 8601 date-time, within 15 min of the grading run |
| "Handle missing or unavailable information gracefully" | with lsusb failing, the script still writes valid JSON with all sections and a boolean has_usb3 |

Every check above comes from the sentence beside it. Nothing beyond the task is checked (no exit code, no key order,
no extra keys forbidden).

## Settled vs contested values
* Root-hub counts, model, flags, disk set and ROTA type are **recomputed** at grading time with the tools the task
  names (lsusb, lscpu, lsblk) and the sysfs data they read. There is no choice of my own beyond reading them.
* GB (1e9 or 2^30) and rounding (floor/round/decimals) are **contested**: the task does not say. Both readings are
  accepted through the tolerance band. No data on this machine can settle a unit convention.
* Full flag list vs a subset of "interesting" flags is **contested** ("e.g." and the example shows a subset). Only
  the weaker property is checked: a subset of the real flags that includes the flags the task names.
* Whether non-disk devices (rom, partitions) are listed is **contested**. Extra entries are allowed if they exist on
  the system.
* "cores" means physical cores (task-text: listed separately from threads). Either of the two lscpu ways of counting
  them is accepted.
There was no fitted or model-based value, so no data-settled / standard-model reasoning was needed.

## Near misses (all rejected; all built from the reference solution with one change)
nm_usb_double_count (counts root hubs from lsusb and from sysfs), nm_threads_assume_smt, nm_rota_inverted,
nm_stale_timestamp, nm_not_executable, nm_fabricated_features (adds avx512f), nm_crash_without_lsusb (set -e +
$(lsusb)), nm_missing_disks, nm_total_doubled, shortcut_example_json (the task's example JSON echoed with a live
timestamp).
Known limit: a script that hard-codes exactly the grading machine's values plus a live timestamp cannot be told
apart by output alone.
