# Judgement

## The verifier as found
- `test.sh` installed curl and uv from the network and then ran pytest. The grading image has no network and no pytest,
  so it could never give reward 1 (evidence.md shows the run itself crashed, so there are no run outputs to use).
  I rewrote it in bash + jq + util-linux, all of which the grading Dockerfile installs.
- `test_outputs.py` only checked types and shapes. It accepted hard-coded output (C and E return another machine's CPU and
  an nvme disk that does not exist here), and it accepted any USB answer, even one that ignores USB.
- It did not delete an old `/tmp/hardware_audit.json` before running the script, so a script that writes nothing could
  pass on a stale file. Fixed: the file is removed first.

## This machine (a Docker container, same image as the grader)
- USB root hubs in sysfs: usb1 has version 2.00 and id 1d6b:0002; usb2 has version 3.00 and id 1d6b:0003. `lsusb` lists
  the same two (it reads sysfs; /dev/bus/usb is absent). So the counts are {"2.0":1,"3.0":1}, max is "3.0" and has_usb3 is true.
- CPU: "AMD EPYC-Rome Processor". CPU(s) is 8 and nproc is 8. Core(s) per socket × Socket(s) is 8×1. Flags are in /proc/cpuinfo.
- Disks: sda is a disk with ROTA 0 and 245761048576 bytes. sr0 is a rom device.

## Candidates (all wrong)
- A: rejected by the USB check. It writes max_version null, all counts 0 and has_usb3 false whatever the machine has,
  so it never detects USB. The task says "Detect all USB root hubs and their versions". The attempt is wrong, not the check.
- B: rejected by the USB check. It only counts lsusb lines with the text "N.N root hub". Here lsusb shows the kernel
  product string "xHCI Host Controller", so it finds 0 hubs while the machine has 2. The detection is wrong.
- C: rejected by the USB check. With that check skipped, it is also rejected on cpu.model. It writes fixed JSON
  (EPYC 9B45, 48 cores, nvme0n1 of 2000 GB): a shortcut.
- D: rejected by the USB check. It reports 3.0 = 2 although only one 3.0 root hub exists. Its lsusb "xhci" heuristic sets
  hubs_30=1, and then the sysfs loop counts usb2 again. That breaks "Count how many root hubs exist for each USB version".
- E: fixed JSON, the same shortcut as C (the CPU, disk and flag list come from another machine).
- With the USB check skipped, A, B and D pass every other check. So no other check rejected them.
- I wrote my own reference script (/tmp/scratch/ref) and it passes. Variants that must also pass do pass: sizes in GiB with
  floor, float sizes, a feature subset, root_hubs with zero entries, a timestamp with an offset, and a list that includes the rom device.

## Requirement → check
| Requirement (task) | Check |
|---|---|
| "placed in /app/hardware_audit.sh", "Be executable" | the script must exist. test.sh runs chmod +x (copying may drop the mode bit) and then runs it (timeout 200 s) |
| "Output a valid JSON file to /tmp/hardware_audit.json" | the stale file is removed, then the output must exist and be a JSON object |
| "Detect all USB root hubs and their versions", "Count how many root hubs exist for each USB version" | the root_hubs map, with zero entries dropped and keys read as numbers, must equal the machine's root hubs. Three readings are accepted: sysfs bcdUSB version, root-hub product id 1d6b:000N, or the 1d6b ids in lsusb output. Values must be non-negative integers |
| "Report the highest USB version" | max_version must be a numeric string equal to the highest hub version. With no hubs, null, "unknown" or "none" |
| "Determine if the system has USB 3.0 or higher" | has_usb3 must be a boolean, true exactly when some hub is ≥ 3.0 |
| "Report CPU model name" | must equal lscpu "Model name" or /proc/cpuinfo "model name". If neither exists, any string or null |
| "number of CPU cores" | a positive integer. Accepted readings: cores/socket × sockets, unique (CORE,SOCKET) pairs, unique CORE, cores/socket, cores/cluster, CPU(s)/threads-per-core |
| "number of threads" | an integer ≥ cores, equal to lscpu CPU(s), nproc, nproc --all, the /proc/cpuinfo processor count or the lscpu -p count |
| "Detect CPU feature flags (e.g., aes, avx, avx2, sse4_1, ...)" | a non-empty array of strings, every entry a real flag of this CPU, and it must include those of aes/avx/avx2/sse4_1/sse4_2 that the CPU has |
| "List all block devices with their type (HDD, SSD, NVMe)" | every lsblk disk must be listed (case-insensitive type): nvme if the name is nvme* or tran is nvme, otherwise hdd when ROTA=1 and ssd when ROTA=0. Extra entries must be real lsblk names |
| "Report total storage capacity in GB" | each disk's size_gb is a number in [bytes/2^30 − 1, bytes/10^9 + 1]. total_capacity_gb is a number within max(1, n) of the sum of the listed size_gb |
| "Create a summary of storage device types" | type_summary has numeric ssd, hdd and nvme keys equal to the counts in devices (the example format shows all three keys) |
| "Include an ISO 8601 formatted timestamp" | ISO 8601 date-time regex. `date -d` must parse it, and it must be within 25 h of the grading time, so a hard-coded value fails |
| "Handle missing ... gracefully (null or "unknown")" | covered by the null/"unknown" branches above (no USB hubs, no model) |

Every check above follows from the task sentence beside it. I removed nothing the task requires. The old shape checks are kept inside the new ones.

## Settled / contested
- USB counts, max and has_usb3: recomputed. The task names lsusb and asks for root hubs and their versions. Both the kernel's
  bcdUSB and the root-hub product id give the same values here. On hosts where they differ (for example, a 3.1 controller
  with product id 0003), either reading is accepted.
- Disk size units (GB or GiB) and rounding: the task does not choose. The check is a band that contains every reading
  (floor, round or float, in GB or GiB). No single value is imposed. Nothing to fit, so no data-settled analysis applies.
- Cores definition: the same approach, any of the listed definitions is accepted.
- Full flag list or a subset: the task's "e.g." does not decide this. The weaker check accepts both.
- disk → ssd/hdd from ROTA: lsblk documents ROTA as "rotational device", and the task's category names fix the mapping. There is no other source on the machine.

## Limits
- A script that hard-codes this exact machine's values (with a live timestamp) cannot be told apart in the same environment.
  If the grader runs on another host, the model, size and USB checks catch it.
- If the grading host exposes no USB, a script that ignores USB (like A) cannot be caught there.
