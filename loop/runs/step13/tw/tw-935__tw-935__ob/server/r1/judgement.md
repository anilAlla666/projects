# Judgement

## Machine facts (this machine, Ubuntu 24.04, usbutils 017)
- USB root hubs: /sys/bus/usb/devices/usb1/version = 2.00 (1d6b:0002, 480M), usb2/version = 3.00 (1d6b:0003, 5000M);
  `lsusb -t` shows both as Class=root_hub, `lsusb -v` bcdUSB 2.00 / 3.00. So the right answer here is
  max_version "3.0", root_hubs 2.0:1 and 3.0:1, has_usb3 true. Note that usbutils 017 prints
  "Linux 6.8.0-... xhci-hcd xHCI Host Controller" and not "Linux Foundation 3.0 root hub".
- CPU: lscpu Model name "AMD EPYC-Rome Processor", CPU(s) 8, 1 thread per core, 8 cores per socket, 1 socket, 86 flags.
- Block devices: sda disk 245761048576 B rota=0 (=> ssd), sr0 rom, loop0-7 size 0.

## Original verifier
It checked only types and structure. It accepted all five candidates, including two that are entirely hard-coded
for another machine (C, E: "AMD EPYC 9B45", 48 cores, nvme0n1 2000 GB). It ran `chmod +x` on the script before
checking "is executable", so that requirement was never really tested. It also needed network/uv to install pytest.

## Candidate decisions (all rejected by the repaired verifier, none by the original)
- A: WRONG. The USB section is hard-coded (`max_version: null`, all zero counts, `has_usb3: false`). The script
  never calls lsusb or reads sysfs. On this machine it reports no USB, but there are a 2.0 and a 3.0 root hub.
  CPU and storage are fine. Rejected by the usb value check and by the source check (no USB detection mechanism).
- B: WRONG. It counts only `lsusb` lines that contain "root hub". The usbutils 017 shipped in this image
  (and in the grading image, also ubuntu:24.04) does not print that text, so it reports no hubs where there are two.
  The task says "The script should work in the current container environment". Rejected by the usb value check.
- C: WRONG. A fully hard-coded heredoc for another machine. Rejected by the CPU, storage, USB and source checks.
- D: WRONG. It adds a "3.0" hub whenever lsusb mentions xhci, then counts the sysfs root hubs again, which gives
  root_hubs 3.0:2 where the system has one ("Count how many root hubs exist for each USB version").
  Rejected by the root_hubs count check only.
- E: WRONG. Hard-coded for another machine. Its flag list has bogus entries ("pdp", "e1gb").
- Empty attempt: rejected (script missing).
My own reference script (/root/scratch) is accepted, along with variants that differ only in allowed ways: GiB
floor/round vs decimal GB sizes, extra sr0 entry, "/dev/sda", "SSD" uppercase, root_hubs with only nonzero keys or
keys "2.00", timestamps with +00:00 or no offset, and a feature list restricted to the named flags.

## Values: settled or contested
- USB root hub versions and counts are **recomputed**. The task asks to "Detect all USB root hubs and their versions".
  The version of a root hub is its bcdUSB (sysfs `usbN/version`, the same value `lsusb -v` prints). If sysfs has
  nothing, the fallback is lsusb's 1d6b:0001/2/3. This is recomputed in the grading container at test time.
  max_version = highest version. has_usb3 = (max >= 3.0), from "Determine if the system has USB 3.0 or higher".
  With no hubs, max_version must be null or a non-numeric string ("use null or "unknown"").
- CPU model: **recomputed**. Must equal lscpu "Model name" or /proc/cpuinfo "model name" (whitespace normalised).
- threads: **recomputed**. Any of lscpu CPU(s), /proc/cpuinfo processor count, nproc, nproc --all,
  _NPROCESSORS_ONLN. Several counts are accepted because the task does not say which one.
- cores: physical cores. **task-text**: the example "i7-1065G7 ... cores: 4, threads: 8" is a 4-core/8-thread
  part. Accepted values: cores/socket*sockets, cores/cluster*clusters, unique (physical id, core id), CPU(s)/threads-per-core.
- features: every entry must be a real flag of this CPU (**recomputed** from lscpu/proc). Every one of the
  flags the task names ("e.g., aes, avx, avx2, sse4_1") plus sse4_2 (in the example) that the CPU has must be present.
  Listing all flags is not required, because the example lists only 5.
- storage type: nvme by name/transport, otherwise lsblk ROTA (rota=1 hdd, rota=0 ssd). Basis: recomputed with lsblk,
  the tool the task names. ROTA is the only data the machine has for HDD vs SSD.
- size_gb: **contested** (GB vs GiB, floor vs round vs decimals). Example 256 does not decide it. Weak check only:
  within 1 of bytes/1e9 or bytes/2^30. (Candidates gave 228 floor GiB / 229 round GiB.) The original verifier
  had no value check, so none is kept beyond this weak property.
- Which block devices to list: **contested** for non-disk devices (rom, loop). Every lsblk TYPE=disk with size>0
  must be present. Extra entries must be real block devices with plausible sizes.
- total_capacity_gb: must equal the sum of the listed devices' size_gb, within rounding tolerance (task-text: "total").
- type_summary: keys ssd/hdd/nvme are required. Each count must equal the number of listed devices of that type.
- Timestamp: must be ISO 8601 date-time and within 15 min of the run (the script must produce it at audit time).
No data-fit readings apply here. All value checks are recomputed from the live system.

## Trace: requirement -> check
- "Be executable and placed in /app/hardware_audit.sh" -> exists + `-x` test (chmod removed from test.sh).
- "Output a valid JSON file to /tmp/hardware_audit.json" -> old file deleted, script run, jq object + python strict parse.
- USB bullets 1-4 -> usb check (root_hubs counts per version, max_version, has_usb3). Also a source check: the script
  must contain some USB detection mechanism (lsusb, /sys/bus/usb, ...). This catches hard-coded output on hosts without USB.
- CPU bullets -> cpu check (model, cores, threads, features) + source check.
- Storage bullets -> storage check (devices/type/size, total, type_summary) + source check.
- "Handle missing ... null or unknown" -> null/"unknown" accepted where the system has no data (USB absent, CPU fields absent).
- "ISO 8601 formatted timestamp" -> regex + date parse + freshness.
- JSON structure -> top-level keys usb/cpu/storage/audit_timestamp.
## Trace: check -> requirement
Every check above follows from a quoted requirement. Removed: the dependency on pytest/uv/network (not a requirement),
and the `chmod +x` (it hid a requirement). The original "threads >= cores" stays (consequence of physical cores <= logical CPUs).

## Limitations
If the grading container exposes no USB root hubs, USB values can only be checked as "none". A hard-coded USB
section is then caught only by the source check. Hard-coding this exact machine's values while still mentioning
the tool names cannot be caught without faking hardware (bwrap/mount namespaces could not be tested here).
