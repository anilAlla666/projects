# Grader report: tw-935

Rounds: 1 · cost $1.44 · candidates judged correct: none · judged wrong: A, B, C, D, E

| basis | checks |
|---|---:|
| task-text | 4 |
| recomputed | 8 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 1 |
| contested | 1 |
| unverified | 0 |

## Needs a human

- **contested:** size_gb / total_capacity_gb unit and rounding: the verifier expects anything between GiB-1 and GB+1 per device; total = sum of listed sizes, the candidates give floor GiB (A, D), round GiB (B); task says GB without saying 1e9 or 2^30, or how to round
- **contested:** which CPU flags to list: the verifier expects subset of real flags including aes/avx/avx2/sse4_1 if present, the candidates give all flags (A, B) or a filtered list (D); task says 'e.g.' and its example lists only five
- **contested:** whether non-disk block devices (rom, partitions) are listed: the verifier expects all disks required; extra real devices allowed, the candidates give disks only; 'all block devices' with only HDD/SSD/NVMe types
- **contested:** size_gb within [bytes/2^30-1, bytes/1e9+1]; total = sum of size_gb within max(1, n) (GB vs GiB and rounding not fixed by the task; both accepted)

## Every check

| check | requirement | basis |
|---|---|---|
| script exists and is executable (test no longer chmods it) | Be executable and placed in /app/hardware_audit.sh | task-text |
| output is a JSON object with usb, cpu, storage, audit_timestamp | Output a valid JSON file to /tmp/hardware_audit.json; The JSON output structure should follow this format | format |
| non-zero root_hubs equal the live system's root hubs (sysfs usbN/version or lsusb 1d6b:000N) | Detect all USB root hubs and their versions; Count how many root hubs exist for each USB version | recomputed |
| max_version = highest version present; null/'unknown' if none | Report the highest USB version supported by the system | recomputed |
| has_usb3 boolean consistent with max version >= 3.0 | Determine if the system has USB 3.0 or higher support | recomputed |
| cpu.model equals lscpu Model name | Report CPU model name | recomputed |
| cpu.cores integer equal to physical core count (cores/socket x sockets or unique core ids) | Report number of CPU cores | recomputed |
| cpu.threads integer equal to logical CPU count (lscpu CPU(s)/nproc/getconf); threads >= cores | Report number of threads | recomputed |
| features: non-empty list of real CPU flags including aes/avx/avx2/sse4_1 where the CPU has them | Detect CPU feature flags (e.g., aes, avx, avx2, sse4_1, etc.) | recomputed |
| every lsblk disk listed, all names real, disk type from nvme name/ROTA | List all block devices with their type (HDD, SSD, NVMe) | recomputed |
| size_gb within [bytes/2^30-1, bytes/1e9+1]; total = sum of size_gb within max(1, n) | Report total storage capacity in GB | contested |
| type_summary ssd/hdd/nvme integers equal counts in devices | Create a summary of storage device types | task-text |
| audit_timestamp extended ISO 8601 and within 15 min of the run | Include an ISO 8601 formatted timestamp in the output | task-text |
| with a failing lsusb on PATH the script still writes valid JSON with all sections | Handle missing or unavailable information gracefully | task-text |
