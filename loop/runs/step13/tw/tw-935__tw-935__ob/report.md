# Grader report: tw-935

Rounds: 1 · cost $1.53 · candidates judged correct: none · judged wrong: A, B, C, D, E

| basis | checks |
|---|---:|
| task-text | 3 |
| recomputed | 7 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 5 |
| contested | 1 |
| unverified | 0 |

## Needs a human

- **contested:** size_gb unit and rounding: the verifier expects within 1 of decimal GB or GiB, the candidates give 228 (floor GiB), 229 (round GiB); task does not say GB vs GiB or rounding
- **contested:** inclusion of non-disk block devices (rom, loop): the verifier expects all disks required, extras allowed, the candidates give disks only; 'all block devices' vs types limited to HDD/SSD/NVMe
- **contested:** size_gb within 1 of bytes/1e9 or bytes/2^30 (weak property only)

## Every check

| check | requirement | basis |
|---|---|---|
| script exists and is executable (no chmod by verifier) | Be executable and placed in /app/hardware_audit.sh | format |
| script run fresh produces /tmp/hardware_audit.json, strict-valid JSON object | Output a valid JSON file to /tmp/hardware_audit.json | format |
| top-level keys usb,cpu,storage,audit_timestamp | The JSON output structure should follow this format | format |
| script contains a detection mechanism for USB, CPU and storage | The script must detect and report on the following hardware capabilities | format |
| root_hubs counts per version equal sysfs root hub bcdUSB counts; zero/missing keys allowed for absent versions | Detect all USB root hubs and their versions / Count how many root hubs exist for each USB version | recomputed |
| max_version equals highest root hub version; null/'unknown' when none | Report the highest USB version supported by the system | recomputed |
| has_usb3 boolean equals max>=3.0 | Determine if the system has USB 3.0 or higher support | recomputed |
| cpu.model equals lscpu Model name or /proc/cpuinfo model name | Report CPU model name | recomputed |
| cpu.cores is a physical core count (any standard derivation) | Report number of CPU cores (example i7-1065G7 cores 4, threads 8) | task-text |
| cpu.threads equals a logical CPU count (lscpu/proc/nproc) | Report number of threads | recomputed |
| features are real CPU flags; include aes/avx/avx2/sse4_1/sse4_2 when present | Detect CPU feature flags (e.g., aes, avx, avx2, sse4_1, etc.) | recomputed |
| every lsblk disk listed with type nvme/hdd/ssd by name/ROTA | List all block devices with their type (HDD, SSD, NVMe) | recomputed |
| size_gb within 1 of bytes/1e9 or bytes/2^30 | Report total storage capacity in GB | contested |
| total_capacity_gb equals sum of device size_gb | Report total storage capacity in GB | task-text |
| type_summary ssd/hdd/nvme counts match device list | Create a summary of storage device types | task-text |
| audit_timestamp ISO 8601 date-time within 15 min of run | Include an ISO 8601 formatted timestamp in the output | format |
