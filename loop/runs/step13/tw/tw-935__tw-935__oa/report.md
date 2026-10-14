# Grader report: tw-935

Rounds: 1 · cost $1.15 · candidates judged correct: none · judged wrong: A, B, C, D, E

| basis | checks |
|---|---:|
| task-text | 0 |
| recomputed | 9 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 4 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** storage size_gb unit and rounding: the verifier expects any value in the GiB..GB band (+-1), the candidates give floor GiB (A,D), rounded GiB (B); the task says GB but does not choose GB or GiB, or a rounding rule
- **contested:** cpu.features: full flag list or a subset: the verifier expects real flags only, including the named examples the CPU has, the candidates give full list (A,B), subset (D); 'e.g.' does not say which flags must be reported

## Every check

| check | requirement | basis |
|---|---|---|
| script exists; run after chmod +x; stale /tmp/hardware_audit.json removed first | Be executable and placed in /app/hardware_audit.sh | format |
| output exists and is a JSON object | Output a valid JSON file to /tmp/hardware_audit.json | format |
| usb.root_hubs (nonzero entries) equals the machine's root hubs per version | Detect all USB root hubs and their versions; Count how many root hubs exist for each USB version | recomputed |
| usb.max_version is the highest hub version (null/unknown if none) | Report the highest USB version supported by the system | recomputed |
| usb.has_usb3 is a boolean, true iff a hub is >= 3.0 | Determine if the system has USB 3.0 or higher support | recomputed |
| cpu.model equals lscpu Model name or cpuinfo model name | Report CPU model name | recomputed |
| cpu.cores is a positive integer matching one accepted core-count definition | Report number of CPU cores | recomputed |
| cpu.threads is an integer >= cores and equals the logical CPU count (lscpu/nproc/cpuinfo) | Report number of threads | recomputed |
| cpu.features is non-empty, every entry a real flag, includes those of aes/avx/avx2/sse4_1/sse4_2 present | Detect CPU feature flags (e.g., aes, avx, avx2, sse4_1, etc.) | recomputed |
| every lsblk disk listed with type nvme/ssd/hdd per name/tran/ROTA; extra entries must be real devices | List all block devices with their type (HDD, SSD, NVMe) | recomputed |
| size_gb within [bytes/2^30-1, bytes/1e9+1]; total_capacity_gb within max(1,n) of the sum | Report total storage capacity in GB | recomputed |
| type_summary has ssd/hdd/nvme counts equal to the devices | Create a summary of storage device types (format example) | format |
| audit_timestamp is ISO 8601, parseable, within 25h of now | Include an ISO 8601 formatted timestamp in the output | format |
