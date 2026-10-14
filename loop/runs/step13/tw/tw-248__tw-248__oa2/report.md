# Grader report: tw-248

Rounds: 1 · cost $0.80 · candidates judged correct: B, C, D · judged wrong: A, E

| basis | checks |
|---|---:|
| task-text | 0 |
| recomputed | 0 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 6 |
| contested | 0 |
| unverified | 0 |

## Needs a human

- **contested:** driver version for veth: the verifier expects no value check (old and new), the candidates give 1.0 (C,D via ethtool), Unknown (B via modinfo); task does not name the source (ethtool driver version vs modinfo version field)

## Every check

| check | requirement | basis |
|---|---|---|
| report file /app/network_driver_report.txt exists and is non-empty | The report should be saved to a file called `network_driver_report.txt` in the working directory. | format |
| per-interface sections present and loopback 'lo' reported | The report must include for each network interface ... The system has various network interfaces including ... loopback | format |
| each interface section has a hardware/interface description | The interface name and hardware description | format |
| each interface names a driver or explicitly states none; at least one non-lo interface has a named driver | identifies all network device drivers currently in use; The currently loaded kernel driver name; Some interfaces may not have drivers loaded | format |
| interfaces with a driver report module path, dependencies, version, parameters (values may be N/A/unknown) | The driver module's full filesystem path; Module dependencies; Driver version information; Module parameters currently in use (if any) | format |
| no placeholder/TODO/TBD/FIXME markers | gather this information programmatically (no fabricated/partial report) | format |
