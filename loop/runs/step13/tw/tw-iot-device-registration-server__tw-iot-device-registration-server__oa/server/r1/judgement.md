# Judgement

## Original verifier
Only compared `/workdir/registered_devices.json` against `answers.json` (2 entries, exact `details`), and wrote
`1.0`/`0.0` (task requires exactly `1`/`0`). The evidence run failed for infrastructure reasons (missing
grader config), so I re-ran everything locally (`/tests` -> `/verifier_out/tests`, candidate copied into `/workdir`).
Under the original verifier every candidate A–E would pass, because all five submitted the identical correct JSON;
C and E are fabrications (C's server.py just writes the JSON, device scripts are stubs; E's device scripts are empty).

## Decisions per candidate (new verifier)
- **A – correct (reward 1).** Threaded server, 1 s accept polling against a 60 s deadline, logs ERROR for the faulty
  device (wrong MAC), keeps running, writes [] when nothing registers, MAC derived from id (hex).
- **B – wrong (reward 0).** Rejected by "server still accepting connections 68 s after start". B's
  `server.settimeout(60)` is reset on every accept, so it is an *idle* timeout, not a master timeout: after the
  45 s probe connection it kept listening until ~105 s. Task: "must have a master timeout of 60 seconds. After 60
  seconds, it should stop listening for new devices". The attempt is wrong, not the check.
- **C – wrong.** server.py exits immediately after writing hard-coded JSON; device scripts do not connect. Fabricated.
- **D – correct (reward 1).**
- **E – wrong.** iot_device.py / iot_device_faulty.py empty; server.py writes hard-coded JSON.
- **empty attempt – wrong (0).**

## Requirement -> check
| Task requirement | Check |
|---|---|
| "you must generate scripts: server.py, iot_device.py, iot_device_faulty.py" | all three exist & non-empty; each is executed |
| "final report ... list of objects stored at registered_devices.json" | submitted file and file produced by re-running are JSON lists |
| "Each object in the list must contain 1 top-level keys: details" | keys == {"details"} (new; original did not check) |
| "details: The mac address and device id" ; Good devices 0001/0002, faulty 0003 | details equal answers.json (kept original exact check, keys `mac_address`/`device_id`, MACs DE:AD:BE:EF:00:01/02) — see contested |
| "create a concurrent server.py" / "handle multiple clients at once" | idle client connection held open while good1 & good2 run concurrently; both must finish within 15 s |
| "server should easily register good devices" | re-run report contains device-0001 and device-0002 |
| "iot_device_faulty.py ... will always fail to register" | re-run report has exactly the 2 good devices (no device-0003) |
| "throw an error ... by logging an error" | server output (stdout/stderr + *.log in /workdir) produced during the faulty run matches error/fail/reject/invalid/denied/refus (weak keyword, works for logging.error, print to stderr etc.) |
| "terminating that specific connection without crashing" | server process alive and still accepting TCP connections after the faulty run, and at 45 s |
| "master timeout of 60 seconds. After 60 seconds, it should stop listening ... save partial results" | at 68 s after launch the discovered port must refuse connections; report on disk by 76 s |
| "If no devices were registered successfully, it should produce an empty list []" | second run with no devices: report == [] |
| "MAC_Address should contain DE:AD:BE:EF:**:** ** dynamic based on device id" | iot_device.py run for device-0010 and device-0011 against a capture socket on the server's port: payload contains the id and a DE:AD:BE:EF:hh:hh MAC; the two MACs differ and are not 00:01/00:02 (holds for decimal or hex mapping) |
| reward exactly 1/0 | test.sh writes 0 first, check.py writes "1" or "0" |

Checks not required by the task: none kept. Not checked: pid files (task never mentions them). Device scripts
are invoked as `python3 <script> <device-id>` with `DEVICE_ID` also set in env; the task does not fix the CLI,
positional argv is the only interface used by all real attempts.

## Contested
- Exact `details` content (key names `mac_address`/`device_id`, MAC suffix mapping 0001 -> 00:01). The task does not
  name the keys nor the id->MAC mapping (decimal vs hex coincide for 1,2). Original check kept unchanged.
No data-fitting question arises in this task (no numeric model), so no data-settled readings.

## Near misses (all based on D, all rejected)
nm_timeout_90s, nm_sequential, nm_silent_faulty, nm_faulty_accepted, nm_crash_on_faulty, nm_extra_key,
nm_no_empty_list, nm_hardcoded_mac (shortcut: MAC table for the two known ids). C and E serve as fabricated shortcuts.
