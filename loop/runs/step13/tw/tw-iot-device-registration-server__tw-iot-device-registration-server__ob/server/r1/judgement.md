# Judgement

## What was wrong with the original verifier
* It only compared the carried-over `/workdir/registered_devices.json` with `answers.json`. Hand-written JSON
  with no working server passed: candidates C (server.py just dumps a hard-coded list, empty device scripts are
  stubs) and E (empty `iot_device*.py`) were ACCEPTED. The task requires working scripts:
  "you must generate scripts ... server.py ... iot_device.py ... iot_device_faulty.py", concurrency, error logging,
  60 s master timeout, partial save, and "If no devices were registered successfully, it should produce an empty list []".
  None of these were checked.
* It wrote `1.0` / `0.0` to reward.txt; the grader needs exactly `1` or `0` (format).
* It did not enforce "Each object in the list must contain 1 top-level keys" (extra top-level keys passed).

## Repaired verifier (`tests/test.sh` -> `tests/verify.py`)
1. Static: carried-over report must be a list of exactly 2 objects, each with exactly one top-level key `details`,
   whose `details` equal those in `answers.json` (order-insensitive). Comparison with answers.json is unchanged
   from the original (contested values kept, see below); only the single-key rule was added (task-text).
2. Behavioural run in a scratch copy of /workdir (python3, no extra deps):
   start `python3 server.py`; detect its new listening socket (/proc/net/tcp*); open an idle TCP connection and hold
   it until t=62 s; immediately run `iot_device_faulty.py device-0003`, then `iot_device.py device-0001` and
   `iot_device.py device-0002` concurrently (an argparse `--...id...` flag is used if the script defines one;
   `DEVICE_ID` env also set). If that run fails, one retry runs each script once with no arguments.
   Requirements: server still running/listening after the faulty device (no crash) and at t≈50 s it accepts a
   connection; at t≈68 s it no longer listens (60 s master timeout); the report it wrote equals the expected two
   good devices (faulty device not registered); server output/log files contain "error" (faulty logged as error).
   Because the idle client is held past the deadline, a non-concurrent server never registers the devices.
3. Empty run: server alone, no devices; after ≤72 s the report must be exactly `[]`.
   Worst case runtime ≈ 225 s (< 300 s); test.sh wraps it in `timeout 290`, reward 0 on any failure.

## Candidates
| cand | old | new | decision |
|---|---|---|---|
| A | accept | accept | correct: threaded TCP server, deadline 60 s enforced by loop with 1 s accept timeout, logging.error on faulty, saves [] at start and results at end. |
| B | accept | reject | **wrong**. Check that rejects: "server still listening ~68 s after start". B's loop calls `accept()` with a 60 s socket timeout and only checks elapsed time between accepts, so any connection before 60 s extends listening up to 60 s more. Demonstrated on this machine: device-0001 at 1 s, a connection at 50 s, then `iot_device.py device-0002` at ~66 s was **registered** and written to the report; server ran until ~110 s. Violates "After 60 seconds, it should stop listening for new devices". |
| C | accept | reject | **wrong** (shortcut): server.py writes a hard-coded list and exits; device scripts do not connect. Rejected by "server.py exited immediately / never opened a listening socket". |
| D | accept | accept | correct: threaded server, 1 s accept timeout, 60 s loop, rejects faulty with printed "Error registering device", saves at end. |
| E | accept | reject | **wrong** (shortcut): server.py only dumps hard-coded JSON, device scripts are empty files. |
| empty attempt | reject | reject | no report. |

## Trace: requirement -> check
* "server.py: A server that can handle multiple device registrations" / "concurrent server.py" / "handle multiple clients at once" -> behavioural run with idle connection held open while devices register; report must contain both good devices.
* "iot_device.py: A device simulator that will successfully register" -> device-0001 and device-0002 run via iot_device.py must appear in the report written by the server run.
* "iot_device_faulty.py: ... will always fail to register" -> device-0003 must not appear (report must equal exactly the two good devices).
* "MAC_Address should contain DE:AD:BE:EF:**:** ** dynamic based on device id" -> answers.json comparison (DE:AD:BE:EF:00:01 / :00:02) — contested value, kept as in the original.
* "throw an error while faulty device registration by logging an error and terminating that specific connection without crashing" -> server must still be listening after the faulty device and at 50 s; output/log must contain "error".
* "master timeout of 60 seconds ... After 60 seconds, it should stop listening for new devices" -> listening at 50 s, not listening at 68 s (8 s slack).
* "save whatever partial results it has gathered" -> report written by the run contains the registered devices.
* "If no devices were registered successfully, it should produce an empty list []" -> empty run must produce `[]`.
* "list of objects store at file registered_devices.json. Each object ... 1 top-level keys: details" -> static and run-report checks: list, exactly key `details`, details is object.
* "details: The mac address and device id" -> details compared to answers.json (keys `mac_address`, `device_id`) — contested key names, kept.
* "use the following device configurations: device-0001, device-0002, device-0003" -> run uses those ids; expected report has the two good ones.
* "run the device scripts immediately after running server" -> devices launched as soon as the server listens.

## Trace: check -> requirement
All checks above map to a quoted sentence. Removed: nothing task-required. The `.pid` files are not required by
the task and are not checked. The reward format changed to exactly 1/0 (format).

## Contested values (kept exactly as the original verifier had them)
* Key names inside `details`: `mac_address` and `device_id`. The task says only "The mac address and device id";
  it does not name the keys. Original verifier expects these exact keys; all candidates use them. Kept, contested.
* MAC suffix mapping: `device-0001` -> `DE:AD:BE:EF:00:01`. The task says "** will be dyanmic based on device id"
  without fixing the mapping (decimal digits vs hex of the number; identical for 1 and 2 but the rule is not stated).
  No data on this machine can settle it (there is no dataset; it is a specification choice). Kept, contested.
* `details` must contain exactly those two keys (equality). Original behaviour, kept.

## Near misses (all copies of candidate A with one change; all rejected)
* nm_timeout_90: master timeout 90 s instead of 60 -> "still listening ~68 s".
* nm_faulty_registers: faulty device sends a valid MAC and gets registered -> report has 3 devices.
* nm_sequential: server handles clients inline (no threads) -> report empty under the held idle connection.
* nm_no_empty_list: no file written when nothing registered -> empty run fails.
* nm_extra_toplevel_key: objects carry an extra top-level "status" key -> static check fails.
* nm_no_error_logged: faulty rejection logged at INFO without "error" -> error-log check fails.
* nm_crash_on_faulty: server exits on the faulty registration -> not listening after devices.
* nm_shortcut_hardcoded_report (shortcut): server writes the hard-coded expected list at shutdown -> empty run gets non-empty list.

Results on this machine (run as the grader runs it, via run_case.sh): A=1, D=1; B, C, E, empty, and all 8 near misses = 0.
