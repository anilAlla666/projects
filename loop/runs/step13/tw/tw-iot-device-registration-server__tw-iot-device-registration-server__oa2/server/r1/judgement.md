# Judgement

## Original verifier
Only compared `/workdir/registered_devices.json` with `answers.json` (two good devices, MACs DE:AD:BE:EF:00:01/00:02).
It never ran the scripts, so it accepted C and E, which fabricate the JSON (C's server.py just writes the
expected list and its device scripts are stubs; E's device scripts are empty files). It also wrote `1.0`/`0.0` to
reward.txt, but the grader needs exactly `1` or `0`.

## Candidate decisions
- A: correct. Threaded TCP server, checks MAC against device id, logs an error with `logging.error` and closes
  the faulty connection, uses a 60 s deadline with a 1 s accept poll, saves `[]` at start and the results at the end.
  The new verifier accepts it (stopped at about 61 s, list correct, `[]` when only faulty devices run).
- B: correct. Thread per client, `faulty` flag rejected with an error printed to stderr, accept timeout of 60 s plus
  a 60 s loop. Note: if a client connects late, accept could wait up to 60 s more. The task scenario (devices start
  right after the server) never triggers that, and the verifier does not test it (see "not tested").
  The first draft of the new verifier rejected B because the test kept an idle socket open until the server
  exited, and B's non-daemon handler thread blocks on recv. That was my check's fault. The idle client is now
  closed once the good devices have finished. B is accepted.
- C: wrong (shortcut). server.py never listens and only writes hard-coded results. Rejected with "server exited immediately".
- D: correct. Same design as B, with a 1 s accept poll. Accepted.
- E: wrong (shortcut). server.py writes hard-coded JSON, and both device scripts are empty. Rejected.
- Empty attempt: rejected (no registered_devices.json).

## Trace: task requirement -> check
| Requirement (task text) | Check |
|---|---|
| "final report ... list of objects store at file registered_devices.json", "Each object ... 1 top-level keys: details: mac address and device id" | static: the submitted file is a list, each item has exactly the key `details` (an object), and the content equals answers.json (order ignored). Re-run: the file the server saves gets the same check. |
| Device configs device-0001/0002 good, device-0003 faulty | the devices are run as `python3 iot_device.py device-000N` (DEVICE_ID env is also set). The faulty one is `iot_device_faulty.py device-0003`. |
| "MAC_Address should contain DE:AD:BE:EF:**:** ** dynamic based on device id" | the saved list must have DE:AD:BE:EF:00:01 / 00:02. device-0002 is registered live, so a hard-coded MAC fails (nm5). |
| "server.py: handle multiple device registrations" / "handle multiple clients at once" | an idle TCP client stays connected to the server's listening port while both good devices run at the same time. They must finish within 8 s. A serial server fails (nm1). |
| "iot_device_faulty.py ... always fail to register" | device-0003 (and device-0004 in the second run) must not appear in the saved list (nm2) |
| "throw an error ... by logging an error and terminating that specific connection without crashing" | the server must still be alive after the faulty device runs, and the good devices must register afterwards (nm6). The word "error" must appear in the server's stdout/stderr or in files it writes (faulty-only run). |
| "master timeout of 60 seconds"; "After 60 seconds, it should stop listening ... save whatever partial results" | the server must exit by itself between 55 s and 85 s after start and save the file (nm3, nm8) |
| "If no devices were registered successfully, it should produce an empty list []" | faulty-only run: the saved file must equal `[]` (nm4) |
| scripts must exist | server.py, iot_device.py and iot_device_faulty.py must exist and be non-empty |

Every check maps to a row above. The removed checks were only the pytest/grader wrapper. The pid files are not
required by the task, so nothing checks them.

## Not tested (deliberately lenient)
- Clients that connect late (near 60 s). A strict reading would reject B, but the task's scenario is "run the
  device scripts immediately after running server".
- How the device scripts are called is not specified. The verifier passes the id as the first positional argument
  and in DEVICE_ID. All valid candidates read the positional argument.
- Timing tolerances (55–85 s exit window, 8 s concurrency bound) are my own margins around the stated 60 s. They
  are not expected values.

## Contested
- The MAC suffix derived from the id (00:01 for device-0001). The original verifier expects DE:AD:BE:EF:00:01 and
  00:02. The task says only "dynamic based on device id" and does not give the mapping. Readings: last four digits
  split in pairs, hex of the integer, or hash-based. The first two give the same value for 0001/0002. A hash reading
  would differ. There is no data on this machine to fit against, so this is contested. The original expected value
  is kept unchanged (all candidates agree with it, but agreement does not settle it).

## Near misses (all rejected by the repaired verifier)
nm1_serial_server (A without threads), nm2_faulty_registers (D, faulty script registers), nm3_short_timeout (A, 30 s),
nm4_no_empty_list (D, no file written when nothing is registered), nm5_hardcoded_mac (D, MAC fixed to 00:01),
nm6_crash_on_faulty (A, server exits on faulty), nm7_fabricated_shortcut (A's device scripts with a server that
only writes the JSON), nm8_no_master_timeout (D, 600 s), nm9_wrong_report (A with device-0003 added to the submitted JSON).

## Repaired verifier results (run here)
A=1, B=1, D=1. C=0, E=0, empty=0, all nine near misses=0. Running test.sh for real at /tests and /workdir on A gave `1`.
