# Judgement: what was wrong with the old verifier and how it was repaired

## What the old verifier did
`test_outputs.py` imported a `grader` module that does not exist in `/tests/` (grader.py present
in the round-0 folder was only referenced; the module never ran on a path that worked in the
grading container layout), and its intended check was: `registered_devices.json` equals
`/tests/answers.json` **exactly** (list order + key order inside `details` matter).

## Every candidate the verifier accepted or rejected, and my decision

| attempt | old verdict | my verdict | why |
|---|---|---|---|
| A | ACCEPTED | **CORRECT** | Re-ran its exact scripts in a scratch copy of the machine: concurrent threaded server (port 5000, burst of 10 handled), rejects device-0003 with an error response and keeps serving, registers device-0001/0002, stops at the 60 s master timeout ("Master timeout reached. Final results saved"), writes `[{details:{device-0001, DE:AD:BE:EF:00:01}},{details:{device-0002, DE:AD:BE:EF:00:02}}]`. Meets every stated requirement. |
| B | ACCEPTED | **CORRECT** | Same end-to-end verification (port 65432, MASTER_TIMEOUT=60, threaded per-client handlers). JSON semantically identical to A. Accepted. |
| C | ACCEPTED (old) | **WRONG — shortcut** | `server.py` is a stub: `def run_server(): pass` plus code that hard-writes the two expected entries to `registered_devices.json` and exits. It never opens a port, never handles a client. My runtime checks reject it: "server.py crashed at startup" (exits instantly) / nothing listening. The old verifier's exact-JSON check is precisely why this shortcut sailed through. |
| D | ACCEPTED | **CORRECT** | Same end-to-end verification (port 65432, MASTER_TIMEOUT=60, threading). Accepted. |
| E | ACCEPTED (old) | **WRONG — shortcut** | `server.py` is *only* a JSON writer for the two expected entries; `iot_device.py` and `iot_device_faulty.py` are empty files (0 bytes). No server, no simulators. Rejected by the static (empty scripts) and runtime (never listens) checks. |
| empty attempt | (would fail on missing files) | **WRONG** | Rejected: missing scripts + missing report. |

## Near misses (each = candidate A's verified-correct set broken in exactly one place)

| name | broken requirement | rejected by |
|---|---|---|
| nm1_wrong_report_mac | MAC not "dyanmic based on device id" (00:11 for device-0001) | report MAC rule |
| nm2_timeout_not_60 | master timeout 999 s instead of 60 | server alive at t≈95 s + late registration rejected |
| nm3_faulty_accepted | server registers the faulty device | faulty id present in report (static + runtime) |
| nm4_empty_saves_null | saves `null` instead of `[]` when nothing registered | live empty-scenario run (60 s, zero clients) |
| nm5_flat_report | entries flat (no single top-level key "details") | report structure |
| nm6_good2_dropped | only device-0001 registered/saved | device-0002 missing (static + runtime) |

Extra anti-shortcut probe (not a stored near miss): A's **correct static report** + A's
simulators but the nm3 server (which accepts the faulty device) → REJECTED by the runtime
("faulty device registration was not reported as an error; device-0003 was registered in
runtime report"). A solver cannot pass by hand-writing the JSON while the real system misbehaves.

## Requirement → check traceability (task sentence → check)

| task sentence | check |
|---|---|
| "You must first create a concurrent server.py" / "nust have abiltiy enough to handle multiple clients at once" | `server.py` non-empty; server listens; burst of 10 simultaneous connections all answered (runtime) |
| "iot MAC_Address should contain DE:AD:BE:EF:**:** ** will be dyanmic based on device id" | runtime probes use MACs derived from the device id (DE:AD:BE:EF:00:01/00:02/00:03) and are accepted; report MACs must embed the device-number as two hex bytes (either byte order) under the `DE:AD:BE:EF:` prefix; prefix-only would accept `DE:AD:BE:EF:DE:AD`, so the dynamic part is enforced |
| "iot_device.py: A device simulator that will successfully register" | non-empty; `iot_device.py device-0001` → ok response; device appears in report; same for device-0002 |
| "iot_device_faulty.py: A device simulator that will always fail to regsiter" | non-empty; its run for device-0003 → error response; device-0003 absent from report |
| "It must throw an error while faulty device registration by logging an error and terminating that specific connection without crashing" | probe to 0003 gets an error response, **and the server stays up afterwards** (subsequent probe to 0002 succeeds); the "logging an error" half is not decidable (only the final JSON + 4 pid files are carried over — no logs survive to the grading container), verified unverified-by-nature: the observable consequence (error to that connection, server unharmed) is what is enforced |
| "It must have a master timeout of 60 seconds" | server still alive at t≈45 s; dead by t≈75 s; nothing registered after the deadline |
| "After 60 seconds, it should stop listening for new devices" | connection-refused probe at t≈70 s; late device-0002 registration rejected |
| "and save whatever partial results it has gathered" | report after main run == {device-0001, device-0002} |
| "If no devices were registered successfully, it should produce an empty list []" | live second run: 60 s, zero clients → saved report is `[]` |
| "The final report should be a list of objects store at file registered_devices.json. Each object in the list must contain 1 top-level keys: 1. 'details': The mac address and device id" | JSON is a list; each entry has exactly one top-level key `details`; `details` contains `device_id` and a MAC field |

## Value decisions (settled vs contested)

- **MAC values** `DE:AD:BE:EF:00:01` / `:00:02`: task says the MAC "should contain
  DE:AD:BE:EF:**:** ** will be dyanmic based on device id". The prefix is task-text; the
  "dynamic" part is settled from the machine's data: all three independently-working
  implementations (A: `%04x` split into two bytes; B: `n//256, n%256`; D: same; the faulty
  variants use 00:03 or an intentionally-off 00:04) encode the numeric suffix of
  `device-00NN` as two hex bytes. That is the only convention with any data support, and it
  is the standard MAC suffix interpretation for a 16-bit id. Both byte orders accepted
  (endian not specified). Contested reading "literal `**` → `DE:AD:BE:EF:00:00` for all":
  rejected — it contradicts "dynamic based on device id" (no variation) and every attempt.
- **Which devices land in the report**: task-text (Good 1 and 2 register; Faulty "always
  fail") → exactly {device-0001, device-0002}. Order not specified → sorted-by-id comparison.
- **60 s timeout**: task-text ("master timeout of 60 seconds", "After 60 seconds"). Checked
  with windows [45 s, 75 s] (alive inside, dead outside) so any sane implementation of "60 s"
  passes; 61 s or 999 s fail.
- **Empty list `[]`**: task-text. Checked live (the carried-over JSON only ever shows the
  successful run).
- **Exact JSON shape (key order, list order, extra keys)**: NOT checked — old verifier's
  exact-equality against `answers.json` was over-strict on order and would have rejected
  semantically correct reports; the task only fixes the one-top-level-key-`details` shape.

## Old checks removed / added
- REMOVED: exact match vs `/tests/answers.json` (wrong direction for C/E; over-strict on
  ordering for correct variants). `answers.json` deleted.
- REMOVED: the broken `grader`-module import (module absent from the grading image).
- ADDED: all runtime behavioural checks above, sandboxed in `mkdtemp` so the solver's JSON
  is backed up/restored (empty run must not destroy the graded report).
- KEPT (as format): report must be a JSON list of one-key `details` objects.

Unverified (no check, by nature): the server's stderr *log text* for the faulty device
("logging an error") — logs are not among the paths carried into the grading container. The
observable consequence (error response + surviving server) is enforced instead.
