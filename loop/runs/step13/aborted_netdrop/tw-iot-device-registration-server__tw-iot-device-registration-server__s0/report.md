# Grader report: tw-iot-device-registration-server

Rounds: 1 · cost $0.00 · candidates judged correct: A, B, D · judged wrong: C, E

| basis | checks |
|---|---:|
| task-text | 9 |
| recomputed | 1 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 0 |
| contested | 0 |
| unverified | 0 |

## Needs a human: nothing flagged


## Every check

| check | requirement | basis |
|---|---|---|
| server.py, iot_device.py, iot_device_faulty.py exist and are non-empty | you must generate scripts: server.py / iot_device.py / iot_device_faulty.py | task-text |
| registered_devices.json: list of objects, each with exactly 1 top-level key 'details' holding device_id and a MAC field | The final report should be a list of objects ... Each object in the list must contain 1 top-level keys: 1. 'details': The mac address and device id of the device connected | task-text |
| report == {device-0001, device-0002} exactly (sorted compare, no device-0003, no extras) | Good Device 1/2 register; Faulty Device device-0003 'will always fail to regsiter' | task-text |
| each report MAC matches DE:AD:BE:EF:xx:yy with xx:yy the device number as two hex bytes (either endian) | iot MAC_Address should contain DE:AD:BE:EF:**:** ** will be dyanmic based on device id | recomputed |
| runtime: solver's server.py starts, listens on a TCP port, and answers a burst of 10 simultaneous connections (concurrency) | You must first create a concurrent server.py ... able to handle multiple clients at once | task-text |
| runtime: iot_device.py device-0001/0002 get success responses and appear in the server-saved report; iot_device_faulty.py device-0003 gets an error response and device-0003 is absent from the report | iot_device.py ... successfully register; iot_device_faulty.py ... always fail to regsiter | task-text |
| runtime: faulty probe's error response is received AND the server still serves good devices afterwards (no crash) | It must throw an error while faulty device registration by logging an error and terminating that specific connection without crashing | task-text |
| runtime: server still alive at t~45s, dead by t~75s; nothing registers after the deadline; connection refused shortly after | It must have a master timeout of 60 seconds. After 60 seconds, it should stop listening for new devices and save whatever partial results it has gathered | task-text |
| runtime (live second run, 60s, zero clients): saved report is exactly [] | If no devices were registered successfully, it should produce an empty list [] | task-text |
| report MACs / registered ids consistent with what the simulators send for the mandated device ids | For the test, you must use the following device configurations: device-0001, device-0002, device-0003 | task-text |
