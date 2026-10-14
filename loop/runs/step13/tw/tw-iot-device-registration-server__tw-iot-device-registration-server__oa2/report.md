# Grader report: tw-iot-device-registration-server

Rounds: 1 · cost $1.06 · candidates judged correct: A, B, D · judged wrong: C, E

| basis | checks |
|---|---:|
| task-text | 5 |
| recomputed | 0 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 2 |
| contested | 2 |
| unverified | 0 |

## Needs a human

- **contested:** MAC suffix for device-0001/0002: the verifier expects DE:AD:BE:EF:00:01 / DE:AD:BE:EF:00:02, the candidates give same (all five); the task says only 'dynamic based on device id' and gives no mapping (digit split vs hex vs hash)
- **contested:** submitted registered_devices.json is a list of objects with only key 'details' and equals answers.json (order-insensitive) (original expected values kept; the MAC mapping is contested, the device ids are task-text)
- **contested:** server run live; file saved after the run equals the expected two good devices (same expected values as the original check, applied to the live output; catches fabricated results)

## Every check

| check | requirement | basis |
|---|---|---|
| submitted registered_devices.json is a list of objects with only key 'details' and equals answers.json (order-insensitive) | The final report should be a list of objects store at file registered_devices.json. Each object in the list must contain 1 top-level keys: details | contested |
| server.py, iot_device.py, iot_device_faulty.py exist and are non-empty | you must generate scripts: server.py ... iot_device.py ... iot_device_faulty.py | format |
| server run live; file saved after the run equals the expected two good devices | The server should easily register good devices; Good Device 1: device-0001, Good Device 2: device-0002 | contested |
| good devices finish within 8s while another TCP client is connected | This server nust have abiltiy enough to handle multiple clients at once | task-text |
| faulty device (device-0003/0004) not in the saved list | iot_device_faulty.py: A device simulator that will always fail to regsiter | task-text |
| server still alive after the faulty registration and registers good devices afterwards; 'error' appears in the server logs | It must throw an error while faulty device registration by logging an error and terminating that specific connection without crashing | task-text |
| server exits by itself 55-85s after start and saves the file | must have a master timeout of 60 seconds. After 60 seconds, it should stop listening for new devices and save whatever partial results | task-text |
| faulty-only run saves exactly [] | If no devices were registered successfully, it should produce an empty list [] | task-text |
| reward.txt holds exactly 1 or 0 | grader contract | format |
