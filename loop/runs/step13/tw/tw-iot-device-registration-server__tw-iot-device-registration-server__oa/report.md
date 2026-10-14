# Grader report: tw-iot-device-registration-server

Rounds: 1 · cost $1.33 · candidates judged correct: A, D · judged wrong: B, C, E

| basis | checks |
|---|---:|
| task-text | 6 |
| recomputed | 0 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 3 |
| contested | 1 |
| unverified | 0 |

## Needs a human

- **contested:** exact details object for the two good devices: the verifier expects {mac_address: DE:AD:BE:EF:00:01, device_id: device-0001}, {mac_address: DE:AD:BE:EF:00:02, device_id: device-0002}, the candidates give the same (all five); task does not name the JSON keys inside details nor the device-id-to-MAC mapping
- **contested:** details equal answers.json: device-0001/DE:AD:BE:EF:00:01 and device-0002/DE:AD:BE:EF:00:02, keys mac_address/device_id, exactly 2 entries (original check kept unchanged; applied to the submitted file and to the file produced by re-running the scripts)

## Every check

| check | requirement | basis |
|---|---|---|
| server.py, iot_device.py, iot_device_faulty.py exist and are non-empty | you must generate scripts: server.py ... iot_device.py ... iot_device_faulty.py | format |
| submitted and re-generated registered_devices.json are JSON lists of objects with exactly one key 'details' | The final report should be a list of objects store at file "registered_devices.json". Each object in the list must contain 1 top-level keys | format |
| details equal answers.json: device-0001/DE:AD:BE:EF:00:01 and device-0002/DE:AD:BE:EF:00:02, keys mac_address/device_id, exactly 2 entries | "details": The mac address and device id of the device connected; Good Device 1: device-0001, Good Device 2: device-0002, Faulty Device: device-0003 | contested |
| good devices complete within 15 s while another client connection is held open | This server nust have abiltiy enough to handle multiple clients at once. | task-text |
| faulty device run produces error log output from the server and is not registered | It must throw an error while faulty device registration by logging an error | task-text |
| server alive and accepting after faulty run and at 45 s | terminating that specific connection without crashing; must have a master timeout of 60 seconds | task-text |
| server port refuses connections 68 s after launch; report saved by 76 s | After 60 seconds, it should stop listening for new devices and save whatever partial results it has gathered. | task-text |
| run with no devices yields [] | If no devices were registered successfully, it should produce an empty list []. | task-text |
| iot_device.py for device-0010/0011 sends its id and distinct DE:AD:BE:EF:hh:hh MACs | iot MAC_Address should contain DE:AD:BE:EF:**:** ** will be dyanmic based on device id | task-text |
| reward file contains exactly 1 or 0 | grader contract | format |
