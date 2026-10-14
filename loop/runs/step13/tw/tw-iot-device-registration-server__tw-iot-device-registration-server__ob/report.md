# Grader report: tw-iot-device-registration-server

Rounds: 1 · cost $1.48 · candidates judged correct: A, D · judged wrong: B, C, E

| basis | checks |
|---|---:|
| task-text | 8 |
| recomputed | 0 |
| data-settled | 0 |
| standard-model | 0 |
| author-answer | 0 |
| task-example | 0 |
| format | 1 |
| contested | 1 |
| unverified | 0 |

## Needs a human

- **contested:** key names inside details: the verifier expects mac_address, device_id (exactly these keys), the candidates give mac_address, device_id; task says only 'The mac address and device id of the device connected', no key names
- **contested:** MAC suffix derived from device id: the verifier expects DE:AD:BE:EF:00:01 for device-0001, DE:AD:BE:EF:00:02 for device-0002, the candidates give same; task says '** will be dyanmic based on device id' without specifying the mapping
- **contested:** report contains exactly the two good devices with details equal to answers.json (mac_address DE:AD:BE:EF:00:01/00:02, device_id device-0001/0002) (key names and MAC suffix mapping not fixed by the task; original check kept unchanged)

## Every check

| check | requirement | basis |
|---|---|---|
| reward.txt is exactly 1 or 0 | grader contract: write exactly 1 or 0 | format |
| registered_devices.json exists, is a JSON list of objects each with exactly one top-level key 'details' (an object) | The final report should be a list of objects store at file "registered_devices.json". Each object in the list must contain 1 top-level keys | task-text |
| report contains exactly the two good devices with details equal to answers.json (mac_address DE:AD:BE:EF:00:01/00:02, device_id device-0001/0002) | Good Device 1: device-0001, Good Device 2: device-0002, Faulty Device: device-0003; details: The mac address and device id | contested |
| server.py starts and opens a listening socket; iot_device.py run for device-0001 and device-0002 get registered by it | server.py: A server that can handle multiple device registrations; iot_device.py: A device simulator that will successfully register | task-text |
| an idle client connection is held open until after the deadline while devices register; devices must still be registered | This server nust have abiltiy enough to handle multiple clients at once / concurrent server.py | task-text |
| faulty device (iot_device_faulty.py device-0003) not in the report | iot_device_faulty.py: A device simulator that will always fail to regsiter | task-text |
| server still listening after faulty registration and accepts a connection at ~50 s | terminating that specific connection without crashing | task-text |
| server output or log files contain 'error' | It must throw an error while faulty device registration by logging an error | task-text |
| server no longer listening ~68 s after start; report saved | must have a master timeout of 60 seconds. After 60 seconds, it should stop listening for new devices and save whatever partial results it has gathered | task-text |
| server run with no devices writes [] | If no devices were registered successfully, it should produce an empty list [] | task-text |
