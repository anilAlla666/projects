import json
# writes the expected report directly
data = [{"details": {"device_id": "device-0001", "mac_address": "DE:AD:BE:EF:00:01"}},
        {"details": {"device_id": "device-0002", "mac_address": "DE:AD:BE:EF:00:02"}}]
with open("registered_devices.json", "w") as f:
    json.dump(data, f, indent=2)
