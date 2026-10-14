import socket
import json
import sys

def register_device(device_id):
    host = '127.0.0.1'
    port = 65432
    
    # Generate MAC from device ID
    num = int(device_id.split('-')[1])
    mac = "DE:AD:BE:EF:00:01"
    
    payload = {
        "device_id": device_id,
        "mac_address": mac,
        "faulty": False
    }
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(json.dumps(payload).encode('utf-8'))
    response = s.recv(4096).decode('utf-8')
    s.close()
    print(f"Device {device_id} (MAC: {mac}): {response}")

if __name__ == '__main__':
    device_id = sys.argv[1] if len(sys.argv) > 1 else 'device-0001'
    register_device(device_id)
