import socket
import json
import sys

def register_faulty_device(device_id):
    host = '127.0.0.1'
    port = 65432
    
    num = int(device_id.split('-')[1])
    mac = f"DE:AD:BE:EF:{num // 256:02X}:{num % 256:02X}"
    
    payload = {
        "device_id": device_id,
        "mac_address": mac,
        "faulty": True
    }
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(json.dumps(payload).encode('utf-8'))
    response = s.recv(4096).decode('utf-8')
    s.close()
    print(f"Faulty device {device_id} (MAC: {mac}): {response}")

if __name__ == '__main__':
    device_id = sys.argv[1] if len(sys.argv) > 1 else 'device-0003'
    register_faulty_device(device_id)
