import socket
import threading
import json
import time
import sys

registered_devices = []
lock = threading.Lock()

def handle_client(conn, addr):
    try:
        data = conn.recv(4096).decode('utf-8')
        device_info = json.loads(data)
        device_id = device_info.get('device_id', '')
        mac_address = device_info.get('mac_address', '')
        
        if not mac_address.startswith('DE:AD:BE:EF:'):
            raise ValueError(f"Invalid MAC address format: {mac_address}")
        
        if device_info.get('faulty', False):
            raise ValueError(f"Faulty device rejected: {device_id}")
        
        with lock:
            registered_devices.append({
                "details": {
                    "mac_address": mac_address,
                    "device_id": device_id
                }
            })
        conn.sendall(b'{"status": "registered"}')
        print(f"Device {device_id} registered successfully with MAC {mac_address}")
    except Exception as e:
        print(f"Error registering device: {e}")
        try:
            conn.sendall(b'{"status": "error", "message": "' + str(e).encode() + b'"}')
        except:
            pass
    finally:
        conn.close()

def main():
    host = '127.0.0.1'
    port = 65432
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    server.settimeout(1.0)
    print(f"Server listening on {host}:{port}")
    
    start_time = time.time()
    master_timeout = 90
    
    try:
        while time.time() - start_time < master_timeout:
            try:
                conn, addr = server.accept()
                t = threading.Thread(target=handle_client, args=(conn, addr))
                t.daemon = True
                t.start()
            except socket.timeout:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        with open('registered_devices.json', 'w') as f:
            json.dump(registered_devices, f, indent=2)
        print(f"Saved {len(registered_devices)} devices to registered_devices.json")

if __name__ == '__main__':
    main()
