#!/usr/bin/env python3
import json
import os
import re
import socket
import sys

DEVICE_ID_RE = re.compile(r'^device-([0-9]+)$')


def faulty_mac_for_device(device_id: str) -> str:
    match = DEVICE_ID_RE.fullmatch(device_id)
    if not match:
        raise ValueError(f'invalid device_id format: {device_id}')
    suffix = (int(match.group(1)) + 1) & 0xFFFF
    return f'DE:AD:BE:EF:{suffix >> 8:02X}:{suffix & 0xFF:02X}'


def recv_line(sock):
    data = b''
    while not data.endswith(b'\n'):
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return data.decode('utf-8').strip()


def main():
    device_id = sys.argv[1] if len(sys.argv) > 1 else 'device-0003'
    host = os.getenv('IOT_SERVER_HOST', '127.0.0.1')
    port = int(os.getenv('IOT_SERVER_PORT', '5000'))

    payload = {
        'device_id': device_id,
        'mac_address': faulty_mac_for_device(device_id),
    }

    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall((json.dumps(payload) + '\n').encode('utf-8'))
        response = json.loads(recv_line(sock))

    print(json.dumps({'sent': payload, 'response': response}, indent=2))
    return 0 if response.get('status') == 'error' else 1


if __name__ == '__main__':
    raise SystemExit(main())
