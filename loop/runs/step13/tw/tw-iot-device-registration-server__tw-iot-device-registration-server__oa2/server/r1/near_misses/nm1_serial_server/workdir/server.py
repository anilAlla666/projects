#!/usr/bin/env python3
import json
import logging
import os
import re
import socket
import threading
import time
from pathlib import Path

HOST = os.getenv('IOT_SERVER_HOST', '0.0.0.0')
PORT = int(os.getenv('IOT_SERVER_PORT', '5000'))
MASTER_TIMEOUT = int(os.getenv('IOT_MASTER_TIMEOUT', '60'))
OUTPUT_FILE = Path('registered_devices.json')

DEVICE_ID_RE = re.compile(r'^device-([0-9]+)$')
MAC_RE = re.compile(r'^DE:AD:BE:EF:[0-9A-F]{2}:[0-9A-F]{2}$')

registered_devices = []
registered_ids = set()
lock = threading.RLock()


def expected_mac(device_id: str) -> str:
    match = DEVICE_ID_RE.fullmatch(device_id)
    if not match:
        raise ValueError(f'invalid device_id format: {device_id}')
    suffix = int(match.group(1)) & 0xFFFF
    return f'DE:AD:BE:EF:{suffix >> 8:02X}:{suffix & 0xFF:02X}'


def save_results():
    with lock:
        snapshot = sorted(
            registered_devices,
            key=lambda item: item['details']['device_id'],
        )
    OUTPUT_FILE.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')


def register_device(device_id: str, mac_address: str):
    normalized_mac = mac_address.upper()
    if not MAC_RE.fullmatch(normalized_mac):
        raise ValueError(f'invalid mac_address format: {mac_address}')

    expected = expected_mac(device_id)
    if normalized_mac != expected:
        raise ValueError(
            f'invalid mac_address for {device_id}: expected {expected}, got {normalized_mac}'
        )

    entry = {'details': {'device_id': device_id, 'mac_address': normalized_mac}}
    was_new = False
    with lock:
        if device_id not in registered_ids:
            registered_ids.add(device_id)
            registered_devices.append(entry)
            was_new = True

    if was_new:
        save_results()
        logging.info('Registered %s with %s', device_id, normalized_mac)
    else:
        logging.info('Duplicate registration ignored for %s', device_id)

    return entry


def recv_line(conn, limit=8192):
    data = b''
    while not data.endswith(b'\n'):
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
        if len(data) > limit:
            raise ValueError('payload too large')

    if not data:
        raise ValueError('empty payload')

    return data.decode('utf-8').strip()


def handle_client(conn, address):
    with conn:
        conn.settimeout(10)
        try:
            raw_message = recv_line(conn)
            payload = json.loads(raw_message)
            device_id = payload.get('device_id')
            mac_address = payload.get('mac_address')

            if not isinstance(device_id, str):
                raise ValueError('missing or invalid device_id')
            if not isinstance(mac_address, str):
                raise ValueError('missing or invalid mac_address')

            entry = register_device(device_id, mac_address)
            response = {'status': 'ok', 'registered': entry}
            conn.sendall((json.dumps(response) + '\n').encode('utf-8'))
        except Exception as exc:
            logging.error('Failed registration from %s: %s', address, exc)
            try:
                response = {'status': 'error', 'error': str(exc)}
                conn.sendall((json.dumps(response) + '\n').encode('utf-8'))
            except OSError:
                pass


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    save_results()
    deadline = time.time() + MASTER_TIMEOUT
    client_threads = []

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        server_socket.settimeout(1.0)
        logging.info(
            'IoT registration server listening on %s:%s for %s seconds',
            HOST,
            PORT,
            MASTER_TIMEOUT,
        )

        while time.time() < deadline:
            try:
                conn, address = server_socket.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                logging.error('Server socket error: %s', exc)
                break

            handle_client(conn, address)

    for thread in client_threads:
        thread.join(timeout=2)

    save_results()
    logging.info('Master timeout reached. Final results saved to %s', OUTPUT_FILE)


if __name__ == '__main__':
    main()
