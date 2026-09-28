"""Phase B Session 1 — minimal length-prefixed JSON IPC over a Unix socket."""
import json
import struct


def _recvn(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def send_msg(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(b)) + b)


def recv_msg(sock):
    hdr = _recvn(sock, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack(">I", hdr)
    body = _recvn(sock, n)
    return json.loads(body) if body is not None else None
