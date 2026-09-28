"""Focused cuIPC round-trip smoke for the formb_ipc wrappers.

Validates the NEW bits over Track-2 SC3: IpcOut/IpcIn shape handling and the
per-token REUSE pattern (producer overwrites its arena, consumer re-reads the
same mapped view and sees fresh bytes). Two real processes (fork BEFORE any CUDA
init — CUDA contexts do not survive fork), fd over AF_UNIX. Run: python3 formb_smoke_ipc.py
"""
import os
import sys
import socket
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SOCK = "/tmp/formb_smoke_ipc.sock"
H = 2048
STEPS = 5


def consumer():
    import torch
    import formb_ipc as ipc
    ipc.bridge_init()
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for _ in range(100):
        try:
            c.connect(SOCK); break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.1)
    ein, _ = ipc.recv_buffer(c)
    ok = True
    for t in range(STEPS):
        ipc.recv_msg(c)                       # producer wrote step t
        got = ein.view[0].clone().float()     # re-read same mapped view
        exp = float(t + 1)
        good = bool(torch.allclose(got, torch.full_like(got, exp)))
        ok = ok and good
        print("  consumer step %d: first=%.1f expect=%.1f match=%s"
              % (t, float(got[0]), exp, good), file=sys.stderr, flush=True)
        ipc.send_msg(c, {"ack": t})
    c.close()
    sys.exit(0 if ok else 2)


def producer():
    import torch
    import formb_ipc as ipc
    ipc.bridge_init()
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK); srv.listen(1)
    conn, _ = srv.accept()
    out = ipc.IpcOut([1, H], torch.float16, tenant_id=900)
    ipc.send_buffer(conn, out)
    for t in range(STEPS):
        out.write(torch.full((1, H), float(t + 1), device="cuda",
                             dtype=torch.float16))
        ipc.send_msg(conn, {"step": t})
        ipc.recv_msg(conn)                    # consumer ack
    conn.close(); srv.close(); os.unlink(SOCK)


if __name__ == "__main__":
    # fork BEFORE importing torch / initializing CUDA in either branch
    child = os.fork()
    if child == 0:
        consumer()
    else:
        producer()
        _, status = os.waitpid(child, 0)
        rc = os.waitstatus_to_exitcode(status)
        print("SMOKE %s (consumer rc=%d)" % ("PASS" if rc == 0 else "FAIL", rc),
              file=sys.stderr, flush=True)
        sys.exit(rc)
