"""W.4b.2 Form B' prototype — cuIPC activation channel + control signals.

Per W4B_FORM_B_DESIGN_MEMO.md §5 item 2. Tensors move via cuIPC (cipher_kv_bridge
VMM arenas, cuMemExport POSIX-FD), never over the socket; the Unix socket carries
only cuIPC handle fds (SCM_RIGHTS via socket.send_fds) at setup + per-token
ready/done JSON signals. Mirrors the verified Track-2 SC3 producer/consumer
(phase_c/sc3_*.py) fd-passing idiom.

A cuIPC consumer mapping is PROT_READ (cipher_rt_kv_alloc.c cuMemSetAccess), so:
  producer creates+writes its own RW arena and exports the fd;
  consumer imports + views read-only and copies out.
Each side writes ONLY to arenas it created. Buffers are persistent: the producer
overwrites its arena each token and signals; the consumer re-reads the same view.
"""
import os
import sys
import json
import struct
import socket

sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch                       # noqa: E402
import cipher_kv_bridge as kvb     # noqa: E402

_DT = {torch.float16: "float16", torch.bfloat16: "bfloat16",
       torch.float32: "float32"}
_ESIZE = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}


def bridge_init(nbytes=64 * 1024 * 1024):
    kvb.init(nbytes)


# --- length-prefixed JSON control signals (reused from batch_ipc) ----------
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


# --- robust (manifest + fd) transfer ---------------------------------------
# socket.recv_fds does ONE recvmsg, so a large manifest (Mistral's 291-tensor
# weight manifest is ~37 KB) is truncated past the socket's atomic delivery.
# Send the manifest length-prefixed (reliable _recvn loop) FIRST, then the fd
# with a 1-byte SCM_RIGHTS payload. Used for the Step-1 weight arena.
def send_arena_fd(sock, manifest, fd):
    send_msg(sock, manifest)
    socket.send_fds(sock, [b"F"], [fd])


def recv_arena_fd(sock):
    manifest = recv_msg(sock)
    _msg, fds, _, _ = socket.recv_fds(sock, 8, 1)
    return manifest, fds[0]


# --- cuIPC producer-side buffer (RW; this process owns it) -----------------
class IpcOut:
    """A persistent device buffer this process writes and exports over cuIPC.
    `shape` is the maximum [.. , hidden/vocab] it will ever hold."""

    def __init__(self, shape, dtype, tenant_id):
        self.dtype = dtype
        self.shape = list(shape)
        esize = _ESIZE[dtype]
        nbytes = 1
        for s in shape:
            nbytes *= s
        nbytes *= esize
        # pad the arena so cuMem granularity rounding never truncates the view
        self.arena = kvb.weight_arena_create(int(nbytes) + (2 << 20), tenant_id)
        self.tensor = self.arena.alloc(self.shape, esize, _DT[dtype])
        self.base = int(self.arena.base)
        self.size = int(self.arena.size)

    def write(self, src, sync=True):
        """Copy `src` into the exported buffer. With sync=True, flush so the peer
        sees it. In the hot loop pass sync=False and do ONE torch.cuda.synchronize()
        after all writes (W.4b.4 rendezvous amortization — one sync/side/step
        instead of one per buffer)."""
        self.tensor[: src.shape[0]].copy_(src.reshape(self.tensor[: src.shape[0]].shape))
        if sync:
            torch.cuda.synchronize()

    def fd(self):
        return self.arena.export_fd()

    def manifest(self):
        return {"base": self.base, "size": self.size,
                "shape": self.shape, "dtype": _DT[self.dtype],
                "itemsize": _ESIZE[self.dtype]}


# --- cuIPC consumer-side mapping (PROT_READ) -------------------------------
class IpcIn:
    """A read-only view of a peer's exported cuIPC buffer."""

    def __init__(self, fd, manifest):
        self.arena = kvb.weight_arena_import(fd, 0, manifest["size"])
        self.view = self.arena.view(manifest["shape"], manifest["itemsize"],
                                    manifest["dtype"], 0)

    def read(self, shape=None):
        """Return a fresh device copy of the current contents (forces a read)."""
        v = self.view if shape is None else self.view.reshape(shape)
        return v.clone()


# --- fd + manifest exchange over the control socket (SCM_RIGHTS) -----------
def send_buffer(sock, ipc_out, extra=None):
    """Send one cuIPC buffer's manifest (+optional extra dict) and its fd."""
    m = ipc_out.manifest()
    if extra:
        m.update(extra)
    fd = ipc_out.fd()
    socket.send_fds(sock, [json.dumps(m).encode()], [fd])
    os.close(fd)


def recv_buffer(sock):
    """Receive a manifest + fd; return (IpcIn, manifest). Closes the fd."""
    msg, fds, _, _ = socket.recv_fds(sock, 1 << 16, 1)
    manifest = json.loads(msg.decode())
    ipc_in = IpcIn(fds[0], manifest)
    os.close(fds[0])
    return ipc_in, manifest
