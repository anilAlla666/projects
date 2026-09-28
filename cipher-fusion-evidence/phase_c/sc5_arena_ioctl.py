#!/usr/bin/env python3
"""Track 2 SC5 — /dev/cipher weight-arena ioctl helper (pure ctypes).

The SC5 kmod arena registry exposes 4 additive ioctls (NRs 21-24, magic 'C')
on /dev/cipher. SC5's anchor decision is KMOD-ONLY rotation: the producer and
consumer issue these ioctls DIRECTLY from Python — cipher_kv_bridge and
libcipher_rt are unchanged. This module is that direct path.

  REGISTER (21) — producer hands the kmod its exported VMM POSIX fd + an
                  opaque metadata blob; the kmod fget's the fd, returns an id.
  IMPORT   (22) — consumer obtains a dup'd fd + the blob by arena id, whether
                  or not the producer is still alive.
  LEAVE    (23) — explicit participant departure (clean-exit fast path).
  QUERY    (24) — read-only operator snapshot of the registry.

The structs mirror `cipher_ioctl.h` exactly (natural alignment, no __packed).
"""
import ctypes
import os
import struct

DEV = "/dev/cipher"

# ---- _IOC encoding (asm-generic/ioctl.h) ---------------------------------
_MAGIC = ord('C')
_NONE, _WRITE, _READ = 0, 1, 2


def _IOC(direction, nr, size):
    return (direction << 30) | (size << 16) | (_MAGIC << 8) | nr


# ---- struct mirrors of cipher_ioctl.h ------------------------------------
class ArenaRegister(ctypes.Structure):
    _fields_ = [("fd", ctypes.c_int32),
                ("arena_id_out", ctypes.c_uint32),
                ("base", ctypes.c_uint64),
                ("size", ctypes.c_uint64),
                ("blob_ptr", ctypes.c_uint64),
                ("blob_len", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 5)]


class ArenaImport(ctypes.Structure):
    _fields_ = [("arena_id", ctypes.c_uint32),
                ("fd_out", ctypes.c_int32),
                ("base", ctypes.c_uint64),
                ("size", ctypes.c_uint64),
                ("blob_ptr", ctypes.c_uint64),
                ("blob_cap", ctypes.c_uint32),
                ("blob_len", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 4)]


class _ArenaQEntry(ctypes.Structure):
    _fields_ = [("arena_id", ctypes.c_uint32),
                ("producer_pid", ctypes.c_uint32),
                ("n_consumers", ctypes.c_uint32),
                ("_pad", ctypes.c_uint32),
                ("size", ctypes.c_uint64)]


CIPHER_WA_MAX_ARENAS = 100  # W6 G1+G2 cap bump 16->100; kmod 0.5.0+


class ArenaQuery(ctypes.Structure):
    _fields_ = [("n_arenas", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32),
                ("arenas", _ArenaQEntry * CIPHER_WA_MAX_ARENAS)]


CIPHER_WA_BLOB_MAX = 65536

CIPHER_ARENA_REGISTER = _IOC(_READ | _WRITE, 21, ctypes.sizeof(ArenaRegister))
CIPHER_ARENA_IMPORT   = _IOC(_READ | _WRITE, 22, ctypes.sizeof(ArenaImport))
CIPHER_ARENA_LEAVE    = _IOC(_WRITE, 23, 4)
CIPHER_ARENA_QUERY    = _IOC(_READ, 24, ctypes.sizeof(ArenaQuery))

# struct-size sanity vs. the kernel ABI (cipher_ioctl.h)
assert ctypes.sizeof(ArenaRegister) == 56, ctypes.sizeof(ArenaRegister)
assert ctypes.sizeof(ArenaImport) == 56, ctypes.sizeof(ArenaImport)
assert ctypes.sizeof(ArenaQuery) == 2408, ctypes.sizeof(ArenaQuery)


def _ioctl(op, arg):
    """Issue an ioctl on a fresh /dev/cipher fd. fcntl.ioctl mutates `arg`
    (a ctypes Structure) in place. The /dev/cipher fd used for the call is
    irrelevant to IMPORT's fd_install — that installs into THIS process."""
    import fcntl
    fd = os.open(DEV, os.O_RDWR)
    try:
        fcntl.ioctl(fd, op, arg, True)
    finally:
        os.close(fd)


def register(fd, base, size, blob):
    """REGISTER an exported VMM fd + opaque blob; return the arena id."""
    if len(blob) > CIPHER_WA_BLOB_MAX:
        raise ValueError("blob %d > CIPHER_WA_BLOB_MAX %d"
                         % (len(blob), CIPHER_WA_BLOB_MAX))
    blob_buf = ctypes.create_string_buffer(bytes(blob), max(len(blob), 1))
    s = ArenaRegister()
    s.fd = fd
    s.base = base
    s.size = size
    s.blob_ptr = ctypes.addressof(blob_buf)
    s.blob_len = len(blob)
    _ioctl(CIPHER_ARENA_REGISTER, s)            # blob_buf alive across call
    return int(s.arena_id_out)


def import_arena(arena_id, blob_cap=CIPHER_WA_BLOB_MAX):
    """IMPORT by arena id; return (fd_out, base, size, blob_bytes).

    fd_out is a fresh fd installed in THIS process — it survives the helper's
    own /dev/cipher fd being closed."""
    blob_buf = ctypes.create_string_buffer(blob_cap)
    s = ArenaImport()
    s.arena_id = arena_id
    s.blob_ptr = ctypes.addressof(blob_buf)
    s.blob_cap = blob_cap
    _ioctl(CIPHER_ARENA_IMPORT, s)
    return int(s.fd_out), int(s.base), int(s.size), blob_buf.raw[:s.blob_len]


def leave(arena_id):
    """LEAVE — explicit participant departure."""
    _ioctl(CIPHER_ARENA_LEAVE, bytearray(struct.pack("=I", arena_id)))


def query():
    """QUERY — return a list of {arena_id, producer_pid, n_consumers, size}."""
    s = ArenaQuery()
    _ioctl(CIPHER_ARENA_QUERY, s)
    return [{"arena_id": int(s.arenas[i].arena_id),
             "producer_pid": int(s.arenas[i].producer_pid),
             "n_consumers": int(s.arenas[i].n_consumers),
             "size": int(s.arenas[i].size)}
            for i in range(s.n_arenas)]


if __name__ == "__main__":
    # bare invocation: dump the registry (operator use)
    import json
    print(json.dumps(query(), indent=2))
