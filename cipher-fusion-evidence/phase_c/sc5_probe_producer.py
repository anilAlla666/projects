#!/usr/bin/env python3
"""Track 2 SC5 Phase A — VMM refcount probe PRODUCER.

Creates a 256 MiB CIPHER VMM weight arena, fills it with a known pattern,
exports the arena fd over a UNIX socket, prints a checksum, then sleeps —
waiting to be SIGKILL'd by the probe orchestrator. The SIGKILL (no cleanup)
is the test: does the consumer's mapping survive the producer's death?
"""
import os
import socket
import sys
import time

sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch                                                  # noqa: E402
import cipher_kv_bridge as kvb                                # noqa: E402

SOCK = "/tmp/sc5_probe.sock"
ARENA_BYTES = 256 * 1024 * 1024
N = ARENA_BYTES // 4                                          # float32 count


def main():
    kvb.init(64 * 1024 * 1024)
    arena = kvb.weight_arena_create(ARENA_BYTES + 1024 * 1024, 7)
    t = arena.alloc([N], 4, "float32")
    torch.manual_seed(0xC1FE)
    t.copy_(torch.randn(N, device="cuda", dtype=torch.float32))
    torch.cuda.synchronize()
    checksum = float(t.double().sum().item())

    fd = arena.export_fd()
    if os.path.exists(SOCK):
        os.remove(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(1)
    print("PRODUCER ready base=0x%x size=%d N=%d checksum=%.6f"
          % (arena.base, arena.size, N, checksum), flush=True)

    conn, _ = srv.accept()
    socket.send_fds(conn, [b"sc5probe"], [fd])
    conn.close()
    os.close(fd)
    srv.close()
    print("PRODUCER served fd; sleeping (awaiting SIGKILL)", flush=True)
    time.sleep(300)


if __name__ == "__main__":
    main()
