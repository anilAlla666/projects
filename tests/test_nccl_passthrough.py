#!/usr/bin/env python3
"""Verify CIPHER's ncclAllReduce passthrough doesn't hang.

Spawned by torchrun. Each rank does one AllReduce. If both finish in <30s,
the fix works. If either rank hangs, NCCL watchdog will kill at 10min.

Usage:
  # Without CIPHER (sanity baseline):
  torchrun --nproc_per_node=2 tests/test_nccl_passthrough.py

  # With CIPHER (the actual test):
  LD_PRELOAD=./libcipher_hook.so:/usr/lib/x86_64-linux-gnu/libcuda.so \\
      torchrun --nproc_per_node=2 tests/test_nccl_passthrough.py

For 8-rank verification:
  LD_PRELOAD=./libcipher_hook.so:/usr/lib/x86_64-linux-gnu/libcuda.so \\
      torchrun --nproc_per_node=8 tests/test_nccl_passthrough.py
"""
import os
import sys
import time
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RT_PATH = ROOT / "libcipher_rt.so"


def main():
    # Import torch FIRST so libcudart is loaded — then ctypes.CDLL libcipher_rt.so
    # can resolve its cudaXxx symbol references
    import torch
    import torch.distributed as dist

    # Set the NCCL_TUNER_PLUGIN to the cipher plugin
    if os.environ.get("LD_PRELOAD") and "libcipher_hook" in os.environ.get("LD_PRELOAD", ""):
        os.environ.setdefault("NCCL_TUNER_PLUGIN", "cipher")
        os.environ.setdefault("LD_LIBRARY_PATH",
                               f"{ROOT}:{os.environ.get('LD_LIBRARY_PATH', '')}")
        if RT_PATH.exists():
            try:
                ctypes.CDLL(str(RT_PATH), mode=ctypes.RTLD_GLOBAL)
                print(f"[rank ?] loaded {RT_PATH}", file=sys.stderr)
            except Exception as ex:
                print(f"[rank ?] RT load skipped: {ex}", file=sys.stderr)

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "2"))

    print(f"[rank {rank}] init dist nccl, world={world}", file=sys.stderr)
    t0 = time.perf_counter()
    dist.init_process_group("nccl", rank=rank, world_size=world,
                              timeout=__import__('datetime').timedelta(seconds=120))
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    print(f"[rank {rank}] init done in {time.perf_counter()-t0:.2f}s", file=sys.stderr)

    sizes = [
        128 * 1024,             # < 256 KB
        2 * 1024 * 1024,        # 4 MB
        16 * 1024 * 1024,       # 32 MB → triggers v4 NVLS bias
    ]
    for s in sizes:
        b = torch.randn(s, dtype=torch.float16, device=dev)
        t0 = time.perf_counter()
        dist.all_reduce(b, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        print(f"[rank {rank}] all_reduce size={s:>10} bytes={s*2:>12} elapsed={dt*1e3:.1f}ms",
              file=sys.stderr)

    dist.destroy_process_group()
    print(f"[rank {rank}] DONE — passthrough works", file=sys.stderr)


if __name__ == "__main__":
    main()
