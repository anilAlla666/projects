#!/usr/bin/env python3
"""NCCL AllReduce benchmark across 2 GPUs.

Launch with torchrun:
    torchrun --nproc_per_node=2 nccl_bench.py

Reads NCCL_TUNER_PLUGIN env var: if set, NCCL loads the CIPHER tuner.
"""
import os, time, torch, torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    SIZE_BYTES = 128 * 1024 * 1024  # 128 MB
    n_elem = SIZE_BYTES // 2  # fp16
    x = torch.randn(n_elem, dtype=torch.float16, device=f"cuda:{rank}")

    # Warmup
    for _ in range(5):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    # Timed run
    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        per_iter_s = elapsed / n_iter
        per_iter_us = per_iter_s * 1e6
        # Standard NCCL AllReduce bus bandwidth (ring): 2*(n-1)/n * size / t
        n = world
        alg_bw = SIZE_BYTES / per_iter_s / 1e9
        bus_bw = SIZE_BYTES * 2 * (n - 1) / n / per_iter_s / 1e9
        tuner = os.environ.get("NCCL_TUNER_PLUGIN", "(none)")
        print(f"[nccl_bench] world={world} size={SIZE_BYTES//1024//1024}MB "
              f"iters={n_iter}")
        print(f"[nccl_bench] tuner_plugin={tuner}")
        print(f"[nccl_bench] per_iter={per_iter_us:.1f}us  "
              f"algBW={alg_bw:.1f}GB/s  busBW={bus_bw:.1f}GB/s  "
              f"total={elapsed:.2f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
