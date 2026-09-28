"""W.7 item 1 (NCCL 2.26.2 binds the plugin) + item 3 (pass-through bit-identical).
2-rank NCCL on the single H100 (both ranks device 0). Exactly-representable data so
any valid algorithm yields the EXACT sum -> bit-identical is the correct Memory #11
expectation regardless of algorithm choice."""
import os, sys, json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N = 1 << 20  # 1M floats = 4 MiB AllReduce

def worker(rank, world, out_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    # exactly-representable integer-valued floats: rank r contributes (r+1)
    t = torch.full((N,), float(rank + 1), device="cuda:0", dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)  # expect world*(world+1)/2 = 1+2 = 3
    torch.cuda.synchronize()
    if rank == 0:
        exp = float(world * (world + 1) // 2)
        ok = bool(torch.all(t == exp).item())
        # checksum of the raw bytes for bit-identical cross-run comparison
        import hashlib
        h = hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()
        json.dump({"world": world, "expected": exp, "exact": ok, "md5": h,
                   "first": float(t[0].item())}, open(out_path, "w"))
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    out = sys.argv[1]
    mp.spawn(worker, args=(2, out), nprocs=2, join=True)
