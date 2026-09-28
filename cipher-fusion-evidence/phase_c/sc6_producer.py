#!/usr/bin/env python3
"""Track 2 SC6 — weight-sharing PRODUCER (N=4 verification).

Same as sc5_producer.py, plus:
  - parameterised by model name (TinyLlama | Mistral-7B);
  - an explicit `gc.collect()` + `torch.cuda.empty_cache()` after packing,
    with framebuffer checkpoints logged before/after — so the producer's
    steady-state footprint is `arena + context`, not `arena + a redundant
    from_pretrained copy` (SC6 design memo §4 honesty correction).

argv: [model_name]
"""
import gc
import json
import os
import sys
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch                                                  # noqa: E402
import cipher_kv_bridge as kvb                                # noqa: E402
import cipher_model_fingerprint as cmf                        # noqa: E402
import sc5_arena_ioctl as aioctl                              # noqa: E402
import sc6_models as M                                        # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

_ARENA = None      # module-scope pin — never GC the arena while alive


def ckpt(tag):
    print("PRODUCER mem %-16s fb=%d MiB reserved=%d MiB allocated=%d MiB"
          % (tag, M.gpu_fb_mib(),
             torch.cuda.memory_reserved() // (1 << 20),
             torch.cuda.memory_allocated() // (1 << 20)), flush=True)


def main():
    global _ARENA
    model_name = sys.argv[1]
    spec = M.MODELS[model_name]
    path, dtype = spec["path"], spec["dtype"]

    kvb.init(64 * 1024 * 1024)
    torch.manual_seed(0)
    ckpt("start")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype).cuda()
    model.train(False)
    ckpt("after_load")

    params = list(model.named_parameters())
    total = sum(p.numel() * p.element_size() for _, p in params)
    arena = kvb.weight_arena_create(int(total) + 16 * 1024 * 1024, 1)
    _ARENA = arena

    manifest = {"base": int(arena.base), "size": int(arena.size),
                "model": path, "model_name": model_name,
                "torch_dtype": str(dtype), "tensors": [],
                "fingerprint": cmf.compute_fingerprint(path)}
    for name, p in params:
        esize = p.element_size()
        vt = arena.alloc(list(p.shape), esize, M.DT[p.dtype])
        vt.copy_(p.data)
        manifest["tensors"].append({
            "name": name, "shape": list(p.shape), "dtype": M.DT[p.dtype],
            "itemsize": esize, "offset": int(vt.data_ptr() - arena.base),
            "nbytes": int(p.numel() * esize)})
        p.data = vt
    torch.cuda.synchronize()
    ckpt("after_pack")

    # Drop the redundant from_pretrained storages: every parameter now points
    # at an arena VMM view, so the original CUDA storages are unreferenced.
    gc.collect()
    torch.cuda.empty_cache()
    ckpt("after_empty_cache")      # fb here should be ~arena + context

    # reference forward pass on the arena-backed weights
    ids = tok(M.CANONICAL_PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        yp = model(ids).logits.float().cpu()
    torch.save(yp, "%s/sc6_%s_producer_logits.pt" % (HERE, model_name))
    ckpt("after_forward")

    blob = json.dumps(manifest).encode()
    if len(blob) > aioctl.CIPHER_WA_BLOB_MAX:
        print("PRODUCER FATAL — blob %d > CIPHER_WA_BLOB_MAX %d"
              % (len(blob), aioctl.CIPHER_WA_BLOB_MAX), flush=True)
        sys.exit(2)

    fd = arena.export_fd()
    arena_id = aioctl.register(fd, arena.base, arena.size, blob)
    os.close(fd)
    print("PRODUCER registered arena_id=%d model=%s base=0x%x size=%d "
          "ntensors=%d blob=%d fingerprint=%s"
          % (arena_id, model_name, arena.base, arena.size, len(params),
             len(blob), manifest["fingerprint"]["combined"][:16]), flush=True)

    time.sleep(900)        # stay a live participant until SIGKILL'd


if __name__ == "__main__":
    main()
