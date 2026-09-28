#!/usr/bin/env python3
"""Track 2 SC5 — weight-sharing PRODUCER (kmod-rendezvous).

Loads TinyLlama, packs every weight tensor into a CIPHER VMM weight arena
(SC2's WeightArena), runs a reference forward pass, then REGISTERs the arena
fd + a metadata blob (the SC3 layout manifest + SC4 fingerprint) with the
kmod weight-arena registry (CIPHER_ARENA_REGISTER, NR 21) and stays alive
until SIGKILL'd by the orchestrator.

Unlike SC3/SC4 there is NO UNIX socket: the kmod is the single rendezvous.
The kmod takes its own `struct file *` reference on the exported fd at
REGISTER, so the arena outlives this process — that is exactly what SC5
proves when sc5_run.py SIGKILLs this producer with a consumer joined.
"""
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
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
DT = {torch.float16: "float16", torch.bfloat16: "bfloat16",
      torch.float32: "float32"}
PROMPT = "The history of computing spans several distinct eras, each defined by"

# pinned at module scope so the arena (and its CUDA handle) is never GC'd
# while the producer sleeps — the SC4 lifetime lesson.
_ARENA = None


def main():
    global _ARENA
    kvb.init(64 * 1024 * 1024)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).cuda()
    model.train(False)

    params = list(model.named_parameters())
    total = sum(p.numel() * p.element_size() for _, p in params)
    arena = kvb.weight_arena_create(int(total) + 16 * 1024 * 1024, 1)
    _ARENA = arena

    manifest = {"base": int(arena.base), "size": int(arena.size),
                "model": MODEL, "tensors": [],
                "fingerprint": cmf.compute_fingerprint(MODEL)}
    for name, p in params:
        esize = p.element_size()
        vt = arena.alloc(list(p.shape), esize, DT[p.dtype])
        vt.copy_(p.data)
        manifest["tensors"].append({
            "name": name, "shape": list(p.shape), "dtype": DT[p.dtype],
            "itemsize": esize, "offset": int(vt.data_ptr() - arena.base),
            "nbytes": int(p.numel() * esize)})
        p.data = vt
    torch.cuda.synchronize()

    # reference forward pass on the arena-backed weights
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        yp = model(ids).logits.float().cpu()
    torch.save(yp, HERE + "/sc5_producer_logits.pt")

    blob = json.dumps(manifest).encode()
    print("PRODUCER blob=%d bytes (cap %d)"
          % (len(blob), aioctl.CIPHER_WA_BLOB_MAX), flush=True)

    # REGISTER with the kmod — it fget's the exported fd; we may then close
    # our copy, the kmod's reference keeps the arena alive past our death.
    fd = arena.export_fd()
    arena_id = aioctl.register(fd, arena.base, arena.size, blob)
    os.close(fd)
    print("PRODUCER registered arena_id=%d base=0x%x size=%d ntensors=%d "
          "fingerprint=%s" % (arena_id, arena.base, arena.size, len(params),
                              manifest["fingerprint"]["combined"][:16]),
          flush=True)

    # stay alive (a live participant) until sc5_run.py SIGKILLs us. The
    # SIGKILL — no cleanup, no ARENA_LEAVE — is the crash-path test.
    time.sleep(600)


if __name__ == "__main__":
    main()
