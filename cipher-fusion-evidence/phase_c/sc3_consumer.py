#!/usr/bin/env python3
"""Track 2 SC3 — weight-sharing CONSUMER (peer tenant).

Receives the producer's arena fd + JSON manifest over a UNIX socket, imports
the shared VMM arena, meta-loads TinyLlama (params on `meta` — the safetensors
weight read is SKIPPED), rebinds every parameter onto a READ-only view of the
imported arena, and runs the forward pass. Its weights are the producer's
exact physical bytes — the forward must be bit-identical.
"""
import json
import os
import socket
import sys

sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch                                                  # noqa: E402
import cipher_kv_bridge as kvb                                # noqa: E402
from transformers import (AutoConfig, AutoModelForCausalLM,   # noqa: E402
                          AutoTokenizer)
from accelerate import init_empty_weights                     # noqa: E402

MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
SOCK = "/tmp/sc3_weightshare.sock"
HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
PROMPT = "The history of computing spans several distinct eras, each defined by"


def set_submodule_attr(model, dotted, kind, value):
    """Replace model.<dotted> in _parameters / _buffers of its owning module."""
    *path, attr = dotted.split(".")
    mod = model
    for seg in path:
        mod = getattr(mod, seg)
    getattr(mod, "_" + kind)[attr] = value


def main():
    kvb.init(64 * 1024 * 1024)

    # ---- receive fd + manifest (SCM_RIGHTS) ----
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(SOCK)
    msg, fds, _, _ = socket.recv_fds(c, 1 << 20, 1)
    c.close()
    manifest = json.loads(msg.decode())
    fd = fds[0]

    # ---- import the shared arena ----
    # SC3_FORCE_OFFSET=1 passes want_base=0 — skips the same-VA attempt,
    # exercising the offset-relative path (the same-VA fallback).
    want = 0 if os.environ.get("SC3_FORCE_OFFSET") else manifest["base"]
    arena = kvb.weight_arena_import(fd, want, manifest["size"])
    os.close(fd)
    same_va = (arena.base == manifest["base"])
    print("CONSUMER imported arena base=0x%x (producer 0x%x) same_va=%s"
          % (arena.base, manifest["base"], same_va), flush=True)

    # ---- meta-load: params on `meta`, buffers real (safetensors NOT read) ----
    config = AutoConfig.from_pretrained(MODEL)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config,
                                                 torch_dtype=torch.float16)

    # ---- rebind every parameter onto a READ-only arena view ----
    mani = {t["name"]: t for t in manifest["tensors"]}
    n_rebound = 0
    for name, p in list(model.named_parameters()):
        e = mani[name]
        vt = arena.view(e["shape"], e["itemsize"], e["dtype"], e["offset"])
        set_submodule_attr(model, name, "parameters",
                           torch.nn.Parameter(vt, requires_grad=False))
        n_rebound += 1

    # ---- buffers (computed locally by init_empty_weights) -> CUDA ----
    for name, b in list(model.named_buffers()):
        set_submodule_attr(model, name, "buffers", b.cuda())
    model.train(False)
    print("CONSUMER rebound %d params + %d buffers"
          % (n_rebound, len(list(model.named_buffers()))), flush=True)

    # ---- forward pass on the shared weights ----
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        yc = model(ids).logits.float().cpu()
    torch.save(yc, HERE + "/sc3_consumer_logits.pt")

    # ---- structural check: a consumer weight tensor is arena memory ----
    big_name, big_p = max(model.named_parameters(), key=lambda kv: kv[1].numel())
    pi = kvb.page_info(big_p.data_ptr())
    res = {"same_va": bool(same_va), "n_params": n_rebound,
           "consumer_base": int(arena.base), "producer_base": manifest["base"],
           "page_info_big": pi, "big_tensor": big_name}
    json.dump(res, open(HERE + "/sc3_consumer_result.json", "w"), default=str)
    print("CONSUMER forward done; page_info(%s)=%s" % (big_name, pi),
          flush=True)


if __name__ == "__main__":
    main()
