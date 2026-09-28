#!/usr/bin/env python3
"""Track 2 SC3 — weight-sharing PRODUCER (tenant 0).

Loads TinyLlama normally, packs every weight tensor into a CIPHER VMM weight
arena (SC2's WeightArena), runs a reference forward pass, then exports the
arena fd + a JSON layout manifest over a UNIX socket (SCM_RIGHTS) and stays
alive while the consumer maps the shared arena (SC3 assumes producer-owned
lifetime — kmod-owned lifetime is SC5).
"""
import json
import os
import socket
import sys
import time

sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch                                                  # noqa: E402
import cipher_kv_bridge as kvb                                # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
SOCK = "/tmp/sc3_weightshare.sock"
HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
DT = {torch.float16: "float16", torch.bfloat16: "bfloat16",
      torch.float32: "float32"}
PROMPT = "The history of computing spans several distinct eras, each defined by"


def main():
    kvb.init(64 * 1024 * 1024)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).cuda()
    model.train(False)                          # inference mode

    params = list(model.named_parameters())
    total = sum(p.numel() * p.element_size() for _, p in params)
    arena = kvb.weight_arena_create(int(total) + 16 * 1024 * 1024, 1)

    # pack each weight tensor into the arena; record its layout
    manifest = {"base": int(arena.base), "size": int(arena.size),
                "model": MODEL, "tensors": []}
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
    torch.save(yp, HERE + "/sc3_producer_logits.pt")

    mbytes = json.dumps(manifest).encode()
    if os.path.exists(SOCK):
        os.remove(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(4)
    srv.settimeout(150.0)      # SC3: producer-owned lifetime (SC5 = kmod-owned)
    print("PRODUCER ready base=0x%x size=%d ntensors=%d"
          % (arena.base, arena.size, len(params)), flush=True)

    # serve the arena to each consumer that connects (a fresh export fd each)
    served = 0
    t0 = time.time()
    while time.time() - t0 < 150:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            break
        fd = arena.export_fd()
        socket.send_fds(conn, [mbytes], [fd])
        conn.close()
        os.close(fd)
        served += 1
        print("PRODUCER served consumer #%d" % served, flush=True)
    srv.close()
    print("PRODUCER exit (served %d)" % served, flush=True)


if __name__ == "__main__":
    main()
