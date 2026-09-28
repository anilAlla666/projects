#!/usr/bin/env python3
"""Track 2 SC4 — weight-sharing PRODUCER (= SC3 producer + model fingerprint).

Identical to sc3_producer.py except the JSON manifest now carries a
`fingerprint` field (cipher_model_fingerprint.compute_fingerprint) so a
consumer can verify model identity before importing the arena.
"""
import json
import os
import socket
import sys
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch                                                  # noqa: E402
import cipher_kv_bridge as kvb                                # noqa: E402
import cipher_model_fingerprint as cmf                        # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
SOCK = "/tmp/sc4_weightshare.sock"
DT = {torch.float16: "float16", torch.bfloat16: "bfloat16",
      torch.float32: "float32"}
PROMPT = "The history of computing spans several distinct eras, each defined by"


def main():
    kvb.init(64 * 1024 * 1024)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).cuda()
    model.train(False)

    params = list(model.named_parameters())
    total = sum(p.numel() * p.element_size() for _, p in params)
    arena = kvb.weight_arena_create(int(total) + 16 * 1024 * 1024, 1)

    manifest = {"base": int(arena.base), "size": int(arena.size),
                "model": MODEL, "tensors": [],
                "fingerprint": cmf.compute_fingerprint(MODEL)}   # SC4
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

    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        yp = model(ids).logits.float().cpu()
    torch.save(yp, HERE + "/sc4_producer_logits.pt")

    mbytes = json.dumps(manifest).encode()
    if os.path.exists(SOCK):
        os.remove(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(4)
    srv.settimeout(150.0)
    print("PRODUCER ready base=0x%x ntensors=%d fingerprint=%s"
          % (arena.base, len(params),
             manifest["fingerprint"]["combined"][:16]), flush=True)

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
