#!/usr/bin/env python3
"""Track 2 SC6 — weight-sharing CONSUMER (N=4 verification).

Extends sc5_consumer.py with a consumer index so 4 concurrent instances do
not collide on output files. Each consumer:

  1. IMPORTs the shared arena from the kmod registry (no producer needed);
  2. verifies the SC4 model fingerprint;
  3. maps the arena, meta-loads the model, rebinds every parameter onto a
     read-only arena view;
  4. forward pass #1 (producer still alive) -> `..._pre_logits.pt`;
  5. WAITs for the sentinel sc6_kill_done (sc6_run.py drops it AFTER the
     producer is SIGKILL'd);
  6. forward pass #2 (producer dead) -> `..._post_logits.pt`.

Both forwards must be bit-identical to the producer's reference — #2 is the
SC5 crux ("a consumer keeps working after the producer dies") scaled to N=4.

argv: [arena_id] [consumer_index] [model_name]
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
import sc6_models as M                                        # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM     # noqa: E402
from transformers import AutoTokenizer                        # noqa: E402
from accelerate import init_empty_weights                     # noqa: E402

SENTINEL = HERE + "/sc6_kill_done"
_ARENA = None


def set_submodule_attr(model, dotted, kind, value):
    *path, attr = dotted.split(".")
    mod = model
    for seg in path:
        mod = getattr(mod, seg)
    getattr(mod, "_" + kind)[attr] = value


def main():
    arena_id = int(sys.argv[1])
    idx = int(sys.argv[2])
    model_name = sys.argv[3]
    spec = M.MODELS[model_name]
    path, dtype = spec["path"], spec["dtype"]
    tag = "C%d" % idx
    out = "%s/sc6_%s_consumer_%d" % (HERE, model_name, idx)

    global _ARENA
    kvb.init(64 * 1024 * 1024)

    # ---- IMPORT from the kmod registry (no producer process needed) ----
    fd, base, size, blob = aioctl.import_arena(arena_id)
    manifest = json.loads(blob.decode())
    print("%s imported arena_id=%d fd=%d base=0x%x" % (tag, arena_id, fd, base),
          flush=True)

    # ---- SC4 fingerprint verification before mapping ----
    cmf.verify_fingerprint(manifest, path, manifest_path="kmod-arena-blob")
    print("%s fingerprint verified" % tag, flush=True)

    arena = kvb.weight_arena_import(fd, 0, manifest["size"])
    _ARENA = arena
    os.close(fd)

    # ---- meta-load + rebind every parameter onto a read-only arena view ----
    config = AutoConfig.from_pretrained(path)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    mani = {t["name"]: t for t in manifest["tensors"]}
    n_rebound = 0
    for name, p in list(model.named_parameters()):
        e = mani[name]
        vt = arena.view(e["shape"], e["itemsize"], e["dtype"], e["offset"])
        set_submodule_attr(model, name, "parameters",
                           torch.nn.Parameter(vt, requires_grad=False))
        n_rebound += 1
    for name, b in list(model.named_buffers()):
        set_submodule_attr(model, name, "buffers", b.cuda())
    model.train(False)
    model._cipher_weight_arena = arena
    print("%s-IMPORTED rebound %d params" % (tag, n_rebound), flush=True)

    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(M.CANONICAL_PROMPT, return_tensors="pt").input_ids.cuda()

    # ---- forward #1 — producer still alive ----
    with torch.no_grad():
        y1 = model(ids).logits.float().cpu()
    torch.save(y1, out + "_pre_logits.pt")
    big_name, big_p = max(model.named_parameters(),
                          key=lambda kv: kv[1].numel())
    pi = kvb.page_info(big_p.data_ptr())
    print("%s-FWD1 done page_info(%s)=%s" % (tag, big_name, pi), flush=True)

    # ---- wait until the producer has been SIGKILL'd ----
    t0 = time.time()
    while not os.path.exists(SENTINEL):
        if time.time() - t0 > 240:
            print("%s FAIL — sentinel never appeared" % tag, flush=True)
            sys.exit(1)
        time.sleep(0.25)

    # ---- forward #2 — producer is dead (SC5 crux, scaled to N=4) ----
    with torch.no_grad():
        y2 = model(ids).logits.float().cpu()
    torch.save(y2, out + "_post_logits.pt")
    res = {"consumer": idx, "model": model_name, "arena_id": arena_id,
           "n_params": n_rebound, "page_info_big": pi, "big_tensor": big_name}
    json.dump(res, open(out + "_result.json", "w"), default=str)
    print("%s-FWD2-DONE forward after producer death complete" % tag,
          flush=True)


if __name__ == "__main__":
    main()
