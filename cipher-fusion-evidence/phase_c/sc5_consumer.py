#!/usr/bin/env python3
"""Track 2 SC5 — weight-sharing CONSUMER (kmod-rendezvous).

Imports a shared weight arena from the kmod registry (CIPHER_ARENA_IMPORT,
NR 22) — by arena id, with NO live producer required — verifies the model
fingerprint (SC4), meta-loads TinyLlama, rebinds every parameter onto a
READ-only view of the imported arena, and runs the forward pass on the
producer's exact physical bytes.

Two modes (argv[2]):
  A — import + rebind, print A-IMPORTED, then WAIT for the sentinel file
      $HERE/sc5_kill_done (sc5_run.py creates it AFTER SIGKILLing the
      producer), then run the forward pass. This proves a consumer's mapping
      keeps working after the producer process has died.
  B — import + rebind + forward immediately. sc5_run.py launches mode B
      only AFTER the producer is dead — so a successful IMPORT here proves
      the kmod rendezvous delivers the fd with no producer alive at all.

argv: [arena_id] [mode A|B]
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
from transformers import AutoConfig, AutoModelForCausalLM     # noqa: E402
from transformers import AutoTokenizer                        # noqa: E402
from accelerate import init_empty_weights                     # noqa: E402

MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
SENTINEL = HERE + "/sc5_kill_done"
PROMPT = "The history of computing spans several distinct eras, each defined by"

# pinned at module scope — the SC4 lifetime lesson: a GC'd arena unmaps the
# weights out from under the model. SC5 makes the arena kmod-owned, but the
# consumer's own import handle must still outlive its forward pass.
_ARENA = None


def set_submodule_attr(model, dotted, kind, value):
    *path, attr = dotted.split(".")
    mod = model
    for seg in path:
        mod = getattr(mod, seg)
    getattr(mod, "_" + kind)[attr] = value


def main():
    global _ARENA
    arena_id = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "B"
    tag = "CONSUMER-" + mode

    kvb.init(64 * 1024 * 1024)

    # ---- IMPORT the arena from the kmod registry (no producer needed) ----
    fd, base, size, blob = aioctl.import_arena(arena_id)
    manifest = json.loads(blob.decode())
    print("%s imported arena_id=%d fd=%d base=0x%x size=%d (producer base 0x%x)"
          % (tag, arena_id, fd, base, size, manifest["base"]), flush=True)

    # ---- SC4 fingerprint verification BEFORE mapping the weights ----
    cmf.verify_fingerprint(manifest, MODEL, manifest_path="kmod-arena-blob")
    print("%s fingerprint verified (%s)"
          % (tag, manifest["fingerprint"]["combined"][:16]), flush=True)

    # ---- map the shared arena (offset-relative; want_base=0) ----
    arena = kvb.weight_arena_import(fd, 0, manifest["size"])
    _ARENA = arena
    os.close(fd)
    same_va = (arena.base == manifest["base"])
    print("%s mapped arena base=0x%x same_va=%s" % (tag, arena.base, same_va),
          flush=True)

    # ---- meta-load: params on `meta`, buffers real (no safetensors read) ----
    config = AutoConfig.from_pretrained(MODEL)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config,
                                                 torch_dtype=torch.float16)

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
    model._cipher_weight_arena = arena      # pin lifetime to the model too
    print("%s rebound %d params" % (tag, n_rebound), flush=True)

    # ---- mode A waits until the producer has been SIGKILL'd ----
    if mode == "A":
        print("A-IMPORTED", flush=True)
        t0 = time.time()
        while not os.path.exists(SENTINEL):
            if time.time() - t0 > 180:
                print("%s FAIL — sentinel never appeared" % tag, flush=True)
                sys.exit(1)
            time.sleep(0.25)
        print("%s sentinel seen — producer is dead; running forward" % tag,
              flush=True)

    # ---- forward pass on the shared weights ----
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        yc = model(ids).logits.float().cpu()
    torch.save(yc, "%s/sc5_consumer_%s_logits.pt" % (HERE, mode))

    big_name, big_p = max(model.named_parameters(),
                          key=lambda kv: kv[1].numel())
    pi = kvb.page_info(big_p.data_ptr())
    res = {"mode": mode, "arena_id": arena_id, "n_params": n_rebound,
           "same_va": bool(same_va), "consumer_base": int(arena.base),
           "producer_base": manifest["base"], "page_info_big": pi,
           "big_tensor": big_name}
    json.dump(res, open("%s/sc5_consumer_%s_result.json" % (HERE, mode), "w"),
              default=str)
    print("%s-DONE forward complete; page_info(%s)=%s"
          % (tag, big_name, pi), flush=True)


if __name__ == "__main__":
    main()
