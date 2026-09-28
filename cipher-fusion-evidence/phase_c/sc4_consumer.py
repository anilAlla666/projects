#!/usr/bin/env python3
"""Track 2 SC4 — weight-sharing CONSUMER with model-identity verification.

Receives the producer's arena fd + manifest, then — BEFORE importing the
arena — verifies the producer's model fingerprint against the consumer's own
model (env SC4_CONSUMER_MODEL, default TinyLlama).

  match    -> import the shared arena, rebind, run (the SC3 shared path).
  mismatch -> FingerprintMismatch: log the details, do NOT import, fall back
              to an independent safetensors load of the consumer's own model.

The explicit try/except + detailed mismatch log is the Item-3 PUSH: a silent
fallback would hide the operator error that caused the mismatch.
"""
import json
import os
import socket
import sys

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch                                                  # noqa: E402
import cipher_kv_bridge as kvb                                # noqa: E402
import cipher_model_fingerprint as cmf                        # noqa: E402
from transformers import (AutoConfig, AutoModelForCausalLM,   # noqa: E402
                          AutoTokenizer)
from accelerate import init_empty_weights                     # noqa: E402

SOCK = "/tmp/sc4_weightshare.sock"
CONSUMER_MODEL = os.environ.get("SC4_CONSUMER_MODEL",
                                "/home/ubuntu/models/TinyLlama-1.1B")
PROMPT = "The history of computing spans several distinct eras, each defined by"


def log(m):
    print("[sc4-consumer] %s" % m, file=sys.stderr, flush=True)


def set_submodule_attr(model, dotted, kind, value):
    *path, attr = dotted.split(".")
    mod = model
    for seg in path:
        mod = getattr(mod, seg)
    getattr(mod, "_" + kind)[attr] = value


def run_shared(manifest, fd):
    """The SC3 shared path — import the arena, rebind, forward."""
    arena = kvb.weight_arena_import(fd, manifest["base"], manifest["size"])
    os.close(fd)
    config = AutoConfig.from_pretrained(CONSUMER_MODEL)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config,
                                                 torch_dtype=torch.float16)
    mani = {t["name"]: t for t in manifest["tensors"]}
    for name, p in list(model.named_parameters()):
        e = mani[name]
        vt = arena.view(e["shape"], e["itemsize"], e["dtype"], e["offset"])
        set_submodule_attr(model, name, "parameters",
                           torch.nn.Parameter(vt, requires_grad=False))
    for name, b in list(model.named_buffers()):
        set_submodule_attr(model, name, "buffers", b.cuda())
    model.train(False)
    # Pin the arena's lifetime to the model: the parameter tensors are
    # from_blob views with a no-op deleter — if the WeightArena is GC'd the
    # arena is unmapped and the views dangle. (SC5 = kmod-owned lifetime.)
    model._cipher_weight_arena = arena
    return model


def run_independent(fd):
    """Fallback — independent safetensors load of the consumer's own model."""
    os.close(fd)                                  # do NOT import the arena
    model = AutoModelForCausalLM.from_pretrained(
        CONSUMER_MODEL, torch_dtype=torch.float16).cuda()
    model.train(False)
    return model


def main():
    kvb.init(64 * 1024 * 1024)
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(SOCK)
    msg, fds, _, _ = socket.recv_fds(c, 1 << 20, 1)
    c.close()
    manifest = json.loads(msg.decode())
    fd = fds[0]

    res = {"consumer_model": CONSUMER_MODEL}
    try:
        # SC4 — verify model identity BEFORE committing any VMM resource
        cmf.verify_fingerprint(manifest, CONSUMER_MODEL,
                               manifest_path="<sc4_weightshare.sock>")
        log("fingerprint MATCH — proceeding with shared import")
        model = run_shared(manifest, fd)
        res["mode"] = "shared"
    except cmf.FingerprintMismatch as e:
        # Item-3 PUSH — explicit, detailed mismatch log (not a silent fallback)
        log("FINGERPRINT MISMATCH — falling back to independent load")
        log("  differing_tier      = %s" % e.differing_tier)
        log("  producer_hash       = %s" % e.producer_hash)
        log("  consumer_hash       = %s" % e.consumer_hash)
        log("  manifest_path       = %s" % e.manifest_path)
        log("  consumer_model_path = %s" % e.consumer_model_path)
        log("  producer_model      = %s" % manifest.get("model"))
        res.update(mode="fallback_independent", mismatch={
            "differing_tier": e.differing_tier,
            "producer_hash": e.producer_hash, "consumer_hash": e.consumer_hash,
            "producer_model": manifest.get("model"),
            "consumer_model": e.consumer_model_path})
        model = run_independent(fd)

    tok = AutoTokenizer.from_pretrained(CONSUMER_MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        yc = model(ids).logits.float().cpu()
    torch.save(yc, HERE + "/sc4_consumer_logits.pt")
    res["forward_ok"] = True
    res["logits_shape"] = list(yc.shape)
    json.dump(res, open(HERE + "/sc4_consumer_result.json", "w"), default=str)
    log("DONE mode=%s forward_ok=True" % res["mode"])


if __name__ == "__main__":
    main()
