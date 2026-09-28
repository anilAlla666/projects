#!/usr/bin/env python3
"""Track 3 SC5 — 5-partition churn test: one PARTITION decode tenant.

Launched by sc5_churn.py under CUDA_INJECTION64_PATH=<SC5 libcipher_rt> with
CIPHER_QOS_CLASS=partition + CIPHER_SM_COUNT + CIPHER_MIGRATABLE=1. Runs
KV-cached greedy decode rounds for SC5_LIFETIME_S seconds — calling the
migrate handler each round — then exits. Process exit → the kmod do_exit
reaper frees this tenant's 8-SM groups: that exit IS the 'partition free'
event whose ordering the seed controls.

Records per-decode-step latency, per-prompt TFGATE KL, and any migrations.
Env: SC5_TENANT_ID SC5_OUT SC5_LIFETIME_S [CIPHER_SC3_FAULT].
"""
import json
import os
import sys
import time

TID = os.environ["SC5_TENANT_ID"]
OUT = os.environ["SC5_OUT"]
LIFETIME = float(os.environ["SC5_LIFETIME_S"])
sys.path.insert(0, "/home/ubuntu/cipher-fusion-evidence/phase_c/track_3")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

import torch                                                  # noqa: E402
import torch.nn.functional as F                               # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import cipher_migrate as cm                                   # noqa: E402

WL01 = "/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01"
MODEL = "/home/ubuntu/models/TinyLlama-1.1B"


def log(m):
    print("[sc5-tenant %s] %s" % (TID, m), file=sys.stderr, flush=True)


def main():
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).cuda()
    model.train(False)
    torch.cuda.synchronize()

    h = cm.MigrateHandler()
    cm._smid_ext()              # warm the L2 %smid compile before the loop
    gold = {g["prompt"]: g for g in json.load(open(WL01 + "/gold.json"))["prompts"]}
    gold_logits = torch.load(WL01 + "/gold_logits.pt", map_location="cuda")
    n_prompts = len(gold)

    start_mask = h.cur_mask()
    decode_ms, migrations = [], []
    kl_max_all, rounds, r = 0.0, 0, 0
    log("model loaded, start_mask=0x%x, lifetime=%.0fs" % (start_mask, LIFETIME))

    while time.time() - t_start < LIFETIME:
        pr = r % n_prompts
        g = gold[pr]
        plen = g["plen"]
        gen = gold_logits[pr].shape[0]

        # KV-cached greedy decode, per-step timed (step 0 = prefill, excluded)
        cur = torch.tensor([g["full_ids"][:plen]], device="cuda")
        past = None
        with torch.no_grad():
            for st in range(gen):
                torch.cuda.synchronize()
                ts = time.perf_counter()
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - ts) * 1e3
                if st > 0:
                    decode_ms.append(dt)
                past = o.past_key_values
                cur = o.logits[:, -1, :].argmax(-1, keepdim=True)

        # teacher-forced KL gate
        full = torch.tensor([g["full_ids"]], device="cuda")
        with torch.no_grad():
            lo = model(input_ids=full).logits[0]
        seg = lo[plen - 1:plen - 1 + gen, :].float()
        lp = F.log_softmax(gold_logits[pr].float(), dim=-1)
        lq = F.log_softmax(seg, dim=-1)
        kl_max_all = max(kl_max_all, float((lp.exp() * (lp - lq)).sum(-1).max()))

        # poll-and-migrate handler at the decode-round boundary
        act = h.step()
        if act["action"] == "migrate":
            migrations.append({"round": r, **act})
            log("round %d MIGRATE -> 0x%x committed=%s"
                % (r, act.get("target_mask", 0), act.get("committed")))
        rounds += 1
        r += 1

    committed = [m for m in migrations if m.get("committed")]
    res = {
        "tenant": TID, "pid": os.getpid(), "rounds": rounds,
        "start_mask": start_mask, "end_mask": h.cur_mask(),
        "decode_ms": decode_ms, "n_decode_steps": len(decode_ms),
        "kl_max_all": kl_max_all, "kl_gate_pass": bool(kl_max_all <= 0.1),
        "n_migrations": len(migrations), "n_committed": len(committed),
        "migrations": migrations,
        "fault": os.environ.get("CIPHER_SC3_FAULT"),
    }
    json.dump(res, open(OUT, "w"), default=str)
    h.close()
    log("DONE rounds=%d kl_max=%.2e migrations=%d/%d"
        % (rounds, kl_max_all, len(committed), len(migrations)))


if __name__ == "__main__":
    main()
