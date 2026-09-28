#!/usr/bin/env python3
"""Track 3 SC6 — barriered PARTITION decode tenant (within-A + clause-2).

Runs SC6_ROUNDS barriered decode rounds (the round count encodes the free
order — a tenant with fewer rounds exits sooner, freeing its groups). Each
round: KV-cached greedy decode (per-round latency), teacher-forced KL gate,
the migrate handler, and a %smid probe (the clause-2 input — contemporaneous
across tenants because the round is barriered). Then exits.

migratable: orchestrator sets CIPHER_MIGRATABLE=1 for the migratable class;
the pinned class leaves it unset (handler.step() then always no-ops).

Env: SC6_TENANT_ID SC6_OUT SC6_ROUNDS SC6_BARRIER_TAG [CIPHER_SC3_FAULT].
"""
import json
import os
import sys
import time

TID = os.environ["SC6_TENANT_ID"]
OUT = os.environ["SC6_OUT"]
ROUNDS = int(os.environ["SC6_ROUNDS"])
TAG = os.environ["SC6_BARRIER_TAG"]
MIGRATABLE = bool(os.environ.get("CIPHER_MIGRATABLE"))
sys.path.insert(0, "/home/ubuntu/cipher-fusion-evidence/phase_c/track_3")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

import torch                                                  # noqa: E402
import torch.nn.functional as F                               # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import cipher_migrate as cm                                   # noqa: E402

WL01 = "/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01"
MODEL = "/home/ubuntu/models/TinyLlama-1.1B"


def log(m):
    print("[sc6-tenant %s] %s" % (TID, m), file=sys.stderr, flush=True)


def barrier(r):
    """Per-round lockstep (cp54_s16 pattern). Drop this tenant's done sentinel,
    block for the orchestrator's go before the next round."""
    open("/tmp/sc6_%s_%s_r%d.done" % (TAG, TID, r), "w").close()
    go = "/tmp/sc6_%s_all_r%d.go" % (TAG, r)
    while not os.path.exists(go):
        time.sleep(0.02)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).cuda()
    model.train(False)
    torch.cuda.synchronize()
    h = cm.MigrateHandler()
    cm._smid_ext()                          # warm the %smid compile
    gold = {g["prompt"]: g for g in json.load(open(WL01 + "/gold.json"))["prompts"]}
    gold_logits = torch.load(WL01 + "/gold_logits.pt", map_location="cuda")
    n_prompts = len(gold)
    log("model loaded; rounds=%d migratable=%s start_mask=0x%x"
        % (ROUNDS, MIGRATABLE, h.cur_mask()))

    rounds, kl_max_all = [], 0.0
    for r in range(ROUNDS):
        pr = r % n_prompts
        g = gold[pr]
        plen = g["plen"]
        gen = gold_logits[pr].shape[0]

        cur = torch.tensor([g["full_ids"][:plen]], device="cuda")
        past, dms = None, []
        with torch.no_grad():
            for st in range(gen):
                torch.cuda.synchronize()
                ts = time.perf_counter()
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - ts) * 1e3
                if st > 0:
                    dms.append(dt)
                past = o.past_key_values
                cur = o.logits[:, -1, :].argmax(-1, keepdim=True)

        full = torch.tensor([g["full_ids"]], device="cuda")
        with torch.no_grad():
            lo = model(input_ids=full).logits[0]
        seg = lo[plen - 1:plen - 1 + gen, :].float()
        lp = F.log_softmax(gold_logits[pr].float(), dim=-1)
        lq = F.log_softmax(seg, dim=-1)
        kl = float((lp.exp() * (lp - lq)).sum(-1).max())
        kl_max_all = max(kl_max_all, kl)

        act = h.step()                       # poll-and-migrate (no-op if pinned)
        migrated = (act["action"] == "migrate" and act.get("committed"))
        sms = cm.probe_sms()                 # clause-2 input (barriered round)
        mean_d = sum(dms) / len(dms) if dms else 0.0
        rounds.append({"round": r, "t": time.time(), "decode_ms": dms,
                       "decode_ms_mean": mean_d, "kl_max": kl,
                       "migrated": bool(migrated), "observed_sms": sms,
                       "cur_mask": h.cur_mask(),
                       "primitive_ms": act.get("primitive_ms"),
                       "l2_ms": act.get("l2_ms")})
        log("round %d kl=%.2e mean=%.2fms migrated=%s |sms|=%d"
            % (r, kl, mean_d, migrated, len(sms)))
        barrier(r)

    committed = [rd for rd in rounds if rd["migrated"]]
    res = {"tenant": TID, "pid": os.getpid(), "migratable": MIGRATABLE,
           "rounds_run": ROUNDS, "kl_max_all": kl_max_all,
           "kl_gate_pass": bool(kl_max_all <= 0.1),
           "n_migrated_rounds": len(committed),
           "fault": os.environ.get("CIPHER_SC3_FAULT"), "rounds": rounds}
    json.dump(res, open(OUT, "w"), default=str)
    h.close()
    log("DONE kl_max=%.2e migrated_rounds=%d" % (kl_max_all, len(committed)))


if __name__ == "__main__":
    main()
