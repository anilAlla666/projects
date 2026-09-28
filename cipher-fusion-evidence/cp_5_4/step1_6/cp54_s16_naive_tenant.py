#!/usr/bin/env python3
"""CP 5.4 Step 1.6B-3 — naive Arm-A tenant (THROWAWAY).

The naive baseline tenant: pure PyTorch + transformers, model via
`from_pretrained`, **no CIPHER** — no kmod ledger query, no green context, no
SM partitioning, no `libcipher_rt`. It runs on the full GPU and time-slices
against every other tenant. This is the honest "multi-tenant with no
substrate" comparison point for the Arm-B mixed deployment.

Deliberately a **stripped `cp54_s16_partition_tenant.py`**: the model load,
the teacher-forced KL gate, `decode_round` (the per-decode-step timing code
path) and the per-round lockstep barrier are **byte-identical** to the
PARTITION tenant — the only difference between Arm A and Arm B must be
substrate-mediated SM partitioning vs none (Step 1.6 design memo §11.2). What
is removed: the kmod `CIPHER_CP54_QUERY` ledger read and the `%smid`
disjointness probe — neither has meaning with no partition.

Orchestrator coordination (same protocol as the PARTITION tenant):
  stdout: `READY <pid>`  then  `ROUND <r>` each round, then `DONE`.
  stdin : blocks for `START` before the timed rounds (skip with NOSIGNAL=1).

Env: TENANT OUT_JSON [ROUNDS=10] [NOSIGNAL=0] [BARRIER_TAG]
     [WL_MODEL] [GOLD_JSON] [GOLD_LOGITS]
"""
import json
import os
import sys
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
import torch                                                       # noqa: E402
import torch.nn.functional as F                                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer       # noqa: E402

WL01 = "/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01"
TENANT = os.environ.get("TENANT", "n0")
OUT_JSON = os.environ["OUT_JSON"]
ROUNDS = int(os.environ.get("ROUNDS", "10"))
NOSIGNAL = os.environ.get("NOSIGNAL", "0") == "1"
BARRIER_TAG = os.environ.get("BARRIER_TAG")        # set => per-round lockstep
MODEL = os.environ.get("WL_MODEL", "/home/ubuntu/models/TinyLlama-1.1B")
GOLD_JSON = os.environ.get("GOLD_JSON", WL01 + "/gold.json")
GOLD_LOGITS = os.environ.get("GOLD_LOGITS", WL01 + "/gold_logits.pt")


def log(m):
    print("[s16-naive %s] %s" % (TENANT, m), file=sys.stderr, flush=True)


def barrier(r, last):
    """Per-round lockstep barrier — byte-identical protocol to the PARTITION
    tenant: after round r drop this member's done sentinel, then block for the
    orchestrator's go sentinel. Keeps all naive tenants' contention windows
    co-extensive (Step 1.6B-2A precedent). No-op when BARRIER_TAG is unset."""
    if not BARRIER_TAG:
        return
    open("/tmp/cp54_s16_%s_%s_r%d.done" % (BARRIER_TAG, TENANT, r), "w").close()
    if last:
        return
    go = "/tmp/cp54_s16_%s_all_r%d.go" % (BARRIER_TAG, r)
    while not os.path.exists(go):
        time.sleep(0.02)


def main():
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).cuda()
    model.train(False)
    torch.cuda.synchronize()
    log("model loaded in %.1fs (naive — full GPU, no partition)"
        % (time.perf_counter() - t0))

    gold = {g["prompt"]: g for g in json.load(open(GOLD_JSON))["prompts"]}
    gold_logits = torch.load(GOLD_LOGITS, map_location="cuda")
    n_prompts = len(gold)

    # ---- teacher-forced correctness gate (TFGATE) — identical to Arm B ----
    kl_pc = []
    for pr in range(n_prompts):
        g = gold[pr]
        plen = g["plen"]
        gen = gold_logits[pr].shape[0]
        full = torch.tensor([g["full_ids"]], device="cuda")
        with torch.no_grad():
            lo = model(input_ids=full).logits[0]
        seg = lo[plen - 1:plen - 1 + gen, :].float()
        lp = F.log_softmax(gold_logits[pr].float(), dim=-1)
        lq = F.log_softmax(seg, dim=-1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        km, kx = float(kl.mean()), float(kl.max())
        kl_pc.append({"prompt": pr, "kl_mean": km, "kl_max": kx})
        log("TFGATE prompt=%d kl_mean=%.6e kl_max=%.6e" % (pr, km, kx))
    kl_max_all = max(x["kl_max"] for x in kl_pc)

    def decode_round(pr, timed):
        """One KV-cached greedy decode of prompt pr. Returns (prefill_ms,
        [decode_ms]). Byte-identical to the PARTITION tenant's decode_round —
        the per-step timing path must not differ between arms."""
        g = gold[pr]
        plen = g["plen"]
        gen = gold_logits[pr].shape[0]
        cur = torch.tensor([g["full_ids"][:plen]], device="cuda")
        past = None
        prefill_ms = None
        decode_ms = []
        with torch.no_grad():
            for st in range(gen):
                torch.cuda.synchronize()
                ts = time.perf_counter()
                out = model(input_ids=cur, past_key_values=past,
                            use_cache=True)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - ts) * 1e3
                if st == 0:
                    prefill_ms = dt
                elif timed:
                    decode_ms.append(dt)
                past = out.past_key_values
                cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
        return prefill_ms, decode_ms

    decode_round(0, timed=False)          # warmup round (cold outlier)
    torch.cuda.synchronize()
    log("warmup round done")

    # ---- rendezvous ----
    print("READY %d" % os.getpid(), flush=True)
    if not NOSIGNAL:
        for line in sys.stdin:
            if line.strip() == "START":
                break
    log("START — %d rounds" % ROUNDS)

    rounds = []
    for r in range(ROUNDS):
        pr = r % n_prompts
        prefill_ms, decode_ms = decode_round(pr, timed=True)
        rounds.append({"round": r, "prompt": pr, "prefill_ms": prefill_ms,
                       "decode_ms": decode_ms})
        print("ROUND %d" % r, flush=True)
        mean_d = sum(decode_ms) / len(decode_ms) if decode_ms else 0.0
        log("round %d prompt %d  decode_steps=%d  mean=%.3fms"
            % (r, pr, len(decode_ms), mean_d))
        barrier(r, last=(r == ROUNDS - 1))

    all_decode = [x for rd in rounds for x in rd["decode_ms"]]
    res = {
        "tenant": TENANT, "pid": os.getpid(), "arm": "A-naive",
        "model": MODEL, "rounds": ROUNDS,
        "n_decode_steps": len(all_decode),
        "kl_per_prompt": kl_pc, "kl_max_all": kl_max_all,
        "kl_gate_pass": kl_max_all <= 0.1,
        "max_mem_allocated_mib": torch.cuda.max_memory_allocated() // (1 << 20),
        "rounds_detail": rounds,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(res, f)
    print("DONE", flush=True)
    log("DONE  decode_steps=%d  kl_max=%.6e  kl_gate=%s  peak_mem=%d MiB"
        % (len(all_decode), kl_max_all, res["kl_gate_pass"],
           res["max_mem_allocated_mib"]))


if __name__ == "__main__":
    main()
