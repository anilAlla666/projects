#!/usr/bin/env python3
"""TPW re-test — naive-concurrent baseline worker (THROWAWAY).

One independent tenant: loads the model privately and runs `generate()` on
the gold prompts with NO substrate — no batching, no kmod, no libcipher.
N of these run concurrently = the naive-N-concurrent baseline that
cross-tenant batching is measured against.

Matches the batched executor's model path EXACTLY (dtype=float16, sdpa,
greedy generate, MAX_NEW) so the only difference batched-vs-naive is the
batch fusion itself.

Env: WL_MODEL GOLD_JSON OUT_JSON GO_FILE [TENANT] [MAX_NEW=128]
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["WL_MODEL"]
GOLD = os.environ["GOLD_JSON"]
OUT = os.environ["OUT_JSON"]
GO = os.environ["GO_FILE"]
TENANT = os.environ.get("TENANT", "n0")
MAX_NEW = int(os.environ.get("MAX_NEW", "128"))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda()
    m.train(False)                          # inference mode
    gold = json.load(open(GOLD))["prompts"]

    # warmup (excluded from timing)
    g0 = gold[0]
    ids = torch.tensor([g0["full_ids"][:g0["plen"]]], device="cuda")
    with torch.no_grad():
        m.generate(input_ids=ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    print("LOADED", flush=True)

    while not os.path.exists(GO):           # rendezvous — all tenants start together
        time.sleep(0.02)

    t0 = time.perf_counter()
    total = 0
    with torch.no_grad():
        for g in gold:
            ids = torch.tensor([g["full_ids"][:g["plen"]]], device="cuda")
            out = m.generate(input_ids=ids, max_new_tokens=MAX_NEW,
                             do_sample=False)
            total += int(out.shape[1] - ids.shape[1])
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    json.dump({"tenant": TENANT, "tokens": total, "decode_wall_s": wall},
              open(OUT, "w"))
    print("DONE tokens=%d wall=%.2f" % (total, wall), flush=True)


if __name__ == "__main__":
    main()
