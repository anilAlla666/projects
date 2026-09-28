"""W.4b.6 deconfounding control (advisor): in-process B=N decode with N idle
co-resident CUDA contexts present — matching the cross run's context count
(executor active + N tenants idle) but with NO cross-process transport and a
SINGLE active decoder. Decomposes cross/naive into:
  batching lever   = inproc_alone / naive        (genuine density, weight-share)
  architecture tax = cross / inproc_alone         (cost of N GPU processes)
This run measures inproc_WITH_idle_contexts; if it ~= cross, the tax IS the
co-resident-context cost (not executor-specific). Power-windowed, same GEN/PLEN.
"""
import os
import sys
import json
import time

MODEL = os.environ["WL_MODEL"]
N = int(os.environ["N_IDLE"])          # number of idle context holders
NB = int(os.environ["NB"])             # in-proc batch size (= gate N)
GEN = int(os.environ["GEN_LEN"])
PLEN = int(os.environ["PLEN"])
SENT = os.environ["SENTINEL"]
RESULT = os.environ["RESULT_JSON"]
DEV = "cuda"
PROMPTS = [
    "The history of computing spans several distinct technological eras that",
    "Renewable energy adoption worldwide has accelerated rapidly over the recent",
    "Modern distributed systems must carefully balance consistency availability and partition",
    "Advances in materials science have enabled lighter stronger and more durable",
    "The global supply chain experienced unprecedented disruption during the pandemic which",
    "Machine learning models trained on large corpora can exhibit surprising emergent",
    "Urban planners increasingly rely on data driven simulations to forecast traffic",
    "Ocean currents play a critical role in regulating the planet climate by",
]


def idle_holder(ready_w, go_r):
    import torch
    x = torch.ones(4 * 1024 * 1024, device=DEV)   # init a real CUDA context
    x.mul_(1.0001); torch.cuda.synchronize()
    os.write(ready_w, b"r")                        # signal ready
    os.read(go_r, 1)                               # block until parent says exit
    sys.exit(0)


def main():
    # spawn N idle context holders, each with its own pipe pair
    ready_r, ready_w = os.pipe()
    go_pipes = []
    kids = []
    for _ in range(N):
        gr, gw = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(ready_r); os.close(gw)
            idle_holder(ready_w, gr)
        os.close(gr)
        go_pipes.append(gw); kids.append(pid)
    os.close(ready_w)
    # parent loads model + builds batch
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

    def enc(p):
        t = tok(p, return_tensors="pt").input_ids[0]
        if t.shape[0] >= PLEN:
            return t[:PLEN]
        padv = tok.pad_token_id or tok.eos_token_id
        return torch.cat([t, torch.full((PLEN - t.shape[0],), padv, dtype=t.dtype)])
    ids = torch.stack([enc(PROMPTS[i % len(PROMPTS)]) for i in range(NB)], 0).cuda()
    with torch.no_grad():
        m(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    # wait for all idle holders to be resident
    got = 0
    while got < N:
        got += len(os.read(ready_r, N - got))
    # power-windowed in-proc B=NB decode
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True); past = out.past_key_values
        nt = out.logits[:, -1, :].argmax(-1); cur = ids.shape[1]
        torch.cuda.synchronize()
        with open(SENT, "w") as s:
            s.write("DECODE_START %.3f\n" % time.time())
        t0 = time.perf_counter()
        for _ in range(1, GEN):
            out = m(input_ids=nt.unsqueeze(-1), past_key_values=past, use_cache=True,
                    cache_position=torch.tensor([cur], device=DEV))
            past = out.past_key_values; nt = out.logits[:, -1, :].argmax(-1); cur += 1
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        with open(SENT, "a") as s:
            s.write("DECODE_END %.3f\n" % time.time())
    dtok = NB * (GEN - 1)
    json.dump({"mode": "inproc_with_%d_idle_contexts" % N, "B": NB, "gen_len": GEN,
               "decode_wall_s": round(wall, 4), "decode_tokens": dtok,
               "decode_tok_s": round(dtok / wall, 3),
               "peak_hbm_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
              open(RESULT, "w"), indent=2)
    print("IDLECTX inproc B=%d with %d idle contexts decode_tok/s=%.1f wall=%.3fs"
          % (NB, N, dtok / wall, wall), file=sys.stderr, flush=True)
    for gw in go_pipes:
        os.write(gw, b"x")
    for pid in kids:
        os.waitpid(pid, 0)


if __name__ == "__main__":
    main()
