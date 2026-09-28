"""CP 4.6.6 — real concurrent decode load test (Option A real-decode arm).

16 tenants (HBM ceiling at 32K fp16: (80-14)/4 = 16), shared Mistral-7B fp16,
16 resident 32K StaticCaches + per-tenant CUDA streams. A single driver loop
issues one decode step per tenant per round, round-robin — this models the
§3b time-shared GPU honestly and avoids the GIL host-enqueue artifact a
16-thread harness would introduce. Run under
CUDA_INJECTION64_PATH=libcipher_rt.so so the CIPHER substrate is live.

Measures (item-6 list): per-tenant tok/s (median/p25/p75); per-tenant p99
inter-token latency (gate D4: p99 <= 2x the §3b roofline mean ITL); peak HBM
(the 16-resident-32K-cache capacity test); aggregate decode throughput.

Roofline (memo §3b, 32K fp16): ~112 realistic decode steps/s aggregate; at
16 tenants -> ~7.0 tok/s/tenant -> mean ITL ~143 ms -> p99 gate <= 286 ms.
"""
import os, time, json, statistics as st

# Caching-allocator fragmentation, not raw HBM physics, capped the first
# Llama-3.1-8B run at 6 tenants (78 GiB reserved / 53 GiB allocated). Fix in
# source per regression discipline: expandable segments removes the
# fragmentation; logits_to_keep=1 (below) removes the ~8.4 GB full-prompt
# logits transient that spiked each prefill.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TENANTS  = int(os.environ.get("T466_TENANTS", "16"))
CTX      = int(os.environ.get("T466_CTX", "32768"))
DECODE_N = int(os.environ.get("T466_DECODE", "64"))
MODEL    = os.environ.get("T466_MODEL", "/home/ubuntu/models/Llama-3.1-8B")
OUT      = os.environ.get("T466_OUT",
                          "/home/ubuntu/cipher-fusion-evidence/cp_4_6_5_6/"
                          "t466_decode_result.json")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

print(f"[t466] loading {os.path.basename(MODEL)} fp16 (shared) ...", flush=True)
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                             device_map="cuda:0").eval()
WEIGHT_BYTES = torch.cuda.memory_allocated()
print(f"[t466] model loaded {time.perf_counter()-t0:.1f}s; "
      f"weights HBM={WEIGHT_BYTES/2**30:.1f} GB", flush=True)

# Per-token KV-cache footprint (fp16): 2 (K+V) * layers * kv_heads * head_dim
# * 2 B. Used for the §3b roofline — recomputed N-specifically below, not the
# stale Mistral-N=16 hardcode.
_cfg = model.config
_head_dim = getattr(_cfg, "head_dim", None) or (
    _cfg.hidden_size // _cfg.num_attention_heads)
KV_PER_TOKEN = (2 * _cfg.num_hidden_layers *
                _cfg.num_key_value_heads * _head_dim * 2)
print(f"[t466] KV/token={KV_PER_TOKEN/1024:.0f} KiB  "
      f"KV/tenant@{CTX}={CTX*KV_PER_TOKEN/2**30:.2f} GB", flush=True)

base = tok("Energy efficiency means doing more useful work per watt. "
           "The future of GPU computing is to make every joule count. ",
           return_tensors="pt").input_ids
reps = (CTX // base.shape[1]) + 1
prompt_ids = base.repeat(1, reps)[:, :CTX].to("cuda:0")
print(f"[t466] prompt: {prompt_ids.shape[1]} tokens", flush=True)


def main():
    streams, caches, nxt, itl, ok = [], [], [], [], []
    # allocate + prefill each tenant; on OOM, stop and report the ceiling
    for tid in range(TENANTS):
        try:
            s = torch.cuda.Stream(device="cuda:0")
            with torch.cuda.stream(s), torch.no_grad():
                c = StaticCache(config=model.config, max_batch_size=1,
                                max_cache_len=CTX + DECODE_N + 8,
                                device="cuda:0", dtype=torch.float16)
                out = model(prompt_ids, past_key_values=c, use_cache=True,
                            logits_to_keep=1)
                tkn = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            streams.append(s); caches.append(c); nxt.append(tkn)
            itl.append([]); ok.append(tid)
            print(f"[t466] tenant {tid} prefilled  "
                  f"HBM={torch.cuda.memory_allocated()/2**30:.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"[t466] tenant {tid} OOM — HBM ceiling at {tid} tenants: "
                  f"{exc}", flush=True)
            break
    N = len(ok)
    print(f"[t466] {N} tenants resident; round-robin decode {DECODE_N} steps",
          flush=True)

    WARMUP = 4
    with torch.no_grad():
        # warmup rounds — first-launch / graph-capture cost excluded
        for _ in range(WARMUP):
            for i in range(N):
                with torch.cuda.stream(streams[i]):
                    out = model(nxt[i], past_key_values=caches[i],
                                use_cache=True, logits_to_keep=1)
                    nxt[i] = out.logits[:, -1:].argmax(-1)
                streams[i].synchronize()
        # measured window
        run_t0 = time.perf_counter()
        for step in range(DECODE_N):
            for i in range(N):
                ts = time.perf_counter()
                with torch.cuda.stream(streams[i]):
                    out = model(nxt[i], past_key_values=caches[i],
                                use_cache=True, logits_to_keep=1)
                    nxt[i] = out.logits[:, -1:].argmax(-1)
                streams[i].synchronize()
                itl[i].append(time.perf_counter() - ts)
    wall = time.perf_counter() - run_t0
    peak_hbm = torch.cuda.max_memory_allocated() / 2**30

    per_tps = sorted(len(x) / sum(x) for x in itl if sum(x) > 0)
    all_itl = sorted(v for x in itl for v in x)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None

    # §3b roofline — N-specific, computed from measured geometry (not the
    # stale Mistral-N=16 0.143 s hardcode). B=1 decode is HBM-bandwidth-
    # bound: the serviced tenant reads weights + its own KV once per step.
    HBM_BW, EFF = 3.35e12, 0.6
    bytes_per_step = WEIGHT_BYTES + CTX * KV_PER_TOKEN
    agg_steps_s = HBM_BW * EFF / bytes_per_step
    roofline_mean_itl = N / agg_steps_s   # mean ITL at N-way time-share
    p99 = pct(all_itl, 0.99)
    doc = {
        "cp": "4.6.6", "arm": "real-decode (Option A)",
        "tenants_target": TENANTS, "tenants_resident": N,
        "hbm_ceiling_hit": N < TENANTS,
        "context_tokens": CTX, "decode_tokens": DECODE_N,
        "wall_s": round(wall, 1), "peak_hbm_gb": round(peak_hbm, 2),
        "per_tenant_tok_s": {
            "median": round(st.median(per_tps), 3) if per_tps else None,
            "p25": round(pct(per_tps, 0.25), 3) if per_tps else None,
            "p75": round(pct(per_tps, 0.75), 3) if per_tps else None},
        "itl_ms": {
            "median": round(pct(all_itl, 0.5) * 1000, 1) if all_itl else None,
            "p99": round(p99 * 1000, 1) if p99 else None},
        "roofline_3b": {
            "bytes_per_step_gb": round(bytes_per_step / 2**30, 2),
            "agg_steps_s": round(agg_steps_s, 1),
            "per_tenant_tok_s_pred": round(agg_steps_s / N, 2),
            "mean_itl_ms": round(roofline_mean_itl * 1000, 1)},
        "p99_gate_d4": {
            "roofline_mean_itl_ms": round(roofline_mean_itl * 1000, 1),
            "budget_ms": round(roofline_mean_itl * 2 * 1000, 1),
            "pass": bool(p99 and p99 <= 2 * roofline_mean_itl)},
        "aggregate_tok_s": round(sum(per_tps), 2) if per_tps else 0,
    }
    json.dump(doc, open(OUT, "w"), indent=2)
    print(f"[t466] DONE N={N} wall={wall:.1f}s peak_hbm={peak_hbm:.1f}GB "
          f"per-tenant_tok/s_med={doc['per_tenant_tok_s']['median']} "
          f"p99_itl={doc['itl_ms']['p99']}ms gate={doc['p99_gate_d4']['pass']}",
          flush=True)
    print(f"[t466] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
