# V0 GATE-1 increment g1.2x: bnb NF4 density into the engine -- DENSITY REAL + CAPTURE-SAFE (non-SWA)

**2026-06-03. POSITIVE verdict (recovers the axis g1.2 did NOT deliver). DENSITY: bnb NF4 stays TRULY packed at
runtime AND in-engine -- ~5.6GB LIVE per 7-8B model (weights + captured static-KV graph + StaticCache, all alive),
with nvml tracking live to within 0.3GB (NO retained-fp16 tail, unlike compressed-tensors' 17GB). 12 distinct-or-replica
nf4 models co-resident in ONE process at 38.7GB. CORRECTNESS: nf4-graph vs nf4-eager KL=0 (capture-safe on NON-SWA
models).** NO `.so` change; g1.2x added ONLY probes (inc-1+inc-2 modules + `cipher_inc4.py` BYTE-IDENTICAL since g1.1);
anchor 1f305ce6 (May-27) / deployed `libcipher_rt.so` (Jun-01) UNCHANGED. `CIPHER_RT_DISABLE_AUTO_INIT=1`. Probes:
`pager_g12x_nf4_capture.py`, `pager_g12x_density.py`, `pager_g12x_coresident.py`.

## The g1.2 trap this increment had to avoid

g1.2 (compressed-tensors W4A16) looked dense -- 5.26GB packed pager region -- but the forward **materialized ~10GB of
fp16 working weights and RETAINED them** (post-`empty_cache` = 17.3GB, even with `run_compressed: true` forced). Captured
int4 ≈ fp16, ~4 co-reside, NO density gain. So g1.2x's gate demanded DENSITY-REAL measured at runtime FIRST (live HBM
after forward + capture, not the load-time packed number), before any co-residence count.

## STEP 1 -- nf4 stays packed at runtime (decisively unlike compressed-tensors)

`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`. Single-model
clean HBM (Qwen2-7B): packed footprint 5.07GB; forward **+0.1GB** (in-kernel fused dequant -- NO fp16 materialization);
**post-`empty_cache` = 6.3GB**. Contrast g1.2 compressed-tensors: forward +9.7GB, post-empty_cache 17.3GB. nf4 is the
real fused-int4 path on this stack.

## STEP 2 -- nf4 CAPTURES into the engine's static-KV graph, KL=0 (capture-safe on non-SWA)

Same-precision oracle (nf4-graph vs nf4-EAGER, not fp16-bit-exact, per inc-1 framing). `pager_g12x_nf4_capture.py`,
Qwen2-7B, N=64: **64/64 KL=0** -- bnb's fused-dequant kernel does NOT fire capture-illegal ops (the risk was that, like
act-order g_idx / Koopman cusolver, it might do a host-side op inside capture; it does not). Confirmed KL=0 on Llama-3.1-8B,
Llama-3.2-1B, TinyLlama. **Mistral-7B nf4 DIVERGES (4/32)** -- the SWA (sliding-window) capture binding term, the SAME
separate issue excluded at g1.2 / inc-3, NOT an nf4 failure. So: **nf4 capture-safe on NON-SWA models** (the same model
set the engine already serves).

## STEP 3 -- density is REAL in-engine (the advisor's reconciling measurement)

The first co-residence probe (`pager_g12x_density.py`) showed a confusing 7.8->20GB/model spread -- which would have meant
NO density (graph/cache eats it, ~4 fit). The reconciling test (advisor): keep model+captured-graph+cache ALIVE per model
(engine-representative -- the engine keeps each model's graph) and print `torch.cuda.memory_allocated` (**LIVE**) alongside
nvml, because `empty_cache` can only reclaim UNUSED cached blocks, never live memory. `pager_g12x_coresident.py`, 12 models
(Qwen2-7B / Llama-3.1-8B / Llama-3.2-1B / TinyLlama, cycled):

```
nf4[0] Qwen2-7B:    LIVE=5.5GB (+5.5/model) | nvml=5.8GB
nf4[1] Llama-3.1-8B: LIVE=11.2GB (+5.7/model) | nvml=11.4GB
nf4[2] Llama-3.2-1B: LIVE=12.2GB (+1.0/model) | nvml=12.4GB
nf4[3] TinyLlama:    LIVE=12.9GB (+0.8/model) | nvml=13.2GB
...
nf4[11] TinyLlama:   LIVE=38.7GB | nvml=39.0GB | free=39.5GB
[RESULT] 12 nf4 models held (model+graph+cache ALIVE) | LIVE=38.7GB | nvml=39.0GB
```

Decisive: **LIVE (38.7GB) tracks nvml (39.0GB) to within 0.3GB** -- there is NO hidden retained cost. The "20GB/model" in
the first probe was eager-burst reclaimable caching transients (exactly the artifact the live measurement rules out).
Per-model LIVE footprint: **7-8B nf4 ~5.6GB** (weights + captured graph + StaticCache), 1B ~0.8-1.0GB. The captured graph
+ cache add negligibly over the packed weights.

**DENSITY NUMBER: ~13 7B-class nf4 models fit/75GB (5.6GB each) vs fp16 ~5 (inc-3) / compressed-tensors ~4 (g1.2) -- ~2.5x.**
12 distinct-or-replica nf4 models demonstrated co-resident in one process. The axis g1.2 missed, g1.2x delivers.

## Honest caveats

- **nf4 capture-safe on NON-SWA models only** -- Mistral SWA diverges (separate known binding term, excluded everywhere).
- **The co-residence probe used `device_map="cuda"`, NOT the pager**, deliberately, to measure HBM-total density without the
  routing confound. nf4-THROUGH-the-pager is already proven at inc-1 ([[cipher-pager-int4-build]]: 3 distinct 7-8B nf4
  co-resident over pager regions, KL=0). `pager_g12x_density.py` showed a pager-region routing FALLBACK for models 3+
  (region footprint read 0.00 -- a `use_mem_pool`/`MemPool`-reuse probe bug; those loaded via default allocator, not an
  nf4 limit). The DENSITY claim is HBM-total and valid regardless of routing.
- Replicas share a checkpoint on disk but are independent HBM instances; only 4 DISTINCT checkpoints exist here (the
  availability ceiling), so "12 co-resident" = 4 distinct x 3 -- the FOOTPRINT (5.6GB/7B) is the transferable number.

## Gate status (Mem #11) + framing

1. **DENSITY-REAL FIRST: PASS** -- nf4 ~5.6GB LIVE/7B in-engine (live==nvml, no retained fp16), ~13 fit vs fp16 ~5.
2. **CORRECTNESS: PASS** -- nf4-graph vs nf4-eager KL=0 (capture-safe, non-SWA).
3. NON-REGRESSION: probes only; inc-1..inc-4 + g1.1 + g1.2 unaffected; OFF byte-identical; anchor UNCHANGED; GPU->0.

**FRAMING:** g1.1 (DVFS) delivered the ENERGY axis (1.53x tok/W). g1.2 (Marlin compressed-tensors) delivered int4
CORRECTNESS/capture-safety but NOT density (retained fp16). g1.2x (bnb NF4) **recovers the DENSITY axis** -- truly packed,
capture-safe, ~2.5x co-residence -- the first density win composed into the engine path. Together g1.1+g1.2x = both the
energy and density actuators compose capture-safe over the multi-model engine.

## STOP for g1.3 (Koopman probe-and-report) decision

g1.2x closed: bnb NF4 density REAL + capture-safe (non-SWA). Remaining gate-1: g1.3 Koopman (cusolver EDMD,
almost-certainly engine-incompatible per inc-1/g1.2 act-order -- a probe-and-honestly-report increment). Anil's call.
Related: [[cipher-gate1-g12-marlin-int4]], [[cipher-gate1-g11-dvfs-into-engine]], [[cipher-pager-int4-build]].
