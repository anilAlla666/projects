# Five-pillar architecture — honest measured outcomes

All measurements vs **eager-mode baseline at the same batch size**, freshly
measured at PREFILL_LEN=1024 with NaN sentinels and decoded-token sanity:

| B | eager tps | eager watts | **eager tok/W** |
|--:|---------:|----------:|----------------:|
| 1 | 48.21    | 215.5     | **0.2236** |
| 8 | 389.08   | 381.8     | **1.0190** |
| 32 | 685.38  | 474.7     | **1.4438** |
| 64 | 793.50  | 521.9     | **1.5205** |

10× targets: B=1 → 2.236, B=8 → 10.19, B=32 → 14.44, B=64 → 15.21.

## Pillar-by-pillar honest verdicts

| Pillar | Best Δtok/W vs same-B eager | Verdict |
|---|---:|---|
| 1A baseline | 0% | reference |
| 1B eager CIPHER (INT4+fused) | +5.7% B=1, −3 to −6% B≥32 | small B=1 win; intercept overhead dominates at high B |
| 1C idle injection sweep (0→5000 µs) | clock pinned at 1980 MHz — premise EMPIRICALLY FALSE on H100 | rules out a class of designs |
| 2A megakernel B=1 | +11.2% | works |
| 2C megakernel M>1 | −60 to −73% | GEMV-style kernel without tensor cores — research-grade fix needed |
| 3 V3 KV at B=1 | −1.9% | dequant cost exceeds bandwidth saving in eager |
| 3 V3 KV at B>1 | crash | layout requires multi-batch redesign (cache is `(token,head)` flat; FA expects `(B,head,T,dim)`) |
| 4 adaptive compute depth | quality FAIL: KL = 9.96 nats per token at 15% skip-rate | Mistral-7B is dense — Step 4A showed every layer has rel-delta > 0.26; needs distillation/finetune |
| 5B clock locking (the win) | **+35-46% across all B** | the only meaningful lever |

## Final combined result (verified-positive levers only)

| Plan | B | clk MHz | tps | watts | tok/W | × eager | last token |
|------|--:|--------:|-----:|------:|------:|--------:|:---|
| safe | 1 | 1000 | 51.2 | 141.2 | **0.3629** | **1.62×** | `'Energy'` |
| safe | 8 | 1600 | 399.2 | 354.8 | 1.1250 | 1.10× | `'is'` |
| safe | 32 | 1980 | 700.3 | 490.2 | 1.4286 | 0.99× | `'.'` |
| safe | 64 | 1980 | 787.6 | 518.7 | 1.5184 | 1.00× | `'.'` |
| aggr | 1 | 1000 | 51.4 | 141.6 | 0.3629 | 1.62× | `'Energy'` |
| aggr | 8 | 1200 | 328.4 | 237.9 | **1.3805** | **1.35×** | `'is'` |
| aggr | 32 | 1000 | 425.3 | 219.4 | **1.9383** | **1.34×** | `'means'` |
| aggr | 64 | 1000 | 467.7 | 224.6 | **2.0824** | **1.37×** | `'make'` |

NaN sentinels passed every row. INT4 GEMV fired 39168× and megakernel
fired 9792× at B=1 (counter-verified). At B≥8 the M=1 paths are
correctly dormant and fp16 cuBLAS handles every linear (counter shows 0
INT4 / 0 mega, 27776-68544 fallbacks).

## What this is and what this isn't

**Best result: 1.62× tok/W at B=1, 1.37× at B=64. Not 10×.**

The 10× target is not reachable on this hardware with this set of levers
because:
1. **Mistral-7B is dense.** Step 4A directly measured per-layer
   contribution; no layer is near-identity. Adaptive layer skipping
   produces incoherent text (KL=9.96 nats per token).
2. **H100 DVFS doesn't engage on µs gaps.** Idle injection sweep
   confirmed clock pinned at 1980 MHz at 0–5000 µs gap. Pillar 1's idle
   premise was empirically wrong on this hardware. Power management
   requires explicit clock locking (Pillar 5).
3. **INT4 GEMV is roughly cuBLAS-parity on tps, not faster.** The win
   is power (~8% drop) at B=1 only; Marlin/wgmma can't beat cuBLAS on
   H100 sm_90.
4. **Continuous batching's 14× win** (kickass_results.json from earlier)
   compares aggregate-server-throughput tok/W vs single-user-baseline
   tok/W — that's the right number for a serving cost model but it's
   different physics from "10× same-B same-context."

## What WOULD deliver 10× (research path; requires work beyond this session)

| Lever | Expected ×tok/W | Cost |
|---|---:|---|
| Speculative decoding w/ 1B draft model | 2-3× at B=1 | requires draft model + verifier |
| AWQ-calibrated INT4 + tensor-core INT4 GEMM (Marlin v2 or wgmma) | 1.5-2× | calibration + new kernel |
| FP8 end-to-end (Hopper-native) | 1.5-2× | full FP8 retraining loop |
| Pruned/distilled 4B Mistral | 1.5-2× | training compute |
| Continuous batching (server) | 5-10× | scheduler, paged KV |

Compound ceiling: ~10-30× tok/W vs B=1 eager. Realistic 2-6 month effort.
The 1.62× shipped today is the verified floor of that ramp.

## What's reproducible right now

```
cd /home/ubuntu/op31-prod-fix

# Eager baseline
python3 p1a_eager_baseline.py
# → 0.224 tok/W at B=1, 1.02 at B=8, 1.44 at B=32, 1.52 at B=64

# Pillar 5 sweep (the win)
python3 p5b_clock_sweep.py
# → 7×4 = 28 (clock,batch) measurements with NaN+token sanity

# Final combined
LD_PRELOAD=$PWD/libcipher_hook.so:/usr/lib/x86_64-linux-gnu/libcuda.so \
  CIPHER_SUBSTITUTE_V2=on CIPHER_WEIGHT_COMPRESS=on CIPHER_FUSION_KERNELS=on \
  python3 p_final_combined.py
# → 1.62× B=1, 1.35× B=8, 1.34× B=32, 1.37× B=64 (aggressive plan)
```

## Files

| Pillar | Code | Data |
|---|---|---|
| 1A | `p1a_eager_baseline.py` | `p1a_eager_baseline.json` |
| 1B | `p1b_eager_cipher.py` | `p1b_eager_cipher.json` |
| 1C | `p1c_idle_sweep.py` | `p1c_idle_sweep.json` |
| 2A-C | `p2a_megakernel_eager.py` | `p2a_megakernel_eager.json` |
| 3 | `p3_kv_v3_eager.py` | `p3_kv_v3_eager.json` |
| 4A-B | `p4ab_probe.py` | `p4ab_probe.json` |
| 4C-D | `p4cd_adaptive_depth.py` | `p4d_quality.json` |
| 5B | `p5b_clock_sweep.py` | `p5b_clock_sweep.json`, `p5_optimal_clocks.json` |
| FINAL | `p_final_combined.py` | `p_final_combined.json` |
