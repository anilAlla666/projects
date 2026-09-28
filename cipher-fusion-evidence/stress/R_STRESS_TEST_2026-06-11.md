> **SUPERSEDED by R_STRESS_TEST_V2_2026-06-11.md** — a verification panel found 10 material confounds; v2 corrects them. Notably v1 Result 4 (spec-decode "breaks under load") was a cudagraph artifact and is RETRACTED; v1 TPW 2.11x was under-saturated (true 1.96x).

# STRESS TEST — do the TPW / MBU / spec-decode claims hold under saturated realistic load? (2026-06-11)

Directive: stress-test the green-scorecard claims and see if they hold. Method: real vLLM, Mistral-7B, **saturated
continuous batching** (NREQ=128 concurrent requests, MIXED workload — chat/instruct/RAG/code, variable prompt AND
output lengths) instead of the fixed-batch/batch-1 synthetic microbenchmarks; power sampled during the run; +
a load sweep (NREQ 1→128) and a measured perplexity (the previously-unmeasured quality caveat). Anchor `2edba0d2`
unchanged; substrate NOT loaded (0/15 runs — still measuring vLLM-native physics, as flagged); power reset.

## Result 1 — TPW 2× : **HOLDS under saturated load (with FP8, not 4-bit)**
tok/W under saturated mixed load, vs fp16@700W baseline (18.08 tok/W):

| config | agg tok/s | power | tok/W | **× baseline** |
|---|---|---|---|---|
| fp16 @ 700W (baseline) | 9479 | 524W | 18.08 | 1.00× |
| fp16 @ 300W (DVFS only, lossless) | 7356 | 297W | 24.80 | 1.37× |
| 4-bit gptq @ 300W (original claim's config) | 8534 | 279W | 30.56 | 1.69× |
| **FP8 @ 300W (composed)** | **11305** | **296W** | **38.18** | **2.11×** |
| FP8 @ 700W | 12906 | 423W | 30.51 | 1.69× |

- **The 2× HOLDS — but the right config is FP8@300W (2.11×), not the gptq@300W the microbench used.** Under
  saturated load 4-bit GPTQ goes **compute-bound** (dequant overhead), so its throughput drops *below* fp16
  (0.90×) and its tok/W gain falls to 1.69×. FP8 handles high batch far better (12906 tok/s) and recovers >2×.
- DVFS-alone is **1.37× under load** (lossless), matching the 1.39× microbench — robust.
- So the original "2.00× via gptq@300" was a batch-16 artifact (gptq drops to 1.69× saturated); **FP8@300 gives a
  stronger, real 2.11× under saturated load, at higher throughput than the fp16 baseline.**

## Result 2 — Quality (the unmeasured caveat) : **now MEASURED**
Perplexity on held-out factual/narrative/technical text (teacher-forced, n=185 tokens):

| precision | PPL | vs fp16 |
|---|---|---|
| fp16 | 3.393 | — |
| **FP8** | 3.405 | **+0.38%** |
| 4-bit GPTQ | 3.544 | **+4.46%** |

FP8's quality cost is small (+0.38%); 4-bit's is **12× larger (+4.46%)**. Combined with Result 1, **FP8 dominates
4-bit on every axis** (under-load throughput, tok/W, AND quality) — the TPW/MBU wins should ride on FP8, not 4-bit.

## Result 3 — MBU / FP8 decode throughput : **holds, but erodes under load**
FP8 vs fp16 decode throughput: **1.53× at batch 1 → 1.36× at saturated load** (12906/9479). Positive lever, smaller
under saturation (FP8's edge is memory-bound; at high batch decode is partly compute-bound). Quality +0.38%.

## Result 4 — Spec-decode 2×+ : **DOES NOT HOLD — it is a single-user latency feature only**
FP8 ⊗ ngram-spec, throughput ratio vs FP8-alone, across load (same harness, mixed workload):

| concurrency (NREQ) | spec ratio | |
|---|---|---|
| 1 (single user) | **3.40×** | HELPS (mixed wl; was 8.6× on verbatim-copy) |
| 4 | 0.78× | **HURTS** |
| 16 | 1.01× | neutral |
| 128 (saturated) | **0.61×** | **HURTS −39%** |

**The 2–8.6× was entirely a batch-1 artifact.** At any realistic concurrency (NREQ≥4) spec-decode is
neutral-to-negative: the draft+verify compute competes with the batched real work once the GPU is busy. Spec-decode
is a **single-user / low-load LATENCY optimization, NOT a server-throughput lever** — the earlier "decode 2×" claim
does not survive a loaded server.

## Result 5 — MBU-itself 2× (custom FP8 GEMV kernel) : still UNSUCCESSFUL (prior, unchanged)
The kernel route to 74% MBU was already shown to fail (cutlass caps 54%/1.53×; my best custom 0.94×). Stress
testing does not change it.

## Honest scorecard after stress
| claim | microbench | **under saturated realistic load** | holds? |
|---|---|---|---|
| TPW 2× | 2.00× (gptq, batch 16) | **2.11× (FP8@300W)** | ✅ YES (with FP8) |
| DVFS lossless | 1.39× | 1.37× | ✅ YES |
| MBU / FP8 decode | 1.53× (batch 1) | 1.36× | ⚠️ smaller but positive |
| Spec-decode 2×+ | 8.6× (batch-1, verbatim) | 0.61–1.01× (NREQ≥4) | ❌ NO (single-user only) |
| MBU-itself 2× | — | — | ❌ NO (kernel ceiling) |
| Quality cost | hand-waved | FP8 +0.38% / 4-bit +4.46% | now measured |

**Net:** TPW 2× survived stress (and improved — FP8 beats the originally-claimed 4-bit on throughput, efficiency,
AND quality). The spec-decode 2× broke under load. The MBU memory lever holds but shrinks. Quality is now real.

## Remaining gaps (honest — what stress STILL does not cover)
1. **Substrate not loaded** (0/15) — still vLLM-native physics, not CIPHER's actuators delivering it in-loop.
2. **Batch-submit, not Poisson arrival** — NREQ=128 submitted at once = a saturation point, not a real arrival
   distribution / latency-SLO test. p50/p99 latency under arrival rate not measured.
3. **Single GPU** — no multi-tenant/fleet, no NCCL.
4. **One model, short-to-mid context** — no long-context (where KV-cache, not weights, dominates the byte budget).

## Integrity
Anchor `2edba0d2` entry==exit; substrate not loaded (`anchor_loaded:false` ×15); power reset to 700W; no leftover
processes. Saturated runs generated ~19k tokens in ~2s (GPU saturated, mean power 420–525W confirms it). PPL
teacher-forced via prompt_logprobs, same 3 texts across precisions. Best-effort power sampling ~10Hz during the
timed window. Load sweep + saturated runs in the SAME harness for apples-to-apples.
