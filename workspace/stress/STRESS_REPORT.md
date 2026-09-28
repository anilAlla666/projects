# CIPHER stress test — Llama-3.1-8B fp16, FP8 + 1200 MHz lock

**Pod**: 2× H100 80GB HBM3, driver 580.105.08, cudart 12.4
**Stack**: torch 2.6.0+cu124, libcipher_hook.so + libcipher_rt.so (build May 1)
**Clock**: `nvidia-smi -lgc 1200` on both GPUs
**CIPHER env**: `CIPHER_FP8_COMPUTE=on`, `CIPHER_SUBSTITUTE_V2=on`,
`CIPHER_FUSION_KERNELS=on`, fusion patches on `LlamaRMSNorm.forward` /
`LlamaMLP.forward` (mirroring `step9_llama70b.py`).
**Date**: 2026-05-02

## Headline

| Test | CIPHER | Baseline | Verdict |
|---|---|---|---|
| 1 — 10 K-token endurance | gen=10000, **all output `!!!!`** | gen=10000, coherent | **FAIL (correctness)** |
| 2 — 8-client × 5 min | **6/8 alive, 0/8 coherent**, 99.8 tok/s, 0.233 tok/W | 8/8 alive, 8/8 coherent, 135.9 tok/s, 0.333 tok/W | **FAIL (OOM + correctness)** |
| 3 — determinism @ T=0 | intra: 100/100 match; inter vs baseline: divergent at token 0 (every output is `[0,0,0,…]`) | intra: 100/100 match; output coherent | **FAIL (perturbs correctness)** |
| 4 — 1000-call leak (ran 200) | front-loaded +3 084 MiB at call 1 then **0 leak** through 200 calls | +14 MiB total over 200 calls | PASS (no per-call leak) |

CIPHER is **broken on Llama-3.1-8B in this stack**. Every model output
under FP8 + fusion is the token id 0 (`!`). Throughput is **24 % slower**
than baseline, tok/W is **17 % worse**, and 25 % of children OOM at
launch under sustained concurrent load.

The hook itself loads cleanly (Intercepts=306 290, GOT patches=10) — the
problem is in CIPHER's actuators producing wrong outputs, not in the hook
plumbing.

---

## Test 1 — endurance (single client, 10 000 tokens)

Single `model.generate(max_new_tokens=10_000, do_sample=False)`. Background
sampler on power + nvidia-smi every 150 ms; checkpoint nvidia-smi every 30 s.

|                       | **CIPHER**          | baseline            |
|---|---|---|
| generated             | **10 000 / 10 000** | 10 000 / 10 000     |
| crash                 | none                | none                |
| wall time             | **446.2 s**         | 338.6 s             |
| tok/s                 | **22.41**           | 29.53 (+31.8 %)     |
| mean watts (GPU 0)    | 273.1 W             | 299.0 W             |
| **tok/W**             | **0.0821**          | **0.0988** (+20 %)  |
| GPU 0 mem (pre→post)  | 19 743 → 24 373 MiB (+4 630) | 16 117 → 17 675 MiB (+1 558) |
| coherence @ tok 1     | `'!!!!!!!!!!!!!!!!'` | `' The future of GPU computing is to make every joule count. '` |
| coherence @ tok 5000  | `'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'` | (similar coherent text) |
| coherence @ tok 10000 | `'!!!!!!!!!!!!!!!'`  | (similar coherent text) |
| FP8 calls fired       | 288                 | n/a |
| FP8 weights quantized | 160 (of 224 linears) | n/a |
| fusion: rmsnorm       | 650 000             | n/a |
| fusion: silu_mul      | 320 000             | n/a |
| fusion: residual_add  | **0**               | n/a |

**Pass/Fail vs spec**:
- "No crash": ✅ both
- "Output coherent at tok 1, 5000, 10000": ❌ all three checkpoints garbage
  (token id 0 emitted continuously)
- "GPU memory doesn't grow": ⚠ baseline grows +1.5 GB (KV cache only);
  CIPHER grows +4.6 GB (KV cache + ~3 GB FP8/fusion buffers). Growth
  stabilises after ~30 s — no progressive leak past warm-up.
- "tok/W stable across run": ⚠ CIPHER per-60 s watt windows climb
  219 → 321 W during the run (+47 %). Some of that is contamination
  from a parallel run on GPU 1; the GPU-0-only nvidia-smi heartbeats
  still show watts climbing 220 → 320 W between minute 1 and 7.

**Test 1 verdict: FAIL (correctness).** Throughput is degraded
(24 % slower at 24 % more memory), but the headline is that the model
emits a single repeated token from the very first decoded position.

---

## Test 2 — sustained concurrent load (8 clients, 5 min)

8 child processes round-robin'd across GPU 0 / GPU 1, each looping
`generate(max_new_tokens=64)` until the 5-minute deadline. CIPHER children
inherit `LD_PRELOAD` and CIPHER env from the driver; baseline children
do not.

|                  | **CIPHER**       | baseline         |
|---|---|---|
| children alive at end | **6 / 8**       | 8 / 8           |
| CUDA errors      | **2** (children 4 & 5 — third spawn on each GPU) | 0 |
| total tokens     | 29 952           | 40 768          |
| total calls      | 468              | 637             |
| **agg tps**      | **99.8**         | 135.9 (+36 %)   |
| mean watts       | 428 W            | 409 W           |
| **tok/W**        | **0.233**        | **0.333** (+43 %) |
| max mem (sum 2 GPUs) | 145 428 MiB    | 136 068 MiB     |
| coherent outputs | **0 / 8**        | 8 / 8           |
| sample child 0 first text | `'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'` | `' The future of GPU computing is to make every joule count.  The future of GPU co'` |

The two CIPHER OOMs are deterministic — under CIPHER each child needs
~24 GB (16 GB fp16 weights + ~3 GB FP8 buffers + ~5 GB cuBLAS workspace
under CIPHER hook), so only 3 fit per 80 GB H100. Children 4 and 5 are
the third on their respective GPUs; they OOM during model load, fall
out, and the 4th-spawned children (6 and 7) succeed because the dead
ones freed memory back. Baseline children fit at ~17 GB each — 4 per
GPU, all 8 succeed.

**Test 2 verdict: FAIL.** No crash *of the harness* — but 25 % of
production clients fail to start, and 100 % of survivors emit garbage.

---

## Test 3 — determinism @ T=0 (100 runs same prompt)

Same prompt (`"The future of GPU computing is"`), 100×
`generate(max_new_tokens=64, do_sample=False)`. Hash each output token
sequence; count matches.

|                       | **CIPHER**         | baseline           |
|---|---|---|
| runs                  | 100                | 100                |
| **intra-run match**   | **100 / 100**      | 100 / 100          |
| unique hashes         | 1                  | 1                  |
| crashes               | 0                  | 0                  |
| elapsed               | 284.7 s            | 220.4 s            |
| first 12 tokens       | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | `[1618, 13, 578, 34661, 12175, 28298, 12175, 356, 7461, 15, 23501, 46879]` |

CIPHER is **bit-deterministic** within a run — every one of the 100
generations produces the same 64-token sequence — but that sequence is
all zeros (the `!` token), and it differs from the baseline at **token
0 and every subsequent token**.

**Test 3 verdict** (per spec, "Any divergence = CIPHER is perturbing
correctness"): **FAIL.** CIPHER perturbs correctness at every position.
The intra-run determinism passes, which is a small comfort: the
corruption is reproducible, not flaky.

---

## Test 4 — memory leak (sequential generate calls)

Spec: 1000 sequential calls × 100 tokens, sample nvidia-smi every 10
calls. **Reduced to 200 calls × 100 tokens** because the leak signal
stabilised by call 30 — extending to 1000 had no incremental signal
and the GPU was needed for Test 2.

| call | CIPHER nvsmi (MiB) | baseline nvsmi (MiB) |
|---|---|---|
| 0 (post-warmup)         | 20 629 | 17 003 |
| 10                      | 23 713 | 17 017 |
| 20                      | 23 713 | 17 017 |
| 30                      | 23 713 | 17 017 |
| 100                     | 23 713 | 17 017 |
| 200                     | 23 713 | 17 017 |
| **growth**              | **+3 084 MiB (front-loaded; 0 per-call after call 10)** | **+14 MiB total** |
| crashes                 | 0      | 0      |
| total wall              | 867 s  | 672 s  |
| sec / call              | 4.34   | 3.36   |

**Test 4 verdict: PASS** for "no progressive leak" — both CIPHER and
baseline are flat after warm-up. CIPHER carries a one-time **+3 GB**
FP8 buffer cost at the first hot generate; that does not accumulate.
This is the test where CIPHER actually behaves: stable working set,
no per-call growth. (My earlier reading of "leak" came from a bug in
the test harness — `nvsmi --id=0` was hardcoded and was reading the
wrong physical GPU when CIPHER tests ran with `CUDA_VISIBLE_DEVICES=1`;
fixed by mapping logical → physical GPU index, after which both runs
go quiet.)

---

## What broke

CIPHER's hook **loads** on torch 2.6.0+cu124 (this stack):
`Intercepts=306 290 / ProcAddr=812 / GOT patches=10` with no
`cudaErrorInvalidResourceHandle` (which kills it on torch 2.11). Where
it goes wrong is its actuators on Llama-3.1-8B fp16:

- **Isolated** `CIPHER_FP8_COMPUTE=on, CIPHER_FUSION_KERNELS=off`: 200-token
  smoke is `'!!!!!!!!!!!!!!!!'` from token 1, leaks 3 GB on first hot generate.
  fp8_delta = 160 weights, 288 calls — only 160 of 224 linears reach the
  stable-pointer threshold to be quantised, but the surviving fp16 paths
  alone aren't enough to keep the residual stream sane.
- **Isolated** `CIPHER_FP8_COMPUTE=off, CIPHER_FUSION_KERNELS=on`: 200-token
  smoke is `'četulum!!!!!!!!!!!!!!'` (slightly different garbage), 74 MiB
  growth. The fused RMSNorm or fused SiLU·Mul kernel is producing wrong
  output at fp16 — `residual_add` count stays at 0, so the residual
  fusion path isn't even being hit.
- **Combined** (the reported `full` config): `'!!!!!!!!!!!!!!!!'` — both
  bugs compound, FP8 output dominates.

This is consistent with `CIPHER_VLLM_TEST.md` (May 1) noting CIPHER
broke against newer torch — but the failure mode here is different:
the *hook* survives torch 2.6, the *actuators* don't preserve
correctness on Llama-3.1-8B fp16.

`step9_llama8b_full.json` from May 1 reported coherent output
(`" means doing more useful work per watt..."`) at batch 1 with the
same stack — between then and now, something in the build pipeline
regressed FP8 / fusion correctness on this model. The libs are dated
May 1 13:48 / 14:32, after that JSON was written.

## Production-readiness summary

| Property                                       | Status under CIPHER |
|---|---|
| Loads under torch 2.6 / cudart 12.4            | ✅ |
| Single-client coherent output                  | ❌ (token 0 from position 1) |
| Multi-client coherent output                   | ❌ (8/8 garbage) |
| Determinism (same input → same output)         | ✅ (deterministic garbage) |
| Memory leak across calls                       | ✅ (no per-call leak) |
| Memory budget vs baseline                      | ❌ (+8 GB per process) |
| Throughput vs baseline (single)                | ❌ (24 % slower) |
| Throughput vs baseline (8-way)                 | ❌ (27 % slower) |
| Energy efficiency tok/W (single)               | ❌ (17 % worse) |
| Energy efficiency tok/W (8-way)                | ❌ (30 % worse) |
| Concurrent capacity (8 clients @ 80 GB × 2)    | ❌ (only 6/8 fit) |

**3 of 4 stress tests fail.** The one pass (no per-call leak) is
overshadowed by garbage output across every other dimension. CIPHER
should not be enabled in production against Llama-3.1-8B on this
build until the FP8 + fusion correctness regression is fixed.

## Reproducibility

All scripts live in `op31-prod-fix/stress/`:

- `stress_common.py` — model load, fusion patches, power sampler,
  nvsmi helper (with the CUDA_VISIBLE_DEVICES → physical-GPU mapping fix)
- `t1_endurance.py`, `t2_driver.py` + `t2_child.py`, `t3_determinism.py`,
  `t4_memleak.py` — per-test drivers
- `report.py` — re-aggregates the JSON outputs into the table above
- `run_t2.sh` — sequences T2 CIPHER then T2 baseline

Raw outputs: `t{1,2,3,4}_*.json` and `*.log` in the same directory.
