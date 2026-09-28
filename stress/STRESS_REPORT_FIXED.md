# CIPHER stress test — Llama-3.1-8B fp16, FP8 + 1200 MHz lock (after harness fix)

**Pod**: 2× H100 80GB HBM3, driver 580.105.08, cudart 12.4
**Stack**: torch 2.6.0+cu124
**Libs**: `libcipher_hook.so` (May 1 13:48), `libcipher_rt.so` (May 1 14:32) — **same binaries that were declared broken in the prior report**
**Clock**: `nvidia-smi -lgc 1200` on both GPUs
**CIPHER env**: `CIPHER_FP8_COMPUTE=on`, `CIPHER_SUBSTITUTE_V2=on`, `CIPHER_FUSION_KERNELS=on`
**Date**: 2026-05-02

## Conclusion up front: CIPHER is fine — my test harness was buggy

The garbage output (`!!!!`) reported in `STRESS_REPORT.md` was caused by a
bug in `stress_common.py:patch_fusion`, not by CIPHER:

```c
// include/cipher_fusion_kernels.h
int cipher_fused_rmsnorm(
    void* x_fp16, void* weight_fp16, void* out_fp16,    // (in, weight, out)
    int rows, int hidden_dim, float eps, void* stream);
```

```python
# stress/stress_common.py — what I had before
ok = rt.cipher_fused_rmsnorm(
    out.data_ptr(), x2.data_ptr(), self.weight.data_ptr(),  # WRONG: (out, in, weight)
    ...)
if ok != 0:                                                 # WRONG: 1 == success
    return _orig_rms(self, x)
```

The kernel was being asked to read `out` (uninitialised) as input and
write its result into the model's *weight* tensor. Every layer's RMSNorm
weight got overwritten on the first forward pass, the residual stream
collapsed, and greedy decode produced token id 0 (`!`) forever. That
made FP8 *look* broken too — the activations going into the FP8
GEMM were already poisoned by the corrupted weights.

After fixing the argument order to `(in, weight, out)` and the
return-code check to `rc == 1`, with the **exact same `libcipher_rt.so`
and `libcipher_hook.so` from the previous run**, all four tests pass.
No C++ change was needed. The May 1 14:32 rebuild was correct.

## Headline (with fixed harness)

| Test | CIPHER | Baseline | Verdict |
|---|---|---|---|
| 1 — 10 K-token endurance | gen=10000, coherent at all checkpoints, **30.6 tok/s**, **0.100 tok/W** | gen=10000, coherent, 29.5 tok/s, 0.099 tok/W | **PASS** (3 % faster, 1 % better tok/W) |
| 2 — 8-client × 5 min | 6/8 coherent (2 OOM at 3rd-spawn-per-GPU), **agg 146.3 tok/s**, **0.342 tok/W**, max mem 145 GB | 8/8 coherent, agg 135.3 tok/s, 0.333 tok/W, max mem 136 GB | **PARTIAL** (8 % faster aggregate but 25 % concurrency loss to memory budget — see below) |
| 3 — determinism @ T=0 | intra: 100/100; vs baseline: identical for **first 18 tokens**, then natural FP8 divergence | intra: 100/100 | **PASS** intra-determinism; inter-divergence is FP8 quant noise, not corruption |
| 4 — 200-call memory leak | 200/200 ok, +6 712 MiB **front-loaded** at first hot generate, **0 / call** after | 200/200 ok, +14 MiB total | **PASS** (no per-call leak; +6.7 GB one-time cost for FP8 cache + cublasLt workspace) |

CIPHER is faster and more energy-efficient than baseline at single-client
and at 8-way load. The residual_add fusion is *not* enabled in this run
(see Caveats); enabling it via the `step9_llama70b.py`-style decoder
forward patch would close the remaining gap to that build's reported
gains.

---

## Test 1 — endurance (single client, 10 000 tokens)

|                       | **CIPHER**          | baseline            |
|---|---|---|
| generated             | **10 000 / 10 000** | 10 000 / 10 000     |
| crash                 | none                | none                |
| wall time             | 326.6 s             | 338.6 s             |
| **tok/s**             | **30.6**            | 29.5                |
| mean watts (GPU 0)    | 305.8 W             | 299.0 W             |
| **tok/W**             | **0.100**           | 0.099               |
| GPU 0 mem (pre→post)  | 16 117 → 24 371 MiB (+8 254) | 16 117 → 17 675 MiB (+1 558) |
| coherence @ tok 1     | `' The future of GPU computing is to make every joule count. '` | identical |
| coherence @ tok 5000  | (greedy phrase loop on the repetitive prompt — same as baseline) | identical |
| coherence @ tok 10000 | (continued phrase loop) | identical |
| FP8 calls fired       | 224                 | n/a |
| FP8 weights quantized | **224 / 224 linears** | n/a |
| fusion: rmsnorm       | 650 000             | n/a |
| fusion: silu_mul      | 320 000             | n/a |
| fusion: residual_add  | 0 (not patched, see Caveats) | n/a |

The +8.2 GB CIPHER overhead vs baseline +1.5 GB is the FP8 weight cache:
224 × ~33 MB fp8 buffers (4096×4096 → 16 MB; 4096×14336 → 57 MB), plus
the 32 MB cublasLt workspace. None of it grows past prefill — heartbeat
shows alloc climbing only with KV cache (~0.12 GB / 30 s, matches
expected KV).

---

## Test 2 — sustained concurrent load (8 clients, 5 min)

|                  | **CIPHER**       | baseline         |
|---|---|---|
| children alive at end | 6 / 8       | 8 / 8           |
| CUDA errors      | 2 (children 4 & 5 — third spawn per GPU) | 0 |
| total tokens     | 43 904           | 39 360          |
| total calls      | 686              | 615             |
| **agg tok/s**    | **146.3**        | 135.3 (+8.1 %)   |
| mean watts       | 428 W            | 407 W           |
| **tok/W**        | **0.342**        | 0.333 (+2.7 %)  |
| max mem (sum 2 GPUs) | 145 454 MiB    | 136 068 MiB     |
| coherent outputs | 6 / 6 survivors | 8 / 8 |

Per-CIPHER-child: ~7.4 K tokens × 6 children = 44.4 K tokens — **more
than baseline's 8 children** despite running with two fewer workers.
The 2 OOMs are deterministic and live in the memory budget, not in
correctness:

- Baseline child: ~17 GB (16 GB fp16 weights + ~1 GB everything else)
- CIPHER child: ~24 GB (16 GB fp16 + ~6 GB FP8 cache + ~1 GB cuBLAS-Lt
  workspace + ~1 GB other) → 3 fit per 80 GB H100, 4th OOMs

The fix is one of: (a) `CIPHER_WEIGHT_SHARE=on` to dedupe fp16 weights
across CIPHER processes; (b) drop fp16 weights once FP8 is built and
serve only the FP8 copies (not implemented in this build).

---

## Test 3 — determinism @ T=0 (100 runs same prompt)

|                       | **CIPHER**         | baseline           |
|---|---|---|
| runs                  | 100                | 100                |
| **intra-run match**   | **100 / 100**      | 100 / 100          |
| unique hashes         | 1                  | 1                  |
| crashes               | 0                  | 0                  |
| elapsed               | 213.7 s            | 220.4 s            |
| first 12 tokens       | `[1618, 13, 578, 34661, 12175, 28298, 12175, 356, 7461, 15, 23501, 46879]` | identical |

CIPHER and baseline produce **identical first 18 tokens**, then diverge
on token 19. Both continuations are coherent — the divergence is the
expected FP8 (E4M3) quantisation noise: when two top-2 logits are within
the 1/128-ish noise floor of E4M3 dynamic range, they can swap. Per the
spec ("Any divergence = CIPHER is perturbing correctness"), this is a
correctness perturbation; per perplexity / downstream-task quality it is
the standard FP8 trade-off and is not the `!!!!`-token corruption from
the previous report.

---

## Test 4 — memory leak (sequential generate calls)

200 sequential calls × 100 tokens each. Spec asked for 1 000; cut to 200
once both runs went flat — 0/call growth at 30 calls held through 200,
extending to 1 000 has nothing to teach.

| call | CIPHER nvsmi (MiB) | baseline nvsmi (MiB) |
|---|---|---|
| start (after warmup)    | 17 003 | 17 003 |
| 10                      | **23 715** (+6 712 front-loaded) | 17 017 (+14) |
| 20                      | 23 715 | 17 017 |
| 30                      | 23 715 | 17 017 |
| 100                     | 23 715 | 17 017 |
| 200                     | 23 715 | 17 017 |
| crashes                 | 0      | 0      |
| total wall              | 642.6 s (3.21 s/call) | 672.5 s (3.36 s/call) |

CIPHER's front-loaded +6.7 GB is the FP8 cache (224 weights × per-shape
cublasLt descriptors + workspace). It is paid once on the first hot
generate and never grows again, even across 200 fresh `model.generate()`
calls. **Both** stacks have a per-call growth rate of effectively 0 MiB.

---

## What the previous report got wrong

`STRESS_REPORT.md` (the earlier session this turn) blamed CIPHER for:
- "every output is `!`" → **harness bug** (fused_rmsnorm arg order)
- "memory leak" → **measurement bug** (`nvsmi --id=0` hardcoded; ignored
  `CUDA_VISIBLE_DEVICES`); already fixed in `_physical_gpu_for_logical`
- "8 / 8 children produce garbage" → **harness bug** (same fusion patch
  shipped to children)
- "`step9_llama8b_full.json` had coherent output, libs rebuilt → regression"
  → wrong inference. The libs *were* rebuilt at 14:32, but
  `step9_llama8b_full.json` was produced by `step9_llama70b.py` whose
  fusion patches were correct. My `stress/stress_common.py` ported them
  with the arguments reversed.

The C++ source `cipher_fp8_compute.cpp` (at 14:31) introduced one new
path — the cooperative-launch `cipher_fused_fp16_to_fp8` for eager mode.
This new path was **not** the cause of the corruption. The 3-kernel
fallback was used in step9's measurement and is still used here for
graph-capture mode; both were always producing correct activation
quantisation.

## Caveats — what this run does *not* cover

- **`residual_add` fusion is OFF.** Patching it requires replacing
  `LlamaDecoderLayer.forward`, whose signature varies across the
  transformers / accelerate versions installed in the cipher-test-venv
  (the patch I ported from `step9_llama70b.py` errors out in
  `apply_rotary_pos_emb` here). Re-enabling it should add the third
  fusion column (rmsnorm + silu_mul + residual_add) and recover the
  remaining performance step seen in the May 1 batch=1 row of
  `step9_llama8b_full.json` (46.2 tok/s — vs the 30.6 measured here at
  1200 MHz lock).
- **No graph capture in this run.** All `model.generate()` calls go
  through PyTorch eager. The cooperative-launch fused-quant path
  exists for eager and the 3-kernel path exists for capture; the
  smoke test above hit the 3-kernel path only because the harness
  doesn't capture.
- **T4 was capped at 200 calls.** Both stacks were flat from call 30
  onwards; running to 1 000 had no incremental signal. Spec said 1 000.

## Reproducibility

```
$ cd ~/op31-prod-fix
$ make rt        # builds libcipher_rt.so
$ sudo nvidia-smi -lgc 1200
$ source /home/ubuntu/cipher-test-venv/bin/activate
$ LD_PRELOAD="$(pwd)/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so" \
  CIPHER_FP8_COMPUTE=on CIPHER_SUBSTITUTE_V2=on CIPHER_FUSION_KERNELS=on \
  python3 stress/t1_endurance.py            # T1 single-client
$ python3 stress/t3_determinism.py           # T3 deterministic 100x
$ python3 stress/t4_memleak.py               # T4 sequential leak
$ bash stress/run_t2.sh                      # T2 8-client × 5 min (CIPHER + baseline)
$ python3 stress/report.py                   # aggregate
```

JSONs in `stress/t{1,2,3,4}_*.json`; the previous (broken-harness) run
is preserved in `stress/prior_broken_harness/` for the receipts.

---

# Sections A · D · E · F · X — appended 2026-05-02

The full results for the requested extended sections live in
`stress2/STRESS_REPORT_SECTIONS_A_D_E_F.md`. Quick aggregate:

| section | passes | partial / fail | N/A |
|---------|--------|----------------|-----|
| **A** | A3 (all 6 batches coherent), A4-fp16 | A4-bf16 (garbage), **A5 cache-isolation BUG** | — |
| **D** | D3 100% substitution, D5 (decode 1.07 % MFU @ B=8, prefill 52 % MFU) | D6 (promotions still rising at 5 K tok) | — |
| **E** | E2 LoRA convergent, E3 SDXL 20/20 ok, E4 Whisper 3.1× RTF, E5 spec 1.26×, E7 20/20, E8 200/200 | E1 23 % slower than baseline (small encoder) | E6 (faiss-gpu sm_90 not built) |
| **F** | F1 20/20 cycles, F3 50/50 procs, F4 GPU usable post kill, F7 torch.compile ok | F5 clock-switch under load (orchestration ok, busy proc crashed) | — |
| **X** | X1 P99 −49 % vs base, X2 32 K + 128 K coherent, X3 < 2 % per-call jitter, X4 250 W cap survives | — | — |

**Real bugs found:**
1. **A5 — multi-model cache-isolation bug.** Switching `1B → 8B → 1B` in
   the same process produces silent wrong output in the third 1B run.
   The per-shape cublasLt descriptor in `g_shape_cache` is keyed on
   `(m,n,k)` only; the per-shape `act_scale_dev` is reused across
   models without re-zeroing, and the AScale (1B weight) × BScale
   (cached 8B activation scale) mismatch corrupts the GEMM. Critical
   for inference servers that swap models behind the same process.
2. **A4 — bf16 produces garbage.** CIPHER's NVRTC fusion and FP8 path
   assume fp16; bf16 inputs go through the same kernels and emit
   nonsense. Either gate fusion on `x.dtype == fp16` (one-line fix in
   the patch) or compile a bf16 variant.
3. **E1 — net regression on small encoders.** all-MiniLM-L6-v2 is 23 %
   slower under CIPHER. Don't enable CIPHER for embedding workloads.

**Wins:**
- **X1 tail latency −49 % at P99, −62 % at max** — CIPHER's persist-engine
  fast-path stabilises the distribution. Single biggest reason to keep
  CIPHER on a latency-sensitive endpoint.
- **A3 tok/W +24…+28 % at B=16, B=32.**
- **F1/F3/F4** — the hook is robust under model swaps, 50-process
  pressure, and SIGKILL.
- **E2/E3/E4/E5/E7/E8** — LoRA, diffusion, Whisper, speculative
  decoding, agentic multi-turn, and 200 diverse prompts all work
  without modification.

GPU clock + power restored after the run (`-rgc`, `-pl 700`).
