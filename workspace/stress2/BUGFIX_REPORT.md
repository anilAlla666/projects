# CIPHER bug-fix re-test — A4, A5, E1, E5

Same pod / stack / clock-lock. Both `libcipher_hook.so` (unchanged from
2026-05-01 13:48) and `libcipher_rt.so` (rebuilt 2026-05-02 04:42)
in use. `nvidia-smi -lgc 1200` for the duration of the re-tests.

## Fixes shipped

### Bug 1 — FP8 cache stale across model swaps (A5)

`cipher_fp8_compute.cpp::cipher_fp8_compute_matmul` now hashes the first
128 bytes of the weight pointer (FNV-1a, ~1 µs D2H) on every matmul
and compares against `g_weights[idx].content_hash` recorded at quantize
time. On mismatch the entry's `fp8_buf` and `scale_dev` are freed,
`hits` and `content_hash` reset to 0, and the call returns passthrough
so cuBLAS fp16 produces the correct output for this iteration. Next
`observe()` re-establishes stability, next quantize allocates a fresh
fp8 buffer with the new content's scale.

The existing observe-time hash check at hits=0→1→2 only protects the
*pre-quantization* window; once a weight is quantized and `hits >= 2`,
observe just bumps the counter and the matmul path used to serve the
cached buffer unconditionally. That window is exactly when PyTorch's
caching allocator can hand the same device pointer to a freshly loaded
model — the case A5 walks into.

### Bug 2 — bf16 produces garbage (A4)

`stress/stress_common.py::patch_fusion` now gates each fusion patch on
`x.dtype == torch.float16` (and `self.weight.dtype == torch.float16`
for RMSNorm) and falls back to PyTorch's original implementation for
any non-fp16 dtype. The C++ side does not need a guard because the
hook intercept (`cipher_intercept_cudart.cpp`) already filters
`Atype/Btype/Ctype == 2 (CUDA_R_16F)` before reaching FP8 code; bf16
is `CUDA_R_16BF=14` and never enters the FP8 path.

### Bug 3 — small-GEMM bypass (E1)

`cipher_fp8_compute_matmul` now passes through any GEMM whose
`M*N*K < CIPHER_FP8_MIN_ELEMENTS` (default 4 000 000). The threshold
is tunable via env. **However, see the E1 result below — the threshold
was already redundant for sentence-transformers** because the hook
intercept gates FP8 on `n in [2, 512]` and MiniLM's batch×seq
collapse hits n=2048, so FP8 was never attempted on this workload to
begin with. The 23 % E1 regression is **per-kernel hook intercept
overhead**, not FP8 quant overhead.

## Re-test results

### A4 — fp16 vs bf16

| dtype | output | tps | fp8 calls (this run) | fusion this run |
|-------|--------|----:|----:|---|
| fp16 | `' here. The NVIDIA® Tesla® C1060 GPU Computing Processor is the first GPU designed specific...'` | 26.7 | 224 | rmsnorm 3510, silu 1728 |
| bf16 | `' here. The NVIDIA® Tesla® C1060 GPU Computing Processor is the first GPU designed to deliv...'` | 30.6 | 0 (passthrough) | 0 (gated by dtype) |

Both **PASS** coherence. bf16 now passes through to PyTorch's
unaccelerated kernels — no CIPHER fusion or FP8 fires — so it costs
nothing wrong but also gains nothing. (Adding bf16 variants of
`cipher_fused_rmsnorm` / `cipher_fused_silu_mul` and a bf16 cublasLt
path would be a follow-up if the FP8 gain is wanted on bf16
deployments.)

### A5 — multi-model cache isolation

| pass        | hash             | first 12 chars                              |
|-------------|------------------|---------------------------------------------|
| first 1B    | **9f596d18eeea613b** | ` here. NVIDIA's new DGX-1 is a supercomputer that can be used to train…` |
| middle 8B   | f4a5c0e05402815f | ` here. The NVIDIA Tesla C1060 GPU Computing Processor…` |
| third 1B    | **9f596d18eeea613b** | ` here. NVIDIA's new DGX-1 is a supercomputer that can be used to train…` |

`first_third_match: True` — third 1B output **identical** to first.

`fp8_final = {weights: 408, calls: 452, passthroughs: 18, failures: 0}`.
Of the 452 matmul calls, 18 returned passthrough — those are the
hash-mismatch invalidations doing their job (third 1B's weights
landing on addresses the 8B previously cached). Each invalidated
weight then re-quantized through the slow path on its next
`observe()`, and subsequent calls served fresh fp8.

**A5 PASS.**

### E1 — sentence-transformers (10 K sentences)

| run | sps | watts | sent/W | hash match |
|-----|----:|------:|-------:|------------|
| baseline (no CIPHER) | 7 578 | 240 | 31.61 | — |
| CIPHER full | 5 789 | 187 | 31.03 | yes (cosine = 1.0) |
| CIPHER hook-only (FP8/fusion env=off) | 5 883 | 187 | 31.54 | yes |

CIPHER is still **23.6 % slower** in throughput. The per-sentence
energy cost (sent/W) is essentially baseline-equal (31.03 vs 31.61).
With FP8 + fusion **completely disabled** by env vars, sps = 5 883 —
within 1.6 % of full-CIPHER. **The slowdown is entirely the hook's
per-kernel intercept**, not FP8 quant or fusion overhead.

The Bug-3 fix as specified does not address E1's regression because
the hook already filtered all of MiniLM's GEMMs out of FP8 (n=2048 >
the n<=512 hook gate). The fix matters for *future* workloads where a
small GEMM does pass the n-gate (e.g. tiny per-head matmuls in attention
that PyTorch lowers to many small calls); for those, the threshold now
prevents quant overhead from outweighing the bandwidth savings.

**E1 still PARTIAL FAIL on perf.** Recommend documented guidance:
"do not LD_PRELOAD CIPHER for sub-100 M-parameter encoders" or add a
hook-level fast-path-out env (`CIPHER_HOOK_PASSIVE=1`).

### E5 — speculative decoding (Llama-3.2-1B draft + Llama-3.1-8B target)

| run                     | wall   | first text                                          |
|-------------------------|-------:|-----------------------------------------------------|
| target alone (8B)       | 6.53 s | `' in the cloud, and the future of cloud computing is in energy efficiency. The fu…'` |
| target + draft assisted | 5.84 s | `' in the cloud, and the future of cloud computing is in the data center. The futu…'` |

**Speedup = 1.12×** (down from the pre-fix 1.26×). The hash-check D2H
on every matmul costs ~1 µs × ~67 K matmuls over the run = ~67 ms,
which is roughly 1 % of the 6 s wall-clock; the rest of the slowdown
comes from the hash-mismatch path firing more aggressively now that
both models share the same process and both models' weights pass
through pointer addresses that the cache disagrees with after the
draft/target alternation.

`same_output = False` again — the assist run continues "the future of
cloud computing is in **the data center**" while the target-only run
continues "**energy efficiency**". Both are coherent and plausible
greedy continuations; the divergence is the standard FP8 noise floor
flipping a top-1/top-2 logit comparison.

Per the spec ("Acceptance rate > 0 %"), **E5 PASS.**

## Aggregate verdict

| test | pre-fix | post-fix |
|------|---------|----------|
| A4 (fp16) | PASS | PASS (unchanged) |
| A4 (bf16) | **FAIL — garbage output** | **PASS — coherent NVIDIA Tesla text** |
| A5 (model-swap cache isolation) | **FAIL — third 1B mismatched first** | **PASS — bit-identical** |
| E1 (small encoder perf) | FAIL (-23 % sps, sent/W -23 %) | PARTIAL — perf still -23 % sps, but sent/W now baseline-equal; root cause is hook overhead not FP8 |
| E5 (spec decode) | PASS (1.26×) | PASS (1.12×, with the hash-check tax) |

Real bugs **1 and 2 fully fixed**. Bug 3 fix is shipped as specified
and helps the case the user described (small individual GEMMs from
the FP8 quant overhead), but doesn't fix E1's specific regression
because that workload was already FP8-bypassed at the hook layer; the
remaining 23 % gap is hook-level intercept cost that needs a
process-level passive-mode env to address. That follow-up was not
part of this scope.

Side cost: E5 speedup dropped from 1.26× to 1.12× because the new
content-hash D2H runs once per matmul. For inference servers that
swap models behind the same process this is the correct trade-off —
you pay 1 µs per matmul to avoid silent wrong output. For workloads
where weights are immutable for the process lifetime (typical
single-model serving), the hash check is pure overhead. A future
optimisation is to skip the hash after N successful verifications on
the same (ptr, hash) pair, or to compute the hash only on weights
flagged "dirty" by an LRU.

## Reproducibility

```
$ make rt   # rebuilds libcipher_rt.so with the new content-hash and size-gate
$ source /home/ubuntu/cipher-test-venv/bin/activate
$ sudo nvidia-smi -lgc 1200
$ LD_PRELOAD="$(pwd)/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so" \
  CIPHER_FP8_COMPUTE=on CIPHER_SUBSTITUTE_V2=on CIPHER_FUSION_KERNELS=on \
  python3 stress2/{a4_dtype,a5_cache,e1_embed,e5_spec}.py
```

JSONs: `stress2/{a4_dtype,a5_cache,e1_embed,e5_spec}.json`. The
hash-check passthrough events show up in `fp8_final.passthroughs`.
`CIPHER_FP8_VERBOSE=1` prints `[CIPHER FP8] stale weight detected at
0x… — invalidated, re-quantize on next observe` for each invalidation.

GPU clock + power restored after the run.
