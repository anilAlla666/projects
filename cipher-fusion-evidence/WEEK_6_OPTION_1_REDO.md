# WEEK_6_OPTION_1_REDO.md

**Date:** 2026-05-23
**Replaces (the numbers from):** Week-5 post-close Option 1 measurements taken with `/tmp/postclose/bench_mfu.py`.
**Audit:** `WEEK_6_BENCH_HARNESS_AUDIT.md`
**New harness:** `/tmp/bench_v2/bench_llm.py` — methodology in `WEEK_6_BENCH_HARNESS_REWRITE.md`.

---

## TL;DR — the honest finding

At the canonical Option-1 configuration **(single vLLM process, Mistral-7B-v0.1, B = [1, 8, 16, 32, 64], 1024-token agentic prompt, 256 output tokens, BF16, H100 SXM5 locked @ 1980/2619 MHz)**:

> **CIPHER substrate is essentially NEUTRAL vs. vanilla at this configuration.** Aggregate throughput deltas are ≤ 0.31% across all batch sizes (Cohen's d ≤ 3.05 at B=1, n.s. for B ≥ 8). Decode MFU deltas are ≤ 0.79%. Power runs **0.25%–0.78% higher** under CIPHER (the page-tracking / KV-dedup metadata overhead) without a compensating throughput lift, yielding a small but statistically significant **−0.25% to −0.59% regression in tok/W** at B ∈ {1, 8, 32}.

This finding is consistent with the prior CIPHER memory anchor `[CIPHER TPW re-test + 3.6× correction]`:
> *"old '3.617× single-instance no-perf-loss' RETRACTED (F1 bug); real lever = cross-tenant batching, re-tested 2026-05-19 […]; '3.6×' is a located point ≈N=9; honest investor claim = '3-6× tok/W by tenant density' NOT single-instance."*

The Option-1 redo on the corrected harness **confirms that retraction directly**: single-process single-instance B-sweep does not engage CIPHER's lift mechanisms (KV-dedup cross-tenant page sharing, VOLT clock-gating, SM partitioning). What it *does* engage is the substrate's metadata overhead, which is small (<1%) but real.

---

## 1. Run parameters

| Item                      | Value                                                           |
|---------------------------|-----------------------------------------------------------------|
| Model                     | mistralai/Mistral-7B-v0.1 (BF16, 32 layers, GQA 32/8 heads)     |
| Batch sizes               | 1, 8, 16, 32, 64                                                |
| Iterations per batch      | 5 (after 2 warmup iterations)                                  |
| Output tokens             | 256 (deterministic, `ignore_eos=True`)                          |
| Prompt length             | 1024 tokens, agentic workload (system + tools + task + trace_id) |
| Hardware                  | H100 SXM5 80GB HBM3, driver 580.105.08, locked 1980/2619 MHz, persistence ON, ECC ON, 700W power limit |
| vLLM                      | 0.21.0 (V1 engine, CUDA graphs PIECEWISE+FULL, FlashAttention v3) |
| Telemetry                 | pynvml direct @ 100 Hz, pinned CPU 0, trapezoidal energy        |
| Vanilla env               | `CIPHER_KV_ALLOC=0, CIPHER_KVDEDUP=0, CIPHER_OFFLOAD=0`         |
| CIPHER env                | `CIPHER_KV_ALLOC=1, CIPHER_KVDEDUP=1, CUDA_INJECTION64_PATH=…libcipher_rt.so` |
| Anchors at run            | libcipher_rt.so md5 `259ac994aead2da8289fc84d6116fbe9` (week-5-complete) |

Two configurations were run to give a complete picture:

- **(a) Prefix-caching ON** (vLLM default `enable_prefix_caching=True`): production-realistic — agentic deployments with shared system prompts hit the cache hard. *But:* the prefill-MFU formula is over-counted on iterations 2+ because cached prefill skips compute. We document the artifact and use decode MFU as the load-bearing metric.
- **(b) Prefix-caching OFF** (`enable_prefix_caching=False`): clean MFU accounting — both prefill and decode FLOP-counts are honest. The headline comparison uses (b) for MFU and (a) for production-realistic latency/throughput.

---

## 2. Vanilla baseline — full per-batch table (cache ON, production-realistic)

`/tmp/bench_v2/results/20260523_mistral7b_vanilla_agentic_fixed1024/result.json`

| B  | n_iter | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | agg_tps | decode_MFU% | prefill_MFU%* | mean_W | peak_W | energy_J | tok/W | tok/J | peak_HBM_MiB | SM_clk_MHz |
|----|--------|-------------|-------------|-------------|-------------|---------|-------------|---------------|--------|--------|----------|-------|-------|---------------|------------|
| 1  | 5      | 10.6        | 10.6        | 6.03        | 6.03        | 165.3   | 0.249       | 141.2*        | 420.8  | 422.1  | 648.1    | 0.393 | 0.395 | 70046         | 1830       |
| 8  | 5      | 19.3        | 19.9        | 6.39        | 6.40        | 1237.9  | 1.877       | 560.8*        | 447.3  | 450.5  | 738.0    | 2.768 | 2.775 | 70214         | 1830       |
| 16 | 5      | 26.4        | 27.9        | 6.77        | 6.77        | 2331.9  | 3.537       | 818.6*        | 485.7  | 488.9  | 850.1    | 4.801 | 4.819 | 70287         | 1830       |
| 32 | 5      | 47.1        | 50.4        | 7.47        | 7.49        | 4186.1  | 6.389       | 928.2*        | 522.6  | 529.4  | 1019.6   | 8.010 | 8.034 | 70336         | 1830       |
| 64 | 5      | 81.4        | 88.5        | 8.96        | 9.03        | 6895.7  | 10.551      | 1064.8*       | 599.8  | 610.3  | 1420.8   | 11.498| 11.532| 70438         | 1830       |

\* prefill_MFU > 100% is a prefix-caching artifact: vLLM's prefix cache hits the shared system-prompt prefix across iterations, so the formula's FLOP numerator (which assumes fully-uncached prefill) is overestimated. See the cache-OFF table below for clean prefill MFU.

---

## 3. Vanilla baseline — clean MFU (cache OFF)

`/tmp/bench_v2/results/20260523_mistral7b_vanilla_agentic_fixed1024_nocache/result.json`

| B  | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | agg_tps | decode_MFU% | **prefill_MFU%** | mean_W | peak_W | energy_J | tok/W | tok/J | peak_HBM_MiB |
|----|-------------|-------------|-------------|-------------|---------|-------------|-------------------|--------|--------|----------|-------|-------|---------------|
| 1  | 25.2        | 25.2        | 6.05        | 6.05        | 163.3   | 0.249       | **59.57**         | 433.7  | 437.6  | 676.5    | 0.377 | 0.378 | 70046         |
| 8  | 120.6       | 180.2       | 6.58        | 6.89        | 1114.0  | 1.702       | **65.05**         | 483.0  | 514.3  | 885.9    | 2.306 | 2.312 | 70214         |
| 16 | 209.0       | 358.6       | 7.33        | 7.92        | 1924.7  | 2.934       | **65.73**         | 534.4  | 592.5  | 1134.1   | 3.601 | 3.612 | 70214         |
| 32 | 379.0       | 731.7       | 8.78        | 9.96        | 3019.1  | 4.575       | **64.89**         | 585.8  | 687.8  | 1586.3   | 5.154 | 5.164 | 70214         |
| 64 | 762.7       | 1521.6      | 11.89       | 14.24       | 4156.1  | 6.292       | **62.06**         | 651.4  | 706.4  | 2565.1   | 6.381 | 6.387 | 70214         |

**Reading the clean prefill MFU:** 60–66% across all batch sizes. This is well within the textbook 50–80% range for compute-bound prefill on H100 BF16. The harness math is verified.

**Reading the decode MFU:** climbs from 0.249% at B=1 (memory-bandwidth-bound; only one token's worth of weight-reuse work per HBM read) up to 6.292% at B=64 (memory-bound → compute-bound transition partially under way; weights amortized over 64 query tokens per step). This is the expected curve for a 7B-class model on H100.

**Reading the TTFT inflation at B≥8 in cache-OFF:** at B=64 cache-OFF, full prefill of 64 × 1024 = 65 536 tokens takes 763 ms median, 1.52 s p99 — this is the *real* prefill compute cost when nothing is cached. Cache-ON reduces this to 81 ms / 88 ms because all 64 prompts share the same system-prompt prefix.

---

## 4. CIPHER arm

### 4.1 Cache ON (production-realistic)

`/tmp/bench_v2/results/20260523_mistral7b_cipher_agentic_fixed1024/result.json`

| B  | agg_tps | decode_MFU% | mean_W | peak_W | tok/W | peak_HBM_MiB |
|----|---------|-------------|--------|--------|-------|---------------|
| 1  | 165.76  | 0.250       | 424.1  | 427.5  | 0.391 | 70058         |
| 8  | 1238.25 | 1.878       | 450.0  | 452.5  | 2.751 | 70226         |
| 16 | 2331.98 | 3.544       | 487.0  | 489.5  | 4.789 | 70354         |
| 32 | 4190.45 | 6.440       | 524.6  | 530.8  | 7.986 | 70354         |
| 64 | 6899.82 | 10.623      | 598.8  | 610.6  | 11.523| 70354         |

### 4.2 Cache OFF (clean MFU)

`/tmp/bench_v2/results/20260523_mistral7b_cipher_agentic_fixed1024_nocache/result.json`

| B  | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | agg_tps | decode_MFU% | **prefill_MFU%** | mean_W | peak_W | energy_J | tok/W | tok/J | peak_HBM_MiB |
|----|-------------|-------------|-------------|-------------|---------|-------------|-------------------|--------|--------|----------|-------|-------|---------------|
| 1  | 25.4        | 25.4        | 6.04        | 6.04        | 163.6   | 0.249       | **59.09**         | 438.5  | 442.1  | 683.7    | 0.373 | 0.374 | 70058         |
| 8  | 120.2       | 178.6       | 6.60        | 6.90        | 1113.6  | 1.702       | **65.63**         | 483.3  | 513.5  | 886.4    | 2.304 | 2.310 | 70226         |
| 16 | 213.9       | 359.1       | 7.35        | 7.94        | 1923.3  | 2.938       | **65.49**         | 534.7  | 592.6  | 1134.6   | 3.597 | 3.610 | 70226         |
| 32 | 393.6       | 730.3       | 8.84        | 10.02       | 3012.1  | 4.584       | **64.46**         | 585.5  | 688.4  | 1588.0   | 5.144 | 5.159 | 70226         |
| 64 | 765.0       | 1522.0      | 11.89       | 14.24       | 4155.2  | 6.292       | **62.05**         | 651.9  | 703.9  | 2568.6   | 6.374 | 6.378 | 70226         |

CIPHER's clean prefill MFU is 59-66%, the same envelope as vanilla (60-66%). The substrate does not regress prefill compute efficiency.

---

## 5. Side-by-side with statistical significance

### 5.0 Cache-ON comparison

Source: `/tmp/bench_v2/results/comparison_cache_on/comparison_table.md` and `significance.json`. Significance: bootstrap 1000-resample 95% CI of (CIPHER − vanilla); "YES" means `0 ∉ CI95`.

### 5.1 Headline metrics

| B | vanilla TPS | CIPHER TPS | Δ% | sig? | vanilla decode_MFU% | CIPHER decode_MFU% | Δ% | sig? | vanilla tok/W | CIPHER tok/W | Δ% | sig? |
|---|-------------|------------|------|------|----------------------|---------------------|------|------|----------------|---------------|------|------|
| 1 | 165.258  | 165.762  | +0.31% | YES | 0.2495 | 0.2502 | +0.31% | YES | 0.3927 | 0.3909 | −0.47% | YES |
| 8 | 1237.928 | 1238.247 | +0.03% | no  | 1.8768 | 1.8785 | +0.09% | no  | 2.7678 | 2.7514 | −0.59% | YES |
| 16 | 2331.914 | 2331.977 | +0.00% | no  | 3.5374 | 3.5437 | +0.18% | no  | 4.8008 | 4.7888 | −0.25% | no  |
| 32 | 4186.145 | 4190.445 | +0.10% | no  | 6.3893 | 6.4399 | +0.79% | YES | 8.0098 | 7.9858 | −0.30% | YES |
| 64 | 6895.669 | 6899.817 | +0.06% | no  | 10.5509 | 10.6232 | +0.68% | YES | 11.4979 | 11.5232 | +0.22% | no  |

### 5.2 Tail latency

| B | vanilla TTFT p50 ms | CIPHER TTFT p50 ms | Δ% | sig? | vanilla TPOT p99 ms | CIPHER TPOT p99 ms | Δ% | sig? |
|---|----------------------|---------------------|------|------|----------------------|---------------------|------|------|
| 1 | 10.6 | 10.7 | +0.33% | no  | 6.03 | 6.01 | −0.31% | YES |
| 8 | 19.3 | 18.6 | −3.78% | no  | 6.40 | 6.40 | −0.03% | no  |
| 16 | 26.4 | 27.2 | +3.01% | no  | 6.77 | 6.77 | −0.07% | no  |
| 32 | 47.1 | 47.7 | +1.34% | no  | 7.49 | 7.49 | −0.04% | no  |
| 64 | 81.4 | 81.4 | +0.04% | no  | 9.03 | 9.04 | +0.11% | no  |

TTFT deltas are noisy at B ≥ 8 (`±3% range`, none significant) — the harness's TTFT measurement at B ≥ 8 is dominated by which request happens to finish prefill first when 8/16/… requests are batched simultaneously. Per-request scheduling jitter produces this noise; it is not a CIPHER signal.

### 5.3 Capacity (HBM)

| B | vanilla peak_HBM MiB | CIPHER peak_HBM MiB | Δ MiB | sig? |
|---|----------------------|---------------------|--------|------|
| 1 | 70 046 | 70 058 | +12 | YES |
| 8 | 70 214 | 70 226 | +12 | YES |
| 16 | 70 287 | 70 354 | +67 | YES |
| 32 | 70 336 | 70 354 | +18 | YES |
| 64 | 70 438 | 70 354 | −84 | (zero-variance — see note) |

The +12 to +67 MiB CIPHER overhead is the page-tracking metadata footprint of the cipher_kv_bridge VMM allocator (slab descriptors, page-mapping tables for the 80 GiB VA pool). This is much smaller than the working pool (~70 GiB) and does not interact with capacity in a meaningful way at single-tenant.

The B=64 row shows −84 MiB but the bootstrap CI is degenerate because vanilla's measurements were identical across iterations (zero variance) — both vanilla and CIPHER pre-allocate to `gpu_memory_utilization * total`; the small numeric delta is sampler-window timing of the HBM read against the allocator state, not a CIPHER lift.

### 5.4 Power

| B | vanilla mean_W | CIPHER mean_W | Δ W | Δ% | sig? |
|---|-----------------|----------------|------|------|------|
| 1 | 420.81 | 424.08 | +3.27 | +0.78% | YES |
| 8 | 447.26 | 450.04 | +2.79 | +0.62% | YES |
| 16 | 485.74 | 486.97 | +1.23 | +0.25% | YES |
| 32 | 522.6 | 524.6 | +2.0 | +0.38% | YES |
| 64 | 599.8 | 598.8 | −1.0 | −0.17% | no  |

CIPHER consistently draws ~0.3–0.8% more power at B ≤ 32 and is within power-drift noise at B = 64. The C.3 self-consistency test established a vanilla run-to-run power drift of +0.53% — the CIPHER overhead at B = 1 (+0.78%) is **above** this drift floor; at B = 16 (+0.25%) it is **at** the drift floor. The CIPHER substrate adds a small real overhead.

### 5.5 Cache-OFF comparison (clean MFU)

Source: `/tmp/bench_v2/results/comparison_cache_off/comparison_table.md`. Same harness, prefix-caching disabled, 5 iter × 2 warmup × 5 batch sizes per arm.

| B | vanilla TPS | CIPHER TPS | Δ% | sig? | vanilla decode_MFU% | CIPHER decode_MFU% | Δ% | sig? | vanilla tok/W | CIPHER tok/W | Δ% | sig? |
|---|-------------|------------|------|------|----------------------|---------------------|------|------|----------------|---------------|------|------|
| 1 | 163.29 | 163.61 | +0.19% | YES | 0.2488 | 0.2493 | +0.21% | YES | 0.3765 | 0.3731 | −0.91% | YES |
| 8 | 1114.0 | 1113.6 | −0.04% | no  | 1.7021 | 1.7018 | −0.02% | no  | 2.3061 | 2.3038 | −0.10% | no  |
| 16 | 1924.7 | 1923.3 | −0.07% | no  | 2.9342 | 2.9380 | +0.14% | no  | 3.6010 | 3.5966 | −0.12% | no  |
| 32 | 3019.1 | 3012.1 | −0.23% | YES | 4.5747 | 4.5840 | +0.20% | no  | 5.1539 | 5.1442 | −0.19% | no  |
| 64 | 4156.1 | 4155.2 | −0.02% | no  | 6.2916 | 6.2917 | +0.00% | no  | 6.3811 | 6.3741 | −0.11% | no  |

**Reading cache-OFF prefill MFU side-by-side:**

| B | vanilla prefill_MFU% | CIPHER prefill_MFU% | Δ% | sig? |
|---|------------------------|----------------------|------|------|
| 1 | 59.57 | 59.09 | −0.81% | YES (Cohen's d = −1.52) |
| 8 | 65.05 | 65.63 | +0.89% | n.s. (CI [−0.30, +1.45]) |
| 16 | 65.73 | 65.49 | −0.36% | n.s. (CI [−0.94, +0.45]) |
| 32 | 64.89 | 64.46 | −0.66% | n.s. (CI [−1.32, +0.04]) |
| 64 | 62.06 | 62.05 | −0.01% | n.s. (CI [−0.62, +0.58]) |

The cache-OFF prefill MFU comparison confirms what the cache-ON one could not: vanilla and CIPHER both deliver ~60–66% prefill MFU within ≤1% of each other, with no consistent direction. The substrate does not pay the overhead inside the GEMM/SDPA kernels; it pays it CPU-side, as predicted.

**The cache-OFF comparison reproduces the cache-ON narrative:** small (<1%) regression in tok/W at B=1 (above noise floor); statistical-noise differences at B ≥ 8; no throughput lift anywhere.

---

## 6. Plots

Generated by `/tmp/bench_v2/compare_runs.py` to two directories — one per cache configuration:

`/tmp/bench_v2/results/comparison_cache_on/` (production-realistic):
- `tps_vs_batch.png` — overlapping vanilla and CIPHER curves; both climb from ~165 to ~6900 tok/s; visually indistinguishable.
- `mfu_decode_vs_batch.png` — overlapping vanilla and CIPHER curves; both climb 0.25% → 10.6%.
- `tok_per_w_vs_batch.png` — CIPHER slightly below vanilla at B ∈ {1, 8, 16, 32}, indistinguishable at B = 64.
- `hbm_vs_batch.png` — both arms tight to ~70.0–70.4 GiB (vLLM-reserved pool dominates; per-iter HBM-used delta is metadata-scale).

`/tmp/bench_v2/results/comparison_cache_off/` (clean MFU):
- `tps_vs_batch.png` — both climb ~163 → ~4156 tok/s; visually indistinguishable.
- `mfu_decode_vs_batch.png` — both climb 0.25% → 6.29%; visually indistinguishable.
- `tok_per_w_vs_batch.png` — CIPHER below vanilla at B=1 by ~0.9%; indistinguishable elsewhere.
- `hbm_vs_batch.png` — both arms tight to ~70.0–70.2 GiB.

---

## 7. Narrative

### 7.1 Where does CIPHER lift, at what batch size, by how much?

**It does not lift on this workload.** Throughput and decode-MFU deltas hover at 0–0.8% with mixed significance (a few CI-pass-zero, a few don't). None is a load-bearing claim of efficiency improvement.

This is not a failure of CIPHER. It is a confirmation of what its lift mechanisms actually do:

- **Cross-tenant KV-dedup** requires ≥ 2 vLLM processes (or ≥ 2 tenants in a multi-tenant orchestration) running the same model with the same shared prefix in their KV. We had a single process with B-batched concurrent requests; vLLM's *intra-process* prefix cache already does the same dedup at the page level. CIPHER's overhead is paid; the lift never arrives.
- **VOLT (clock-gating / DVFS)** is not engaged by default — would require `CIPHER_VOLT=1` and a clock target. Not set in this run.
- **SM partitioning** is not engaged at single-tenant — would require multiple tenants with `CIPHER_SM_PARTITION` partitions specified.
- **Marlin INT4 / weight-sharing** does not apply to BF16 Mistral-7B and not to single-instance configurations.

### 7.2 Where does it regress?

Small, real overhead at B ∈ {1, 8, 32}:

- **Power +0.25 to +0.78%** — CIPHER's KV-page tracking, kvdedup hash bookkeeping, and runtime symbol interception (`CUDA_INJECTION64_PATH=libcipher_rt.so`) collectively cost about 2–3 W steady-state.
- **tok/W −0.25 to −0.59%** — direct consequence of the power overhead with throughput unchanged.

### 7.3 Is any delta within noise?

Per the C.3 self-consistency drift floor (TPS within ~0.2%, power within ~0.5%):
- TPS deltas: ALL within drift at B ≥ 8; B=1's +0.31% is just above (CI marginally excludes zero, but only ±0.4 tok/s in magnitude).
- Power deltas: B ∈ {1, 8, 32} are above the drift floor; B = 16 is at the floor; B = 64 is within.
- tok/W deltas: B ∈ {1, 8, 32} are real (above drift); B = 16 is at floor; B = 64 is within.

### 7.4 Prefill MFU vs decode MFU — where does CIPHER's overhead live?

Comparing cache-OFF clean MFU (where the formula is honest — section 5.5 above):

- *Prefill MFU* — vanilla 59.6–65.7%, CIPHER 59.1–65.6% across all batch sizes. Deltas are ≤ ±1% and not statistically significant at B ≥ 8. The B=1 CIPHER number is *slightly* below vanilla (−0.81% significant) — this is the small TTFT inflation visible in the cache-OFF B=1 line (vanilla 25.18ms → CIPHER 25.39ms, p99 +0.21ms).
- *Decode MFU* — climbs with B because B query-tokens-per-step share the same weight read. CIPHER and vanilla within ±0.3% across all batch sizes, none significant under bootstrap CI at B ≥ 8.

CIPHER's overhead lives in the **CPU-side scheduling + runtime injection** (page-tracking metadata for the VMM allocator; dedup-hash tracking, even when no peer process exists; CUDA_INJECTION64_PATH symbol-table walks at first launch), **not in the model compute kernels**. This is the right place for it — the substrate is designed to introduce zero work into the model's hot kernel path, and the cache-OFF prefill-MFU equivalence verifies that empirically.

### 7.5 HBM trajectory — KV-dedup capacity effect visible?

**Not at single-tenant.** vLLM pre-allocates `gpu_memory_utilization × HBM` (~70 GiB) up front, so peak HBM is dominated by the pool reservation, not by KV-cache fill. To see capacity lift, we need multi-process tenancy with shared system prompts — that's the W5 N=2 / N=4 measurement track, which produced the 42 GiB saving headline and lives in `cipher_vllm_plugin/tests/sc_kvdedup_n2.py` (single-process B-sweep cannot reproduce it).

---

## 8. What this redo does and does not retract

**Retracts (about the old harness):**

- The number `0.127% Mistral-7B B=1 decode MFU` reported in the Week-5 post-close as the "single-instance MFU" — that figure was the result of (TPS-divides-by-prefill+decode) ÷ (peak FLOPs) and is dimensionally inconsistent. The corrected B=1 decode MFU on this harness is **0.249%** (cache ON) / **0.249%** (cache OFF) — within ~2× of the legacy number; the legacy number was off-by-formula, not off-by-substrate.

- The pattern of attributing single-instance B=1 deltas at the 1-5% magnitude to "CIPHER lift" — on the corrected harness, single-instance deltas are *real* but typically a small *regression* (power), not a lift.

**Does NOT retract:**

- The Week-5 N=2/N=4 KV-dedup HBM headline (45.8 GiB / 42 GiB) — that is a different harness, a different (multi-tenant) configuration, and a different mechanism (cross-process page sharing). It is unaffected by anything in this redo.
- The 2026-05-19 cross-tenant TPW re-test result `Mistral-7B N=4 3.06× / TinyLlama N=8 3.30× / N=12 4.73× / N=16 5.98×`. Those measurements use multi-tenant orchestration, which the single-process bench harness here cannot reach.

**Confirms:**

- `[CIPHER TPW re-test + 3.6× correction]` memory: single-instance has no lift; the lever is cross-tenant batching.

---

## 9. Artifacts

```
/tmp/bench_v2/bench_llm.py
/tmp/bench_v2/compare_runs.py
/tmp/bench_v2/results/20260523_mistral7b_vanilla_agentic_fixed1024/         (cache ON)
/tmp/bench_v2/results/20260523_mistral7b_cipher_agentic_fixed1024/          (cache ON)
/tmp/bench_v2/results/20260523_mistral7b_vanilla_agentic_fixed1024_nocache/ (cache OFF)
/tmp/bench_v2/results/20260523_mistral7b_cipher_agentic_fixed1024_nocache/  (cache OFF)
/tmp/bench_v2/results/comparison_cache_on/                                  (table + plots + significance JSON)
/tmp/bench_v2/results/comparison_cache_off/                                 (table + plots + significance JSON)
```

All directories contain `result.json` (full per-request data) and `summary.md` (the human-readable table) per the harness's reporter.

---

## 10. Next

- **B1 / B2 multi-tenant Option-1** (≥ 2 tenants per GPU, KV-dedup engaged): this is where the lift mechanism is supposed to fire. The next harness step is to add multi-process orchestration to `bench_llm.py` *or* use `cipher_vllm_plugin/tests/sc_kvdedup_n*.py` directly. The current harness is the right *measurement instrument*; we need to pair it with the right *workload*.
- **VOLT + SM-partition envelopes:** these need separate enable flags and a workload structured to expose them (decode-only, or co-resident multi-tenant). Not on the Option-1 redo critical path.
- **Open-ended `gen_len` distribution** (ShareGPT replay): the deterministic `ignore_eos=True` workload here is required for apples-to-apples FLOP accounting; an open-ended ShareGPT replay would give a more realistic latency picture and is a natural Week-6 follow-on.
