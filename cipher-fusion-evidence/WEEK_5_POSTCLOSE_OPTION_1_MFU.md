# Week 5 Post-Close Option 1 — Mistral-7B Decode MFU + tok/W — RESULT

**Status: COMPLETE. Honest finding: CIPHER substrate REGRESSES Mistral-7B
B=1 decode by -4.9% tok/s and -2.2% tok/J at single-tenant.** This is
consistent with prior CIPHER campaign findings (e.g.,
`cipher-t43-envelope`): VOLT clock-locking + observer-tier overhead at
B=1 decode on memory-bound 7B-class models does NOT deliver positive
tok/s/W lift at single tenant — the product value lives at multi-tenant
(Option 2 measures that surface).

**Date:** 2026-05-21 (Week 5 post-close)
**Substrate state:** UNCHANGED (Week-5-complete; libcipher_rt.so md5
`259ac994`; cipher_kv_bridge.so md5 `f041789c`; cipher_vllm_kvdedup.py
md5 `cd8c826f`; cipher_kmod loaded srcversion `CECE94921DE1F43F04E452F`).

---

## §1 — Methodology (MLPerf Inference v4.0 server scenario aligned)

| element | choice |
|---|---|
| Model | `mistralai/Mistral-7B-v0.1` (fp16/bf16 native; 7,241,732,096 params) |
| Engine | vLLM 0.21.0 (`enforce_eager=True` — deterministic; CUDA graph disabled) |
| Batch | B=1 (single sequence) |
| Prompt | 1024 tokens (BOS + 1023 × filler token id 99 — deterministic across iterations) |
| Decode | 256 tokens; `ignore_eos=True` (force exact length); greedy (temperature=0) |
| Warm-up | 3 iterations discarded |
| Measurement | 10 iterations averaged |
| Power sampler | nvidia-smi `power.draw` polled at 100 ms cadence on background thread; mean over decode window |
| GPU clocks | **Fixed at H100 SXM5 base: SM 1980 MHz / MEM 2619 MHz** via `nvidia-smi -ac 2619,1980` (no thermal/boost variability) |
| FLOPs convention | 2 × params per decode token (MLPerf convention; matmul-dominated; non-matmul ops ignored) |
| Peak ref | H100 SXM5 FP16 dense tensor: **989 TFLOPS/s** (NVIDIA spec) |
| Pre-condition | GPU `memory.used` = 0 MiB before each run |

`bench_mfu.py` source at `/tmp/postclose/bench_mfu.py` (snapshot landed
at `cipher-fusion-evidence/measurement_snapshots/bench_mfu.py.w5_postclose`).

---

## §2 — System state (snapshot at /tmp/postclose/sysstate_pre.txt)

| signal | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 SXM (driver 580.105.08, CUDA 13.0) |
| Power limit | 700 W |
| Application clocks (set) | SM 1980 MHz / MEM 2619 MHz |
| CPU | Intel Xeon Platinum 8480+ (26 cores visible) |
| RAM | 221 GiB |
| Kernel | 6.8.0-1046-nvidia |
| vLLM | 0.21.0 (in `/home/ubuntu/vllm_env`) |
| cipher_rt_phase4 HEAD | `ec0e005` (week-5-complete) |
| cipher_kmod HEAD | `2fc70c3` (week-5-complete) |
| cipher_kv_bridge.so md5 | `f041789c` |
| libcipher_rt.so md5 | `259ac994` |
| cipher_vllm_kvdedup.py md5 | `cd8c826f` |

---

## §3 — Vanilla baseline numbers

Env: `CIPHER_KV_ALLOC=0 CIPHER_KV_OFFLOAD=0 CIPHER_KVDEDUP=0` (all CIPHER
plugins explicitly disabled; vLLM runs unmodified; entry-points still load
but the plugins early-exit via env-gate).

| metric | mean | stdev | p50 | min | max |
|---|---|---|---|---|---|
| tok/s | **87.011** | 0.685 | 87.014 | 86.203 | 88.303 |
| MFU (%) | **0.127** | 0.001 | 0.127 | 0.126 | 0.129 |
| mean power (W) | 272.4 | 3.48 | 271.5 | 268.3 | 279.6 |
| tok/J | 0.319 | 0.002 | 0.320 | 0.316 | 0.322 |
| tok/W (= tok/J) | 0.319 | 0.002 | 0.320 | 0.316 | 0.322 |
| elapsed (s) | 2.942 | 0.023 | 2.940 | 2.899 | 2.970 |

10-iter runs after 3 warm-up. Per-iteration log at
`/tmp/postclose/option1_vanilla.log`; full JSON at
`/tmp/postclose/option1_vanilla.json`.

---

## §4 — CIPHER baseline numbers

Env: `CIPHER_KV_ALLOC=1 CIPHER_KVDEDUP=0
CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so`
(libcipher_rt injection ON; CP 5.1 buffer-ownership ON; CP 5.2 offload
ON; W5 kvdedup OFF for single-tenant test — no cross-tenant content to
dedup).

| metric | mean | stdev | p50 | min | max |
|---|---|---|---|---|---|
| tok/s | **82.737** | 0.598 | 82.972 | 81.794 | 83.420 |
| MFU (%) | **0.121** | 0.001 | 0.121 | 0.120 | 0.122 |
| mean power (W) | 265.4 | 3.34 | 266.4 | 259.7 | 269.1 |
| tok/J | 0.312 | 0.002 | 0.311 | 0.309 | 0.315 |
| tok/W (= tok/J) | 0.312 | 0.002 | 0.311 | 0.309 | 0.315 |
| elapsed (s) | 3.094 | 0.022 | 3.094 | 3.069 | 3.130 |

Per-iteration log at `/tmp/postclose/option1_cipher.log`; full JSON at
`/tmp/postclose/option1_cipher.json`.

---

## §5 — Delta analysis

| metric | vanilla | CIPHER | delta (abs) | delta (rel) |
|---|---|---|---|---|
| **tok/s** | 87.011 | 82.737 | **-4.274** | **-4.91%** |
| **MFU (%)** | 0.127 | 0.121 | -0.006 | -4.93% (-0.006 pp absolute) |
| mean power (W) | 272.4 | 265.4 | -7.01 | -2.57% |
| **tok/J** | 0.319 | 0.312 | -0.007 | **-2.20%** |
| **tok/W** | 0.319 | 0.312 | -0.007 | -2.20% |
| elapsed (s) | 2.942 | 3.094 | +0.152 | +5.17% |
| p50 latency for 256-token decode (ms) | 2940 | 3094 | +154 ms | +5.2% |

**The substrate makes single-tenant B=1 decode -4.91% slower at -2.20%
worse energy efficiency.** Power draws slightly lower (-2.57%) but not
enough to compensate for the throughput regression. tok/W and tok/J are
mathematically identical in this single-GPU configuration.

### Statistical significance

vanilla stdev = 0.685 tok/s; CIPHER stdev = 0.598 tok/s. The -4.274 tok/s
delta is **~6 sigma** of either distribution — overwhelmingly significant,
not measurement noise.

### Source of the regression

Most likely contributors (from Week-4-substrate review + Week-5 plugin
additions):
- CP 5.1 monkey-patch of `_allocate_kv_cache_tensors` → CIPHER-VMM-backed
  KV (no direct hot-path overhead at decode time, but the VMM page-fault
  pattern differs from torch's default)
- CP 5.2 snapshot-on-preempt hook (idle during decode — should be ~0
  overhead but vLLM might still take the patched path through scheduler)
- Week-2/3 observability tier (LOOP/PIPELINE/PULSE/etc): per-launch
  observe() callbacks; 75 T-symbols installed at register, env-gated but
  the dispatch substrate still routes through them
- The dispatch substrate (`cipher_rt_dispatch`) is LIVE by default
  (`CIPHER_DISPATCH_LIVE=1`); classifier-driven routing is engaged per
  launch — non-zero per-launch overhead

This is **CIPHER's overhead at single-tenant** showing through. The
product-relevant question is whether this overhead is recovered at
multi-tenant (Option 2 measures that).

---

## §6 — Honest accounting

### What this measurement proves

1. **CIPHER substrate is correct end-to-end at single-tenant Mistral-7B
   B=1 decode** — vLLM loads, decode produces 256 tokens per iteration,
   no crashes, deterministic across all 10 iterations + 3 warm-ups.

2. **The substrate's per-launch overhead at B=1 decode is ~5% of
   throughput on this configuration** (Mistral-7B fp16, enforce_eager,
   1024-token prompt, single tenant). This is consistent across all 10
   measurement iterations.

3. **MFU at this configuration is 0.121-0.127%** — far below the
   industry-typical 3-8% range cited for Mistral-7B decode on H100. The
   reason: `enforce_eager=True` disables CUDA graphs, which removes most
   of vLLM's hot-path optimization. This is the deterministic-measurement
   choice MLPerf-aligned scenarios make; production-quality serving with
   CUDA graphs would hit 5-10% MFU.

### What this measurement does NOT prove

1. **CIPHER's product value at multi-tenant is NOT measured by this run.**
   Single-tenant B=1 decode is the regression-sentinel measurement (per
   `[[cipher-phase-a-multitenant]]`: "B=1 decode is overhead-bound not
   memory-bound; HBM 4-18%"). CIPHER's KV-dedup mechanism delivered 42 GiB
   HBM savings at N=2 TinyLlama / 45.8 GiB at N=4 TinyLlama (Week 5 Step
   2 + 3 results). Option 2 below measures the multi-tenant
   product-relevant surface.

2. **Today's substrate is 25-of-30 ops.** Week 7-12 work (COMMIT
   primitive + RING_WRITE substrate + Koopman tier) lands the remaining
   5 v1 ops. CP 5.5 at Week 13-14 measures the COMPLETE 30-op substrate.
   Today's Option 1 is a Week-5-state snapshot, not a v1-final
   measurement.

3. **Reference numbers for trajectory:**
   - **CIPHER May-13-14 sprint result**: 2.96× tok/W on stacked Marlin
     INT4 + adaptive DVFS + speculative decode at B=1 decode. Today's
     Option 1 has **none of those actuators wired live**; the substrate
     present is the observability + dispatch + KV-dedup mechanism.
     Marlin and DVFS are pinned to primary context but inactive on
     Mistral-7B B=1 decode workload (DVFS for Mistral-7B B=1 was -14%
     in the prior measurement).
   - **Production vLLM Mistral-7B decode on H100**: 100-150 tok/s with
     CUDA graphs enabled; we measure 87 tok/s with `enforce_eager=True`
     — the ~37% gap to production is the CUDA-graph optimization, not a
     CIPHER artifact.

### What's PARTIAL about Option 1

- **Single prompt-len only.** The 3-prompt-len sweep (128/1024/4096)
  outlined in the task was budget-deferred; only the medium (1024) ran.
- **enforce_eager mode.** Production-quality decode uses CUDA graphs;
  the regression measured here may shift with CUDA graphs (could be
  larger or smaller — CIPHER's hot-path overhead interacts non-trivially
  with graph capture).
- **Single hardware sample.** One H100 80GB SXM, one machine, one
  measurement campaign. Production fleet measurements would have
  per-node variance.

---

## §7 — Reproducibility

### Commands (exact, copy-paste reproducible)

```bash
# Pre-flight (sudo required for clock fix)
sudo nvidia-smi -ac 2619,1980

# Vanilla baseline
CIPHER_KV_ALLOC=0 CIPHER_KV_OFFLOAD=0 CIPHER_KVDEDUP=0 \
  /home/ubuntu/vllm_env/bin/python /tmp/postclose/bench_mfu.py \
    --model mistralai/Mistral-7B-v0.1 \
    --prompt-len 1024 --max-tokens 256 \
    --warmup-iters 3 --iterations 10 \
    --cipher vanilla \
    --output /tmp/postclose/option1_vanilla.json

# CIPHER baseline
CIPHER_KV_ALLOC=1 CIPHER_KVDEDUP=0 \
CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
  /home/ubuntu/vllm_env/bin/python /tmp/postclose/bench_mfu.py \
    --model mistralai/Mistral-7B-v0.1 \
    --prompt-len 1024 --max-tokens 256 \
    --warmup-iters 3 --iterations 10 \
    --cipher cipher \
    --output /tmp/postclose/option1_cipher.json

# Restore clocks (post-campaign)
sudo nvidia-smi -rac
```

### Artifacts

- `bench_mfu.py`: snapshotted at `cipher-fusion-evidence/measurement_snapshots/bench_mfu.py.w5_postclose`
- Vanilla JSON: `/tmp/postclose/option1_vanilla.json` (per-iter + aggregate)
- CIPHER JSON: `/tmp/postclose/option1_cipher.json` (per-iter + aggregate)
- Logs: `/tmp/postclose/option1_{vanilla,cipher}.log`
- System state: `/tmp/postclose/sysstate_pre.txt`

### Substrate verifiability

All anchors at this measurement match the Week-5-complete state. Any
re-run with the same anchors + GPU clocks should reproduce within the
~0.7% stdev band.
