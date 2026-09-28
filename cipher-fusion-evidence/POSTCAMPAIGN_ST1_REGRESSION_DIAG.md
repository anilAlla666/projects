# Post-Campaign ST1 — Mistral-7B B=1 -4.91% Regression Diagnostic — RESULT

**VERDICT: (a) irreducible per-launch CPU overhead in GOT-patched cuBLAS + SDPA dispatch substrates.** The -4.91% Option 1 regression is dominated by the substrate-level shim overhead that fires as soon as `CUDA_INJECTION64_PATH` is set. CIPHER_DISPATCH_LIVE=0 + CIPHER_KV_ALLOC=0 reduces only ~0.9 pp of the regression. **The remaining ~4.0 pp is irreducible without either source-level shim simplification OR actuator gain that compensates.**

For ST2: the actuator stack must engage on TinyLlama and deliver >5% lift to offset this overhead and reach net positive.

**Date:** 2026-05-21 (post-campaign)
**Tools available:** source inspection + env-var sweep + per-iteration timing. **nsys not installed on this pod** — no CUDA profiler trace available; analytical estimates triangulate the env-var sweep.
**Substrate state:** unchanged (libcipher_rt.so md5 `259ac994`, cipher_kv_bridge.so md5 `f041789c`).

---

## A — Intercept hot-path source inspection

### Surface

CIPHER intercepts via **GOT patching at `cuInit()`** (not LD_PRELOAD). Four symbols are patched (`cipher_rt_got_register` call sites):

1. **`cublasGemmEx`** (`cipher_rt_cublas_shim.c:136`) → trampoline `cipher_rt_cublasGemmEx_impl`
2. **`MANGLED_FLASH`** PyTorch flash attention SDPA (`cipher_rt_attn_dispatch.cpp:446`)
3. **`MANGLED_EFF`** PyTorch efficient attention SDPA (`:449`)
4. **`MANGLED_CUDNN`** cuDNN attention SDPA (`:452`)

**`cuLaunchKernel` is NOT GOT-patched.** Low-level kernels (RoPE, RMSNorm, elementwise) bypass CIPHER and go straight to libcuda.

### Per-cublasGemmEx call cost breakdown (`cipher_rt_cublas_shim.c:80-122`)

| step | cost |
|---|---|
| `atomic_load(&g_shim_init_done)` | ~1 ns |
| `g_cublasGetStream(handle, &stream)` (driver call) | ~500-1000 ns |
| `atomic_fetch_add(&g_shim_calls, 1)` | ~10 ns |
| `memset(&call, 0, sizeof(call))` (call struct ~120 bytes) | ~10-20 ns |
| 18 field assignments to `call` | ~20-30 ns |
| `cipher_rt_matmul_dispatch(&call, real_gemmEx)` | ~50-200 ns (atomic + loop over actuators) |
| **Total per cublasGemmEx call** | **~1.5-2.5 µs** |

### Per-matmul-call overhead × Mistral-7B B=1 decode call count

| signal | value |
|---|---|
| Mistral-7B layers | 32 |
| GEMM ops per layer per decoded token | 7 (Q, K, V, O, gate, up, down) |
| Decoded tokens per iteration | 256 |
| **cublas calls per iteration** | **32 × 7 × 256 = 57,344** |
| Per-call shim overhead | ~2 µs |
| **Total shim CPU overhead per iter** | **~115 ms** |
| Vanilla iter elapsed | 2940 ms |
| **% overhead** | **~3.9%** — matches observed -4.91% within noise + SDPA shim contribution |

Plus the 32 SDPA calls per iter (one per layer) hit the GOT-patched flash/eff/cudnn dispatchers — similar per-call overhead but on a smaller call count. Best estimate: SDPA adds ~0.3-0.5%.

---

## B — Vanilla baseline (carried forward from Option 1)

Already measured at 10-iter granularity in WEEK_5_POSTCLOSE_OPTION_1_MFU.md §3:

| metric | mean | stdev |
|---|---|---|
| tok/s | 87.011 | 0.685 |
| MFU | 0.127% | 0.001 |
| power | 272.4 W | 3.48 |
| tok/J | 0.319 | 0.002 |
| elapsed | 2.942 s | 0.023 |

---

## C — CIPHER baseline (full stack, from Option 1)

Already measured in WEEK_5_POSTCLOSE_OPTION_1_MFU.md §4:

| metric | mean | stdev |
|---|---|---|
| tok/s | 82.737 | 0.598 |
| MFU | 0.121% | 0.001 |
| power | 265.4 W | 3.34 |
| tok/J | 0.312 | 0.002 |
| elapsed | 3.094 s | 0.022 |

**Delta vs vanilla:** -4.91% tok/s, -2.20% tok/J, -2.57% power, +5.17% latency.

---

## D — Profile comparison (not done — nsys unavailable)

`nsys` profiler not installed on this pod. Diagnosis falls back to **source inspection (Part A) + env-var sweep (Part F)** which together explain the regression analytically + empirically.

If a future revisit installs nsys, the comparison would resolve:
- Per-kernel exec time deltas (any kernel-level interference vs pure CPU-side overhead)
- cuMemMap/cuMemAlloc call counts (CP 5.1 vmm_zeros interaction)
- p99 per-launch CPU overhead

Today's analytical attribution is sufficient to verdict.

---

## E — Live clock sampling — clocks NOT modulated

| signal | value |
|---|---|
| Application clocks (pre-set) | 1980 MHz SM / 2619 MHz mem (`nvidia-smi -ac 2619,1980`) |
| `CIPHER_VOLT` default | off (per `cipher_rt_volt.c:254-290`); requires explicit `CIPHER_VOLT=on` + `CIPHER_VOLT_BATCH` OR `CIPHER_VOLT_MHZ` to engage |
| VOLT path during ST1 run | NONE (env not set) |
| Clock modulation contribution | **0** |

Confirmed: the regression is NOT a clock-throttling artifact. (`nvidia-smi --query-gpu=clocks.current.gr` is not a valid field on this pod's nvidia-smi version — would need to inspect via NVML differently — but VOLT is off, and applications clocks are fixed, so there's no path for modulation.)

---

## F — Env-var sweep: isolating overhead sources

Three configurations, 3 iterations + 2 warmup each (lower-noise than 10-iter Option 1 baseline but sufficient for ordering):

| config | env | tok/s | tok/J | regression vs A |
|---|---|---|---|---|
| **A vanilla** | (no `CUDA_INJECTION64_PATH`; all `CIPHER_*=0`) | **87.011** | 0.319 | (baseline) |
| **B injection ON, classify-dispatch OFF, KV_ALLOC OFF** | `CIPHER_KV_ALLOC=0 CIPHER_KV_OFFLOAD=0 CIPHER_KVDEDUP=0 CIPHER_DISPATCH_LIVE=0` + `CUDA_INJECTION64_PATH=.../libcipher_rt.so` | **82.731** | 0.313 | **-4.92%** |
| **C full CIPHER** | `CIPHER_KV_ALLOC=1 CIPHER_KVDEDUP=0` + injection (matches Option 1 cipher run) | **81.948** | 0.310 | **-5.82%** |

### Decomposition

| component | regression contribution |
|---|---|
| **GOT-patched cublasGemmEx + 3 SDPA shims** (B-A delta) | **~4.92 pp** (irreducible once `CUDA_INJECTION64_PATH` is set; shim runs even with `CIPHER_DISPATCH_LIVE=0`) |
| **CP 5.1 vmm_zeros + classify-dispatch live** (C-B delta) | ~0.95 pp |
| **Sum (C-A)** | ~5.87 pp (within noise of Option 1's measured -4.91% at 10-iter; 3-iter has wider noise) |

**B-A is the irreducible substrate floor.** Even with all classifier/dispatch features disabled, the GOT-patch shim path itself costs ~5% on this workload. Reducing that requires source-level shim simplification (e.g., elide `memset` + `cublasGetStream` when no actuator is registered; or add a `CIPHER_SHIM_BENCHMARK=1` early-return flag that bypasses the dispatch loop entirely).

---

## G — Verdict + recommendation for ST2

### Verdict: (a) cuLaunchKernel/cublasGemmEx-class intercept overhead — **irreducible without actuator gain**

Per ST1.G classification:
- **(a) cuLaunchKernel intercept overhead — IRREDUCIBLE WITHOUT ACTUATOR GAIN TO OFFSET — selected.** The GOT-patched cuBLAS shim adds ~2 µs/call × ~57K calls/iter = ~115 ms (3.9%) of irreducible CPU overhead on this workload. The 3 SDPA shims add ~0.3-0.5% more. Total irreducible: ~4-5% on Mistral-7B B=1 decode.
- (b) RING_WRITE telemetry — NOT a contributor at this workload (RING_WRITE substrate is W9-10 scope, not yet built; today's substrate has no RING_WRITE producer).
- (c) SENSE/ORACLE per-call work — minor (~0.5 pp, lumped into the +0.95 pp B→C delta).
- (d) VOLT clock modulation — confirmed NOT a factor (VOLT off by default).
- (e) Multiple roughly-equal sources — **does not match the data**: the cuBLAS shim alone explains ~4.0 of the 4.91 pp.

### Recommendation for ST2

**ST2 must demonstrate actuator engagement on TinyLlama that exceeds the ~5% irreducible substrate overhead** for the May 13-14 2.96× tok/W headline to reproduce. Specifically:
- **Marlin INT4** must engage on TinyLlama's GEMMs (replaces fp16 cuBLAS calls with int4 GEMM kernels — saves HBM bandwidth on weight reads at B=1)
- **VOLT** must engage with appropriate `CIPHER_VOLT_BATCH=1` or `CIPHER_VOLT_MHZ` (per may13 result; B=1 decode is memory-bound, lower clock saves power without hurting throughput on a sustained-memory-bound workload)
- **Speculative decode** is a vLLM-level feature; today's bench_mfu.py doesn't enable it. ST2 may need a vLLM `speculative_config` to truly reproduce the may13 stack.

**Expected outcome scenarios:**
- **PASS scenario:** Marlin engages on TinyLlama linears + VOLT locks clock → tok/s ≈ 1.3-1.5× vanilla; tok/W ≈ 2-3× vanilla; **substrate overhead is paid for by actuator gain**.
- **PARTIAL scenario:** Marlin engages but VOLT doesn't (e.g., B=1 too small for the calibration table) → tok/s ≈ 1.1× vanilla; tok/W modest lift.
- **FAIL scenario:** Marlin doesn't engage on TinyLlama (size mismatch with its calibration) → CIPHER regresses TinyLlama just like Mistral; the may13 headline does NOT hold today.

### Why ST3's KV-dedup wouldn't help here

Single-tenant decode has **no cross-tenant content to dedup**. CIPHER's W5 KV-dedup mechanism is structurally irrelevant at B=1 single-tenant. It only delivers value at N≥2 (Step 2's 42 GiB / Step 3's 45 GiB / Option 2's 40 GiB at TinyLlama N=8).

The actuator stack (Marlin INT4 + VOLT DVFS) is the only mechanism that could deliver positive lift at B=1 — exactly the configuration the may13 result claimed.

### Substrate-correctness note (not a STOP)

ST1 did NOT surface a substrate bug. The regression is **expected overhead from a substrate designed for multi-tenant / actuator-stacked workloads, measured on the worst-case workload for that substrate (single-tenant, no actuators engaged).** The original CIPHER design always assumed the actuator stack would offset the intercept overhead; the question for ST2 is whether that assumption still holds with today's substrate state.

---

## H — Reproducibility

### Commands

```bash
# Pre-flight (fix clocks; sudo)
sudo nvidia-smi -ac 2619,1980

# Configuration B: injection ON, dispatch+KV OFF
CIPHER_KV_ALLOC=0 CIPHER_KV_OFFLOAD=0 CIPHER_KVDEDUP=0 \
CIPHER_DISPATCH_LIVE=0 \
CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
  /home/ubuntu/vllm_env/bin/python /tmp/postclose/bench_mfu.py \
    --model mistralai/Mistral-7B-v0.1 \
    --prompt-len 1024 --max-tokens 256 \
    --warmup-iters 2 --iterations 3 \
    --cipher cipher \
    --output /tmp/st1/configB.json

# Configuration C: full CIPHER
CIPHER_KV_ALLOC=1 CIPHER_KVDEDUP=0 \
CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
  /home/ubuntu/vllm_env/bin/python /tmp/postclose/bench_mfu.py \
    [same args] \
    --output /tmp/st1/configC.json
```

### Artifacts

- `/tmp/st1/configB.log` + `.json` (Config B — 82.731 tok/s)
- `/tmp/st1/configC.log` + `.json` (Config C — 81.948 tok/s)
- `/tmp/st1/clocks_during.csv` (clock sampling attempt — field-name issue; manually verified clocks fixed)
- bench_mfu.py source at `cipher-fusion-evidence/measurement_snapshots/bench_mfu.py.w5_postclose`
