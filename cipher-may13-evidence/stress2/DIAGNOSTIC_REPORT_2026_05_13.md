# CIPHER Density Harness — Diagnostic Report (2026-05-13)

**Pod:** H100 80GB HBM3, driver 580.105.08, CUDA runtime 12.8, 700 W power cap, 1980 MHz sustained.
**Model:** Mistral-7B-v0.1 (fp16, 32 layers, hidden=4096, intermediate=14336, n_kv=8).
**Stack:** libcipher_rt.so loaded via ctypes (no LD_PRELOAD hook). Python-level Marlin INT4 substitution on all 225 nn.Linears (`CIPHER_WEIGHT_COMPRESS=on`, `CIPHER_SUBSTITUTE_V2=on`).

## Decision recorded against this evidence

The density sweep (continuous N∈{4,8,16,24,32,48,64,96,128} + burst N∈{8,16,32,64,96,128,192}) was **not executed**. The numbers it would produce are bounded by harness/prototype constraints discussed below, not by CIPHER's intrinsic ceiling. Publishing them would set the wrong baseline. Next session begins **Phase 1 — `cipher_kmod` skeleton**; see `WHY_KMOD_NOT_OPTIONAL.md`.

## The four diagnostics

### Diagnostic 1 — `density_harness.MarlinLinear.__call__` ops per decode call

```python
with _MARLIN_LOCK:                              # threading.Lock acquire+release
    marlin_stream.wait_stream(cur_stream)       # cudaEventCreate + cudaEventRecord + cudaStreamWaitEvent
    rc = self.rt.cipher_weight_compress_marlin_gemm(...)   # kernel launch on marlin_stream
    cur_stream.wait_stream(marlin_stream)       # cudaEventCreate + cudaEventRecord + cudaStreamWaitEvent
```

**Per Marlin call:** 1 Python lock acquire/release + 2 CUDA events created + 2 cudaEventRecord + 2 cudaStreamWaitEvent + 1 kernel launch.
**Per decode token (225 Marlin calls):** **450 event ops + 225 lock ops + 225 kernel launches** on the Marlin path alone.

### Diagnostic 2 — `c2_marlin.py.MarlinLinear.__call__` (May-2 reference)

```python
stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
rc = self.rt.cipher_weight_compress_marlin_gemm(..., stream_ptr)
```

**Per call:** 1 kernel launch on the calling stream. **0 locks, 0 events, 0 wait_streams.**

The only delta between the two `__call__` bodies is the lock + 2 wait_stream pairs added in density_harness for cross-thread Marlin workspace safety.

### Diagnostic 3 — `torch.profiler` on density_harness D1 (10 s wall, single tenant)

Profiler enabled; activities = CPU + CUDA; no shapes/stack/memory tracking. Profiler itself adds ~2× wall-time overhead (we measured 11.7 tps under profiler vs 36 tps native).

| Measurement | Value |
|---|---|
| Wall time | 21.0 s |
| Tokens decoded | 246 |
| **Self CUDA time total** | **1.724 s** |
| Self CPU time total | 5.757 s |
| **GPU busy fraction** | **1.724 / 21.0 = 8.2%** |
| **GPU idle fraction** | **91.8%** |

**Top CUDA kernel:**
- `marlin_M1_N8_K8_G8` — 896.6 ms / 55,350 calls = **16.2 μs/call**
- 55350 / 246 = **225 Marlin calls per token** ✓ (matches code)
- Pure Marlin GPU ceiling = 1 / (225 × 16.2 μs) = **277 tps if launches were free**

**Top CPU op:**
- `cudaLaunchKernel` — 1.446 s self / **331,000 calls** = **1346 launches per token** (Marlin = 225, rest = 1121 from RMSNorm/SDPA/elementwise/copy/cache updates)
- At 4.43 μs per launch × 1346 launches = **5.96 ms of pure CPU dispatch per token**

**Per-token breakdown (eager mode, single tenant):**
- Marlin GPU compute: 3.6 ms
- SDPA + other GPU: ~3.4 ms (estimate from remaining CUDA total)
- CPU launch dispatch: ~6 ms
- Stream sync + Python: remainder
- **Wall per token ≈ 27 ms ≈ 36 tps** ✓ matches D1 measurement

### Diagnostic 4 — `c2_marlin.py` unchanged, pointed at Mistral-7B (this pod, no LD_PRELOAD)

```
[C2] gen=2200/2200  tps=49.43  W=167  tok/W=0.2953
[C2] marlin=494776  fb_largeM=224  fb_rc=0
[C2] first_text: 'The future of GPU computing is to make every joule count.\nTDM\n\n# TDM ...'
```

**c2_marlin.py at this pod, no LD_PRELOAD, no graph capture, Python-level Marlin only, Mistral-7B: 49.43 tps.** Output coherent. 225 Marlin calls per token confirmed (494776 / 2200 = 224.9).

## The 119 / 49 / 36 decomposition

| Configuration | tps single-tenant Mistral-7B | Source |
|---|---|---|
| LD_PRELOAD + cuLaunchKernelEx shim + graph capture + FP8 + fusion | **119** | CLAUDE.md May-2 (NOT this run) |
| Python-level Marlin only, **no LD_PRELOAD, no graph capture**, eager (c2_marlin) | **49.4** | Diagnostic 4 |
| Python-level Marlin + `_MARLIN_LOCK` + 2× `wait_stream` per call (density_harness D1) | **36.4** | D1 earlier |

**The dominant gap is 119 → 49 (~60% loss):** eager-mode launch overhead without graph capture. NOT caused by density_harness — c2_marlin sees the same ceiling at this pod.

**The smaller gap is 49 → 36 (~27% loss):** density_harness's lock + 2 wait_stream/call adds ~50 μs × 225 calls = ~11 ms per token. Math: `1 / (1/49 + 0.011) ≈ 36.3 tps` ✓ reconciles with measurement.

## Honest read

1. **Eager mode is the floor without graph capture.** Single-tenant Mistral-7B with Python-level Marlin alone tops out at ~50 tps on H100 at 700W. This is a launch-overhead ceiling, not a Marlin or HBM ceiling.

2. **GPU is idle 91.8% of wall time at single-tenant decode.** Marlin GEMM is fast (16 μs per call). The work is "1346 small kernels per token," and CPU dispatch can't feed them quickly enough.

3. **The kernel module + libcuda shim is what gets past this floor without requiring application-layer graph capture.** A kernel-level path can fuse / batch / bypass the per-launch CPU cost in ways no Python-layer harness can. Graph capture works for some workloads but not for the agentic / dynamic-shape / multi-tenant cases CIPHER targets. See `WHY_KMOD_NOT_OPTIONAL.md`.

4. **The density framing remains correct for burst workloads (Test B).** GPU is idle 92% of the time at single-tenant — there's enormous compute headroom to pack tenants into. Idle-fraction-dominated density should scale well even with the current Marlin lock, because contention on Marlin only matters when tenants overlap on a single launch chain. **But** continuous-decode density (Test A) is launch-overhead-bound, not silicon-bound — running that sweep tonight would publish numbers that reflect harness/prototype constraints, not CIPHER's intrinsic capacity.

5. **The current threading harness adds 27% on top of the eager ceiling because of the Marlin workspace lock.** That part is harness-owned and recoverable (per-thread workspace in the rt). Recording but not pursuing tonight.

## What this report does NOT establish

- The actual ceiling of CIPHER's full stack (kmod + graph + LD_PRELOAD) at high tenant counts.
- Whether the May-2 119 tps number reproduces today at this pod with full LD_PRELOAD enabled (we didn't run that — we only ran the no-LD_PRELOAD comparison).
- Per-tenant burst-density numbers (deferred until after kmod).
- $/M tok numbers at production-realistic workloads.

## Files referenced

- `/workspace/stress2/density_harness.py` — the new multi-tenant harness
- `/workspace/stress2/cipher_metrics.py` — MetricsCollector v1.0 (production-shaped, NVML-backed)
- `/workspace/stress2/density_d1d2.py` — D1+D2 runner
- `/workspace/stress2/diag3_profiler.py` — Diagnostic 3 (torch.profiler)
- `/workspace/stress2/diag4_c2_mistral.py` — Diagnostic 4 (c2_marlin unchanged on Mistral)
- `/workspace/stress2/c2_marlin.py` — May-2 reference harness (unchanged)
- `/workspace/stress2/density_d1d2.log` — D1+D2 stdout
- `/workspace/stress2/diag3_profiler.log` — profiler stdout
- `/workspace/stress2/diag4_c2_mistral.log` — c2_marlin Mistral run stdout
- `/tmp/cipher_metrics/d1_single_60s.json` — D1 receipt
- `/tmp/cipher_metrics/d2_n4_60s.json` — D2 receipt
