# CIPHER Density Harness — Known Limitations (recorded 2026-05-13)

This document captures harness-layer constraints discovered while building the multi-tenant density-sweep substrate on 2026-05-13. Read this before iterating on the harness or interpreting any sweep result it produces.

## 1. Marlin workspace race forces serialization

**Issue.** `cipher_weight_compress_marlin_gemm` (in `libcipher_rt.so`) uses a module-scope `cudaMalloc`'d workspace for split-K coordination, atomic counters, and partial-reduce buffers. The workspace is single-instance for the whole rt module. Two threads concurrently launching Marlin GEMMs to different CUDA streams cause the kernels to execute concurrently on the GPU and to touch the same workspace bytes → corruption.

**Current mitigation (density_harness.py).** A module-scope `_MARLIN_LOCK: threading.Lock` plus a shared `_MARLIN_STREAM: torch.cuda.Stream`. Every `MarlinLinear.__call__` does, under the lock:

1. `marlin_stream.wait_stream(cur_stream)` — Marlin stream waits for input to be ready
2. enqueue `marlin_gemm` on `marlin_stream`
3. `cur_stream.wait_stream(marlin_stream)` — calling stream waits for output

This serializes all Marlin work on a single CUDA stream (GPU-side FIFO order = no concurrent workspace touch) while letting non-Marlin ops (RMSNorm/SDPA/elementwise) overlap across tenant streams.

**Cost.** Per Marlin call: 2 `cudaEventCreate` + 2 `cudaEventRecord` + 2 `cudaStreamWaitEvent` ≈ ~50 μs of CPU overhead. At 225 Marlin calls per Mistral-7B decode token, this is ~11 ms/token. Empirically: single-tenant tps drops from 49.4 (c2_marlin baseline, no lock) → 36.4 (density_harness D1, with lock) = **27% overhead**.

**Long-term fix.** Per-thread or per-stream workspace in the rt. Two viable approaches:
- Workspace pool: rt maintains N workspace buffers; each Marlin call reserves one. Bounded contention.
- Caller-supplied workspace: extend `cipher_weight_compress_marlin_gemm` signature to take a workspace pointer; harness allocates per-thread.

Eliminates the lock, the shared marlin_stream, and the per-call event pair. Expected single-tenant tps recovers to ~49 (eager-mode floor; the further ceiling is separate — see §3).

**Why we did not fix tonight.** ~2-3 hours of C++ work in `cipher_weight_compress.cpp` plus rebuilding `libcipher_rt.so`. Doing it without first deciding whether the harness is the right vehicle (versus kmod) would have been wasted scaffolding.

## 2. Graph capture and ctypes Marlin calls do not coexist

**Issue.** The May-2 119 tps number for Mistral-7B (CLAUDE.md, "End-to-end Mistral-7B speedups") required CUDA graph capture of the decode forward. In the current Python-level harness, every Marlin GEMM is called through ctypes into `libcipher_rt.so`. `cudaStreamBeginCapture` does not capture ctypes-issued kernel launches: they execute eagerly inside the capture region rather than being recorded as graph nodes. Result: a graph captured around our forward replays everything *except* the Marlin GEMMs, which run once during capture and never again — same bug class as the "stream=NULL" capture bug discovered in CLAUDE.md Phase 3.

**Implications.**
- The May-2 119 tps cannot be reproduced today using `density_harness` plus `torch.cuda.graph` — the Marlin path won't capture.
- To benefit from graph capture in the current architecture, Marlin would need to be invoked via a PyTorch custom op (registered through `torch.library`) rather than ctypes. That's ~1-2 days of glue.
- Alternatively the rt's graph_inspect / graph node rewriter could substitute cuBLAS GEMM nodes into Marlin nodes inside an externally-captured graph (CLAUDE.md Phase 3 has infrastructure for this; the actual rewriting code isn't used).
- Or — preferred — kmod sits underneath all of this and the launch-overhead problem dissolves regardless of capture (see `WHY_KMOD_NOT_OPTIONAL.md`).

**State.** Unsolved. Recorded for future sessions. Do not propose graph capture in `density_harness` as a Path-2-style fix; it will silently produce wrong numbers.

## 3. Continuous-decode density and burst density measure different things

**Issue.** The "60-150 tenants per H100 at 95-99% idle" framing is workload-conditional:

- **Burst workload (Test B):** tenants are mostly idle. At single-tenant the GPU is idle 91.8% of wall time anyway (Diagnostic 3). Adding many idle tenants packs into that headroom without causing Marlin lock contention as long as tenants don't fire bursts simultaneously. Density × idle_fraction scaling holds. **This framing is valid.**

- **Continuous-decode workload (Test A):** every tenant is always trying to decode. CPU launch dispatch is the bottleneck (1346 launches/token × N tenants overwhelms one Python interpreter quickly even before Marlin lock contention kicks in). Density does NOT scale because the rate-limiter is host-side, not silicon. **This framing is misleading for continuous decode.**

**Operational consequence.** Test A and Test B must be reported and interpreted separately. The headline external metric (tenants × idle_fraction → effective density) lives on Test B. Test A measures a different thing — the eager-mode launch ceiling — and that ceiling is what kmod is built to break.

**State.** This is by design once understood. Tonight's deferred sweep should, when run, lead with Test B as the headline and Test A as the prototype-ceiling demonstration.

## 4. Memory footprint is dominated by `orig.weight` retention

**Issue.** Mistral-7B fp16 weights = 14 GB. `MarlinLinear.__init__` creates a transposed `_wt` copy (another 14 GB) which is the rt's lookup key. We do NOT free `orig.weight` because prefill at M=128 exceeds Marlin's MAX_M=8 (per CLAUDE.md "M≥128 has no compatible Marlin kernel") and falls back to `F.linear`, which reads `orig.weight`. Total resident shared-model memory: ~28-31 GB.

**Implication for tenant count.** Per-tenant marginal memory (StaticCache + activations) is only ~50 MB for a Mistral-7B decode at prefill=128 + max_decode=200. The 80 GB - 31 GB = 49 GB available headroom comfortably fits >900 tenants by cache memory alone. **Memory is not the binding constraint at the tenant counts in this sweep.**

**Long-term.** When Marlin gains a parallel-M ≥128 path (or a separate prefill kernel), `orig.weight` can be freed, recovering 14 GB.

## 5. NVML sampling vs nvidia-smi subprocess

**Issue noted, no problem yet.** The MetricsCollector defaults to pynvml (`import pynvml; nvmlInit; nvmlDeviceGetHandleByIndex`) and falls back to `nvidia-smi --query-gpu=...` subprocess if NVML init fails. At 5 Hz, NVML adds ~1 ms per sample; nvidia-smi adds ~30-50 ms per sample (subprocess fork+exec). Receipts record which backend was used. Both D1 and D2 ran on NVML cleanly (0 errors, 0 dropped samples).

**Recorded so future maintainers don't replace NVML with subprocess "for portability."** NVML is the correct default; the subprocess fallback is an exception path.

## 6. FAIRNESS observer cap

**Issue.** `cipher_fairness_shm.cpp` has `MAX_TENANTS = 64`. Above 64 tenants, slot registration fails ("all slots full!" stderr warning) but workloads continue. Useful as a soft signal but means the Test B sweep at N=192 would see partial FAIRNESS observability.

**Workaround.** Compute `fairness_ratio = max(per_tenant_tps) / min(per_tenant_tps)` at the parent (harness) level from MetricsCollector data. This is what the harness already does — the rt's FAIRNESS observer is supplementary, not load-bearing for the metric.

**Long-term.** Bump `MAX_TENANTS` to 256 or 1024 in `cipher_fairness_shm.cpp` and rebuild. ~5 lines of code, included in the next observer-cleanup pass.

## 7. The `_orig_forward` rebind brittleness

**Issue.** `MarlinLinear.__init__` captures the original Linear's forward bound method:
```python
self._orig_forward = type(orig).forward.__get__(orig, type(orig))
```
This works for plain `nn.Linear`. If a model wraps Linear in a custom subclass or uses an accelerate hook (`_old_forward`), this capture may not see the wrapped path. CIPHER's `stress_common.patch_fusion` works around this for LlamaRMSNorm/LlamaMLP by rebinding `_old_forward` separately. The density_harness does NOT do that.

**State.** Mistral-7B in this run did not have accelerate offloading active, so plain `nn.Linear` works. Multi-GPU / device_map="auto" / disk-offload workloads would need the `_old_forward` rebind treatment. Recorded.

## Summary — what NOT to do in future sessions

- Do not run the density sweep on `density_harness` as-is and treat the numbers as CIPHER's ceiling. They include the 27% Marlin-lock overhead AND the eager-mode launch ceiling.
- Do not propose CUDA graph capture as a fix at the density_harness level — it silently breaks Marlin via the ctypes/capture incompatibility.
- Do not "fix the harness" without first deciding the harness is the right vehicle. The right vehicle for the next throughput milestone is `cipher_kmod`.
- Do not measure continuous decode and call it "tenant density." Burst is the framing that holds.
