# Why the kernel module is not optional

**Architectural intelligence gathered on 2026-05-13.**
Companion to `DIAGNOSTIC_REPORT_2026_05_13.md` and `HARNESS_LIMITATIONS.md`.

## The empirical wall

A single H100 80GB on this pod, decoding Mistral-7B at batch=1 in eager mode through `libcipher_rt.so`'s Python-level Marlin INT4 substitution, peaks at **~49 tps**. Measured via `c2_marlin.py` unchanged. No LD_PRELOAD shim. No CUDA graph capture. The GPU is **idle 91.8% of wall time** during that run — confirmed by `torch.profiler` (Self CUDA total 1.72 s out of 21 s wall).

The bottleneck is not Marlin (16.2 μs/call × 225 calls = 3.6 ms/token of pure Marlin GPU compute, well under the 27 ms/token wall). The bottleneck is **CPU-side kernel launch dispatch**: 1,346 `cudaLaunchKernel` invocations per decoded token at 4.43 μs each = 5.96 ms/token just in driver dispatch, on top of Python overhead, on top of stream sync.

This is a property of eager-mode PyTorch on a transformer decode at batch=1. It is not a CIPHER property. It is also not a problem CIPHER can solve from where it currently sits — at the user-space library/LD_PRELOAD layer, with userspace stream sync and userspace dispatch.

## What graph capture gives you, and why it isn't enough

CUDA graph capture replays a recorded sequence of kernel launches as one driver call. On Mistral-7B decode, capturing the forward replaces ~1,300 of those 1,346 per-token launches with a single graph launch. The CLAUDE.md May-2 entry "Phase 2 — Marlin INT4 GEMM via NVRTC" reports the resulting boost: **Mistral-7B B=1 from baseline 115 → 119 tps** (graph + Marlin + fusion + FP8); B=8 jumps to **1.38× over baseline** because the launch ceiling fully relaxes.

So graph capture works. The catch: **it works only where the workload is graph-capturable.**

- **Agentic workloads** with branch-on-output: cache positions advance, control flow depends on the decoded token, stop conditions vary by tenant. Capture-once, replay-many doesn't apply.
- **Dynamic shapes**: variable batch sizes, variable cache lengths, variable sequence lengths across requests. Each shape combination needs its own captured graph. Combinatorial explosion.
- **Multi-tenant scheduling**: tenants enter and leave continuously; the schedule is dynamic; the graph would need to be rebuilt per scheduling decision. Defeats the point.
- **Custom op + ctypes interaction**: every kernel launched outside PyTorch's tracer (every ctypes call into a third-party rt — i.e., CIPHER's Marlin path itself) is invisible to capture. See `HARNESS_LIMITATIONS.md` §2 for the empirical evidence: a graph captured around a density_harness forward would replay everything *except* the Marlin GEMMs. The exact bug class CLAUDE.md's Phase 3 stream-capture audit caught.

Graph capture is a workload-specific optimization. It does not generalize to the workloads CIPHER targets as a device-layer hypervisor underneath vLLM / TensorRT-LLM / SGLang / arbitrary customer code.

## What kmod gives you that nothing else can

The kernel module + libcuda shim moves CIPHER below the eager-mode dispatch layer entirely. Three things become possible that no application-layer optimizer can do:

### 1. Dispatch-batching without graph capture

The shim intercepts `cuLaunchKernel` / `cuLaunchKernelEx` calls below the userspace driver. It can buffer launches submitted to the same stream within a small time window (microseconds) and submit them as one driver call — same effect as graph capture, no graph capture required. The user code remains eager. Their dynamic-shape, branch-on-output, multi-tenant scheduling all still works. **The 1,346-launches-per-token CPU dispatch tax disappears at the layer the user neither sees nor has to refactor.**

This is the structural answer to the 91.8% GPU idle problem and is not achievable from libcipher_rt.so alone because user-space can't dispatch-batch without breaking ordering guarantees the application depends on. Below userspace, the driver shim controls ordering and can prove ordering equivalence.

### 2. Per-stream context partitioning (Green Context productionization)

The current LD_PRELOAD prototype's Marlin workspace race forces serialization (`HARNESS_LIMITATIONS.md` §1). With kmod controlling stream allocation at the driver level, every tenant gets a kernel-enforced resource partition (SM count, L2 quota, HBM bandwidth budget). The workspace race goes away because tenants don't share workspaces — they share *the GPU*, allocated by the kernel, with quotas the kernel enforces. **Multi-tenant scales without harness gymnastics.**

### 3. Compliance + audit at the trust layer customers actually accept

LD_PRELOAD is a procurement blocker for serious enterprise / sovereign-cloud / regulated deployments (`feedback_cipher_engineering_priorities.md` §4). A signed kernel module installed via `.deb`/`.rpm` with `nvidia-modprobe`-style registration is the trust shape every neocloud security team will sign off on. The compliance observers (CARBON/TRACE/RECEIPT/FAIRNESS/COMPLY) get hardware-rooted trust instead of user-space-overridable trust. **That's the regulatory wedge made real.**

## The structural advantage no application-layer optimizer can match

Every competitor in the inference-acceleration space operates above the userspace driver:

- **vLLM / SGLang / TGI**: PyTorch / Python level. Subject to the same eager-mode launch ceiling we measured tonight.
- **TensorRT-LLM**: TensorRT graph level. Excellent for static graphs; brittle for dynamic / agentic.
- **CUTLASS / FlashAttention kernels**: kernel level. Improves individual op throughput; cannot fix dispatch.
- **MIG / NVIDIA Triton**: hardware/server level. Coarse partitioning; not workload-aware; tenant changes require restart.

CIPHER with kmod sits in a layer none of them occupy: between the userspace driver and the kernel-side GPU driver, with workload-awareness inherited from the rt's observers and substitution engine. Anyone trying to copy this would need to either:
- Convince NVIDIA to ship the equivalent in CUDA (long lead time, conflicts with their hardware-selling incentives)
- Build their own kernel module with the same set of substitution kernels + audit trails (years of work + the Marlin/KIVI/fusion engineering is non-trivial)
- Match it at the userspace layer and fail at the same launch-overhead wall we hit tonight

This is **marvel-grade differentiation**: not a 1.5× benchmark win that gets eroded by the next library release, but a layer of the stack that nobody else has standing in.

## Phase 1 entry criteria (next session)

The diagnostic evidence in `DIAGNOSTIC_REPORT_2026_05_13.md` makes the case. Phase 1 = `cipher_kmod` skeleton. Minimum viable scope:

1. Out-of-tree Linux kernel module (`cipher_kmod.ko`) registering a character device `/dev/cipher` with `ioctl` interface.
2. Userspace shim `libcipher_cuda.so` interposing on `cuLaunchKernel` / `cuLaunchKernelEx` / `cudaStreamSynchronize`, forwarding to kmod via the `ioctl` interface or shared memory ring.
3. Dispatch-batching path: collect launches submitted within W μs, batch-submit. Default OFF; gated by `CIPHER_KMOD_BATCH=on`.
4. Single end-to-end test: Mistral-7B B=1 single-tenant decode → kmod path → measure tps. Target: > 49 tps (the eager ceiling). Success metric: shows the launch tax is genuinely paid by kmod, not just shuffled.
5. Compatibility test: c2_marlin.py runs unchanged through the kmod path. (Validates `LD_PRELOAD`-equivalent semantics.)

Out of scope for Phase 1: SM partitioning, multi-tenant scheduling, signed-package distribution, observability hooks. Those land in Phase 2-4.

## TL;DR for the slide

> **The LD_PRELOAD prototype hits a wall at ~50 tps single-tenant Mistral-7B decode because eager-mode CUDA dispatch costs 1,346 kernel launches per token. CIPHER's kernel module + libcuda shim moves dispatch-batching, partition enforcement, and audit below the userspace driver — solving the same problem CUDA graph capture solves, without requiring graph-capturable workloads. This is the architectural advantage that makes CIPHER the layer every neocloud needs: not a faster kernel library, a *different layer of the stack* that nothing else occupies.**
