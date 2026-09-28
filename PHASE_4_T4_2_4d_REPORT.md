# Phase 4 T4.2.4d — GREEN_CTX enforcement gap fixed

**Date:** 2026-05-14 afternoon session (continuing post-T4.2.4c).
**Trigger:** T4.2.4c bomb-throughput diagnostic was A=B (93.8 vs 94.4 prefills/s, Δ -0.6%) — proving the GREEN_CTX actuator was cosmetic, not enforcing. T4.2.4d closes the gap.
**Kmod substrate:** 0.4.5 unchanged (`b47db4500e226f5ddc010f9148d3d6ca`).
**Build under test:** `libcipher_rt.so.v0.2.0_T4_2_4d` md5 `50414674ddad2689191d13a92377e492`.
**Pre-T4.2.4d artifact saved:** `libcipher_rt.so.v0.2.0_T4_2_4b_v2.pre_T4_2_4d` md5 `8fbc350fb322b0733ce6d4af1468a88f`.

## TL;DR

| Question | Answer |
|---|---|
| Was enforcement actually fixed? | **YES.** Bomb solo drops from 114.1 prefills/s (libcipher_v2) to 10.4 prefills/s (T4.2.4d 8-SM partition) — **11× slowdown**. Binding-diagnostic threshold (<40 prefills/s) met decisively. |
| Root cause? | PyTorch routes 100% of kernel launches through the NULL (default) stream. The NULL stream resolves to whichever CUcontext is current to the calling thread. Push-then-create on cuStreamCreate (T4.2.4b-c approach) only affected the ~32 explicit streams PyTorch creates, not the hot path. |
| Code change? | Add `cipher_rt_green_ctx_make_current()` (persistent `cuCtxSetCurrent(green_cuctx)` with fast-path) in `cipher_rt_green_ctx.c`; call it from `cipher_cupti.c` at every cuLaunchKernel ENTER. ~30 lines net. |
| Re-run A/B verdict? | Bomb 13.9× slower under partition (enforcement confirmed in A/B context). Victim p99 +8.0% worse under partition (not the isolation we predicted). |
| Phase 4.2 ship status? | Actuator now **really enforces** (not cosmetic). It does NOT deliver tail-latency isolation on this workload pair due to the workload-resource-mismatch (already documented in T4.2.4c Caveats). |
| What's next? | Either: (a) demonstrate enforcement value on a workload pair where it has a foothold (two SM-bound tenants), or (b) ship P4.2 on existing G1/G2/G4 MFU gates + the enforcement substrate. User decides. |

## Phase 1 — CUDA Green Context docs

Two doc passages were in tension:

- **Programming Guide (Green Contexts section):** "Various execution context APIs … take an explicit `cudaExecutionContext_t` handle and thus **ignore the context that is current to the calling thread**." Implied that `cuStreamCreate` with a pushed green ctx would not bind.
- **Driver API reference (CUDA_STREAM module):** "If during stream creation the context that was active in the calling thread was obtained with `cuCtxFromGreenCtx`, that green context is returned in `phCtx`." Implied push works.

Empirically resolved by Phase 3 microbench: **the Driver API reference is correct.** `cuStreamCreate` does honor a pushed `cuCtxFromGreenCtx`-derived `CUcontext`. The Programming Guide's "ignore current context" rule applies to a different subset of APIs (cudaExecutionCtx* family).

CUDA environment: 12.8.93, driver 580.105.08, H100 SXM5 80 GB cc=9.0. All Green Context APIs present in `libcuda.so.1` including `cuGreenCtxStreamCreate`, `cuGreenCtxGetDevResource`, `cuStreamGetGreenCtx` (the last is critical for the diagnostic).

## Phase 3 — Microbench: enforcement at raw CUDA

`/tmp/test_green_ctx_enforcement.cu`: 1024-block × 256-thread compute-bound kernel, 10 launches per condition.

| Test | per-launch | ratio | partition? |
|---|---:|---:|---|
| TEST 1: primary ctx, default stream | 102.0 ms | 1.0× | No (full 132 SMs) |
| TEST 2: `cuCtxPushCurrent(green_cuctx)` + `cuStreamCreate` | 1487.5 ms | **14.6×** | **YES** |
| TEST 3: `cuGreenCtxStreamCreate` directly | 1487.5 ms | 14.6× | YES |

`cuStreamGetGreenCtx` confirmed TEST 2 and TEST 3 streams bound to the same green ctx pointer. **Outcome A from the user's plan:** enforcement works at raw CUDA level; bug is in libcipher_rt's hot-path coverage.

## Phase 3.5 — Extended microbench: NULL-stream behavior

`/tmp/test_green_ctx_nullstream.cu`: same kernel, varying current-context vs stream.

| Test | per-launch | ratio | partition? |
|---|---:|---:|---|
| TEST A: primary current, NULL stream | 102.0 ms | 1.0× | No |
| TEST B: `cuCtxSetCurrent(green_cuctx)`, NULL stream | **1487.5 ms** | **14.6×** | **YES** |
| TEST C: same as B, repeated via runtime API | 1487.5 ms | 14.6× | YES |
| TEST D: switched back to primary, NULL stream | 101.9 ms | 1.0× | No |

This is the load-bearing finding: **persistently setting the green CUcontext as the calling thread's current context makes NULL-stream launches go through the green ctx's 8-SM partition.** This identified the fix architecture (Option 1 of advisor's three) as viable.

## Phase 3 diagnostic — what PyTorch actually does

Diagnostic build added launch-time classification: for each kernel launch, classify by (a) stream==NULL, (b) `cuStreamGetGreenCtx(stream) == our green`, (c) some other green, (d) primary.

Bomb solo, 30 s under T4.2.4d diagnostic build (libcipher_rt with classification, no enforcement yet):

```
DIAG-T4.2.4d: total_launches=884493 streams_observed=32
DIAG-T4.2.4d: on_our_green=0 on_other_green=0 on_primary=0 on_null_stream=884493
```

**100.000% of bomb's 884,493 kernel launches in 30 s land on `stream==NULL`.** The 32 streams PyTorch created (and which our cuStreamCreate hook correctly bound to green ctx via push) **are never used for kernel launches in the bomb hot path**. PyTorch's `m.generate(...)` decode/prefill kernels go through the per-context default stream, which is the primary context's default stream by default.

This is the "case (b)" outcome the advisor flagged at the architecture decision point — the hot-path bypasses our cuStreamCreate hook entirely.

## Phase 4 — The fix

`cipher_rt_green_ctx.c` — new function:

```c
int cipher_rt_green_ctx_make_current(void)
{
    CUcontext cur = NULL;
    if (!atomic_load(&g_green_init_succeeded)) return -1;
    if (cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur == g_green_cuctx)
        return 0;             /* already green — fast path */
    if (cuCtxSetCurrent(g_green_cuctx) != CUDA_SUCCESS) return -1;
    return 0;
}
```

`cipher_cupti.c` — at every cuLaunchKernel ENTER (driver + runtime APIs):

```c
if (cipher_rt_green_ctx_ensure() == 0) {
    if (cipher_rt_green_ctx_make_current() == 0)
        atomic_fetch_add(&g_ctx_swaps_to_green, 1);
}
```

`cuCtxGetCurrent`/`cuCtxSetCurrent` are TLS reads/writes; the fast path is a single comparison. Per-launch overhead is in the sub-µs range — negligible against a 200 ms kernel.

The push/pop on cuStreamCreate is left intact (defensive — covers the few streams PyTorch does create). The new make_current call is the load-bearing piece.

## Binding diagnostic — the user's pass/fail test

User's threshold: "bomb solo on T4.2.4d w/8-SM partition → <40 prefills/s expected."

| Build | Bomb solo prefills/s | Verdict |
|---|---:|---|
| libcipher_v2 (no partition baseline) | **114.1** | reference |
| T4.2.4c libcipher_rt (cosmetic partition) | ~110-112 | (Δ 1-3% — cosmetic) |
| **T4.2.4d libcipher_rt (real partition)** | **10.4** | **11.0× slowdown, ✅ <40** |

`ctx_swaps_to_green=83,640` — every one of bomb's 83,640 launches in 30 s went through the make_current hook. Of those, the fast path skipped most after the first launch on each thread; only the first launch on each new thread actually called cuCtxSetCurrent.

**Enforcement is verified.**

## Phase 5 — Noisy-neighbor A/B re-run with T4.2.4d

Same protocol as T4.2.4c (Mistral-7B prefill bomb + TinyLlama decode victim, 120 s, 5 s lead). Matched-pair (B9 rule): A and B captured in same wall-clock window (13:21–13:26 UTC).

| Metric | A (T4.2.4d, partition ON, enforced) | B (libcipher_v2, partition OFF) | Δ A vs B |
|---|---:|---:|---:|
| Bomb prefills (125 s) | **848** | 11,776 | **-92.8%** |
| Bomb prefills/s | **6.78** | 94.2 | **-92.8%** |
| Victim decode iters (120 s) | 182 | 203 | -10.3% |
| Victim new tokens | 5,824 | 6,496 | -10.3% |
| Victim tok/s | 48.5 | 53.7 | -9.7% |
| Victim mean inter-token ms | 20.57 | 18.45 | **+11.5%** |
| Victim p50 inter-token ms | 20.69 | 19.14 | +8.1% |
| Victim p95 inter-token ms | 21.87 | 19.37 | +12.9% |
| Victim **p99** inter-token ms | **35.64** | **33.01** | **+8.0%** |
| Victim max inter-token ms | 286.8 | 74.3 | (single outlier in A) |
| Device mean power W | **172.8** | 493.7 | **-65.0%** |
| Device max power W | 196.4 | 643.5 | -69.5% |
| Device mean SM util % | 80.6 | 78.1 | +3.2% |

**Two headline observations:**

1. **Enforcement is fully realized in the multi-tenant setting:** bomb prefills crash from 11,776 to 848 (13.9× slower), and device power drops 65% (494 W → 173 W) because the GPU is no longer being saturated. The actuator now genuinely constrains tenant compute.

2. **Tail-latency isolation hypothesis is now CLEANLY FALSIFIED on this workload pair:** with enforcement working, victim p99 is **worse** under partition (+8.0%), not better. The reason is the workload-resource-mismatch from T4.2.4c Caveats: the victim's decode is HBM-bound and was not actually being starved in B's unrestricted contention. Forcing the victim into its own 8-SM partition in A removes its access to the other 124 SMs without freeing any HBM bandwidth — net negative.

## Verdict on the T4.2.4d hypothesis

- **Enforcement gap closed.** GREEN_CTX is no longer cosmetic. The actuator can demonstrably constrain a tenant's compute to a configured SM partition.
- **Tail-latency isolation on this workload pair: falsified.** Not a measurement artifact this time — the mechanism works, the workload is the wrong vehicle. Predicted in T4.2.4c Caveats, confirmed here.

The bomb (compute-bound, SM-heavy) and victim (HBM-bound, SM-light) bottleneck on different resources. SM partitioning helps when both tenants compete for the same SMs; here they didn't, even without partitioning. The clean tail-latency-isolation demonstration would require two SM-bound tenants (e.g., two prefill tenants), where unrestricted scheduling causes one to starve the other.

## Phase 4.2 ship status

| Gate | Target | Current state |
|---|---|---|
| G1 | ≥85% MFU per WL | Passing where measured |
| G2 | ≥90% MFU on WL05 ×8 | Passing at 100% (T4.2.4b_v2 600 s) |
| G3 | ≥30 tenants on WL05 | Not measured at this scale |
| G4 | zero oops, taint ≤+1, 24 h soak | Passing (taint 12288 stable) |

Plus a new positive datum: **the GREEN_CTX actuator is real**, not substrate-only. It can be invoked to constrain compute. T4.2.4c's "substrate-verified, product-value-undemonstrated" framing now updates to "actuator-verified, product-value-conditional-on-workload":

- For workloads with same-resource contention: predicted to deliver QoS isolation (untested).
- For workloads with cross-resource bottleneck (this pair): partition hurts.
- For "limit a tenant's compute share" use case (power budget, fairness enforcement, etc.): **delivers the constraint directly** — bomb's power consumption dropped 65% under partition. This IS a real product capability.

## Discipline gate at session end

| Gate | Result |
|---|---|
| Phase 3 ABI 12/12 PASS | ✅ (3+6+3) |
| Fallback kmod md5 55ab8c0c | ✅ unchanged |
| Fallback libcipher_v2 md5 86618c30 | ✅ unchanged |
| Taint 12288 | ✅ unchanged |
| Kmod 0.4.5 (srcversion 4618FD1FC28BEE5EBC32944) | ✅ loaded |
| dmesg oops/WARN/BUG since start | none |
| `cipher_kmod: alloc_failures` post-reload | 0 |

The NVRM `refcntRequestReference_IMPL` lines at 13:21–13:24 are NVIDIA driver internal refcount info-level messages (not kernel oops/WARN); they appeared during normal CUDA process teardown of the A/B tenants. No taint delta.

## Artifact map

| Artifact | md5 | Purpose |
|---|---|---|
| `libcipher_rt.so.v0.2.0_T4_2_4b_v2.pre_T4_2_4d` | `8fbc350fb322b0733ce6d4af1468a88f` | discipline pre-snapshot |
| `libcipher_rt.so.v0.2.0_T4_2_4d_diag` | `15fe74d07de19f502a558ddab7efa18b` | intermediate diagnostic build (PyTorch NULL-stream finding) |
| **`libcipher_rt.so.v0.2.0_T4_2_4d`** | **`50414674ddad2689191d13a92377e492`** | **enforcement fix shipped** |
| `cipher_rt_green_ctx.c.pre_T4_2_4d` | `8326bc752933eb26fc376ada1e6d2453` | source pre-snapshot |
| `cipher_cupti.c.pre_T4_2_4d` | `3e049315cedd5798c1d8d898cfd3edaf` | source pre-snapshot |
| `/tmp/test_green_ctx_enforcement.cu` | new | raw CUDA microbench (push vs explicit) |
| `/tmp/test_green_ctx_nullstream.cu` | new | NULL-stream-under-green-current microbench |
| `cipher-phase4-evidence/t4_2_4d/A_summary.json` | new | T4.2.4d A (partition ON) summary |
| `cipher-phase4-evidence/t4_2_4d/B_summary.json` | new | T4.2.4d B (partition OFF) summary |
| `cipher-phase4-evidence/t4_2_4d/{A,B}_nn_*.{log,csv,json,txt}` | new | raw timestamps, device CSV, progress, stderr |

## Cross-link to backlog

- **B-T4.2.4d-1 — closed:** "GREEN_CTX actuator is cosmetic (substrate-verified, partition not reaching kernels)." Fixed by persistent cuCtxSetCurrent at every kernel launch.
- **B-T4.2.4d-2 — open:** "Tail-latency isolation on cross-resource workload pairs (bomb HBM-light, victim HBM-bound) is mechanically unsupported and was confirmed empirically under T4.2.4d." Status: documented limitation, not a regression.
- **B-T4.2.4d-3 — open (informational):** "Persistent cuCtxSetCurrent affects ALL CUDA APIs in the calling thread, not just kernel launches. This means cudaMalloc/cudaMemcpy from the tenant's threads also run on the green ctx. Acceptable for current scope (green ctx inherits primary's memory pool); document for future review when cipher_rt is wired into multi-context workloads."
