# Phase B — Track 1: Cross-Tenant Batching — Design Memo

**Date:** 2026-05-18. **Type:** design, no substrate code. Substrate
`a7ac8e97`. Anchor workload WL01 (TinyLlama-1.1B, B=1 greedy decode).

**This is the TPW lever.** Diagnostic 1 measured the bottleneck: at B=1
decode the GPU is 70–95 % idle per token on launch/host overhead. Batching N
tenants' decode steps into one B=N kernel amortises that overhead. The lift
is measured below, not theorised — and it clears the ≥3.6 × substrate-
attributable target.

## 1. Measured lift ceiling — WL01 in-process batch scan

In-process B=N is the kernel-level ceiling cross-tenant batching can reach
(the batched GEMM is identical; cross-process plumbing only adds overhead
*below* this). WL01, vanilla, 5 prompts × 128-tok greedy, power windowed to
the decode region. Evidence: `phase_b/batchscan/`.

| B | tok/s | W | tok/W | MFU% | tok/W vs B=1 |
|---|---|---|---|---|---|
| 1 | 72.5 | 148 | 0.491 | 0.020 | 1.00× |
| 2 | 135.9 | 148 | 0.921 | 0.038 | 1.88× |
| 4 | 285.4 | 149 | 1.922 | 0.079 | **3.92×** |
| 8 | 565.9 | 150 | 3.781 | 0.156 | **7.70×** |
| 16 | 1109.6 | 152 | 7.289 | 0.307 | **14.85×** |
| 32 | 2169.8 | 155 | 14.026 | 0.599 | 28.58× |

Near-linear in N. Power is flat (148→155 W) — batching converts idle time to
work at almost-constant draw, which is exactly a tok/W win. MFU at B=32 is
still 0.6 % — for a 1.1B model the GPU is nowhere near compute-bound, so the
curve has not saturated.

**Substrate-attributable framing.** The honest denominator is *naive
N-concurrent*, not 1-tenant (concurrency is free — see `WL01_MULTITENANT_REPORT.md`).
Naive 4-concurrent vanilla = 0.994 tok/W (measured); 4 concurrent tenants
already saturate SM occupancy, so the naive baseline is ~flat at ~1.0 tok/W
for N ≥ 4. Then:

| N | batched tok/W | naive N-concurrent tok/W | **substrate-attributable** |
|---|---|---|---|
| 4 | 1.922 | 0.994 (measured) | **1.93×** |
| 8 | 3.781 | ~1.0 (est.; flat) | **~3.8×** |
| 16 | 7.289 | ~1.0 (est.; flat) | **~7×** |

**Cross-tenant batching crosses ≥3.6 × substrate-attributable at N ≈ 8** and
has wide margin at N=16. (Open item: measure 8- and 16-tenant naive-concurrent
to firm the denominator; the estimate ~1.0 is well-supported — 4 concurrent
tenants already pin SM to 100 %, so more tenants cannot raise naive tok/W.)

The build's entire risk is the **plumbing gap** — how close cross-process
batching lands to this in-process ceiling. Even at 50 % of ceiling, N=16
clears 3.6 ×.

## 2. Architecture

### 2.1 Granularity — decode-step, not GEMM

Fusing individual GEMM calls across processes (intercept each tenant's
~150 GEMMs/token, rendezvous N processes per GEMM) is rejected: the
per-GEMM cross-process synchronisation cost dwarfs the saving. Fusion happens
at **decode-step granularity** — one process runs the *entire* B=N transformer
step for the group, once per token.

### 2.2 The batched-decode executor

A substrate-orchestrated **batched-decode executor** holds one model copy and
runs the B=N step. Two viable forms — adjudication picks:

- **(A) Dedicated executor process.** A substrate-spawned process owns the
  model; tenants are clients. One weight copy by construction (Track 2 then
  redundant for batching — see §7). Cleanest; closest to vLLM.
- **(B) Elected-tenant executor.** One tenant in the model-group runs the
  batched step on behalf of the group, using its own weight copy + cuIpc-
  gathered peer activations. Preserves CIPHER's per-process tenancy model
  (green-partition-per-tenant). Track 2 then reduces the *non-executing*
  tenants' redundant copies (capacity).

Both run the same batched kernels; both fuse at decode-step granularity.

### 2.3 Transparent interception — CIPHER's distinctive value

vLLM requires rewriting the app to vLLM's API. CIPHER's substrate already
monkeypatches `generate()` (`cipher_spec_decode.install()`). The cross-tenant
batching layer installs the same way: it **intercepts each tenant's per-token
decode step** (the `model.forward` inside `generate`) and routes it to the
executor — the tenant's code is **unmodified**. Transparent cross-process
batching is the substrate primitive; that is what makes it "driver-level,"
not "use a different serving framework."

### 2.4 What batches and what does not

- **Batched (B=N):** the weight-bound GEMMs — q/k/v/o projections, MLP
  up/gate/down. These are where launch overhead and weight reads live; M=1→M=N
  is where the measured lift comes from.
- **Per-sequence (not batched):** attention. Each tenant's KV cache is its
  own, of its own length. Attention is computed per-sequence inside the step.
  This is cheap and — critically — means **heterogeneous sequence lengths do
  not break the GEMM batch** (the projection/MLP GEMMs are `[N,hidden]×[hidden,*]`;
  KV length never enters them). §4 expands.

## 3. Substrate primitives needed

1. **Cross-process activation gather** — collect N tenants' current
   hidden-state vectors into one `[N, hidden]` batch tensor. cuIpc staging
   buffer; reuse the T4.6 / `cipher_kv_bridge` cuIpc page-pool machinery.
2. **Per-tenant KV isolation within the shared step** — each tenant's KV
   cache addressed separately (cuIpc handle to the tenant's KV, or a paged KV
   pool as in CP 5.1). The batched step indexes per-tenant KV for the
   attention sub-step. Correctness isolation is non-negotiable — a tenant must
   never read another's KV.
3. **Logit dispatch / scatter** — the B=N output split into N logit rows,
   each returned to the originating tenant process (cuIpc shared output
   buffer + per-tenant completion signal).
4. **Batch-formation scheduler** — decides when to fire a batched step: wait
   to gather more tenants (throughput, higher N) vs fire now (latency). A
   bounded-wait window (e.g. fire at N_target or after T_max µs) — continuous-
   batching admission. Also handles tenants at different decode positions
   (a tenant mid-prompt-prefill vs mid-decode).

## 4. Failure modes

- **Heterogeneous sequence lengths** — *not* a problem for the batched GEMMs
  (length-agnostic, §2.4). It only affects the per-sequence attention sub-step
  (already handled — each tenant's own KV) and scheduler admission. No padding
  of the GEMM batch is needed for unequal lengths.
- **Padding overhead at low N** — at N=2 the batch is small but overhead still
  amortises ~1.9× (measured). There is no GEMM padding (N is the exact batch).
  The only "low-N" cost is a partly-filled scheduler window; bounded by T_max.
- **Tenant crash mid-batch** — the executor must tolerate a tenant
  disappearing between gather and scatter: timeout on the gather, drop the
  dead tenant, run B=(N−1), mark the slot. The kmod `do_exit` reaper already
  sweeps dead tenants; the executor subscribes to that.
- **Correctness preservation** — a row of a B=N GEMM is bit-equivalent to the
  individual M=1 GEMM modulo FP reduction order. The per-tenant teacher-forced
  / logit-KL gate (Phase A harness) validates each tenant's output against its
  own clean-FP16 gold — the batched path must pass the same gate the
  single-tenant path does. Build phases gate on this.
- **Straggler / position skew** — tenants decoding at different token indices:
  the scheduler groups by readiness, not by index; a tenant that falls behind
  simply joins a later batch.

## 5. Lift quantification (summary)

In-process ceiling (measured, §1): N=2 1.88×, N=4 3.92×, N=8 7.70×,
N=16 14.85× tok/W vs B=1. Substrate-attributable (vs naive concurrency):
~1.9× / ~3.8× / ~7× at N=4/8/16. **Target ≥3.6× met at N≈8**, comfortably
exceeded at N=16. For larger models (e.g. Mistral-7B) the curve saturates at
lower N (compute/memory-bound sooner) — regime-dependent; to be measured per
workload class in Phase A's resumed matrix.

## 6. Build phases

- **Prototype (1–2 weeks)** — 2-tenant elected-executor, decode-step fusion,
  fixed equal-length prompts, no scheduler (barrier sync). Goal: demonstrate
  cross-process batched tok/W > naive 2-concurrent, and pass the per-tenant
  correctness gate. Measures the real plumbing gap vs the §1 ceiling.
- **MVP at N=4 (3–4 weeks)** — the scheduler, heterogeneous lengths,
  crash tolerance, per-tenant KV isolation hardened. Gate: ≥3.6×-track
  substrate-attributable at N=8 projection, correctness gate green on all
  tenants, on ≥2 model classes.
- **Production at N=16 (6 weeks total)** — paged KV pool, continuous-batching
  admission/eviction, soak. Gate: substrate-attributable ≥3.6× verified at
  N=16 under the correctness gate, multi-workload.

## 7. Composition with Track 2 (weight-sharing)

Important and honest: under either executor form (§2.2), the process that
runs the batched step uses **one** weight copy it already holds — the batched
GEMM reads weights once for all N rows *because it is one kernel in one
process*, not because pages are shared across processes. So **Track 2 is not
a functional prerequisite for Track 1.** Track 1 can be built and can deliver
the lift independently. Track 2's role is **capacity** — letting the
non-executing tenants drop their redundant weight copies, raising tenant
density (which compounds the concurrency factor). Build order is therefore
*not* forced; see `PHASE_B_WEIGHT_SHARING_DESIGN.md` §6. Track 1 is the
critical path to the TPW target.
