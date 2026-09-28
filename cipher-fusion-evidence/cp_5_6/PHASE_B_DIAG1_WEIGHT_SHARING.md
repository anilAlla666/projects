# Phase B — Diagnostic Session 1: Cross-Tenant Weight Page Sharing

**Date:** 2026-05-18. **Type:** diagnostic + design, no substrate code.
Substrate `a7ac8e97`. Workload WL01 (TinyLlama-1.1B, B=1, 128-tok greedy
decode), single H100 80GB. Supersedes the interim `PRIORITY_1_WEIGHT_CACHE_MEMO.md`.

**Recommendation: MODIFY-AND-BUILD + PIVOT.** Build the weight-sharing
primitive — but as a **capacity** primitive (N× → 1× weight HBM), where it is
genuinely valuable and cheap (~3-5 days, reuses the T4.6 cuIpc pool). Do
**not** build it as the tok/W lever: the diagnostic shows it cannot be one.
Pivot the substrate-attributable ≥3.6× tok/W work to **cross-tenant
batching**, for which weight-sharing is the natural enabler.

---

## Reading the premise

The task premise states B=1 decode is "memory-bound, ~5% of HBM bandwidth
ceiling." Those two clauses contradict each other, and the contradiction is
the whole diagnostic: **a workload using 5% of HBM bandwidth is not
memory-bound.** It is bound by whatever consumes the other 95% of the time.
The measurements below establish what that is, and what weight-sharing can
and cannot do about it.

## Tooling note (per the constraint to document measurement approach)

`ncu`, `nsys`, `dcgmi` are not installed. `nvprof` is present but is the
legacy profiler and does **not** support metric collection on Hopper (CC 9.0)
— it cannot profile this H100. A custom CUPTI memory-counter tool was not
built (out of scope for a 2-3 h diagnostic). Measurement approach used
instead, and why it is conclusive:

- **`nvidia-smi dmon -s mu`** — `sm%` (SM-occupancy time-fraction) and `mem%`
  (NVML `utilization.memory` = fraction of time the HBM interface was
  transferring). `mem%` is a utilization-*time* proxy, not a byte count — but
  a low value is dispositive (an idle interface is not a bottleneck), and its
  *scaling* with tenant count answers the read-coalescing question directly.
- **Per-process GPU memory** (`nvidia-smi --query-compute-apps`) and **CUDA
  allocation probe** (`weight_probe.py`) — these answer the duplication
  question *more* decisively than a byte counter would: they show the
  physical copies are disjoint, which makes silicon-level read-coalescing
  impossible by construction.

Evidence: `phase_b_diag1/dmon_1t.txt`, `dmon_4t.txt`, `wprobe{1..4}.log`,
`weight_probe.py`.

---

## Objective 1 — Memory-traffic profile (WL01 4-tenant vs 1-tenant)

| run | SM% avg/peak | **HBM mem% avg/peak** | FB used |
|---|---|---|---|
| 1-tenant decode | 18 / 30 | **4 / 6** | 2.96 GB |
| 4-tenant decode | 85 / 100 | **14 / 18** | 11.76 GB |

**Finding 1a — decode is not HBM-bandwidth-bound.** Single-tenant decode runs
with the HBM interface active **4-6 % of the time**, SMs **18-30 %**. The GPU
is idle 70-95 % of every decode token. At 4 tenants saturating SM occupancy
to 100 %, HBM activity still only reaches 18 % — the memory bus is 82 % idle.
The bottleneck is **per-token host/launch overhead** (the Python `generate`
loop, kernel-launch latency, M=1 GEMV under-occupancy), not HBM traffic.

**Finding 1b — HBM traffic scales ~linearly with tenant count; no silicon
reuse.** HBM active-time goes 1-tenant ≈5 % → 4-tenant ≈16 %, i.e. ~3-4× more
HBM activity for 4× the tenants — linear within the metric's resolution. If
the 4 tenants were reading shared weight bytes with L2/HBM read-coalescing,
4-tenant HBM activity would stay near the 1-tenant ≈5 % (the weight portion
served once); it does not. The tenants read independently. (Objective 2
explains why this is structurally forced, not incidental.)

A byte-exact CUPTI count would refine 3-4× to a precise multiple, but cannot
change the conclusion: traffic is linear, not sublinear — no reuse exists.

## Objective 2 — Process memory-map analysis

`weight_probe.py` loads TinyLlama-1.1B and reports its CUDA allocations; run
1× then 4× concurrently.

| | single probe | 4 concurrent probes |
|---|---|---|
| per-process GPU mem (`--query-compute-apps`) | 2720 MiB | **2720 MiB each** (×4) |
| total FB | 2.96 GB | **10.9 GB** (≈ 4×) |
| `torch.cuda.memory_allocated` | 2.202 GB | 2.202 GB each |
| param bytes | 2.200 GB | 2.200 GB each |
| `first_param.data_ptr()` | 0x7924d8000000 | **0x7a65…, 0x7c38…, 0x772a…, 0x7681…** — four disjoint ranges |
| `layers.10.mlp.gate_proj` ptr | 0x792490000000 | four disjoint ranges |
| allocator | `PYTORCH_CUDA_ALLOC_CONF` unset → default `cudaMalloc` caching allocator (no `cuMemMap`/VMM, no `expandable_segments`) | same |

**Finding 2 — the duplication is structural and total.** Each of the 4
tenants is a separate OS process → separate CUDA context → separate
`cudaMalloc` arena. Each independently runs stock
`AutoModelForCausalLM.from_pretrained(...).cuda()` and allocates its **own
full 2.2 GB physical copy** of the weights. The four copies occupy four
disjoint physical address ranges (per-process memory 4×2720 MiB; pointers
disjoint; `torch_alloc` identical per process). No `cuMemMap`/VMM is in play —
plain `cudaMalloc`. **CIPHER's hooks do not touch model-weight allocation at
all** — `libcipher_rt` intercepts GEMM dispatch, `libcipher_v2` does
tenant/green-context tracking, `cipher_kv_bridge` handles KV pages; none sees
`from_pretrained`. The duplication lives entirely in stock HF + PyTorch, one
copy per process, and the substrate currently does nothing about it.

This is *why* Objective 1 sees linear traffic: L2 tags and HBM bank accesses
key on **physical address**. Four disjoint physical copies cannot alias in L2
and cannot coalesce at the memory controller. Cross-tenant read reuse is not
"absent" — it is **physically impossible** while the copies are disjoint.

---

## Objective 3 — Architectural design memo

### 3a. What "shared weight pages" would actually change

Make tenants 1..N map **one** physical weight copy: tenant 0 loads the
weights, exports them (`cuIpcGetMemHandle`, or VMM `cuMemExportToShareableHandle`);
tenants 1..N `cuIpcOpenMemHandle` and rebind the mapped memory as their model
parameter storage. The mechanism already exists in CIPHER — the **T4.6 cuIpc
cross-process page pool** (`cipher_kv_bridge.cpp`, `cipher_rt_kv_alloc.c`),
proven real==sim 1.000 for cross-tenant KV-dedup (CP 4.6). Extending it from
KV pages to a weight arena is a contained reuse, not a new subsystem.

**What this delivers — capacity.** Weight HBM goes N×2.2 GB → 1×2.2 GB. This
is real and valuable: on an 80 GB H100, independent copies cap TinyLlama at
~29 tenants on weight memory alone; a shared copy makes weight memory a
constant, so tenant count is then bounded by KV+activations (tens of MB
each) — **far more tenants per GPU**. And tenant count is exactly the
*concurrency factor* the WL01 multi-tenant measurement showed is the only
thing currently producing tok/W lift (2.04× at 4 tenants). So weight-sharing
**compounds the concurrency pillar via density**.

**What this does not deliver — tok/W.** Two independent reasons:
1. *Mechanism.* Mapping one physical copy into N contexts does not reduce the
   transaction count. Each tenant still issues its own reads of every weight
   to feed its own matmul; N reads of one address at N different times are N
   transactions. Transactions divide only if N reads are coalesced *in time* —
   L2 reuse (bounded: 50 MB L2 vs 2.2 GB model → ≤~2 % opportunistic) or a
   single batched kernel with N consumers. Page-sharing is neither; it is a
   *precondition* for the batched kernel, not a substitute.
2. *Headroom.* Even an idealised weight-sharing that drove weight HBM traffic
   to zero can only remove the fraction of runtime spent on it. Objective 1
   measures that fraction at ≲5 % (HBM active 4-6 % of the time).

### 3b. Theoretical ceiling for weight-sharing lift

The task asks: given ~95 % of HBM bandwidth idle, what is the ceiling? The
honest answer inverts the premise's intuition. "95 % HBM idle" is **not**
headroom for weight-sharing — it is proof that 95 % of the runtime is spent
on something other than weight HBM traffic. Weight-sharing can only touch the
≤5 % that *is* weight traffic:

```
ceiling(weight-sharing tok/W lift)  =  1 / (1 - f_mem)   with f_mem ≈ 0.05
                                    ≈  1.05x      (~5%, single-digit percent)
```

That is the ceiling for *this* workload (1.1B model, B=1, overhead-bound).
For a genuinely HBM-bandwidth-bound regime — a 70B-class model B=1 decode,
where weight reads dominate — `f_mem` approaches 1 and the *capacity* problem
also becomes acute. But even there, weight-*sharing* (page mapping) still does
not coalesce the transactions; the lever is still a batched kernel. The model
size changes which resource is scarce; it does not change pages-vs-batching.

### 3c. Where the substrate-attributable ≥3.6× actually is

The 70-95 % idle GPU in Objective 1 is real headroom — claimable by attacking
the *overhead* bottleneck, not the memory one:

**Cross-tenant batching.** Intercept N tenants' per-token decode GEMMs at the
substrate, gather their activation vectors, issue **one B=N batched GEMM**,
scatter results. This amortises the launch/overhead cost — the dominant cost —
across N tenants, and reads each weight once for N consumers (the *real*
transaction division). It is vLLM-class continuous batching done *across
processes* at the driver layer. Weight-sharing is its natural input: one
physical weight, N activations, one kernel.

- Primitives that exist to compose: `cipher_kv_bridge` cuIpc page pool
  (cross-process buffers); `libcipher_rt` GEMM-dispatch hooks (the
  interception point); `libcipher_v2` tenant tracking (group membership).
- Primitives missing: a cross-process activation gather/scatter buffer; a
  decode-step scheduler aligning N async tenants to a common batch boundary;
  unequal-length tail handling (continuous-batching admission/eviction).
  Attention stays per-sequence on the existing CP 4.6/5.1 KV substrate — only
  the weight-bound projection/MLP GEMMs batch.

A complementary single-tenant lever: **CUDA-graph capture** of the decode
step removes per-token launch overhead with no cross-tenant machinery —
small, and it de-risks the batched build (shared graph machinery).

### 3d. Estimated engineering effort

| item | scope | effort |
|---|---|---|
| weight-sharing **capacity** primitive | cuIpc-export weight arena from tenant 0; peer import; rebind HF params to external storage (a custom multi-tenant load path — fiddly but bounded); correctness validation | **~3-5 days** |
| CUDA-graph decode spike | capture/replay the decode step single-tenant | ~2-4 days |
| cross-tenant batching (the tok/W lever) | gather/scatter buffers + cross-process decode scheduler + tail handling + correctness gating | **~3-6 weeks** |

---

## Recommendation

One clear call, mapping to the task's three options:

- **MODIFY-AND-BUILD** the weight-sharing primitive — but re-scoped as a
  **capacity** primitive (N×→1× weight HBM, raising tenant density). ~3-5
  days, reuses the cuIpc pool. It is worth building because density compounds
  the concurrency factor — but it must not be represented as a tok/W lift;
  its direct tok/W ceiling here is ~1.05×.
- **PIVOT** the substrate-attributable ≥3.6× tok/W work to **cross-tenant
  batching** (Phase B Build Session candidate). That is the lever that
  attacks the measured bottleneck (70-95 % idle from launch overhead).
  Weight-sharing feeds it. Recommended first build step once adjudicated: a
  minimal 2-tenant batched-decode prototype, tok/W vs 2×B=1, scale if it lifts.
- Run the **CUDA-graph spike** alongside — cheap, attacks the same overhead.

**Do not** build weight-sharing as the ≥3.6× mechanism. The diagnostic is
unambiguous: it is a capacity primitive. Presenting it — or a concurrency
multiplier — as a substrate tok/W lift would repeat the CP 5.6 P2 finding
(the 3.617× was a loop artifact; honest single-tenant 1.54×; multi-tenant
2.0× concurrency, substrate factor 1.0×). The honest ≥3.6× is reachable, by
cross-tenant batching — that is the real engineering, and it is worth doing.

## Artefacts

`phase_b_diag1/` — `dmon_1t.txt`, `dmon_4t.txt` (HBM/SM traces),
`wprobe{1..4}.log`, `weight_probe.py` (process memory-map). WL01 multi-tenant:
`WL01_MULTITENANT_REPORT.md`. cuIpc prior art: `cipher_kv_bridge.cpp`,
`cipher_rt_kv_alloc.c`, `t4_6_3_dedup_report.md`.
