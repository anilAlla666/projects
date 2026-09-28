# Priority 1 — Cross-Tenant Weight Page Cache — Diagnostic + Architecture Memo

**Date:** 2026-05-18. **Type:** design + measurement, no substrate code.
Substrate `a7ac8e97`. Workload WL01 (TinyLlama-1.1B, B=1, 128-token greedy
decode). Single H100 80GB.

**Bottom line:** the cross-tenant idea is right; the *mechanism* in the
premise is wrong, on two counts the measurement makes unambiguous. A shared
weight page cache delivers a **capacity** win (4×→1× weight HBM), not the
"N-fold tok/W from dividing memory transactions." And B=1 decode here is
**not HBM-bandwidth-bound** — it is launch/overhead-bound — so amortizing
weight reads cannot yield N-fold regardless of mechanism. The
substrate-attributable lift lives in **cross-tenant batching**, for which the
weight cache is a useful *enabler*, not the lever itself.

---

## 1. The premise under test

> 4 tenants running the same model read the same bytes from HBM. Cache
> weights as shared HBM pages → memory transaction count divides by N →
> N-fold tok/W for memory-bound B=1 decode.

Two claims: (A) decode is memory(-bandwidth)-bound; (B) shared pages divide
the memory transaction count by N. Both are tested below. **Both fail.**

## 2. Diagnostic measurements

Profiler note: `ncu`/`nsys` are not installed on this pod; `nvidia-smi dmon
-s mu` is used — `sm%` (SM-occupancy time-fraction) and `mem%` (NVML
`utilization.memory`: fraction of time the HBM interface was actively
transferring). `mem%` is a utilization-*time* proxy, not a bandwidth
fraction, but a low value is dispositive: an idle memory interface cannot be
the bottleneck. Evidence: `priority1/dmon_1t.txt`, `dmon_4t.txt`.

| run | SM% (avg/peak) | **HBM mem% (avg/peak)** | FB used |
|---|---|---|---|
| WL01 1-tenant vanilla decode | 18 / 30 | **4 / 6** | 2.96 GB |
| WL01 4-tenant vanilla decode | 85 / 100 | **14 / 18** | 11.76 GB |

**Claim A is false.** Single-tenant B=1 decode runs with the HBM interface
active **4–6 % of the time** and SMs **18–30 %**. The GPU is idle ~70–95 % of
every decode token. The bottleneck is **per-token host/launch overhead** —
the Python generate loop, kernel-launch latency, and M=1 GEMV under-occupancy
— *not* HBM bandwidth and *not* compute. Even at 4 concurrent tenants
saturating SM occupancy to 100 %, HBM utilization only reaches **18 %** — the
memory bus is 82 % idle. Weight reads are nowhere near the bottleneck.

Consequence: even a *perfect* division of all weight-read traffic by N
amortizes something that is ≤ ~6 % (1-tenant) to ~18 % (4-tenant) of the time
budget. The tok/W ceiling from weight-traffic amortization alone is
**single-digit percent, not N-fold.**

**Weight duplication confirmed.** FB memory scales 2.96 GB → 11.76 GB ≈ 4×
from 1 to 4 tenants — four independent physical copies of the weights. This
is correct: 4 tenants are 4 OS processes → 4 CUDA contexts → 4 independent
`cudaMalloc`s. The duplication happens in stock
`AutoModelForCausalLM.from_pretrained(...).cuda()` — each process loads
safetensors and allocates its own device tensor. **CIPHER's hooks do not
touch weight allocation today** — libcipher_v2/libcipher_rt intercept GEMM
dispatch and green-context binding, not model load. So the tenants read
*different physical addresses*; there is no cross-tenant reuse to exploit at
the cache/L2 level as things stand.

## 3. Why claim B is also false — pages vs transactions

Sharing the physical weight copy across N tenants (one `cudaMalloc`, N
mappers) does **not** divide the HBM transaction count. Each tenant still
issues its own reads of every weight to feed its own matmul; reading the same
physical address from N contexts at N different times is N transactions, not
one. Physical sharing reduces *transactions* only if the N reads are
**coalesced in time** so a weight tile fetched once is consumed by all N
tenants before eviction:

- **L2 reuse** needs the N tenants marching the *same* weight tile within the
  L2 residency window. H100 L2 = 50 MB; TinyLlama is 2.96 GB and even one
  layer (~135 MB FP16) exceeds L2. Lockstep scheduling buys only partial,
  per-tile reuse — bounded, and it requires a cross-tenant layer-sync
  scheduler that does not exist.
- **Register/SMEM reuse** — the real division — needs the N tenants'
  activations co-located so one weight fetch multiplies against N activation
  vectors in a single kernel. That is a **batched GEMM** (B=1×N → B=N), i.e.
  **batching**, not page caching.

So: shared pages → **capacity** saving (4×→1× weight HBM). Transaction/
bandwidth division → **batching**. They are different primitives.

## 4. What the weight page cache *does* deliver — and it is worth having

Reframed honestly, a cuIpc-backed shared weight pool is a **capacity**
primitive: 4 tenants (or 16, or 40) map one 2.96 GB physical copy instead of
N copies. CIPHER already has the enabling mechanism — the **cuIpc
cross-process page pool from T4.6** (`cipher_kv_bridge.cpp`,
`cipher_rt_kv_alloc.c`; cross-process sharing proven real==sim 1.000 for
KV-dedup, CP 4.6). Applying it to weights instead of KV pages is a contained
extension: one tenant loads + registers the weight tensor via
`cuIpcGetMemHandle`, peers `cuIpcOpenMemHandle` and bind it as their model
parameters.

Value: weight HBM goes from `N × 2.96 GB` to `2.96 GB`. On an 80 GB H100 the
weight copies, not compute, cap tenant count — 1× weights frees the budget to
pack **many more tenants**. And tenant count is exactly the **concurrency
factor** the WL01 multi-tenant measurement showed is the only thing currently
producing lift (2.0× at 4 tenants). So the weight cache compounds the
concurrency pillar — indirectly, via density — but it is **not itself a
tok/W lift** and must not be reported as one.

## 5. Where the substrate-attributable lift actually is — cross-tenant batching

The headroom is explicit in §2: single-tenant decode leaves the GPU 70–95 %
idle because it is overhead-bound. The lever that claims that headroom is
**cross-tenant continuous batching**: intercept each tenant's per-token
decode GEMMs at the substrate, gather N tenants' activation vectors, issue
**one B=N batched GEMM**, scatter results. This attacks the *actual*
bottleneck:

- **Launch/overhead amortized N×** — one set of kernel launches serves N
  tenants instead of N separate launch loops. This is the dominant win,
  because overhead is the dominant cost.
- **Weight reads amortized N×** — a B=N GEMM reads each weight once into
  registers/SMEM and multiplies against N activations. This is the real
  "transaction count ÷ N" — and here the shared weight pool is the natural
  substrate: one physical weight, N activations, one kernel.
- It is what vLLM continuous batching does *within* one process; doing it
  *across* processes/tenants at the driver layer is the primitive CIPHER
  does not have.

**Missing primitives** for cross-tenant batching:
1. a cross-process activation gather/scatter buffer (the cuIpc page pool
   covers the mechanism);
2. a substrate decode scheduler that aligns N tenants to a common step
   boundary and forms the batch (tenants decode asynchronously today);
3. per-sequence attention handling inside the batched step (KV caches stay
   per-tenant; only the weight-bound projection/MLP GEMMs batch — attention
   is already handled per-sequence by the CP 4.6/5.1 KV substrate);
4. tail handling when tenants have unequal sequence lengths / finish at
   different times (continuous-batching admission/eviction).

A cheaper, single-tenant, complementary lever also falls out of §2:
**CUDA-graph capture** of the decode step removes per-token launch overhead
without any cross-tenant machinery — worth a separate spike.

## 6. Estimated lift (honest ranges, not point claims)

| lever | mechanism | tok/W effect | capacity effect |
|---|---|---|---|
| weight page cache *alone* | shared physical weights | ~0 (≤ single-digit %; decode isn't HBM-bound) | **4×→1×** weight HBM (∝ tenant density) |
| cross-tenant batching (B=N) | one launch + one weight read serves N tenants | **large multiple** — attacks the 70–95 % idle; vLLM-class continuous batching routinely 5–20× over naïve B=1 serving. Realistic CIPHER target: a real multi-× substrate-attributable lift | uses shared weights as input |
| CUDA-graph decode (single-tenant) | kill launch overhead | 2–5× tok/s on this overhead-bound profile | none |

The 3.6×+ substrate-attributable target is **plausible** — but it comes from
cross-tenant batching (optionally graph-accelerated), with the weight cache
as an enabler. It does **not** come from a weight page cache used as a
bandwidth primitive.

## 7. Recommendation

1. **Do not build the weight page cache as a tok/W primitive.** Build it (or
   schedule it) as a **capacity** primitive — cheap, reuses the T4.6 cuIpc
   pool, and raises the tenant ceiling that the concurrency factor feeds on.
2. **Re-aim Priority 1's prototype at cross-tenant batching.** That is the
   substrate-attributable lever. Recommended next step: a minimal
   2-tenant batched-decode prototype — gather 2 tenants' decode activations,
   one B=2 GEMM through the weight-bound layers, measure tok/W vs 2×B=1. If
   2-tenant shows real lift, scale the batch.
3. Run a **CUDA-graph single-tenant spike** in parallel — it is small, and
   §2 says launch overhead is the single biggest cost; it also de-risks the
   batched prototype (same graph machinery).
4. Keep the honest framing intact: concurrency (2.0× measured) is not
   CIPHER's; the weight cache is capacity; batching is the lift. Investors
   should see the corrected picture (CP 5.6 P2 retraction, 1.54× single-
   tenant, 2.0× concurrency, substrate factor 1.0× today) and the
   engineering plan to earn a real substrate factor — not a re-derived 3.6×.

## 8. Artefacts

`priority1/dmon_1t.txt`, `priority1/dmon_4t.txt` (raw dmon traces).
WL01 multi-tenant evidence: `WL01_MULTITENANT_REPORT.md`, `phase_a/WL01/`.
Prior cuIpc page-pool art: `cipher_kv_bridge.cpp`, `cipher_rt_kv_alloc.c`,
`t4_6_3_dedup_report.md`.
