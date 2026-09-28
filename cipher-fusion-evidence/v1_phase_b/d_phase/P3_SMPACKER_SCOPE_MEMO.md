# P.3 SM-packer / persistent-kernel — SCOPE + BUILD-NOW-vs-DEFER MEMO

**Date:** 2026-05-30. **Type:** READ-ONLY scope discovery + recommendation. No build, no commit,
anchors unchanged. STOP for Anil's build-now-vs-defer decision. Every claim cited file:line, not memory.

**Bottom line up front: DEFER P.3 to V.1 / CP 5.5.** P.3 has **no single-GPU-measurable cross-tenant
aggregate-MFU/TPW deliverable** that isn't already delivered by W.4 (POOL batched executor, closed). Its
two own pieces are (a) the SM-packer = observe-only hint whose actuator (PERSIST_ENGINE) was
de-prioritized by FWD-1, and (b) SM-partition = isolation/variance that is *contention-gated* and
explicitly "not a tok/W lever." Both pay off only at CP-5.5 density / the deferred multi-GPU 85%.

---

## PART 1 — SCOPE (cited)

### 1. What the plan says P.3 IS
- **SM-packer:** `CIPHER_REENGINEERING_PLAN.md:265` — *"`cipher_rt_sm_packer.{c,h}` | SM-pack **hint
  computation** | **setup**"* — a setup-time hint, not a hot-path actuator.
- **persistent-kernel / POOL executor:** `:1038` — *"**POOL batched executor (libcipher_v2) fuses N
  tenants' decode steps** — this is where the **3-6× lift** comes from at scale."* `:1164` — the
  memory-bound-decode row: *"TPW lift, target ≥2× **composed at N ≥ 4 tenant density** | cuBLAS-FP16 +
  **POOL cross-tenant batching (3-6×)** + fleet DVFS."*
- **Goal × workload:** Goal 3 = *"TPW (tok/W) composed lift via DVFS + density + **cross-tenant
  batching**"* (`:102`, with the measured curve *"**3.06× Mistral-7B N=4 → 5.98× TinyLlama-1.1B
  N=16**"*). **85% MFU is explicitly post-CP-5.5, NOT a v1 deliverable:** `:1019` — *"85% MFU on declared
  compute-bound WLs is the **post-CP-5.5 depth-win-program goal, not a v1 deliverable**... v1's job is to
  ship the substrate and measure the per-WL baseline."*

### 2. Current runtime state — "observation-only" (cited)
- **SM-packer = detection-only.** `cipher_rt_sm_packer.h:3-7` — *"This T4.2.3 build **instruments only —
  it counts**... **Actual substitution requires a resident persistent kernel (T4.2.4 PERSIST_ENGINE)**."*
  `cipher_rt_sm_packer.c:41-102` `cipher_rt_smp_observe()` detects per-stream small-kernel streaks
  (<4096 threads), called from `cipher_cupti.c:173`. Counters (`total/small_launches`, `longest_streak`,
  `:104-117`) are **exposed but consumed by no actuator** (`.h:17`: *"/proc exporter does not currently
  surface them; that's a Phase 4.4/4.7 wiring step"*).
- **POOL (live rt_phase4) = decision-only.** `cipher_rt_pool.c:1-4` — *"W.4a cross-tenant POOL
  eligibility... **DECISION-ONLY: no operand transport, no coalescing**"*; `:124-135`
  `cipher_rt_pool_observe_gemm()` returns 0 unconditionally (*"W.4a performs NO coalescing — transport is
  W.4b"*). The **batching that yields 3-6× is the POOL *batched executor* in libcipher_v2** (`:1038`),
  i.e. **W.4**, not the rt_phase4 eligibility hook.
- **SM-partition exists** (green-ctx). `cipher_rt_green_ctx.c:224-393` — CP 5.4 ALLOCATE → grp_mask →
  `cuDevSmResourceSplitByCount(MIN_SM=8)` → 15×8-SM groups (H100) → `cuGreenCtxCreate` over the tenant's
  groups. So SM *partitioning* is built; **what's missing to actuate the packer is the PERSIST_ENGINE**.
- **Missing to actuate:** the resident persistent kernel (PERSIST_ENGINE, `sm_packer.h:7`) that would
  consume the pack hints and pull work — never built.

### 3. FWD-1 de-prioritization (cited, `FWD_1_FINDINGS.md`)
FWD-1 verdict: *"The W.4b.7 ~52ms launch-bound forward ceiling was **largely an HF-DynamicCache
prototype-harness artifact**... `StaticCache` is **~2.6× faster** (eliminable with **zero graphs**) — and
the **intended product (vLLM) already uses static paged KV**, so production never paid that tax. The
**graph lever (torch.compile / manual CUDA-graph capture) is absent in this stack** (wrong + slower /
hard-abort). The forward is **near-bandwidth (~1.8× over floor) after static KV**... **⇒ No new substrate
lever; recommend → W.7."** `compile_static` = wrong (8/256) + slower; manual `CUDAGraph` **HARD-ABORTS on
replay** (`index_copy_` device assert; growing-KV not graph-safe). **Reasoning:** the persistent-kernel /
graph dispatch does not pay on the single-stream forward on this HF stack, and vLLM already does static-KV.

### 4. The compose relationship (cited, `W5_FLASHATTN_CLOSE_REPORT.md:143-144`)
*"Goal 3 (85% MFU) is a **composition target**: attention intercept (W.5) + **POOL cross-tenant batching
(W.4)** + **SM packer (P.3)** stack. W.5 wires the attention lever; the **MFU headline needs W.4 + P.3 to
compose**."* — **W.4 (now CLOSED: `w4a-pool-close` + `w4b-close`) delivers the cross-tenant batching
(3-6×).** What **P.3 ADDS** on top is the **SM-partition packing** of those batched tenants onto SM
groups — which, per §below, is *isolation/variance*, not aggregate-MFU.

---

## PART 2 — THE BINDING QUESTION (build-now vs defer) — **answer: DEFER**

**Q: Does P.3 have a single-GPU-measurable cross-tenant aggregate-MFU/TPW lever (build-now), or only the
deferred multi-GPU 85% payoff (defer)?**

1. **The single-GPU cross-tenant aggregate lever exists — but it is W.4 (POOL), not P.3.** The measured
   3.06× Mistral N=4 → 5.98× TinyLlama N=16 (`:102`, `cp_5_6/CP_5_6_CLOSEOUT.md`: N=4 Mistral 3.26× / N=8
   TinyLlama 3.69×) is the **POOL batched executor fusing tenant decode steps** (`:1038`) — **W.4, already
   closed + measured**. P.3 does not add it.
2. **P.3's own SM-partition packing is isolation/variance, NOT aggregate-MFU — and it can LOSE.**
   `CP_5_4_STEP_1_6B_4_STOPPED.md:19-28`: OP-2 (low contention, 7 tenants) — CIPHER per-tenant CV **0.256
   vs naive 0.160** (CIPHER **worse** variance, CIs disjoint; *"falsified at the low-contention operating
   point"*). `CP_5_4_CLOSEOUT.md:42-50`: OP-5 (high contention) ~6% tighter but *"NOT CI-confirmed"*;
   *"SM-partition isolation is **contention-gated** — it **hurts at low contention** and appears to help
   at high; a definitive statement was not produced and is **deferred**."* `:38-55`: *"**SM partitioning
   is not a tok/W lever** — established earlier in the campaign."*
3. **P.3's actuator (PERSIST_ENGINE / persistent-kernel) was de-prioritized by FWD-1** (PART 1.3): no
   single-stream-forward payoff on this stack; vLLM already static-KV; graphs absent/broken.
4. **The plan puts 85% post-CP-5.5** (`:1019`), and the regime where SM-partition isolation *might* pay
   (high-contention 100-tenant density) **is exactly CP 5.5**.

**⇒ P.3's only genuine payoff is at CP-5.5 density / the deferred multi-GPU 85% composition.** It has no
build-now single-GPU aggregate-MFU deliverable that W.4 doesn't already own. **Defer.**

---

## PART 3 — DEFER RECOMMENDATION (not a build-now design memo)

**Recommendation: defer P.3 to V.1 / CP 5.5**, alongside the 85% multi-GPU composed measurement, because
its payoff is density-gated (SM-partition isolation at 100-tenant high-contention) and/or multi-GPU
(85% composition) — neither single-GPU-measurable-as-a-new-lever now.

**What of P.3 is worth building single-GPU NOW: essentially nothing MFU-bearing.**
- The SM-packer **observe** op already exists (counts opportunities) — no build needed.
- The only trivial single-GPU item is **wiring the SM-packer counters to `/proc/cipher`** (the
  `sm_packer.h:17` "Phase 4.4/4.7 wiring step") — pure telemetry exposure, **adds no MFU**, not worth a
  substep on its own.
- Everything MFU-bearing — PERSIST_ENGINE actuation (FWD-1-deprioritized), SM-partition-at-density
  (CP-5.5-gated), the 85% composition (post-CP-5.5) — defers.

**Where P.3 lands in the roadmap:** it is **subsumed by V.1 CP 5.5**, whose gate is the 100-tenant hybrid
(5 prefill + 80 decode + 15 burst) + ≥5 model families on the full runtime (`PLAN.md:75`,`:1623`) — the
exact high-contention density where SM-partition isolation's value (if any) becomes CI-testable, composed
with W.4 batching + the D.9 FP8 lever. The deferred 85% MFU is measured there (with the NCCL/AllReduce
multi-GPU portion that cannot appear single-GPU, `PLAN.md:1721`).

**Honest residue / confidence (Marvel std):** HIGH confidence P.3 has no single-GPU build-now MFU lever
(SM-packer observe-only by design; SM-partition "not a tok/W lever" + contention-gated, can lose;
persistent-kernel FWD-1-deprioritized; cross-tenant lever owned by closed-W.4; 85% is post-CP-5.5 per the
plan). The one open question P.3 legitimately answers — *does SM-partition isolation tighten per-tenant
p99 at real 100-tenant density?* — is a **CP-5.5 measurement**, not a single-GPU build-now.

### If Anil instead wants a single-GPU build-now P.3
The only defensible single-GPU build-now would be a **cross-tenant SM-partition *isolation* gate** (not
MFU) — measuring per-tenant p99 tightening at moderate contention on a stock multi-tenant workload
(e.g. N=8 Mistral-7B bf16). But CP 5.4 already found this contention-gated and inconclusive at small N
(`CP_5_4_CLOSEOUT.md:42-50`); a real verdict needs CP-5.5 density. So even this collapses into V.1.
**SUBSTRATE-LINE note (Mem #24):** were P.3 built, the persistent-kernel/SM-pack actuation must live at
the CUDA driver dispatch boundary (green-ctx via `cuGreenCtxCreate` + driver `cuLaunchKernel` on the
resident kernel) — **no vLLM/torch scheduler bypass**; flagged for the design if/when un-deferred.

### STOP — Anil's call: defer P.3 to V.1/CP-5.5 (recommended, cited), or specify a single-GPU build-now
target. No code, no commit, anchors unchanged.
