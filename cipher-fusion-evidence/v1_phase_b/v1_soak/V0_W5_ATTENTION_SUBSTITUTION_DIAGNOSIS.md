# W.5 attention substitution — WHY observe/passthrough (build vs research-wall)

**Date:** 2026-05-31. **Type:** READ-ONLY (cite file:line; no build, no commit, anchors unchanged).
Answers: why did W.5 close as intercept+passthrough (substitution v1.x), what would substitution mean,
and is it a build or a research-wall — so Anil decides v1 vs v1.5.

**Headline:** W.5 deferred substitution **by spec design** — its bar was "intercept + passthrough,
byte-identical." The deferred lever is **attention-MATH/kernel substitution** (an *optimized attention
kernel* + a per-pattern **KL acceptance gate**), which is **NOT a simple "wire HANDLED" build**: it
requires either beating vLLM's already-Hopper-optimal FA3 (exact, marginal) or an approximate kernel
(quality-gated, OOD-risk — the accuracy-oracle class). **The deliverable attention-HBM lever — KV
footprint — is ALREADY delivered, KL=0, by the KV-dedup memory substrate.** ⇒ attention-math
substitution is correctly **v1.5 quality/kernel research**, not a v1 gap.

---

## 1. WHY v1.x — what the W.5 close states (cited)

The deferral is **not** "it failed" — it is **the W.5 spec bar by design**, plus the missing pieces:
- **(a) The W.5 bar was intercept+passthrough, byte-identical** — `W5_FLASHATTN_CLOSE_REPORT.md:49-50`
  *"v1 ships intercept + passthrough only — the trampoline forwards every call to the original kernel, so
  output is byte-identical to baseline"*; `:98-100` *"the W.5 bar is 'patterns intercept + passthrough
  (byte-identical correctness)'; substitution kernels are explicit §D items, NOT silent narrowing."* So
  substitution was **explicitly out of scope for W.5**, scheduled as §D v1.x — not hidden.
- **(b) No HANDLED path is built** — the 6-pattern trampolines are naked count-and-forward: `:32-36`
  *"Each bumps intercepts + passthrough then tail-jmps to the resolved original."* There is **no actuator
  / HANDLED branch in the vLLM-attention trampolines at all**. And on the (separate, HF) torch-SDPA path,
  `cipher_rt_attn_dispatch.h:92-96`: *"no actuator returns HANDLED on the attn substrate today"* — the
  trampoline HANDLED-return-construction is v1.5. So substitution is mechanically un-plumbed.
- **(c) The KL acceptance gate is itself deferred** — `:52-53` *"The per-pattern KL acceptance gate
  facilities land with the substitution kernels (§D, v1.x)."* I.e. substitution is expected to **diverge**
  from baseline (else no KL gate needed) → it needs a quality gate that does not yet exist.
- **(d) The deferred work is "optimized attention kernels"** — `:96-98` *"Substitution kernels for all 4
  armed patterns … Substrate-side optimized attention kernels (FA3-shape-matched, MLA, PagedAttn) are
  v1.x."* Plus `:106` Hopper FA3 TMA-descriptor handling = v1.x. FlashInfer P3/P4 (`:92-95`) is a separate
  JIT-intercept residue (can't GOT-patch JIT), also v1.x.

**So the reason is a combination of (b) not-built + (c)/(d) quality-gated-and-kernel-work — NOT (a) "it's
impossible." The W.5 bar was deliberately intercept+passthrough; substitution was scoped forward.**

## 2. WHAT substitution concretely means here — math/kernel, NOT KV-representation

Two distinct attention-HBM levers; the W.5 deferred one is the second:
- **(i) KV footprint / representation (memory substrate)** — deduplicate/share identical KV blocks across
  tenants → cuts KV HBM capacity + redundant reads. **This is the KV-dedup plugin, and it is ALREADY
  DELIVERED, KL=0** (exact dedup of identical blocks): `cipher_vllm_kvdedup.py` (vLLM plugin, monkey-patch
  of `_allocate_kv_cache_tensors` + cuMemMap swap), measured on stock vLLM decode (N=2/N=4 cross-tenant
  hits, ≥30 GiB saved, 50.62% dedup ratio — per `V0_ENGAGEMENT_MAP_THREE_SUBSTRATE.md` §1.3, cited there).
  **This is NOT what W.5 deferred** — it's a separate, shipped substrate.
- **(ii) Attention MATH / kernel substitution (W.5's deferred lever)** — replace the attention **kernel**
  (FA3/MLA/PagedAttn) with a CIPHER kernel: `W5:96-98` "optimized attention kernels (FA3-shape-matched,
  MLA, PagedAttn)". The HBM lever here is the attention **compute's** KV-read efficiency, not the
  footprint. This is the **attention-math** option — the Koopman-class problem (substitute compute, risk
  divergence).

**⇒ The deliverable KV-HBM lever (footprint) is already captured by KV-dedup (i). W.5 deferred (ii), the
attention-math/kernel substitution — a different, harder thing.**

## 3. THE HONEST CALL — build or research-wall?

**Attention-math substitution is NOT a simple "wire HANDLED" build, and the deliverable lever is already
shipped — so it is correctly v1.5, not a v1 gap.** Precisely:
- **It is not "just wire HANDLED."** Even the plumbing (HANDLED-return reconstruction in the trampolines,
  `attn_dispatch.h:92-96`) is unbuilt — but that is the *small* part. The substantive part is the
  **kernel** + the **KL gate** (`W5:52-53,96-98`).
- **Exact-faster kernel → a hard build with marginal upside (not a wall, but low-ROI):** to substitute
  *without* divergence you must write an attention kernel that **beats vLLM's FA3**, which on H100 is
  already the state-of-the-art Hopper hot path (`W5:20` "FA3 … the H100 FLASH_ATTN hot path", intercepting
  5280/cell). Beating an already-optimal vendor kernel is a hard build for marginal gain → low ROI, not a
  v1 priority.
- **Approximate kernel → a research-wall (quality-gated):** a cheaper *approximate* attention (sparse /
  low-rank / linear) diverges → it needs the **per-pattern KL acceptance gate** the W.5 report explicitly
  defers (`:52-53`) — i.e. an accuracy oracle / quality bar. This is the **same class as the Koopman
  rank-wall**: substitute-compute-with-acceptable-error, OOD-risk, KL≠0. Genuine research.
- **Either way it is v1.5+:** exact = hard/low-ROI build; approximate = quality research. Neither is a v1
  "just turn it on."

**Does KV-dedup already capture the deliverable attention-HBM lever?** **Yes for footprint** — the KL=0,
measured, shipped KV-dedup memory substrate delivers the exact KV-HBM-capacity lever now. The
attention-*math* lever (faster/approximate compute) is additive headroom, not the same deliverable, and is
quality-gated/kernel-work → v1.5.

**Founder verdict:** W.5 deferred attention-math substitution **by design** (its bar was intercept+
passthrough), and that deferral is **correct**: the deliverable attention-HBM lever (KV footprint) is
**already shipped via KV-dedup (KL=0)**, while attention-math/kernel substitution is **v1.5** — either a
hard low-ROI build (beat FA3 exactly) or a quality-research item (approximate + KL acceptance gate). It is
**not a v1 gap and not a quick HANDLED-wire.** Note (`W5:134,142-144`): W.5's role for Goal 3 is the
attention **intercept** (wired, FA3 5280/cell); the MFU headline composes intercept + W.4 POOL + P.3 — the
attention *substitution kernel* is explicitly the v1.x part of that composition.

**Anil decides:** accept attention-math substitution as v1.5 (recommended — KV-dedup delivers the v1
attention-HBM lever; FA3 is already optimal so an exact kernel is low-ROI and an approximate one is
research), or scope a v1.5 attention-kernel + KL-acceptance-gate research item. No build, read-only.
