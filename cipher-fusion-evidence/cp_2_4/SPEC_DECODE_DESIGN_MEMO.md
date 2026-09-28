# CP 2.4 sub-task (iii) — Speculative decode: design memo

**Date:** 2026-05-16. **Status:** DESIGN MEMO — awaiting adjudication before
any code (atomic STEP: design-memo → approve → build). Builds on
`CP_2_4_DESIGN_MEMO.md` §4.

---

## 0. Audit — what exists in the v2 stack

Searched `cipher_rt_phase4/`, `cipher_workloads/`, the phase-4 evidence trees.
**No CIPHER speculative-decode actuator exists.** The one spec-related asset is
`cipher_workloads/drivers/wl10_speculative.py` (33 lines) — a **vLLM**-based
reference *workload* (`LLM(..., speculative_model=...)`), a separate stack
used only for baseline measurement. Nothing to port. Memo §4 "build, not
port" confirmed: **build from near-zero.** `wl10_speculative.py` is retained
as a baseline-comparison reference only.

## 1. Draft models

- **Llama arm:** **Llama-3.2-1B-Instruct** draft + Llama-3.1-8B target — the
  May-13 2.96× scorecard pairing. First-party, same model family.
- **Mistral arm:** **n-gram draft** (prompt-lookup / op31 path) — smallest
  dependency: no draft model to load, no extra weights, no extra GPU memory.
  Resolves memo §4.3 / §8-decision-2 (Mistral-7B has no first-party 1B
  sibling) in favour of the lowest-dependency option.

## 2. Draft–target token alignment

Speculative decoding compares **token IDs** between draft and target — they
must share a tokenizer/vocabulary.

- **Llama arm:** Llama-3.2-1B and Llama-3.1-8B are the same Llama-3 family and
  use the **identical 128 K tiktoken-based tokenizer** — token IDs align
  exactly. Compatible. *Pre-build verification (cheap):* hash both
  `tokenizer.json` / vocab and assert equality before the build proceeds.
- **Mistral arm:** the n-gram draft has **no tokenizer of its own** — it
  predicts continuations directly from the target's own emitted token history
  (target token space). No alignment question — N/A by construction.

## 3. Integration point — where spec decode hooks in

Speculative decode is **not** a matmul/attention-dispatch actuator — it does
not intercept a GEMM or an SDPA call. It restructures the **decode loop**
(draft k tokens → target batched verify → accept/reject). It therefore sits
**above** the v2 dispatch layer:

- **Mechanism (recommended): a generate-path injection**, operator-policy,
  transparent to customer code — the same philosophy as op #1 S2b ("owns the
  KV cache by *being* the cache class"). CIPHER's injection bootstrap installs
  a speculative-decode generate path; the customer's `model.generate()` is
  routed to it; customer code is unchanged.
- **Composition with Marlin is free, by construction.** The draft forwards
  and the target verify forward each still flow through the v2 matmul/attn
  dispatch — so their GEMMs hit the Marlin actuator normally. Spec decode ×
  Marlin compose because spec sits above, Marlin below.
- **Coverage-immune.** Unlike interception-layer actuators (the `pause_note.md`
  43 %-coverage problem), spec decode *owns* the decode loop — there is no
  dispatch to "miss." It is a control-ownership-layer port, standard ship gate.

This is the main item for adjudication — see §7.

## 4. Verify path (memo §4.2, restated)

Draft generates k tokens (k ≈ 4–5) autoregressively. Target runs **one
batched forward** over the k draft tokens → k+1 logit vectors. Accept the
longest prefix where `sample(target_logits[i]) == draft[i]`; on first reject,
resample that position from the target distribution and discard the rest.
At temperature 0 (greedy) the accepted output is **byte-identical** to plain
target decoding — spec decode preserves the target distribution exactly.

## 5. Draft placement — correction to memo §4.1

Memo §4.1 said "draft runs in its own CUDA stream / **green context**." Post
CP 2.4 Fix A this needs correction: the green context is the 8-SM partition
that Marlin cannot run in (`MARLIN_HANG_ROOT_CAUSE.md`,
`../PHASE_5_ARCHITECTURE_REVISION.md`). For CP 2.4 **single-tenant** scope
there is no partitioning — Fix A runs everything on the full GPU.

Revised: the draft model is co-resident on the same H100 (a 1B draft is
~2 GB fp16 / ~0.7 GB INT4; the H100 has the headroom), running on **its own
CUDA stream in the primary context** — own stream gives draft/verify overlap
without false-serialization; primary context keeps it Fix-A-correct (a
Marlin-INT4 draft is fine — Fix A runs Marlin full-GPU). Green-context
placement of the draft is a Phase 5 multi-tenant question, gated on
partition-aware Marlin.

## 6. Performance target & gate

- **Target:** spec-decode lift **1.75×** on top of Marlin INT4.
  Marlin 1.62× × spec 1.75× ≈ **2.84×** composed. (With DVFS likely OUT —
  memo §3.3 disable rule, pending the running DVFS sweep — the composed claim
  is Marlin × spec ≈ 2.84×, not the original 2.96×.)
- **Gate (per memo §6):** spec-decode per-lever lift, **n = 5 matched pairs**
  (spec off vs on, both Marlin-on), Llama arm and Mistral arm separately;
  report mean lift + 95 % CI and the **acceptance rate** (the lift driver);
  output equivalence — byte-identical at temp 0.

## 7. Open decisions — adjudication needed before build

1. **Integration mechanism** (§3) — confirm the generate-path injection
   (operator-policy) approach, vs. an alternative (e.g. an explicit
   `CipherSpeculativeModel` wrapper the operator instantiates).
2. **Gate workload** (memo §8 decision 1) — Llama-3.1-8B, Mistral-7B, or
   both? The Llama arm has the clean 1B draft; the Mistral arm is n-gram-only
   (~0.59 match per memo §4.3 → a smaller lift). This sets which arm the 1.75×
   target is held against.
3. **k** (draft tokens per round) — memo says k ≈ 4–5; pick a value (or make
   it adaptive on running acceptance rate).

Build STEP begins on adjudication of §7. Anchors held: kmod 0.4.8 `e2f50452`,
libcipher_v2 `86618c30`, libcipher_rt `5e304549`.
