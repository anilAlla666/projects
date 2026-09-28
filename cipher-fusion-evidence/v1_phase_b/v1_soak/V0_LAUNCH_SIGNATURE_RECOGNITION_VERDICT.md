# Launch-signature agnostic recognition — IS IT ACHIEVABLE? (empirical verdict)

**Date:** 2026-05-31. **Type:** READ-ONLY (cite file:line + captured geometry; no build, no commit,
anchors unchanged). Challenges the prior scope's "multi-week research → defer" with the actual data.
Question (bounded, not "perfect universal classifier"): do FA2/FA3/FlashMLA/PagedAttn + decode-GEMMs have
**distinctive, stable launch GEOMETRIES** (grid/block/shmem — algorithm-keyed, framework-agnostic) that a
**fail-safe** recognizer can key on, NAME-independently?

**VERDICT: BOUNDED — the agnostic geometry recognizer is largely ALREADY BUILT.** My prior "research"
claim was wrong; it missed `cipher::classify_launch`. A 7-class geometry-fingerprint classifier exists,
is **calibrated from H100 profiling**, is **wired on the hot path**, and has a **fail-safe default**. The
genuine remaining work is a **~1-2 day capture-and-validate** (confirm/tune the ATTENTION thresholds
against the *actual* vLLM FA3 main-kernel geometry, using the 6-pattern fn-labels as the oracle) + a
recognition→substitution routing wire (~days). **The substitution KERNEL stays v1.5 (W.5 wall) — but
that is the LIFT, separate from RECOGNITION.** "Recognition is a build; substitution is v1.5."

---

## What is BUILT (cited — the prior scope missed this)

1. **Geometry-fingerprint classifier — `cipher::classify_launch`** (`include/may13/cipher_classify.hpp`):
   7 op-classes from grid/block/shmem ALONE (no name): GEMM(0), **ATTENTION(1)**, CONVOLUTION(2),
   ELEMENTWISE(3), REDUCTION/RMSNorm/softmax(4), MEMCPY_TRANSPOSE(5), ITERATIVE_CUSTOM(6), UNCLASSIFIED
   (0xFF) (`:45-51`). **ATTENTION rules are GEOMETRY, not name:**
   - `:102-104` FA2/FA3: `is_2d_grid && bx==128 && shared_bytes≥32768 && grid_aspect≥8.0 → ATTENTION`.
   - `:108-110` FMHA: `shared_bytes≥49152 && bx≤256 && by==1 && grid_aspect≥4.0 → ATTENTION`.
   - `:114-121` GEMM: `2d_grid && 1d_block && large_shared && grid_aspect<32` (+ a square-block variant).
   **Thresholds "derived from profiling H100 Llama-3 70B workload"** (`:73`) — i.e. calibrated against
   captured geometry. **This IS the framework-agnostic recognizer the audit asked for.**
2. **Fail-safe by construction:** unmatched geometry falls through to `ITERATIVE_CUSTOM`/UNCLASSIFIED
   (`:170` tail) — i.e. **unrecognized → not a substitution class → passthrough.** Substitution would only
   fire on a positive high-confidence ATTENTION match. Never wrong-substitutes by default.
3. **Wired on the hot path:** `cipher_cupti.c:191-199` (Week 2 Step 6) passes the full captured geometry
   (`call.grid_x/y/z, block_x/y/z, shared_bytes`) into `cipher_rt_classify_route()`; the registered
   `may13_default_classifier` (`cipher_rt_classify_substrate.cpp:53-60`) calls `classify_launch`.
4. **Capture pipeline + REAL captured geometry:** `cipher_cupti.c:156-169` captures grid/block/shmem for
   every launch; `src/may13/cipher_kernel_table.cpp:210` dumps `grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u`
   per fn-pointer. Captured examples: **Marlin GEMM `grid=(132,1,1) block=(256,1,1) shmem=98304`** (96 KB
   via `cuFuncSetAttribute` attr 8, `cipher_rt_marlin_engine.cpp:410`); **FlashAttn combine `grid=4
   block=256`** (`k1_diagnostic/.../A4_llama3...prefill.log`). So geometry IS captured + dumpable.
5. **6-pattern provides labeled fn-pointers:** `cipher_rt_attn_6pattern.c:43-52` stores `g_attn_p{1,2,5,6}_
   orig` = the resolved FA2/FA3/FlashMLA/PagedAttn fn-pointers; `cipher_cupti.c:160,169` captures the SAME
   `fn` per launch → a cross-reference yields (fn = FA3) ↔ (fn launched with geometry G) = **labeled
   ground truth.** (Today the cross-reference isn't wired — a small build, §gaps.)

## What is NOT YET done (the honest, data-earned gaps)

- **The MAIN FA3 `run_mha_fwd` launch geometry is NOT captured/logged** — only the FA3 *combine* kernel
  (`grid=4 block=256`) and Marlin's GEMM. So I **cannot claim from captured data that FA3-main is
  distinctive from a GEMM** — I can only say the classifier *has* distinctive rules and Marlin-GEMM's
  point (96 KB shmem, persistent `grid=132`) is a known GEMM anchor. **NOT FOUND: the vLLM FA3 main-kernel
  grid/block/shmem.** (The capture pipeline + 6-pattern label make this a ~1-day measurement.)
- **The ATTENTION thresholds are UNVALIDATED on the actual vLLM FA3** (calibrated on Llama-3 70B, not
  confirmed to fire on vLLM-FA3-on-this-H100). The rule's `bx==128` (`:102`) is specific — Marlin uses
  `bx=256`, FA3-combine uses `block=256`; if vLLM's FA3 main kernel uses `bx=256` it would fall to the
  broader FMHA rule (`:108`, `bx≤256`) — **a tuning question the capture answers**, not a wall.
- **classify=ATTENTION is NOT routed to substitution** — it feeds workload-detection today, not a
  substitute actuator.
- **The 6-pattern labels are not cross-referenced** to validate/tune the classifier (the data exists; the
  link is unwired).

## DISTINCTIVENESS — the structural discriminator (from captured + calibrated data)

- **shmem is a real discriminator, and the classifier already uses it:** Marlin GEMM = 96 KB (captured);
  the ATTENTION rule needs `shmem≥32-49 KB` + high `grid_aspect`; quantize/elementwise/reduction kernels
  use shmem 0 or `bx×dtype` (`:148-152`); FA3 on Hopper is expected to request large dynamic shmem (the
  rule's premise). So shmem + grid-aspect (seq≫heads vs balanced GEMM tiles) is a calibrated separator —
  **the rules encode exactly this.** The UNVALIDATED piece is only whether the real vLLM FA3 main-kernel
  numbers land inside the existing ATTENTION band (the ~1-day capture).
- **This is a SUPERVISED validation, not blind clustering:** the 6-pattern labels (fn=FA3) + CUPTI geometry
  give labeled examples → "does `classify_launch(FA3-geometry)==ATTENTION`?" is a checkable supervised
  question. **Prior vLLM-specific work ACCELERATES it** (it is the validation oracle), contrary to "thrown
  away." (One reader called the 6-pattern "doesn't accelerate" — correct only in that it's *not wired
  today*; the labeled data it produces is exactly the oracle.)

## THE VERDICT — BOUNDED (a build), not research/defer

**Agnostic geometry-based recognition for OUR kernels is BOUNDED** — the classifier, the calibrated
ATTENTION/GEMM rules, the fail-safe default, the capture pipeline, and the 6-pattern labels all EXIST. It
is **not** "perfect universal classifier" research; it is **validate-and-wire** of an existing,
hot-path-wired, H100-calibrated recognizer.

**Day estimate (recognition only; substitution kernel is v1.5):**
- Capture the vLLM FA3/MLA/PagedAttn + decode-GEMM main-kernel geometries (run vLLM with the
  `cipher_kernel_table` dump on; the 6-pattern fn-labels them): **~1-2 ED.**
- Cross-reference 6-pattern fn → CUPTI geometry → validate the `classify_launch` ATTENTION verdict; tune
  the thresholds if the real FA3 numbers sit outside the band: **~2-3 ED.**
- Wire recognition→substitution routing actuator on the classify substrate (default-OFF, fires only on
  high-confidence ATTENTION): **~2-3 ED.**
- Fail-safe confirm (UNCLASSIFIED→passthrough is the default; confirm substitution gating): **~1 ED.**
- **≈ 1-1.5 weeks for agnostic RECOGNITION + routing.** (vs the prior scope's "2-6 weeks research" — that
  was wrong; the recognizer exists.)

**File manifest (recognition build):**
- `include/may13/cipher_classify.hpp` — tune ATTENTION thresholds IF the captured vLLM FA3 geometry needs
  it (else unchanged).
- `cipher_rt_classify_substrate.cpp` — register a substitution-routing actuator (default-OFF) that fires
  on classify=ATTENTION.
- `cipher_cupti.c` and/or a net-new validation harness — cross-reference 6-pattern fn-pointers to the
  classify verdict (the labeled-validation oracle).
- (v1.5, SEPARATE) the substitution KERNEL + per-pattern KL acceptance gate — the W.5 wall; the LIFT.

**Build vs defer — RECOGNITION is a BUILD (~1-1.5 wk), the LIFT (substitution kernel) is v1.5:**
- The framework-agnostic *recognition* keystone (de-couple from `_vllm_fa3_C` symbol-keying) is achievable
  now — the classifier exists; this is validate-and-wire. **Build it.**
- The *substitution* (the attention-math kernel + KL gate) remains v1.5 (W.5 quality research) — but that
  is the LIFT, not the agnostic-recognition keystone. **So: build recognition now (CIPHER becomes
  framework-agnostic at the launch boundary); the substitution kernel lands in the v1.5 bundle.**
- **Correction to `V0_CULAUNCHKERNEL_SUBSTRATE_SCOPE.md`:** that scope's "agnostic classifier = 2-6 weeks
  research, defer" is **superseded** — the classifier is built + calibrated + fail-safe + hot-path-wired;
  recognition is ~1-1.5 wk of validate-and-wire. The multi-week item was a phantom (I missed
  `cipher_classify.hpp`). The genuine v1.5 item is the substitution KERNEL, not the recognizer.

**Earned caveat (honesty):** the one thing not provable from *current* captured data is that the vLLM FA3
*main* kernel's geometry lands in the existing ATTENTION band — only the combine kernel + Marlin GEMM are
logged. That confirmation is the ~1-2 day capture (supervised by the 6-pattern labels), not research. So:
**BOUNDED, pending a 1-2 day capture-and-validate that the structural prior + the calibrated rules + the
labels all point to PASSING.** Read-only — no build, no commit, anchors unchanged. Anil decides.
