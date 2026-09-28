# Koopman fix — staged design (confirm the floor live → maybe beat it)

**Date:** 2026-06-05
**Author:** CIPHER work arc (Anil), via Claude Code brainstorming
**Status:** Approved design → writing-plans next
**Scope boundary:** This is the **first action in the entire CIPHER arc that leaves read-only scope and rotates the anchor.** Anchor `cipher_rt_phase4/libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` stays **byte-frozen** (backed up ×2: `cipher_rt_phase4/libcipher_rt.so.ANCHOR_2edba0d2` and `/home/ubuntu/libcipher_rt.so.ANCHOR_2edba0d2_20260605`). All rebuilds land as **separate tagged artifacts**; canonical `libcipher_rt.so` is promoted only after explicit validation.

---

## 1. Why / goal

The 2026-06-05 `KOOPMAN_FIRING_DIAGNOSTIC.md` left a verdict of **"not dead, not reproduced"** with concrete, *out-of-read-only-scope* next steps. This spec executes them. The user chose a **staged** goal:

- **Stage 1 — confirm the (c) accuracy floor LIVE.** Rebuild so the rank-r EDMD surrogate actually *fires* on the GPU at the working config, then measure its output residual / quality / speed. The diagnostic *predicts* this confirms an earned-negative (the ~0.25 output-residual floor, rank-flat r16≈r64, in-sample≈held-out, attributed to a linear map of the nonlinear SwiGLU MLP). Stage 1 turns that offline prediction into a **live** result — at the cost of the anchor.
- **Stage 2 — attempt to BEAT the floor.** Only if Stage 1 fires cleanly. The lever is **nonlinearity** (the floor is a linearity wall, not a rank wall) and/or the **corrected output-cache memoization** path. Separate spec; gated on Stage 1 evidence.

The deliverable is **a conclusive experiment**, not merely a code change. The failure mode to avoid is an *inconclusive* live number that spends the anchor for nothing.

## 2. The object being measured (read from source)

The live firing path is the **scalar** surrogate kernel `cipher_koopman_fp16_decode_generic` (`block_sub_kernel.cu:743`) — WMMA is off by default (`CIPHER_USE_WMMA=1` to enable; comment `:725` notes it is slower for r=16). The surrogate computes `out = W · (K_op · (Vₜ · x))` where `K_op` is built as **identity** (`edmd_live.cpp:537–539`), i.e. a pure rank-r **linear projection** `W·Vₜ·x` with `Vₜ` from a randomized rank-r SVD of the activation manifold and `W` a Tikhonov-regularized recovery (`:541–559`). This is exactly the "linear-map-of-nonlinear-MLP" object whose residual the diagnostic says floors at ~0.25.

## 3. Code changes (three)

All in `cipher_rt_phase4/src/may13/`. Each lands as a tagged `.so`; the anchor is never overwritten in place.

### 3.1 Runtime-selectable rank — `CIPHER_KOOPMAN_RANK` (default 16)
The knob must drive **both** TUs consistently (the EDMD fit rank and the kernel/buffer sizing — currently coupled by `KR_RANK=64` "must match `FP16_R`").

- `block_sub_kernel.cu`
  - Keep `#define FP16_R 64` (`:171`) as the compile-time **MAX** — shared-mem arrays `__shared__ float alpha[FP16_R]` (`:199, :260, :854`) must stay compile-time sized. Use a runtime `R ≤ 64` for **loop bounds** and **strides**.
  - Add `int R;` to `struct ShapeEntry` (`:~429`).
  - Add an `R` parameter to `cipher_koopman_fp16_register_shape` (`:478`); store `se->R`; size buffers by `R` not `FP16_R` (`:487–488, :500–501, :512–514`).
  - Scalar kernel `cipher_koopman_fp16_decode_generic`: take `R` as a runtime arg; replace every `FP16_R` loop bound and the **`K_op` stride** `K_op[tid*FP16_R+j]` → `K_op[tid*R+j]` (`:218, :281` and the strided projection loops `:206, :230, :268, :297`). **Every stride/index/sizing site must change together** (silent-corruption risk — see §5 TDD).
  - WMMA path: guard to `R==64` (off by default), or leave at compile-time `FP16_R` — do **not** silently run WMMA with `R≠64`.
- `edmd_live.cpp`
  - `KR_RANK=64` constexpr (`:35`) → runtime `g_kr_rank` from `CIPHER_KOOPMAN_RANK` (default 16). `const int r = KR_RANK` (`:431`) → `r = g_kr_rank`. Arrays declared `[KR_RANK]` (`:96–98, :114, :487`) stay sized to the max (64); loops use `r`.
  - Pass `r` into `register_shape(K, N, VT_buf, KOP_buf, W_buf, r)` (`:607`).
- **Free side-effect — fixes the (b) wiring bug:** the legacy `cipher_koopman_fp16_set_ptrs` (`:786–793`, `LayerPtrs (16,4096)`) will pass `R=16` into `register_shape`, sizing the 16-row buffers correctly instead of over-reading a 64-row copy 4×.

### 3.2 Real `force_fit` — complete calibration
`cipher_edmd_live_force_fit` (`:764`) is currently a no-op (the reason the diagnostic's live run got `handled=0` only stopped at rung 3 — no shape registered). Make it run the existing fit path (rand-SVD → build matrices → `register_shape`) on **currently-collected rows**, with a hard floor for a valid rank-r SVD (`≥ r + OVERSAMPLE`). Make `TARGET_ROWS=2000` (`:36`) env-tunable (`CIPHER_KOOPMAN_TARGET_ROWS`), default scaled `~20·r` (≈320 at r=16). This lets a short, controlled run register a shape without waiting for 2000 organic rows.

### 3.3 Cache content-check — close the stale-output landmine
`cache_lookup_ptr` (`:602–611`) matches on `input_ptr` identity with **no content check**, and the fast path returns a memcpy of stale output *before* compute (`:687–701`). In autoregressive decode the same buffer pointer is reused with new content → fast and wrong. Fix: the ptr-identity hit must **confirm content** (verify the partial hash) before returning, or fall through to the existing content-hash path (`:704`). Default-correct.

## 4. Four experimental gates (make the result conclusive)

- **G1 — Control build first (BLOCKING).** Rebuild **r=64 from current source** as `libcipher_rt.so.r64_ctrl`. Behaviorally diff against anchor `2edba0d2` (teacher-forced KL=0 on a fixed prompt; OFF-path output identical). The Makefile states the build is not bit-reproducible ("only the harness marker is added"), so the control will not be byte-identical — *behavioral* equivalence is the bar. Rationale: the diagnostic's own framing is that the binary is config-drifted, the tree has uncommitted changes. Without a same-source r=64 control, the r=16 live residual is confounded with source-vs-anchor drift and cannot be attributed to rank. **Gate: do not trust any r=16 number until the control is behaviorally equivalent to the anchor.** **If the control is NOT behaviorally equivalent, that is itself a finding** — the current source has drifted from whatever built the anchor — and a stop-and-report point, not something to paper over (it would mean no clean attribution is possible from this tree without first reconciling the source).
- **G2 — Cache OFF for the floor measurement; cache-correctness tested separately (BLOCKING).**
  - *Floor run:* `CIPHER_USE_CACHE=0` + OOD gate forced open (`CIPHER_KOOPMAN_OOD_THRESHOLD=0.99`), on the **narrow repetitive workload** (the working condition — the only input on which the surrogate fires) → measure **raw surrogate** output residual `‖Y−XÂ‖/‖Y‖` per fired op, plus PPL/KL delta and tok/s vs OFF on that same workload. Gate forced open because Stage 1 measures the surrogate's correctness *when it fires*, not whether the gate admits it (on narrow prompts the max-row gate 0.73–0.99 would refuse).
  - *Cache-correctness run:* cache ON; a decode that **reuses a buffer pointer with new content**; assert the output is **not stale** (matches the fresh-compute reference). Validates §3.3. Separate run, separate assertion.
  - The cache-ON-repetitive "memoization speedup" is a real but **Stage-2** number — it must **not** bleed into the floor run (it would hide the 25×-slower kernel behind memcpy hits).
- **G3 — Pre-registered falsifiable threshold (BLOCKING, cheap).** Decision rule written *before* the run, to prevent confirmation bias on a predicted quantity:
  - **CONFIRM the floor** if live raw-surrogate output residual **≥ ~0.15** *and* approximately flat across `r ∈ {16, 32, 48, 64}`.
  - **REFUTE** (offline replica was wrong → the whole premise reopens, Stage 2 unnecessary in its current form) if residual **< 0.08**.
  - Intermediate (0.08–0.15): inconclusive → investigate before claiming either way.
- **G4 — Row collection specified (BLOCKING).** The harness pre-feeds **N distinct-pointer rows** (Part-B style — that approach already registered shapes by feeding 2000 distinct rows) to reach the §3.2 fit floor, so firing does not die at rung 3 again for a new reason. State the exact row count and distinct-pointer count in the harness.

## 5. TDD guardrail (into writing-plans)
The `FP16_R`→`R` edit has a **silent-corruption** failure mode: miss one stride/index/sizing site and the kernel produces plausible-but-wrong output with no crash, which would masquerade as "the floor." **Before any live number is trusted**, gate the kernel with an arithmetic **unit test at both r=16 and r=64** against a torch/CPU reference. Reuse the repo's `zeta_kernel_test.py` (it previously hit rel_err 2e-4 = fp16 noise). Tests before implementation.

## 6. Success criteria

**Stage 1 is DONE when:**
1. Control build G1 passes (r=64-from-source ≡ anchor behaviorally).
2. Kernel unit test (§5) passes at r=16 and r=64 (rel_err ≈ fp16 noise).
3. The surrogate **fires live** (`handled > 0`) at `CIPHER_KOOPMAN_RANK=16` on the controlled harness (G4).
4. The floor run (G2) yields a live raw-surrogate residual + PPL/KL + tok/s, evaluated against the pre-registered rule (G3).
5. The cache-correctness run (G2) confirms the §3.3 fix (no stale output).
6. A written report records the live verdict (confirm / refute / inconclusive), the residual-vs-rank curve, and the speed result — in the project's measured-verdict style.

**Stage 2 gate:** proceed only if Stage 1 fires cleanly (criteria 1–5 met) and G3 = CONFIRM. Stage 2 = nonlinear surrogate and/or corrected-cache memoization, its own spec.

## 7. Anchor handling
- Frozen: `libcipher_rt.so` md5 `2edba0d2` — backed up ×2, never overwritten in place.
- Control: `libcipher_rt.so.r64_ctrl`. Experiment: `libcipher_rt.so.r{16,32,48,64}_calibfix`.
- Canonical `libcipher_rt.so` is promoted to a rebuilt binary **only** after explicit user validation. "Rotation" = a new identity alongside the frozen anchor, not a destroyed one.

## 8. Out of scope (Stage 1)
- WMMA-path runtime rank (off by default; guard to R==64).
- bf16 production firing (separate known dtype-gate issue at rung 1).
- Any non-Koopman actuator.
- Promoting a new canonical anchor without validation.

## 9. Open risks
- Even with `force_fit` + G4, a given `(K,N)` may not collect enough distinct rows in a controlled decode → `handled=0` for a new reason. Mitigation: harness explicitly feeds the floor count (G4); verify registration before measuring.
- The OOD-gate-forced-open measurement is a *measurement* knob, not a production claim — the report must state that at the default gate the surrogate correctly refuses to fire on diverse inputs (rung 4 doing its job).
