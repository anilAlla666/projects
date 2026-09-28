# Koopman firing diagnostic — does the WORKING config reproduce, and where does firing break?

**2026-06-05. READ-ONLY. No `.so` change** (anchor `cipher_rt_phase4/libcipher_rt.so` md5 `2edba0d2` unchanged, GPU
reset clean). Harnesses: `koopman_diag_step2.py` (offline 2×2 + gate metric), `koopman_step5_live.py` (live gate ladder).
Results: `koopman_step2_results.json`. This re-tests the **working config** (per-layer r=16, narrow prompts, selected
layers) — *not* tonight's frontier test (shipped uniform r=64 on diverse activations).

## Verdict (combination — the question allows it)

**Not dead, not reproduced.** It's a combination, led by two *workload-independent* findings — **(b)** the shipped binary is
config-drifted so the working config can't run read-only, and **(c)** the working operator has an empirical accuracy floor
that fixing the wiring would not clear — plus an **(a)** kernel-of-truth: the narrow-manifold premise is real, but it only
buys *firing*, not *correctness*.

- **(b) wiring / config drift [code-proven, durable].** The binary is **FP16_R=64** (`cipher_block_sub_kernel.cu:171`); the
  working **r=16** MLP-block path (`cipher_wrapper.py` `set_ptrs` → `register_shape`) feeds 16-row factors into a 64-row
  copy → **over-reads 4×** (`:487–488`). So the working r=16 config cannot run correctly on the shipped binary read-only.
  The historical **M-hash registry bug is gone** (`find_shape` is M-agnostic, `:440`).
- **(c) accuracy floor on the working operator [offline-measured, durable].** Even in the working condition (narrow, r=16,
  in-sample, selected layers), the MLP-block **linear** rank-r surrogate's output residual **floors at ~0.25 (range
  0.20–0.73), never <0.08, and is flat from r=16 to r=64, in-sample ≈ held-out.** So more rank and more fitting do not
  help — fixing the wiring would still leave a lossy surrogate. (Likely cause: a linear map of the *nonlinear* SwiGLU MLP;
  also consistent with the narrow data being only ~rank-4 so extra rank fits noise. The *empirical floor* is the
  load-bearing fact; the mechanism is the likely explanation.)
- **(a) narrow premise is real [measured].** Narrow repetitive prompts **do** collapse the input manifold (effective rank
  3.5–7.5 vs 25–68 diverse; input-gate *mean* residual 0.003–0.25 vs 0.74–0.92). So the surrogate would *fire* on
  low-diversity inputs by a mean-residual gate — but "fires" ≠ "correct" (see (c)). On *diverse* inputs the OOD gate
  correctly rejects (max-residual 0.96).
- **Live did not reproduce, and did not even reach the math walls in this run:** all three arms tok/s ≈ 8.8 (flat),
  `handled=0`. In *this short 128-token decode* firing stopped at the registry rung (no shape registered) — but that is a
  **test-design artifact, not a wall**: registration is achievable (the prior task's Part B registered shapes by feeding
  2000 distinct rows). The durable blockers are (b) and (c) above, plus rung-4 OOD-on-diverse.
- The headline numbers (**9.3% / 81.9 vs 74.9 tok/s / fit_error<0.08**) are **not locatable in the repo** (like the prior
  7.43× phantom) and **do not reproduce live**.

---

## Step 1 — RANK: is the shipped kernel r=64 uniform or per-layer?

**Uniform r=64, compile-time, (K,N)-keyed — no per-layer rank.**
- `cipher_rt_phase4/src/may13/cipher_block_sub_kernel.cu:171` → `#define FP16_R 64`. The kernel rank is a **compile-time
  constant** (confirmed in the *shipped* source tree, the object `may13_cipher_block_sub_kernel.o`).
- `register_shape` copies `FP16_R*K_dim` / `FP16_R*N_dim` elements (`:487–488`, `:500–501`) — always 64 rows.
- `find_shape(K_dim, N_dim)` (`:440–446`) keys on **(K,N) only**, no rank field, no per-layer field.
- **Config drift:** the legacy per-layer API still documents **r=16** — `struct LayerPtrs { float* V_T; // (16,4096) … }`
  (`:775–779`), and `cipher_koopman_fp16_set_ptrs` (`:786–794`) feeds those 16-row buffers straight into
  `register_shape(4096,4096,…)`, which then reads `64*4096` elements → **reads 4× past the buffer**. The working r=16
  factors cannot be loaded correctly into the shipped r=64 binary. Running r=16 read-only would require zero-padding to
  64 rows (→ r=16 *quality* at r=64 *speed*), so it cannot reproduce a speedup.

## Step 2 — CALIBRATION INPUT: does narrow-prompt r=16 drop residual below the gate? (load-bearing)

Operator = the **MLP block** (the working config: `cipher_wrapper.py` `install_all_mlp_hooks` → `mlp.forward` replaced;
X = MLP input hidden 4096, Y = MLP output 4096; matches `LayerPtrs (16,4096)`). Two metrics, two sub-questions:
**fires** = input-subspace per-row-max residual `max‖x−VₓVₓᵀx‖/‖x‖` vs β=0.05 (what `cipher_koopman_fp16_ood_max_residual`
computes); **correct** = output residual `‖Y−XÂ‖/‖Y‖` (Tikhonov) + the working `fit_error<0.08`. Exact SVD = optimistic.

**Narrow r=16 (the working condition) vs Diverse r=64 (tonight's frontier), side by side** (layer 8, representative):

| condition | input effrank (pr / @90%) | gate max-resid (fires≤.05?) | gate **mean** | output resid in-samp / held-out (fit<.08?) |
|---|---|---|---|---|
| **narrow, r=16** | **3.5 / 11** | 0.994 (**no**) | 0.162 | 0.292 / 0.282 (**no**) |
| narrow, r=64 | 3.5 / 11 | 0.976 (no) | 0.077 | 0.258 / 0.248 (no) |
| **diverse, r=64** | 68.1 / 565 | 0.981 (no) | 0.864 | 0.756 / 0.914 (no) |

Across all selected layers {0,8,12,23}: narrow input effrank 3.5–7.5 (vs diverse 25–68) — **the narrow premise is real**.
But (1) the **max-row** gate passes only at **L0 r=64** (0.021); everywhere else 0.73–0.99 even narrow. (2) The **mean**
gate *does* drop on narrow (0.003–0.25) — so a mean-based gate would fire in-scope. (3) **The load-bearing fact: output
residual floors at ~0.25 (range 0.20–0.73), never reaches <0.08, is flat from r=16 to r=64, and in-sample ≈ held-out.**
So it is not overfitting and not rank-limited — extra rank and extra fitting don't help. `fit_error<0.08` never reproduces
(best ~0.25, in-sample). *Likely* mechanism: a linear map cannot reproduce the nonlinear SwiGLU MLP except on near-identical
inputs (the output-cache/memoization regime); this also fits the narrow data being only ~rank-4. The empirical floor is
proven; the nonlinearity attribution is the most probable cause, not separately proven here.

## Step 3 — REGISTRY / WIRING: does the shape lookup find the registered shape?

**The historical M-hash bug is GONE; the live break is "no shape ever registered."**
- `find_shape(K_dim, N_dim)` (`:440`) ignores M → no M-mismatch. The `e->M==M` check survives only in the **output cache**
  (`KoopmanCacheEntry`, `:546–624`), a perf path, not the fire gate. **Correctness landmine:** `cache_lookup_ptr` (`:603`)
  returns the cached output on **pointer identity with no content check** — in decode the same buffer pointer is reused with
  new content, so a cache "hit" would return **stale output** (fast because wrong). Cache is default-ON (`:677`).
- **Live (Mistral-7B fp16, narrow prompt, 128-tok decode):** counters `calls_total=524, handled=0, skipped=524,
  skip_dtype=0`. So calls reach the actuator and the registry, and **100% miss `find_shape` → skipped**. The shape is
  never registered because live `cipher_edmd_live_collect` requires accumulation that this short 128-token decode didn't
  reach — it needs **2000 rows (≤8/call) and/or ≥8 distinct activation pointers** before the give-up at 20 calls
  (`cipher_edmd_live.cpp:35,53,61,507`); my counters don't separate which sub-cause fired. **This is a test-design
  artifact, not a wall:** the prior task's Part B *did* register shapes by feeding 2000 distinct-pointer rows. So in *this*
  run firing stops at the registry rung (3), upstream of the OOD gate — but registration is achievable; the durable
  blockers are rungs 4–5 (OOD-on-diverse + the accuracy floor).

## Step 4 — DTYPE/SHAPE GATE: is the test hitting FP16, not bf16 early-exit?

**Yes — FP16, dtype gate passes.** Live `skip_dtype=0` and `calls_total=524>0` (vs the historical bf16 run's `total=0`,
all blocked at the `CUDA_R_16F` gate, `WEEK_14_FOLLOWUP_OPTION_2_STEP_0_5`). So the dtype early-exit is **not** the blocker
here; the model ran fp16 and calls reached the actuator. (bf16 production *would* block at rung 1 — a separate, known issue.)

## Step 5 — REPRODUCTION: does the 9.3% reproduce? (gate ladder, live)

Three arms, Mistral-7B fp16, narrow repetitive prompt, locked 1200 MHz, 128-tok decode:

| arm | tok/s | calls_total | handled | skipped | skip_dtype |
|---|---|---|---|---|---|
| OFF (no injection) | 8.76 | 0 | 0 | 0 | 0 |
| Koopman ON, gate β=0.05 | 8.85 | 524 | **0** | 524 | 0 |
| Koopman ON, **forced** β=0.99 | 8.76 | 524 | **0** | 524 | 0 |

**No reproduction: tok/s flat within 1% noise; `handled=0` in every arm.** Forcing the OOD gate open (β=0.99) changes
nothing → the drop is **upstream** of the OOD gate, at the registry (no shape registered). Output text is byte-identical
across arms (no substitution occurred, so quality is trivially preserved — but so is the lack of speedup).

**The gate ladder, with where it breaks:**

| rung | check | result (this config) |
|---|---|---|
| 1 | dtype `CUDA_R_16F` (`koopman_engine.cpp:102–105`) | **PASS** (fp16, skip_dtype=0) |
| 2 | classifier `koopman_engage` (`koopman_engine.cpp:~120`) | **PASS (inferred)** — `calls_total=524>0` and the engage-gate sits before the `g_calls_total` increment, so calls passed it; I did not dump `workload_class`/`koopman_engage` this run |
| 3 | registry-hit `find_shape` (`block_sub_kernel.cu:440`) | **first gate hit in this test → skipped=524** (no shape registered; registration didn't complete in a 128-tok decode — **test artifact, registration is achievable**, see Part B) |
| 4 | OOD residual ≤ β (`koopman_engine.cpp:207–214`) | **not reached this run** (forcing β=0.99 had zero effect) — offline proxy: rejects diverse (max-gate 0.96) and most narrow r=16 (0.73–0.99) |
| 5 | surrogate kernel (`block_sub_kernel.cu:668`) | **not reached this run** — offline/prior proxy: lossy (output floor ≥0.25) and slow (prior task: scalar kernel **0.04× = 25× slower** than cuBLAS) |

Rungs 4–5 are the **durable** walls (workload-independent: OOD-on-diverse + the accuracy floor); rung 3 is the first gate
hit *in this short-decode test* and is fixable (calibration completion), not a wall.

---

## Why it once "worked," reconciled

The working build predates the W14 OOD gate and ran **r=16** (cipher_wrapper.py path). On narrow repetitive prompts the
input manifold is ~3–7 dim, so the surrogate registered and fired; the **9.3%-class speedup, if real, was Amdahl-possible
only for the whole MLP block** (single 4096→4096 op on 12/32 layers caps at **0.96%**; whole MLP block caps at **~11%** —
`koopman_diag_step2.py` Amdahl block). But the quality-preserving speedup path is the **output cache** (O(1) memcpy on
*identical* inputs = memoization), not the lossy linear surrogate. On the **shipped** binary none of this reproduces:
r=64 drift over-reads the r=16 factors, calibration never registers in a normal decode (rung-3 skip), and the surrogate
itself hits a nonlinearity wall. The specific 9.3% / 81.9 / 74.9 / fit<0.08 figures are **not located in the repo**.

**STOP at the report.** Fixable (out of read-only scope): rebuild FP16_R=16, complete calibration (force_fit is a no-op —
`cipher_edmd_live.cpp:538–543`), and the cache needs a content check. The math limit (linear surrogate of a nonlinear MLP)
is real but is about linearity, not rank.
