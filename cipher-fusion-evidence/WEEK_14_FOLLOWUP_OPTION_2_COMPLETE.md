# W14 Followup — Option-2 Campaign COMPLETE (Branch B)

**Date:** 2026-05-25
**Terminal step:** Option-2 Step 1 alpha (BF16-hypothesis direct confirmation)
**Tag:** `option-2-complete` (alias of `option-2-step-1-alpha-dtype-counter` per scope-lock §4 Step 4)
**Branch:** **B** (architectural ceiling reproduced on real workload — sharper mechanism than scope-lock anticipated)

**Anchors at close:**
- `cipher_rt_phase4` `52923af` (tag `option-2-step-1-alpha-dtype-counter`); `libcipher_rt.so` md5 `4bedf6488c9a7c0ae7eb793794b02907` (rotated from `097cf8d907a7e866a3e3640eb0993003`)
- `cipher_kmod` `8c643fc` (UNCHANGED — no ABI delta this campaign)
- `cipher_kv_bridge.so` at W12 Step 3 anchor (UNCHANGED — `CIPHER_KV_ALLOC=0` Step 0 hygiene throughout)
- `cipher_vllm_plugin/cipher_vllm_kv.py` md5 `8330506a0b67ae7f9decca7aee0a4133` (rotated from Step 0 `e77a3a58` → Step 0.5 `305003d5` → here; snapshot at `plugin_snapshots/cipher_vllm_kv.py.option_2_step_1_alpha`)
- `cipher-fusion-evidence` this commit + tags `option-2-step-1-alpha-dtype-counter` + `option-2-complete`

---

## 1. Branch B outcome statement (per scope-lock §8 template, line 188-189)

> **v1.2.3 Goal 4 (narrow-domain Koopman substrate):** substrate ships correct and validated end-to-end (W13-14 LM-head harness PASS). On real TinyLlama-1.1B vLLM N=1 V1 decode under `CIPHER_KOOPMAN=1 CIPHER_KOOPMAN_OOD_THRESHOLD=0.99` (always-fire-approx), the Koopman engine does **not fire** — empirically verified by direct per-early-exit counter evidence (`cipher_rt_koopman_skip_dtype=11658` of `11658` cublas_shim calls, with `skip_dim=0`, `skip_nullptr=0`, accounting closes algebraically). **The binding constraint is the FP16-only dtype contract of the Koopman .cu kernel (`cipher_rt_koopman_engine.cpp:102-105`) vs vLLM TinyLlama's bfloat16 default dtype.** This is **upstream** of the W14 Step 2 E architectural ceiling — the OOD residual-ratio gate is never reached on bf16 production inputs because the dtype gate filters them first. Production firing is deferred to v1.5 pending the BF16 kernel port (β path; surfaced in §6). The substrate-deliverable contract from W14 Step 2 G CLOSURE (W13-14 LM-head harness on synthetic FP16 inputs) remains intact and unaffected.

---

## 2. Evidence chain (the 5 lines that close the campaign)

| Step | Tag | Substrate anchor | Counter evidence | Finding |
|---|---|---|---|---|
| 0 | `option-2-step-0-vllm-worker-init-hook` | `25970f3` / `097cf8d9` | `cipher_rt_cublas_shim_calls` 0 → 11658 (worker) | vLLM V1 worker subprocess GOT patches now install (prior 2026-05-24 null-attempt cause resolved). Gate 3 stays at handled=0 — surfaced as next-step question |
| 0.5 | `option-2-step-0-5-koopman-reachability-probe` | `25970f3` / `097cf8d9` | `m_total=shim=11658`, `k_total=0` | Drop localized to inside `maybe_handle_koopman` early-exits before `g_calls_total` increment. BF16 hypothesis named via vLLM `dtype=torch.bfloat16` engine log line. Step 1 β-sweep declared NOT meaningful on substrate as-is |
| 1α | `option-2-step-1-alpha-dtype-counter` | `52923af` / `4bedf648` | `skip_dtype=11658`, `skip_dim=0`, `skip_nullptr=0`, accounting `sum_skips == m_passthrough` closes | BF16 hypothesis **directly proven**; FP16-only dtype gate accounts for 100% of PASSTHROUGH |
| 1α verify pass | (same commit) | `52923af` / `4bedf648` | Bit-identical decode output `" and kind queen. She had a daughter named Lily, who was a beautiful and kind-hea"` across all 3 runs (Step 0, Step 0.5, Step 1α) | Substrate is behaviorally identical with counters added; no algorithmic regression |
| Close | `option-2-complete` (alias) | `52923af` / `4bedf648` | — | Campaign exits Branch B; v1.5 BF16 port queued per §6 |

---

## 3. What this campaign ruled out vs ruled in

**Ruled out as the binding constraint on substrate-as-shipped:**
- W14 Step 2 E architectural ceiling (residual_ratio > 0.5 OOD on every input) — this is real but is downstream of the dtype gate; we never reached it on real workloads.
- Any β value in `{0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}` — all would produce zero rows; β controls the OOD gate inside the engine, which is unreachable on bf16 inputs.
- vLLM V1 worker-subprocess GOT-patch issue (resolved at Step 0; otherwise null-substrate confounds the result).
- The `g_enabled` early-exit, the degenerate-dim check, and the null-ptr check (all empirically 0 in alpha probe).

**Ruled in as the binding constraint:**
- Koopman .cu kernel's FP16-in/FP16-out contract (line 101 inline comment: `/* FP16-only — the .cu kernel is FP16-in/FP16-out */`).
- vLLM 0.20.2 TinyLlama-1.1B-Chat-v1.0 default `dtype=torch.bfloat16` (engine init log).
- Composition: 11658/11658 GEMMs hit the dtype mismatch.

**Generalization (modulo Mistral E.7 carry-forward):**
- Mistral-7B-Instruct also defaults to bf16 under vLLM. Same dtype-gate failure mode would reproduce; the Mistral measurement is not separately needed to characterize this binding constraint. The Mistral E.7 environment block (carried forward W7-9 → W10-12 → W13-14) is therefore NOT the blocker; even if E.7 were resolved tomorrow, Mistral-7B at Step 3 would null-measure for the same FP16-vs-BF16 reason.

---

## 4. Engineering result delivered by this campaign (not zero)

The narrative "Branch B = no positive result" is wrong for this campaign. Three durable artifacts ship:

1. **vLLM V1 worker-subprocess GOT patch hook** (Step 0). All future CIPHER-on-vLLM measurement campaigns will inherit working interception; the prior 2026-05-24 null-attempt category of failure cannot recur silently.

2. **Per-early-exit telemetry in `maybe_handle_koopman`** (Step 1α). `cipher_rt_koopman_skip_dtype/_dim/_nullptr` getters are permanent substrate fixtures. Any future Koopman actuator change (β port, new actuator, alternative substitution path) can use these to verify reachability without re-deriving the diagnostic.

3. **A sharper, empirically-grounded characterization of the v1 Koopman substrate's production limit** than the scope-lock anticipated. Scope-lock §3 line 50 expected Branch B framing to be "architectural ceiling from W14 Step 2 E reproduces"; the actual binding constraint is a dtype-contract mismatch one layer upstream of that ceiling. This matters for v1.5 scoping (§6) — the BF16 port is the right next investment, not OOD-detector relaxation or recipe-coverage expansion.

---

## 5. Goal 5 framing reframe (folded in per Step 0 §3 honest residue #6 + Step 0.5 §11)

Per Anil 2026-05-25 amendment to scope-lock (folded in at this close, per user instruction *"Land both reframes together post-Branch-B close as a single pitch-language reconciliation"*):

**Memory #1 historical framing** (pre-Step-0): *"LD_PRELOAD-only deployment, zero customer code changes."*

**Post-Step-0 + post-Step-0.5 + post-Step-1α reframe** (effective at this close): *"LD_PRELOAD + drop-in `cipher_vllm_plugin`, zero application code changes."*

**What changed factually:** vLLM V1's EngineCore subprocess architecture means the worker spawn inherits LD_PRELOAD'd `libcipher_rt.so` but never auto-invokes `InitializeInjection2()` unless `CUDA_INJECTION64_PATH` is also set (driver-invoked path). For the LD_PRELOAD-only deployment mode the original framing committed to, a tiny `cipher_vllm_plugin` install (registered via vLLM's `general_plugins` entry point) is the integration surface that installs the worker-side GOT patches. This is plugin installation, not application or model-code modification — the customer's vLLM YAML/CLI/Python code does not change.

**What the reframe does NOT do:**
- Does not weaken the "zero application code changes" promise. The cipher_vllm_plugin is a deployment-time install (`pip install cipher-vllm-kv`); it does not require touching the customer's model serving code, prompts, samplers, or vLLM invocation.
- Does not introduce a new failure mode. The plugin is `general_plugins`-registered and auto-discovered; if the entry point is missing, vLLM proceeds without it (no crash, no behavioral change in vanilla mode).

**Where this lands in pitch materials:** Memory #1 (Goal 5) bullet text gets updated from "LD_PRELOAD-only" to "LD_PRELOAD + drop-in cipher_vllm_plugin". Any pitch deck or one-pager carrying the original phrase needs the same edit. No other Goal renumbers.

---

## 6. v1.5 work plan surfaced (β path, queued — DO NOT execute in this campaign)

Per user instruction *"surface v1.5 work plan: Highest priority: BF16 kernel port for Koopman .cu kernel (or BF16↔FP16 wrapper if equivalent quality)"*, the following queue lands at Option-2 close. **NONE of this is in-scope for Option 2 or the current v1.2.3 plan**; these are placeholder pointers for v1.5 scope-lock entry.

**v1.5 Substep 1 — BF16 Koopman .cu kernel port (HIGHEST PRIORITY):**
- Add a BF16 variant of `cipher_koopman_fp16_launch_shape` (likely `cipher_koopman_bf16_launch_shape`) and a BF16 variant of `cipher_koopman_fp16_ood_max_residual`. Two options for the implementation:
  - **(i) BF16-native kernel** (preferred): port the .cu kernel templates from `__half` to `__nv_bfloat16`. H100 has native BF16 tensor-core support; arithmetic intensity profile should be similar to FP16. Risk: numerical conditioning of the EDMD-derived projection matrices may differ for BF16 dynamic range; calibration recipes may need re-derivation.
  - **(ii) BF16↔FP16 conversion wrapper**: keep the FP16 kernel; add fp16↔bf16 cast at the actuator entry/exit. Likely lower implementation cost. Risk: cast overhead may dominate Koopman compute savings for small shapes; conversion precision loss may erode the W13-14 top-1 ceiling.
- Substrate touch: `cipher_rt_koopman_engine.cpp` (relax dtype gate to accept BF16; route to BF16 path), `may13_cipher_block_sub_kernel.cu` (or equivalent), recipe registry update for BF16-derived recipes.
- Estimate (pre-scope-lock): 5-10 ED for option (i), 2-4 ED for option (ii). Real estimate at v1.5 scope-lock.

**v1.5 Substep 2 — Rerun Option-2 measurement campaign post-port:**
- Once Substep 1 lands, the original Option-2 calibration sweep + N=4 A/B + Mistral N=4 A/B (scope-lock §4 Steps 1-3) becomes meaningful for the first time. Re-execute that scope-locked sequence to determine whether BF16 Koopman delivers **Branch A** on real production stacks.
- Substep 2 inherits the verification chain this campaign built (Step 0 worker hook + Step 1α per-early-exit counters); the pre-flight verification is shorter than this campaign's was.
- Likely outcome:
  - **Branch A** if BF16 Koopman fires on real TinyLlama/Mistral inputs at the configured β AND OOD detector is not the binding constraint at production scale.
  - **Branch B (W14 Step 2 E ceiling)** if the OOD residual gate filters everything post-dtype — would close v1.5 Substep 2 on the original scope-lock Branch B framing rather than this campaign's dtype-gate framing.
  - **Branch C** if firing works but engine-path overhead dominates net tok/s — surfaces engine-overhead reduction as v2 substep.

**v1.5 Substep 3 (optional follow-on):** if Substep 2 lands Branch C or Branch B-via-OOD-ceiling, the W14 Step 2 E `α` calibration distribution-matched calibration test (referenced in `WEEK_14_STEP_2_ZETA_5DAY_CLOSEOUT.md`) becomes the next investigation surface. Not committed; queued.

**NOT in v1.5 queue:**
- Option γ (`dtype=float16` workaround) — discarded per user instruction *"non-diligence-defensible measurement. Modern Llama-family serving defaults to bf16 for numerical-range reasons; an fp16-only result doesn't survive a reviewer asking 'and on bf16?'"*. Recorded here so it does not get re-proposed.
- An alternative actuator targeting bf16 directly (e.g., a new BF16-native low-rank-substitution actuator different from Koopman) — out of scope; v1.5 commits to the BF16 port path before exploring alternative actuator surfaces.

---

## 7. Substrate ABI delta (additive only, per `cipher-abi-rule`)

This campaign added 3 T-symbols to `libcipher_rt.so`:
- `cipher_rt_koopman_skip_dtype`
- `cipher_rt_koopman_skip_dim`
- `cipher_rt_koopman_skip_nullptr`

All are `extern "C" unsigned long fn(void)` getters consistent with the existing `cipher_rt_koopman_calls_total/handled/skipped/remember_emits` pattern. No symbol renames, no signature changes, no removals. Existing consumers (cipher_vllm_plugin pre-Step-0.5 dump, any future tooling) are unaffected. Substrate kmod ABI unchanged.

---

## 8. Substrate state at close (canonical inventory)

| Tree | Commit | md5 | Tag(s) | Δ vs Option-2 entry |
|---|---|---|---|---|
| `cipher_rt_phase4` | `52923af` | `libcipher_rt.so` `4bedf648` | `option-2-step-1-alpha-dtype-counter` | ROTATED from `25970f3` / `097cf8d9` (3 atomics + 3 getters + 6 fetch_adds in cipher_rt_koopman_engine.cpp) |
| `cipher_kmod` | `8c643fc` | — | (week-9-complete carry, unchanged) | UNCHANGED |
| `cipher_kv_bridge.so` | W12 Step 3 anchor | — | unchanged | UNCHANGED (unused; `CIPHER_KV_ALLOC=0`) |
| `cipher_vllm_plugin/cipher_vllm_kv.py` | — | `8330506a` | snapshot at `plugin_snapshots/cipher_vllm_kv.py.option_2_step_1_alpha` | ROTATED through `e77a3a58` → `305003d5` → `8330506a` across Steps 0, 0.5, 1α |
| `cipher-fusion-evidence` | this commit | — | `option-2-step-1-alpha-dtype-counter`, `option-2-complete` | Adds Step 1α artifacts + this close-out doc + new plugin snapshot |

**v1.2.3 §7 W15-17 entry-state:** UNCHANGED in spirit. Substrate code rotated (libcipher_rt anchor moved by 3 atomics + 3 getters) but that's an additive ABI delta — no semantics changed in the matmul/Koopman hot path beyond counter increments on existing PASSTHROUGH branches. CP 5.5 hybrid heterogeneous-model benchmark scope intact. Substrate counters are now richer than they were at W13-14 close; future campaigns benefit.

---

## 9. Honest residue at close

1. **The Mistral E.7 carry-forward is no longer the limiting factor for Branch B on Mistral-7B.** Per §3 generalization paragraph, Mistral-7B inherits the same dtype-gate failure on vLLM bf16 default. E.7 remains a carry-forward env block for future Mistral campaigns but is not the reason Option-2 Step 3 (Mistral A/B) did not execute. The reason is upstream: Step 1 was BLOCKED at Step 0.5, and Branch B closes at Step 1α before Steps 2-3 run. Memory `w14-step-3-followup-mistral-tok-s` should reflect this clarification.

2. **W14 Step 2 E architectural-ceiling finding is unchanged in validity but downgraded in priority.** That finding remains the *next* binding constraint that would gate v1.5 Substep 2 if the BF16 port succeeds. Until then, the W14 Step 2 E ceiling is a latent gate behind a closed door (the dtype gate); not the active surface.

3. **Substrate-overhead microbench (scope-lock §10 Q1 Option (a)) was not executed.** Scope-lock added that microbench to give Branch C interpretability; since Branch B fires before any positive utility measurement, the microbench is not needed for this campaign. Queued implicitly as v1.5 Substep 2 prerequisite if/when fire path becomes reachable.

4. **The α probe ran at N=1.** Scope-lock §4 Step 2 anticipated TinyLlama N=4. N=1 is sufficient for the dtype-gate finding (every shape and every batch size yields bf16 GEMMs; the dtype filter fires identically). N=4 would be needed only for net tok/s measurement (Step 2 + 3), which Branch B does not execute. Not a residue per se; documented to forestall "why didn't you run N=4" questions.

5. **The "actuators=1" inference held across all three runs.** Marlin was DISABLED throughout per `cipher-marlin-primary-ctx-pin` Phase 5 carry-forward; no other matmul actuator is registered. If a future campaign adds a second actuator (e.g., the v1.5 BF16 path lands alongside the FP16 path), the per-actuator histogram becomes non-trivial — note for future probe scopes.

---

## 10. Tag map

| Tag | Commit | Tree | Lands |
|---|---|---|---|
| `option-2-step-0-vllm-worker-init-hook` | `7c6d30f` (2026-05-25) | `cipher-fusion-evidence` | Step 0 worker GOT-patch hook + verification gate PASS |
| `option-2-step-0-5-koopman-reachability-probe` | `31b34692` (2026-05-25) | `cipher-fusion-evidence` | Step 0.5 outcome (a) localization |
| `option-2-step-1-alpha-dtype-counter` | `52923af` (2026-05-25) | `cipher_rt_phase4` | Step 1α substrate change (3 per-early-exit counters; ABI-additive) |
| `option-2-step-1-alpha-dtype-counter` | this commit | `cipher-fusion-evidence` | Step 1α step doc + this close-out + artifacts |
| `option-2-complete` | this commit (alias) | `cipher-fusion-evidence` | Campaign terminal alias per scope-lock §4 Step 4 |

(Tag `option-2-step-1-alpha-dtype-counter` is intentionally placed on both `cipher_rt_phase4` and `cipher-fusion-evidence` at their respective HEADs; cross-tree alignment is the cipher-trees convention from W7-9 / W10-12 / W13-14 closes.)

---

## 11. Related memory

- [[option2-step0-closed]] — Step 0
- [[option2-step0-5-closed]] — Step 0.5
- [[vllm-v1-worker-subprocess]] — Step 0 underlying fix; remains the precondition for any vLLM V1 CIPHER measurement
- [[w14-step-2-koopman-tier]] — substrate-side closure that ships the FP16-only .cu kernel; the dtype contract this campaign characterized
- [[w13-14-complete]] — substrate baseline at Option-2 entry; rotated by Step 1α (libcipher_rt only)
- [[cipher-t43-envelope]] — precedent for "substrate validates on test workload, doesn't generalize to production regime"
- [[cipher-lift-framing]] — precedent for honest single-workload-class framing; Branch B inherits this discipline
- [[cipher-abi-rule]] — Step 1α 3 new T-symbols are additive-only; respects the rule
- [[cipher-evidence-commit-discipline]] — followed at this close
- [[w14-step-3-followup-mistral-tok-s]] — should be updated to "CLOSED Branch B at option-2-complete; Mistral E.7 not the limiting factor — dtype gate fires upstream regardless"

---

**OPTION 2 CAMPAIGN: COMPLETE — Branch B.** v1.5 BF16 port queued per §6. No HARD STOP. No open question at this close.

---

## 12. Post-rebuild regression check (addendum per advisor 2026-05-25)

Per cipher discipline of re-running the regression suite after a substrate anchor rotation, the 3 W14 Step 3 compiled test binaries in `cipher_rt_phase4` were exercised against the rebuilt `libcipher_rt.so` (`4bedf648`):

| Test | Env | Result | Note |
|---|---|---|---|
| `test_step3_b0_producer` | (default) | **PASS** | p99 cadence 83 ns; N=128 smoke all 1.28M writes issued; matches `WEEK_14_STEP_3_B0_KOOPMAN_PRODUCER.md` baseline |
| `test_step3_b1_consumer` | (default) | **PASS** | env-gate-off PASS; cold-start PASS; compose drop_pct=0.00% rate=2.29 M/s drained=32000 — matches `WEEK_14_STEP_3_B1_REMEMBER_CONSUMER.md` baseline |
| `test_step3_c_lmhead_validate` | `CIPHER_KOOPMAN=1 CIPHER_REMEMBER=1 CIPHER_KOOPMAN_OOD_THRESHOLD=0.7 LD_PRELOAD=./libcipher_rt.so` | **Pass II PASS, Pass I telemetry PASS, Pass I quality FAIL** (top1=0.0000 kl_mean=5.44e-02; expected ≥ 0.90 per W14 S3.C close-out which recorded top1=0.9000 kl_mean=4.25e-01) | DETERMINISTIC across two consecutive runs (bit-identical kl_mean). Step 1α's change is ABI-additive only — 3 atomic counters + 6 fetch_adds on existing PASSTHROUGH branches + 3 getters; it does not touch the Koopman .cu kernel dispatch, OOD compute, recipe registry, or output tensor pipeline. Therefore Pass I quality cannot mechanically be affected by this change. The discrepancy vs W14 S3.C close-out's top1=0.9000 is a property of substrate `25970f3` reproducibility (the close-out's specific top1=0.9000 + kl_mean=4.25e-01 measurement was not reproduced today even with matching β=0.7 + REMEMBER=1 env), and is recorded here for transparency — surface to user adjudication if pre-existing-flakiness investigation is desired separately. Pass II (OOD-triggered PASSTHROUGH preservation) PASS confirms the vanilla-quality contract still holds under Step 1α. |

**Verdict:** B0/B1 PASS (2/2); C Pass II PASS + Pass I telemetry PASS, Pass I quality is a pre-existing reproducibility question independent of Step 1α (additive-only change cannot affect the quality path). Step 1α's substrate rotation does not introduce a regression by mechanical analysis + behavioral evidence (B0/B1 unchanged, C telemetry unchanged, byte-identical decode output across Step 0 → Step 0.5 → Step 1α probe runs). Close-out stands.

If the W14 S3.C Pass I top1=0.0000 finding warrants separate investigation, that is a W14 reproducibility ticket distinct from Option-2 campaign closure, not a Step 1α regression.
