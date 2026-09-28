## Section 4 — Goal 4 (O(1) Koopman substitution) audit

**Scope.** Goal 4 = an O(1), Koopman-operator-predicted substitute that replaces
expensive kernel/attention compute on the hot path. This audit traces the
substitution path against code reality. Source root:
`/home/ubuntu/cipher-may13-evidence/src/` and `/home/ubuntu/cipher_rt_phase4/`.

All claims below carry `file:line` evidence. Where no evidence exists the
finding is recorded as "not found" / "zero", not inferred.

---

### (a) The hot-path TODO / fallback-to-passthrough

There is no single TODO — there are **four disconnected substitution attempts**,
each with its own gate that falls through to passthrough. Ranked by load-bearing
position on the hot path:

**1. THE hot-path gate — `cipher_dispatch.cpp:570-574`** (every kernel launch
passes through `cipher_dispatch`):

```
570    if (!entry) {
571        // No surrogate yet. If this op appears frequently, queue for EDMD.
572        // (L3.5 EDMD pipeline — wired in Week 4-5)
573        return CIPHER_PASS_THROUGH;
574    }
```

The registry lookup (`cipher_registry_lookup`, line 567) returns no entry for any
shape the workloads actually produce, and the "L3.5 EDMD pipeline — wired in
Week 4-5" comment is the deferral: there is no code path that dynamically
populates the registry from collected EDMD surrogates. The registry is only ever
seeded with 32 fixed, hardcoded entries at init (`cipher_recipes.cpp:346-405+`),
all at Llama-3-70B / HyperFlux scale (e.g. `gemm-4096x4096x4096`,
`llama3-ffn-up-4096x28672x8192`). No stress workload (Llama-3.2-1B, Mistral-7B,
TinyLlama) emits those exact shapes, so `apply_recipe()` is never reached.

**2. Attention Koopman substitute TODO — `cipher_attn_koopman.cpp:495-499`:**

```
494            if (is_atv_gemm(M, N, K) && s->fused_ready) {
495                // Would run fused kernel here (TODO: wire up once the
496                // per-shape (V_T, K_op, V_compressed) registry is populated
497                // by a prefill-end hook). For now, fall through to revert
498                // — defaulting to CORRECT output on any uncertainty.
499                revert_and_reset_locked(s, "fused_path_not_yet_wired");
```

The attention FSM reaches the `@V_cache` step and then reverts instead of
substituting; the fused-kernel path is explicitly "not yet wired".

**3. Attention REDIRECT not implemented — `cipher_rt_attn_dispatch.cpp:158-164`:**

```
157        case CIPHER_RT_ATTN_REDIRECTED:
158            // T4.6.3+ — substitution mechanism deferred. For now,
159            // treat as PASSTHROUGH but count separately so we know
160            // an actuator wanted to redirect.
161            g_redirected.fetch_add(1, std::memory_order_relaxed);
162            fprintf(stderr, "[cipher-attn] actuator '%s' returned REDIRECTED "
163                    "but substitution path not implemented in T4.6.1 — "
164                    "passing through\n", ...);
```

(This is Q/K/V pointer redirection, a different lever from O(1) compute, but it
is also a deferred substitution path.)

**4. Flow-substitute SHIPS AS A STUB — `cipher_flow_substitute.cpp:490-492`** (the
only Koopman-adjacent path actually invoked from the hot path, at
`cipher_intercept_cudart.cpp:2630`):

```
489            fprintf(stderr,
490                "[CIPHER FLOW SUBST] STUB count #%llu  recipe=%d  "
491                "(X/W/Y not yet extracted from struct args — passthrough)\n",
```

The real-substitution branch (lines 472-485) gates on
`t_win.x_ptr != 0 && t_win.w_ptr != 0 && t_win.y_ptr != 0 && t_win.hidden > 0`.
`x_ptr / w_ptr / y_ptr / hidden` are **declared at lines 66-70 and never assigned
anywhere in the file** — so `can_real` (line 472) is structurally always false
and the path always falls through to the STUB counter at line 486 (`return 0` =
no suppression, real kernel runs).

---

### (b) What exists vs. what is missing

| Component | Status | Evidence |
|---|---|---|
| Koopman/EDMD math (operator solve `K = ΨY·ΨX†`, predict) | EXISTS | `cipher_edmd.cpp`, `cipher_edmd_live.cpp`; ADAPT solves with `fit_error=0.0002` (per_op_v2.log) |
| KEN — Koopman Eigenfunction Network (encode/evolve/decode/predict) | EXISTS | `cipher_10ops_impl.cpp:99-225` (`ken_*`); inits at runtime |
| Runtime Koopman derivation module (state machine COLLECTING→DERIVED, feature extraction, `cipher_kr_decide`/`_predict`) | EXISTS but **DEAD CODE** | `cipher_koopman_runtime.cpp` — zero callers; `cipher_kr_decide`/`_predict`/`_init`/`_record_output`/`_find_or_create` are not referenced anywhere outside the file itself, incl. rt_phase4 |
| Hot-path dispatcher (`cipher_dispatch`) with CLASSIFY → ORACLE → registry → SUBSTITUTE | EXISTS, wired | `cipher_dispatch.cpp:402-590`; called from `cipher_intercept*.cpp` |
| O(1) substitute kernels (block-level `O(Kr)`, cached-output `O(1)`) | EXISTS | `cipher_dispatch.cpp:258-333`, `cipher_block_sub_kernel.cu`, `cipher_attn_koopman_kernel.cu` |
| Dynamic registry population from collected surrogates | **MISSING** | No `cipher_registry_add/insert/register` anywhere; registry only seeded with 32 static entries (`cipher_recipes.cpp:346`); `cipher_dispatch.cpp:571-572` "wired in Week 4-5" |
| Wiring of `cipher_koopman_runtime` to the hot path | **MISSING** | no caller of `cipher_kr_*` |
| Flow-substitute X/W/Y argument extraction from struct args | **MISSING** | `cipher_flow_substitute.cpp:66-70` declared, never assigned; `:490-492` admits "not yet extracted" |
| Attention fused-kernel substitution wiring | **MISSING** | `cipher_attn_koopman.cpp:495-499` "fused_path_not_yet_wired" |
| Attention REDIRECT (Q/K/V) substitution mechanism | **MISSING** | `cipher_rt_attn_dispatch.cpp:158-164` "not implemented in T4.6.1" |

**Net:** the Koopman/EDMD solver and the O(1) substitute kernels both exist. What
is missing is the *connective tissue* — nothing populates the dispatch registry
with a derived surrogate, the standalone runtime-Koopman module is never called,
and the only hot-path-wired Koopman-adjacent module (flow_substitute) ships as a
declared stub. The substitution branch is bypassed at the registry gate before
any Koopman predict ever runs.

---

### (c) Substitution-firing finding

**Confirmed: zero O(1) Koopman substitutions fire across the stress tests.**

The word "substitution" appears in stress reports for three unrelated systems —
they must not be conflated:

- **Goal-4 Koopman O(1) compute substitution — ZERO firings.**
  Searched `stress/` and `stress2/` logs (incl. the 5.1 MB `t4_cipher.log`,
  5.0 MB `e8_cipher.log`, 21 MB `per_op_v2.log`) for the substitution-success
  prints `[CIPHER L3.2] SUBSTITUTE`, `[O(1)-block]`, `[CIPHER L1.3] Registry HIT`,
  and `[CIPHER FLOW SUBST] ... real_substituted` — **all return 0 matches**.
  `stress2/per_op_validation.json` reports the `KOOPMAN` op as
  `status: SILENT, evidence: "saw_qk=0 saw_softmax=0"`.
  `stress2/OPS_VALIDATION_REPORT.md:16` confirms KOOPMAN never enters its
  calibration window.
  `stress2/per_op_validation.log:108`:
  `[CIPHER F1] Teardown. Intercepts: 0 | Substitutions: 0 (0.0%)`.

- **ADAPT Koopman solver — runs, but produces no substitutions.**
  `stress2/per_op_v2.log` contains **84,611** `→ Koopman update applied` lines
  (`op_class=0 fit_error=0.0002`). Per `cipher_10ops_impl.cpp:678-709` this only
  feeds `cipher_lnn_koopman_update` — it nudges a *shadow LNN's* weights. It
  emits zero `L3.2 SUBSTITUTE` / `O(1)-block` / `Registry HIT` events in the same
  log. The EDMD pipeline it solves (`s_adapt_edmd[0]`) IS the one
  `cipher_get_edmd_pipeline(0)` returns, so `koopman_converged` *could* be true —
  but `apply_recipe` is never reached because the registry gate
  (`cipher_dispatch.cpp:570`) fails first.

- **FP8/Marlin "100% substitution rate" (`STRESS_REPORT_SECTIONS...md:75-79`,
  D3) — a different, non-Koopman system** (`cipher_fp8_compute.cpp` /
  `cipher_marlin_src.cpp`). Not evidence for Goal 4.

---

### (d) Path-to-close verdict

**Verdict: architectural (multi-week).**

Justification, grounded only in the code:

- This is **not** "the matrix solve is done, only the dispatch-time lookup is a
  TODO" (which would be ~1 week). The matrix solve *is* done — but there are
  **four independent, mutually-disconnected substitution substrates**
  (`cipher_koopman_runtime.cpp`, `cipher_dispatch.cpp`+registry,
  `cipher_attn_koopman.cpp` FSM, `cipher_flow_substitute.cpp`, plus the
  `rt_attn_dispatch` REDIRECT path), each with its own gate and its own TODO.
  No single one of them, completed, closes Goal 4.

- The cleanest single piece of evidence: `cipher_koopman_runtime.cpp` — the
  module literally named "Runtime Koopman Derivation" — is **fully built and
  has zero callers**. A complete state machine + feature extractor + predict
  path that nothing invokes. Closing Goal 4 means deciding whether to wire it
  in or delete it in favour of one of the other three.

- The remaining gaps are integration-architecture, not single TODOs:
  (i) build dynamic registry population (the registry is hardcoded to 32 static
  70B-scale shapes; `cipher_dispatch.cpp:571-572` defers this);
  (ii) wire flow_substitute's X/W/Y extraction from CUDA struct args
  (`cipher_flow_substitute.cpp:66-70` are unassigned — and CLAUDE.md's own
  Stage-8b notes show struct-arg layout reverse-engineering is fragile,
  multi-session work);
  (iii) wire the attention fused-kernel registry + prefill-end hook
  (`cipher_attn_koopman.cpp:495-499`);
  (iv) pick one substrate and delete/merge the other three to avoid four
  half-built paths.

Estimate: **multi-week.** Not because any one algorithm is missing, but because
the work is choosing a single substitution substrate, building the
surrogate→registry population pipeline behind it, wiring hot-path argument
extraction, and retiring the three competing stubs. A 1-day or 1-week framing
would be unsupported by the code.
