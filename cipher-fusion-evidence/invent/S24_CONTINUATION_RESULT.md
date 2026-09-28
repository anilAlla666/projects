# S24 CONTINUATION — cell H resolved (EAGLE NEGATIVE off-distribution), 2:4 quality gate FAILS on
# the artifact (serving path is free), acceptance break-even quantified, FP8 detection seams scoped
# (2026-06-12)

Continues HEADROOM_2X_RESULT.md. The 2026-06-11 session died at 18:56 mid-cudagraph-capture of
s24_hv (cell H on varied prompts); this session re-ran it and closed the named next-gates #1
(quality validation) and #3 (detection-seam scoping). Protocol: H100, vLLM 0.20.2, cudagraph ON,
NREQ=32, `VLLM_PLUGINS=` empty (verified in every retained JSON: `plugins_env: ""`),
`VLLM_DEEP_GEMM_WARMUP=skip`, best-of-3 unless stated, T=0, forced 256 tok/req. Anchor 2edba0d2
verified at entry and exit. **Prereg discipline: PREREG_S24_CONTINUATION.md saved BEFORE the first
vLLM process of the session (P1-taint non-repeat); P6/P6b/P7a-c locked, decision rule locked.**
5-lens adversarial panel (artifact/stats/confounds/prereg/mechanism, each material finding
adversarially re-verified): 3 confirmed MATERIAL findings, ALL applied — (1) MMLU pairing
unsupported → arms re-run with per-question bits + real McNemar; (2) DOUBLE-BOS in all chat
measurements → single-BOS re-measurement (§2, reverses the on-distribution story upward);
(3) §4 capture-vs-trace-time conflation → re-grounded. Findings (1) and (2) were fixed by
RE-MEASUREMENT, not wording.

## 1. THROUGHPUT GRID — all cells on the IDENTICAL varied-prompt set (first clean in-harness
## triple-vs-bundle comparison; all prior bundle numbers were chat-harness)

| cell | config | tok/s | tok/W | mean W | note |
|---|---|---|---|---|---|
| Bv | dense Instruct bf16 | 4712.3 | 9.26 | 509 | ≈ degenerate-prompt B 4710.0 and chat dense 4699 |
| Dv | dense Instruct + fp8 | 6630.7 | 14.29 | 464 | ≈ degenerate D 6634.8 |
| Iv | Dv + EAGLE-3 (matched head) | **4921.3** | 9.15 | 538 | **0.742x of Dv — EAGLE NEGATIVE** |
| G2v | sparse 2of4 via torchao FP8+2:4 | 8005.9 (06-11) | 18.62 | 430 | 1.207x/Dv — reproduces degenerate-set 1.203x |
| Hv | G2v + EAGLE-3 (mismatched head) | **6413.3** | 12.14 | 528 | **0.801x of G2v — EAGLE NEGATIVE** |

- Non-spec cells are prompt-set-INSENSITIVE (Bv/Dv reproduce yesterday's B/D to <0.1%); the 2:4
  kernel gain is distribution-robust (1.203x degenerate / 1.207x varied over FP8-dense).
  **Spec cells are the only prompt-sensitive cells → the entire H/I collapse is EAGLE acceptance.**
- EAGLE exactness holds: Iv output text == Dv/Bv text; Hv text == G2v text, byte-identical 80-char
  prefixes. Spec cells burn MORE power (528–538 W vs 464–509 W) for FEWER tokens — wasted verify
  FLOPs, the saturation failure mode.
- Best config on this workload = G2v (no EAGLE): 1.699x tok/s / 2.011x tok/W over dense bf16.

**P6 MISS (predicted Hv/G2v 1.05–1.45; got 0.80). P6b MISS (predicted Iv/Dv 1.20–1.50; got 0.74).**
Shared cause, nailed by the acceptance probe below. Prediction error: I treated prompt *coherence*
as the axis; the real axis is drafter-training-DISTRIBUTION match.

## 2. ACCEPTANCE PROBE — the mechanism, quantified (same engine, two prompt sets; v1 Prometheus
## spec counters; single-rep, mechanism-only)

| config | prompts | accepted tok/step (k=5) | tok/s | vs no-EAGLE |
|---|---|---|---|---|
| dense FP8+EAGLE (matched) | raw continuation | **0.52** (2802 acc / 5385 drafts) | 4498.7 | 0.68x |
| dense FP8+EAGLE (matched) | chat-templated, DOUBLE-BOS (bug, see below) | **1.19** (4453/3732) | 6587.9 | 0.99x |
| dense FP8+EAGLE (matched) | chat-templated, SINGLE-BOS (corrected; best-of-3, TokensPrompt) | **2.37** | **10062.5** | **1.523x** (vs single-BOS FP8-only 6605.7) |
| (06-11 chat harness, retained) | chat, eagle_test2 set — DOUBLE-BOS, lower bound | implied ~1.8 (model below) | 8355 | 1.26x |
| sparse+EAGLE (mismatched) | raw continuation | 0.83 (3716/4497) | 6435.7 | 0.80x |

**DOUBLE-BOS BUG (panel catch, MATERIAL — lineage-wide):** every prior "chat" measurement
(06-11 eagle_test2 chat harness incl. the 8355/1.26x/1.43x bundle numbers, and this session's
first chat probe point) rendered `apply_chat_template(tokenize=False)` (emits BOS) and passed the
STRING to `llm.generate()`, whose text path re-adds BOS (`renderers/base.py:350`,
add_special_tokens=True) — the drafter never saw its training encoding. Corrected single-BOS
run (token ids via apply_chat_template(tokenize=True), assert exactly one BOS): acceptance
DOUBLES 1.19→2.37, EAGLE stage goes 0.99x→1.52x. A ONE-TOKEN prefix error halved acceptance —
acceptance is fragile even to tokenization drift, which sharpens the gating consequence below.
All retained double-BOS chat numbers are LOWER BOUNDS for properly-encoded chat traffic. Caveats:
sbos pair has no power sampling (no tok/W claim); EAGLE-on vs EAGLE-off outputs diverge at a
near-tie ~token 10 (both coherent; spec is target-greedy-lossless modulo FP8 numeric
nondeterminism across engine builds — unlike the byte-identical Iv/Dv raw pair).

**THREE probe points fit ONE saturation cost model: speedup ≈ (1+acc)/C with C = 2.24 (raw),
2.21 (double-BOS chat), 2.21 (single-BOS chat) — verify+draft costs ≈ 2.2x a plain step at
NREQ=32, k=5. Break-even acc = C−1 ≈ 1.2; the 06-11 double-BOS chat result back-solves to
acc ≈ 1.8 under its (buggy) encoding.** Acceptance ranged
0.52→2.37 across prompt styles AND encodings with the SAME matched drafter — a 4.6x swing that
flips the stage from −32% to +52%. Probe internals match the harness cells (Hv probe 6435.7 vs
harness 6413.3). Caveats: chat sets inherit ignore_eos=256-tok forcing (post-EOS text may inflate
chat acceptance); the double-BOS probe is 1 rep (Iv probe 4498.7 vs harness best-of-3 4921.3 —
spec-run rep variance is real, direction unaffected); the single-BOS pair is best-of-3.

**Thesis refinement (first-class):** per-model co-design is NOT sufficient for the spec stage —
it is per-model AND per-WORKLOAD — where "workload" includes the exact ENCODING (one stray BOS
halved acceptance). The 06-11 "trained drafter HOLDS under saturation" finding
(PER_MODEL_BUNDLE_RESULT.md) is now scoped AND strengthened: properly encoded on-distribution it
delivers +52% at saturation (better than every retained number, which were double-BOS-depressed);
off-distribution it inverts to −26/−32%. Product consequence: the bundle's spec stage must ship
with acceptance-aware gating on MEASURED accepted/step vs the ~1.2 break-even (vLLM exposes the
counters) — label-based gating ("this is chat traffic") is insufficient because acceptance is
fragile even to tokenization drift; a customer with continuation-style traffic (or a template bug)
gets a −26–32% regression from the same "validated" bundle.

## 3. QUALITY GATE (prereg P7) — the SERVING PATH is free; the ARTIFACT fails PPL; MMLU [MMLU]

WikiText-2-raw test, teacher-forced prompt_logprobs, identical 32x1024-token chunks (32,736 scored
tokens), Llama-3.1 tokenizer for all arms, as-served engine config (cudagraph ON):

| arm | model / route | PPL | ratio |
|---|---|---|---|
| Qc | dense Llama-3.1-8B base bf16 | 7.1124 | 1.000 |
| Qd | dense base + fp8 on-the-fly | 7.1875 | 1.0106 /Qc — P7c CONFIRMED (≤1.02) |
| Qa | Sparse-Llama-3.1-8B-2of4 bf16 (dense kernels) | 8.875 | **1.2478 /Qc — P7b MISS** (predicted 1.02–1.12) |
| Qb | sparse via torchao FP8+2:4 (as-served G2 route) | 8.958 | **1.0094 /Qa — P7a CONFIRMED** (1.00–1.03); 1.2595 /Qc |

- **P7a (THE gate on the route): CONFIRMED.** FP8-PerRow+SPARSE_CUTLASS serving of the pruned
  checkpoint costs +0.94% PPL — same as fp8-on-dense (+1.06%). The G2 route adds ~nothing.
- **P7b: MISS.** The pruned ARTIFACT itself carries +24.8% wikitext PPL vs the dense base it was
  pruned from. NM's "~98.9% recovery" is an Open-LLM-task claim, not a PPL claim; my prediction
  wrongly transferred it to PPL.
- **LOCKED DECISION RULE: FAILS** (Qb/Qa = 1.009 ≤ 1.05 ✓ but Qb/Qc = 1.26 > 1.15 ✗). By prereg,
  **the 2:4 stage does NOT join the bundle on this artifact.** No re-scoping: the rule bound the
  artifact's total quality, and the artifact misses it.
- **MMLU adjudication (ADDITIONAL post-prereg evidence, labeled as such; 500 5-shot test Qs, seed
  0, letter-logprob argmax, identical paired questions all arms, 0 dropped/0 unparseable):**
  canonical run (per-question bits retained, mmlu_*.json `bits`): dense base **66.8%**, sparse
  bf16 **61.8%**, sparse G2 route **61.2%**. (A first run without bits retention gave
  67.0/61.6/61.2 — consistent; panel caught that "paired, outside noise" was then unsupportable,
  so the arms were re-run WITH pairing.) **Real McNemar on the retained bits: dense-vs-sparse
  n01=59/n10=34, z=2.59, p=0.0095 — the artifact drop IS outside noise, paired; sparse-vs-G2
  n01=14/n10=11, z=0.60, p=0.55 — the route cost IS noise.** MMLU corroborates the PPL verdict
  instead of rescuing it: ~92.5% retention vs dense. The NM-claimed ~98.9% recovery does not
  reproduce on this subset/protocol (unpaired bound vs 0.989x-dense reference: z≈1.6–2.2,
  borderline — claim-level, not artifact-level, disagreement).

**Net:** the 2:4 throughput lever (1.20x) and its serving route are validated and quality-free,
but productizing requires a BETTER 2:4 ARTIFACT (per-model pipeline work: prune+recover to PPL
parity — the paired-significant −5 pt MMLU drop (p=0.0095) rules out "task-parity-only"
acceptance too). The "+2:4 joins the bundle" claim stays GATED on artifact quality, with the route
itself fully validated.

## 4. DETECTION SEAM SCOPING (gate #3) — two choke points, one plugin covers both

Read-only source scoping (vLLM 0.20.2 + torchao, local site-packages; agent-mapped, file:line
verified):
- **Seam 1 (vLLM-native FP8):** `vllm/_custom_ops.py:875–923 cutlass_scaled_mm` — the single
  Python choke point for every CutlassFP8ScaledMMLinearKernel linear (q/k/v/o, gate/up/down; the
  sm90 default). Sees a=[M,K], b=[K,N], scales, out=[M,N].
- **Seam 2 (torchao 2:4 route):** `vllm/model_executor/layers/quantization/torchao.py:328–334
  TorchAOLinearMethod.apply` → `F.linear(x, Sparse2x4CUTLASSFloat8Tensor)` → tensor-subclass
  dispatch (NOT vllm._C) — separate seam required; G2/Hv confirmed empirically non-cuBLAS.
- **No single seam covers both**, but ONE `vllm.general_plugins` entry-point plugin can wrap both
  at import time (plugins load in engine+worker procs before first forward —
  `v1/engine/core.py:103`, `v1/worker/worker_base.py:237`). No vLLM fork.
- **Capture-safety (panel-corrected):** these linears live inside torch.compile-compiled regions,
  so the wrapper's ops are baked in at TRACE time (dynamo/inductor), not at cudagraph capture —
  the plugin patch precedes compilation (plugins load at engine/worker init, before first forward),
  which is what makes it land. Two added requirements: detector ops must be fully dynamo/inductor-
  TRACEABLE (no graph breaks / host branches), and the plugin must participate in (or invalidate)
  the torch.compile cache key, else a cached compiled artifact silently bypasses the detector.
  Host-side wrapper logic does NOT re-execute per replayed step — detector state must live in GPU
  buffers read out-of-band. This is ANALOGOUS in effect to coop-hook Finding 1 (insertion before
  freeze survives; post-hoc mutation rejected), not the identical mechanism — a one-cell smoke
  test on the compiled FP8 path is the required next probe before costing (−3.9% @N=45 periodic
  dual-graph design expected to carry over).
- **Cost expectation:** always-on inline recompute ≈ −50% (coop-hook measured); periodic dual-graph
  ≈ −3.9%. The build is "fork-1 verify machinery re-pointed at two new seams", not a new detector.

## 5. PREREG SCOREBOARD

P6 MISS / P6b MISS (same cause, mechanism nailed: acceptance 0.52–0.83 < ~1.2 break-even) /
P7a CONFIRMED / P7b MISS / P7c CONFIRMED. Decision rule: FAIL → 2:4 stage stays out of the bundle
pending a quality-parity artifact. Misses reported as misses; no re-scoping.

## VERDICT
The interrupted cell H is resolved and it REVERSES the compose hope: **FP8+2:4+EAGLE is NOT
additive — EAGLE is workload-fragile (−20–32% off-distribution at saturation, +52% properly-
encoded on-distribution; break-even ≈1.2 accepted tok/step; cost model (1+acc)/2.2, 3 points)
while FP8 and 2:4 are workload-robust. Best measured config on continuation traffic = FP8+2:4
WITHOUT EAGLE: 1.70x tok/s / 2.01x tok/W over dense bf16. On chat traffic, corrected single-BOS
FP8+EAGLE = 10062.5 tok/s ≈ 2.14x dense bf16 (4710, shown prompt-insensitive across 3 sets but
not run on this exact set) — the founding 2x-throughput-at-saturation goal is MET on-distribution,
no 2:4 needed, and the 06-11 1.78x headline was a double-BOS-depressed lower bound.** The 2:4 serving route is quality-free (+0.9% PPL) but the available artifact fails
the locked quality gate (+25% PPL vs dense base; MMLU corroborates: −5 pt, McNemar p=0.0095) —
per-model pipeline work (prune+recover to parity) is the unlock, exactly the pivot's economic
premise. Detection on
FP8/torchao paths is buildable without forking vLLM via two scoped seams + the proven coop-hook
capture mechanics. Bundle composition is now: FP8 (always) ⊕ 2:4 (artifact-gated) ⊕ EAGLE
(workload-gated, needs acceptance-aware gating) ⊕ DVFS (cap-level) — per-model AND per-workload.
Anil's call.
