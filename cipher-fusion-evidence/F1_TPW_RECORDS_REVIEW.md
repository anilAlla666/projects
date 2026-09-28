# Records review — the 3.617× tok/W measurement vs finding F1

**Date:** 2026-05-18. **Type:** paperwork, no GPU. **Purpose:** before the
CP 5.6 scope memo, determine whether the anchored 3.617× tok/W headline
engaged the Marlin full-GPU path that [[cipher-f1-fullgpu-marlin-broken]]
(finding F1) showed produces degenerate decode — i.e. whether the F1 audit
must carry re-measurement scope.

**Verdict: SUSPECT.** The measurement engaged the path, with the F1
degeneracy precondition present, and carried **no output-correctness gate**.
The number is a valid *throughput timing*, but unverified as efficiency on
*useful* tokens. It needs re-measurement (on `dc804eb3`, with a correctness
gate) before re-assertion.

---

## Source of the number

CP 2.4 composed gate (`cp_2_4/CP_2_4_REPORT.md` §4, md5-anchored report):
**all-on / vanilla = 3.617× tok/W [3.591, 3.642]**, 1.795× tok/s, n=5.
Re-confirmed CP 4.8 Task B (`cp_4_8/taskB_cp24_remeasure/TASK_B_RESULT.md`):
exact reproduction, 3.6166×.

## The four questions

**1. Substrate version.**
- Original CP 2.4 (2026-05-16): libcipher_rt **`5e304549`** — the "CP 2.4
  Marlin Fix A build" (CP_2_4_REPORT §8). Fix A = the `PrimaryCtxGuard` that
  pins the Marlin GEMM to the primary context — the exact construct F1's
  leading hypothesis implicates.
- CP 4.8 Task B re-measurement (2026-05-17): libcipher_rt **`c2c5d313`** —
  **the exact substrate F1 proved degenerate.** It reproduced 3.6166×.

  → The headline has already been re-run on the F1-broken substrate and
  reproduced — because the harness measures only timing (see Q4). Task B's
  "the investor one-pager number stands" conclusion is itself now suspect.

**2. Workload.** Mistral-7B-v0.1, **B=1**, **greedy** decode
(`spec_varied_driver.py:76` — `do_sample=False`), a fixed 5-prompt varied set
(factual / narrative / code / reasoning / conversational), 128-token budget.
Prompts were *deliberately chosen so decode never hits EOS early*
(`spec_varied_driver.py:34`) — which also means a degenerate non-terminating
loop and a coherent long generation are indistinguishable in the recorded
metrics.

**3. Methodology — n=5 matched pairs.** 5 pairs; each pair runs **three arms**
(`run_composed.sh`):
- `vanilla` — stock HF FP16, **no CIPHER at all** (clean baseline);
- `marlin` — `CIPHER_MARLIN=on`, substrate active;
- `all-on` — Marlin + n-gram speculative decode + DVFS clock-lock @ 1000 MHz.
Per (pair, arm): per-prompt tok/s JSON + sentinel-windowed `watts.csv`. Lift
ratios are formed per pair and averaged over the 5. The 3.617× is the
`all-on / vanilla` tok/W ratio. The pairs compare **substrate-on throughput
vs substrate-off throughput** — never output against output.

**4. Coherent output, or pure throughput timing?** **Pure throughput timing —
zero correctness verification.** `spec_varied_driver.py` records only `tok_s`,
`gen_tokens`, `accept_rate` (lines 113–118); it never dumps token IDs or text
and never compares to a reference. The composed JSONs (`composed/p*_*.json`)
and logs (`composed/p*_*.log`) confirm this — `RESULT` lines and nothing else.
There is **no** token-correctness gate anywhere in the composed gate. (CP 2.4
§3.1's token-agreement gate covers only the spec-decode *verify loop*, not the
generated output of the composed measurement.)

## Did it engage the F1-degenerate path? — Yes, affirmatively

From the composed-gate logs themselves (`composed/p1_marlin.log`,
`p1_allon.log`):
- **Marlin full-GPU engaged.** `MARLIN: actuator ENABLED`,
  `MATMUL: exit totals — calls=288000 handled=287325` — Marlin handled
  essentially every matmul, grid=132 full-GPU primary-pinned (Fix A).
- **A green context was active** — `GREEN: green context bound to group 3
  with 8 SMs` — F1's degeneracy precondition (green ctx present + Marlin
  primary-pinned).
- **The context/stream accounting was in the confused F1 state** —
  `DIAG-T4.2.4d: on_our_green=0 on_other_green=0 on_primary=0
  on_null_stream=3018900`, with 2.73M `ctx_swaps_to_green`. Kernels executed
  on the null stream while the green ctx was bound and being swapped to.
- **Positive evidence the output was degenerate:** the all-on arm's n-gram
  `accept_rate = 1.000` on every prompt — CP 2.4 §3.2's own words: *"a
  loop-replay signature, not draft quality… base-model output collapses into
  exact repetition."* A 3-gram draft cannot hit 100% acceptance on coherent
  text; 1.000 means the output was an exact loop.
- **Corroborating asymmetry:** vanilla prompt 3 stopped at 71 tokens (natural
  EOS); the `marlin` arm generated the full 128 on the same prompt — the
  substrate arm's decode diverged from the clean baseline and did not reach
  the baseline's stopping point.

CP 2.4 §3.2 attributed the looping to a *base-model greedy property* ("on
every prompt genre"). That attribution is **unverified and now doubtful**:
F1's gold run — stock FP16 Mistral-7B, the same model, greedy — produced
**coherent, non-looping** 128-token output. Plain greedy on this model does
not inherently collapse into repetition; the CP 2.4 looping is at least as
consistent with substrate-induced degeneracy as with base-model behaviour.

## Verdict and scope implication

**SUSPECT** (not AMBIGUOUS — the records affirmatively show the path engaged,
the green-ctx precondition present, the 1.000 loop signature, and no
correctness gate; this is positive evidence, not absence of evidence).

What is and isn't invalidated:
- **Suspect:** the `all-on` and `marlin` arm *throughput* as a measure of
  useful work — hence the composed **3.617× tok/W and 1.795× tok/s** as
  efficiency-on-real-output claims. The number is a real timing of an
  unverified — probably degenerate — decode.
- **Not invalidated:** the `vanilla` baseline (no substrate, clean); the DVFS
  power component (212 W → 105 W is a power fact, independent of output
  correctness — though "tok/W" still inherits the suspect numerator).

**The F1 audit must include re-measurement scope.** Recommended for the
CP 5.6 / F1-audit memo:
1. Re-run the composed gate on `dc804eb3` (STEP 2 build). Note `dc804eb3`
   fixes only the **partitioned** path; the **full-GPU** path is still
   F1-broken — the re-measurement must pin which path the gate exercises and
   ensure it is a coherent one.
2. Add a **token-correctness gate** to `spec_varied_driver.py` — dump token
   IDs, compare against a no-substrate FP16 gold (logit-KL / perplexity, not
   strict identity — INT4 legitimately diverges; see CP 5.3 STEP 2 §4).
3. Until then, the 3.617× headline should not be re-asserted in the investor
   one-pager / Ditlev brief without the "throughput-only, correctness
   unverified" caveat.
