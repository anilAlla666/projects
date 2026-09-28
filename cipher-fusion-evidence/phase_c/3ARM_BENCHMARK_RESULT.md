# 3-Arm Cross-Process Benchmark — Result (STOP / ADJUDICATE)

**Date:** 2026-05-18. Workload WL01 (TinyLlama-1.1B, B=1/tenant, greedy,
128 tok, 5 prompts replicated). N=8, single H100 SXM5. Anchor `a7ac8e97`.

**Verdict: BOTH stop conditions triggered — this is NOT a Peak-XV-ready
result.** Arm 2 (vLLM) is 2.18× Arm 1 (not the predicted ≈1×); Arm 3/Arm 2 =
**1.75× < 2.5×**. Per the brief's instructions, stop and adjudicate before
drawing conclusions.

## Results — 3 arms, per-rep

| arm | rep 1 | rep 2 | rep 3 | **mean tok/W** | tok/s | power |
|---|---|---|---|---|---|---|
| **Arm 1 — naive 8-concurrent (HF `generate()`)** | 0.863 | 0.887 | 0.881 | **0.877** [0.863–0.887] | ~180 | ~205 W |
| **Arm 2 — vLLM intra-process, 8 instances (B=1 each)** | 1.915 | 1.911 | *failed* | **1.913** [1.911–1.915] | ~529 | ~277 W |
| **Arm 3 — CIPHER cross-process fusion (B=8 fused)** | 3.341 | 3.379 | 3.325 | **3.348** [3.325–3.379] | ~492 | ~147 W |

tok/W = aggregate useful tokens ÷ aggregate watts ÷ wall, decode window only.
Arm 2 rep 3 failed to start twice — 8 concurrent vLLM instances lose the
startup memory race intermittently (operational fragility; see §Notes). The
two successful Arm-2 reps agree to 0.2%.

## Correctness gates — all PASS

- **Arm 1** — teacher-forced logit-KL (pillar_driver gate), vanilla FP16,
  KL ≈ 5.5e-5. PASS.
- **Arm 2** — vLLM output vs FP16 gold: **100.0000% exact token match**
  (4224 tokens/rep). vanilla FP16 → identical to gold. PASS.
- **Arm 3** — teacher-forced logit-KL, KL_max ≈ 3.1e-5. PASS.

## Substrate-attributable lifts

| ratio | value | meaning |
|---|---|---|
| **Arm 2 / Arm 1** | **2.18×** | vLLM intra-process benefit — *predicted ≈1.0×* |
| **Arm 3 / Arm 1** | **3.82×** | CIPHER vs naive (the Phase-B substrate-attributable) |
| **Arm 3 / Arm 2** | **1.75×** | CIPHER vs vLLM — *the intended Peak-XV number* |

## Stop condition 1 — Arm 2 ≫ Arm 1: TRIGGERED

The prediction (methodology memo, `FUTURE_SCOPE/C`) was **Arm 2 ≈ Arm 1** —
vLLM cannot batch across separate processes, so 8 vLLM B=1 instances ≈ 8 naive
HF instances. **The prediction is falsified: Arm 2 = 2.18× Arm 1.**

The cause is *not* cross-process batching (vLLM has none across the 8
instances — verified: 100% exact-match to gold, and 8 separate engine
processes cannot share a batch). The cause is that **vLLM is a far more
efficient single-stream engine than naive HF `generate()`** — paged
attention, optimised kernels, an efficient runtime. Even at B=1 single-stream,
vLLM decodes ~2.2× more efficiently per watt. The prediction conflated "vLLM
can't fuse across processes" (true) with "so vLLM ≈ naive" (false — the
engine itself is just better).

## Stop condition 2 — Arm 3 / Arm 2 < 2.5×: TRIGGERED

Arm 3 / Arm 2 = **1.75×**, below the 2.5× bar. The cross-process-fusion
advantage **measured against a real engine** is 1.75×, not the 3.82× headline
(which was against naive HF). Most of the 3.82× is the engine-quality gap that
vLLM also has; only 1.75× survives a fair baseline.

## Architectural interpretation — and a confound that must be adjudicated

The honest decomposition: `Arm3/Arm1 (3.82) ≈ Arm2/Arm1 (2.18) × Arm3/Arm2
(1.75)`. The 3.82× over naive splits into a ~2.2× "any decent engine beats
naive HF" factor and a ~1.75× residual.

**But the 1.75× is confounded — and against CIPHER.** Arm 3 is CIPHER's
cross-process fusion on an **HF-`generate()`-based executor**. It is *not* the
methodology memo's intended Arm 3 = "CIPHER substrate **over vLLM**." So 1.75×
compares (HF engine + CIPHER cross-process batching) against (vLLM engine, no
batching) — it mixes a batching *gain* with an engine *deficit*. The clean
architectural measurement — CIPHER's cross-process fusion composed onto
**vLLM's** engine — requires `FUTURE_SCOPE/A` (CIPHER + vLLM composed
architecture), which is **not built**. Whether CIPHER-over-vLLM clears 2.5× is
an **open question**; it is not pre-judged here.

A second confound, also against a clean reading: Arm 2 was measured with
`enforce_eager=True` (8 concurrent vLLM instances would not fit with CUDA
graphs — memory). vLLM with graphs is faster → Arm 2 ≥ 1.913 → Arm 3/Arm 2
**≤ 1.75×**. The architectural distinction *today* is at most 1.75×.

**Mechanism of CIPHER's win — it is a power win, not a throughput win.**
Arm 2 (vLLM-8) actually has *higher* aggregate throughput than Arm 3 (529 vs
492 tok/s) — but draws **277 W vs CIPHER's 147 W**. CIPHER wins tok/W by
consolidating 8 tenants into one lower-power process; 8 separate vLLM engine
processes burn ~1.9× the power. So CIPHER's cross-process fusion benefit at
this operating point is **power-efficiency from process consolidation**, not
raw speed.

## Does the result match the cross-process-vs-intra-process prediction?

Partly. The architectural claim — vLLM cannot fuse across separate tenant
processes — **holds** (Arm 2's 8 instances did not batch across each other;
the throughput is 8× single-stream, not a fused batch). But the *magnitude*
prediction (Arm 2 ≈ Arm 1, so Arm 3/Arm 2 ≈ Arm 3/Arm 1 ≈ 3.7×) is **wrong**:
vLLM's engine quality lifts Arm 2 to 2.18× Arm 1, compressing the
fusion-attributable advantage to ≤1.75×.

## Notes

- 8 concurrent vLLM instances on one 80 GB H100 are operationally fragile —
  rep 3 failed twice on the startup memory race (`gpu_memory_utilization=0.10`,
  8-way). This fragility *itself* supports the consolidation argument: running
  one engine per tenant does not scale cleanly to many tenants on one GPU.
- Power profile (decode window): Arm 1 ~205 W, Arm 2 ~277 W, Arm 3 ~147 W —
  stable within each window (±few W; windows 8–13 s, the static-barrier
  scenario).

## Recommendation — adjudicate

This is **not** a verified Peak-XV result. Two paths:

1. **Build `FUTURE_SCOPE/A` (CIPHER + vLLM composed) first, then re-run.**
   The clean Arm 3 is CIPHER's cross-process fusion *on vLLM's engine*. Only
   then does Arm 3/Arm 2 isolate the cross-process-fusion factor (batching on
   the same engine) rather than confounding it with an HF-vs-vLLM engine gap.
   This is the methodology memo's intended Arm 3.
2. **Accept 1.75× (≤, today) as the cross-process-fusion advantage over a real
   engine** and decide whether that — a power-efficiency win from process
   consolidation — is a sufficient external claim. It is below the 2.5× the
   brief set as the adjudication threshold.

Recommended: path 1. The 1.75× is real but confounded; the honest Peak-XV
number needs CIPHER measured *on* vLLM, not *instead of* it. No external claim
until then.

## Artefacts

`phase_c/arm1_rep{1,2,3}/`, `phase_c/arm2_rep{1,2}/`,
`phase_b/session1/arm3_rep{1,2,3}/`; harnesses `run_arm2.sh`/`vllm_tenant.py`,
`run_multitenant.sh` (Arm 1), `run_arm3.sh`/`cipher_batch_executor_gen.py`.
