# CP 4.8 — integration soak / Phase 4 close — DESIGN MEMO

**Date:** 2026-05-17. **Status:** pre-build memo — **no soak, no gate
measurement until the memo is adjudicated.** This is the Phase 4 ship-gate CP.
Per Phase-4+ engineering-marvel discipline the memo opens with mechanism +
per-WL roofline bounds; the gate then measures *against pre-registered
physics-derived ceilings*, not arbitrary thresholds.

**Structure.** §0 scope · §1 mechanism + two physics corrections · §2
pre-registered 21-WL ceiling table · §3 analytical-vs-PMU cross-check · §4
pre-registered G1–G4 pass criteria · §5 24 h soak protocol · §6 honest
caveats · §7 build STEP · §8 calendar. **Complete — held for adjudication.**

**Anchors:** kmod 0.4.8 `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2
`86618c30` (not loaded by the soak runtime — D6). The CP 4.8 soak validates
the composed production stack as it ships; **no substrate code changes** —
the only build artifact is the `tensor_mfu_pct` measurement-path refinement
(`mfu_compute.py`, build-STEP item 2).

---

## 0. Scope adjudication carried in (user, 2026-05-17)

- **D1** — gate metric is `tensor_mfu_pct`, evaluated as **≥80% of a
  pre-registered per-WL roofline ceiling**. Method: analytical FLOP-roofline
  (Option A) — no DCGM, no perturbing CUPTI profiling; soak-safe.
- **D2/D3** — fetch 5 reference models (done: MiniLM, SDXL, Whisper-large-v3,
  CLIP ViT-L, LLaVA-1.5-7B). Defer WL15 (Mixtral, 90 GB / multi-GPU) and
  WL24 (AWQ — Phase-5 partition-aware-Marlin scope).
- **D4** — 24-hour soak per the G4 spec.
- **D5** — fold in B5 (defines the gate); B2 closes with a benign-at-cadence
  note; B8/B9/B13 codified as soak discipline.
- **D6** — RESOLVED (`D6_LIBCIPHER_V2_ANCHOR_RESOLUTION.md`): soak anchors on
  `c2c5d313` + `e2f50452`; `libcipher_v2.so` is not in the soak runtime.

**New finding from driver audit — WL18 is a third deferral.** `wl18_multi_gpu_tp.py`
itself reports *"deferred-multi-gpu (pod has <2 GPUs)… multi-GPU path not
implemented."* Tensor-parallel on a single-GPU pod degenerates to TP=1, which
is just WL01 — not a TP test. **CP 4.8 honestly covers 21 of 24 WLs**; three
deferred, each with a physical reason: WL15 (Mixtral multi-GPU scale), WL18
(multi-GPU TP, single-GPU pod), WL24 (AWQ INT4 — Phase 5). Flagged for
adjudication; the memo proceeds on 21.

---

## 1. Mechanism — the tensor-MFU roofline, and two physics corrections

### 1.1 What `tensor_mfu_pct` measures

`tensor_mfu_pct = workload_tensor_FLOPs / (wallclock × peak_tensor_FLOPS)` —
the fraction of the H100's peak tensor-core throughput the workload actually
realises. It replaces the legacy `mfu_compute.py` SM-cycle proxy
(`sm_util_pct × clock_ratio`), which counts an SM *stalled on memory* as
"utilised" and therefore reports ~100% for memory-bound decode where the
tensor cores are ~99% idle (B5 — WL14 hit 100% under the proxy). The new
metric is **measured analytically** (FLOP count from model geometry × tokens;
wallclock from the harness) — zero added overhead, soak-safe, and
cross-validated against the existing `cipher_flops.c` PMU counter (§3).

### 1.2 Correction 1 — inference is 2N, not the Lambda 6N

The Lambda-MFU formula the D1 spec cites — `6N + 12·L·H·Q·S` — is the
**training** coefficient (forward + backward + weight-grad ≈ 6 FLOP/param).
An **inference forward pass is 2 FLOP/param** (`2N`) plus the attention term
`4·L·H·Q·S` (Q = head-dim, S = sequence). This memo uses:

- **`2N + 4·L·H·Q·S` per forward** for the 19 inference WLs (decode, prefill,
  serving, agentic, spec-decode, RAG, code-gen, vision, embeddings, diffusion-
  step, multimodal).
- **`6N + …` (training)** for **WL07** (LoRA fine-tune) and **WL17** (training
  full) only.

Using 6N for inference would overstate FLOPs 3× and inflate `tensor_mfu_pct`
3×. The per-WL coefficient is stated explicitly in the §2 table.

### 1.3 Correction 2 — Hopper has no INT4 tensor cores

H100 SXM5 peak **dense** tensor throughput (NVIDIA H100 datasheet, no
2:4 sparsity):

| dtype | peak | note |
|---|---|---|
| fp16 / bf16 | **989.4 TFLOPS** | |
| fp8 (E4M3/E5M2) | 1978.9 TFLOPS | |
| int8 | 1978.9 TOPS | |
| **int4** | **— not supported** | Hopper tensor cores dropped the Ampere INT4 path |

The D1 spec's "H100 INT4 = 1978.8 TFLOPS" is **incorrect for Hopper**. Marlin
INT4 GEMM on H100 **dequantises INT4 weights to fp16 and runs the matmul on
fp16 tensor cores** — its compute peak is **989.4 TFLOPS (fp16)**. Consequence
for the ceilings: INT4 raises the *memory-bound* ceiling ~4× (4× weight
compression ⇒ 4× arithmetic intensity) but **does not raise the compute-bound
peak**. The 21 WLs as their drivers ship run **fp16** (no Marlin actuator
gated on); §2 ceilings are therefore fp16-based. (WL24, the INT4 workload, is
deferred — D2.)

### 1.4 The roofline and the regime taxonomy

B=1 decode reads each weight once per token; a 32K prefill reuses each weight
across thousands of token-positions. The **roofline ridge point** (fp16):

```
AI_ridge = peak_FLOPS / HBM_BW = 989.4e12 / 3.35e12 = 295 FLOP/byte
```

A workload with arithmetic intensity `AI < 295` is **memory-bound**; `AI > 295`
is **compute-bound**. The pre-registered tensor-MFU ceiling:

- **Memory-bound:** `ceiling = AI_workload / 295` — **exact physics**. (B=1
  fp16 decode: AI = 2N FLOP / 2N weight-bytes = **1 FLOP/byte** ⇒ ceiling
  **0.34%** — the tensor cores *cannot* exceed this; the GPU spends 99.7% of
  the step waiting on HBM. This is physics, not pessimism, and it is exactly
  why the legacy SM-cycle MFU was misleading.)
- **Compute-bound:** `ceiling = peak × tensor-eligible-time-fraction` — the
  realistically-attainable fraction, discounted for the non-tensor ops
  (softmax, norms, elementwise, optimizer) that run at non-tensor rates. Stated
  per-WL with basis.
- **Mixed:** time-weighted blend of a compute phase and a memory phase.

Substituted models (B5, pre-accepted): the drivers run **TinyLlama-1.1B** for
Llama-3.2-1B and **Mistral-7B** for Llama-3.1-8B. Ceilings below are computed
for the **substitute geometry actually executed**. WL09/WL19 are run with the
fetched **whisper-large-v3** and **clip-vit-large-patch14** (`WL_` env
override) to match the WL-taxonomy reference tier.

---

## 2. Pre-registered per-WL ceiling table (21 WLs)

**Locked before any measurement.** `peak = 989.4 TFLOPS fp16`,
`AI_ridge = 295 FLOP/byte`, `HBM_BW = 3.35 TB/s`. Gate (§4) = measured
`tensor_mfu_pct ≥ 0.80 × ceiling`.

| WL | workload / model (fp16) | regime | derivation | **ceiling** | **pass ≥** |
|---|---|---|---|---|---|
| WL01 | Decode B=1 · TinyLlama | memory-bound | AI=1 FLOP/B ⇒ 1/295 | **0.34%** | 0.27% |
| WL02 | Decode B=8 · TinyLlama | memory-bound | AI≈8 (weights shared across 8 tok) | **2.7%** | 2.2% |
| WL03 | Prefill B=8×1024 · Mistral-7B | compute-bound | GEMM-dominated, attn/norm discount 0.80 | **80%** | 64% |
| WL04 | vLLM serving · TinyLlama | memory-bound | continuous batch, effective B≈8 ⇒ AI≈8 | **2.7%** | 2.2% |
| WL05 | Multi-tenant ×8 · TinyLlama (G2/G3) | memory-bound | 8 procs each B=1, no weight share ⇒ device AI=1 | **0.34%** | 0.27% |
| WL06 | Embeddings batch-512 · MiniLM-L6 | compute-bound | large batch; tiny model ⇒ launch-overhead discount 0.45 | **45%** | 36% |
| WL07 | LoRA fine-tune · TinyLlama | training (6N) | fwd+bwd+opt; frozen-base, adapter grad; discount 0.55 | **55%** | 44% |
| WL08 | Diffusion 20-step · SDXL UNet | compute-bound | conv+attn+groupnorm; non-GEMM discount 0.60 | **60%** | 48% |
| WL09 | Speech · whisper-large-v3 | mixed | enc compute-bound / dec memory-bound; enc-weighted ≈0.25 | **25%** | 20% |
| WL10 | Speculative decode · vLLM TinyLlama | memory-bound | target+draft decode, AI≈4 (spec batching) | **1.4%** | 1.1% |
| WL11 | Agentic multi-turn · TinyLlama | memory-bound | B=1 decode, AI=1 | **0.34%** | 0.27% |
| WL12 | Batch-64 · TinyLlama | compute-bound | batch-64 prefill-dominant; discount 0.65 | **65%** | 52% |
| WL13 | Long-context 32K prefill · Mistral-7B | compute-bound | attn O(N²) compute-heavy; discount 0.80 | **80%** | 64% |
| WL14 | torch.compile decode · TinyLlama | memory-bound | B=1 decode, AI=1 (compile doesn't move the roofline) | **0.34%** | 0.27% |
| WL16 | Prefix caching · vLLM TinyLlama | memory-bound | shared-prefix batch-8, AI≈6 | **2.0%** | 1.6% |
| WL17 | Training full · TinyLlama (SGD) | training (6N) | fwd+bwd+opt, all params; discount 0.55 | **55%** | 44% |
| WL19 | Vision · CLIP ViT-L/14 batch-32 | compute-bound | vision encoder, batched patches; discount 0.62 | **62%** | 50% |
| WL20 | Multimodal · LLaVA-1.5-7B | mixed | vision-enc compute + 20-tok LLM decode (memory); decode-tail-weighted | **8%** | 6.4% |
| WL21 | Code generation · TinyLlama | memory-bound | B=1 decode 64 tok, AI=1 | **0.34%** | 0.27% |
| WL22 | RAG pipeline · MiniLM + TinyLlama | mixed | embed (compute) + gen (memory-bound decode) | **15%** | 12% |
| WL23 | Model-switch stress · TinyLlama↔Mistral | **not a compute WL** | dominated by weight load (HBM/PCIe/disk); tensor cores idle during churn | **n/a** | **stability-only (G4)** |

**Deferred (documented, not gated):** WL15 Mixtral (multi-GPU scale),
WL18 multi-GPU TP (single-GPU pod), WL24 AWQ INT4 (Phase 5).

### 2.1 Notes on the table — honest pre-registration

- **Memory-bound ceilings (WL01/02/04/05/10/11/14/16/21) are exact physics**
  — `AI/295`. The sub-1% absolute values for B=1 decode are correct: those
  workloads are bandwidth-bound and the tensor cores are structurally idle.
  The legacy 85%/90% G1/G2 thresholds were never physically attainable for
  these WLs under a true tensor-MFU metric — that mis-calibration is exactly
  what B5 + D1 correct.
- **Compute-bound / training / mixed ceilings carry a modelling discount**
  (the `tensor-eligible-time-fraction`). These discounts — 0.45–0.80 — are
  the **declared modelling-uncertainty** of the pre-registration: they are
  literature/architecture-grounded estimates, not measured. §3 (next
  installment) gives the PMU cross-check that bounds this uncertainty; §6 will
  state it as the memo's principal caveat. The memory-bound rows carry **no**
  such uncertainty.
- **B=8 / batched AI estimates are short-context approximations.** WL02/WL04
  `AI≈8` assumes weights reused across 8 batch elements *and* short context
  (the drivers use `max_new_tokens=32`) so the per-sequence KV term — which is
  **not** shared across the batch — stays small. Long-context B=8 decode would
  lower the effective AI as KV reads grow. WL16's `AI≈6` comes from the
  shared prefix (the unique per-sequence tails are not shared); it is likewise
  an approximation for the driver's 8-prompt config. Memory-bound *B=1* rows
  (AI=1) carry no such approximation.
- **WL23 is not a tensor-MFU workload.** Model-switch stress is weight-load-
  bound (HBM writes / PCIe / disk); tensor cores are idle during the switch.
  Assigning it a tensor-MFU ceiling would be meaningless. WL23 is gated on
  **G4 stability only** (no oops/WARN/leak across its rotation slot) — its
  per-WL G1 entry is "stability-pass," not an MFU number. Flagged for
  adjudication.
- **G2/G3 gate WL is WL05.** Its ceiling (0.34%, memory-bound) means the
  legacy "90% MFU on WL05" is replaced by "≥80% of the 0.34% bandwidth
  roofline = ≥0.27% tensor MFU, *and* ≥30 concurrent tenants sustained" — the
  density gate G3 is unchanged and remains the substantive WL05 bar.

---

---

## 3. Cross-check methodology — analytical vs `cipher_flops.c` PMU

The analytical `tensor_mfu_pct` (§1.1) is cross-validated against an
**independent** FLOP path: the `cipher_flops.c` PMU counter shipped in CP 3.3
(the `[CIPHER MFU] … FLOPs` substrate telemetry). Two independent estimates of
the same quantity, agreeing, is the engineering-marvel cross-check.

**Protocol.** For each WL, per measurement window:
`analytical_FLOPs` (model geometry × tokens) vs `pmu_FLOPs` (substrate
counter). Agreement gate: **within ±5%**. Disagreement >5% is a **reported
finding** — either a methodology bug (geometry/token-count error) or a
substrate-behaviour anomaly (the counter mis-attributing kernels) — and is
investigated before the WL's gate verdict is trusted.

**Coverage — stated honestly.** The PMU counter is wired to the substrate's
GEMM/kernel-dispatch path. Its cross-check strength is **not uniform**:

| WL class | PMU cross-check coverage |
|---|---|
| LLM decode/prefill/serving/agentic/spec/code/RAG-gen (WL01-05,10-14,16,17,21,22) | **Strong** — cuBLAS-GEMM-dominated; PMU sees the FLOPs directly |
| Embeddings, CLIP vision, LLaVA LLM-half (WL06,19,20) | **Strong** — transformer GEMMs |
| SDXL UNet, Whisper encoder (WL08, WL09) | **Partial** — convolution-heavy; conv kernels may not be GEMM-classified by the counter. Cross-check is advisory here; the analytical figure is primary and the coverage gap is declared. |
| WL23 model-switch | n/a (not a compute WL) |

Where coverage is **partial**, the report states it; the analytical number
stands as primary and the PMU number is advisory, not a veto.

---

## 4. Pre-registered pass criteria — G1–G4

Mechanical: pass/fail computed from measured numbers vs the §2 / below
pre-registered thresholds. **All four gates must pass simultaneously,
reproducibly, on real silicon. 3-of-4 = FAIL** (`PHASE_4_ARCHITECTURE.md`).

| Gate | Pre-registered criterion |
|---|---|
| **G1** | Each of the 21 gated WLs achieves `tensor_mfu_pct ≥ 0.80 × ceiling` (§2 "pass ≥" column), sustained ≥10 min. WL23 → **stability-pass** (no oops/WARN/leak in its slot), not an MFU number. |
| **G2** | WL05 device-aggregate `tensor_mfu_pct ≥ 0.27%` (= 0.80 × 0.34% memory-bound roofline). |
| **G3** | **≥30 concurrent tenants** sustained in the WL05 scenario, per-tenant non-trivial SM time; `density_pack.sh` 2→5→10→20→30→50→100. |
| **G4** | **Zero kernel oops/WARN; taint-bit Δ ≤ 1** across the 24-hour soak rotating the 21 WLs. |

**Honest pre-statement (per D1 / user).** Some compute-bound WLs may miss the
80%-of-ceiling bar — the modelling discount (§2.1) is an estimate, and real
substrate overhead is unknown until measured. **A miss is reported as data,
not hidden, and not retro-fitted.** If a WL misses, the report states whether
the gap is (a) substrate overhead, (b) a too-optimistic discount estimate
(cross-checked via §3), or (c) a workload-harness artifact — mechanically,
with evidence. The gate is the physics; the verdict is the measurement.

**G2 actuation note.** The legacy "G2 requires bare-metal" caveat does not
bind here — VOLT/DVFS actuated successfully on this pod in CP 2.4
(DVFS@1005 MHz); clocks are queryable. G2 is measurable. Confirmed at soak
start.

---

## 5. Measurement + soak protocol

**G1 and G4 are separate runs** — the architecture doc specifies
`mfu_per_tenant.sh per workload; sustained-window check` for G1 and
`soak_24h.sh extended for WL rotation; dmesg + taint deltas` for G4. They are
not conflated:

- **G1 (build-STEP item 4) — dedicated per-WL measurement runs.** Each of the
  21 WLs run standalone ≥10 min, `tensor_mfu_pct` sampled cleanly in a
  sustained window, analytical + PMU cross-check (§3) per WL. Clean
  measurement conditions — no rotation noise. ~21 × ~12 min ≈ 4–5 h.
- **G4 (build-STEP item 6) — the 24-hour rotation soak.** `soak_24h.sh`
  extended to rotate the 21 gated WLs, background. The soak's job is
  **kernel-path stability**, not MFU measurement — so a slot need only run
  long enough to exercise each WL's kernel/ioctl paths and survive repeated
  WL transitions. Cadence: **~5 min/WL × 21 = ~105 min/cycle ⇒ ~13–14 cycles**
  across 24 h (rationale: maximise WL-transition count and cumulative
  kernel-path coverage rather than dwell time — latent oops/leaks surface at
  transitions and under accumulation).
- **G2 + G3 (item 5)** — WL05 device-aggregate `tensor_mfu_pct`, and
  `density_pack.sh` 2→100-tenant push — a separate run, not inside the soak.
- **Monitoring during the soak (sampled, non-perturbing):** `dmesg` +
  `/proc/sys/kernel/tainted` delta every 5 min (G4 evidence); cipher-exporter
  `/metrics` snapshot per cycle; per-WL wallclock + token counts logged (a
  soak-time `tensor_mfu_pct` is recorded as a *health* sample, but the binding
  G1 number is the dedicated item-4 run).
- **Discipline (B8/B9/B13 codified):**
  - **Matched-pair, no stale baselines** (B8/B9) — every WL's gate number is
    measured *in the soak window*, never compared to a prior-day baseline;
    pod-state drift is ~5% same-condition, so only same-window numbers are
    gate-valid.
  - **Single-context per process** (B13) — the persistent-`cuCtxSetCurrent`
    side-effect is bounded to single-context tenants; WL05's 8 tenants are
    separate single-context processes, so B13 does not bind. Documented.
  - **B2 closed with note** — `libcipher_v2/cipher_cupti.c`'s process-wide fd
    is benign at soak cadence (~40 ioctl/s ≪ 8 M/s VFS cap); and the soak
    runtime does not load `libcipher_v2.so` at all (D6). B2 closes "still
    benign at chosen cadence."
- **Anchor verification:** kmod `e2f50452`, libcipher_rt `c2c5d313` md5-verified
  at soak start *and* end — an anchor change mid-soak invalidates G4.
- **Density (G3):** `density_pack.sh` run separately (not inside the 24 h
  rotation) — the 2→100-tenant push to find the sustained-tenant ceiling.

---

## 6. Honest caveats (carried into the report)

1. **Compute-bound ceiling discounts are modelled, not measured** — the
   principal uncertainty of this pre-registration. The 0.45–0.80
   tensor-eligible-time fractions (§2) are literature/architecture estimates.
   The §3 PMU cross-check bounds the *FLOP-count* half; the *time-fraction*
   half is only validated by the measurement itself. If a compute-bound WL
   lands far from its ceiling, the report distinguishes substrate overhead
   from a mis-estimated discount. **Memory-bound WL ceilings carry no such
   uncertainty — they are exact `AI/295` physics.**
2. **Memory-bound WLs cap tensor MFU structurally below 1%** — this is HBM-
   bandwidth physics, not a substrate failure. The gate is roofline-relative
   (≥80% of the WL's own ceiling) precisely so a 0.34%-ceiling WL and an
   80%-ceiling WL are tested equally. The legacy absolute 85/90% G1/G2
   thresholds were never physically attainable under a true tensor-MFU metric.
3. **3 of 24 WLs deferred** — WL15 (Mixtral, multi-GPU scale), WL18 (multi-GPU
   TP, single-GPU pod), WL24 (AWQ INT4, Phase 5). CP 4.8 closes Phase 4 on
   **21/24** with documented physical reasons for the three.
4. **WL23 is gated on stability only** — model-switch is weight-load-bound;
   tensor MFU is not a meaningful metric for it.
5. **Substituted models** — TinyLlama-1.1B / Mistral-7B stand in for
   Llama-3.2-1B / Llama-3.1-8B (B5, pre-accepted); ceilings use the substitute
   geometry. WL09/WL19 run the fetched whisper-large-v3 / clip-vit-large-patch14.
6. **PMU cross-check is partial on SDXL/Whisper** (conv path, §3).

---

## 7. Build STEP — items (2–N gated on memo adjudication)

1. **Design memo** — this document. Hold for adjudication.
2. **`tensor_mfu_pct` path** — refine `mfu_compute.py`: analytical
   FLOP-roofline `tensor_mfu_pct` alongside the legacy `mfu_pct`; per-WL
   FLOP-geometry table; wire the PMU cross-check readout.
3. **Per-WL ceiling pre-registration artifact** — freeze the §2 table as a
   machine-readable `cp48_ceilings.json` before any measurement.
4. **G1 measurement** — `mfu_per_tenant.sh` across the 21 WLs; sustained
   10-min windows; analytical + PMU cross-check per WL.
5. **G2 + G3** — WL05 device-aggregate `tensor_mfu_pct`; `density_pack.sh`
   2→100-tenant push.
6. **G4 — 24-hour soak** — `soak_24h.sh` 21-WL rotation, background.
7. **Backlog close** — B2 benign-at-cadence note; B8/B9/B13 discipline
   recorded.
8. **Report** — `CP_4_8_REPORT.md`: per-WL measured-vs-ceiling table, G1–G4
   verdict, Phase 4 close recommendation. CP 2.4/2.5/4.6.5+6 register.

---

## 8. Calendar & anchors

| Step | Estimate |
|---|---|
| `tensor_mfu_pct` build + ceiling artifact (items 2–3) | ~0.5 day |
| G1 + G2 + G3 measurement (items 4–5) | ~1 day |
| 24-hour soak (item 6) | 1 day wall-clock, background |
| Backlog close + report (items 7–8) | ~1 day |

**Envelope ≈ 3–4 days to Phase 4 close.** Anchors held throughout: kmod
`e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30` (not soak-loaded).
No substrate code changes — the only build artifact is the measurement-path
refinement.

**Phase 4 close.** CP 4.8 is the Phase 4 ship gate. On a clean G1–G4 pass,
Phase 4 is **DONE** — not "components validated" but "the composed production
stack passed sustained-load ship-gate criteria measured against pre-registered
physics-derived ceilings." Any gate miss is reported honestly with its cause;
a partial pass is a documented FAIL, not a soft close.

---

**Items 2–8 HELD for memo adjudication.** §0–§8 complete.

