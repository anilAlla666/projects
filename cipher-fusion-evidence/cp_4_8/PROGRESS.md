# CP 4.8 — integration soak / Phase 4 close — build progress

The Phase 4 ship-gate CP. Validate the composed production stack against the
G1–G4 criteria, measured vs pre-registered physics-derived per-WL ceilings.
Durable cross-session checkpoint; `CP_4_8_REPORT.md` is the report at close.

Anchors: kmod 0.4.8 `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2
`86618c30` (not loaded by the soak runtime — D6).

## Status

- **D6 — DONE.** `D6_LIBCIPHER_V2_ANCHOR_RESOLUTION.md`. Soak runtime does not
  load `libcipher_v2.so` (v2 symbols statically linked into libcipher_rt);
  anchor stays `86618c30`; `cc0479b8` preserved as
  `libcipher_v2.so.v0.3.0-cupti.unpromoted`, not promoted.
- **Step 2 (model fetch) — DONE.** MiniLM, CLIP ViT-L, Whisper-large-v3, SDXL,
  LLaVA-1.5-7B fetched. 22/24 WL models present; driver audit found WL18
  (multi-GPU TP) un-runnable on a single-GPU pod ⇒ **21/24 gated**, 3 deferred
  (WL15, WL18, WL24).
- **Step 3 (tensor-MFU method) — RESOLVED.** No DCGM; CUPTI profiling perturbs
  the soak ⇒ analytical FLOP-roofline `tensor_mfu_pct` (D1 Option A,
  user-confirmed). Built as build-STEP item 2, not pre-built.
- **Step 4 (design memo) — DONE.** `CP_4_8_DESIGN_MEMO.md` §0–§8 complete:
  21-WL pre-registered ceiling table, two physics corrections (2N inference
  not 6N; Hopper has no INT4 tensor cores → Marlin peak = fp16 989 TFLOPS),
  analytical-vs-PMU cross-check, G1–G4 criteria, 24 h soak protocol.
- **Memo ADJUDICATED 2026-05-17** — 4 review questions accepted (2N/INT4
  corrections, 21/24 WL scope, discount estimates pre-registered, WL23
  stability-only). Build STEP proceeding.
- **Item 2 — DONE.** `mfu_compute.py` extended: analytical `tensor_mfu_pct`
  (FLOP-roofline), per-WL FLOP table, PMU cross-check, `crosscheck()`.
  Self-tested: WL01 @ bandwidth roofline = 0.340% (matches §2 ceiling).
- **Item 3 — DONE.** `cp48_ceilings.json` — 21-WL ceilings frozen pre-measurement.
- **Task B — DONE.** CP 2.4 composed re-measure on c2c5d313: 3.6166× tok/W
  [3.5908, 3.6423] — exact reproduction. `taskB_cp24_remeasure/TASK_B_RESULT.md`.
- **WL deps verified** — vllm/diffusers/peft/sentence_transformers/whisper all present.
- **§3 cross-check fork — RESOLVED (Option C).** cipher_flopd not deployed →
  live PMU path dead; independent FLOP source = PyTorch FlopCounterMode
  (ATen op-graph trace). cipher_flopd-not-deployed logged as a Phase-5/operator
  backlog item.
- **§3 spot-check — DONE, methodology VALIDATED.** `cp48_flop_spotcheck.py` /
  `_result.json`. First run disagreed >5% on WL01 (+5.84%) and WL13 (+9.78%) —
  paused + investigated per instruction:
  * **WL01 — real bug found + fixed.** Analytical `2N` counted the input
    embedding *lookup* table (0 matmul FLOPs). Gap was exactly `2×vocab×hidden`.
    Fix: `mfu_compute.py` coefficient term now uses `(params − embed_params)`;
    lm_head stays (real matmul). Attention also added for training (×3).
  * **WL13 — convention, not a bug.** FlopCounterMode counts DENSE attention
    (no causal mode); the gate analytical correctly models CAUSAL-half (what
    the real SDPA/FlashAttention workload executes). Cross-check matched on
    the dense convention validates the geometry.
  * Re-run: **WL01 0.06%, WL17 1.10%, WL13 1.40% — all within ±5%. VALIDATED.**
- **LEAN PIVOT (user, 2026-05-17).** The per-WL ceiling pass/fail apparatus
  was over-engineered — grading against the 85/90% MFU criterion, which is
  not physically achievable for B=1 agentic decode (tensor cores near-idle by
  workload nature; WL01 validation measured 4% of the bandwidth roofline).
  CP 4.8 cut to: **G4 soak (real ship-gate question — survives sustained
  load), G3 density, MFU descriptive-only (no gate)**. Report simplified to
  §1–§6. Fork-ceremony dropped — ordinary engineering calls from here.
- **Item 4 (descriptive MFU) — RUNNING.** `cp48_g1_runner.py --all`,
  180s windows (descriptive, not 600s gate windows), libcipher_rt c2c5d313,
  background pid 1450210.
- **G3 density / G4 soak — scripts ready.** `cp48_density.sh`, `cp48_soak.sh`
  (copied from cipher_measurement, repointed to libcipher_rt c2c5d313).
  cipher-gpustate (root) + cipher-exporter running. Sequence: MFU run →
  density → 24h soak (background) → report.

## Key memo content for review

- Gate = measured `tensor_mfu_pct ≥ 0.80 × pre-registered per-WL ceiling`.
- Memory-bound WL ceilings are exact `AI/295` physics (B=1 decode → 0.34%);
  compute-bound ceilings carry a declared modelling discount (0.45–0.80).
- WL23 (model-switch) gated on G4 stability only — not a tensor-MFU workload.
- 3 WLs deferred with physical reasons. Some compute-bound WLs may miss the
  80%-of-ceiling bar — pre-stated as honest data, not retro-fitted.

## Session log

### 2026-05-17 — STEP started. D6 + model fetch + memo done; held at memo.
Six scope decisions adjudicated by user. D6 resolved, 5 models fetched,
tensor-MFU method confirmed analytical (Option A). Design memo §0–§8 written.
Awaiting memo adjudication before build-STEP items 2–8.
