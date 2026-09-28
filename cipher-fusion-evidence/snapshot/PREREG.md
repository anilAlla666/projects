# PREREG — CIPHER RESULTS SNAPSHOT (2026-06-10)

Written BEFORE any measurement in this session. All Part-A measurements are dispatch-boundary-legal:
clean vLLM/torch baseline (NO CIPHER substrate loaded) or scratch LD_PRELOAD shim copies in snapshot/.
Anchor libcipher_rt.so md5 2edba0d2136f8ede4713d90a8f7cd55f frozen (entry manifest: fork1_md5_entry.txt).

## Fixed analytic models (stated up front so measurements can't bend them)

- Model: mistralai/Mistral-7B-v0.1, fp16, vLLM 0.20.2 **eager** (enforce_eager=True).
  Caveat carried everywhere: eager host tax vs cudagraph is +51% (R_A_REAL_VLLM_2026-06-08.md, 629 vs 1289 tok/s).
- Param count (exact, from architecture: h=4096, L=32, kv_heads=8, inter=14336, vocab=32000):
  N_params = 7,241,732,096 ≈ 7.242e9 (embed 131.07M + 32×218.112M + final norm + lm_head 131.07M).
- Decode FLOP model: FLOPs/token = 2·N_params = 14.483 GFLOP/token (linear + lm_head matmuls; attention
  score/value FLOPs excluded — at ctx ≤ a few hundred they are <2% extra; stated as a lower bound on work).
- Decode byte model (weight-stream): bytes/step = N_params·2 B (fp16) = 14.483 GB; bytes/token = 14.483/B GB.
  KV-cache reads/writes EXCLUDED (small at short ctx) → MBU computed this way is a LOWER bound.
- Prefill FLOP model: FLOPs = 2·N_params·T_prompt_total (same per-token matmul model; attention quadratic
  term excluded, <4% at 2048 ctx).
- Peaks (cited): H100 SXM BF16/FP16 dense 989.5 TFLOP/s (NVIDIA H100 datasheet, no sparsity);
  HBM3 3.35 TB/s (same datasheet). MFU = achieved FLOP/s ÷ 989.5e12. MBU = achieved bytes/s ÷ 3.35e12.
- TPW: tokens ÷ trapezoid-integrated NVML energy (power.draw @50 ms) over the measured generate window
  (validated method from rc_ab/ab_driver.py).
- Consistency cross-check: W6 B=1 163 tok/s → 163×14.483e9/989.5e12 = 0.239% ≈ the quoted 0.25% decode MFU,
  so the A1 model is the same family as W6's. A1 (eager) is expected to differ from W6 where W6's config
  was not eager-constrained; differences will be noted, not hidden.
- Clock: locked 1980 MHz before runs; achieved clock + SwPowerCap throttle reasons recorded per run; -rgc at exit.

## A1 pre-registered expected ranges (eager fp16, this H100)

| Workload | tok/s | MFU | MBU (weight-stream LB) | TPW |
|---|---|---|---|---|
| Decode B=1 (latency) | 60–130 | 0.09–0.19 % (LOW = memory/launch-bound truth, not a defect) | 26–56 % | 0.2–0.8 tok/J |
| Decode B=64 (throughput) | 2000–4500 | 3.0–6.6 % | 14–31 % (per-step weight stream ÷ shared over 64 seqs) | 5–14 tok/J |
| Prefill B=8×~2048 (compute-bound) | (prompt tok/s) 25k–45k | 50–65 % | n/a primary (compute-bound) | 30–80 prompt-tok/J |
| LoRA train step (wi1 harness) | — | 15–45 % (model: ~6·N FLOPs/token fwd+bwd, stated with result) | — | reported if stable |

Surprise rule: anything outside these bands is reported as a surprise with a mechanism guess, not silently absorbed.
If the wi1 LoRA step cannot be made stable quickly (no NaN reintroduction), train-MFU is marked UNAVAILABLE.

## A2 pre-registered expectations (scratch shim, heartbeat OFF, no /dev/cipher, ldd shows no libcipher)

- R.A overhead, B=8 eager bench (prior: 629 clean / 609 @N=45 [-3.1%] / 554 @N=8 [-11.9%] tok/s,
  R_A_REAL_VLLM_2026-06-08.md; rc_ab B=1 measured +4.0%@N45 / +12.6%@N8):
  expect clean 629±5%, N=45 overhead 2–5%, N=8 overhead 8–14%.
  NOTE: the task brief says "~6% @N=8"; the validated priors say ~12%. Discrepancy is pre-registered here;
  whatever measures is reported against BOTH and reconciled in the report.
- R.A 0-FP: 0 detections on clean run, max_clean_residual exactly 0 (T=0 bit-identical recompute); checks count reported.
- R.A latency: injected persistent bit-14 SDC caught at the FIRST check step after onset → ≤N steps (both N=45, N=8).
- R.B (batch mode): persistent ONSET SDC detected within ≤N steps of onset and BEFORE serve (batch returns at end);
  clean run 0 false quarantine; transient injected BETWEEN check steps → 0 detections (the honest ~1/N miss wall).
- R.C: injected SDC → DEGRADED verdict with onset step + descriptor (SDC-pid==NVML-pid, VLLM::EngineCore);
  clean run → 0 FP across all checks. NVML-clean on software flip EXPECTED (orthogonal counters — honest framing).

## Part B (NOT measured; quoted with source)

Mux 3.06×/3.30× tok/W, DVFS +57.3% tok/W, W6 MFU/MBU rows, eager-vs-cudagraph 629/1289 — each must be
located verbatim in its prior report; anything not found is marked UNVERIFIED and excluded from the table.
Reason-not-re-run (mux): engagement requires a vLLM-cooperating host = the walled, partner-gated measurement
per the Nebius memo; re-deriving it would cross the substrate line. NOT faked here by design.
