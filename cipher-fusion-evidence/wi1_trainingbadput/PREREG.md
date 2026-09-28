# WI-1 TRAINING-GOODPUT / BADPUT-AVOIDED — RULE-4 PRE-REGISTRATION (before measurement)

**Goodput definition used: TRAINING goodput** = compute-progress-time / total-time (Nebius/Google/Meta). SDC detection
reduces BADPUT (wasted compute from rollback after a corruption is discovered). This is the definition under which the
R.A detector is a WIN. (NOT correctness-goodput, which the prior R_AB wrongly used.)

## Model of the wasted compute
Real Mistral-7B LoRA finetune step loop on one H100; measure step time `t_s` (fwd+bwd+optimizer) and checkpoint write
time. Checkpoints at steps 0, C, 2C, … Persistent SDC injected at step `S` (a forward linear-GEMM bit-flip), caught by
the R.A cuBLAS-intercept detector with latency `N` training steps (WITH CIPHER). Definitions:
- `K_good` = last checkpoint strictly before onset S (checkpoints at ≥S capture the corrupted state ⇒ bad).
- WITHOUT CIPHER: no online SDC check ⇒ corruption silent until discovered at downstream validation `D_without`
  (next eval / loss-divergence / end-of-window). Rollback to `K_good`. **wasted_without = (D_without − K_good)·t_s.**
- WITH CIPHER: caught at `S+N`. Rollback to `K_good`. **wasted_with = (S+N − K_good)·t_s.**
- **BADPUT-AVOIDED per fault = wasted_without − wasted_with = (D_without − S − N)·t_s GPU-seconds.** (K_good cancels —
  both roll back to the same last-good checkpoint; the difference is purely *discovery latency*.)

## NET of detector overhead (fair)
Detector runs continuously at fractional overhead `ovh` (measured fresh: step time with checks-every-N vs without).
Over a run of `T_total` steps with `F` faults:
- **NET badput-avoided = F·(D_without − S − N)·t_s − ovh·T_total·t_s.**
- **Breakeven fault rate r\* = ovh / (D_without − S − N)** faults/step. **This is step-time-INVARIANT** (badput and
  overhead both scale with t_s) ⇒ a model-size-independent result; the GPU-seconds scale with t_s (report at measured t_s).
- **Effective-training-time** = progress_steps·t_s / (progress_steps·t_s + wasted·t_s + overhead·t_s), WITH vs WITHOUT.

## PRE-REGISTERED EXPECTATIONS (commit before measuring)
1. **Badput-avoided is POSITIVE whenever `D_without − S − N > 0`** — i.e. whenever the silent-discovery delay exceeds the
   detector latency N. For SDC this is essentially always: silent corruption can hide for many steps/checkpoints (the
   whole reason SDC is dangerous), while N is a few-to-tens of steps. Expect a large positive badput-avoided when
   `D_without` is a full validation interval or end-of-window.
2. **Breakeven fault rate r\* = ovh/(D_without−S−N).** With ovh≈3% and `D_without−S−N` ≈ one checkpoint interval
   (e.g. C=100, N=45 ⇒ ~55 steps), r\* ≈ 0.03/55 ≈ **5e-4 faults/step**; if SDC strikes more often than ~1 per ~2000
   steps the continuous overhead is repaid. If discovery is much later (silent for a full eval interval V≫C), r\* drops
   far lower (CIPHER repays at much rarer faults). Anchor a representative SDC cadence from Meta's failure logs
   (~1 failure / 3 h at 16K GPUs ⇒ per-GPU MTTF ~ years for hard failures; SDC is rarer per-GPU but fleet-frequent) and
   STATE the assumption; report badput-avoided as a function of discovery-delay rather than a single hero number.
3. **Detector overhead in training ≈ a few %** (recompute of sampled linears every N steps); MEASURE fresh, do not reuse
   the inference 3%.
4. **Bounds:** linear-GEMM coverage only (attention/lm_head/backward-grad GEMMs not all checked); mercurial-core blind
   (same-GPU recompute reproduces it); single-process LoRA loop is NOT a real distributed job (real 7B pretraining is
   multi-GPU with sharded optimizer — full 7B backward does not fit one 80GB GPU, itself the reason training is
   distributed); checkpoint/rollback modeled at single-node granularity.

## What is REAL vs MODELED
REAL: step time (fwd+bwd+optimizer), checkpoint write time, the injected SDC + detector catching it, detector overhead.
MODELED (stated): the discovery delay `D_without` (assumption, reported as a range/function), the fault rate (cited),
the multi-GPU distributed-job context (single-GPU LoRA stands in; full 7B backward doesn't fit → that non-fit IS the
real distributed-training fact).
