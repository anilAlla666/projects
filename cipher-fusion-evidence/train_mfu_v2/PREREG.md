# PREREG — FP8 ACTUATOR SUCCESSOR (TRAIN-MFU v2), 2026-06-11

Written BEFORE the successor is built and BEFORE any measurement. This file is never edited after
measurement begins (corrections, if any, go in the report as dated disclosures).

## What is being tested

The 2026-06-10 build measured the FP8 actuator as-built at −24/−25 % MFU + convergence FAIL, and the
06-11 probes attributed BOTH walls to implementation artifacts (per-step 159-weight requant churn from
pointer-only cache keying; wrong-math bwd GEMMs from ignored transa/lda). This is Option 1 from that
report: a scoped successor that removes both causes, decided by re-measurement.

**Successor scope (exactly four changes, all in the FP8 engine/actuator pair; nothing else):**
1. Weight cache keyed on (pointer, K, N) — fwd/bwd orientation swap no longer resets the slot.
2. Actuator refuses any call that is not the forward-linear layout the engine hardcodes
   (transa=T, transb=N, lda=k, ldb=k, ldc=m) → passthrough to fp16 cuBLAS (correct by construction).
   Backward input-grads therefore go fp16. New counter `layout_refused` for attribution.
3. cuBLASLt algo autotune per shape (time the heuristic's top candidates once at first use, keep the
   fastest) — replaces first-heuristic-only.
4. pack_key shape-cache hash hardened to non-overlapping bit packing (latent collision hazard).

**Build discipline:** successor built in a COPIED tree (`/home/ubuntu/cipher_fp8v2_build/`); the frozen
anchor `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` (md5 `2edba0d2136f8ede4713d90a8f7cd55f`, verified
at entry 2026-06-11) is never rebuilt, never moved; `make` is never run in `cipher_rt_phase4/`. The
successor tree starts from the current working tree (HEAD b2304d3 + the uncommitted default-OFF fairness
shim hook); attribution does not depend on anchor-source equality because all substrate arms
(obs-only, FP8) use the SAME successor .so — the obs-only arm re-measures the hosting tax directly.

## Workload + harness (unchanged from 06-10 for comparability)

Real LoRA-finetune Mistral-7B, B=2×seq=2048, sdpa, AdamW, fp32 adapters, seed 0, fixed batch; MFU =
tokens·4N/step_time ÷ 989.5 TFLOP/s (finetune 4N model, same conservative caveats). Harness =
`train_step.py` copied to `train_mfu_v2/` with ONE change: the FP8-counter read takes the lib path as a
parameter (the original hardcodes the anchor path). Engagement only via caller env
(`CUDA_INJECTION64_PATH` + `CIPHER_FP8=1`, `CIPHER_VOLT=off`, `VLLM_PLUGINS=""`); harness never sets it.

## Run matrix

1. **Probe A (correctness gate, runs FIRST):** fwd/bwd per-call rel-err vs fp32 on q/kv/gate shapes.
2. **Vanilla baseline** 50 steps (fresh same-day pair; 06-10 gave 24.6 %).
3. **Obs-only successor** (injection, FP8 unset) — hosting tax.
4. **FP8-engaged successor** default-clock full-window (8 warm + 50).
5. **FP8-engaged successor** steady-state (20 warm + 40).
6. **Iso-clock pair** (pin 1830 MHz): vanilla + FP8; clocks reset after.
7. **Convergence pair** 100 steps: baseline + FP8 loss curves.

## Pre-registered expectations + decision rules

- **Probe A gate:** fwd handled rel-err ≈ 0.03–0.05 (per-tensor FP8 noise, 06-10 measured 0.0375);
  bwd handled = 0 (every bwd call layout-refused); bwd rel-err ≤ 1e-3 (fp16-exact path).
  ANY violation ⇒ stop, fix, re-run probe before MFU runs.
- **Hosting tax (obs-only vs vanilla):** expected ≈ +3 % step time (06-10: +3.1 %). >6 % ⇒ successor
  host regressed; investigate before interpreting FP8 arms.
- **Engagement model:** handled/step ≈ 225 (224 fwd linears + lm_head fwd), layout_refused/step ≈ 222
  (all bwd input-grads), requant churn ELIMINATED after warmup (weights quantized once; steady-state
  prequant events = 0; successor counters must show this).
- **MFU (the honest expectation): roughly neutral.** Removing the churn recovers the as-built −21 %
  actuator overhead, but what remains is fwd-only coverage (~15–39 % of GEMM FLOPs; GEMM bucket itself
  only 45.5 % of step time) at MIXED per-shape FP8-Lt quality (gate_up fwd was +82 % slower
  pre-autotune). Pre-registered band: **−5 % to +5 % vs vanilla.**
  Decision rule: LIFT iff FP8-engaged median-step MFU > vanilla by >1 % in BOTH default-clock runs AND
  iso-clock, AND convergence passes. NO-LIFT (neutral) if |Δ| ≤ 1 % or signs disagree across regimes.
  LOSS if < −1 % in all regimes. Any outcome is reportable; none triggers a silent re-scope.
- **Convergence PASS (pre-registered):** loss decreases (beats step-1 loss by ≥10× at step 100),
  0 NaN, final loss ≤ 0.02 (≤ ~10× the 06-10 baseline 0.0018). Rationale: bwd is now fp16-exact, fwd
  carries per-tensor-FP8 activation+weight quant noise; memorization should proceed, possibly slower.
  FAIL ⇒ the fwd-only per-tensor recipe is quality-broken even with correct math — reportable as a
  recipe result (unlike 06-10, which could not rank recipes).
- **What this run can and cannot conclude:** It CAN settle whether the substrate's per-tensor-scalar
  fwd-FP8 actuator, correctly implemented, lifts realistic-batch finetune MFU. It CANNOT rank FP8
  recipes generally (one recipe, one config), and a PASS here is a fixed-batch memorization proxy, not
  a real-data convergence qualification.

## Known risks (pre-stated)

- At-exit segfault 139 on engaged runs (known, post-results; JSONs written before exit).
- Autotune timing runs during first use (warmup window) — must not leak into measured steps; verified
  by steady-state run agreement.
- fp16 weights cached per (ptr,K,N): pointer-reuse hazard if torch frees/reallocs a weight at the same
  address with the same shape — base weights are frozen for the whole run; hazard unchanged from
  as-built, disclosed.
- +~7 GB device memory for one-time e4m3 copies of all fwd weights (was churn-bounded before);
  baseline peak 55.6 GB ⇒ fits 80 GB. OOM would be a reportable cost of the fix.
