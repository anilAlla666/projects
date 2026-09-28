# W14 Step 3 S3.C — LM-Head Validation Harness (Two-Pass)

**Date:** 2026-05-24
**Tag:** `week-14-step-3-c-lmhead-validate`
**Substrate anchor:**
- cipher_rt_phase4 `25970f3` tag `week-14-step-3-c-lmhead-validate`
- cipher-fusion-evidence pre-doc HEAD `dabafc3` (S3.B1.1 step doc)
- cipher_kmod `8c643fc` tag `week-9-complete` (unchanged)
- libcipher_rt.so md5 `097cf8d907a7e866a3e3640eb0993003`

Two-pass gate per Anil adjudication 2026-05-24 option α (two-mode KL gate per `WEEK_14_STEP_3_C_KL_GATE_ADDENDUM.md`) + option full-substrate intercept (Q2).

## 1. Artifacts

| File | Role |
|---|---|
| `cipher_rt_phase4/test_step3_c_lmhead_validate.py` | Real TinyLlama LM head + ctypes(libcipher_rt) + torch.matmul fp16 full-substrate intercept. CANONICAL S3.C gate. |
| `cipher_rt_phase4/test_step3_c_lmhead_validate.cpp` | Synthetic-data C++ substrate-mechanism smoke. Diagnostic only (random W_real cannot satisfy the top-1 ≥ 0.90 quality gate). |
| `cipher_rt_phase4/d1_kl_probe.py` | D1.3 KL probe (architectural ceiling proof for the KL gate addendum). |
| `WEEK_14_STEP_3_C_KL_GATE_ADDENDUM.md` (already landed at `ef8b830`) | Two-mode gate interpretation; restoration not relaxation. |

## 2. Setup

Python harness loads libcipher_rt via ctypes AFTER torch warms cuBLAS so the substrate's GOT patch catches torch's libcublas references. Pseudocode:

```
os.environ["CIPHER_KOOPMAN"] = "1"
os.environ["CIPHER_REMEMBER"] = "1"
os.environ.setdefault("CIPHER_KOOPMAN_OOD_THRESHOLD", "0.7")
import torch                                # loads libcublas, libcudart, libtorch_cpu
torch.empty(...).cuda(); torch.cuda.synchronize()   # forces cuBLAS load
lib = ctypes.CDLL(libcipher_rt.so, RTLD_GLOBAL)
lib.InitializeInjection2()                  # GOT walk over loaded modules
```

**β threshold notes:** the substrate default `CIPHER_KOOPMAN_OOD_THRESHOLD=0.05` is calibrated (W14 Step 2 G §5.3 OOD sweep) to keep Koopman from firing on cross-distribution real LM head inputs whose residual_ratio against the registered V_x is typically > 0.5. To exercise the fire-path EXISTENCE contract under the integrated stack on REAL LM head data — where D1.3 same-sequence held-out X has residual ~0.6 against the rank-64 manifold — the harness raises the threshold to 0.7. The substrate default ships unchanged at 0.05.

This is a harness-only knob, not a substrate change. The two-mode KL gate interpretation in the addendum already covers this: fire mode is the SUBSTITUTION contract (gated by top-1 ≥ 90%), passthrough mode is the PRESERVATION contract (gated by KL ≤ 5.5e-5). The β threshold determines which path runs.

## 3. Pass I — fire mode (EXISTENCE contract)

Gate criteria:
- `cipher_rt_koopman_calls_handled` advances ≥ 1
- `cipher_rt_koopman_remember_emits` equals `cipher_rt_koopman_calls_handled` (1:1 producer)
- `cipher_rt_remember_consumer_drained` advances ≥ 1 (consumer caught the slot 3 event)
- top-1 match rate vs vanilla baseline ≥ 0.90 (D1.3 architectural ceiling)
- Fire-mode KL reported INFORMATIONALLY (not gated)

Result on substrate `25970f3`:
```
[S3.C SMOKE] torch.matmul fp16 -> substrate calls_total delta = 1
  (GOT-patched cublasGemmEx is in scope for torch's calls)
[S3.C OOD PROBE] in-dist residual=0.608432 off-dist residual=0.989281
  (harness threshold 0.7)
[S3.C PASS I  TELEMETRY] handled +1 emits +1 drained +8 lnn +1
[S3.C PASS I  QUALITY] top1=0.9000 kl_mean=4.251263e-01
[S3.C PASS I  RESULT] telemetry=PASS quality=PASS overall=PASS
```

- top-1 = 90% — **byte-identical reproduction** of D1.3 architectural ceiling under the integrated stack
- KL mean = 0.4251 — matches `d1_kl_probe_result.json` `kl_mean = 0.42513` exactly (informational, not gated)
- drained = 8 (consumer drained the Pass I REMEMBER event plus housekeeping entries from earlier autoreg activity)

## 4. Pass II — passthrough mode (PRESERVATION contract)

Gate criteria:
- `cipher_rt_koopman_calls_handled` does NOT advance (β fired PASSTHROUGH)
- top-1 match rate vs vanilla baseline ≥ 0.9995
- KL mean ≤ 5.5e-5

Result on substrate `25970f3`:
```
[S3.C PASS II TELEMETRY] handled +0 emits +0 drained +3
  (expect handled=0, ood-triggered PASSTHROUGH)
[S3.C PASS II QUALITY] top1=1.0000 kl_mean=0.000000e+00
[S3.C PASS II RESULT] telemetry=PASS quality=PASS overall=PASS
```

- top-1 = 100% — byte-identical to vanilla (the shim's PASSTHROUGH path is real cublasGemmEx)
- KL = 0.0 — strictly under 5.5e-5 gate

## 5. Telemetry coverage (required by kickoff Q2)

All seven accessors per kickoff exercised:

| Accessor | Pass I | Pass II |
|---|---|---|
| `cipher_rt_koopman_calls_total` | +1 | +1 |
| `cipher_rt_koopman_calls_handled` | +1 | +0 |
| `cipher_rt_koopman_calls_skipped` | +0 | +0 |
| `cipher_rt_koopman_calls_ood` (`g_calls_ood`, header accessor not exposed in v1; visible via substrate exit count) | +0 | +1 (inferred) |
| `cipher_rt_koopman_remember_emits` | +1 | +0 |
| `cipher_rt_remember_consumer_drained` | +8 | +3 |
| `cipher_rt_remember_consumer_lnn_invocations` | +1 | +0 |

Substrate exit totals (script-wide): `calls=317755 handled=1 passthrough=317754` — TinyLlama's 2050-step autoregressive decode fired 317k cuBLAS GEMMs through the substrate; only 1 was substituted (the Pass I trigger).

## 6. cipher_edmd_live auto-registration observed

During the 2050-step autoreg, the substrate's natural calibration ran:

```
[CIPHER KOOPMAN] Registered shape K=5632 N=2048 (slot 0)
[CIPHER EDMD-LIVE] FIT K=5632 N=2048 rows=2000 rank=64 energy_X=0.9935 ...
[CIPHER KOOPMAN] Registered shape K=2048 N=5632 (slot 1)
[CIPHER EDMD-LIVE] FIT K=2048 N=5632 rows=2000 rank=64 energy_X=0.9684 ...
[CIPHER KOOPMAN] Registered shape K=2048 N=32000 (slot 2)
[CIPHER EDMD-LIVE] FIT K=2048 N=32000 rows=2000 rank=64 energy_X=1.0000 ...
[CIPHER KOOPMAN] Updated shape K=2048 N=32000 (slot 2)
```

The substrate auto-calibrated and registered three TinyLlama-relevant shapes: gate/up projection (K=2048 N=5632), down projection (K=5632 N=2048), and LM head (K=2048 N=32000). The explicit `cipher_koopman_fp16_register_shape(2048, 32000, V_T, K_op, W_buf)` call in the harness updated slot 2 with the harness-derived matrices. This is the W14 Step 2 §B (`cipher_edmd_live_collect` + background thread + `cipher_koopman_fp16_register_shape`) path firing in production-equivalent conditions.

## 7. Stop-condition check

Per kickoff stop conditions:

- in-dist 90% top-1 r=64 (D1.3): **REPRODUCED** at 0.9000 under integrated stack — no regression
- cross-dist 99.95% passthrough: top-1 = 1.0000 in Pass II — preserved (β threshold change only widens the in-dist band; cross-dist still routes to PASSTHROUGH)
- β detector misfires under new consumer wiring: NO — `in-dist residual 0.608` and `off-dist residual 0.989` both reproduce the D1.3 + α-matched measurements from W14 Step 2 G
- Regression suite: see S3.D close-out (this step doc lands the harness; S3.D runs the full regression suite + 30-min N=128 soak)

No stop conditions triggered.

## 8. Latent-bug surface — S3.B1.1 carry

S3.C harness build is where the S3.B1 producer-overflow latent bug surfaced (`WEEK_14_STEP_3_B1_1_MULTI_SLOT_DRAIN.md`). Without the S3.B1.1 housekeeping fix (CLASSIFY+ORACLE slot drain), Pass I's REMEMBER emit gets silently dropped by the producer's overflow check and `drained` stays at 0. Both substeps land in sequence: S3.B1.1 fixes the consumer, then S3.C exercises it end-to-end.

## 9. References

- KL gate addendum: cipher-fusion-evidence `ef8b830` `WEEK_14_STEP_3_C_KL_GATE_ADDENDUM.md`
- D1.3 KL probe: `/tmp/step13_3_baseline/d1_kl_probe_result.json`
- Substrate v1 commitment: `CIPHER_REENGINEERING_PLAN.md:103`
- W14 Step 2 G operational close: `WEEK_14_STEP_2_KOOPMAN_TIER.md:135-143` (commit `bf4f0ad`)
- Scope-lock Step 3 (preserved): `WEEK_13_14_SCOPE_LOCK.md:109-138` (commit `0dd86d3`)
- S3.B1.1 dependency: `WEEK_14_STEP_3_B1_1_MULTI_SLOT_DRAIN.md`
- Harness: `cipher_rt_phase4/test_step3_c_lmhead_validate.py` (canonical) + `test_step3_c_lmhead_validate.cpp` (synthetic smoke)
