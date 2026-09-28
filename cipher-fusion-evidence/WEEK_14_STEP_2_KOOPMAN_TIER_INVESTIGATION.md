# W14 Step 2 Koopman tier — investigation closeout (ζ-deep day 3)

**Date:** 2026-05-23
**Adjudication:** 2026-05-23 (ζ-deep day 3, 3-day box)
**Status:** **Step 2 G NOT closed**. Two substantive bug fixes landed; remaining
            architectural finding surfaced for Anil re-adjudication.
**Surface for:** v1 Koopman compute substitution at top-1 ≥ 99% gate — substrate
                  cannot deliver this with current investigation scope.

## 1. Summary

ζ-deep day 3 investigation (per Anil 2026-05-23 adjudication) characterized the
failure path of S2.E top-1 token match gate (≥ 99%, workload-grounded), found
and fixed TWO substrate bugs, and surfaced a third architectural concern that
remains unresolved within the 3-day investigation box.

Bugs found + fixed during day 3:

| # | Bug                                 | Fix commit               | Verified |
|---|-------------------------------------|--------------------------|----------|
| H2| rand_svd MGS loss-of-orthogonality on near-linearly-dependent oversample columns produces σ_i inflated by √2 and energy_X > 1.0 | `3590f4d` tag `week-14-step-2-h2-cgs2-fix` | yes (Python ground-truth + cipher post-fix energy ≤ 1) |
| GPU prep | cipher_edmd_live.cpp allocates VT_buf/KOP_buf/W_buf via CPU malloc; cipher_koopman_fp16_register_shape and downstream kernels read these as GPU pointers — undefined behavior | `42b05e9` tag `week-14-step-2-zeta-day3-gpu-upload` | yes (direct kernel diagnostic D1: rel_err 2e-4 with proper GPU pointers) |

S2.E gate status with both fixes applied:

| Configuration              | Engine fires | top-1 match | KL mean | residual_ratio |
|----------------------------|--------------|-------------|---------|----------------|
| Pre-investigation baseline | yes (98%)    | 0.00%       | 14.58   | 0.836          |
| +H2 CGS2 only              | yes (98%)    | 0.00%       | 14.78   | 1.36           |
| +H2 +GPU-upload (all)      | yes (98%)    | **0.00%**   | 14.94   | (not measured) |
| +H2 +GPU-upload (LM-head only) | yes (100%) | **5.90%** | 7.96    | (not measured) |
| v1 gate target             |              | **≥ 99%**   |         |                |

**Top-1 token match remains catastrophically below the v1 gate** despite two
real bug fixes landing. The substrate's actual quality ceiling on TinyLlama LM
head appears to be far below 99%.

## 2. Day 3 step-by-step findings

### 2.1 Step 1: H2 CGS2 fix (commit `3590f4d`)

See `WEEK_14_STEP_2_H2_CGS2_FIX.md` for full diagnosis. Summary:
- Bug: `cipher_rs::mgs_qr_thin` single-pass MGS loses orthogonality when
  oversample columns are near-linearly-dependent (||Y^T·Y - I||_F = 2.45)
- Fix: apply MGS twice ("CGS2 / twice is enough", Giraud-Langou-Rozloznik 2005)
- Post-fix: ||Y^T·Y - I||_F = 6.6e-14, σ values match TRUE SVD exactly

### 2.2 Step 2: Direct kernel diagnostic (test harness)

Test design per Anil 2026-05-23 adjudication:
- Python ctypes harness loads libcipher_rt.so
- Hand-crafted V_T (identity-padded), K_op (identity), W_buf (known linear
  pattern), and x (known content) — all uploaded to GPU via PyTorch
- Direct call to `cipher_koopman_fp16_register_shape` then
  `cipher_koopman_fp16_launch_shape`
- Compare cipher's output against Python formula evaluation

Test file: `/tmp/step13_2_baseline/zeta_kernel_test.py`

**Outcome D1: aggregate rel_err = 1.99e-4** (fp16 precision noise floor).
Kernel arithmetic implements `out = x · V_T^T · K_op · W_buf` correctly.

### 2.3 Step 2 follow-up: GPU pointer upload fix (commit `42b05e9`)

The direct kernel test passed because matrices were uploaded to GPU via
`torch.from_numpy(...).cuda()`. cipher_edmd_live's fit_and_register_locked
allocated matrices via CPU `malloc` and passed CPU pointers to
`cipher_koopman_fp16_register_shape`, which:
1. Stores the pointers as-is in `s_shapes[].V_T/K_op/W`
2. Launches `fp32_to_fp16_kernel<<<>>>` on the CPU pointers (undefined)
3. The downstream `cipher_koopman_fp16_decode_generic` kernel reads them
   as device pointers

Fix: introduce CPU staging buffers (VT_cpu, KOP_cpu, W_cpu) for the
computation step, allocate GPU buffers via cudaMalloc, copy with
cudaMemcpy, and free CPU staging.

### 2.4 Persistent failure mode (D3-equivalent architectural finding)

After both H2 and GPU-upload fixes landed, the S2.E workload-grounded gate
still fails by orders of magnitude:

**Negative Pearson correlation (-0.29) between vanilla and cipher logits.**
Cipher's output is not just noise — it's anti-correlated with vanilla. Signs
of a systematic transformation error not captured by the rank-r approximation
math.

**LM-head-only post-fix reports energy_X = 3.03** (mathematically impossible
for rank-r approximation; should be ≤ 1.0). The H2 fix is CONFIRMED in the
binary (md5 4dad976243...) yet this number recurs under LM-head-only
restriction. Suggests a second, distinct numerical issue in the SVD pipeline
that activates under specific calibration data distributions.

**FFN-shapes register BEFORE LM head** in the natural calibration order:
1. K=2048 N=256
2. K=2048 N=2048
3. K=5632 N=2048 (FFN down)
4. K=2048 N=5632 (FFN gate)
5. K=2048 N=32000 (LM head — registers LAST)

Once FFN shapes register, FFN substitution starts firing, perturbing the
hidden state activations that flow into the LM head GEMM. The LM head
calibration samples accumulated during this period are corrupted by upstream
Koopman substitution. The H2 + GPU-upload fixes do not address this
calibration-order cascade.

## 3. Substrate state at day-3 close

| Tree | Anchor | Status |
|------|--------|--------|
| `cipher_rt_phase4` HEAD | `42b05e9` tag `week-14-step-2-zeta-day3-gpu-upload` | NEW (this work) |
| H2 landmark | `3590f4d` tag `week-14-step-2-h2-cgs2-fix` | already committed |
| Port-landmark | `ce4c1b8` tag `week-14-step-2-port-landmark` | unchanged |
| `cipher_kmod` | `week-9-complete` (0.6.5) | UNCHANGED |
| `cipher_vllm_kv.py` | `2b6cedab` | UNCHANGED |
| `libcipher_rt.so` md5 | `4dad976243e4717b966b5652b0f6b5dd` | NEW |

**Step 2 G NOT closed** (no `week-14-step-2-koopman-tier` tag). The two real
bug fixes are committed; the workload-grounded gate is not satisfied.

## 4. Remaining work scope estimate

Continuing investigation beyond day 3 to deliver v1 Koopman at top-1 ≥ 99%
likely requires:

| Hypothesis | Remaining work | Risk |
|------------|----------------|------|
| Calibration ordering: FFN registers first → LM head Y_calib corrupted | Re-architect cipher_edmd_live to either freeze substitution during multi-shape calibration OR re-calibrate all shapes against an FFN-disabled pass | ~2-4 days; touches concurrency model in cipher_edmd_live |
| LM-head-only energy_X = 3.03 anomaly: second distinct SVD bug | Debug rand_svd on the specific data distribution; possibly need to replace with Householder QR | ~1-2 days; touches cipher_randsvd.h further |
| Negative Pearson correlation root cause | Direct logit-level comparison; trace Y_calib values at capture time vs at SVD-input time; check kernel output value range vs vanilla | ~1 day; instrumentation-heavy |
| Y_calib timing (cudaMemcpyAsync vs PyTorch stream sync) | Wrap cipher_edmd_live_collect's gpu_capture_async to use the GEMM's stream (not stream 0) | ~0.5 day; small change in cipher_edmd_live.cpp |
| Fundamental: rank-r approximation may not deliver top-1 ≥ 99% on real LM heads | Re-architect Koopman or escalate scope to attention-only / smaller-N-only | architectural decision |

**Total estimate: 4-8 more eng-days for diagnostic + fix. Risk: even with
all bugs fixed, top-1 ≥ 99% may not be achievable for rank-64 approximation
of full-rank LM head weights.**

## 5. Anil re-adjudication surface

Per day-3 close decision tree (Anil 2026-05-23):

> If Step 2 fails or surfaces D3 architectural finding:
>   - H2 landmark remains committed [DONE: `3590f4d`]
>   - Step 2 does NOT close S2.G [confirmed: no `week-14-step-2-koopman-tier` tag]
>   - Step doc WEEK_14_STEP_2_KOOPMAN_TIER_INVESTIGATION.md documents... [this doc]
>   - Surface to Anil for re-adjudication

Options for Anil:
- **(a)** Extend Step 2 budget by 5 days for deep kernel + calibration work
- **(b)** Defer remaining bug to v1.5; ship v1 with H2 + GPU-upload fixes +
        Koopman substrate inactive at runtime (CIPHER_KOOPMAN default off
        already; no behavioral change vs pre-W14 substrate for default users)
- **(c)** Re-scope v1 narrow-domain Koopman to a subset the substrate can
        deliver at top-1 ≥ 99% (e.g., attention QKV only, smaller K and N)
- **(d)** Other path Anil identifies

## 6. Files modified during ζ-deep day 3

- `cipher_rt_phase4/include/may13/cipher_randsvd.h` (H2 CGS2 fix, 13+1)
- `cipher_rt_phase4/src/may13/cipher_edmd_live.cpp` (GPU upload fix, 38+10)
- `/tmp/step13_2_baseline/zeta_kernel_test.py` (direct kernel diagnostic harness)
- `/tmp/step13_2_baseline/h2_randsvd_clone.py` (Python clone of cipher's algorithm)
- `/tmp/step13_2_baseline/h2_verify.py` (Python ground-truth SVD comparison)
- `/tmp/step13_2_baseline/h2_lmhead_xhost.bin` (cipher's dumped LM head X_host)
- `cipher-fusion-evidence/WEEK_14_STEP_2_H2_CGS2_FIX.md` (H2 closeout doc)
- `cipher-fusion-evidence/WEEK_14_STEP_2_KOOPMAN_TIER_INVESTIGATION.md` (this doc)

Tasks unchanged: S2.E remains in_progress per investigation result.
