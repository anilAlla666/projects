# TRAINING-MFU OPTION B — POINTWISE CEILING CONFIRMED + DISPATCH-BOUNDARY REACHABILITY (2026-06-10/11)

## VERDICT: **CONFIRMED + CLOSED.** The build's pointwise ceiling is real (re-measured 48.4 % vs the build's 49.1 %, within 0.7 pt) and **0 % of it is reachable from the substrate's as-built substitution surface** (source-grounded classification of every component — argued, not demonstrated) — every component is framework-level (opaque ATen elementwise kernels / recipe choices), and no pointwise-relevant symbol is on the as-built surface (`cublasGemmEx` + `cublasLtMatmul` for GEMM; the injector's other GOT patches — SDPA `at::_ops` ::call, vLLM flash-attn, machete, dlsym — do not touch pointwise; see B2 correction). The ceiling is **part recipe artifact, part intrinsic**: removing the fp32-adapter recipe (bf16 adapters) cuts pointwise 48.4→40.3 % and lifts MFU **24.7 %→34.1 % (+9.4 pts — ~3× FP8's entire projected ceiling; quality untested — fp32 master adapters are standard finetune practice, see B3 caveat)**, but that lever is a framework-side recipe change available to the customer, NOT to the substrate. Even leaned out, pointwise holds ~40 % — intrinsic to HF-eager training. **Single-tenant training MFU closes as out-of-moat for HF-eager LoRA finetune (Mistral-7B, B=2×2048, single-GPU); optimized/compiled or full-pretrain stacks (fused pointwise, higher GEMM share) are a different regime, NOT closed by this evidence.**

NO build, NO successor, NO actuator armed. Pure-torch profile (substrate_loaded:False proven in every JSON via the
build's RTLD_NOLOAD probe; `CUDA_INJECTION64_PATH` unset, recorded in the 3 profile JSONs; MFU JSONs carry the probe only). Frozen anchor md5
`2edba0d2136f8ede4713d90a8f7cd55f` entry == exit. Scratch only in `train_mfu_optB/`. Default app clocks (verified
app==default 1980); per-run SM clocks sampled — profile cells 1965–1980, MFU cells throttle under sw-power-cap
(see Integrity; conservative direction for the headline). MFU = 4·N analytic FLOPs ÷ 989.5 TFLOP/s (finetune model,
stated; not a pretrain proxy). Method = the build's `profile_train.py` (copied unedited as
`profile_train_ORIGINAL_COPY.py`), extended in `profile_itemized.py` (sub-buckets, op-level producers, probe).

---

## B1 — POINTWISE DECOMPOSITION CONFIRMED (build's exact config: fp16 base + fp32 LoRA adapters, B=2×2048, sdpa, AdamW)

Coarse buckets (`b1_itemized_fp16_fp32adapters.json`, 461 ms/step kernel time, clocks 1980):

| bucket | build (06-10) | OptB re-measure | abs ms/step |
|---|---|---|---|
| POINTWISE | 49.1 % | **48.4 %** | 223 |
| GEMM | 45.5 % | 45.9 % | 212 |
| ATTENTION | 4.1 % | 4.0 % | 18 |
| OTHER | 1.2 % | 1.3 % | 6 |
| OPTIMIZER | 0.0 % | 0.4 % | 2 |

**CONFIRMED within 0.7 pt.** One small correction: the build's bucketer missed `multi_tensor_apply` (AdamW foreach)
→ OPTIMIZER is 0.41 %, not 0.0 % (was inside POINTWISE/OTHER). A bigger correction to the build's *panel estimate*:
the fp32 adapter GEMMs (`sm80_xmma_gemm_f32f32_*`, CUDA-core FFMA, not tensor-core) are **≥15.8 % of step kernel
time** (10 `sm80_xmma` kernels summed; +0.18 % `cutlass_80_simt_sgemm` also fp32-CUDA-core; sub-0.1 % kernels
truncated from the list), not the "~7 pts of the GEMM bucket" the panel estimated — so the FP8-addressable
tensor-core GEMM share was only ≈ 30 %, even less than the build assumed.

**Itemized pointwise (the deciding table; named kernels in the JSON, % of total step kernel time):**

| component | % | named kernels (top) | producer (op-level evidence) |
|---|---|---|---|
| PW_CAST_COPY | **16.6** | `direct_copy_kernel_cuda` 8.6, `float16_copy_kernel_cuda` 7.7 | `aten::_to_copy` 1,473×/step, 75.4 ms/step + `ToCopyBackward0` 574×/step — peft fp32-adapter boundary casts (x→fp32 in, out→fp16 back, and their backward) + HF intrinsic upcasts (RMSNorm fp32, CE `.float()`) |
| PW_ADD_MUL_VECTORIZED_SAMEDTYPE | **16.8** | fp32 `AUnaryFunctor` mul 6.9, fp16 `CUDAFunctor_add` 3.8, fp16/fp32 `BinaryFunctor` 4.2, `pow_tensor_scalar` 0.6 | `aten::mul` 1,381×/step 72.6 ms + `aten::add(_)` residuals + RMSNorm variance/scale + rope rotations + grad accumulation — HF-eager unfused elementwise |
| PW_ADD_MUL_SAMEDTYPE (nocast unrolled) | 6.5 | `gpu_kernel_impl_nocast` add/binary | same producers, non-vectorizable layouts |
| PW_MIXED_DTYPE_ADD_MUL | **5.2** | `CUDAFunctor_add<float>` with casting load/store | fp32 LoRA output meeting fp16 base residual — pure fp32-adapter artifact |
| PW_ACTIVATION | 1.3 | `silu_kernel` 0.5 / `silu_backward` 0.8 | SwiGLU act |
| PW_FILL_CAT | 1.0 | `CatArrayBatchedCopy` | rope cos/sin cat, bookkeeping |
| PW_COPY_SAMEDTYPE | 1.0 | `direct_copy (nocast)` | contiguity copies |

(Optimizer correction, panel: the AdamW-foreach 0.41 % sits in the coarse OPTIMIZER bucket (`buckets_pct`,
classifier-token-attributed — `multi_tensor_apply` itself falls below the 0.1 % per-kernel listing threshold and is
not named in the JSON), NOT in a `PW_OPTIMIZER_FOREACH` pointwise sub-bucket; the itemized pointwise rows above sum
to the 48.4 % POINTWISE bucket exactly without it.)

(Classifier note, disclosed: a first pass substring-matched PyTorch's `gpu_kernel_impl_nocast` into the cast bucket
— "nocast" contains "cast" — inflating PW_CAST_COPY to 29.3 %; fixed before any conclusion was drawn, mixed-dtype
`LoadWithCast` adds split into their own line. The fix is in `profile_itemized.py` with the ordering comment.
Panel-added disclosure: `profile_itemized.py` also diverges from the original bucketer in three definitional ways —
adds ProfilerActivity.CPU, excludes Memcpy/Memset/`empty` from the kernel-time total (the original's token list
counted them into POINTWISE), and filters op rows differently — so the 0.7 pt build-vs-OptB delta includes
bucketer-definition drift, not only run-to-run noise; the CONFIRMED verdict is unaffected at this magnitude.)

## B2 — REACHABILITY CLASSIFICATION: every component is FRAMEWORK-LEVEL; 0 % reachable on the as-built surface

**The substrate's substitution surface, source-grounded (CORRECTED by panel — original prose was wrong):** the
GEMM-substitution surface is **`cublasGemmEx` and `cublasLtMatmul`** (`cipher_rt_cublas_shim.c:325-335`;
`cublasGetStream_v2` is dlsym-resolved for stream tracking, not patched; the Lt intercept as-built is
passthrough+telemetry — narrower still). That is NOT the entire surface: `cipher_inject.c` additionally arms GOT
patches for three ATen SDPA symbols `at::_ops::_scaled_dot_product_{flash,efficient,cudnn}_attention::call`
(trampolines receive full `at::Tensor` arguments), six vLLM flash-attn patterns, `machete::mm_dispatch`, and
dlsym/dlvsym. **None of those touch the pointwise bucket** (the cudnn-SDPA patch covers this workload's ATTENTION
bucket, 4.0 %), so the 0 % conclusion survives — but the original "entire surface" premise was false, and the
categorical "tensor semantics do not exist at the boundary" wall below is restated as a composite (see the
correction after the component list). The CUPTI observation plane can *see* other launches but their arguments
are opaque.

**Why every pointwise component is off that surface (per-component):**
1. **All add/mul/norm/rope backward-elementwise kernels (≈23.3 % = vectorized 16.8 + nocast 6.5)** — ATen `TensorIterator` kernels launched via
   `cudaLaunchKernel` with template-specialized argument structs (no stable ABI; dtype/layout baked into the
   template). Substituting them at the CUDA boundary requires tensor semantics that do not exist there. This is
   the **same wall V.1 already established** for non-GEMM driver substitution ("opaque ATen args") — re-confirmed
   here, it applies to the *entire* training pointwise bucket. FRAMEWORK-LEVEL (the fix is torch.compile /
   fused-kernel / fused-AdamW recipe choices — framework tools).
2. **Casts (16.6 %)** — also ATen kernels (same opaque-args wall), with a second, independent wall: **dispatch
   ordering + buffer ownership**. The cast executes *upstream* of the GEMM (peft casts `x` to fp32 *before*
   calling `mm`), so by the time the substrate's intercepted GEMM call arrives, the fp32 buffers already exist —
   there is nothing left to fold. Folding the cast away requires the framework to stop allocating/launching it,
   i.e. framework cooperation. Conversely, a cast *after* a GEMM writes a framework-owned buffer the framework
   reads back — the substrate cannot change that buffer's dtype contract. FRAMEWORK-LEVEL.
3. **Mixed-dtype residual adds (5.2 %)** — would require an Lt epilogue expressing "add my output in fp32 to a
   fp16 tensor and store fp16" *and* suppressing the framework's own subsequent add kernel — suppression of an
   opaque ATen launch is semantically blind at the boundary and not substrate-legal. (And cuBLASLt epilogues
   cannot express the RMSNorm/rope/SwiGLU chains at all; standard epilogues are bias/GELU-class.) FRAMEWORK-LEVEL.
4. **Activation/fill/cat/same-dtype copies/optimizer (≈3.7 % = 1.3+1.0+1.0+0.4)** — same opaque-args wall; fused
   alternatives exist *inside* torch (fused AdamW, flash SwiGLU) — recipe choices. FRAMEWORK-LEVEL.

(Partition note, panel: the four components now tile to 48.8 % = POINTWISE 48.41 + OPTIMIZER 0.41 with no overlap;
the original prose double-counted activation+optimizer in components 1 and 4 and omitted the 1.0 % same-dtype
copies.)

**Why the demonstrated `at::_ops` GOT mechanism (SDPA) does not rescue pointwise (panel-prompted composite,
replaces the categorical wall):** the substrate's own attention dispatch proves tensor-semantics interception IS
possible at the ATen-op boundary, so the wall is not "tensor semantics do not exist at the boundary". The accurate
composite: (a) no pointwise-relevant symbol is on the **as-built** surface — extending registration to e.g.
`at::_ops::mul_Tensor::call` / `_to_copy::call` would be a new build; (b) per-op substitution of memory-bound
elementwise kernels gains ~nothing — each kernel is already at the bandwidth floor; the cost is the UNFUSED CHAIN,
and cross-op fusion cannot be expressed as a per-symbol swap (it requires suppressing the framework's subsequent
launches = the same suppression/buffer-ownership wall as component 3); (c) the casts' dispatch-ordering/
buffer-ownership wall (component 2) is mechanism-independent. (b) and (c) are argued, not built-and-measured.

**Reachable fraction of the pointwise ceiling on the as-built surface: 0 %** (source-grounded classification —
argued, not demonstrated; no component survives the composite walls above). The GEMM-intercept lever cannot move
the training pointwise ceiling at all.

## B3 — ARTIFACT-VS-INTRINSIC: part recipe artifact (big), intrinsic remainder still ~40 %

Lean cell = bf16 base + **bf16 adapters** (`b3_lean_bf16_nativeadapters.json`). Gotcha caught and disclosed: peft's
`get_peft_model` creates LoRA weights **fp32 even on a bf16 base** — the first lean attempt silently kept fp32
adapters (`adapter_param_dtypes` recorded in-JSON; discarded as `b3_DISCARDED_fp32adapters_notlean.json`); the
valid cell force-casts adapters to bf16.

**Single-variable control (panel-prompted — the "discarded" cell is exactly the base-dtype-only ablation):**
bf16-base + fp32-adapters re-measures ≈ the build recipe — pointwise **47.9 %** (vs 48.41), PW_CAST_COPY 16.48 %
(vs 16.62), PW_MIXED_DTYPE 5.2 % (identical), kernel 467.5 vs 461 ms/step — so the base fp16→bf16 change alone is
**null** and the entire B3 effect is the **adapter dtype**, resolving the two-variables-at-once confound in the
lean comparison. (No MFU was separately measured for that arm — kernel-time evidence only. Script provenance: that
cell came from the pre-fix script version, since overwritten by the archived `profile_itemized.py` which
force-casts native adapters; the JSON is kept and labeled.)

| metric | build recipe (fp16+fp32 adapters) | lean (bf16+bf16) | meaning |
|---|---|---|---|
| POINTWISE share | 48.4 % | **40.3 %** | recipe artifact removed, intrinsic remains |
| PW_CAST_COPY | 16.6 % | **4.0 %** | surviving 4 % = HF-intrinsic upcasts (RMSNorm fp32, CE float; `_to_copy` 1,473→359×/step, 75→12 ms) |
| PW_MIXED_DTYPE_ADD_MUL | 5.2 % | **0 %** | pure fp32-adapter artifact, gone |
| fp32 FFMA adapter GEMMs | 15.8 % | ~0 (tensor-core bf16) | GEMM bucket 212→170 ms/step |
| kernel ms/step | 461 | 329 | |
| **MFU (unprofiled, `train_step_optB.py` = build harness + dtype flags)** | **0.2470** (replicates build 0.2464 ✓) | **0.3406** | **+9.4 MFU pts from a recipe change** |
| step s (median) | 0.4854 | 0.3521 | −27.5 % |
| stability (proxy, 50-step MFU runs) | 0.032→0.003, 0 NaN | 0.038→0.004, 0 NaN | both stable on the memorization proxy (28-step profile runs also stable) |

**Reading:** the build's 49 % was **recipe-inflated** (fp32 adapters: casts + mixed-dtype adds + CUDA-core FFMA
GEMMs), but the ceiling **survives the lean harness at ~40 %** — intrinsic to HF-eager training (residuals,
RMSNorm fp32 upcast+mul, rope, SwiGLU, unfused backward elementwise). So the finding generalizes with structure
**at this config** (Mistral-7B, B=2×2048, sdpa, AdamW): the three measured dtype-recipe cells span pointwise
40–48 %; extension to other batches/models/attention impls is inference, not measurement (the build's own B=1×512
cell is a different, launch-bound regime). The reachability conclusion (0 % as-built) is structural and
recipe-independent. **Numerics caveat (stated, not tested):** fp32 master adapters are standard practice for
finetune quality; the lean cell is performance-share evidence + proxy-stable (50-step MFU runs + 28-step profile
runs, fixed batch), not a quality qualification of bf16 adapters on real data. At lean MFU 34.1 % with GEMM share
51.8 %, GEMM-time MFU ≈ 0.66–0.70 (0.66 = MFU ÷ wall-applied profiled GEMM share; 0.70 = FLOPs ÷ profiled GEMM
kernel time × peak) — the practical HF-eager headroom above the lean point is mostly non-GEMM.

## B4 — VERDICT (closes single-tenant training MFU)

- **CONFIRMED** (with two small corrections): the training-MFU ceiling IS the pointwise bucket — re-measured
  48.4 % vs build's 49.1 %; corrections: OPTIMIZER 0.41 % not 0.0 % (bucketer token), and the fp32-adapter
  CUDA-core GEMM share is ≥15.8 % of step time (supersedes the panel's ~7 pt estimate; FP8-addressable ≈ 30 %).
- **REACHABILITY: 0 % on the as-built surface** (argued classification, not a demonstration). GEMM-substitution
  surface = `cublasGemmEx`+`cublasLtMatmul`; the injector's further GOT patches (SDPA `at::_ops`, flash-attn,
  machete, dlsym) are not pointwise-relevant; every pointwise component fails the composite walls (no symbol
  as-built; per-op swap can't fuse cross-op chains at the bandwidth floor; dispatch-ordering/buffer-ownership).
- **GENERALIZES, with structure, at this config:** recipe artifact (fp32 adapters) accounted for ~8 pts of
  pointwise share and −9.4 MFU pts (single-variable control: base-dtype change alone is null); the remaining
  ~40 % pointwise is intrinsic to HF-eager training and survives the lean harness.

**THE CLOSING SENTENCE:** Single-tenant training-step MFU on the dispatch-boundary position is **ceiling-limited
by framework-level pointwise (40–48 % of step kernel time, 0 % reachable on the as-built surface)**; the
GEMM-intercept lever **cannot** move it; this **closes single-tenant training MFU as out-of-moat for the measured
regime — HF-eager LoRA finetune (Mistral-7B, B=2×2048, single-GPU)** — the largest measured lever (bf16 adapters:
24.7 %→34.1 % MFU, +9.4 pts; quality untested, fp32 master adapters are standard practice) is a customer-side
recipe change, and the next levers (fused elementwise/optimizer, torch.compile) are framework tools, not substrate
actuators. Optimized/compiled or full-pretrain stacks (fused pointwise, higher GEMM share) are a different regime,
not closed by this evidence.

## Integrity
- Frozen anchor md5 `2edba0d2136f8ede4713d90a8f7cd55f` entry == exit (asserted at entry, re-verified at exit and
  by panel — matches; no in-dir artifact records it, adopt in-harness md5 capture next time); no build/successor/
  actuator; pure torch proven per-run (`substrate_probe: substrate_loaded:false` via RTLD_NOLOAD in **all 5**
  JSONs; `injection_env: unset` recorded in the 3 profile JSONs — the MFU harness records the probe only, which is
  sufficient). Originals (`train_mfu/`, fork-1) cited, not edited; `profile_train.py` copied unedited
  (`profile_train_ORIGINAL_COPY.py`, md5-identical to fork-1); `train_step_optB.py` was NOT preceded by an
  unedited copy — its provenance is the diff vs `train_mfu/train_step.py` (dtype/recipe flags only,
  panel-verified), which keeps the 0.2470-vs-0.2464 replication comparison valid.
- Default APP clocks both cells (app==default 1980, verified at exit). Per-run sampled SM clocks (corrected by
  panel — original prose said "1965–1980 sampled both cells", which is wrong for the lean MFU run): profile cells
  1980/1965; build MFU run 1935–1980 med 1965; **lean MFU run 1710–1965 med 1920 under sw-power-cap at 685 W**
  (build cell 578 W). Conservative direction: at equal clocks the lean cell's 0.3406 MFU and the +9.4 pt headline
  would only grow; profiled shares are clock-insensitive ratios. Profiled-cell shares compared profiler-on vs
  profiler-on (apples-to-apples); MFU numbers from unprofiled runs only. Build baseline independently replicated
  (0.2470 vs 0.2464).
- Every number → a `train_mfu_optB/` JSON: `b1_itemized_fp16_fp32adapters.json`, `b3_lean_bf16_nativeadapters.json`,
  `b3_DISCARDED_fp32adapters_notlean.json` (kept, labeled), `mfu_build_recipe.json`, `mfu_lean_recipe.json`.
  No retracted-number reuse; 85 % appears only as the closed target.
- Self-caught and disclosed: classifier "nocast"-substring bug (fixed before conclusions); peft fp32-adapter
  default (first lean cell discarded, not papered).

## Verification panel (2026-06-11, post-hoc close-out)

4-lens adversarial panel (numbers traceability / logic+methodology / honesty+overclaim / discipline+integrity),
each material finding then independently attacked by a skeptic agent (9 agents total). Full findings + skeptic
verdicts: `panel_findings_full.json`. **28 findings: 5 raised material → 4 confirmed (ALL FIXED above), 1
downgraded (duplicate of the surface finding), 0 refuted; 23 minor (all worthwhile ones applied).**

**Confirmed material (all applied in-place, marked "panel" at each site):**
1. **[logic] "Entire substitution surface" premise FALSE per substrate source** — `cipher_inject.c` additionally
   GOT-patches three SDPA `at::_ops::…::call` symbols (full `at::Tensor` args), six vLLM flash-attn patterns,
   `machete::mm_dispatch`, dlsym/dlvsym. None pointwise-relevant, so 0 % survives — but the categorical
   "no tensor semantics at the boundary" wall was replaced with the composite (as-built surface / per-op-swap-
   can't-fuse-chains / dispatch-ordering), and "0 %" is now labeled argued-not-demonstrated.
2. **[honesty] Closing verdict scope** — "closes single-tenant training MFU as out-of-moat" now restated to the
   measured regime (HF-eager LoRA finetune, Mistral-7B, B=2×2048, single-GPU) at both verdict sites; optimized/
   compiled/pretrain stacks explicitly named as a different regime not closed.
3. **[honesty] bf16-adapter quality caveat** was present only in B3 but absent at both pitch sites — added to the
   VERDICT line and the closing sentence.
4. **[discipline] Integrity clock claim "1965–1980 sampled" false for the lean MFU run** (recorded 1710–1965 med
   1920, sw-power-cap, 685 W) — corrected with per-run ranges + conservative-direction note.

**Notable minors applied:** B2 components made a partition (were double-counting 1.7 pt, omitting 1.0 pt);
optimizer row moved out of the itemized-pointwise table (provenance was misstated); fp32 adapter GEMM share
"15.8 %" → "≥15.8 %" (11th SIMT kernel + truncation); GEMM-time MFU "≈0.66" → "0.66–0.70" (two defensible
derivations); discarded cell promoted to the **single-variable control** it actually is (base-dtype change null —
strengthens B3's attribution); bucketer-definition drift inside the 0.7 pt "CONFIRMED" delta disclosed;
integrity-provenance wording scoped to what artifacts actually record (no train_step.py copy; injection_env in
3/5 JSONs; md5/app-clock asserted-not-archived); stability rows cited to the 50-step runs; generalization band
scoped to this config. **Numbers lens: every headline number reproduces bit-exactly from the JSONs (MFU recomputed
from 4·N·tokens/(step·989.5e12), N=7,241,732,096: 0.24704/0.34056, +9.35 pt).**

Process note for next time: archive the pre-fix script version alongside any discarded cell; record entry/exit
anchor md5 + app-clock query as artifacts, not prose.
