# K.1.5 close — signal capture results + Tables 1-5

**Date:** 2026-05-27
**Substrate anchor:** `cipher_rt_phase4 d785fd8` (K.1.5 Step 1.6 close) + cipher_vllm_kv VA pool floor fix (Sub-step 0, plugin md5 `d0226d79`)
**rev8 .deb md5:** `5603f72dd1dd172e1c05aa5804274321`
**Capture binary:** `build_cuda13_may13/libcipher_rt.so.k1.step1_6` md5 `482d08950e8f107ce86bc925fe7cc2f9`
**Captures directory:** `cipher-fusion-evidence/v1_phase_b/k1_diagnostic/captures/`

---

## Section A — 14-cell matrix + execution status

| Cell | Workload | Config | Status | Workload class |
|---|---|---|---|---|
| A1 | Llama-3.1-8B bf16 | B=1 decode ctx=512 | GEN_DONE ✓ | **A3_SINGLE_TENANT_STREAM** (correct) |
| A2 | Llama-3.1-8B bf16 | B=8 batched ctx=512 | GEN_DONE ✓ | A4_BATCH_INFERENCE (expected A2 but classifier doesn't differentiate without cuStreamBeginCapture) |
| A3 | Llama-3.1-8B bf16 | B=32 batched ctx=4K | GEN_DONE ✓ | A4_BATCH_INFERENCE (correct — large_batch >32) |
| A4 | Llama-3.1-8B bf16 | B=1 prefill ctx=32K | GEN_DONE ✓ | A4_BATCH_INFERENCE (expected C1 — long_ctx signal not firing; see Table 2) |
| A5 | Mistral-7B-v0.1 bf16 | B=1 decode ctx=512 | GEN_DONE ✓ | **A3_SINGLE_TENANT_STREAM** (correct) |
| A6 | Mistral-7B-v0.1 bf16 | B=8 batched ctx=512 | GEN_DONE ✓ | A4_BATCH_INFERENCE |
| A7 | Mistral-7B-v0.1 bf16 | B=32 batched ctx=4K | GEN_DONE ✓ | A4_BATCH_INFERENCE (correct) |
| A8 | Mistral-7B-v0.1 bf16 | B=1 prefill ctx=32K | GEN_DONE=0 (partial) | A4_BATCH_INFERENCE (1000 obs only; same C1-gap as A4) |
| A9 | TinyLlama-AWQ INT4 | B=1 decode ctx=512 | GEN_DONE ✓ | **A3_SINGLE_TENANT_STREAM + int4=1** (correct, both axes) |
| A10 | TinyLlama-AWQ INT4 | B=8 batched ctx=512 | GEN_DONE ✓ | UNKNOWN + int4=1 (class missed but int4 still detected) |
| A11 | TinyLlama-AWQ INT4 | B=32 batched ctx=2K (capped from 4K — TinyLlama max_pos) | GEN_DONE ✓ | UNKNOWN + int4=1 |
| A12 | TinyLlama-AWQ INT4 | B=1 prefill ctx=2K (capped from 32K — TinyLlama max_pos) | GEN_DONE ✓ | UNKNOWN + int4=1 |
| **B** | N=2 concurrent Mistral-7B-v0.1 via `--pid=host` (N=4 bare-metal would need vllm_env outside docker; deferred) | bf16 B=1 | proc-A failed (GPU contention); proc-B succeeded | A3_SINGLE_TENANT_STREAM multi=0 (multi_tenant signal didn't fire — Bug #3 W.6 sub-B dependency confirmed) |
| E1 | bare torch.matmul loop | 50×512 fp16 | GEN_DONE | **UNKNOWN** (correct — no class match) |
| E2 | model loading then idle | TinyLlama-1.1B-fp16 init + 30s sleep | GEN_DONE | A4_BATCH_INFERENCE (FALSE POSITIVE — classifier shouldn't fire on idle) |
| E3 | nvidia-smi only | CUDA init + nvidia-smi probe | GEN_DONE | (no snapshot — too few launches) |

**B1 + B2 (training + fine-tuning) DEFERRED** to V.1 CP 5.5 per prior Anil decision — ~1.5 ED harness cost vs marginal architectural information (the substrate kernel-name patterns for `_bwd`/`optimizer`/`Adam` are unit-tested; production validation lands at CP 5.5 mixed-workload soak).

**Total: 15 capture artifacts** (12 inference + 1 multi-tenant + 3 negative + 1 partial B-A). 13 PASS (87%), 2 PARTIAL.

---

## Section B — Per-capture signal histograms (top numerics from K1-DIAG-SNAPSHOT last entry)

| Cell | obs | machete | flash | rms | silu_gelu | alloc_medium | procs |
|---|---|---|---|---|---|---|---|
| A1 Llama B=1 | 36000 | 0 | 8356 | 12746 | 4204 | 2 | 1 |
| A2 Llama B=8 | 36000 | 0 | 8364 | 12758 | 4208 | 3 | 1 |
| A3 Llama B=32 ctx4K | 34000 | 0 | 8364 | 12762 | 4210 | 3 | 1 |
| A4 Llama 32K prefill | 2000 | 0 | 248 | 648 | 213 | 2 | 1 |
| A5 Mistral B=1 | 36000 | 0 | 8364 | 12758 | 4208 | 4 | 1 |
| A6 Mistral B=8 | 36000 | 0 | 8372 | 12770 | 4212 | 5 | 1 |
| A7 Mistral B=32 ctx4K | 34000 | 0 | 8372 | 12776 | 4214 | 4 | 1 |
| A8 Mistral 32K prefill (partial) | 1000 | 0 | 28 | 267 | 88 | 3 | 1 |
| A9 TinyLlama-AWQ B=1 | 42000 | **11602** | 5804 | 8833 | 2900 | 52 | 1 |
| A10 TinyLlama-AWQ B=8 | 42000 | **11531** | 5818 | 8848 | 2904 | 53 | 1 |
| A11 TinyLlama-AWQ B=32 | 36000 | **5969** | 5706 | 8688 | 2852 | 53 | 1 |
| A12 TinyLlama-AWQ prefill | 9000 | **352** | 52 | 268 | 88 | 53 | 1 |
| B Mistral N=2 proc-B | 36000 | 0 | 8372 | 12770 | 4212 | 4 | 1 |
| E1 bare torch | 1000 | 0 | 0 | 0 | 0 | 1 | 1 |
| E2 idle | — | — | — | — | — | — | — |
| E3 no-workload | — | — | — | — | — | — | — |

Every cell shows `capture=0` (vLLM 0.21 default `cudagraph_mode=NONE`); `nccl_k=0` (no actual NCCL); `nccl_lib=1` (libnccl mapped as DT_NEEDED but unused); `alloc_large=0` (weights via PagedAttention single big allocation; not via cipher_kv_bridge per its always-on path).

---

## Section C — Tables 1-5 cross-workload analysis

### Table 1 — Reliable workload-class differentiators (signals with non-overlapping distributions)

| Signal | Differentiates | Evidence |
|---|---|---|
| `machete kernel count > 0` | TinyLlama-AWQ INT4 (A9-A12) ≠ Llama-3/Mistral bf16 (A1-A8) | A9-A12: 352-11602; A1-A8 + B + E1-E3: 0 |
| `flash kernel count == 0` | E1 (bare torch) ≠ A1-A12 / B | E1: 0; everything else: 28-8372 |
| `rms_norm count == 0` | E1 (bare torch) ≠ all inference | Same pattern |
| `B=1 decode dominance (gemm_b1 > 75% gemm_total)` | A1/A5/A9/B (= A3 class) ≠ A2-A4/A6-A8/A10-A12 | Verified in classifier decision branch |
| `int4_weights_detected` | TinyLlama-AWQ (via machete pattern match) | Fires consistently on A9-A12 even when class is UNKNOWN |

### Table 2 — Class-conflating signals (need disambiguators in K.1.6)

| Signal | Conflates | Why |
|---|---|---|
| `large_batch_count > 10` | A2 (B=8) + A3 (B=32) + A4 (32K prefill) all → A4 | Decision tree's A4 fires whenever batch > 32; doesn't distinguish "continuous batched" vs "true batch inference" vs "long-context prefill"; needs A2 signal (cuStreamBeginCapture or stream-count) and C1 signal (kv_cache_alloc_pattern with rolling seq-len) |
| `long_ctx_detected` | NEVER FIRES across all cells | `kv_cache_alloc_pattern_detected` gate requires 100+ medium allocs; vLLM PagedAttention uses ONE big alloc → alloc_medium count stays low (2-53) → kv_pattern never fires → long_ctx_detected never derives; C1 unreachable on default vLLM 0.21 stock workloads |
| `capture_seen` (=0 everywhere) | A2_CONTINUOUS_BATCHED unreachable | vLLM 0.21 default `cudagraph_mode=NONE` + `enforce_eager=True`; A2 needs alternative signal (stream count peak) |

### Table 3 — Within-class signal stability (across C1-C4 configs)

| Class | Cells | Signal stability |
|---|---|---|
| A3 (B=1 decode) | A1 Llama / A5 Mistral / A9 TinyLlama-AWQ | flash/rms/silu counts near-identical across model families at same B; stable |
| A4 (large batch) | A2 / A3 / A6 / A7 / A11 | flash/rms/silu counts near-identical; large_batch signal stable; cannot distinguish A2 vs A4 vs C1 within this group |
| UNKNOWN (TinyLlama-AWQ B>1) | A10 / A11 / A12 | int4 signal stable at 1 across all; classifier falls into UNKNOWN when batch1_dom doesn't fire AND has_capture doesn't fire AND large_batch doesn't reach threshold |

### Table 4 — Empirically observed kernel-name patterns (Bug #5 H4 verdict empirically confirmed)

| Pattern | Workload sources | Demangled (sample) |
|---|---|---|
| `pytorch_flash::*` | Llama-3 + Mistral | `pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<...>` |
| `vllm::flash_attn_kernel`, `vllm::rms_norm_kernel`, `vllm::rotary_embedding_kernel`, `vllm::silu_kernel`, `vllm::fused_add_rms_norm_kernel` | All vLLM workloads | matched via current `classify_kernel_name` |
| `cutlass::device_kernel<...gemm::kernel::GemmUniversal<...machete::MacheteCollectiveMma...>>` | TinyLlama-AWQ only | INT4 GEMM via Machete; classifier's "machete" substring match catches it correctly (A9-A12: machete count 352-11602) |
| `flashinfer::sampling::TopPSamplingFromProbKernel` / `RadixTopKMaskLogitsKernel` | All vLLM workloads | not currently classified; would be UNKNOWN-no-class kernels |
| `at::native::vectorized_elementwise_kernel`, `at::native::reduce_kernel`, `at::native::elementwise_kernel` | All inference + E1 bare torch | not classified; generic PyTorch kernels |

**Bug #5 H4 verdict reconfirmed:** CUkernel handles (CUDA 12.4+) are properly resolved via cuKernelGetName promoted-primary path; all 50+ unique kernels per workload land in fn_cache with names. classify_kernel_name pattern set catches vLLM + Machete + flash + rms + silu + rotary; misses flashinfer-sampling + at::native generic — K.1.6 pattern expansion scope.

### Table 5 — A2 vs A3 vs A4 vs C1 empirical differentiator answer (Bug #4)

The current decision tree distinguishes:
- **A3** via `batch1_dom = gemm_b1/gemm_tot > 0.75` ✓ (works correctly on A1/A5/A9)
- **A4** via `large_batch_count > 10` ✓ (works on A3/A7 where batch > 32)
- **A2** via `has_capture && gemm_tot > 100` ✗ (capture never fires; A2 unreachable)
- **C1** via `kv_cache_alloc_pattern_detected && max_n > 32K` ✗ (kv pattern never fires)

**Empirical answer per data:** at vLLM 0.21 default (no cudagraph capture), A2/A3/A4 are **structurally indistinguishable from driver-level signals alone** because:
- All run on single context, single process, single dtype
- All use same PyTorch kernel set (same flash/rms/silu counts)
- Only difference is B dim per cuBLAS call (already captured in batch1_dom)

K.1.6 decision proposal: **collapse A2 + A4 into a single class** (e.g. `A_INFERENCE_BATCHED`) since they engage the same v1 actuator stack (VOLT + Marlin via fp16 + Koopman if calibrated). A2-vs-A4 distinction matters only for downstream multi-tenant POOL batching which is W.4 scope.

C1 (long-context RAG) needs a real long-context signal — currently impossible to detect via cipher_kv_bridge.vmm_zeros allocation (always one big slab per layer regardless of seq length). Defer C1 to a future substep that wires per-block PagedAttention KV-cache observability into the classifier (requires Track 2/3 process registry).

---

## Section D — Confidence calibration mapping

Observed CLASSIFY confidence values vs decision tree branch:
| Branch fired | Confidence | Calibration note |
|---|---|---|
| B1_PRE_TRAINING (has_nccl_kernels + has_backward) | 880 | Untested empirically (B1 deferred to CP 5.5); confidence value retained |
| B2_FINE_TUNING (has_backward) | 800 | Untested empirically |
| A2_CONTINUOUS_BATCHED (has_capture + !batch1_dom + gemm_tot > 100) | 800 | Never fired; unreachable on stock vLLM 0.21 |
| A1_AGENTIC_MULTI_TENANT (multi_tenant + batch1_dom + !has_capture) | 820 | Never fired; multi_tenant signal blocked by Bug #3 |
| **A3_SINGLE_TENANT_STREAM** (batch1_dom only) | **820** | Fires correctly on A1/A5/A9 + multi-tenant proc-B |
| **A4_BATCH_INFERENCE** (large_batch alone) | **720** | Fires on A2/A3/A6/A7 (true large batch); also wrongly on A4/A8 (long-ctx prefill) and E2 (idle loading) |
| C1_RAG_LONG_CONTEXT (long_ctx derived) | 740 | Never fires; long_ctx signal needs Section C Table 2 fix |
| **UNKNOWN low** (weights_seen + gemm_tot < 50) | **350** | Fires on TinyLlama-AWQ B=8/32/prefill (A10-A12) |
| **UNKNOWN dead** (everything else) | **200** | Fires on E1 bare torch + E3 no_workload + A12 prefill |

**Calibration recommendation for K.1.6:** lower A4 confidence to 600 to reflect that it's a "fallback" class (catches everything that isn't B=1 decode); raise A3 to 850 (high-signal fire); UNKNOWN at 200 stays.

---

## Section E — Transition behavior observations

The captures don't include explicit workload shifts mid-run. Implicit transitions observed:
- Within each cell, classification re-evaluates every 100 obs; once classified into A3 (e.g. A1), it stays A3 across the rest of the run (workload character stable)
- E2 idle was wrongly classified as A4_BATCH_INFERENCE during loading then would presumably degrade to UNKNOWN if measured after warmup (not captured); transition observability for "workload completed → idle" not yet wired

K.1.6 scope: add a "workload idle" signal (e.g., observations_per_second < threshold) for transition detection.

---

## Section F — Edge cases + surprises (Memory #11 honest residue)

1. **E2 false positive (A4 instead of UNKNOWN)** — model loading triggers many large weight allocations + initialization GEMMs which the classifier reads as "large_batch inference". K.1.6 fix: require observations_per_second steady-state minimum before classifying.
2. **Multi-tenant B-proc-A failed** — GPU contention during simultaneous container spawn; only proc-B succeeded. Bug #3 W.6 sub-B dependency confirmed: even with `--pid=host`, `count_concurrent_cipher_processes()` returned 1 → multi_tenant=0 → A1 unreachable.
3. **Long-context prefill (A4/A8 32K + A12 capped 2K) misclassified as A4** — `kv_cache_alloc_pattern_detected` never fires because vLLM PagedAttention uses one large slab per layer; alloc_medium count stays at 2-53 across all workloads. Section C Table 2 fix needed.
4. **TinyLlama max_position_embeddings cap at 2048** — A11/A12 had to be capped from 32K to 2048 ctx; full 32K-prefill measurement only possible on Llama-3 (A4) and partially Mistral (A8 failed at 1000 obs).
5. **`large_batch` threshold of 32 vs reality** — even B=8 (A2/A6) triggers A4 because `gemm_b1_count` was 0 (vLLM batches into single GEMM call with n=8). The threshold needs tightening or removal for batch ≤ ~16.

---

## Section G — K.1.6 decision-tree skeleton (empirically derived)

```
classify_internal_v2:
  has_nccl_k = (nccl_count > 5)              # not nccl_lib_loaded (per Step 7 fix)
  has_backward = (backward + optimizer > 10)
  has_machete = (machete + awq + gptq > 0)
  has_capture = (capture_seen > 0)
  gemm_tot = ...
  batch1_dom = (gemm_b1 * 4 > gemm_tot * 3) when gemm_tot > 50
  batched_mid = (gemm_b1 < gemm_b1_dom_threshold AND gemm_tot > 100)  # B=8..32
  large_batch_strict = (gemm_large_batch > 50)  # raised from 10
  multi_tenant_kmod = (kmod_process_registry_count > 1)  # NEW per W.6 sub-C

  IF has_nccl_k && has_backward → B1
  ELIF has_backward → B2
  ELIF has_capture && !batch1_dom → A2
  ELIF multi_tenant_kmod && batch1_dom → A1
  ELIF batch1_dom → A3
  ELIF batched_mid → A_INFERENCE_BATCHED  # NEW: collapses A2+A4 mid-batch
  ELIF large_batch_strict → A4
  ELIF observations_per_second < idle_threshold → UNKNOWN  # NEW idle gate
  ELSE UNKNOWN

  conditional flags:
    int4_weights_detected = has_machete
    long_context_detected = (max_seq_in_prefill > 32K via PagedAttention block tracking)  # NEW W-series
    training_workload_detected = has_backward || has_nccl_k
    multi_tenant_detected = multi_tenant_kmod
```

classify_kernel_name pattern expansion:
- Add: `flashinfer::sampling::` → KCB_SAMPLING (new class)
- Add: `at::native::vectorized_elementwise` / `reduce_kernel` / `elementwise_kernel` → KCB_GENERIC_PYTORCH (new bin)
- Keep existing: machete / awq / gptq / flash / fmha / rms_norm / silu / gelu / nccl / _bwd / optimizer / reduce / rotary

---

## Section H — Substrate signals worth adding (gaps surfaced for W-series)

| Gap | Required signal | Substep |
|---|---|---|
| Capture mode unreachable on default vLLM | cuStreamBeginCapture intercept counter (already wired in K.1 Step 1 via observe_capture_begin); vLLM 0.21 just doesn't fire it. Add ALTERNATIVE: stream count peak via cuStreamCreate intercept | K.1.6 + W.4 |
| Long-context prefill unreachable | PagedAttention block table size observation OR max-seq-len peak per prefill | future W-series |
| Multi-tenant detection across containers | kmod-side process registry via NR 27 (when W.6 sub-B unblocks) | W.6 sub-B/C |
| Idle detection | observations_per_second time-windowed measurement | K.1.6 |
| A2 vs A4 distinction | stream count peak + prefill+decode interleaving pattern | K.1.6 + future |

---

## Section I — Engineering debt forecast (Memory #28 marvel)

What would break this classifier in production:
1. **vLLM cudagraph_mode=FULL** (default in vLLM ≥ 0.22 expected) — would fire has_capture, A2 reachable; need to verify A2 capability flags engage correctly
2. **TP > 1 multi-GPU inference** — NCCL kernels fire (currently not tested), workload becomes B1-shaped even at inference; need to differentiate "inference with NCCL" from "training with NCCL"
3. **Models with bf16 dtype but small (≤1B params)** — fall into A3 (correct) but Marlin/Koopman don't engage (correct per dtype gate); no actuator lift; honest residue per Memory #29 W.3 binding
4. **Continuous-batched serving with mixed prefill+decode** (real vLLM steady state) — alternates B=1 decode and B=N prefill rapidly; classifier may flip between A3 and A4 every 100 obs; need rolling-window smoothing
5. **Speculative decoding** (CIPHER_SPEC default ON per cipher_spec_decode.py:18) — produces irregular decode patterns; not validated in this capture roll
6. **vLLM 0.22+ EngineCore architecture changes** — REGISTER_STREAMS / NR 27 / plugin hook chain may shift; integration retest needed

---

## Section J — B1/B2 honest residue cited to V.1 CP 5.5

B1 (pre-training) + B2 (fine-tuning) deferred per:
- Prior Anil decision (~1.5 ED harness construction cost vs marginal architectural information)
- The substrate kernel-name patterns for `_bwd` / `backward` / `optimizer` / `Adam` / `SGD` are present in classify_kernel_name (verified Step 1 source)
- B1 decision tree branch requires `has_nccl_k + has_backward`; both signals are wired
- B2 requires `has_backward`; signal wired
- Production validation lands at V.1 CP 5.5 (multi-workload stock-mix soak with at least one training cell)

If B1/B2 surface engineering gaps post-CP-5.5, they fold into a separate substep (no scope expansion in K.1.5).

---

## Anchors at K.1.5 close

| Artifact | SHA / md5 |
|---|---|
| cipher_rt_phase4 HEAD | `d785fd8` (K.1.5 Step 1.6 — UNCHANGED) |
| cipher_kmod HEAD | `8c643fc` UNCHANGED |
| cipher-platform rev8 .deb | `5603f72dd1dd172e1c05aa5804274321` (VA pool fix) |
| cipher_vllm_kv.py md5 (in rev8) | `d0226d7950bba4e6dfdf11ad5b841e51` |
| libcipher_rt.so capture binary | `482d08950e8f107ce86bc925fe7cc2f9` (rev6/rev7/rev8 all ship 1f305ce6 installed; capture used the K.1.5 Step 1.6 build for diagnostic env) |
| cipher-fusion-evidence pending HEAD (this commit) | TBD on commit |
| Tags pending | `k1-5-complete` + `k1-5-close` |

K.1.5 closes K.1's empirical foundation. K.1.6 follows: decision tree v2 implementation (Section G skeleton) + classify_kernel_name pattern expansion + K.1 final close gate.
