# G3 — Marlin INT4 on vLLM batched DECODE (B≥8): the single yes/no measurement

**Date:** 2026-05-31. **Type:** MEASUREMENT (not a close). **No build, no `.so` change, no commit, no tag,
no rotation.** Staging `.so` = `7cdc68ef` (unchanged); deployed `/usr/lib/cipher` = `1f305ce6` (unchanged).
Marlin's bf16 substitution path (`cipher_rt_marlin_actuator.c:182-253`) was flipped ON via env
`CIPHER_MARLIN=on`; no actuator code written. Arming = `CUDA_INJECTION64_PATH=$SO` (driver cuInit hook →
`InitializeInjection2` → GOT patch, incl. the vLLM V1 EngineCore worker subprocess). All GEMMs eager
(`enforce_eager=True`).

---

## 0. THE YES/NO (the whole point)

- **Does Marlin SUBSTITUTE at B≥8?** **YES.** 97% of decode GEMMs handled, on both torch-HF and the real
  vLLM V1 worker, all via `cublasGemmEx` (no cuBLASLt routing gap).
- **Does it deliver a tok/s or tok/W LIFT?** **NO — it REGRESSES, and the regression worsens with batch.**
  −11%→−18%→−28% (torch Mistral-7B B=8/32/64); −32%→−53% (vLLM TinyLlama B=8/32). tok/W tracks tok/s down.
- **KL=0 / output-correct?** **FAILS the hard stop.** Teacher-forced argmax-agreement 88.5% (TinyLlama) /
  93.75% (Mistral-7B) — not 100%. Free-running vLLM generation: 19.9% token-match, 0/8 sequences identical.

**Verdict = the user's "handled>0 but NO lift" branch, compounded by a KL=0 hard-stop failure.** Marlin
INT4 substitutes cleanly on the cross-tenant batched-decode path, but the regime does **not pay**. The
prior "win" evidence is a **per-GEMM fp16 microbench** (`cipher-may13-evidence/BUILD_STATE.md:536-542`:
isolated kernel µs, 1.09×/q-o · 1.58×/k-v · 1.21×/gate-up · 1.55×/down "vs cuBLAS fp16") carrying an
**unvalidated assertion** (`:544` "4× weight bandwidth reduction translating to net tok/s gain"). There is
**no prior end-to-end tok/s measurement** to contradict — this measurement is the first to test that
assertion end-to-end, and it **does not hold**: end-to-end decode regresses. **This REFUTES the
`V0_VLLM_DECODE_ACTUATION_DIAGNOSIS.md` hypothesis that "the GEMM-substitution lift's home is batched
decode (B≥8)"** — batched decode is where it regresses *most*. INT4 RTN also perturbs the logits enough to
break greedy determinism (KL=0 fail). STOP for Anil — no fix attempted.

---

## 1. SUBSTITUTION — handled vs passthrough (counter-cited, `cipher_rt_matmul_dispatch.c` atexit totals)

| arm | calls | handled | passthrough | handled % | lt_shim | route |
|---|---|---|---|---|---|---|
| torch Mistral-7B B=8 | 30375 | **29702** | 673 | 97.8% | 0 | cublasGemmEx |
| torch TinyLlama B=8 | 31155 | **21980** | 9175 | 70.5% | 0 | cublasGemmEx |
| **vLLM TinyLlama B=8** | 11836 | **11484** | 352 | **97.0%** | 0 | cublasGemmEx (V1 worker) |
| **vLLM TinyLlama B=32** | 11836 | **11484** | 352 | 97.0% | 0 | cublasGemmEx (V1 worker) |

- `marlin_bf16_substituted` == `matmul_handled` on every arm (the bf16 path at
  `cipher_rt_marlin_actuator.c:189-253` is what fires; fp16 path untouched).
- **`lt_shim_calls = 0` everywhere ⇒ NO cuBLASLt-routing gap.** vLLM batched decode issues `cublasGemmEx`,
  which reaches the dispatch. The "D10 LT-route" item is moot for this path (`CIPHER_LT_ROUTE` left OFF, as
  scoped — no speculative wiring).
- Per-shape substitution map (`CIPHER_MARLIN_VERBOSE=1`, TinyLlama): lm_head (K2048/N32000), attn q/o
  (2048/2048), FFN gate/up (2048/5632), FFN down (5632/2048). Passthrough = prefill/profiling GEMMs (gated
  by M>64) + GQA k/v (N=256 < 1024) + embeddings.

## 2. THE M-DIMENSION — confirmed batched (M = batch), three independent ways

The actuator's gate is `bf16_M = call->n` (the batch/N-of-C dim), gated `1 ≤ M ≤ 64`
(`cipher_rt_marlin_actuator.c:200-208`).
1. **Batch-invariant GEMM count (the decisive one):** vLLM `handled` = **11484 at B=8 AND 11484 at B=32**.
   The count is identical because batch is the **N-dimension of each decode GEMM**, not a multiplier on the
   number of GEMMs — the defining signature of batched decode. ⇒ at B=8 each handled GEMM has N=8; at B=32,
   N=32. Both ≤64 → substituted.
2. **Gate logic + prefill exclusion:** prefill/profiling GEMMs were koopman-observed at N=104 / N=16384
   (`[CIPHER KOOPMAN-BF16-OBS] n=16384`, `cipher_rt_koopman_engine.cpp:161`) → M>64 → gated out into the 352
   passthrough. Only decode GEMMs (N=batch ≤64) are in the 11484 handled.
3. **Architecture:** vLLM continuous batching merges B concurrent decode steps into one forward; nn.Linear
   rows = num_tokens = B → cuBLAS N = B = Marlin's M. Confirmed B=8 and B=32.

## 3. THE LIFT — matched OFF/ON pairs (same workload, substrate present in both arms)

| regime | OFF tok/s | ON tok/s | Δ tok/s | Δ tok/s/W |
|---|---|---|---|---|
| torch Mistral-7B B=8 | 68.4 | 61.1 | **−10.6%** | −3.1% |
| torch Mistral-7B B=32 | 290.7 | 238.8 | **−17.8%** | −11.5% |
| torch Mistral-7B B=64 | 585.8 | 424.9 | **−27.5%** | −21.1% |
| torch TinyLlama B=8 | 87.0 | 76.6 | **−12.0%** | −11.4% |
| **vLLM TinyLlama B=8** | 430.7 | 292.0 | **−32.2%** | −25.0% |
| **vLLM TinyLlama B=32** | 1660.4 | 774.8 | **−53.3%** | −47.5% |

- **Monotonic with batch** in both frameworks → the opposite of the "B≥8 is the win regime" hypothesis.
- vLLM regresses harder than torch because vLLM's bf16 baseline is far more optimized (430 vs 87 tok/s at
  B=8 TinyLlama), so injecting per-GEMM INT4 dequant costs proportionally more.
- **Fingerprint isolation** (`CIPHER_MARLIN_GC_FP=0`, Mistral-7B): B=8 62.6 vs shipped-on 61.1 (vs OFF
  68.4); B=64 427.8 vs shipped-on 424.9 (vs OFF 585.8). **The GC fingerprint accounts for only ~1–3%; the
  loss is the substitution path itself, not the GC-on-free safety overhead.**
- **What is NOT isolated (honest boundary):** the regression is attributable to the **bf16 substitution
  path as implemented** — which is cast(bf16→fp16) → Marlin INT4 GEMM → cast(fp16→bf16), i.e. ~3 launches
  replacing cuBLAS's 1 (`cipher_rt_marlin_actuator.c:210-240`). I isolated the GC fingerprint but **not the
  bf16↔fp16 cast wrapper**, so this does **not** prove the bare INT4 kernel is intrinsically slower than
  cuBLAS — only that the end-to-end path *as shipped* loses. (Note: the may13 microbench above was the
  **native fp16** path — no cast wrapper — so it is not directly comparable to this bf16-cast path.)

**Physics (why batched decode is the wrong regime — and consistent with the batch trend):** Marlin INT4
saves *weight-read bandwidth* 4×. At B=1 the GEMM is weight-bandwidth-bound but launch/dequant overhead
dominates the tiny GEMM (known B=1 regression). As batch grows (B=8→64) each weight is reused M times → the
GEMM becomes **compute-bound**, the bandwidth saving becomes irrelevant, while the INT4 dequant + the
3-launch cast wrapper cost stays. That the regression **grows** with batch (not shrinks) is the tell: pure
launch overhead would shrink as a % of a larger GEMM, so the dominant cost is compute-bound, not fixed
overhead. cuBLAS bf16 tensor-core GEMM is already near-optimal for these shapes on H100. Net: no batch
regime where this Marlin path beats cuBLAS bf16 end-to-end on this substrate.

## 4. CORRECTNESS — KL=0 HARD STOP (Mem #11): FAILS

- **Teacher-forced** (clean: identical input tokens to both arms, per-decode-step logit comparison —
  isolates the per-step INT4 perturbation, no cascade):
  - TinyLlama B=8: argmax-agreement **88.54%** (88 flips / 768 positions), KL(ref‖marlin) mean 0.054, max 0.287.
  - Mistral-7B B=8: argmax-agreement **93.75%** (32 flips / 512), KL mean 0.021, max 0.189.
  - Neither is 100% ⇒ **FAIL the KL=0 hard stop.** INT4 RTN on lm_head (K2048/N32000) perturbs logits enough
    to flip greedy argmax. (Reported as a fail — NOT waved through via the CP 5.3 "benign divergence"
    precedent; that is Anil's call, not the substrate's.)
- **Free-running** (the user's literal "batched output == OFF ref"), vLLM B=8: per-token match **19.9%**,
  **0/8 sequences identical**, first divergence as early as token 1 — the cascade of the per-step flips.

## 5. THE PRODUCTION-PATH CAVEAT (first-class finding, not a footnote)

This entire measurement is **eager** (`enforce_eager=True`). It is mandatory: production vLLM captures decode
into **CUDA graphs** by default, and **graph replay bypasses the live `cublasGemmEx` GOT-patch** — Marlin
would substitute **0** there regardless of this result (graph node-rewrite is Phase-4-deferred / inspect-only
per `MFU_MECHANISM_DOSSIER_2026-05-30.md §1`). So the eager A/B validly answers *"does Marlin help when the
substrate is live"* (no — it regresses), but a hypothetical positive must **not** be read as *"delivers in
production vLLM"*: in graph mode the intercept never fires.

## 6. SCOPE ADHERENCE

- No build / no fix (handled>0 but regresses → reported, stopped — per the prompt's "do not speculatively
  fix" branch). `CIPHER_LT_ROUTE` left OFF (no speculative LT wiring; lt_shim=0 made it moot anyway).
- Anchors unchanged (md5 re-verified post-run, no rebuild): staging `.so` =
  `7cdc68ef902a887e67ce58ed4ee5a881`, deployed `/usr/lib/cipher` = `1f305ce61e9acae65920dc8fa9474ee1`
  (mtime 05:05, untouched), marlin-gc `0e2e853`. No tag, no rotation.
- **Not covered:** vLLM × Mistral-7B (EngineCore init fails on *both* arms — the known env-block, not
  Marlin; OFF arm fails identically). torch-Mistral-7B (regime) + vLLM-TinyLlama (product path) bracket it.

## 7. WHAT ANIL DECIDES NEXT (this is a measurement, not a recommendation)

The number is: **Marlin INT4 substitutes ~97% on vLLM batched decode but regresses tok/s −32% (B=8) →
−53% (B=32) and fails KL=0.** Options for adjudication: (a) accept that G3 GEMM-substitution via Marlin
INT4 does **not** deliver on batched decode (the lift is neither at B=1 nor B≥8 end-to-end) and retire the
"B≥8 POOL is the home" framing; (b) pursue a different GEMM lever for the compute-bound batched regime —
FP8 (already wired, large-M forward 64-68% MFU) or CUTLASS persistent GEMM (NOT BUILT) — where the win is
compute-throughput, not weight-bandwidth; (c) treat the substrate-bypass-under-CUDA-graphs as the prior
blocker for *any* in-decode GEMM actuator on production vLLM. No code changed; awaiting adjudication.
