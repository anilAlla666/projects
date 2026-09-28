# READ-ONLY VERIFICATION — does a BUILT, graph-captured fused/megakernel exist in the tree?

**Date:** 2026-05-31. **Type:** READ-ONLY fact-find (no build, no fix, no commit, anchors unchanged). Fed by
a 6-agent parallel source inventory + my own primary reads + on-pod test runs. Every claim file:line.
**Bottom line: the SOURCE exists (cited) but NONE of it reproduces WORKING on this pod, and even per its own
2026-04-28 records it was never the production substrate.** Qualifications, each load-bearing: (a) all
fused/graph/megakernel source is in the **may13 archive, NOT the production runtime** (`cipher_rt_phase4`
has only graph-*inspect*); (b) the working path was Python **monkey-patch + CIPHER-owned `torch.cuda.graph`,
NOT the transparent substrate**; (c) **NOT a true persistent/cooperative megakernel** (the only grid.sync
kernel is an fp8 *quant* kernel); (d) it does **NOT** solve the vLLM graph-bypass that killed Marlin (CIPHER
owns the capture, doesn't inject into vLLM's graph); and (e) on this pod the `.so` won't load without a
libcuda RTLD_GLOBAL preload (`undefined symbol cuGreenCtxDestroy` — a real load-order quirk).

> **CORRECTION (2026-05-31, same day — supersedes §3's "build drift / `<unk>` garbage / non-functional"):**
> that was a **misdiagnosis**. The `<unk>` garbage + megakernel "quantize failed" were a **missing enablement
> env**, not build drift: `cipher_substitute_v2_init` gates the NVRTC path on `getenv("CIPHER_SUBSTITUTE_V2")`
> (`cipher_substitute_v2.cpp:78`); unset → kernels no-op → the wrapper returned an uninitialized buffer →
> garbage. With **`CIPHER_SUBSTITUTE_V2=1`** the fused RMSNorm/SiLU kernels compile (sm_90) and are **correct**
> (teacher-forced argmax 100%, KL ~5e-6) and deliver a real, graph-integrity-proven **~+9–12% graph-mode tok/s**
> at B=8–32 — see `V0_FUSION_LEVER_SIZING.md`. The "+14.5%" was a confounded *harness* number (corrected to
> ~9–12% there). The fusion kernels are **functional on cu13**. (The two qualifications that stand: it's
> archive-only + Python-monkey-patch, not the transparent production substrate; and capturing it in vLLM is a
> per-framework build, not a wire.)

---

## 0. THE ANSWER TO ANIL'S RECOLLECTION

"We already built kernel fusion / megakernels AND captured CUDA graphs successfully" → **TRUE, in
`cipher-may13-evidence` (op31-prod, 2026-04-28, cu13)**, but with four load-bearing qualifications that
decide whether it's the G3 batched-decode unlock:
1. **Not in production.** `cipher_rt_phase4` (current runtime) compiles **only** `cipher_rt_graph_inspect.o`
   (`Makefile:105`) — read-only graph *inspection*. grep for `cipher_fusion|cipher_graph.o|weight_compress|
   int4_silu_mul` in its Makefile = **0**; no fusion/graph init in `cipher_inject.c:cipher_v2_init_body`;
   **zero** `grid.sync|cooperative_groups|megakernel` source hits in the whole tree. The fused/graph code is
   archive-only.
2. **Not transparent.** The working path is `run_mistral_fused.py:73-74` monkey-patching
   `MistralRMSNorm.forward`/`MistralMLP.forward` + `:154 with torch.cuda.graph(g)`. It requires replacing the
   model's module code — the opposite of the LD_PRELOAD/injection substrate.
3. **Not a true megakernel.** No persistent-grid / cooperative whole-block kernel exists. The "MLP
   megakernels" are multi-op fused **GEMV** kernels (one warp/column, register intermediates, single grid,
   no grid.sync), and the MLP is **split across two** of them (gate/up/silu, then down/residual) — not one
   block-spanning kernel. The only `cudaLaunchCooperativeKernel`+`grid.sync()` kernel in any repo is a
   **quant** kernel (`cipher_fp8_fused_quant.cu`), not a transformer megakernel.
4. **Does not solve the vLLM graph-bypass.** "Survives graph capture" works because CIPHER **owns** the
   capture (monkey-patch the model, then `torch.cuda.graph` captures its forward incl. CIPHER's kernels as
   nodes). It does **NOT** rewrite nodes into someone else's graph (that infra is INERT). In vLLM production
   vLLM owns the model + the graph, so the transparent intercept is still bypassed.

---

## 1. FUSED / MEGAKERNEL — built code (ALL in cipher-may13-evidence; NONE in cipher_rt_phase4)

| kernel / entry | file:line | fuses | kind | on-this-pod state |
|---|---|---|---|---|
| `cipher_int4_silu_mul_gemv` | `src/cipher_weight_compress.cpp:678` (host `:1151`, NVRTC compile `:1075`) | gate-INT4-GEMV + up-INT4-GEMV + SiLU + mul; register accumulators `acc_g/acc_u` never touch HBM (`:732`,`:742-744`) | multi-op fused GEMV (loosely "megakernel"; **not** persistent/cooperative) | **quantize FAILS on this pod** (build drift; `op31-prod-fix` absent) |
| `cipher_int4_down_residual_gemv` | `:758` (host `:1175`) | down-INT4-GEMV + residual add (`C=acc+r` `:804-806`) | multi-op fused; reads prior kernel's **global** output → MLP split across 2 kernels, **no full-block megakernel** | same (quantize path) |
| `cipher_rmsnorm_fp16` | `src/cipher_fusion_kernels.cpp:22` (host `:212`, NVRTC `:166`) | RMSNorm decomposition (one op) | op-fusion (NVRTC string) | **emits `<unk>` garbage on this pod** (see §3) |
| `cipher_silu_mul_fp16` | `:73` (host `:231`) | silu(gate)*up elementwise | op-fusion | **emits `<unk>` garbage on this pod** (see §3) |
| `cipher_residual_add_fp16` | `:92` (host `:248`) | a+b (single elementwise) | op-fusion (degenerate) | untested |
| `cipher_fp8_fused_quant.cu` | `:67` (`grid.sync()`, `cudaLaunchCooperativeKernel`) — *per the inventory sweep, not read by me line-by-line* | fp8 absmax→scale→quantize | **the ONLY true cooperative-grid kernel** in any repo — but a QUANT kernel | n/a (not on G3 path) |

- **NOT-FOUND anywhere** (source, all 4 repos): on-GPU interpreter, instruction queue, paged SMEM,
  hand-rolled software-barrier / `__threadfence`+atomic-spin cross-block sync, whole-transformer-block
  persistent megakernel.
- `cipher_fusion.cpp:41` `cipher_fusion_decide` is a **pattern matcher with hardcoded predicted_speedup
  floats** (`:27-30`); launches no kernel. `apply_recipe` (`cipher_dispatch.cpp:242`) does **not** launch
  fusion — case 2 returns false: "substitution happens at the Python wrapper level" (`:319-320`).

## 2. CUDA-GRAPH CAPTURE — built code + what works

| component | file:line | kind | state |
|---|---|---|---|
| may13 graph engine | `cipher-may13-evidence/src/cipher_graph.cpp:239` `cuStreamBeginCapture`, `:248` `cuGraphInstantiate`, `:277` `cuGraphLaunch` | **real capture+replay** (replays graph WHOLESALE) | archive-only; **NO node rewrite** |
| current-runtime graph | `cipher_rt_phase4/src/may13_intercept/cipher_graph_inspect.cpp` (`Makefile:105` `cipher_rt_graph_inspect.o`) | **inspect/observability only** | the only graph code in production |
| node-rewrite infra (`cuGraphExecKernelNodeSetParams`) | phase4 | **INERT** — never called | not built/used |

- `CIPHER_GRAPH_REPLAY` is a **phantom env flag** — no `getenv("CIPHER_GRAPH_REPLAY")` exists (the
  dossier's "replay opt-in behind CIPHER_GRAPH_REPLAY" was wrong; `CIPHER_GRAPH_REPLAYING` is an internal
  state enum, `cipher_graph.h:22`).
- **On THIS pod today:** plain `torch.cuda.graph` capture+replay WORKS — baseline Mistral-7B B=1 eager
  18.8 → **graph 114.2 tps (6.07×)** (`/tmp/run_mistral_fused_local.py`, no patches). CUDA-graph capture is
  functional on cu13.

## 3. THE INTERSECTION — does a graph-captured fused kernel RUN here? (tests run on-pod) → NO

Corrected after an advisor catch: my first pass reported the fusion path "+14.5% verified working." That was
**wrong** — I measured throughput but not OUTPUT. The numbers are an artifact of broken/no-op kernels.

- **Fusion path: produces GARBAGE on this pod.** Isolated greedy decode, Mistral-7B fp16
  (`fusion_graph_correctness.py`): bf16 reference = *"it more accessible to the masses."*; the monkey-patched
  fused RMSNorm/SiLU path = `<unk><unk><unk>…` (**0% token match**) in **both eager and graph**, and with
  **both** `stream=NULL` (the pre-fix `run_mistral_fused.py` default) **and** `stream=current` (the documented
  one-line capture-stream fix). So the "+14.5% graph" I first saw was a **faster-but-WRONG** result: the
  kernels silently no-op (NVRTC build drift — same family as the quantize failure) and the Python wrapper
  returns the uninitialized output buffer regardless of the C return code. The fused kernels do **not** run
  correctly here, let alone "survive graph capture."
- **The `.so` won't even load cleanly:** `undefined symbol: cuGreenCtxDestroy` unless `libcuda.so.1` is
  dlopened RTLD_GLOBAL first (build/link drift; `op31-prod-fix` absent).
- **INT4-megakernel path: also fails.** `tests/test_megakernel.py` (repointed) fails at
  `cipher_weight_compress_quantize` ("quantize failed").
- **What DOES work on this pod:** plain `torch.cuda.graph` capture+replay with NO CIPHER kernels — baseline
  Mistral-7B B=1 eager 18.8 → graph 114.2 tps (6.07×). CUDA-graph capture is functional on cu13; it's the
  CIPHER fused/megakernel `.so` that is build-drifted and non-functional here.
- **The recorded op31-prod numbers are NOT reproduced here.** The CLAUDE.md "Final corrected end-to-end table"
  **1.38× B=8 / 1.79× tok/W** (2026-04-28, fp16) came from the **Marlin linears in a captured graph + fusion**
  (CLAUDE.md ablation: INT4/Marlin-only graph 1.176×, RMSNorm 1.138×, all-on 1.388×; the captured graph had
  *"224 marlin nodes"*) — i.e. a **different** code path from the build-drifted `int4_silu_mul`/`down_residual`
  megakernels. Repairing those two megakernels would **not** by itself resurrect the 1.38×.

## 4. HONEST STATE (rigor classification)

| thing | classification | basis |
|---|---|---|
| Fused RMSNorm/SiLU kernels (op-fusion) | **BUILT (source); NOT-WORKING on this pod** (emits `<unk>` garbage; recorded-working op31-prod 2026-04-28) | source cited; 0% token match eager+graph here |
| INT4-fused GEMV "megakernels" (silu_mul, down_residual) | **BUILT (source); NOT-WORKING on this pod** (quantize fails; recorded-working op31-prod 2026-04-28) | source cited; quantize fails here |
| may13 graph capture+replay engine | **BUILT** (archive); replays wholesale, no node-rewrite | APIs cited; plain torch.cuda.graph works on cu13, but CIPHER `.so` build-drifted |
| `cipher_fusion.cpp` engine / `apply_recipe` | **BUILT-BUT-STUB** | pattern-matcher, hardcoded speedups, launches nothing; returns false |
| True persistent/cooperative transformer megakernel | **NOT-FOUND** | zero grid.sync/cooperative/interpreter source |
| Any of the above in cipher_rt_phase4 (production) | **NOT-FOUND** | Makefile=graph_inspect only; grep=0; no init wire |

## 5. THE G3 IMPLICATION (what this means for the batched-decode lever — facts only, Anil decides)

- The graph-bypass that killed Marlin is **NOT** transparently solved by this path. The may13 stack survives
  graph capture by **owning** it (model monkey-patch + `torch.cuda.graph`), not by injecting into vLLM's
  graph. To reach vLLM production you must either monkey-patch CIPHER kernels into vLLM's own modules (a
  real per-framework Python integration build) or build the node-rewrite into vLLM's captured graph (INERT /
  Phase-4-deferred). Neither is "just wiring."
- The lever it provides is the **fusion** (RMSNorm/SiLU = non-GEMM, ~17% of decode time per CLAUDE.md
  Amdahl table), **not** the compute-bound GEMM that batched decode is bottlenecked on
  (`V0_G3_MARLIN_VLLM_BATCHED_DECODE_MEASUREMENT.md`). The INT4-GEMM part is the build-drifted megakernel.
- So: the graph-captured fused/megakernel SOURCE **exists** (cited) — but it is an **archive Python-monkey-
  patch + owned-graph stack** that **does not run correctly on this pod** (build-drifted) and was never a
  production-substrate drop-in. "Already built, just wire it" is **not** accurate: it is neither wired, nor
  transparent, nor currently functional here, and the lever it targets is the non-GEMM fusion, not the
  compute-bound GEMM that batched decode is bottlenecked on. Anil decides among **builds** (none is a wire):
  (a) port + repair the fusion kernels into cipher_rt_phase4 on cu13 AND add a vLLM-module monkey-patch
  integration; (b) repair the INT4-fused-GEMV megakernels on cu13 — but note this alone would **not**
  resurrect the recorded 1.38× (that was Marlin-linears-in-graph + fusion, fp16, op31-prod — a different
  path); (c) build the graph-node-rewrite to inject CIPHER kernels into vLLM's own captured graph
  (Phase-4-deferred, INERT today). No code changed here; this is a fact-find.
