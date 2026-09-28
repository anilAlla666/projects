# R.A PATH-1 BUILD — detector inside a real vLLM cudagraph (cudagraph ON): WALL-WITH-MECHANISM

**Date:** 2026-06-09. **Model:** mistralai/Mistral-7B-v0.1, vLLM 0.20.2 / torch 2.11, **cudagraph ON** (not
enforce_eager). **Injector:** scratch `LD_PRELOAD` shim (`ra_path1/build_shim.so` + `path1_kernels.cubin`), NOT the
anchor, NOT a vLLM source edit. **Discipline:** anchor `2edba0d2…` frozen (md5 entry==exit); no edit to
`ra_e2e/`/`step_b_e2e/`/`ra_keystone/`/`ra_realvllm/`/`ra_fork2/`; all new work in `ra_path1/`; clock 1980 → `-rgc`;
vLLM children reaped. **Evidence:** `ra_path1/{probe_shim.c, probe_v3.c, path1_kernels.cu+.cubin, build_shim.c,
probe2.txt (the 64-match log), p1c.txt/p1sk.txt/p1k1.txt/p1ms.txt/p1b.txt (the wall logs), 16, 48 (build-side
instrumented:1/steps:0)}`. NOTE: `probe_result.json` reads all-zeros and `build_result.json` was never written — both
are per-process flush artifacts (the parent vLLM process's destructor overwrote/owned the file while the child
EngineCore that did the work crashed before its flush); cite the `.txt` logs and `16`/`48`, not those JSONs.

---

## VERDICT: WALL-WITH-MECHANISM (valid DONE per the mandate)
**Part A (GEMM-output-pointer recovery) is FEASIBLE — but Part B (hosting the check inside vLLM's real cudagraph)
WALLS:** adding *any* node (even a trivial memset) to vLLM's owned decode `cudaGraph_t` at the
`cudaGraphInstantiateWithFlags` seam crashes vLLM's **init-time `profile_cudagraph_memory`** with
`cudaErrorInvalidValue`. The node *adds* successfully (`rc=0`) — the seam is interposable, as Track B's toy-graph demo
showed — but vLLM's own graph lifecycle (it captures, instantiates, and re-processes its graphs during startup memory
profiling) does **not tolerate an external structural mutation it did not make**. The mandate's "do not fake recovery"
bar is therefore honestly **not cleared**: the check never executes in replay and no injected fault is caught
cudagraph-on, because vLLM crashes during init before any inference replay. Per the mandate, hosting that needs vLLM
source cooperation is OUT of scope → **the shipping host remains eager (+51% tax) from the real-vLLM run.**

---

## STEP-0 PRE-REG (paper, before building)
A device check node added after the target GEMM runs **every replay** (a static graph replays in full); "every N" is
enforced host-side (read/act on the device flag every N) — so the *in-graph* cost is paid per step, amortized only if
the kernel early-exits on non-N steps via a device counter. Pre-registered cudagraph-frame marginal for one ABFT
row-sum check node ≈ tiny (one reduction over C + x per checked GEMM); for the full detector (all 128 linear GEMMs/
forward) ≈ the per-step ABFT cost, which the real-vLLM run estimated at ~6% (the number this run set out to measure).
**This number was NOT measured — the host walls before any replay (see Part B).**

## PART A — GEMM-output-pointer recovery: FEASIBLE (located; not proven-by-catch, because the host walls)
- **Discovery:** vLLM fp16 linear = `cublasGemmEx`; the decode graph instantiate carries **417 nodes / 385 kernel
  nodes** (`probe2.txt:83`, instantiate#67).
- **Located:** scanning each kernel node's `*kernelParams[0]` (first 8 bytes, a safe read) against the
  interception-logged output pointers found **64 nodes whose first arg == a logged GEMM output C** (`probe2.txt:75-83`,
  e.g. node#5 qkv m=6144 cap=1). So the GEMM output pointer **is** locatable in the captured node params (corroborating
  the cleaner method actually used in the build: the shim logs the live C/x/W at capture, and torch's frozen capture
  buffers make those the exact replay addresses — no param extraction needed). **Caveat:** that probe run itself
  **stack-smashed and terminated immediately after printing the 64 matches** (`probe2.txt:84`
  `*** stack smashing detected ***`) — the matches are valid (printed pre-crash) but the run did not complete; the
  crash was my own bug (a `cudaPointerAttributes` write overran a 64-byte stack buffer, `probe_shim.c`), not a vLLM issue.
  `probe_result.json` reads all-zeros (parent-process flush overwrote the child's data) and must not be cited for this.
- **Research (cited):** `cudaGraphKernelNodeGetParams` gives `kernelParams` (void**) with **no discoverable count**,
  and cuBLAS passes operands in one opaque by-value struct (Hopper TMA blobs) — so blind window-scanning is unsafe.
  The shipped, version-proof method is **capture-time logging of the live C pointer** (frozen buffers), not post-hoc
  struct parsing. (Failure modes actually observed while getting here, disclosed for accuracy: the first probe died on
  `cudaErrorInvalidDeviceFunction` in flash_attn `probe_err.txt:76`; the param-parse probe **stack-smashed** as above —
  neither was a clean segfault.)
- **Proof obligation NOT met:** the mandate requires proving the recovered buffer is the GEMM output by catching an
  injected fault via the in-graph check. That check never ran (Part B wall), so this is reported as **located, not
  catch-proven** — no recovery is claimed beyond pointer location.

## PART B — in-graph check node: WALL-WITH-MECHANISM
- **The node adds:** `cuGraphAddKernelNode` (ABFT check) and `cudaGraphAddMemsetNode` (trivial primitive) both return
  **`rc=0`** into the real 417-node decode graph, with leaf dependencies — the interposition seam accepts the mutation
  (`addCheck rc=0` in p1c/p1sk/p1k1/p1b.txt:9; `addMemset rc=0` in p1ms.txt:9). The surviving build outputs `16`/`48`
  show `instrumented:1, steps:0` — the node was added but **zero replay steps executed** (vLLM crashed pre-replay).
- **The wall (robust, multi-config):** vLLM then crashes in `gpu_model_runner.py:5992 profile_cudagraph_memory` with
  `torch.AcceleratorError: CUDA error: invalid argument` (`cudaErrorInvalidValue`). Confirmed across **every** variant:
  ABFT check node; **trivial memset node** (gap6's primitive that worked standalone); with and without the out-of-graph
  `colsum_W` launch; and deferring instrumentation past the first/second/third big (profiling) graphs (`RV_SKIPBIG`).
  The crash is the **mutation itself**, not the kernel and not the out-of-graph launch.
- **Mechanism:** vLLM owns its decode `cudaGraph_t` and **re-processes it during init-time cudagraph memory
  profiling** (`determine_available_memory` → `profile_cudagraph_memory`, captures/replays/accounts the graph to size
  the KV cache). An externally-injected node changes the graph vLLM is tracking; vLLM's own subsequent graph operation
  rejects the structurally-altered graph it didn't author → `cudaErrorInvalidValue`. Track B's primitive works when the
  graph's **owner does not independently re-process it** (a standalone `torch.cuda.graph` we control); it does **not**
  survive when vLLM owns and re-processes the graph during startup. Hosting the check cudagraph-on therefore requires
  **vLLM source cooperation** (vLLM adding the check node at its own capture, or a first-class capture hook) — which is
  OUT of scope per the substrate line.

## WHAT WAS / WASN'T MEASURED
| DONE item | result |
|---|---|
| (1) check node injected at the seam | ✅ adds (`rc=0`), both ABFT and memset, into the real 417-node decode graph |
| (1) executes in replay + recovery catch-proven | ❌ WALL — vLLM crashes at init before any replay |
| (2) cudagraph-frame detector marginal (N=45/best-N) | ❌ NOT measured (can't host) — remains the ~6% estimate |
| (3) injected SDC caught ≤N + 0-FP, cudagraph-on | ❌ WALL (no replay) — *(eager-hosted catch + 0-FP were proven in the real-vLLM run)* |
| (4) fork-1 md5 entry==exit | ✅ anchor `2edba0d2` + all 7 ra_e2e files byte-identical |

## FORK-1 INTEGRITY (md5 entry==exit)
anchor `2edba0d2136f8ede4713d90a8f7cd55f`; ra_e2e: common `8276b08`, run_e2e `2de5398`, calib `ade20e2`,
trace `7d55b91`, summary `e5ac91b`, throughput `8131a2b`, detect `cefa606` — all unchanged. No vLLM source edit.
Clock reset `-rgc` (idle 375 / max 1980); no procs left.

## HONEST BOTTOM LINE
**R.A is NOT deployable inside cudagraph-on vLLM via substrate-legal external graph-node injection.** It does **not**
wall on the part everyone feared (the GEMM-output pointer is locatable, and the node adds cleanly) — it walls one layer
deeper: **vLLM re-processes its own decode cudagraph during init-time memory profiling, and an externally-mutated graph
fails that profiling (`cudaErrorInvalidValue`), regardless of node type.** The cudagraph-preserving host therefore
requires vLLM to add the check node *itself* (a vLLM plugin/hook at its capture site) — beyond the scratch-injector
substrate line. **Absent that, the shipping host stays eager: +51% eager tax with a ~3% detector marginal** (measured
in the real-vLLM run); the cudagraph-frame ~6% marginal remains unmeasured because it cannot be hosted externally.
The eager fallback detector (fires, catches persistent SDC ≤N, 0 FP) remains validated; this run closes the question of
whether the eager tax can be removed by external in-graph injection: **no — not without vLLM source cooperation.**
