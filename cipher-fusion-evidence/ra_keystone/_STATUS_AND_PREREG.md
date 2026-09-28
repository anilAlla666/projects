# R.A KEYSTONE + PRODUCT INTEGRATION — STATUS + PRE-REGISTERED VERDICTS
**Resume-from-cold file.** Mandate: build the checksum-fused WGMMA GEMM keystone (Track A), close
Gap-6 vLLM hosting (Track B), assemble R.A as a shipping feature (Track C). READ-ONLY on anchor
`2edba0d2136f8ede4713d90a8f7cd55f` (frozen, confirm at exit). One run; parallel only on NON-GPU work;
GPU measurements SERIAL on the single H100 (guard `nvidia-smi --query-compute-apps` empty before each).
Clock locked 1980 (`sudo nvidia-smi -lgc 1980`; runs ~1500–1830 under load — power/thermal, documented);
`-rgc` at exit. A concurrent claude session (pts/3) is alive but not on GPU; defensive guard, do not pause.
Consolidated output: `R_A_KEYSTONE_AND_PRODUCT_2026-06-08.md`. Last updated 2026-06-08.

## RUN COMPLETE 2026-06-08. Report: cipher-fusion-evidence/R_A_KEYSTONE_AND_PRODUCT_2026-06-08.md.
Final verdicts MATCH pre-reg: Track A = WALL-WITH-MECHANISM (over-determined: parity +10–45% ⊕ wave-quantized
reference epilogue +3–19%; +27–72% vs cuBLAS all shapes; pingpong implausible from CUTLASS headers, no build; coverage
200/200 0 FP T=1e-3). Track B = CLOSED-WITH-BOUND (Path-1 capture-seam node injection DEMONSTRATED on a real
torch.cuda.graph — refutes capture-invalidation wall; Path-2 enforce_eager tax MEASURED ~47%). Track C = ships
persistent-within-N (linear +2.96%@N45 ra-e2e / unified +2.51%@N64·+3.57%@N45 full-coverage-closure; hosting⊥cost).
4-agent adversarial verify ran (wrryqzqa6): Track-A table EXACT match, 0 faked closes; 7 overclaim/attribution fixes
APPLIED (coverage T was 1e-3 not 3e-5; dropped unsourced r-relerr-6e-7; Path-1 = toy graph not real vLLM graph +
constant-writer node; unified <3% only @N64 not the N45-64 band; unified mis-cited to ra-e2e → full-coverage-closure;
T=0 blanket → attention uses T=8×roundoff). Anchor 2edba0d2 UNCHANGED, clock -rgc reset, procs reaped.

## PRE-REGISTERED VERDICTS (written before measurement, advisor-reviewed)
- **TRACK A keystone = WALL-WITH-MECHANISM (pre-reg).** Two independent terms each exceed <3% on every
  real Mistral shape: (1) WGMMA-vs-cuBLAS PARITY GAP, (2) wave-quantized independent-reference epilogue.
  Even at hypothetical 100% parity the reference epilogue alone fails. CLOSED-WITH-BOUND only if a lever
  simultaneously reaches parity AND cuts the reference <3% on the largest shapes — wave-count physics
  (real N≤14336≈6.8 waves; 3% needs ~9–10 waves) makes that unreachable.
- **TRACK B hosting = Path-1 feasibility-gated on the cudaStreamEndCapture→Instantiate seam + node
  dependency ordering (advisor correction: vLLM uses STREAM capture, not the manual node API, so a
  cuGraphAddKernelNode interposition only bites at the EndCapture→Instantiate seam); Path-2 enforce_eager
  tax = the reliable CLOSED-WITH-BOUND fallback.** Verify the seam in installed vLLM source BEFORE building.
- **TRACK C** = assemble per A+B verdicts; product datasheet (coverage matrix as shippable claims).

## DISCIPLINE
Hold goal fixed, bring mechanisms, never scope-down. Honesty override: at a genuine floor, do NOT cross
the substrate line (no app-layer / cuBLAS-source / monkeypatch / vLLM-source edit), do NOT paper — record
WALL-WITH-MECHANISM and continue. Rule 4: paper ceiling before kernel. Every number from THIS run.

## RESULTS (live, append per track)
### TRACK A — keystone. (GPU done: cost + coverage. Pending: pingpong decision, paper ceiling from research.)
Files: `ra_keystone/trackA/{trackA_vs_cublas.py, trackA_interleaved.py, coverage_inject.py,
base_vs_cublas_1980.json, interleaved_1980.json, coverage_result.json}`. Kernel reused:
`step_b_prefill_build/libkernel_dedup.so` (CUTLASS sm90 KernelTmaWarpSpecializedCooperative, fused dedup
row-sum checksum; mode0=GEMM-only, mode3=fused). C rel-err vs cuBLAS = 0 (bit-exact) all shapes.
- **COST vs cuBLAS — FRESH, interleaved drift-canceling, median 200 reps @ locked-1980 (ran 1500–1830):**

  | shape | N | waves | parity gap | checksum | **TOTAL vs cuBLAS** |
  |---|---|---|---|---|---|
  | k/v_proj | 1024 | 0.48 | +45.0% | +18.8% | **+72.3% FAIL** |
  | q/o_proj | 4096 | 1.94 | +20.2% | +6.9% | **+28.5% FAIL** |
  | down_proj | 4096(K14336) | 1.94 | +10.3% | +15.0% | **+26.9% FAIL** |
  | gate/up_proj | 14336 | 6.79 | +28.4% | +3.3% | **+32.6% FAIL** |

  WALL is OVER-DETERMINED: (1) parity gap alone (+10–45%) fails every shape; (2) even at hypothetical
  perfect cuBLAS parity, the checksum epilogue alone (gate/up +3.3% best → k/v +18.8%) fails every shape.
  Checksum is wave-count-limited as predicted (gate/up 6.79 waves → +3.3% lowest; k/v 0.48 waves → +18.8%).
  => pingpong-hides-r (which only attacks the checksum term) CANNOT rescue: best case leaves parity +10–45%.
- **COVERAGE — CONFIRMED:** epilogue checksum (s=rowsum_N(C) vs independent r=A·wref, fp32) catches
  Step-A bit-13/14 flips **200/200 = 100%** all 4 shapes, **0 clean FP**, T from fp32 clean residual
  (~3e-5), |delta| 0.01–33K. Matches Step-A settled coverage; r bit-exact (rrel 6e-7).
- **PAPER CEILING (Rule 4, research w3ids3s56) — DONE, corroborates measurement:** s free (1/(2K)=0.003–0.012%),
  r FLOP 1/N (0.007–0.098%) but realized cost WAVE-COUNT-LIMITED (r = exposed serial CUDA-core FMA chain length K,
  not hideable in WGMMA mainloop until ~10–13.6 waves); crossing <3% at N≈22–28K > any real Mistral N (max 14336).
  Parity gap 2nd separately-fatal term; the parity↔epilogue fight is structurally unwinnable (closing parity shrinks
  base → r larger %, removes mainloop slack). Verdict: NO real shape admits <3%.
- **PINGPONG FALSIFIER — RESOLVED ANALYTICALLY (Rule 4) = IMPLAUSIBLE, no build warranted.** From CUTLASS 4.1.0
  headers (sm90_gemm_tma_warpspecialized_pingpong.hpp): (1) pingpong overlaps MMA(WGa)↔EPILOGUE(WGb), but r is
  MAINLOOP-resident (reads A as K-loop streams it; full-row A 3.7MB ≫ 228KB smem) → wrong phase; (2) no spare
  warpgroup in tensor-core-bound prefill (cooperative already uses both WGs for WGMMA+r); (3) cost = serial-chain(K)
  + wave-count, both SCHEDULE-INDEPENDENT; (4) pingpong static_assert-FORBIDS stream-K (line 105) — the one low-wave
  lever, exactly where worst FAILs are. Asymmetric warp-specialization (dedicate a WG to r) is the only plausible
  variant but has no spare WG + same serial/wave gate. Reinforced by fresh data: parity (+10–45%) dominates, so even
  a pingpong win (checksum→0) leaves parity FAIL. => WALL-WITH-MECHANISM settled measured+analytical; pingpong build
  cannot change verdict and source says it won't hide r. (reconcile-advisor confirming before final.)
- DECODE/PREFILL NUANCE preserved: decode M=1 +1.1% PASS (memory-bound, r hides under 117MB W-read) carries SEPARATE
  +10.1% GEMV substitution gap (cuBLASLt can't host row-sum-over-N → custom GEMV ~92% cuBLAS bw). Prefill FAIL stands.

### TRACK B — vLLM hosting. SEAM VERIFIED (research w3ids3s56, binary-confirmed). vLLM 0.20.2 / torch 2.11 uses
torch.cuda.graph STREAM capture (cuda_graph.py:308-312), keep_graph=False → EndCapture+InstantiateWithFlags+Destroy
inside ONE capture_end() C++ call (no Python seam). BUT libtorch_cuda.so imports cudaStreamEndCapture +
cudaGraphInstantiateWithFlags as UNDEF from separate libcudart.so.13 → INTERPOSABLE via CUDA_INJECTION64_PATH.
Path-1 = hook cudaGraphInstantiateWithFlags, receive populated cudaGraph_t pre-instantiate, cudaGraphGetNodes/GetEdges
→ leaves → cudaGraphAddKernelNode(check) + cudaGraphAddDependencies → real instantiate. Device flag in shim's own
cudaMalloc, host reads post-replay (no in-capture sync). NO vLLM source edit. VERDICT: FEASIBLE-BUT-FIDDLY (hard part
= recovering which node-param is the last GEMM output buffer via cudaGraphKernelNodeGetParams; ordering = easy).
VERDICT = CLOSED-WITH-BOUND. Files: ra_keystone/trackB/{gap6_inject_shim.c/.so, gap6_path1_demo.py,
gap6_path1_result.json, gap6_path2_eager_tax.py, gap6_path2_{graph,eager,tax}.json}.
- **PATH-1 (in-graph node injection) — DEMONSTRATED, refutes the prior capture-invalidation wall.** Built a pure-C
  CUDA_INJECTION-style shim (dlsym RTLD_NEXT, no cudart link/headers/nvcc — version-safe across nvcc-12.8/torch-cu13)
  that interposes cudaGraphInstantiateWithFlags, receives the populated cudaGraph_t pre-instantiate, enumerates
  nodes/edges, computes leaves, and adds a device node (memset sentinel) ordered after the leaves. On a REAL
  torch.cuda.graph-captured 7-node graph: hook fired, node EXECUTES IN REPLAY (flag 0→0xC1), host reads flag
  post-replay via own cudaMalloc'd buffer (no in-capture sync), graph stays VALID (re-replay OK, correct output with
  new input). => prior Gap-6 "an injected detector can't be hosted in cudagraphed vLLM" WALL is DOWNGRADED: a device
  check-node CAN be injected at the EndCapture→Instantiate seam without a vLLM source edit. BOUND: a raw out-of-pool
  node-add perturbs torch's NEXT fresh allocation once (cudaErrorInvalidValue, self-recovering; control w/ injection
  disabled = clean) — production needs pool-compatible insertion + the (unrun) GEMM-output-param recovery.
- **PATH-2 (enforce_eager) tax — MEASURED ~47% (prior estimate 10-30% REFUTED).** Qwen2-7B fp16, vLLM 0.20.2,
  cipher plugins EXCLUDED (VLLM_PLUGINS=lora-only) + VLLM_USE_DEEP_GEMM=0 (clean/uninjected). cudagraph speedup
  1.89× @batch1 / 1.91× @batch32 => enforce_eager TAX 47.1% / 47.5%. Hosting the detector by disabling cudagraph
  nearly HALVES decode throughput — an EXPENSIVE host, not 10-30%.
- NOTE (cipher auto-load found): vLLM general-plugins cipher_vllm_kv / cipher_vllm_kvdedup auto-register the cipher_v2
  runtime in vLLM (NOT in plain torch — Track A confirmed uninjected). Excluded for the clean tax. Anchor .so md5
  2edba0d2 UNCHANGED throughout (re-verified post-vLLM).

### TRACK C — product datasheet. IN PROGRESS. Advisor blind-spot (BINDING): Path-1 hosts a PER-STEP in-graph check
= exactly what Track A cost-walled; the SHIPPING detector is PERIODIC RECOMPUTE (+2.96% GEMM / ~2.8% unified @N≈64,
ra-e2e anchor) = a DIFFERENT execution model (out-of-graph/separate pass, NOT single in-main-graph node). "hostable"
(Track B) and "cheap-per-step" (Track A) are ORTHOGONAL axes. Hosting matrix must read: per-step check =
hostable(Path-1 feasible-but-fiddly) BUT cost-walled; periodic recompute = ships <3% (N≈64) hosted via periodic
out-of-graph recompute (keeps cudagraph, +2.96% ra-e2e — designed/anchored, not fresh-in-vLLM-measured) OR the
measured enforce_eager ~47% tax (expensive fallback). Do NOT claim "detector hosts in vLLM ✓" without this split.

## BACKGROUND
Research workflow `w3ids3s56` (NON-GPU): agent1 vLLM seam (Track B gate), agent2 Track-A paper ceiling,
agent3 Marlin parity + pingpong-hides-r mechanism + decode/prefill nuance. Decode (M=1) is a SEPARATE
GEMV kernel (+1.1% PASS / +10.1% substitution gap) — do NOT collapse into the prefill WALL.
