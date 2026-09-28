# MFU LEVER PORT — DESIGN MEMO (design-first; STOP for approval before any build)

**Date:** 2026-05-30. **Status:** DESIGN ONLY. No code, no build, no `.so` change, nothing committed.
Two levers POC'd in `cipher-may13-evidence` but unwired in the live runtime (per
`MFU_MECHANISM_DOSSIER_2026-05-30.md`): **CUDA Graph promotion** (Lever 1) and **L2 weight persistence**
(Lever 2). This memo specifies the port into the CURRENT runtime `cipher_rt_phase4` (HEAD `2909eb3`,
where Marlin+FP8 fire), the isolating measurement, and the **honest expected outcome + confidence stated
up front** for each. Mem #13: this is a PORT = fresh additive substrate work (default-OFF, anchors
unchanged, no W7-12 modification, staging `.so` only). STOP for approval before porting.

---

## 0. The honest bottom line, up front (both levers, on the B=64 single-GPU forward)

| lever | expected B=64-forward-MFU contribution | confidence | basis (measured/structural, in hand) |
|---|---|---|---|
| **1. CUDA Graph promotion** | **~0 (low single-digit % at most)** | **HIGH it's ~0 at B=64** | the forward's CUDA **self-time 939.8 ms/fwd ≈ wall-clock ~940 ms** (`d9_fp8_decomp.py` + `d9_fp8_mfu` sdpa) → GPU is ~100% busy → **negligible launch-gap idle for graph to reclaim**; the POC 4.07× (synthetic 64-kernel launch-bench) and 2.05× (torch's *own* `torch.cuda.graph` on **decode**) do NOT transfer |
| **2. L2 weight persistence** | **~0 — structural, guaranteed** | **HIGH** | **nothing model-relevant fits in 50 MB L2** for Mistral-7B (weights 14 GB, 1 linear 117 MB, KV@B64 ~4 GB); the B=64 forward is compute-bound and streams each weight once (no reuse) → no L2 lever exists |

**Neither lever is a single-GPU-forward-MFU lever for a 7B model.** This matches the task's own
pre-acceptance ("L2-persist may reproduce 0% on the forward… record, do not tune"; "honest residue
closes nothing"). The **value of the port+measure** is: (i) close the residue by measuring the LIVE-runtime
contribution (the POC numbers were on different probes), (ii) confirm the real single-GPU forward ceiling
with {FP8, +graph, +L2} composed in the shipping `.so`, (iii) for graph, a 20-minute **feasibility kill**
(Build Step 0) likely ends it before integration. The memo specifies the full port regardless — approval,
not this memo, decides whether to build.

---

## 1. The current-runtime integration pattern a ported lever MUST mirror

The live actuators register/init in `cipher_inject.c:cipher_v2_init_body()`:
- `cipher_rt_volt_init()` (`:60`), `cipher_rt_marlin_init()` (`:62`), **`cipher_rt_fp8_init()` (`:63`)**,
  `cipher_rt_koopman_init()` (`:66`) — each: **net-new TU(s), default-OFF env gate (`CIPHER_*`), one init
  call here, lazy CUDA resources** (the D.9 FP8 pattern: `cipher_rt_fp8_engine.cpp` resolves symbols at
  init, allocates CUDA on first use). FP8 registers as a matmul-dispatch actuator at priority 20
  (`cipher_rt_fp8_actuator.c`). A ported lever follows the same shape: new `cipher_rt_<lever>_*.{c,cpp}`,
  `getenv("CIPHER_<LEVER>")` default-OFF, one `cipher_rt_<lever>_init()` line in `cipher_v2_init_body()`,
  added to `Makefile` OBJS, **W7-12 substrate untouched**, **byte-identical when OFF**.

---

## 2. LEVER 1 — CUDA Graph promotion

### 2.1 POC source → current integration point
- **POC (`cipher-may13-evidence`, commit `83b76da`):** `src/cipher_graph.cpp` — per-thread sliding-window
  kernel-signature sequence detector (`promote_after`, `g_replays`) + the real capture/replay path
  (`cuStreamBeginCapture_v2` `:185`, `cuStreamEndCapture`, `cuGraphInstantiate`, `cuGraphLaunch` `:157-171`)
  gated "behind a second env flag" (`:4-5`). `include/cipher_graph.h` = the OBSERVE→DECIDED→CAPTURED→
  REPLAYING→INVALID state machine.
- **POC numbers are non-transferable (cited):** 4.07× is a 64-kernel fp32-elementwise **launch-bound
  microbench**; 2.05× Mistral is **torch's own** `torch.cuda.graph(g)` on the **decode** loop
  (`run_mistral_graph_capture.py:122`) with CIPHER merely loaded — NOT CIPHER-driven promotion of an
  eager forward.
- **Current runtime hook:** `src/may13_intercept/cipher_graph_inspect.cpp` — intercepts
  `cuGraphInstantiate*`, **INSPECT-ONLY**, "With surgery deferred (Phase 4), this file is read-only — it
  never edits the executable graph" (`:13-14`), env `CIPHER_GRAPH_INSPECT`. Compiled into the `.so`
  (`Makefile:429`).
- **Integration target:** a new TU `cipher_rt_graph_promote.{cpp,h}` (mirror FP8), env `CIPHER_GRAPH_REPLAY`,
  `cipher_rt_graph_promote_init()` added to `cipher_v2_init_body()`. It would CIPHER-drive capture of the
  repeating per-layer launch sequence (the may13 `cipher_graph.cpp` detector + `cuStreamBeginCapture`/
  `EndCapture`/`cuGraphLaunch`), NOT reuse the inspect hook (which only edits torch-captured graphs).

### 2.2 Premise correction (must be in the report)
The task frames graph as attacking "the non-GEMM 31.2% floor (D9 decomp 67.5% GEMM / 31.2% norm-elem)."
**The 31.2% is GPU self-time of the norm/SiLU/elementwise *kernels*. Graph replay reduces CPU-side launch
overhead / inter-kernel idle gaps — a DIFFERENT quantity. Graph cannot reduce the 31.2% GPU-time at all.**
And per §0, the launch-gap idle at B=64 is ~0 (self-time ≈ wall-clock). So graph's leverage on the B=64
forward is the (near-zero) launch overhead, not the 31.2%.

### 2.3 BUILD STEP 0 — capture-feasibility probe (cheap kill, do FIRST, gates everything)
Before ANY `cipher_inject.c` integration: inject CIPHER, attempt `cuStreamBeginCapture` around ONE real
Mistral-7B B=64 forward and check whether it errors. **Torch's eager forward does dynamic `cudaMalloc`s,
which are not capture-safe** → capture likely returns `cudaErrorStreamCaptureUnsupported`/alloc error. If
capture fails, **Lever 1 is dead before integration** (a CIPHER-driven graph of an eager torch forward is
infeasible from the driver boundary; torch's own `torch.compile(mode=reduce-overhead)` solves this with a
capture-safe pool CIPHER can't replicate). ~20 minutes; report the result either way.

### 2.4 The port (only if Step 0 passes) + measurement
- Additive new TU, default-OFF `CIPHER_GRAPH_REPLAY`, registered in `cipher_inject.c`; lazy CUDA; builds
  into the live `.so`. W7-12 untouched.
- **Measure:** Mistral-7B B=64 forward, `CUDA_INJECTION64_PATH`, 700 W (watts+clock), CIPHER-graph ON vs
  OFF — forward-MFU from→to. **AND composed with FP8** (already live): FP8-only vs FP8+graph, measured
  against an **FP8-ON baseline** (so composition isn't conflated).
- **Guards (Mem #11):** (a) **correctness — the replayed graph must actually do the work**, not just
  KL=0 on one window (my own prior stale-capture bug: "ran once during capture, replayed stale output,
  looked fast"); verify output matches OFF on a fixed window AND that GPU work actually executes per
  replay (counter/timing sanity). (b) engaged-count > 0, telemetry-proven.

### 2.5 Honest expected outcome + confidence
**~0 to low-single-digit % on the B=64 forward** (HIGH confidence it's ~0): self-time≈wall-clock means
almost no launch idle to reclaim. **Likely killed at Step 0** (capture-safety). Where graph genuinely
helps — **B=1 decode** (launch/Python-bound) — torch's own cudagraphs already apply; CIPHER duplicating it
from the driver is hard and redundant. If the user wants the decode regime measured, that's a separate
(non-B64-forward) target; flag for the approval decision.

---

## 3. LEVER 2 — L2 weight persistence

### 3.1 POC source → current integration point
- **POC (`83b76da`):** `src/cipher_persist_engine.cpp:201` `out->hitProp = cudaAccessPropertyPersisting`
  (+`:234,:269`); `src/cipher_l2_persist.cu` (239 lines); fractional-knapsack budget vs `l2_persist_max`
  (31.2 MB on H100); 5/5 tests. **POC pinned CIPHER's own <3 MB LNN weights** (`cipher_l2_persist.h:2-12`)
  — *not* model data.
- **Current runtime:** the window-injection actuation is ALREADY present —
  `src/may13_intercept/cipher_intercept_cudart.cpp:2811-2826` appends
  `CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW` to `cuLaunchKernelEx` — **but `en_fn()` returns false because
  the persist engine is never initialized**: `cipher_init()` (`src/may13/cipher_runtime.cpp:25`, which
  calls `cipher_l2_persist_init` at `:49`) is **never invoked from `cipher_v2_init_body()`**.
- **Integration target:** new `cipher_rt_l2_persist_init()` (env `CIPHER_L2_PERSIST`) in `cipher_inject.c`
  that inits the persist engine + registers a target region (so `en_fn()` returns true and the existing
  injection at `:2811` engages). Additive; the W7-12 hot path is untouched (the injection site already
  exists, just inert).

### 3.2 NAME the target region — or the measurement is a no-op by construction
The injection only helps if a **model-relevant hot region fits in the 50 MB (31.2 MB persist-max) L2**.
For Mistral-7B: full weights **14 GB**, one decoder layer **~440 MB**, one linear (gate_proj)
**117 MB**, embeddings **262 MB**, KV@B=64×S=512 **~4 GB** — **none fit.** The B=64 forward streams each
weight **once** (no intra-forward reuse), and decode evicts weights long before reuse (14 GB ≫ 50 MB).
**⇒ there is no model-relevant region to pin → ~0 is structural/guaranteed, not merely empirical.** The
only pinnable data is CIPHER's own small scratch (LNN weights), which does not touch model MFU. (L2-persist
becomes a real lever only for ≤~25 M-param models whose weights fit in L2, or a genuinely-reused <31 MB
hot tensor — neither exists in 7B inference.)

### 3.3 The port + measurement
- Additive new TU, default-OFF `CIPHER_L2_PERSIST`, registered in `cipher_inject.c`; register the chosen
  region (for the measurement, the honest choice is "no model region qualifies" → measure the structural
  ~0, OR pin a named small tensor to demonstrate the mechanism fires).
- **Measure BOTH** (report both honestly): **(a) compute-bound B=64 forward** (POC said 0% — confirm/refute)
  and **(b) memory-bound B=1 decode** (where it *would* help if anything fit — expect ~0 too for 7B,
  weights ≫ L2). CIPHER-on vs off, 700 W, watts+clock, `CUDA_INJECTION64_PATH`; report window-hits AND
  ΔTFLOPS.
- **Guard (Mem #11):** output **byte-identical** with persist OFF — the access-policy window is a cache
  *hint*, must not change math; verify bitwise.

### 3.4 Honest expected outcome + confidence
**~0 on B=64 forward (HIGH) and ~0 on B=1 decode (HIGH) for Mistral-7B**, structural: nothing model-relevant
fits in L2. The honest finding ("L2-persist is not a 7B single-GPU lever; it needs data that fits in 50 MB
L2, which 7B inference does not provide") is recorded, not tuned toward a number — per the task.

---

## 4. Sequence + discipline + STOP

1. **Lever 1 Build Step 0** (capture-feasibility probe) — likely kills it; ~20 min; report.
2. If Step 0 passes: port Lever 1, measure (CIPHER-graph on/off + FP8±graph), guards, report → STOP.
3. Then Lever 2: port, measure (B=64 forward + B=1 decode), byte-identical guard, report → STOP.

**Report:** `MFU_LEVER_PORT_RESULT.md` per lever — the PORT (new TU + `cipher_inject.c` integration
file:line + commit + new `libcipher_rt.so` md5), CIPHER-on vs off forward-MFU (+decode for L2), composed
FP8+graph, what each ACTUALLY added vs its POC and WHY-if-different, guards. Honest delta: which ported
lever moves single-GPU forward-MFU and by how much, which reproduces ~0 (→ decode/multi-GPU lever), and
the real single-GPU forward-MFU ceiling with {FP8, +graph, +L2} composed in the LIVE runtime.

**Discipline:** design-memo→approve→build→STOP per lever. Mem #13 additive PORT (anchors UNCHANGED, no
W7-12 modification, staging `.so` only, no rotation). Mem #11 KL=0/byte-identical = HARD STOPs. Mem #24
substrate-level (graph capture at the CUDA driver graph API, L2 via `cudaAccessPolicyWindow` at the driver
— no framework source). Cite file:line + result-file for every number; NOT FOUND if not measured. Commit
as Anil, no co-author.

### STOP — awaiting approval before porting. Note for the approval decision: both levers are expected ~0
on the B=64 single-GPU forward (measured/structural, §0); Lever 1 is likely killed at the cheap Step-0
capture probe. If the goal is genuinely to move single-GPU forward-MFU, the evidence says neither lever
does it — the honest residue these close is "they are decode/memory-bound/multi-GPU levers, not
single-GPU-forward levers." Proceeding still has value (close the residue in the live runtime, confirm the
composed ceiling) — your call.
