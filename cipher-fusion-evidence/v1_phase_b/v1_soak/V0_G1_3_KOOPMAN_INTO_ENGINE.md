# V0 GATE-1 increment g1.3: Koopman / EDMD into the engine -- DOES NOT COMPOSE (never actuates + calibration capture-unsafe)

**2026-06-03. PROBE-AND-REPORT verdict, and it REFINES the inc-1 prior rather than parroting it. HEADLINE: with the
shipped Koopman ON (`CIPHER_KOOPMAN=1`), the engine's real static-KV `torch.cuda.graph` capture SUCCEEDS, 48/48 KL=0,
on TinyLlama AND Qwen2-7B -- capture does NOT throw. BUT that KL=0 is VACUOUS w.r.t. Koopman: `handled=0` (zero
substitutions), so graph==eager==plain-fp16; it proves nothing about Koopman composing. Koopman does NOT compose into
the engine -- not because capture fails, but because (1) it never actuates there (`handled=0`, no shape `registered`,
no benefit, the rank-r kernel unreached/untested) and (2) its EDMD calibration feed contains capture-illegal ops
(`cudaHostAlloc` proven illegal in isolation; cusolver masked) that can't be trusted capture-safe and are correctly
kept OFF (inc-1).** NO `.so` change; g1.3 added ONLY probes (deployed `libcipher_rt.so` Jun-01 + anchor 1f305ce6
UNCHANGED; inc-1+inc-2 modules + `cipher_inc4.py` BYTE-IDENTICAL). Probes: `g13_koopman_capture_probe.py` (isolation,
"Probe B"), `g13_engagement_probe.py` ("Probe A"), `g13_endtoend_capture.py` (the real engine-path test).

## What the shipped Koopman path is (so the attribution is exact)

`cipher_rt_koopman_engine.cpp`: a matmul-dispatch actuator (`maybe_handle_koopman`) that intercepts FP16 `cublasGemmEx`
and, for shapes already in the calibrated `.cu` registry, substitutes `cipher_koopman_fp16_launch_shape()` (a rank-r
.cu kernel). Calibration is fed by the cuBLAS shim (`cipher_rt_cublas_shim.c:239-244`) calling
`cipher_edmd_live_collect` on every GEMM; once a shape accumulates `TARGET_ROWS=2000` snapshots a **detached background
thread** runs the EDMD SVD (`cipher_edmd_live.cpp:269+`, `cusolverDnSgesvd`) and registers the shape. Two distinct
sub-paths with distinct capture concerns: the **calibration/collect feed** (telemetry) and the **rank-r substitution
kernel** (the actuation).

## Engagement reality (Probe A) -- it DOES engage; not a null probe

`LD_PRELOAD=$SO CIPHER_KOOPMAN=1`, real torch fp16 GEMMs + a TinyLlama decode: **`koopman calls_total=1320`,
`handled=0`**. Torch 2.11 on H100 routes fp16 matmul through `cublasGemmEx` (NOT cublasLt for fp16), so the shim DOES
intercept and the collect feed DOES run. Engagement witness positive -- the probes below exercise a live path, not a
dead one.

## Isolation (Probe B) -- WHICH op is capture-illegal, named by the runtime (correcting the brief's hypothesis)

Calling the shipped `cipher_edmd_live_collect` directly inside a `torch.cuda.graph` capture, with a no-collect control:
- **CONTROL (no collect): capture+replay SUCCEEDS.** (The same-stream fp16 GEMM substrate is capture-safe.)
- **Steady-state snapshot path -> `cudaMemcpyAsync(...,cudaMemcpyDeviceToHost, stream 0)` @ `cipher_edmd_live.cpp:217`
  (`gpu_capture_async`): NOT capture-illegal.** In-capture `cudaPeekAtLastError=(0,'no error')`, capture completes. An
  async D2H to pinned host on stream 0 runs fine during capture. (My first hypothesis -- the memcpy -- was WRONG.)
- **Cold buffer-allocation path -> `cudaHostAlloc(cudaHostAllocDefault)` @ `cipher_edmd_live.cpp:171`
  (`ensure_capture_buffers_locked`): CAPTURE-ILLEGAL.** Driving the distinct-input gate so the 8th-distinct call
  allocates inside capture: `cudaPeekAtLastError=(900,'operation not permitted when stream is capturing')` ->
  `cudaErrorStreamCaptureInvalidated`, capture FAILS. A memory allocation during capture -- same illegality class as
  g1.2's act-order `g_idx` host op.
- **cusolver EDMD solve -> `cudaMalloc` @ `:287` + `cusolverDnSgesvd_bufferSize`/`cusolverDnSgesvd` @ `:306`:
  source-identified capture-illegal (alloc + cuSOLVER workspace), but MASKED behind `TARGET_ROWS=2000` + a detached
  thread -- never the op that trips in a short run.** Honest correction of the brief: cusolver is real and would be
  illegal, but it is NOT "the failing op"; the calibration ALLOCATION is the first illegal op, and even that only fires
  if a shape's buffers are first allocated during capture.

Attribution (held precisely): the illegal ops are in the EDMD **calibration/collect feed**, NOT in the rank-r
substitution kernel `cipher_koopman_fp16_launch_shape`.

## The real engine-path test (end-to-end) -- capture SUCCEEDS, and why (verified, not asserted)

`LD_PRELOAD=$SO CIPHER_KOOPMAN=1`, the inc-1/g1.2 static-KV harness (eager prefill+warmup -> `torch.cuda.graph`
decode capture -> replay), witnessed by the shipped `cipher_edmd_live_report()`:
- **TinyLlama-1.1B: capture+replay SUCCEEDS, graph-vs-eager 48/48 KL=0.** `handled=0`. Report: 4 shapes reached
  `rows=2000` (collect engaged -- the distinct-input gate DID pass on eager-warmup activations), lm_head `failed=1`,
  all `registered=0`.
- **Qwen2-7B: capture+replay SUCCEEDS, 48/48 KL=0.** `handled=0`. Report: most shapes hit the distinct-gate give-up
  (`failed=1`), one reached `rows=1676`.
- **CONTROL `CIPHER_KOOPMAN=0` (same preload): SUCCEEDS, 48/48, `calls_total=0`** -- isolates that the preloaded `.so`
  itself does not break capture.
- **Mechanism, instrumented (not asserted):** `report()` bracketing the captured forward shows the capped shape
  `K=2048 N=2048` (rows=2000, fit_launched) stays `calls=1764` FLAT across the capture block (collect early-exits at
  `:662`), while the non-capped `K=5632 N=2048` goes `calls 1078->1122`, `rows 1152->1350` -- collect DOES fire during
  the capture block but only the capture-LEGAL stream-0 memcpy. Every engaged shape had `rows>0` BEFORE the capture
  block, i.e. its one-time `cudaHostAlloc` landed during eager warmup, pre-capture. So no capture-illegal op fires
  inside the capture window in this lifecycle -> capture succeeds.

This is contingent, not a safety property: a shape first-seen DURING capture would trip `cudaHostAlloc` (Probe B), and
a `rows==2000` fit-trigger landing inside a capture window is a race with the detached SVD thread. It is correctly kept
OFF in the engine (inc-1 `CIPHER_RT_DISABLE_AUTO_INIT`).

## The load-bearing fact: `handled=0` -- and the KL=0 is vacuous w.r.t. Koopman

Koopman SUBSTITUTED zero GEMMs in every run (`handled=0`; no shape `registered` -- the background fit ran but did not
register, `registered=0` even at `rows=2000`). So during the captured decode Koopman was absent: graph==eager==
plain-fp16, and 48/48 KL=0 is trivially true. **This is NOT the brief's "surprise: capture-safe -> third composed
actuator" branch** -- that branch requires Koopman to actually substitute (`handled>0`) and THAT to capture KL=0. It
did not substitute. Boundary (not a permanence claim): whether a *registered* shape's rank-r substitution captures KL=0
is UNTESTED here -- the kernel was never reached/masked.

## Verdict (Mem #11) + framing

1. **Verdict: Koopman does NOT compose into the engine path.** Not via capture-failure (capture SUCCEEDS, 48/48 KL=0),
   but via (a) **no actuation** -- `handled=0`, no registration, the rank-r kernel unreached/untested, zero benefit; and
   (b) **calibration feed is capture-unsafe** -- `cudaHostAlloc` (`cipher_edmd_live.cpp:171`) is capture-illegal in
   isolation (CUDA err 900 -> StreamCaptureInvalidated; Probe B, control-isolated), cusolver (`:287`/`:306`) masked.
   The legacy single-tenant dispatch path RETAINS Koopman (it ships there, `CIPHER_KOOPMAN=1`); the engine path keeps
   `CIPHER_RT_DISABLE_AUTO_INIT`. Do NOT jam it into the engine graph.
2. **NON-REGRESSION:** probes only; NO `.so` change; deployed `libcipher_rt.so` + anchor 1f305ce6 UNCHANGED;
   inc-1+inc-2 modules + `cipher_inc4.py` BYTE-IDENTICAL; default-OFF (`CIPHER_KOOPMAN` unset) -> engine unaffected;
   GPU->0; subprocesses reaped.

**FRAMING -- gate-1 actuator sweep CLOSES:** g1.1 DVFS = ENERGY axis, composes (1.53x tok/W). g1.2 Marlin = int4
CORRECTNESS/capture-safe but no density. g1.2x bnb NF4 = DENSITY axis, composes (~2.5x). g1.3 Koopman = does NOT
compose (never actuates in-engine + calibration capture-unsafe). Refines the inc-1 prior: the ops are capture-illegal
in ISOLATION, but the engine path captures CLEAN -- Koopman just delivers nothing there. Two of three CIPHER compute
actuators (energy, density) compose capture-safe over the multi-model engine; the third (Koopman) stays on the legacy
single-tenant path.

## STOP -- next axis is G-O3 (MFU) then V.0/V.1 = Anil's call

The gate-1 actuator sweep is complete. Koopman closes it as a probe-and-report negative (engine does not benefit; legacy
path keeps it). Related: [[cipher-gate1-g11-dvfs-into-engine]], [[cipher-gate1-g12-marlin-int4]],
[[cipher-gate1-g12x-bnb-nf4-density]], [[cipher-go1-increment1-pager-graphdecode]].
