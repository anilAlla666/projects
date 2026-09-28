# CIPHER Build State — op31-prod snapshot

## Pod
ssh root@103.207.149.87 -p 18850   # current as of 2026-04-15 perf-fix snapshot
ssh root@103.207.149.84 -p 17008   # prior pod (21-ops snapshot)
Working dir: /workspace/op31-prod

## Built — 21 ops complete, all gates green

### Phase 1 — Session Intelligence (3/3)
- Op 13 SENSE — session classification (HUMAN/AGENT/BATCH)
- Op 14 SHIELD — latency protection + band priority hints
- Op 15 SUSTAIN — KV pressure slope detection

### Phase 2 — Energy Layer (4/4)
- Op 20 THERMOSTAT — predictive thermal throttle prevention
- Op 22 PULSE — hardware fault early warning (score cap 2, Signal 2 deferred)
- Op 30 VOLT — AI classifier + frequency steering (Path B, pod-degraded)
- Op 31 HIBERNATE — idle detection + power gating (Path B, pod-degraded)

### Phase 3 — Straggler/NCCL (1/2)
- Straggler Detection — local slowdown + algorithm hint + tools/straggler_aggregate.py
- NCCL P2P CPU Proxy — DEFERRED (requires libibverbs + multi-node, not buildable here)

### Phase 4 — Agentic AI (3/3)
- Op 26 LOOP — agentic runaway detection (3 signals: shape cycle + decode/prefill ratio + prefill drought; score 0..3, runaway ≥ 2; hint-only v1)
- Op 19 CONTINUITY — incremental KV state checkpoint tracking (v1 observer: per-session region map + manifest gate every 500 events, AGENT-gated; v2 Tier-A KV capture deferred)
- Op 27 PIPELINE — multi-agent session correlation (v1 observer: per-session bounded shape set + pairwise Jaccard ≥ 0.5 edges in pipeline-graph JSON; v2 priority inheritance deferred to ARBITRATE v2)

## Proven results
- 7.76x peak speedup at M=4096
- 694 TFLOPS / 69.1% MFU
- max_diff = 0.000000
- 7/7 hardware validation: ALL PASS
- All 21 ops passing simultaneously with all ops ON (verified 2026-04-15, 20-test regression with CIPHER_SENSE/SHIELD/SUSTAIN/THERMOSTAT/PULSE/VOLT/HIBERNATE/STRAGGLER/LOOP/CONTINUITY/PIPELINE/PREDICT/GUARD/DETERMINISM/TOPOLOGY/TRACE/FAIRNESS/CARBON/RECEIPT/COMPLY all on)
- **Compute-bound GEMM (4096³ fp16) baseline post-fix (2026-04-15)** measured on this pod (H100):
  - M1 no preload:                 749 TFLOPS
  - M2 hook only:                  742 TFLOPS  (-1%)
  - M3 hook+rt observers off:      721 TFLOPS  (-4%, target ≥715 met)
  - M4 hook+rt all 20 on:          745 TFLOPS  (-0.5%)
  - Pre-fix M3 was 629 TFLOPS (13% regression). Two patches recovered it: cuLaunchKernelEx dispatch guard + lazy Stage-1/2 thread spawn. See "Performance fixes" below.

## Performance fixes (2026-04-15)
- **cuLaunchKernelEx dispatch guard** — when the parent `cublasGemmEx` shim has already classified a GEMM (`tls_shape_valid==1`), the cuBLAS-internal `cuLaunchKernelEx` calls now short-circuit `dispatch_and_log` instead of running full classify+oracle+registry per kernel. The cublasGemmEx shim emits ONE ring entry per GEMM with synthetic grid=M·N, block=K so observers (FAIRNESS, CARBON) see honest M·K·N work units. Mirrors the same `!tls_shape_valid` guard that `cudaLaunchKernel` already had. (`src/cipher_intercept_cudart.cpp` ~lines 380, 750)
- **Lazy Stage-1/2 thread spawn** — sum the return values of all 20 `cipher_<op>_init()` calls. Spawn `stage1_shadow` + `stage2_background` only if `observers_enabled > 0`. With all observers off, no threads, no CPU contention, no GPU launch-pipeline stall. (`src/cipher_10ops_impl.cpp:914-973`)
- **Diagnostic env guards left in tree** — `CIPHER_NO_BG_THREADS=1` (force-skip threads) and `CIPHER_NO_DISPATCH=1` (force-null `g_cipher_dispatch`). Both default off, useful for future bisects.
- **Test recalibration** — `tests/test_fairness.py` and `tests/test_comply.py` updated to the new M·K·N work-unit semantics (1e12 / 1e10 / 1e14 quotas).
- LOOP wiring repair: cipher_loop_init/observe were not wired into cipher_10ops_impl.cpp despite prior docs claim; repaired alongside Op 19 CONTINUITY wiring

## Remaining to build

### Phase 3 remainder
- NCCL P2P CPU Proxy — deferred to multi-node session

### Phase 4 — Agentic AI (complete)

### Phase 5 — Compliance and Observability (9/9 done)
- Op 17 PREDICT — proactive L2 preloading (v1 observer)
- Op 16 GUARD — KV cache privacy enforcement (v1 observer)
- Op 21 DETERMINISM — reproducible dispatch-sequence fingerprint (v1 observer)
- Op 25 TOPOLOGY — NVLink/PCIe peer adjacency (v1 init-time cudaDeviceCanAccessPeer matrix; JSON report at /tmp/cipher_topology_report.json; single-GPU pod: n=1, 0 edges)
- Op 28 TRACE — bounded kernel-trace exporter (v1 ring of 8192 compact records → JSONL at /tmp/cipher_trace.jsonl + summary JSON; drop-on-full)
- Op 24 FAIRNESS — per-tenant work quota (v1 observer: per-session grid*block accumulator; overrun flag when configurable quota crossed; env CIPHER_FAIRNESS_QUOTA)
- Op 23 CARBON — per-session carbon certificate (v1 observer: grid*block work * joules/unit * grid intensity → gCO2; env CIPHER_CARBON_J_PER_UNIT, CIPHER_CARBON_GCO2_PER_KWH)
- Op 18 RECEIPT — per-session signed proof of compute (v1: FNV-64 chain over params_hash + HMAC-SHA256(chain||launches||order) signed at report; env CIPHER_RECEIPT_KEY allows reproducible MAC)
- Op 29 COMPLY — regulatory compliance artifact bundler (v1 aggregator: stitches RECEIPT/CARBON/GUARD/DETERMINISM/FAIRNESS/TOPOLOGY C-API state into /tmp/cipher_comply_report.json with compliance_ok flag)
- Op 16 GUARD — KV cache privacy enforcement
- Op 21 DETERMINISM — reproducible dispatch sequence
- Op 23 CARBON — per-session carbon certificate
- Op 24 FAIRNESS — kernel-level tenant FLOP quota
- Op 25 TOPOLOGY — NVLink topology inference
- Op 28 TRACE — kernel-level execution trace export
- Op 29 COMPLY — regulatory compliance artifact generation

### Deferred Stage 0 hooks (one focused plan)
- cache_aggressive → Op 3 SUBSTITUTE
- oracle_aggressive → cipher_oracle.cpp
- sustain_compress → Op 3 SUBSTITUTE
- thermostat_aggressive → Op 3 SUBSTITUTE
- sentinel comparison → PULSE Signal 2

### Track B — Instance 4 (CUDA Graph Capture)
- Re-run Phase 4.0 on Mistral-7B first (TinyLlama wrong model)
- Then Phase 4.1 → 4.3 if gate passes
- **Plan-of-record (2026-04-15 brainstorm, not yet executed):** before committing to graph-capture code, run a real Mistral-7B inference with `CIPHER_PERSIST=on` and report `cipher_persist_fast_path_count / observe_count / promotion_count`. Decision rule:
  - If promotion fraction (fast_path / observe) ≥ 30%, build **Option B** — CIPHER-driven `cuStreamBeginCapture` of promoted stable sequences, replay via `cuGraphLaunch`, with synthesized ring entries on replay so all 21 observers keep working.
  - If promotion fraction < 30%, **skip Instance 4** entirely and prioritize **Instance 3 megakernel** instead. Graph capture buys little when stable sequences are rare.
  - **Option A** (non-interference: detect `cuStreamIsCapturing` and forward cleanly when PyTorch user code starts its own `torch.cuda.graph` block) is a defensive prerequisite to either path; ship A first regardless.
- No CUDA Graph code exists in the tree yet (verified 2026-04-15: zero `CUgraph*` / `cuStreamBeginCapture` / `cuGraphInstantiate` / `cuGraphLaunch` references in `src/`, `include/`, `docs/`).

### Track C — Instance 3 (Fused Megakernel)
- Depends on Instance 4 gate

### IMPLANT CUZ (after all tracks complete)
- Priority 1: Persistent kernel dispatch
- Priority 2: Flash Attention injection
- Priority 3: Cross-request KV prefix cache
- Priority 4: Skinny GEMM fusion
- Priority 5: NCCL compute overlap

## Next task when resuming
Phase 5 complete. Remaining tracks (outside Phase 5):
- Phase 3 remainder: NCCL P2P CPU Proxy (needs multi-node)
- Stage-0 deferred hooks (cache_aggressive, oracle_aggressive, sustain_compress, thermostat_aggressive, sentinel comparison)
- Track B (CUDA Graph Capture), Track C (Fused Megakernel)
- IMPLANT CUZ priorities
Same discipline: read files, report, propose, wait for approval.
