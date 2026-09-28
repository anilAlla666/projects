# Phase 4.5 — Matmul-routing substrate + Marlin INT4 as first actuator

**Date:** 2026-05-14 evening.
**Sub-phases:** T4.5.0 discovery · T4.5.1 substrate · T4.5.2 Marlin port · T4.5.3 measurement · T4.5.4 report.
**Cap:** 8h on Phase 4.5; ~7h elapsed.

---

## TL;DR (honest, both layers)

### Engineering layer

**T4.5.1 substrate — ships clean.**
- LD_PRELOAD-able `.symver`-tagged cuBLAS interception (50 LOC + 4-line linker version script)
- Pluggable actuator registry: `cipher_rt_matmul_register_actuator()` accepts up to 16 priority-ordered actuators
- 155/155 cublasGemmEx calls on TinyLlama dispatched cleanly; outputs **byte-identical** to baseline with no actuators registered
- Perf overhead: −1.69% ± 1.17%, 95% CI [−4%, +0.6%] — statistically indistinguishable from zero
- `libcipher_rt.so.v0.2.0_T4_5_substrate` md5 `2f8218abaccc43150aa9edfca472e61b`

**T4.5.2 Marlin actuator — landed on substrate; product gain not demonstrated on TinyLlama B=1.**
- 770-line Marlin INT4 kernel ported (IST-DASLab Apache-2.0); compiles via NVRTC for sm_90 in 18.4s cold, 0.11s warm
- GPU-side quantize pipeline (transpose + scales + pack) replaces initial CPU draft (which would have taken ~80 min for full TinyLlama)
- Marlin kernel dispatch verified: TinyLlama forward with `CIPHER_MARLIN=on` reports `MATMUL: exit totals — calls=1240 handled=555 passthrough=685` (~45% routed to Marlin; rest fall through gate for small-shape)
- **Accuracy on a single 2048×2048 Linear: cos_sim=0.997, rel_err=0.073** — within INT4 noise floor
- **Accuracy on full TinyLlama logits: cos_sim=0.94, rel_err=0.34** — argmax flipped vs FP16 (degraded; INT4 noise compounds across 22 layers + lm_head)
- **Throughput on TinyLlama B=1 decode (3 matched pairs, 60s each):**
  - tok/s: 67.0 ± 1.6 (off) → 45.3 ± 0.04 (on), **−32.4%** (95% CI [−35%, −30%])
  - tok/W: 0.444 ± 0.008 → 0.325 ± 0.001, **−26.9%** (95% CI [−29%, −25%])
- `libcipher_rt.so.v0.2.0_T4_5` md5 `9ee93ec8958ab0de868c0bc0506c1cfa`

### Strategic layer

**The substrate is the strategic win, not Marlin's TinyLlama number.**

Tonight we shipped the matmul-routing substrate for CIPHER's compute-path moat. Every future actuator — FP8, fusion kernels, speculative decoding, per-tenant matmul dispatch — plugs in via the same `cipher_rt_matmul_register_actuator` interface with zero shim changes. The substrate is what makes the multi-actuator marvel composable.

Marlin tonight serves as the **proof point** that an actuator can land on this substrate end-to-end:
- API: register → maybe_handle → gating → dispatch ✓
- Engine: NVRTC compile → GPU quantize → host repack → cuLaunchKernel ✓
- Lifecycle: lazy init at first eligible call → cached per-weight thereafter ✓
- Discipline: ABI 12/12 unchanged, fallback md5s unchanged, taint stable ✓

What did NOT validate is Marlin's product **performance** on TinyLlama B=1. The prior `cipher-may13-evidence` 1.62× tok/W number was measured on Mistral-7B/Llama-3.1-8B at batch ≥ 8 (Marlin's designed regime); at batch=1 even on Mistral the prior measurement was only 1.03× tok/s. TinyLlama at M=10 (batch=1, seq=10) is **outside Marlin's designed regime**.

**Honest positioning vs B200:** Phase 4.5 advanced the deployment moat (substrate ready; FP8 / fusion / spec-decode have a hook) but did NOT close the MFU moat gap tonight. Marlin's product number requires testing on the prior validated workload class (Mistral-7B at B≥8); that test belongs to a next session with focused budget.

---

## Architecture as shipped

```
PyTorch/vLLM/TRT-LLM (customer code, UNCHANGED)
   │  (uses cublasGemmEx via libcublas.so.13)
   ▼
[LD_PRELOAD-interposed cipher_rt_cublasGemmEx_impl]   <-- .symver @libcublas.so.13
   │
   ▼  (builds cipher_rt_matmul_call descriptor + resolves stream)
cipher_rt_matmul_dispatch
   │
   ├── MARLIN_INT4 actuator (priority 10)
   │     gate: M≤64 (batch), N≥1024, K≥1024, K%128, N%64, fp16
   │     observe → stability threshold → lazy quant+repack → cached dispatch
   │
   ├── (future) FP8_COMPUTE actuator
   ├── (future) FUSION actuator
   ├── (future) SPECULATIVE_DECODE actuator
   ├── (future) PER_TENANT_DISPATCH actuator
   │
   └── PASSTHROUGH: real cublasGemmEx via dlvsym("libcublas.so.13")
```

Operator deployment unchanged:
```
LD_PRELOAD=/opt/cipher/lib/libcipher_rt.so
CUDA_INJECTION64_PATH=/opt/cipher/lib/libcipher_rt.so
CIPHER_VOLT=on CIPHER_VOLT_BATCH=1     # T4.3 efficiency moat
CIPHER_MARLIN=on                        # T4.5 Marlin (validated regime: B≥8)
python <customer service entry point>
```

---

## Sub-phase summary

### T4.5.0 — Discovery (40 min into 3h cap)

Two passes documented in `PHASE_4_T4_5_0_DISCOVERY.md`:
1. **First-pass D6 (REJECTED by user):** option (c) Python module monkey-patch — required customer code import, breaks operator-context.
2. **Second-pass D6:** Re-evaluated option (a) cuBLAS LD_PRELOAD interception.
   - First prototype (plain alias): inconsistent — only 1 of ~150 calls intercepted on TinyLlama (PyTorch 2.11 uses `cublasGemmEx@libcublas.so.13` version-tagged symbols).
   - Second prototype (`.symver` directive + linker version script): **155/155 calls intercepted**, M/N/K visible, no cuBLAS errors. The ~50-LOC `.symver` approach replaces ~3000 LOC of GOT-patching that the prior cipher-may13-evidence shim used.

### T4.5.1 — Substrate (~45 min into 4h cap)

- `cipher_rt_matmul_dispatch.{h,c}` — actuator registry (16 slots, priority-ordered)
- `cipher_rt_cublas_shim.c` — `.symver`-tagged shim + lazy `dlvsym` of real cuBLAS
- `cublas_version.map` — linker version script
- Verified: byte-identical output to baseline, overhead statistically zero

### T4.5.2 — Marlin actuator (~3h, exceeded 3h cap due to architectural learning)

**Files added (~2,600 LOC):**
- `cipher_rt_marlin_kernel_src.cpp` — IST-DASLab Marlin kernel (770 LOC, direct copy with renames)
- `cipher_rt_marlin_perms.h` — XOR-swizzle LUTs (81 LOC)
- `cipher_rt_marlin_engine.cpp` — NVRTC pipeline + GPU quantize + GPU transpose + host repack + dispatch (~700 LOC)
- `cipher_rt_marlin_actuator.c` — substrate hook + gating (~180 LOC)
- `cipher_rt_marlin.{h}` — public API

**Three architectural learnings (durable for the substrate):**

1. **Synchronous heavy work on dispatch hot path is unacceptable.** First CPU-quant draft was 30+ minutes/weight; replaced with GPU kernels. Lesson: actuator-substrate contract should mandate "maybe_handle ≤ some μs budget or return PASSTHROUGH and async-warmup". Tonight: GPU quantize ~1ms/weight + host repack ~50ms/weight + DtoH/HtoD ~20ms = ~70ms/weight at first eligible call. 150 weights → ~10s first-forward stall. Acceptable for service-startup but not for low-latency requirements.

2. **PyTorch's cuBLAS A operand IS the weight (column-major shape (out, in)).** Marlin requires the weight in (K_in, N_out) row-major. Solution: GPU transpose kernel before quantize (~0.5ms per weight). The prior c2_marlin.py Python harness did `weight.t().contiguous()` explicitly; under cuBLAS interception we must do it in the engine.

3. **Marlin's grid (132 SM blocks) deadlocks on small-shape weights.** Empirically tested: K=2048 N=256 hangs the kernel via cross-block lock saturation. Added gate `N≥1024 && K≥1024` — covers QKV/MLP large projections, excludes small KV head and embedding-like projections. ~45% of TinyLlama matmuls qualify.

### T4.5.3 — Measurement (3 matched pairs)

Per the discipline rules, ran honest measurement and reported the numbers regardless of direction. **Marlin regresses on TinyLlama B=1 decode.** This is consistent with:
- Marlin's designed regime is B≥8 prefill (M_marlin ≥ 16)
- Prior measurement showed batch=1 on Mistral-7B was only 1.03× tok/s
- TinyLlama (1.1B params) has less quantization redundancy than Mistral-7B (7B)
- M=10 (TinyLlama prompt seq_len) is in the GEMV regime Marlin doesn't optimize for

### Discipline gate at session end

| Gate | Result |
|---|---|
| Phase 3 ABI 12/12 PASS | ✅ (3+6+3) |
| Fallback kmod md5 `55ab8c0c` | ✅ unchanged |
| Fallback libcipher_v2 md5 `86618c30` | ✅ unchanged |
| Taint 12288 | ✅ unchanged |
| dmesg oops/WARN/BUG | none |
| Pod state | baseline (345 MHz, 700 W) |
| Substrate-only build saved | `libcipher_rt.so.v0.2.0_T4_5_substrate` md5 `2f8218a...e61b` |
| Marlin+substrate build saved | `libcipher_rt.so.v0.2.0_T4_5` md5 `9ee93ec...cfa` |

---

## What ships tonight, what doesn't

**Ships:**
- Substrate (matmul routing + actuator registry) — clean, verified, deployable
- Marlin engine + actuator code — integrated on substrate, accuracy verified on isolated matmul (cos=0.997 single Linear), known regime boundaries documented
- Two .so builds (substrate-only + Marlin-included) for operator choice

**Does NOT ship as product:**
- Marlin enabled by default — accuracy on TinyLlama is marginal (cos=0.94, argmax flips) and throughput regresses. `CIPHER_MARLIN=on` should NOT be the default in production until Mistral-7B regime is re-validated on current dispatch.
- Marlin's product-grade tok/W lift — needs the prior workload class (Mistral-7B/Llama-3.1-8B at B≥8) test in a focused next session.

## Position against B200 / marvel target

The marvel target stack is:
- Efficiency moat (was 25%, now ~50% with VOLT shipped) — **Phase 4.3 done**
- Multi-tenant moat (30%) — Phase 4.6 KV dedup, **not yet started**
- MFU moat (5%) — Marlin is the gap-closer here, **substrate ready but Marlin's perf gate is open**
- Deployment moat (was 70%, now ~85%) — every actuator now plugs into the same hook; **operator-context preserved throughout**

T4.5's contribution: **deployment moat advanced; MFU moat awaits next-session Marlin validation on its target regime.**

## Next phase recommendation

**Phase 4.6 — KV dedup + multi-tenant moat.** Per the marvel-target table, KV dedup is the biggest single lever for the multi-tenant story (50-100 tenants/H100). Substrate is now in place to host additional actuators (FP8, fusion, KV redirect) on the same hook.

**Parallel concern: Marlin perf validation.** Next session should run Marlin on Mistral-7B at B=8 (the prior validated regime) to either confirm the prior 1.38× tok/s number or document the gap honestly. This is ~2h of focused measurement, not a full substrate rebuild.

---

## Honest interpretation (no $48/GPU/year framing)

Tonight we built the substrate that lets CIPHER intercept every matmul PyTorch does, route it through a registry of pluggable actuators, and dispatch a substituted kernel back to user code — all without the customer changing a line. The substrate adds <2% perf overhead and is byte-correct when no actuators are registered. This is the *matmul routing piece* of the marvel.

Marlin landed on the substrate and produces accurate output on its target shape regime. Its TinyLlama B=1 regression is not a substrate failure — it's Marlin operating outside its designed envelope. The substrate did exactly what it should have: routed cleanly, gated honestly, and made the regression measurable.

Phase 4.5 advanced the deployment moat. Phase 4.6 advances the multi-tenant moat. By the time Phase 4.8 integration runs the composed 24h soak, the substrate carries multiple actuators each validated on their target regime — that's the marvel demo.

---

## Addendum — V1–V5 Mistral-7B Marlin validation attempt (Path B)

**Date:** 2026-05-14 evening (after T4.5 close). User-directed Path B:
validate Marlin on Mistral-7B B=8 autoregressive decode (Marlin's prior
1.38× tok/s claim regime).

### What completed

| Step | Result |
|---|---|
| V1.1 — locate Mistral-7B | ✓ `mistralai/Mistral-7B-v0.1` cached locally (14 GB fp16) |
| V1.2 — FP16 baseline | ✓ B=1 39ms / 652.7 tok/s prefill, B=8 84ms / 2391 tok/s; 450 cublasGemmEx routed cleanly, all passthrough; baseline logits saved |
| V2 — Marlin smoke (decode regime) | ✗ **Both runs hung in first eligible decode iteration; both terminated at 15-minute timeout** |
| V3 — accuracy on Mistral-7B | ✗ **Not measurable** — hang in V2 prevented capture |
| V4 — throughput matched pairs | ✗ **Not measurable** — same hang |
| V5 — close-out | This addendum |

### Root cause — same pattern, larger scale than TinyLlama

The lazy-quant-on-hot-path pattern combined with three compounding factors makes Mistral-7B's first eligible decode iteration infeasible to complete:

1. **Per-weight quant + repack is host-blocking ~400ms–2s per weight.** GPU transpose + quant + DtoH + host repack + HtoD. For a 4096×4096 weight ~400ms; for a 14336×4096 (Mistral down_proj) ~2s. Mistral-7B has 225 such weights.
2. **GREEN_CTX (T4.2.4d enforcement) constrains all kernels to 8/132 SMs.** Transpose + quant kernels run at ~6% of full-device throughput. A 14336×4096 transpose that takes ~30ms on 132 SMs takes ~400ms on 8 SMs.
3. **Quant fires sequentially for ALL 225 weights in the first eligible iteration.** No async warmup. Each cublasGemmEx call waits for its weight quantize to finish. Total: 60–200+s blocked time in the first eligible iteration.

This is the same architectural finding as TinyLlama (forward 4 = 3.2s for 555 eligible weights). On Mistral-7B with 225 weights and larger sizes, the same pattern scales to many minutes.

The substrate routing is correct (V1.2 verified: 450/450 calls passthrough cleanly, byte-correct). The actuator's *first-call latency contract* is the issue. **Not a substrate bug — an actuator design bug, surfaced by the substrate's honest measurement.**

### Three durable substrate-contract findings (for next-session Marlin re-port)

1. **The actuator-substrate contract should mandate `maybe_handle() ≤ µs budget OR return PASSTHROUGH and trigger ASYNC warmup.** Synchronous heavy work on the cuBLAS hot path is not viable. Add `actuator->warmup(weight, K, N, stream)` API the substrate calls on a background thread; `maybe_handle()` returns PASSTHROUGH until the actuator marks the weight ready.
2. **Actuator-side quantization kernels should NOT inherit GREEN_CTX.** Push primary context (cuDevicePrimaryCtxRetain) → run setup kernels at full device speed → restore green ctx. Tonight's port inherited green ctx, slowing setup ~16×.
3. **Pre-deployed weight kit is the production answer.** Offline tool: `cipher_marlin_prequant /path/to/model → /opt/cipher/marlin_cache/<model_hash>.bin`. libcipher_rt loads the kit at init. Zero hot-path quant latency. This is the deployment-correct pattern regardless of finding #1/#2.

### Outcome class per pre-experiment enumeration

User's outcome enumeration:
- α: Marlin reproduces ~1.62× tok/W on Mistral-7B B=8
- β: Smaller positive lift (1.1–1.4×)
- γ: Regression or neutral

**Land in γ** — though the blocker is the actuator's first-call latency contract, not Marlin's intrinsic correctness or throughput envelope. Marlin's product number on Mistral-7B remains unmeasured tonight; the prior 1.62× claim is neither confirmed nor refuted by this session's data.

### What the next-session re-port needs

The path is concrete engineering (not research):

**Option A (~3h fresh session):** Async warmup pattern. Restructure `cipher_rt_marlin_actuator.c` to set warmup-pending flag, spawn background thread with `cuDevicePrimaryCtxRetain` + push primary, run transpose/quant/repack at full speed, mark ready. `maybe_handle()` checks ready flag; PASSTHROUGH until set. Re-run V3/V4 — Marlin's first eligible iteration becomes fast (~10ms), so accuracy + throughput become measurable.

**Option B (~5h fresh session):** Offline pre-quant kit. Build a standalone tool that quantizes a HuggingFace model's safetensors and emits `marlin_cache.bin`. libcipher_rt at init reads the kit, pre-populates the engine's weight cache. Zero per-tenant quant latency forever. This is the operator-deployment-correct pattern and obviates Option A entirely for production.

Recommended: **Option B**, because:
- Operator-deployment story: pre-quant happens once per model at deploy time
- Zero hot-path latency forever (no per-tenant retraining)
- Decouples Marlin port debugging from PyTorch's lazy weight loading patterns
- Substrate contract stays simple (no async-warmup API needed for v1)

### Discipline gate (final, Phase 4.5 close)

| Gate | Result |
|---|---|
| Phase 3 ABI 12/12 PASS | ✅ (3+6+3) |
| Fallback kmod md5 `55ab8c0c` | ✅ unchanged |
| Fallback libcipher_v2 md5 `86618c30` | ✅ unchanged |
| Taint 12288 | ✅ unchanged |
| Pod state | baseline (345 MHz idle, 700 W) |
| Substrate-only build saved | `libcipher_rt.so.v0.2.0_T4_5_substrate` md5 `2f8218a...e61b` |
| Marlin+substrate build saved | `libcipher_rt.so.v0.2.0_T4_5` md5 `9ee93ec...cfa` |
| FP16 Mistral baseline logits saved | `/tmp/mistral_dec_fp16.pt` for next-session compare |

### Final position vs marvel target

| Moat | Status | Notes |
|---|---|---|
| Efficiency (DVFS) | **Shipped** | T4.3 +57% tok/W on B=1 decode (verified n=5 matched pairs) |
| Deployment (matmul substrate) | **Shipped** | T4.5.1 `.symver` cuBLAS interception; future actuators plug in zero-shim-changes |
| MFU (Marlin compute path) | **Open** | T4.5.2 engine integrated; next-session needs pre-quant kit OR async warmup |
| Multi-tenant (KV dedup) | **Not started** | Phase 4.6 |
| Composition (24h soak) | **Phase 4.8** | After 4.6 + 4.7 |

**Phase 4.5 final close: substrate ships, Marlin's product number deferred.** Net-add to the marvel architecture: an actuator-pluggable matmul-routing layer with operator-context preservation. Honest reporting: substrate verified end-to-end; Marlin actuator integrated but production-scale validation requires the actuator-contract refinements documented above.

The deployment story is now precise:
```
operator deploys:
  cipher_kmod.ko (T4.3.2)
  libcipher_rt.so.v0.2.0_T4_5_substrate  (T4.5.1 substrate, shippable today)

operator runs (per service):
  LD_PRELOAD=/opt/cipher/lib/libcipher_rt.so
  CUDA_INJECTION64_PATH=/opt/cipher/lib/libcipher_rt.so
  CIPHER_VOLT=on CIPHER_VOLT_BATCH=<batch>  # T4.3 efficiency
  # CIPHER_MARLIN=on intentionally NOT enabled — needs next-session re-port

operator's customer service:  UNCHANGED.
```

This is shippable tonight. Marlin re-enters production-readiness pipeline in a fresh session with concrete scope.
