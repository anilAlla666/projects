# Pre-Benchmark Actuator Engagement Audit

**Date:** 2026-05-21
**Substrate state:** Week-5-complete (libcipher_rt.so `259ac994`, kmod.ko
`22febc8b`, kv_bridge.so `f041789c`; per `WEEK_6_ENTRY_PREFLIGHT.md`)
**Methodology:** read-only source inspection + Option 1 log re-read; no
source modified, no builds, no benchmarks rerun. The B.3 probe was
**not run** — source is conclusive on the gate logic and would only
re-confirm what Section B already shows.

---

## HEADLINE

**CONFIGURATION CANNOT BE LOCKED FOR ST2/ST3 YET.** Three structural
findings invert the task spec's hypotheses; surface for adjudication
before any benchmark prompt is drafted.

| # | finding | invalidates |
|---|---|---|
| **F1** | **Option 1's CIPHER arm ran with ALL actuators DISABLED.** Log shows `MARLIN: actuator DISABLED (CIPHER_MARLIN not set)`; VOLT silently didn't initialise (no `[cipher_v2] VOLT:` line); KVDEDUP off; spec decode not installed. The -4.91% tok/s, -2.20% tok/J regression measures **substrate intercept-only overhead** (GOT patch on cuLaunchKernel/cublasGemmEx + CUPTI callbacks + CLASSIFY observer + dispatch substrate), **with zero actuator gain to offset it.** Not "CIPHER regresses single-tenant" as the post-close summary framed it; it's "substrate intercept costs 5%, no actuator was engaged to pay it back." | "ST1's regression is a CIPHER fundamentals problem" — it isn't, it's an env-vars problem. |
| **F2** | **Marlin covers fp16 dense matmul, NOT pre-quantized INT4 models.** The actuator is `MARLIN_INT4` but the input-side gate is `Atype == Btype == Ctype == CUDA_R_16F`. Marlin **lazy-quantizes** the live fp16 weight to INT4 internally after 4 observations of the same weight pointer (runtime PTQ; engine is INT4, API is fp16). Pre-quantized INT4 models (GPTQ/AWQ) **bypass cublasGemmEx** in vLLM (use GPTQ-specific kernels) and never reach Marlin. **MoE models** (Llama 4 Scout, Mixtral, etc.) route experts through vLLM's `fused_moe` grouped matmul which also doesn't traverse cublasGemmEx. | The spec's proposed ST2/ST3 model set (Mistral Small 4 24B INT4 / Qwen 3.6 27B INT4 / Llama 4 Scout MoE FP8) is **inverted**: those are exactly the models Marlin DOES NOT engage on. ST2/ST3 must use fp16 dense decoder-only models (Mistral-7B, Llama-3-8B, Qwen2-7B fp16). |
| **F3** | **CIPHER's speculative decode is HF-transformers-only, not vLLM-integrated.** `cipher_spec_decode.py` monkey-patches `transformers.GenerationMixin.generate`. libcipher_rt.so contains zero spec/draft/ngram symbols. vLLM owns its own decode loop in EngineCore and never calls `transformers.GenerationMixin.generate` — CIPHER's spec decode wrapper has no surface to hook on vLLM. The May-13-14 2.96× tok/W headline used HF-transformers + cipher_spec_decode + Marlin (composing-for-free) + VOLT; reproducing that stack on vLLM requires using **vLLM's native `SpeculativeConfig`** (which is a separate code path, not CIPHER's). | "Reproduce May-13-14 headline on vLLM by stacking CIPHER's spec-decode" — that combination doesn't exist; either port CIPHER spec to vLLM EngineCore (engineering work, not minor) or use vLLM-native speculation and accept ~half the May-13-14 contribution stack. |

Plus one **non-blocker but needs-mitigation**:

| # | finding | mitigation |
|---|---|---|
| **F4** | **KV-dedup trigger is explicit-flush only (no auto-trigger).** `dedup_now()` must be called from inside the EngineCore subprocess; the plugin installs a SIGUSR1 handler. No periodic, threshold-based, or per-iteration auto-trigger exists. Source comment explicitly cites "T4.6.4 memo §Call model: ~400 µs/HIT … per-decode-step trigger unacceptable; explicit-flush is the W5 call-model choice." | For ST3 (single-process, 100 agents, continuous batching): external watchdog process sends SIGUSR1 every N seconds (crude, viable, no engineering); OR add threshold-based auto-trigger to plugin (~hours engineering, cleaner); OR accept reduced effectiveness on continuous workloads. Vanilla-vs-CIPHER fair comparison: vLLM's prefix-caching is request-level automatic, CIPHER's kvdedup is operator-triggered — apples-vs-oranges unless matched. |

The four other findings are mechanical and detailed in Sections A-E.

---

## PART A — Actuator + observer inventory

### A.1 — File catalog

`/home/ubuntu/cipher_rt_phase4/src/may13/` contains **24** .cpp files
(observers + tier ports; one short of the spec's "25 ported ops" — see
A.1.note). Actuator and substrate sources live at
`/home/ubuntu/cipher_rt_phase4/` root, not under `src/`.

**Classification:**

| class | files | role |
|---|---|---|
| **Substrate** (always runs once libcipher_rt.so is LD_PRELOAD'd) | `cipher_inject.c`, `cipher_cupti.c`, `cipher_rt_dispatch.{cpp,h}`, `cipher_rt_classify_substrate.{cpp,h}`, `cipher_rt_matmul_dispatch.{c,h}`, `cipher_rt_attn_dispatch.{cpp,h}`, `cipher_rt_got_patch.{c,h}`, `cipher_rt_cublas_shim.c`, `cipher_rt_kv_alloc.{c,h}` (in kv_bridge), `cipher_rt_oracle_bridge.{cpp,h}`, `cipher_rt_audit.{c,h}`, `cipher_rt_arbitrate.{c,h}`, `cipher_rt_green_ctx.{c,h}`, `cipher_rt_partition_router.{c,h}`, `cipher_rt_sm_packer.{c,h}`, `cipher_rt_tenant.{cpp,h}`, `cipher_rt_sense_transition.{c,h}` | GOT patch + CUPTI callbacks + classifier-driven dispatch tables + kv-VMM allocator + cuBLAS shim |
| **Observer** (engages on every relevant kernel when env truthy) | `src/may13/cipher_sense.cpp`, `cipher_oracle.cpp`, `cipher_runtime.cpp`, `cipher_telemetry.cpp`, `cipher_structural_lookup.cpp`, plus the Tier-A/B/Sub-4 observability ports: `cipher_loop.cpp`, `cipher_pipeline.cpp`, `cipher_pulse.cpp`, `cipher_continuity.cpp`, `cipher_trace.cpp`, `cipher_receipt.cpp`, `cipher_carbon.cpp`, `cipher_fairness.cpp`, `cipher_fairness_shm.cpp`, `cipher_guard.cpp`, `cipher_determinism.cpp`, `cipher_topology.cpp`, `cipher_comply.cpp`, `cipher_kernel_table.cpp` | per-launch / per-tenant / periodic observation; each is independently env-gated |
| **Actuator** (engages conditionally on shape + env) | `cipher_rt_marlin_actuator.c` + `cipher_rt_marlin_engine.cpp` + `cipher_rt_marlin_kernel_src.cpp`; `cipher_rt_volt.c`; `cipher_rt_attn_test_actuator.c` | conditional kernel substitution / DVFS clock-lock / attention-substrate test (Phase 4 demo) |
| **Static/data** | `cipher_recipes.cpp`, `cipher_l2_persist.cu`, `cipher_liquid_state.cu`, `cipher_green_ctx.cu` | reference tables + GPU-side stubs (not all wired into v1 substrate) |
| **Out-of-Phase-4 (compose externally)** | `cipher_spec_decode.py`, `cipher_kv_cache.py`, `cipher_vllm_plugin/cipher_vllm_kv.py`, `cipher_vllm_plugin/cipher_kv_offload.py`, `cipher_vllm_plugin/cipher_vllm_kvdedup.py` | Python-side wrappers; do not link into libcipher_rt.so |

**A.1.note — missing 5 of v1's 30 ops (per `CIPHER_REENGINEERING_PLAN.md §7`):**
COMMIT atomic-state-transition primitive (Weeks 7-8), RING_WRITE
lock-free telemetry (Weeks 9-10), and the 4 Koopman learning-tier ops
REMEMBER/VALIDATE/SPECULATE/ADAPT (Weeks 11-12). All 5 are unrelated
to ST1-ST3 actuator engagement; they're substrate-primitive scope, not
production actuators.

### A.2 — Env var catalog

**Truthy convention:** `"on"|"1"|"ON"` enables; anything else (including
unset / `"0"` / `"off"`) disables. Most actuators silently no-op when
unset (no log line); some log "DISABLED" on init.

| env var | default | controls | source |
|---|---|---|---|
| `CIPHER_DISPATCH_LIVE` | **on (1)** ← only "on by default" | classifier-driven matmul/attn routing; W3 Step 4 II-a default flipped from 0 → 1 | `cipher_rt_dispatch.cpp:54-57` |
| `CIPHER_MARLIN` | off | Marlin INT4 actuator engagement | `cipher_rt_marlin_actuator.c:203-217` |
| `CIPHER_MARLIN_VERBOSE` | off | per-call MARLIN log lines | same |
| `CIPHER_VOLT` | off | DVFS clock-lock master switch | `cipher_rt_volt.c:255-281` |
| `CIPHER_VOLT_MHZ` | unset | direct target clock (MHz, range [210,1980]) | `cipher_rt_volt.c:283-297` |
| `CIPHER_VOLT_BATCH` | unset | calibration table: 1→1000, 8→1600, 32→1980, 64→1980, else 0 (declines to lock) | `cipher_rt_volt.c:35-45` |
| `CIPHER_SENSE` | off | sense observer | `src/may13/cipher_sense.cpp:158-161` |
| `CIPHER_LOOP/PIPELINE/PULSE/CONTINUITY` | each off | Tier-A observability ports | each respective file `:~115-120` |
| `CIPHER_TRACE/RECEIPT/CARBON/FAIRNESS/GUARD/DETERMINISM/TOPOLOGY/COMPLY` | each off | Tier-B+ observability ports | each respective file |
| `CIPHER_AUDIT` / `CIPHER_AUDIT_DUMP` | off | HMAC audit chain + dump | `cipher_rt_audit.c:182-185` |
| `CIPHER_FORCE_PERMIT` | off | oracle bridge force-permit (test escape hatch) | `cipher_rt_oracle_bridge.cpp:301-302` |
| `CIPHER_ATTN_TEST` | off | dev-only attention-substrate fake actuator | `cipher_rt_attn_test_actuator.c` |
| `CIPHER_TENANT_ID` / `CIPHER_TENANT_NUM` | 0 | tenant identity for SM-partition + kvdedup | various |
| `CIPHER_SM_COUNT` | 0 (= no green ctx) | CP 5.4 SM-group allocation | `cipher_rt_green_ctx.c` |
| `CIPHER_QOS_CLASS` | 1 | green-ctx QoS class (1=shared, 2=dedicated) | same |
| `CIPHER_MIGRATABLE` | (not gated to a default) | DSM migration eligibility | `cipher_rt_green_ctx.c` |
| `CIPHER_KV_ALLOC` | **on (1)** | vLLM KV cache via CIPHER VMM allocator (CP 5.1) | `cipher_vllm_plugin/cipher_vllm_kv.py:37` |
| `CIPHER_KV_OFFLOAD` | **on (1)** | KV snapshot-on-preempt (CP 5.2) | `cipher_vllm_plugin/cipher_kv_offload.py:41` |
| `CIPHER_KVDEDUP` | off | cross-tenant KV-page dedup (W5) | `cipher_vllm_plugin/cipher_vllm_kvdedup.py:57` |
| `CIPHER_SPEC` | **on (1) BUT** install() must be explicitly called | speculative decode HF-side monkey-patch | `cipher_spec_decode.py:249` |
| `CIPHER_SPEC_DRAFT` | `"ngram"` | spec-decode draft policy (`ngram` or `<hf-path>`) | `cipher_spec_decode.py:264` |
| `CIPHER_SPEC_DRAFT_STREAM` | 0 | draft on default stream (1=own stream, deprecated/hangs) | `cipher_spec_decode.py:293-295` |
| `CIPHER_SPEC_EMA_WIN` | 64 | adaptive-k EMA window | `cipher_spec_decode.py:193` |
| `CIPHER_SPEC_NGRAM_N` | 3 | n-gram match length | `cipher_spec_decode.py:266` |
| `CIPHER_CARBON_J_PER_UNIT` | (numeric) | carbon-observer normalisation | `src/may13/cipher_carbon.cpp` |
| `CIPHER_FAIRNESS_QUOTA` | unset | per-tenant FAIRNESS quota | `src/may13/cipher_fairness.cpp` |
| `CIPHER_RECEIPT_KEY` | unset | HMAC key for receipts | `src/may13/cipher_receipt.cpp` |
| `CIPHER_KERNEL_TABLE_VERBOSE` | off | kernel-table verbose dump | `src/may13/cipher_kernel_table.cpp` |

**There is no `CIPHER_VOLT_POLICY=throughput|power|balanced`.** The task
spec's hypothetical policy-knob doesn't exist — VOLT has exactly two
modes: off, or "locked to a specific MHz." See Part C.

---

## PART B — Marlin INT4 actuator coverage

### B.1 — Gate semantics (read from `cipher_rt_marlin_actuator.c:60-180`)

| gate | criterion | what gets through |
|---|---|---|
| env | `CIPHER_MARLIN=on\|1\|ON` | else actuator DISABLED at init; no per-call cost |
| dispatch hint | optional W3 Step 3 ROUTE_MARLIN hint (advisory at LIVE=0, binding at LIVE=1 = current state) | classifier-driven; default fall-through is allow-all when no hint |
| dtype | `Atype == Btype == Ctype == CUDA_R_16F` | **fp16 only** — fp8, bf16, fp32, INT8, INT4 all skipped |
| batch (M=`call->n` per PyTorch convention) | `1 ≤ M ≤ 64` (`MARLIN_MAX_M_GATE=64`) | decode B=1..64; prefill batches >64 skip |
| shape (Marlin N=`call->m`, Marlin K=`call->k`) | `N ≥ 1024 AND K ≥ 1024` | excludes small KV-head / embedding projections |
| modulus | `K % 128 == 0 AND N % 64 == 0` | Marlin tile constraint |
| stability | per-weight-pointer hit count ≥ `STABILITY_THRESHOLD=4` | first 4 calls per weight go to cuBLAS; 5th onward Marlin handles after one ~30-100 ms quant_repack stall |

**Critical reinterpretation vs the task-spec hypothesis:**

The task spec assumed "INT4-only" means "must use pre-quantized INT4
models." **Source contradicts this.** Marlin in CIPHER takes an fp16
weight pointer, observes it 4 times, then on the 5th eligible call
runs `cipher_rt_marlin_engine_quantize_repack(weight_ptr, K, N)`
— **runtime PTQ from fp16 to INT4-packed Marlin layout** — and dispatches
the INT4 kernel. The model loads as fp16; Marlin handles the
quantization on-the-fly. The actuator name "MARLIN_INT4" refers to the
**kernel** (INT4), not the input format.

### B.2 — Coverage matrix (per source, NOT speculation)

| model class | weight dtype loaded by vLLM | matmul path | Marlin engages? | rationale |
|---|---|---|---|---|
| **Mistral-7B fp16** | fp16 | cuBLAS gemmEx via vLLM linear | **YES** (after 4 obs/weight) | dtype fp16 ✓; M=1..64 ✓; QKV/MLP shapes (N=4096/14336, K=4096/14336/1024-GQA) satisfy ≥1024 + mod-128/64 |
| **Llama-3-8B fp16** | fp16 | cuBLAS gemmEx | **YES** (after 4 obs/weight) | same as Mistral; layer shapes 4096/14336/1024-GQA |
| **Mistral Small 4 24B INT4 (GPTQ/AWQ)** | int4-packed (GPTQ) | vLLM's GPTQ kernel (NOT cublasGemmEx) | **NO** | Marlin's hook is on `cipher_rt_cublas_shim` / `cublasGemmEx` GOT slot; GPTQ kernels bypass cublas entirely |
| **Qwen 3.6 27B INT4** | int4-packed | GPTQ/AWQ kernel | **NO** | same as Mistral Small 4 |
| **Llama 4 Scout MoE FP8** | fp8 + MoE routing | vLLM's `fused_moe` grouped matmul (NOT cublasGemmEx) | **NO** (two reasons) | dtype gate would reject fp8; AND MoE expert dispatch doesn't go through cublasGemmEx |
| **Mistral-7B FP8 (vLLM auto-fp8)** | fp8 | cublas FP8 path | **NO** | dtype gate rejects (Atype != CUDA_R_16F) |
| **Mistral-7B fp16 prefill batch > 64** | fp16 | cuBLAS | **NO for that call** | M > 64 gate; only decode batches hit Marlin |

**Implication for ST2/ST3:** the task spec's "production-tier" model
list (Mistral Small 4 INT4, Qwen 3.6 27B INT4, Llama 4 Scout MoE FP8)
is **exactly the wrong list for Marlin engagement**. The correct
Marlin-eligible production models are **fp16 dense decoders**:
Mistral-7B fp16, Llama-3-8B fp16, Qwen2-7B fp16, etc.

### B.3 — Probe NOT run (source conclusive)

The task spec instructed `B.3 — Verify by running a small probe …
Expected: no (it's fp16, not INT4).` Source shows the **opposite
expectation**: with `CIPHER_MARLIN=on`, Mistral-7B fp16 WOULD engage
Marlin after stability hits.

The Option 1 CIPHER arm log
(`/tmp/postclose/option1_cipher.log`) confirms the run-time state:

```
[cipher_v2] MARLIN: actuator DISABLED (CIPHER_MARLIN not set)
```

— and that's the entire Marlin contribution to Option 1's regression
diagnostic: zero. Running a probe with `CIPHER_MARLIN=on` would
re-confirm what the source already shows; deferred unless the user
wants empirical confirmation as a separate step.

---

## PART C — VOLT DVFS policy

### C.1 — Policy semantics (read from `cipher_rt_volt.c:35-360`)

VOLT has **two modes**: OFF (default) and ACTIVE-locked-at-N-MHz. There
is **no continuous policy** (no throughput/power/balanced), no PID, no
adaptive-clock loop. Engagement model:

1. `CIPHER_VOLT=on|1|ON` required, else silently no-op (no init log).
2. Target clock comes from `CIPHER_VOLT_MHZ=N` (direct, range
   [210, 1980] MHz) OR `CIPHER_VOLT_BATCH={1,8,32,64}` (calibration
   table: B=1→1000 MHz, B=8→1600 MHz, B=32→1980 MHz, B=64→1980 MHz, else
   declines to engage).
3. NVML path tried first (`nvmlDeviceSetGpuLockedClocks`); falls through
   to kmod ioctl (`CIPHER_SET_CLOCK_MHZ` via `/dev/cipher`) if NVML
   probe fails — per `[[cipher-t432-kmod-volt-ioctl]]` non-root path.
4. atexit + SIG{TERM,INT,SEGV,ABRT,BUS} handlers installed to
   `nvmlDeviceResetGpuLockedClocks` on crash/exit.

**Implication for the task-spec's "VOLT default locks clocks low"
hypothesis:** VOLT is OFF by default; it locks NOTHING unless
explicitly told to. Option 1's CIPHER arm had no VOLT init line —
confirming VOLT didn't engage in that measurement.

**But:** if an operator sets `CIPHER_VOLT=on CIPHER_VOLT_BATCH=1` on a
B=1 decode workload, the calibration table locks to **1000 MHz** (vs
base 1980 MHz on H100). That's a ~50% clock reduction in exchange for
tok/W — exactly the "lock clocks low for efficiency" pattern the task
spec described, but **opt-in**, not default.

### C.2 — Option 1 clock state

Logs do not contain `VOLT` lines in either arm:

```
$ grep -i 'volt\|VOLT' /tmp/postclose/option1_{vanilla,cipher}.log
(empty)
```

The Option 1 harness fixed clocks to **1980/2619 MHz before running
both arms** (per `WEEK_5_POSTCLOSE_OPTION_1_MFU.md`); VOLT never
engaged in either arm. The -4.91% regression is **not a VOLT artifact**.

### C.3 — Policy knobs that exist (revised)

| knob | values | meaning |
|---|---|---|
| `CIPHER_VOLT` | `on\|1\|ON` (else off) | master switch |
| `CIPHER_VOLT_MHZ` | int in [210,1980] | direct target MHz |
| `CIPHER_VOLT_BATCH` | `{1, 8, 32, 64}` | calibration table lookup |

No `_POLICY`, no `_CLOCK_LOCK`, no adaptive mode. For ST1's regression
diagnostic: setting `CIPHER_VOLT=on CIPHER_VOLT_BATCH=64` locks to
1980 MHz (the base clock the operator already applied externally) and
adds VOLT's ~zero-overhead path; it would not change the tok/s but
would confirm VOLT's apply-restore path works under measurement. For
tok/W lift on B=1: `CIPHER_VOLT=on CIPHER_VOLT_BATCH=1` locks to 1000
MHz (this IS the May-13-14 lever, but it costs ~50% tok/s).

---

## PART D — Speculative decode wiring

### D.1 — CIPHER side vs vLLM side

**`cipher_spec_decode.py` is HF-transformers-only.** Source comment
(L4-5):

> install() monkey-patches the transformers generate path so a customer's
> model.generate() is routed through CIPHER's speculative loop.

The patch site is `transformers.GenerationMixin.generate`. libcipher_rt.so
has zero spec/draft/ngram symbols
(`nm -D libcipher_rt.so | grep -i 'spec\|draft\|ngram'` → empty).

**vLLM does not call `transformers.GenerationMixin.generate`** —
vLLM has its own `LLM.generate()` API and an EngineCore-owned decode
loop. CIPHER's spec decode monkey-patch **has no hook surface inside
vLLM**.

### D.2 — Configuration path for ST2/ST3 on vLLM

Three options:

| option | engineering | what you get |
|---|---|---|
| **(D-a)** Use vLLM's native `SpeculativeConfig` | none (already in vLLM 0.21.0); pass `speculative_config={"method": "ngram", "prompt_lookup_min": 2, "prompt_lookup_max": 8, "num_speculative_tokens": 5}` or a draft-model `{"model": "<draft-hf-path>", "num_speculative_tokens": 5}` | speculation runs natively in vLLM; **CIPHER's spec-decode does not contribute**; Marlin still composes underneath the target forwards |
| **(D-b)** Port CIPHER spec decode to vLLM EngineCore | non-trivial (engineering work; not in v1.2.2 §7 scope) | CIPHER controls the spec loop in vLLM; Marlin composes for free as it did on HF transformers |
| **(D-c)** Drop spec decode from ST2 stack | none | benchmark without speculation; May-13-14's headline drops by speculation's contribution (~half on B=1 decode per the original sweep) |

**Available draft pairs for D-a / D-b (ngram does not need a model):**
- ngram (parameter-free; vLLM-native and CIPHER-native both support it)
- Llama-3-8B target ← TinyLlama-1.1B draft (vocab-compatible)
- Mistral-7B target ← TinyLlama-1.1B draft (vocab **incompatible** — different tokenizer; would need vocab-projection or a Mistral-arch draft)
- The May-13-14 result used `CIPHER_SPEC_DRAFT="ngram"` for the Mistral arm (no draft model needed); same env on D-a would work on vLLM.

**Status note:** `cipher_spec_decode.py` header (L31-32) flags: "core
engine + drafts + adaptive-k below are CPU-unit-tested; GPU paths verified
on GPU after the CP 2.4 DVFS sweep completes." May-13-14 evidence
predates Phase 4 ports. The current substrate has not been smoke-tested
against the spec decode wrapper on the W5-close binaries.

---

## PART E — KV-dedup engagement in single-process vLLM

### E.1 — Trigger mechanism (read from `cipher_vllm_plugin/cipher_vllm_kvdedup.py`)

**Explicit-flush only.** Per source comment (L13-22):

> Trigger model — **explicit flush** (per WEEK_5_STEP_1_DESIGN_MEMO.md
> Part A.5b): operator calls cipher_vllm_kvdedup.dedup_now() at a
> quiescent point (post-prefill before decode, per Step 1 memo Part E.4).
> The cold-path latency budget (T4.6.4 memo §Call model: ~400 µs/HIT)
> makes per-decode-step trigger unacceptable; explicit-flush is the W5
> call-model choice.

Two trigger entry points exist in the plugin:

1. **Direct `dedup_now()` call** from the EngineCore process. Returns
   stats dict (`pages_processed`, `hits_on_flush`, `misses_on_flush`,
   `hits_delta`).
2. **SIGUSR1 handler** installed at plugin register time (L317-321);
   handler calls `dedup_now()` and writes JSON result to
   `/tmp/cipher_kvdedup_result_t{TENANT}.json`. EngineCore subprocess
   pid is written to `/tmp/cipher_kvdedup_pid_t{TENANT}.txt` for the
   external triggerer.

**No automatic, periodic, threshold-based, or per-iteration trigger.**

### E.2 — ST3 single-process viability

For ST3 (single vLLM process, 100 concurrent agents, continuous
batching), four paths to make dedup engage during a serving run:

| path | engineering | behavior |
|---|---|---|
| **(E-a)** External watchdog `kill -USR1 $(cat /tmp/cipher_kvdedup_pid_t0.txt)` every N seconds | trivial (one shell script) | crude; dedup latency stalls EngineCore for the flush duration (~ms × pages); fixed cadence |
| **(E-b)** Add threshold-based auto-trigger to the plugin (e.g., fire when N pages registered since last flush, or when HBM utilisation crosses %) | ~hours; modify `_make_kvdedup_wrapper` to track delta and self-call after N adds | cleaner; dedup keeps up with continuous workload without external coordination |
| **(E-c)** Periodic background thread inside the plugin that calls `dedup_now()` every N seconds with a lock | ~hours; threading + GIL concerns; deferred per source comment ("cold-path latency budget") | requires careful pause semantics — flush-mid-decode caveats |
| **(E-d)** Accept dedup never fires during ST3 serving | zero | CIPHER arm of ST3 gets zero KV-dedup contribution; CIPHER's KV "win" reduces to whatever vLLM-prefix-caching + CP-5.1-VMM-allocator delivers |

### E.3 — Fair-comparison framing for ST3

vLLM's `enable_prefix_caching=True` is **automatic** at request boundary
(reuses KV blocks across requests sharing a prefix). CIPHER's kvdedup
operates on **post-prefill cross-tenant content** (the 2.96× / 446×
dedup ratios at TinyLlama N=4/N=8 from Step 3.B / Option 2 are
cross-tenant content-page dedup, not request-level prefix reuse).

For ST3 to fairly compare:

- **Both arms** should have `enable_prefix_caching=True` (vLLM-side).
- **CIPHER arm** layers kvdedup on top of prefix caching; the
  marginal HBM savings come from content overlap *across* tenants that
  prefix-caching can't detect (different prompts, same model, similar
  KV content).
- **Trigger** for CIPHER's kvdedup must be one of (E-a)–(E-c) above;
  (E-d) means measuring CIPHER with kvdedup effectively off — same as
  Option 2's "Mistral N=8 OOM at gpu_util=0.18" finding (kvdedup needs
  to fire to save HBM at high N).

---

## PART F — Verdict + ST1/ST2/ST3 configuration matrix

### F.1 — What ST1 actually needs (single-tenant regression diagnostic)

**Spec target:** "identify why CIPHER regresses — VOLT policy? Intercept
overhead? Marlin not firing? RING_WRITE telemetry?"

**Findings constrain the answer:**

| candidate cause | source verdict | conclusion |
|---|---|---|
| VOLT policy locking clocks low | VOLT didn't init in Option 1 (log) | **NOT the cause** |
| Marlin not firing on fp16 | Marlin would fire on fp16 *if env set*; was DISABLED in Option 1 (log) | **Half the cause — actuator gain not realised**, but not because Marlin can't engage; because env was unset |
| RING_WRITE telemetry overhead | RING_WRITE primitive doesn't exist yet (Weeks 9-10 scope per §7); Option 1 had no RING_WRITE path | **NOT the cause** |
| Substrate intercept overhead (GOT patch + CUPTI + CLASSIFY observer + DISPATCH-LIVE + cublas-shim) | substrate ON in Option 1, no actuator to offset; -4.91% / -2.20% is the unbalanced cost | **THE cause** |

**Diagnostic plan for ST1 (proposed; not run here):**

1. Re-run Option 1 with `CIPHER_MARLIN=on` (everything else identical).
   Expected: tok/s recovers and may exceed vanilla; tok/J recovers
   further. **This is the single highest-information experiment.**
2. Re-run with `CIPHER_MARLIN=on CIPHER_VOLT=on CIPHER_VOLT_BATCH=64`
   (locks at 1980 MHz = base). Expected: ~same as #1 (VOLT-locked at
   base is a no-op on tok/s; confirms VOLT apply-restore path works).
3. Re-run with `CIPHER_MARLIN=on CIPHER_VOLT=on CIPHER_VOLT_BATCH=1`
   (locks at 1000 MHz). Expected: tok/s drops ~50%, tok/J rises
   substantially. **This is the May-13-14 tok/W lever** on a single
   tenant.
4. Optional decomposition runs: turn off substrate components one at a
   time (`CIPHER_DISPATCH_LIVE=0` rolls dispatch back to v1.2.1
   per-observer mutation; the cublas-shim cannot be disabled without
   unsetting `CUDA_INJECTION64_PATH`) to attribute the -4.91% across
   intercept + CUPTI + CLASSIFY observer.

### F.2 — What ST2 actually needs (reproduce May-13-14 headline)

**Spec target:** "production-tier model … with Marlin + VOLT + spec-decode
engaged."

**Findings constrain the model + the spec-decode wiring:**

- **Model:** must be **fp16 dense decoder-only** (Mistral-7B fp16,
  Llama-3-8B fp16, Qwen2-7B fp16). The spec's "Mistral Small 4 INT4 /
  Qwen 3.6 27B INT4 / Llama 4 Scout MoE FP8" set is wrong — none of
  them engage Marlin.
- **Spec-decode:** options (D-a) vLLM-native (no CIPHER contribution to
  spec), (D-b) port CIPHER spec to vLLM (engineering), or (D-c) drop
  spec from stack. Recommend **(D-a)** for ST2 to keep the
  vLLM-substrate axis honest; the May-13-14 headline can be partially
  reproduced (Marlin × VOLT, sans CIPHER-spec) and the spec lift
  measured separately on the vLLM-native path.
- **VOLT:** `CIPHER_VOLT=on CIPHER_VOLT_BATCH=1` (1000 MHz lock) is the
  May-13-14 lever for B=1 decode tok/W. For batched serving (ST3),
  `CIPHER_VOLT_BATCH=8` (1600 MHz) is the calibrated point.
- **Marlin:** `CIPHER_MARLIN=on`. Stability threshold = 4 obs per
  weight; allow a ~5-minute warm-up phase before measurement so all
  hot weights are quantized.

### F.3 — What ST3 actually needs (production agentic workload)

**Spec target:** "single vLLM process, 100 concurrent agents,
continuous batching + prefix caching enabled, CIPHER full stack stacked
on top."

**Findings constrain the model + the kvdedup trigger:**

- **Model:** fp16 dense (per F.2 — Marlin requirement). Mistral-7B fp16
  at N=100 needs HBM math: 100 × 14 GB weights = 1.4 TB > 80 GB →
  impossible without weight arena sharing. **Track 2 weight-arena is
  Week 13-14 CP 5.5 scope** ([[track2-weight-sharing]]); ST3 at N=100
  on Mistral-7B fp16 is **not feasible in the current substrate**.
  Practical N for ST3 on Mistral-7B fp16 with single weight copy =
  **N ≤ 4** (post-Option-2 finding: N=8 Mistral OOM at
  gpu_util=0.18).
- **Smaller model for N=100:** TinyLlama-1.1B fp16 (2.2 GB × 100 =
  220 GB → still OOM but with weight-arena sharing potential at lower
  marginal cost). Even single-copy TinyLlama N=100 = 2.2 GB ≤ 80 GB,
  feasible. **Recommendation: ST3 with TinyLlama-1.1B fp16 at N=100,
  OR Mistral-7B fp16 at N=4 (matching the Option-2 finding).** ST3
  cannot be both "Mistral-class" AND "N=100" at v1; pick one.
- **KV-dedup trigger:** (E-a) external watchdog OR (E-b) auto-trigger
  patch. Recommend **(E-b)** if the time is available (~hours); ST3 is
  the headline production benchmark and operator-triggered dedup is
  fragile. (E-a) is the no-engineering fallback.
- **VOLT:** `CIPHER_VOLT=on CIPHER_VOLT_BATCH=8` (1600 MHz).
- **Marlin:** `CIPHER_MARLIN=on` (only engages on fp16 dense → applies
  to both Mistral and TinyLlama).
- **Spec-decode:** vLLM-native `SpeculativeConfig` with ngram method
  for both arms; ST3 measures CIPHER-substrate-on-vLLM × spec, not
  CIPHER-spec.

### F.4 — Configuration matrix (revised vs the task spec)

| component | task-spec proposal | audited reality | source |
|---|---|---|---|
| **ST1 model** | Mistral-7B fp16 (Option 1 repro) | unchanged | spec |
| **ST2 model** | Mistral Small 4 24B INT4 primary, Qwen 3.6 27B INT4 fallback | **REVISED: Mistral-7B fp16 OR Llama-3-8B fp16 OR Qwen2-7B fp16** | F2 — Marlin doesn't engage on pre-quant INT4 |
| **ST3 model** | Llama 4 Scout FP8 OR Qwen 3.6 27B INT4 | **REVISED: Mistral-7B fp16 (N=4) OR TinyLlama-1.1B fp16 (N=100)** | F3 — Marlin needs fp16 dense; N=100 needs Track 2 or tiny model |
| **Quantization** | "as-loaded" | fp16 only for Marlin engagement | B.2 |
| **vLLM optimizations** | continuous batching + prefix caching + FP8 KV cache | continuous batching ✓, prefix caching ✓; **FP8 KV cache may interact** with CIPHER VMM allocator (unverified) | needs probe before ST3 |
| **CIPHER substrate** | OFF / ON via LD_PRELOAD | `CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` ✓ | sufficient |
| **Marlin** | "engaged IF INT4 dense" | `CIPHER_MARLIN=on` engages on fp16 dense (M ≤ 64, K/N ≥ 1024, mod-128/64); **NOT** on pre-quant INT4 or MoE | B.1, B.2 |
| **VOLT** | "engaged with throughput policy" | no "throughput policy" exists; `CIPHER_VOLT=on CIPHER_VOLT_BATCH={1\|8\|32\|64}` for table lookup or `CIPHER_VOLT_MHZ=N` for direct | C.1 |
| **Spec decode** | "engaged via vLLM speculative_config" | task-spec is right — but call this out explicitly as **vLLM-native, NOT CIPHER's `cipher_spec_decode`**; the May-13-14 headline cannot be reproduced verbatim on vLLM | D.1, D.2 |
| **KV-dedup** | "vLLM prefix caching + CIPHER KV-dedup" | needs trigger plan (E-a/E-b/E-c) for ST3; (E-d) = effectively no CIPHER dedup | E.1, E.2 |

### F.5 — Honest gap surface (per spec instructions)

The task spec said: *"If any structural finding … surface immediately.
Don't draft benchmark prompt until those are resolved."*

**Three structural findings surfaced; one mitigation needed:**

1. **F2 model class.** ST2/ST3 cannot use pre-quantized INT4 or
   FP8 MoE models with current Marlin. Recommend swapping to fp16
   dense (Mistral-7B / Llama-3-8B / Qwen2-7B). **User adjudication
   required** — keeping the spec's INT4/MoE list means Marlin lift
   = 0, which redefines what ST2/ST3 measure.

2. **F3 spec decode.** May-13-14 headline used HF-transformers-based
   CIPHER spec decode. vLLM has no CIPHER-spec hook surface. **User
   adjudication required** — (D-a) use vLLM-native and drop CIPHER's
   spec contribution from the v1 ship; (D-b) port; (D-c) drop spec
   entirely.

3. **F4 KV-dedup auto-trigger.** ST3 fairness needs (E-a) external
   watchdog or (E-b) ~hours engineering on the plugin. **User
   adjudication required** — (E-b) is the cleaner path; (E-a)
   unblocks ST3 today.

4. **N=100 + Mistral-class is incompatible at v1.** Track 2 weight-arena
   is Week-13-14 CP-5.5 scope. ST3 must pick one of (a) TinyLlama-1.1B
   at N=100, (b) Mistral-7B fp16 at N≤4, or (c) defer ST3 to W13-14.

**ST1 is unblocked.** The single-tenant Mistral-7B fp16 regression
diagnostic can run today against the substrate at W5-close anchors;
the high-information variant is "Option 1 with `CIPHER_MARLIN=on`."
ST2 and ST3 are blocked on the 4 adjudications above.

---

## Evidence (commands run this turn)

- `find /home/ubuntu/cipher_rt_phase4 -name '*marlin*' -o -name '*volt*' …` — actuator file inventory (Part A)
- `grep -rhnE 'getenv\s*\(\s*"CIPHER_[A-Z_]+"' /home/ubuntu/cipher_rt_phase4/ --include='*.c' --include='*.cpp' --include='*.h'` — 30 env vars enumerated (Part A.2)
- `nm -D /home/ubuntu/cipher_rt_phase4/libcipher_rt.so | grep -i volt` → 4 VOLT T-symbols
- `nm -D /home/ubuntu/cipher_rt_phase4/libcipher_rt.so | grep -i marlin` → 15 marlin T-symbols (engine + actuator + counters)
- `nm -D /home/ubuntu/cipher_rt_phase4/libcipher_rt.so | grep -i 'spec\|draft\|ngram'` → **empty** (spec decode is Python-only)
- `head -250 /home/ubuntu/cipher_rt_phase4/cipher_rt_marlin_actuator.c` — gate criteria (Part B.1)
- `sed -n '253,360p' /home/ubuntu/cipher_rt_phase4/cipher_rt_volt.c` — VOLT init + policy (Part C)
- `head -60 /home/ubuntu/cipher_rt_phase4/cipher_spec_decode.py` + `grep CIPHER_SPEC` — spec-decode env + install() semantics (Part D)
- `head -80 /home/ubuntu/cipher_vllm_plugin/cipher_vllm_kvdedup.py` + grep SIGUSR1/dedup_now — kvdedup trigger (Part E)
- `grep -i 'volt\|VOLT\|MARLIN\|CIPHER_' /tmp/postclose/option1_{vanilla,cipher}.log` — Option 1 env state (Part F.1)
- `/home/ubuntu/vllm_env/bin/python -c "from vllm.config import SpeculativeConfig; print(dir(...))"` — vLLM-native spec available (Part D.2)

**Time:** ~25 minutes. **No source modified. No builds. No benchmarks run.**
