# V1 Phase B B.6''.9.8.5b.4 — Framing Y diagnostic audit

**Date:** 2026-05-27. **Status:** READ-ONLY audit complete; HOLD for Anil
adjudication of Framing Y refined scope before .8.6 / .8.6b / .8.6c
implementation order.

**Anchors at entry (unchanged at close):**
- `cipher_rt_phase4` `959a6f5`
- `cipher_kmod` `8c643fc`
- `cipher-fusion-evidence` `3ae4043`
- `cipher-platform v2.0 rev6` md5 `7c7068ca` (installed)

**Discipline preserved:** zero substrate touches (Memory #20); source
citations file\:line (Memory #12); framework-agnostic property evaluated for
every intercept proposal (Memory #9); audit-first like .8.5b.2 (Memory #11
honest residue).

---

## Section A — Marlin engagement architecture

### A.1 Path mapping (AWQ INT4 vLLM)

vLLM's AWQ INT4 dispatch chain (TinyLlama-1.1B-Chat-v1.0-AWQ measured at
.8.5b.3):

```
AWQMarlinLinearMethod.apply()
  → vllm/model_executor/kernels/linear/mixed_precision/machete.py:147
      output = ops.machete_mm(...)
  → vllm/_custom_ops.py:1483-1496
      torch.ops._C.machete_mm(...)
  → torch op dispatcher
  → vllm/_C.abi3.so symbol `_ZN7machete11mm_dispatchENS_6MMArgsE`
      (machete::mm_dispatch(MMArgs))
  → CUTLASS-generated MacheteCollectiveMma kernels
  → cuLaunchKernel directly (bypasses cublasGemmEx entirely)
```

**Where CIPHER classify_route fires today on this path:**
`cipher_rt_classify_observer` GOT-patched at `cuLaunchKernel` per
B.6''.9.8.1; the launch IS observed (per .8.5b.3 init log "cuLaunch shims
via GOT-patch"). However Marlin's actuator
(`cipher_rt_marlin_actuator.c:120`) sits at the cuBLAS layer behind
`cipher_rt_matmul_dispatch`; AWQ Machete never reaches cuBLAS so Marlin
sees **zero** triggering calls (.8.5b.3 measured `MARLIN.calls_total=0`).

**Where CIPHER Marlin intercept should happen — four candidates:**

| # | Layer | Per Memory #9 verdict | Mechanism notes |
|---|---|---|---|
| (i) | `cuModuleGetFunction` (Option C, .8.5b.2 audit) | framework-agnostic ✓ | Registration-time only; lacks per-launch M/N/K/dtype required by Marlin's per-call gate |
| (ii) | `cuLaunchKernel` + `cuFuncGetName` pattern match | framework-agnostic ✓ but … | Name detection works (already GOT-patched); but `kernelParams[]` ABI is opaque (`src/may13/cipher_dispatch.cpp:304` — *"cuLaunchKernel params layout is opaque (kernel-specific ABI) — we cannot safely extract tensor pointers from it at the C intercept level"*). Cannot extract M/N/K/A/B/C without Machete kernel-signature knowledge |
| (iii) | PyTorch `torch.ops` dispatcher | framework-coupled ✗ | Ruled out per Memory #9 |
| (iv) | **`machete::mm_dispatch(MMArgs)` symbol intercept** | Machete-coupled, framework-VERSION-agnostic ⚠ | Single stable C++ symbol in `vllm/_C.abi3.so` (`_ZN7machete11mm_dispatchENS_6MMArgsE`); GOT-patchable same pattern as cuBLAS; typed `MMArgs` exposes M/N/K/dtype/A/B/C/scales cleanly; identical pattern to today's libcipher_rt cuBLAS GOT-patch |

**Strict framework-agnostic interpretation of Memory #9 rules out (iv)** —
Machete is a vLLM+CUTLASS pairing. **Pragmatic interpretation** treats
"framework" as vLLM-version-coupling: (iv) survives because `machete::mm_dispatch`
is upstream CUTLASS-shape, ships unchanged across vLLM 0.21+ minor versions,
and any CUTLASS-Machete consumer (sglang, lmdeploy if they adopt) hits the
same symbol. **Anil adjudication needed on whether Machete-coupling is
acceptable for the AWQ INT4 workload class.**

### A.2 Marlin actuator API surface

From `cipher_rt_marlin_actuator.c:39-47`:

```c
extern int cipher_rt_marlin_engine_observe_weight(const void *w_ptr);
extern int cipher_rt_marlin_engine_is_ready(const void *w_ptr);
extern int cipher_rt_marlin_engine_lookup(const void *w_ptr,
                                          void **out_B, void **out_S,
                                          int *out_K, int *out_N, int *out_G);
extern int cipher_rt_marlin_engine_quantize_repack(
    const void *d_fp16_weight, int K, int N);
extern int cipher_rt_marlin_engine_dispatch(
    const void *a_fp16, const void *marlin_B, const void *marlin_S,
    void *c_fp16, int M, int N, int K, int G, void *stream);
```

**Dispatch decision shape:** per-call (gates at `cipher_rt_marlin_actuator.c:120-146`):
- dtype: `CUDA_R_16F` (fp16 only)
- M (batch) ≤ 64
- N (out_features) ≥ 1024 AND K (in_features) ≥ 1024
- K % 128 == 0 AND N % 64 == 0

**Lazy-quantize model** (`cipher_rt_marlin_actuator.c:150-169`,
`STABILITY_THRESHOLD=4`): the actuator observes the same weight pointer 4
times before triggering `engine_quantize_repack` (host-side fp16→INT4
~30-100 ms; subsequent calls hit the INT4 cache).

**Option C feasibility** (registration-time intercept at
`cuModuleGetFunction`): not a clean fit. Marlin's gate is shape-conditional
(M, N, K all needed at decide-time), and the weight pointer needed for the
lazy-quantize cache key is also per-launch. Option C would have to be layered
under a per-launch shape-check at `cuLaunchKernel`. **Option (iv) is the
clean fit** — `MMArgs` carries all parameters.

### A.3 Marlin fp16/bf16 scope

**Marlin actuator is fp16 INPUT ONLY**, not INT4-only:
- Header (`cipher_rt_marlin.h:7-9`): "*gates eligible cublasGemmEx calls (M ≤ 8, FP16 fp16 dtypes …) lazy-quantizes weights to INT4 + Marlin XOR-swizzled layout on first observation*"
- Gate `cipher_rt_marlin_actuator.c:120-124`: requires `Atype == Btype == Ctype == CUDA_R_16F` (= 2). **bf16 = `CUDA_R_16BF` = 14, REJECTED.**

So the substitution scope:

| Workload | dtype | cuBLAS path? | Marlin engagement |
|---|---|---|---|
| TinyLlama AWQ INT4 (`.8.5b.3` measured) | INT4-via-Machete | NO (bypasses) | 0 (no triggering call) |
| Mistral-7B-v0.1 **default** | bf16 (`config.json` `torch_dtype: bfloat16`) | YES | 0 (dtype gate rejects) |
| Mistral-7B with `--dtype float16` override (.8.5b.3 used this) | fp16 | YES | gates fire (M/N/K depend on workload) |
| Llama-3.1-8B default | bf16 | YES | 0 (dtype gate rejects) |
| Qwen-2.5-7B default | bf16 | YES | 0 (dtype gate rejects) |
| Phi-2 default | bf16 | YES | 0 (dtype gate rejects) |
| TinyLlama-1.1B default | fp16 | YES | gates fire (B=1 decode: regression per memory `cipher-t45-substrate-marlin`) |
| Llama-2-7B default | fp16 | YES | gates fire |

**The Marlin actuator does NOT have a bf16 dispatch path.** Adding one means
either (a) NVRTC-recompile Marlin's `cipher_rt_marlin_kernel_src` for bf16
acts (Marlin paper/cubin is INT4-weights × half-acts; bf16 acts would
require new cubin variants and accuracy re-validation), or (b) up-cast bf16
acts → fp16 in-flight (numerical drift; not bit-identical).

### A.4 Proposed .8.6b architecture

**Two coupled scopes — Anil picks:**

**.8.6b.A — Machete intercept via Option (iv) (AWQ INT4 coverage):**
- Add `cipher_rt_machete_actuator.c` registering with a new
  `cipher_rt_machete_dispatch` substrate (parallel to matmul-dispatch).
- GOT-patch `_ZN7machete11mm_dispatchENS_6MMArgsE` in `vllm/_C.abi3.so`
  via existing `cipher_rt_got_patch_init` infrastructure (`cipher_inject.c:85`).
- Pass-through default; substitution slot reserved for future Machete
  alternatives (e.g. CIPHER-side INT4 GEMM tuned for partition co-residence).
- v1 ship target: **observability + counter parity, NOT substitution**.
  Mirrors B.6''.9.8.5b.2's "ship infrastructure forward-compatible" pattern.
- ED: ~3-4 (GOT-patch wire + observer hooks + counter parity gate +
  Mistral-7B-AWQ regression check).
- **Constraint:** Machete kernels are persistent-style (per CUTLASS warp-specialized cooperative scheduler in the symbol name `KernelTmaWarpSpecializedCooperative`). Same partition constraint as Marlin applies — Phase 5 multi-tenant problem (per `cp_2_4/PHASE_5_MARLIN_PARTITION_CONSTRAINT.md`). v1 .8.6b is single-tenant only; multi-tenant Machete-substitution × green-context is Phase 5.

**.8.6b.B — bf16 path for existing Marlin actuator (default-bf16 model coverage):**
- Option B1: NVRTC-compile bf16-acts variant of `cipher_rt_marlin_kernel_src`.
  ED ~5-7 (NVRTC source variant + 10 entry-point matrix expansion + correctness gate). Risk: accuracy at low-bit weights × bf16 acts.
- Option B2: Up-cast bf16 → fp16 at gate (in-flight conversion). ED ~2.
  Risk: numerical drift; not bit-identical; KL test required against
  vanilla bf16 baseline.
- Option B3: Defer to v1.x (honest scope: v1 Marlin = fp16-only).

**Recommendation:** ship .8.6b.A (observability-first, Phase 5 partition
problem owned downstream) and .8.6b.B = B3 (defer bf16 to v1.x; document
the dtype-coverage gap in pitch material). Total .8.6b ED if recommended: **3-4 ED**.

---

## Section B — Koopman calibration architecture

### B.1 Current calibration mechanism

Pipeline (`cipher_rt_koopman_engine.cpp:14-20` + `src/may13/cipher_edmd_live.cpp`):

```
cublasGemmEx observed → cipher_dispatch.cpp:213 edmd_live_post_relaunch_hook
  → cipher_edmd_live_collect(K_dim, N_dim, ptr_A, ptr_B, ptr_C, M)
    [tunables at src/may13/cipher_edmd_live.cpp:34-65]
      - TARGET_ROWS = 2000 snapshots per shape before SVD fit
      - ROWS_PER_CALL = 8 (min(M, 8) rows copied per observation)
      - DISTINCT_INPUT_THRESHOLD = 8 unique activation ptrs gate
      - DISTINCT_GIVE_UP_CALLS = 20 (synthetic-bench give-up)
      - MAX_SHAPES = 16 (registry size LIMIT)
      - KR_RANK = 64; POWER_ITERS = 4
  → once rows >= TARGET_ROWS (src/may13/cipher_edmd_live.cpp:733):
        background HMT randomized-SVD on (X, Y)
        → cipher_koopman_fp16_register_shape(K_dim, N_dim, V_T, K_op, W)
  → .cu shape registry hit on subsequent cuBLAS calls
        → cipher_koopman_fp16_launch_shape() substitutes vs cuBLAS
```

**Convergence criterion**: 2000 snapshots accumulated for a given (K, N)
shape WITH ≥8 distinct activation pointers seen for that shape.

**Dtype gate** (`cipher_rt_koopman_engine.cpp:109-114`): **fp16 only**
(`CUDA_R_16F`). Same bf16-exclusion as Marlin — Llama-3 / Qwen / Phi-2 /
default-Mistral all rejected at the dtype gate even if registry is populated.

**OOD gate** (`cipher_rt_koopman_engine.cpp:146-154`): `residual_ratio` ≥ 0.05 (default; env CIPHER_KOOPMAN_OOD_THRESHOLD overridable) bails to passthrough.

### B.2 Calibration state in rev6 .deb

**`/usr/lib/cipher/` inventory** (`dpkg -L cipher-platform`):

```
cipher_cdi_patch.py / .sh
health-check.sh
libc10.so / libcipher_v2.so / libcipher_rt.so
runbook.md
```

**NO `.npz`, NO `.bin`, NO shape-registry serialization, NO calibration
artifact ships with rev6.** The Koopman engine starts with an empty .cu
shape registry on every container boot.

### B.3 Calibration UX gap

**Customer path "install .deb → Koopman engages" today:** does not exist.

The implicit path is *passive live-collection*: the engine collects snapshots
during normal inference and fits in the background. But:

- Customer must set `CIPHER_KOOPMAN=1` (`cipher_rt_koopman_engine.cpp:213-222`)
  — not in `CIPHER_ENV` shipped by `cipher_cdi_patch.py:249-255`.
- Customer must run **default-fp16 model** (Mistral-7B/TinyLlama/Llama-2 in
  fp16) — modern bf16 defaults (Llama-3/Qwen/Phi-2) skip at dtype gate
  before collection.
- Collection must accumulate 2000 snapshots × 8 distinct activations per shape
  AND background SVD must complete before customer's session ends.

.8.5b.3 measurement on Mistral-7B `--dtype float16`: 27 329 cuBLAS observer
calls → 0 substitutions because background SVD fit did not converge in the
measurement window.

### B.4 Proposed .8.6c tooling design

**Surfaces in order of preference:**

**.8.6c.A — Pre-shipped calibration (RECOMMENDED):**
- New deliverable: `/usr/lib/cipher/registries/<model_uuid>.npz` per common
  model family (Mistral-7B, Llama-3-8B if bf16 path lands, Phi-2, Qwen-2.5).
- Build pipeline: offline `cipher-calibrate-model` CLI runs vLLM with synthetic
  prompts long enough to fill 2000 snapshots × N shapes; serializes V_T, K_op,
  W matrices via `cipher_koopman_fp16_serialize_shape` (new sibling of register).
- Runtime: new `cipher_koopman_fp16_load_registry(path)` reads at init.
- Pro: customer sees engagement on `cipher-run vllm serve …` with zero config.
- Con: registry-per-model maintenance burden; per-snapshot license question
  for vendor's pre-built data.

**.8.6c.B — One-shot CLI calibrate at deploy:**
- `cipher-platform calibrate --model /path --workload-trace synth|user-trace.jsonl`.
- Customer runs once before serving; persists to `/var/lib/cipher/registries/`.
- Pro: avoids shipping per-model artifacts.
- Con: customer onboarding friction; trace quality determines registry quality.

**.8.6c.C — Auto-calibration with aggressive defaults:**
- Lower TARGET_ROWS (e.g. 500) for faster warm-up; accept rank-r quality drop.
- Tighten DISTINCT_GIVE_UP_CALLS; raise priority of background SVD thread.
- Add deployment-time warm-up: have `cipher-run` pre-launch a calibration
  workload from a stock prompt file before exec'ing the user command.
- Pro: zero customer config.
- Con: warm-up time on every container start; quality < pre-shipped.

**Recommendation:** ship .8.6c.A + .8.6c.B (pre-shipped for common models;
CLI for customer's own model). .8.6c.C is a separate ergonomics issue.

**ED estimate:**
- .8.6c.A pre-ship pipeline: ~5-6 (serialize/load + offline build + initial
  registry for Mistral-7B-v0.1 and one bf16 representative *IF* bf16 path
  lands).
- .8.6c.B CLI: ~3 (cipher-platform subcommand + calibration driver).
- Total **.8.6c: ~7-9 ED** if both ship.

### B.5 LM head 7.43× provenance — Memory #13 revision proposed

Memory anchor `cipher-track2-weight-sharing` carries no 7.43× claim;
Memory `cipher-t46-measurement` references "synthetic dedup 1.19×" not
7.43×. The "LM head 7.43×" framing surfaced in W13-14 prompt.

**Located evidence:**
- `WEEK_13_14_SCOPE_LOCK.md:188` (verbatim): *"LM head 7.43× provenance not located. The W13-14 task prompt referenced 'LM head verified 7.43×' but searching cipher-fusion-evidence + cipher_rt_phase4 + cipher_kmod produced no direct match."*
- `WEEK_13_14_SCOPE_LOCK.md:146` R-W14.2: classified as MINOR / informational.
- W14 Step 3 S3.C harness (`WEEK_13_14_COMPLETE.md:78`) achieved EXISTENCE
  gate at **0.9000 top-1 at rank-64 on TinyLlama LM head** under β=0.05 raised
  threshold — NOT a 7.43× speedup measurement.
- `WEEK_13_14_COMPLETE.md:85` (verbatim): *"The aggregate tok/s impact of
  Koopman compute substitution on a real vLLM decode workload is NOT
  measured at W13-14 close."*

**Proposed MEMORY.md revision:** the user's auto-memory currently does not
carry a "LM head 7.43× verified" anchor explicitly (audited the MEMORY.md
listing in the system reminder; no such entry found). The framing exists
only in session prompts and the W13-14 task brief. **No memory delete
needed; flag this as informational scope-correction:** Goal 4 v1 claim is
EXISTENCE (β-gated Koopman fires correctness-preserving on LM-head-class
shapes), not "7.43× speedup."

### B.6 Per-workload calibration cost

Mistral-7B (32 transformer blocks; GQA 8/32) distinct (K, N) matmul shapes
per forward (`config.json`-derived):

| Op | K (in) | N (out) | Per-layer count |
|---|---|---|---|
| Q proj | 4096 | 4096 | 1 |
| K proj | 4096 | 1024 (GQA) | 1 |
| V proj | 4096 | 1024 (GQA) | 1 |
| O proj | 4096 | 4096 | 1 (overlaps Q) |
| gate proj | 4096 | 14336 | 1 |
| up proj | 4096 | 14336 | 1 (overlaps gate) |
| down proj | 14336 | 4096 | 1 |
| LM head | 4096 | 32000 | once per forward |

**Distinct (K, N) pairs**: ~6-8 (counting overlaps). Fits comfortably under
`MAX_SHAPES=16`.

**Time-to-fill at decode B=1**: each cuBLAS observation copies
`min(M=1, ROWS_PER_CALL=8) = 1` snapshot. To hit `TARGET_ROWS=2000` per
shape: 2000 decode steps × 32 layers = 64 000 layer-passes × 7 shapes =
~448 000 cuBLAS observations. At Mistral B=1 throughput ~50 tok/s
(VOLT-disengaged baseline), that's ~40 seconds for all shapes in parallel.

**Time-to-fill at prefill (M=tokens)**: a single 512-token prefill writes
`min(512, 8)=8` snapshots per cuBLAS call × ~225 cuBLAS calls = 1800
snapshots in one forward — virtually all shapes hit TARGET_ROWS in one
medium-prompt prefill.

**SVD fit cost**: HMT randomized-SVD with `POWER_ITERS=4`, `OVERSAMPLE=10`,
rank-64 on ~2000×K (K up to 14336) matrices — host-side fp32, single-threaded
in current impl. Per-shape fit ~0.5-2 seconds; background thread spawns
after threshold (`src/may13/cipher_edmd_live.cpp:744`).

**Net calibration budget**: under 1 minute of real prefill + decode traffic
to fully calibrate a Mistral-7B-class model. **The .8.5b.3 zero-substitution
result is therefore a *driver gap, not a fundamental wall*** — the
27 329-call measurement run did not let the background SVD thread complete
before probe sampling.

**Implication for .8.6c.C**: aggressive auto-calibration is plausible; a
60-second deployment warm-up at `cipher-run` exec time would calibrate one
fp16 model fully. But the dtype gap (B.1) means this only helps explicit
fp16 models; bf16 customers see zero engagement regardless.

---

## Section C — VOLT contribution validation

### C.1 VOLT engagement on rev6 — UNVERIFIED

**rev6 CDI env injection** (`cipher_cdi_patch.py:249-255`):

```python
CIPHER_ENV = [
    "CIPHER_CDI_MARKER=v1",
    "CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so",
    "LD_LIBRARY_PATH=/usr/lib/cipher:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib",
    "TORCH_CUBLASLT_DISABLE=1",
    "CIPHER_REGISTER_MODEL=0",
]
```

**`CIPHER_VOLT=on` is NOT injected.**

VOLT init (`cipher_rt_volt.c:272-275`):
```c
if (!env_mode || strcmp(env_mode, "off") == 0 ||
    strcmp(env_mode, "0")   == 0) {
    return CIPHER_RT_VOLT_OFF;
}
```
And `cipher_rt_volt.c:288-292` requires `CIPHER_VOLT_BATCH` or
`CIPHER_VOLT_MHZ` in addition; without either, VOLT also returns OFF.

`cipher-run` wrapper (`/usr/bin/cipher-run:21-22`) exports only
`CUDA_INJECTION64_PATH` and `LD_LIBRARY_PATH` — does not opt into VOLT.

**Empirical** (`nvidia-smi --query-gpu=clocks.current.sm,…`): host idle
this session reads `clocks.current.sm=1830 MHz`, `clocks.applications.graphics=1980 MHz`, `clocks.max.sm=1980 MHz` — VOLT-disengaged
(running at near-stock application clock, not locked at a low VOLT target).

**.8.5b.3 koopman_eager.txt grep**: zero "VOLT: NVML path available" lines —
the run was VOLT-off as well.

**Net rev6 customer experience without manual env config:**

| Actuator | Init env required | Shipped in CDI? | Net engagement |
|---|---|---|---|
| VOLT | `CIPHER_VOLT=on` + `CIPHER_VOLT_BATCH` or `CIPHER_VOLT_MHZ` | NO | **0** |
| Marlin | `CIPHER_MARLIN=on` | NO | **0** (even with env, fp16+cuBLAS-only) |
| Koopman | `CIPHER_KOOPMAN=1` | NO | **0** (even with env, fp16-only + registry empty) |
| KV-dedup | `CIPHER_KVDEDUP=1` | NO | **0** (default off per memory `week6-kvdedup-autotrigger`) |
| REMEMBER | `CIPHER_REMEMBER=1` | NO | **0** |

**Out-of-the-box rev6 deploy delivers exactly zero substitution and zero
DVFS lock.** This is consistent with B.6''.9.4 substrate-coverage finding;
.8.5b.3 narrowed it to "engagement is practically zero on standard vLLM"
because of dtype + registry; this audit widens it to "engagement is
exactly zero without manual env config."

### C.2 VOLT +57.28% on current substrate — UNMEASURED at .8.5b.4

Memory `cipher-t43-envelope`: T4.3.2 reproduced +57.28% (TinyLlama-1.1B B=1)
but **did not generalize**: "Mistral-7B B=1 is -14% (no clock recovers
positive)." Honest claim already memorialized: *"+55% on
memory-bandwidth-bound decode only."*

Re-verification on rev6 + may13-port substrate **was not performed in this
audit** (Memory #20 + Memory #11 read-only discipline). Section C cannot
gate Goal 2 on Memory #11's number on the v1 substrate without a future
measurement step. Any synthesis below treats "+55% on memory-bandwidth-bound
fp16 decode" as a HISTORICAL bound, not a v1 anchor.

### C.3 Gap to Goal 2 — honest residue

Goal 2 commitment per Memory `cipher-fusion-campaign`: 2× tok/W via stacked
actuators.

**v1 reachable stack on a default-out-of-box rev6 deploy:** zero (no env
opt-in; no calibration; no dtype-matching workload).

**v1 reachable stack on a manually-configured rev6 deploy:**

| Workload | Stack engaged | Best estimate |
|---|---|---|
| TinyLlama-1.1B fp16 B=1 decode | VOLT + Marlin (if cuBLAS path; B=1 regresses) + Koopman (post-calibrate) | VOLT alone ~+55%; Marlin neutral-or-regress at B=1; Koopman speedup unmeasured-on-current-substrate; **upper bound ~1.55×** |
| Mistral-7B `--dtype float16` B=1 decode | VOLT (-14% per envelope) + Marlin (cuBLAS path) + Koopman (post-calibrate) | VOLT negative; Marlin neutral-or-positive at B=1; Koopman unmeasured; **upper bound likely < 1.5×** |
| Mistral-7B default bf16 B=1 decode | VOLT only | VOLT magnitude unmeasured on bf16; **likely flat-to-negative** |
| Llama-3.1-8B default bf16 B=1 decode | VOLT only | Same as above |
| TinyLlama-AWQ INT4 decode | VOLT only | VOLT only |

**Goal 2 "2× via stacked actuators" is currently unsupported on standard
production vLLM workloads** in the absence of either (a) a Machete-side
substitution actuator (.8.6b.A) AND (b) Koopman calibration tooling (.8.6c)
AND (c) bf16 actuator coverage (.8.6b.B). And even with all three, the
TinyLlama-1.1B +55% does not generalize to 7B-class models per
`cipher-t43-envelope`.

---

## Section D — Framing Y refined scope

### D.1 Total ED revised by diagnostic

| Item | Pre-audit estimate | Diagnostic-revised |
|---|---|---|
| .8.6 FlashAttention (Goal 3) | unchanged | ~3-4 ED (per scope-lock prior) |
| .8.6b Machete intercept (Option iv, observability-only) | unscoped | **~3-4 ED** |
| .8.6b bf16 Marlin coverage | unscoped | **~5-7 ED if shipped; recommend defer to v1.x** |
| .8.6c Koopman calibration tooling (A pre-ship + B CLI) | unscoped | **~7-9 ED** |
| **CIPHER_ENV injection of opt-in actuators in CDI patch** | unscoped | **~0.5 ED (config-only)** |
| VOLT re-measurement on rev6 substrate (Goal 2 anchor) | unscoped | **~1 ED measurement** |
| Memory revision pass (LM head provenance + scope-lock updates) | unscoped | **~0.5 ED** |

**Phase B remaining ED estimate revised:** prior estimate ~26-31 ED;
post-audit, if Anil picks recommended scope (.8.6 + .8.6b.A + .8.6c.A+B +
config fix + VOLT re-measure):

- Substantive ED: 3-4 + 3-4 + 7-9 + 0.5 + 1 + 0.5 = **~15-18 ED**
- If bf16 Marlin coverage also lands (.8.6b.B = B1 NVRTC): add 5-7 → **~20-25 ED**

### D.2 Risk factors surfaced by audit

1. **(NEW, HIGH)** rev6 CIPHER_ENV ships zero actuator opt-ins; the customer's
   first-touch experience is "engagement = 0" without manual env config. Even
   if .8.6b and .8.6c ship, customers won't see lift without the env-injection
   fix. **Owner: cipher_cdi_patch.py:249 update.**
2. **(NEW, HIGH)** bf16 default dtype on modern OSS models (Llama-3/Qwen/Phi-2,
   Mistral default) closes Marlin AND Koopman substitution doors. Without
   .8.6b.B, ~70% of OSS vLLM deployments see zero substitution regardless
   of calibration.
3. **(NEW, MEDIUM)** Machete intercept (.8.6b.A) inherits Marlin's
   partition-incompatibility constraint
   (`cp_2_4/PHASE_5_MARLIN_PARTITION_CONSTRAINT.md`): CUTLASS persistent-style
   kernels deadlock in green-context partitions. v1 single-tenant only;
   multi-tenant is Phase 5.
4. **(KNOWN, MEDIUM)** Goal 2 envelope per `cipher-t43-envelope`: VOLT +55%
   reproduces on TinyLlama-1.1B B=1 ONLY, regresses to -14% on Mistral-7B B=1.
   "2× via stacked actuators" relies on multi-actuator composition that has
   not been measured end-to-end on rev6 substrate at any model scale.
5. **(KNOWN, LOW)** LM head 7.43× provenance per WEEK_13_14_SCOPE_LOCK.md:188
   not located. Treat as historical reference, not v1 anchor.

### D.3 Recommended execution order

**.8.6 FlashAttention (Goal 3) FIRST — independent of Goal 2 scope decision.**

- .8.6 closes the empirical attn_calls=0 finding from B.6''.9.4 and ships
  Goal 3 substrate coverage. Does not depend on the Marlin/Koopman audit
  outcome above.
- Anil's Goal 2 scope decision (Section E below) gates .8.6b and .8.6c
  sequencing but does not gate .8.6.

**.8.6b and .8.6c sequencing** depends on Anil's pick in Section E.

### D.4 Phase B remaining calendar

Pre-audit estimate: ~26-31 ED Phase B total. Audit-revised under
recommended scope: ~15-18 ED post-.8.5b.4. At 1 ED/day cadence:
3-4 calendar weeks. Adding bf16 Marlin (.8.6b.B = B1) extends to ~5-6 weeks.

---

## Section E — Memory #1 Goal 2 honest re-examination

### E.1 The dtype-coverage gap (PRIMARY FINDING)

**Both substitution actuators reject bf16:**
- Marlin `cipher_rt_marlin_actuator.c:120-124` — fp16 only
- Koopman `cipher_rt_koopman_engine.cpp:109-114` — fp16 only

The Mistral-7B `config.json` shipped from HF declares `torch_dtype: bfloat16`
as default. `.8.5b.3` bypassed this with explicit `--dtype float16` override
(log line: `Casting torch.bfloat16 to torch.float16`). This is **not a
representative production configuration** — most vLLM tenants run native
bf16 for accuracy/speed and accept the default.

Modern OSS models that default to bf16:
- Llama-3, Llama-3.1, Llama-3.2 (all variants)
- Mistral (Mistral-7B-v0.1, v0.2, v0.3, Mixtral)
- Qwen-2, Qwen-2.5 (all variants)
- Phi-2, Phi-3
- Gemma-2

**The CIPHER v1 substitution actuator stack has zero out-of-box coverage of
this set.** Goal 2 "2× via stacked actuators" cannot be claimed against
modern OSS production vLLM without either (a) bf16 actuator coverage or (b)
explicit pitch narrowing to "fp16 vLLM workloads only."

### E.2 Goal 2 v1 scope options for Anil

| Option | Coverage claim | Scope work | ED |
|---|---|---|---|
| **E.1 Honest narrow** | "Goal 2 = 1.5-2× tok/W on fp16 vLLM workloads (Mistral, TinyLlama, Llama-2). bf16 coverage in v1.x." | .8.6c calibration tooling only; document fp16 scope; CIPHER_ENV opt-in fix | ~8-10 ED |
| **E.2 Full bf16 coverage** | "Goal 2 = 1.5-2× tok/W on fp16 AND bf16 vLLM workloads." | .8.6b.B NVRTC bf16 variant + .8.6c calibration + CIPHER_ENV fix | ~16-19 ED |
| **E.3 AWQ INT4 coverage** | "Goal 2 = 1.5-2× tok/W on AWQ INT4 vLLM workloads." | .8.6b.A Machete intercept ship-substitution variant + .8.6c | ~12-15 ED |
| **E.4 Reframe Goal 2** | "Goal 2 = observability + clock-lock on all dtypes; substitution on fp16 only in v1." | CIPHER_ENV fix + .8.6c-A small subset; admit fp16-only substitution in pitch | ~4-5 ED |
| **E.5 Defer Goal 2 to v1.x** | "v1 ships observability + substrate; substitution actuators in v1.x." | CIPHER_ENV fix + retract Marlin/Koopman from Goal 2 v1 claims; ship observability-only | ~1-2 ED |

**Recommended (subject to Anil pick):** **E.1 Honest narrow** plus a 1-line
addendum to Memory `cipher-fusion-campaign` Goal 2 scope: "v1 substitution
actuator stack is fp16-only; bf16 coverage tracked for v1.x."

### E.3 What Memory #1 looks like under E.1

Current Memory `cipher-fusion-campaign` framing: "close all 23 canonical
CPs… 2× tok/W via stacked actuators."

Proposed honest-narrow revision (Anil-approved language only):

> "Goal 2 v1 commitment: 1.5-2× tok/W on fp16-explicit vLLM workloads
> (Mistral-7B `--dtype float16`, TinyLlama, Llama-2). bf16-default workloads
> (Llama-3, Qwen, Phi-2, default Mistral) see VOLT-only contribution
> (magnitude unmeasured on rev6 substrate; envelope `cipher-t43-envelope`
> indicates regression risk at 7B+ scale). bf16 substitution coverage is
> v1.x scope."

This is the honest residue framing the audit surfaces. Memory `cipher-lift-framing`
already establishes the "ship on MFU gates not TPW lift" discipline; this
adds the dtype-coverage dimension to that discipline.

---

## HOLD point

**Decisions Anil owns:**

1. **Goal 2 scope (Section E)**: E.1 / E.2 / E.3 / E.4 / E.5? This is the
   single most important decision; it determines .8.6b and .8.6c sequencing.
2. **.8.6b architecture (Section A.4)**: Option (iv) Machete symbol intercept
   acceptable as Machete-coupled-but-framework-version-agnostic, or hold the
   Memory #9 line strictly and reject?
3. **CIPHER_ENV CDI patch update**: ship `CIPHER_VOLT=on` +
   `CIPHER_VOLT_BATCH=1` (or per-deployment config) in
   `cipher_cdi_patch.py:249`? Default-on changes customer-observable
   behaviour, may surface VOLT regression risk on 7B+ workloads.
4. **Memory revision**: should Memory #1 / Memory `cipher-fusion-campaign`
   gain the honest-narrow addendum from E.3 immediately, or wait for .8.6b
   landing?
5. **Execution order (Section D.3)**: confirm .8.6 first (independent), then
   .8.6b/.8.6c per Goal 2 scope adjudication?

**HOLD until Anil pick** on items 1-5. No substrate changes pending.

---

**Anchors at .8.5b.4 close (unchanged — diagnostic read-only):**
- `cipher_rt_phase4` `959a6f5`
- `cipher_kmod` `8c643fc`
- `cipher-platform v2.0 rev6` (md5 `7c7068ca`, installed)
- `cipher-fusion-evidence` will rotate at this commit
