# CP 2.4 — build progress (atomic STEP across sessions)

CP 2.4 is one atomic STEP executed over 2–3 weeks / multiple sessions; the
single `CP_2_4_REPORT.md` lands only when all gate criteria are measured. This
file is the durable checkpoint between sessions — **not** the report.

Build order (approved): Marlin per-stream registry → DVFS wire + envelope
sweep → speculative decode build → composition + gate.

---

## Session 2026-05-15 — progress

### Done
- **CP 2.1 addendum** appended to `CP_2_1_REPORT.md` — NCCL row 30 PORT→DELETE
  (`libnccl-tuner-cipher.so` exports `ncclTunerPlugin_v1/v2`). Closed.
- **CP 2.4 memo §2.2 amended** — Marlin registry key swapped per-shape →
  **per-stream**, user-adjudicated. Density-sweep evidence: one shared model →
  all tenants share ~10 shapes, so per-shape caps concurrency at ~10; per-stream
  scales N-way. Original §2.2 text preserved with a dated amendment note.
- **Marlin per-stream workspace registry — IMPLEMENTED.**
  `cipher_rt_phase4/cipher_rt_marlin_engine.cpp`:
  - Removed the single `g_marlin_workspace` / `g_marlin_workspace_sz`.
  - Added `MarlinWsSlot g_ws_slots[256]` — registry keyed on the caller's CUDA
    stream; `marlin_ws_for_stream()` returns a private `locks` buffer per
    stream (lazy `cudaMalloc`, LRU). 256 slots ≥ CP 0.5 peak N=192, so the LRU
    victim is always a destroyed stream → eviction is race-free.
  - `marlin_gemm_launch()` now takes a per-stream workspace; the GEMM still
    launches on the caller's own stream — no cross-stream sync.
  - Added `cipher_rt_marlin_engine_ws_slots()` diagnostic (distinct live
    streams = concurrency width).
- **`libcipher_rt.so` rebuilt** — clean (`make`, one benign pre-existing
  `_GNU_SOURCE` warning).

### Artifact state
- `cipher_rt_phase4/libcipher_rt.so` — **new build `a0d6cddacb116f2d51ad1d1f867ef564`**, 141400 B.
- `cipher_rt_phase4/libcipher_rt.so.pre_cp2_4` — rollback point `4f5cf543439d64a34b18742faef2ce93`.
- `cipher_rt_marlin_engine_ws_slots` symbol present (verified `nm`).

### Anchors — held
- kmod **0.4.8 `e2f50452f668859a96b1e25a2cba4e10`** — new campaign anchor
  (devnode-codify STEP 2026-05-15; supersedes 0.4.7 `2a69f9de`). The Marlin
  sub-task was pure userspace; the kmod bump is the separate `/dev/cipher`
  fix. **No ABI change** in either — no ioctl nrs touched.
- libcipher_v2 anchor `86618c30896470b642fcc6985d8dc632` — untouched.
- libcipher_rt `a0d6cdda…` (CP 2.4 Marlin build) — untouched by the kmod fix.
- kmod fallback anchor `55ab8c0c` — untouched. Taint 12288.

### Marlin sub-task (i) regression — test B DONE, test A next

**Finding:** verbatim `density_sweep_a.py` cannot test the new engine — it is
wired to the old op31 LD_PRELOAD API (`cipher_weight_compress_marlin_gemm`,
absent from the v2 lib) and carries its own Python `_MARLIN_LOCK`. User
adjudicated (2026-05-15): regression = **Both** — B microbenchmark + A
end-to-end.

**Test B — DONE (2026-05-15).** `marlin_regression/marlin_concurrency_bench.cu`
drives the v2 Marlin engine from N threads. Result (`REGRESSION_NOTE.md`):
- Serialization removed: shared-stream control flat ~86.5K GEMM/s (the
  CP 0.4/0.5 ceiling); per-stream lifts to ~152K (**1.76×**).
- `ws_slots` confirms N-way slotting; `rc_fail=0` at 64 concurrent streams
  (race-free).
- Honest ceiling: per-stream saturates at N≈4 — a **GPU-occupancy** ceiling
  (M=1 GEMM = grid 132 blocks fills the SM array), NOT a software lock. The
  registry removed the software ceiling; the microbench hit the silicon one.
- Verdict: **mechanism PASS.** Test A is decisive for the tok/s claim (real
  decode is ~99.98% GPU-idle, so occupancy won't bind there).

### Test A — v2-lib activation invocation (RESOLVED 2026-05-15)

Source: `PHASE_4_T4_5_REPORT.md:74-77,269-272` and `PHASE_4_T4_6_1_REPORT.md:203`
(*"operator deployment is `CUDA_INJECTION64_PATH + LD_PRELOAD`"*). The v2 lib is
activated by **both** env vars pointing at the same `libcipher_rt.so`:

```
LD_PRELOAD=<...>/libcipher_rt.so          # .symver cuBLAS interposition
CUDA_INJECTION64_PATH=<...>/libcipher_rt.so  # driver calls InitializeInjection2
CIPHER_MARLIN=on                          # route GEMMs to the Marlin actuator
```

- **LD_PRELOAD** makes the `.symver`-tagged `cipher_rt_cublasGemmEx_impl`
  interpose `cublasGemmEx@libcublas.so.13` (T4.5.1) and the plain ATen-SDPA
  trampolines (T4.6.1). Symbol interposition needs LD_PRELOAD load order.
- **CUDA_INJECTION64_PATH** makes the CUDA driver call `InitializeInjection2`
  at `cuInit` → runs the actuator init chain in `cipher_inject.c`
  (`cipher_rt_volt_init`, `_matmul_dispatch_init`, `cipher_rt_marlin_init`,
  `_attn_dispatch_init`). Without it the actuators never register and the
  cuBLAS shim passes everything through.
- Both must point at the **same** file. For test A: the CP 2.4 build
  `cipher_rt_phase4/libcipher_rt.so` (`a0d6cdda`).
- `CIPHER_MARLIN=on` arms the Marlin actuator (default OFF). T4.5 verified
  routing: `MATMUL: exit totals — calls=… handled=… passthrough=…`.

Note (from T4.5 §finding): Marlin's *per-call* lift is validated for B≥8 and
regresses at B=1 on TinyLlama — that is **Marlin sub-task (ii)**, separate.
Test A measures only the **concurrency regression** (does removing the lock
lift the aggregate ceiling); `CIPHER_MARLIN=on` is needed to put GEMMs on the
per-stream registry path, regardless of per-call speed.

**Test A — DONE (2026-05-15).** `marlin_regression/test_a_density.py` —
N-tenant Mistral-7B decode under the v2 lib, no Python `_MARLIN_LOCK`. Routing
confirmed: 97% of GEMMs handled by the Marlin actuator. Result vs CP 0.4:

| N | test A agg tok/s | CP 0.4 | lift |
|---|---|---|---|
| 4  | 24.07 | 12.76 | 1.89× |
| 16 | 14.82 | 8.86  | 1.67× |
| 64 | 13.44 | 8.89  | 1.51× |

The CP 0.4/0.5 ~8.5 tok/s ceiling is **lifted** — uniform 1.5–1.9× at matched
N. Honest residual: aggregate still *declines* with N (the registry shifted
the curve up, did not make it scale) — that decline is the eager-mode
launch-overhead floor (CP 4.3 persistent-kernel scope, not Marlin).

### ✅ MARLIN SUB-TASK CLOSED (2026-05-15)
Mechanism proven (test B: race-free, N-way slots, serialization removed),
end-to-end ceiling lifted (test A: 1.5–1.9×). Full evidence:
`marlin_regression/REGRESSION_NOTE.md`. Marlin's per-call tok/W lift (scorecard
1.62×) is measured later as memo §6 gate criterion (b), not as a separate
sub-task.

### DVFS envelope sweep — UNBLOCKED (kmod 0.4.8, 2026-05-15)

The DVFS sweep harness is built (`dvfs_envelope/run_dvfs_sweep.sh`,
`envelope_driver.py`, `analyze_dvfs.py` — adapted from the T4.3 envelope
harness). It was blocked: VOLT couldn't actuate — NVML clock-lock is
`NOT_SUPPORTED` for a user process, and the kmod-ioctl fallback needs
`/dev/cipher`, which CP 3.3's kmod reload left at `0600 root`.

**Regression audit run first** (`../REGRESSION_AUDIT_2026_05_15.md`,
independently re-verified this session):
- Check 1 — T4.6.4 KV-dedup 5 indicators: **PASS 5/5**.
- Check 2 — T4.5.1 `.symver` cuBLAS capture: **PASS**.
- Check 3 — Op #1 allocator unit test: **PASS 14/14**.
- Check 4 — `/dev/cipher` access: **FAIL** (the one regression). The audit's
  first Decision section had a wrong premise (corrected in the report): the
  kmod's ABI *documents* 0666 (`cipher_ioctl.h:249`, `cipher_clock.c`), and
  `SET_CLOCK_MHZ` has no CAP check by deliberate T4.3.2 design.

`chmod 0666` workaround **rejected** — fix codified in the kmod instead.

**RESOLVED — user adjudicated Posture A** (keep T4.3.2's design). kmod
**0.4.7 `2a69f9de` → 0.4.8 `e2f50452`**: `devnode` callbacks codify
`/dev/cipher` + `/dev/cipher_kvdedup` at 0666 (reload-stable); stale
`cipher_clock.c` CAP comment fixed. CP 3.3 gate re-verified on 0.4.8 — all 4
PASS (`cp_3_3/reverify_0_4_8/`); regression checks 1/3 re-PASS. Rollback:
`kmod_0.4.7_pre_devnode.ko`. **DVFS sweep now runs — non-root, as T4.3.2
intended.**

### DVFS sweep — harness windowing fix + BLOCKER found (2026-05-15)

**Harness windowing fix — DONE & verified.** The sweep harness averaged power
over the wrong window: the sampler started at python-launch, so watts.csv
covered model load + ~12 s Marlin NVRTC compile + warmup, not the decode
loop — and the Marlin cubin-compile asymmetry (12 s first run, 0.1 s cached)
broke the matched pairs. Fix (user-adjudicated, Option 1 — sentinel-windowed):
`envelope_driver.py` writes a `DECODE_START`/`DECODE_END` sentinel at the
decode-loop boundary; `run_dvfs_sweep.sh` `run_one` launches the power sampler
on `DECODE_START` and stops it on `DECODE_END` — watts.csv is now exactly the
decode window. `analyze_dvfs.py` trim cut 5 s/2 s → 1 s/1 s. Verified: the
isolation run below produced a clean `DECODE_START 1778871449.487 /
DECODE_END 1778871461.723` 12.24 s window. Clock grid reordered to 1200 first.

**BLOCKER — Marlin actuator hangs when the v2 substrate is co-active.**
The M1200 smoke hung; gdb on the stuck process showed the main thread in
`cipher_rt_marlin_engine_quantize_repack` → `cudaMemcpy` DtoH, GPU pegged 100 %,
decode loop never reached. Isolation test (both with the substrate active
post-kmod-fix):
- `CIPHER_MARLIN=on`  → **HANG** in `quantize_repack`.
- `CIPHER_MARLIN=off` → decode **completes**, 34 tok/s, sentinel clean.

So the Marlin INT4 actuator hangs **when the v2 substrate (GREEN ctx + ARB,
now active because kmod 0.4.8 restored `/dev/cipher`) is co-active**. The
substrate alone is fine; Marlin alone was fine in CP 2.4 test A — but test A
ran pre-kmod-fix (substrate OFF). This is the first time `CIPHER_MARLIN=on`
met the live substrate. **Implication beyond DVFS:** the Marlin sub-task was
closed on test A's substrate-OFF validation; post-kmod-fix that closure has a
caveat — any `CIPHER_MARLIN=on` workload hangs with the substrate active.

The DVFS sweep needs `CIPHER_MARLIN=on` (workload = post-Marlin-INT4). It is
**BLOCKED** pending a fix in `libcipher_rt` (the `a0d6cdda` anchor) — needs
user adjudication. Anchors held meanwhile: kmod 0.4.8 `e2f50452`, libcipher_v2
`86618c30`, libcipher_rt `a0d6cdda` — all untouched. No CP 2.4 report.

> **CORRECTION — 2026-05-15 (supersedes the "v2 substrate is co-active"
> framing above).** The substrate-interaction framing was **wrong** — proven so:
> - `MARLIN=on` + `CIPHER_SUBSTRATE=off` retest → **still hung**.
> - test A retest, post-kmod-fix, `SUBSTRATE=off` → **still hung**.
> - The green context initializes regardless of the substrate gate (the gate
>   covers ARB/SMP/PR; green-ctx has an independent trigger).
>
> Root cause, confirmed by full gdb backtrace + source elimination: the hang
> is the **Marlin quant path's mixed driver-API NULL-stream `cuLaunchKernel`
> + runtime-API `cudaMemcpy`** in `quantize_fp16_to_int4_groupwise_gpu`. Post-
> kmod-fix, CUPTI subscribes (it couldn't pre-fix — `/dev/cipher` was EPERM);
> its per-launch callback runs the T4.2.4d enforcement `green_ctx_make_current`
> → `cuCtxSetCurrent(green)` on every `cuLaunchKernel`. Marlin's quant kernels
> launch via `cuLaunchKernel`, so each lands in the green context, then the
> synchronous `cudaMemcpy` DtoH hangs on the green/primary context split.
> Pre-fix CUPTI never subscribed → green never current (`ctx_swaps_to_green=0`)
> → quant path ran in the primary context → test A passed. Hypothesis (a)
> cuBLAS-shim reentrancy: eliminated (single `cipher_rt_matmul_dispatch`
> frame; quant path has no cuBLAS). Full detail → `MARLIN_HANG_ROOT_CAUSE.md`
> (written once the fix shape is adjudicated). Fix-shape decision pending.

### Marlin hang — RESOLVED (Fix A, 2026-05-16)

Root cause confirmed by an instrumented build: the Marlin GEMM kernel
(`grid = SM count = 132`, persistent-style, inter-CTA split-K `locks`) was
launched into the 8-SM green context — kmod 0.4.8 restored `/dev/cipher`,
which activated the CUPTI-driven T4.2.4d green-context enforcement. 132
blocks into 8 SMs → inter-CTA deadlock. Full trail: `MARLIN_HANG_ROOT_CAUSE.md`
and `MARLIN_HANG_INSTRUMENTATION.md`.

**Fix A applied** (user-adjudicated): `PrimaryCtxGuard` (RAII) now wraps both
`cipher_rt_marlin_engine_quantize_repack` and `cipher_rt_marlin_engine_dispatch`
— all Marlin engine GPU work runs in the device-0 primary context (132 SMs),
restored on every exit. `libcipher_rt` **`a0d6cdda` → `5e304549`** (clean
build, instrumentation stripped) — **new campaign anchor.** Verified: smoke
`CIPHER_MARLIN=on` + substrate ON completes (GEMM confirmed in the primary
context); post-kmod-reload smoke clean (0 ENOSPC); CP 3.3 gate 4/4; regression
Checks 1/3 PASS.

**Marlin sub-task — CLOSED, unconditionally.** Fix A makes Marlin's GPU
execution run in the full 132-SM primary context — *identical* to the context
the original test A measurement ran in (pre-kmod-fix, green context never
current). The §2.3 sub-task result therefore stands without the post-kmod-fix
caveat: mechanism proven (test B), end-to-end ceiling lifted (test A 1.5–1.9×),
and Fix A guarantees that execution context regardless of substrate state.
The multi-tenant N-sweep retest did not re-run cleanly (unrelated kmod
partition-allocator `ENOSPC` — `../PHASE_5_ALLOCATOR_DIAGNOSTIC.md`); it was a
sanity check, not a gate criterion (the gate is single-tenant).

**Phase 5 forward work** (does not affect CP 2.4): the Marlin GEMM kernel is
structurally full-GPU and cannot run in an SM partition — Marlin and
green-context partitioning do not compose. See
`PHASE_5_MARLIN_PARTITION_CONSTRAINT.md`.

### Gate criteria (from approved memo §6) — none measured yet
(a) Marlin lock-fix regression · (b) per-lever lift n=5 matched pairs ·
(c) composed lift n=5 · (d) Llama ≥2.7× verdict + Mistral measurement ·
(e) anchors held.

### Sub-task (iii) — Speculative decode — STARTED (2026-05-16)

Design memo `SPEC_DECODE_DESIGN_MEMO.md` **approved**. Adjudicated decisions:
1. Integration = generate-path injection (operator-policy, like op #1's cache
   class). No `CipherSpeculativeModel` wrapper — would force tenant code
   changes, breaking the unmodified-workload product story.
2. Gate = **both arms, Llama primary**. Llama-3.1-8B + Llama-3.2-1B draft is
   the pass criterion: **1.75× lift over Marlin-alone** (not unaccelerated),
   n=5 matched pairs + 95% CI. Mistral-7B + n-gram draft is a *secondary*
   measurement — document the lift, no target. Two arms, two findings.
3. k = adaptive on running acceptance rate (EMA), `k = clamp(2, round(
   target_accepts / EMA_accept_rate), 8)`; EMA window documented in code.

Draft placement corrected: own CUDA stream, **primary context** (Marlin's
full-GPU pin applies to the draft too — it is a transformer forward through
Marlin-routed shapes). Atomic STEP, ~1.5–2 weeks (campaign calendar
dominator); one report at STEP close, part of the CP 2.4 close.

**Build status:** core module `cipher_rt_phase4/cipher_spec_decode.py` —
engine written; GPU verification deferred until the DVFS sweep finishes (no
GPU contention — the sweep is a power/throughput measurement).

GPU-free progress, 2026-05-16:
- **Unit tests** (`test_cipher_spec_decode.py`): **22/22 PASS** — AdaptiveK
  (seed, clamp, low/mid/high regimes, mixed-rate bounds), NgramDraft lookup,
  greedy_accept (full/partial/empty/k=1/all-rejected/k_max boundary). Pure
  logic CPU-verified.
- **n-gram tuning (item 2): SKIPPED** — no cached Mistral-7B token-id traces
  on disk (the `t4_6_3_dedup/traces/*.jsonl` are KV-dedup hash-id traces, not
  token sequences). n stays at the memo default 3; tune during GPU measurement.
- **Tokenizer check (item 1) + Llama-3.2-1B download (item 4): BLOCKED** —
  no Llama-3 models on disk or `/lambda/nfs`, no HF auth token, and
  Llama-3.1-8B / 3.2-1B are gated. The **Llama arm is the primary gate
  criterion** — needs an HF token (Meta license accepted) or the models
  placed on disk before the Llama-arm build/measurement can run.

Anchors held: kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`,
libcipher_rt `5e304549`.

### DVFS envelope sweep — RESULT (2026-05-16) — verdict: DVFS IN

Sweep complete: 5 clock points × 5 matched pairs × 2 arms = 50 clean
sentinel-windowed 60 s decode runs (Mistral-7B B=1, `CIPHER_MARLIN=on`,
substrate ON, libcipher_rt `5e304549`). `analyze_dvfs.py`:

| clock | off tok/W | on tok/W | Δtok/W % | 95% CI | on tok/s ratio | on W ratio |
|---|---|---|---|---|---|---|
| 1000 | 0.2943 | 0.4782 | **+62.45** | +62.0..+62.9 | 1.003 | 0.618 |
| 1200 | 0.2929 | 0.4417 | +50.80 | +49.3..+52.3 | 0.988 | 0.655 |
| 1400 | 0.2951 | 0.3848 | +30.40 | +29.6..+31.3 | 1.000 | 0.767 |
| 1600 | 0.2929 | 0.3342 | +14.09 | +12.9..+15.3 | 1.010 | 0.885 |
| 1800 | 0.2934 | 0.2996 | +2.10 | +1.3..+2.9 | 1.007 | 0.986 |

**Verdict (memo §3.3): DVFS IN** — every clock point's 95% CI lower bound
> 0. Mechanism is clean: **tok/s ratio ≈ 1.00 at every point** (locking the
SM clock does not cost throughput → the post-Marlin-INT4 decode is
memory-bandwidth-bound, as the memo §3.2 thesis predicted); power drops up to
38% (W ratio 0.618 at 1000 MHz). Result: `dvfs_envelope/dvfs_envelope_result.json`.

**Honest flag — magnitude.** +62% at 1000 MHz is ~15× the memo §3.2
expectation (+4.2% scorecard contribution) and overturns the T4.3 Mistral-7B
**FP16** envelope (−2..−14%). Most likely reconciliation: this sweep used the
**corrected sentinel-windowed harness** (the windowing bug fixed this
session); the T4.3 harness is its ancestor and carried that bug, so the prior
−14%/+4.2% numbers are suspect. This measurement is internally coherent —
flat tok/s, strictly monotonic envelope, tight CIs, off-arm reproducible
across 25 independent runs (~0.293 tok/W).

**Null-check — PASS (2026-05-16).** `run_dvfs_nullcheck.sh`: same harness,
n=5 matched pairs, **both arms VOLT=off** (nominal, no lock). Result:
**Δtok/W = −0.02 %, 95 % CI [−0.99 %, +0.95 %]** — straddles 0, CI width ~1.9
(within the sweep's 0.9–3.0 range). The harness reports ≈0 for a no-op → no
residual bias. **DVFS verdict IN is confirmed, no longer provisional.** The
T4.3 Mistral-FP16 −2..−14 % envelope and the scorecard's +4.2 % DVFS
contribution are **retracted** — pre-fix-harness windowing-bug artifacts,
superseded by this null-validated measurement. Full detail: `DVFS_NULL_CHECK.md`.
DVFS is a major composed-gate lever (+62 % tok/W at 1000 MHz), not a rounding
term — the §6c math accounts for this.

### Sub-task (iii) — spec_generate fixed + CPU-verified (2026-05-16)

`spec_generate` verify loop **rewritten** — three bugs found by code review,
fixed: (a) logit/draft alignment off-by-one, (b) bonus-token index range,
(c) KV crop off-by-one. The fix feeds `[pending] + proposed` each round
(`pending` = last committed token, KV not yet in `past`) so `res.logits`
yields a length-(k+1) `tgt_full` aligned 1:1 with `proposed` — bonus index
n∈0..k always in range, KV crop keeps exactly `pending`+`accepted(n)`.

**CPU integration test `test_spec_generate.py` — 12/12 PASS.** Deterministic
mock target (RULE Markov-2, tracks its seen sequence via `past` so a wrong KV
crop diverges) + scripted drafts. Covers: all-accepted at k_max, all-rejected
at k=2, partial accept (n=1/3/5), multi-round 200-token (cross-round state),
and token-agreement vs target-only at temp 0 for oracle/wrong/partial/ngram
drafts. Pure-helper unit test still 22/22. `install()` implemented (operator-
policy monkey-patch of `GenerationMixin.generate`).

`spec_smoke.py` (Mistral-arm GPU smoke + real token-agreement) launched;
Llama-3.1-8B / 3.2-1B downloads retrying (first attempt hit a transient Hub
connection error). Anchors held: kmod 0.4.8 `e2f50452`, libcipher_v2
`86618c30`, libcipher_rt `5e304549`.

### Sub-task (iii) — Mistral-arm correctness verified; lift measurement running (2026-05-16)

**Correctness — verified.** `spec_smoke.py` / `spec_verify.py`: two confounds
resolved — (1) Marlin lazy quantization (warm Marlin before the reference),
(2) GPU greedy is **not bit-deterministic** — plain `generate(do_sample=False)`
run 3× on the identical prompt diverges from *itself* at logit near-ties
(prompt 0 @ idx 6, prompt 1 @ idx 9; prompt 2 deterministic). `spec_generate`
diverges from stock **only at those same indices**, never where stock is
deterministic. With the CPU integration test (12/12, deterministic mock →
exact byte-identical) this confirms `spec_generate` is correct: it adds zero
divergence beyond GPU fp-tie noise. The temp-0 token-agreement gate is **met**
(criterion honestly reframed — see `SPEC_DECODE_CORRECTNESS.md`).

**Lift measurement — running.** `run_spec_measure.sh` (n=5 matched pairs,
CIPHER_SPEC off vs on, both Marlin-on, Mistral-7B B=1 decode) — the secondary
criterion; analysis → `spec_measure_result.json`.

**Llama models — ON DISK (2026-05-16).** Meta license granted on Anil0666 +
new Read token; both downloaded to `/home/ubuntu/models/`:
`Llama-3.1-8B` (15 G, 4 safetensors) and `Llama-3.2-1B-Instruct` (2.4 G, 1
safetensors) — config + tokenizer present, `original/` excluded. The Llama
arm (primary 1.75× criterion, n=5 over Marlin-alone) is **unblocked** — runs
after the Mistral lift analysis. Anchors held: kmod 0.4.8 `e2f50452`,
libcipher_v2 `86618c30`, libcipher_rt `5e304549`.

### Sub-task (iii) — varied-prompt methodology + ModelDraft fix (2026-05-16)

**Pathological-best finding.** The first Mistral-arm measurement
(`spec_measure_result.json`) used a single trivial prompt — `"Hello, world."`
greedy ×32. Mistral loops into repetitive text, so the n-gram draft hit at
**accept 0.992 → 1.50× lift** — ~1.7× the memo §4.3 expectation (~0.59) for
n-gram on real text. **1.50× is workload-favorable, not representative.** It
is kept as a documented **upper-bound artifact** (`run_spec_measure.sh`,
`spec_measure_result.json`); the honest secondary number is the varied-prompt
re-measurement.

**Varied prompt set adopted (user-directed).** 5 prompts ×
factual/narrative/code/reasoning/conversational, 128 tokens each; n=5 matched
pairs → **25 lift points/arm**, flat mean ± 1.96·SE, per-prompt subgroup means
also reported (the 25 points are not i.i.d. — documented). Rationale +
pathological-best artifact → `SPEC_DECODE_METHODOLOGY.md`.

**ModelDraft KV-reuse bug — FOUND & FIXED.** Code review (pre-GPU) found
`ModelDraft.propose()` carried `self._kv` across `propose()` calls *and across
`generate()` calls* — at round 2+ the saved KV holds the rejected draft tokens
and misses the committed bonus; across prompts it is a different prompt
entirely. Never exercised (the ModelDraft GPU path was deferred). **Fix:**
`propose()` is now **stateless** — a fresh full-sequence draft prefill each
round. Unconditionally correct; for a 1B draft cheap (~few ms). A feedback-
keyed incremental draft KV is a ~10–20 % Phase-5 optimisation — deliberately
off the gate path (same off-by-one surface as the verify-loop bugs). The
Llama-arm lift is therefore measured with a **conservative draft** — a slight
underestimate; a pass at 1.75× is a robust pass. Also added: `_load_draft`
now **asserts draft/target `vocab_size` match** (was promised in a docstring,
not enforced); `_last_stats` exposes per-call accept stats to the driver.
`cipher_spec_decode.py` only — no lib rebuild, anchors untouched.

**Harness:** `run_spec_varied.sh` / `spec_varied_driver.py` /
`analyze_spec_varied.py` — parameterised by TAG/TARGET/DRAFT for both arms.
First launch caught a warm-up bug: a 16-token warm left the first 128-token
generate paying a ~3.7 s cuDNN/Marlin JIT → prompt 0 read 20.6 tok/s vs ~50,
corrupting prompt-0's lift. **Fixed:** warm now runs the full 5-prompt set at
the 128-token budget before timing.

**Order:** (1) Mistral re-measurement on varied prompts — RUNNING; (2) Llama
smoke (`spec_llama_smoke.py` — 5 checks: draft load, vocab align, spec
completes, accept plausible, token-agreement) then Llama n=5 gate; (3)
composed gate; (4) CP 2.4 report.

**Composed-gate baseline (pinned, from memo §6):** composed = per-lever
product — Marlin 1.62× × spec lift (throughput); DVFS adds tok/W on top
(DVFS is IN, flat tok/s). Step-3 composed measurement: all-on
(Marlin+spec+DVFS) vs **vanilla HF**, report composed tok/s lift + tok/W
lift, and validate tok/s lift ≈ Marlin-lift × spec-lift (clean composition).

Anchors held: kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`,
libcipher_rt `5e304549`.

### Sub-task (iii) — Llama-arm BLOCKED: cuDNN-attention × Marlin hang (2026-05-16)

The Llama-arm smoke hung ~indefinitely (killed at 1 h 01 m, GPU pegged).
gdb stack (sudo) + **6-run bisection** — full detail `CUDNN_ATTN_MARLIN_HANG.md`.
Stack: hang in **cuDNN SDPA `cuKernelSetAttribute`** (driver spin) — NOT a
Marlin kernel deadlock, NOT the spec logic. The bisection scoped **two
distinct problems**:

- **3a — the hang = the draft's separate CUDA stream.** A(hang) vs T3(pass)
  differ only in whether the model draft runs on its own `torch.cuda.Stream()`
  or the default stream. T1 (stock) + T2 (n-gram spec) pass — not a broad
  Marlin×cuDNN issue. **One-line fix:** run the draft on the default stream
  (the own-stream gave zero benefit — draft↔target handoff is fully synced).
  Knob added: `CIPHER_SPEC_DRAFT_STREAM` (default 1; 0 → default stream).
- **3b — SEPARATE: model-draft acceptance collapses under the substrate.**
  T3 (no hang) shows `accept_rate` **0.036** vs **0.490** substrate-OFF (R1) —
  same draft, prompt, 64 tok. T3 took 53.9 s vs stock T1 36.2 s → **model-
  draft Llama arm = negative lift under the substrate.** Not yet root-caused
  (leading hypothesis: Marlin INT4 quantizing the 1B draft + 8B target so the
  draft no longer tracks the target). NOT a one-line fix.
- R1 confirmed the **stateless `ModelDraft` fix is correct** (substrate-OFF:
  byte-identical to stock, accept 0.490, 3.2 tok/round).
- **n-gram Llama arm WORKS** (T2 — clean under the full substrate).
- **Blocks the model-draft Llama primary 1.75× criterion.** Awaiting user
  adjudication: Option 3 (n-gram Llama arm — viable now, two n-gram arms +
  composed gate), Option A (diagnose+fix 3b — open-ended), Option B (scope
  Llama out).

### Sub-task (iii) — Mistral arm varied-prompt RESULT + greedy-loop finding (2026-05-16)

`spec_varied_mistral_result.json` — n=5 pairs × 5 prompts × 128 tok, 25
matched points:

| metric | value |
|---|---|
| **Mistral-arm spec lift** | **1.639× — 95 % CI [1.602, 1.677]** (SD 0.096) |
| lift driver | 1.96 tokens / verify round |
| per-prompt | factual 1.51× · narrative 1.67× · code 1.69× · reasoning 1.65× · conversational 1.69× |

Secondary criterion (n-gram draft) — **lift documented, no pass target.**

**Finding — greedy decoding loops; `accept_rate` is a loop signature.** The
n-gram conditional acceptance (`accepted/proposed`) pinned at **exactly 1.000
on all 25 generations.** Cause: under greedy (temp-0) decoding — required by
the correctness gate — a base model's output collapses into **exact
repetition within ~30–50 tokens on every prompt genre**; the n-gram draft
proposes only on an exact n-gram match and then replays the loop perfectly.
So:
- `accept_rate` is NOT draft quality — the honest driver is **tok/round**
  (`gen_tokens/rounds`, counts miss rounds). `analyze_spec_varied.py` reports
  both; the driver now records raw `accepted`/`proposed`/`rounds`.
- The original `"Hello,world."`×**32** measurement (1.50×) was **NOT an upper
  bound** — varied×**128** gives 1.64× (*higher*). Generation length is the
  first-order lift factor (more tokens → more in-loop fraction); prompt genre
  is second-order. `SPEC_DECODE_METHODOLOGY.md` §1.1–1.2 corrected.
- The lift is a **greedy-decode-regime** number; sampled decoding (temp>0)
  falls through to stock — outside spec's scope. Stated in the CP 2.4 report.
- **Implication for the Llama gate:** Llama-3.1-8B greedy×128 will also loop.
  But the Llama arm uses a *model* draft (1B) — unlike n-gram it also proposes
  correctly on pre-loop novel text, so its lift is not purely loop-driven. The
  1.75× gate is measured greedy-128 as specified; the loop component is
  documented, and `analyze_spec_varied.py` reports tok/round to expose it.

Anchors held: kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`,
libcipher_rt `5e304549`.

### Sub-task (iii) — Option 3 adjudicated: n-gram Llama arm (2026-05-16)

User adjudicated **Option 3** — close CP 2.4 with two n-gram arms; the
model-draft 1.75× primary criterion is NOT met (deferred to Phase 5).
Rationale: the 1.75× target was scorecard-anchored; under greedy decode
(the correctness regime) n-gram and model-draft converge in the loop, and
the model draft's value is pre-loop (a small fraction of 128 tokens). The
3b acceptance collapse is Phase-5 architectural work (same drawer as
Marlin × green-context).

Execution:
- **Stream fix shipped** — `CIPHER_SPEC_DRAFT_STREAM` default flipped to `0`
  (model draft on the default stream). R1 regression smoke re-run on the new
  default: PASS — model-draft, no substrate, default stream → byte-identical
  to stock, accept 0.490. `cipher_spec_decode.py` only (Python; no lib
  rebuild, no anchor drift).
- **Llama n-gram arm** — `run_spec_varied.sh TAG=llama DRAFT=ngram`, n=5
  varied prompts, RUNNING. Llama-8B decode confirmed full-speed (~51 tok/s;
  the slow hang_bisect numbers were cold-start, the harness warms properly).
- **Composed-gate harness built** — `run_composed.sh` + `analyze_composed.py`:
  Mistral-7B, n=5, three arms (vanilla / marlin / all-on = Marlin+spec+DVFS
  @1000 MHz), sentinel-windowed power → composed tok/s + tok/W lift, with the
  lever-decomposition cross-check. `spec_varied_driver.py` gained
  DECODE_START/END power sentinels (harmless to the spec-arm runs).
- **Phase 5 doc updated** — `PHASE_5_ARCHITECTURE_REVISION.md` §10 records the
  substrate × model-draft Marlin INT4 constraint alongside §2 (Marlin ×
  green-context); both feed the Song Han engagement scope.

Anchors held: kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`,
libcipher_rt `5e304549`.

### STEP CLOSE — Llama n-gram arm + composed gate DONE; CP_2_4_REPORT.md landed (2026-05-16)

- **Llama n-gram arm — DONE.** `spec_varied_llama_result.json` (md5 `370ec185`):
  **1.597× [1.514, 1.680]**, n=25, SD 0.213, tok/round 1.93. One transient
  outlier (pair 4 / prompt 0 = 0.690×, 35 tok/s on-arm — on<off, physically
  impossible at conditional-acceptance 1.000). Pre-stated rule excludes it as a
  documented sensitivity check: **n=24 → 1.635× [1.595, 1.675], SD 0.100**.
  Not re-run (secondary criterion, no pass target). Both n-gram arms agree.
- **Composed gate — DONE.** `composed.runlog` (md5 `19fe3af1`) — one clean
  15-invocation run (5 pairs × 3 arms), `composed_result.json` (md5 `9b126710`):
  - all-on/vanilla **tok/W 3.617× [3.591, 3.642]**, tok/s 1.795× [1.777, 1.813]
  - marlin/vanilla tok/W 1.307×, tok/s 1.090× (Marlin UNDER scorecard 1.62× —
    B=1 weak regime; this run also serves as criterion (b)'s Marlin per-lever)
  - all-on/marlin tok/s 1.647× — reproduces standalone Mistral n-gram 1.639×
  - composition clean multiplicative: 1.090 × 1.647 = 1.7955× == composed tok/s
  - **VOLT verified** — all-on watts.csv clock = 1005 MHz; vanilla/marlin 1980.
  - Composed 3.617× tok/W exceeds the old 2.96× scorecard, but the lever mix
    differs (Marlin under, DVFS dominant) — attributed in the report, not
    reframed as "scorecard reproduced".
- **Launch-integrity incident** (logged in report §4): two earlier composed
  launch attempts aborted at pair 1 — a leftover prior-session `run_composed.sh`
  was duplicated, and a background wrapper hit a 2-min timeout while the script
  orphaned and survived. Both killed, GPU verified idle (0 MiB), `composed/`
  wiped before the single clean run. No reported number from a contaminated run.
- **`CP_2_4_REPORT.md` LANDED** — md5 `573c50d6`. Gate scorecard: (a) PASS,
  (b) PASS, (c) PASS, (d) NOT MET (Llama model-draft 1.75× — 3b collapse,
  Phase 5), (e) PASS. Honest partial close: 13/23 → 14/23 on adjudication.

**CP 2.4 atomic STEP COMPLETE.**

### ADJUDICATED — CP 2.4 CLOSED, CP 2.5 STARTED (2026-05-16)

User adjudicated: **CP 2.4 closed as honest partial — 13/23 → 14/23.**
Marlin + DVFS + n-gram spec shipped and composed (3.617× tok/W); model-draft
spec deferred to Phase 5 with documented root cause. No re-measurement
directed. → CP 2.5 next (Phase 2 close: 14/23 → 15/23).

**CP 2.5 STARTED** — scope: drop LD_PRELOAD in favour of
CUDA_INJECTION64_PATH-only deployment; produce `cipher-platform.deb` for clean
install on neocloud nodes. The deployment story for the May 28 Ditlev demo.
Atomic STEP, design-memo-first. Calendar 3–5 days. Progress tracked in
`cp_2_5/PROGRESS.md` (new). Anchors held: kmod 0.4.8 `e2f50452`, libcipher_v2
`86618c30`, libcipher_rt `5e304549`, taint 12288.
