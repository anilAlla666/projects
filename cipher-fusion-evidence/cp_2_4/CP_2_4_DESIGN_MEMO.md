# CP 2.4 — Design Memo: Marlin + DVFS + speculative ported to the v2 stack

**Date:** 2026-05-15. **Status:** DESIGN — awaiting approval. **No code is
written until this memo is approved.**

CP 2.4 canonical scope: *"Marlin + DVFS + speculative ported, 2.96× tok/W
reproduces through new dispatch."* The decisive Phase-2 CP — it delivers the
*payload* (perf actuators on the v2 dispatch), the thing Phase 2 plumbing-only
never delivered.

This memo is long and front-loads three findings that change the shape of the
work. Read §0 first — it contains a workload conflict that needs your decision
before the gate can be defined.

---

## 0. Audit-before-build findings

### 0.1 What is ALREADY in the v2 stack (`cipher_rt_phase4/`)

The v2 injection lib is further along than "port from scratch" implies.
`cipher_inject.c` (`InitializeInjection2`) already wires:

```
cipher_rt_volt_init()           — DVFS / VOLT  (T4.3.1)
cipher_rt_matmul_dispatch_init()— matmul-routing substrate  (T4.5)
cipher_rt_marlin_init()         — Marlin actuator  (T4.5.2)
cipher_rt_attn_dispatch_init()  — attention substrate  (T4.6.1)
```

- **Marlin** — `cipher_rt_marlin_engine.cpp` + actuator + kernel_src are in the
  v2 tree. Marlin is *integrated*, not unported. Env-gated `CIPHER_MARLIN=on`,
  **OFF by default**.
- **DVFS** — `cipher_rt_volt.c` is in the v2 tree, derived from the op31
  `cipher_volt.cpp`. Env-gated `CIPHER_VOLT=on` + a target
  (`CIPHER_VOLT_BATCH` / `CIPHER_VOLT_MHZ`), **OFF by default**. The kmod ioctl
  it can fall through to (`CIPHER_SET_CLOCK_MHZ`, nr 10) shipped in T4.3.2.
- **Speculative decode** — **absent.** There is no spec-decode source in the v2
  tree. The only artifact is the op31 `spec_decode_measure.py`, an *n-gram
  feasibility study* (M1 draft-match 0.59, M2 batch-overhead 1.13×, M3 power
  headroom 116 W) — **not a working actuator**. The 2.96× scorecard used a real
  **Llama-3.2-1B draft + Llama-3.1-8B target**. So "port speculative" is in
  truth **build speculative decode from near-zero**.

**Consequence:** CP 2.4 is not three ports. It is (1) a Marlin **lock-saturation
fix**, (2) a DVFS **wire-and-characterize**, (3) a speculative-decode **build**.
The third dominates the schedule — see §9.

### 0.2 CP 2.1 open item — RESOLVED

You asked CP 2.4's audit STEP to confirm whether `libnccl-tuner-cipher.so`
supersedes the LD_PRELOAD `ncclAllReduce` port. **It does.** `nm -D` shows it
exports `ncclTunerPlugin_v1` and `ncclTunerPlugin_v2` (type `D` — the NCCL
external-tuner-plugin ABI structs). That is the *supported* NCCL tuner
mechanism, no LD_PRELOAD required. **CP 2.1 inventory row 30 (`ncclAllReduce`)
moves PORT → DELETE.** (NCCL is outside CP 2.4's Marlin/DVFS/spec scope; this
only closes the CP 2.1 open item. An optional one-line addendum to
`CP_2_1_REPORT.md` row 30 + §10 can record it — your call.)

### 0.3 The workload conflict — needs your decision (see §8, decision 1)

The CP brief says two things that cannot both be the gate:

- **"2.96× reproduces through v2 dispatch."** The 2.96× is **Llama-3.1-8B**,
  B=1, 200-tok decode, H100 pinned **1200 MHz**, measured on the LD_PRELOAD
  stack (`SCORECARD.md`).
- **"Workload + gate: Mistral-7B decode."**

The 2.96× is a Llama-3.1-8B number. It cannot be literally reproduced on
Mistral-7B. §8 decision 1 lays out three options; the memo recommends one but
does not pre-decide.

---

## 1. The CP 2.1 PORT inventory as binding input

CP 2.1 classified the old `libcipher_hook.so` surface. The rows that bind CP 2.4:

| CP 2.1 item | Class | CP 2.4 obligation |
|---|---|---|
| `cublasLtMatmul` / `cublasGemmEx` — FP8 + **Marlin** payload | PORT → `rt` `.symver` substrate | Marlin already landed (T4.5.2). CP 2.4 fixes its lock saturation and verifies its lift. |
| `cipher_set_gemm_ptrs` / `cipher_tls_get_gemm_*` / `cipher_tls_relaunch` | PORT → `rt` substrate | The TLS GEMM-shape pipeline feeds Marlin shape selection. CP 2.4 must keep these consumers intact when the per-shape registry lands. **Prerequisite: enumerate external callers** (`cipher_wrapper.py` et al.) before touching the registry. |
| `cuLaunchKernel*` + persist + kernel-table | PORT → `v2` | DVFS clock decisions and spec-decode verify launches ride the v2 launch path; CP 2.4 does not re-port these (done) but composes on top. |
| `ncclAllReduce` | ~~PORT~~ → **DELETE** | Resolved §0.2 — not CP 2.4 scope. |

The FP8-compute substitution payload (CP 2.1 items 28–29) is **out of CP 2.4
scope** — CP 2.4's three named actuators are Marlin, DVFS, speculative. FP8 is a
separate payload; flagged, not done here.

---

## 2. Marlin port — per-shape context registry

### 2.1 The defect (measured, CP 0.4 / CP 0.5)

CP 0.4 (continuous) and CP 0.5 (burst) both hit a hard ceiling: **~8.5 tok/s
aggregate, peak MFU ~0.012 %**, flat across N=8…192 tenants. Root cause, quoted
from `CP_0_5_REPORT.md`: the Marlin path *"uses one module-scope workspace, so
the harness funnels all tenant Marlin GEMMs through a single `_MARLIN_LOCK` +
shared `_MARLIN_STREAM`."* Every tenant's compute becomes one serial GPU queue;
`s/burst` rises linearly with N — tenants spend their time *waiting in a queue*,
not computing.

The v2 Marlin engine carries the **same defect**:
`cipher_rt_marlin_engine.cpp` has a single `g_api_mu` (line 97) and a single
`g_weight_mu` (line 448). Porting Marlin to v2 without fixing this just moves the
serialization ceiling into the v2 stack.

### 2.2 The fix — per-stream context registry

> **AMENDED 2026-05-15 (after build-audit, before any code).** The original
> §2.2 below proposed keying the registry on Marlin *shape*
> `(m_blocks,n_blocks,k_blocks,group_blocks)`. Reading the actual CP 0.4/0.5
> regression harness (`density_sweep_a.py:3` — *"one shared Mistral-7B process,
> per-tenant StaticCache + CUDA stream"*) showed all N tenants run **one shared
> model** → they share the same ~10 Marlin kernel shapes. Per-shape keying
> therefore caps concurrency at ~10 lanes regardless of N and forces
> cross-stream sync. The contended resource is the single `g_marlin_workspace`
> `locks` buffer; each tenant already owns a distinct CUDA stream
> (`cipher_rt_cublas_shim.c:92` resolves it via `cublasGetStream_v2`). **The key
> is swapped to the caller's CUDA stream** — per-stream (≈ per-tenant) — giving
> N-way concurrency, the launch staying on the caller's stream (no cross-stream
> sync). User-adjudicated 2026-05-15. The bullets below read with
> "shape→stream"; the original wording is preserved for audit history.

Replace the single workspace/lock/stream with a **registry keyed on the
caller's CUDA stream** (≈ per-tenant) — the `stream` argument already threaded
into `marlin_gemm_launch`. (Original wording: *"a registry keyed on Marlin
shape `(m_blocks, n_blocks, k_blocks, group_blocks)`"* — superseded per the
amendment above.)

- Each registry **slot** is keyed on a caller `stream` and owns its own
  `locks` workspace buffer (sized to `MIN_WS`, covering N up to 32K) + a mutex.
- Slots are created **lazily** on first sighting of a stream; bounded count
  (64) with LRU eviction (stream pointers can be reused after destroy — LRU is
  acceptable for the gate; a stream-destroy callback closes it for production).
- The Marlin GEMM launches on the **caller's own stream** (unchanged) — no
  cross-stream sync. Each stream's `locks` buffer is private → no race.
- **N tenants → N streams → N private workspaces → N-way concurrency.** The
  CP 0.4/0.5 single-`_MARLIN_LOCK`/`_MARLIN_STREAM` serialization ceiling lifts
  and aggregate throughput scales with N.
- A registry mutex guards only slot insert/lookup (sub-microsecond); it never
  wraps the GEMM. `g_weight_mu` (per-weight quant cache) is left as-is — its
  critical section is a hashmap lookup, observably not the ceiling.

### 2.3 Marlin gate — sub-task (i) lock-fix + (ii) lift

- **(i) Lock-fix regression test:** re-run the CP 0.4 / CP 0.5 density sweep
  (`density_sweep_a.py` / `_b.py`) against the registry build. PASS = the
  ~8.5 tok/s aggregate ceiling lifts and aggregate throughput scales with N
  (target: super-linear departure from the flat plateau through at least
  N=32). The CP 0.4/0.5 curve is the regression baseline.
- **(ii) Lift verification:** Marlin INT4 reproduces its **1.62×** tok/W
  contribution (305.8 W → 122 W power, ~1.9× tok/s) through the v2 dispatch on
  the chosen gate workload (§6).

---

## 3. DVFS port — kmod ioctl exists; wire + characterize through v2

### 3.1 State

DVFS is **already wired** into the v2 inject path (`cipher_rt_volt_init()`),
and the kmod actuator ioctl `CIPHER_SET_CLOCK_MHZ` (nr 10, non-root) shipped in
T4.3.2. "Wire through v2" at the linkage level is **done**. What is *not* done:
the actuator is OFF by default (`CIPHER_VOLT` unset) and its contribution inside
the *composed* stack on the gate workload is unmeasured.

### 3.2 The honest DVFS problem

The T4.3 envelope is unambiguous: DVFS delivers **+55 % tok/W only on
memory-bandwidth-bound decode** (TinyLlama-class). On **Mistral-7B B=1 FP16**
it is **−2 % to −14 %** — neutral to negative. The 2.96× scorecard's DVFS
contribution was **+4.2 %**, and that was credible *because it was applied
after Marlin INT4 quantization* — the INT4-quantized model is ~4× lighter in
weight bytes, far more bandwidth-bound, so DVFS recovers a small positive. The
same effect should appear on a Mistral-7B-INT4 workload, but the **magnitude is
unmeasured**.

### 3.3 Build commitment — pre-gate DVFS sweep

Before DVFS enters the composed gate, run a **pre-gate clock sweep on the
post-Marlin-INT4 workload**: 4–5 `CIPHER_VOLT_MHZ` points × matched-pair tok/W.
Two branches:

- DVFS positive on Marlin-INT4 decode → include in the composed stack; record
  the contribution.
- DVFS neutral/negative → **set DVFS off in the composed gate**, document as
  "DVFS-disabled on this workload class," composed claim becomes
  "Marlin × spec, DVFS-off."

This makes the DVFS honest-fallback a *build step*, not a post-mortem.

---

## 4. Speculative decode — build (not port)

There is nothing to port (§0.1). CP 2.4 builds a speculative-decode actuator.

### 4.1 Draft-model placement

The 2.96× composition used **Llama-3.2-1B-Instruct draft + Llama-3.1-8B
target**, both on one H100. Design:

- Draft model resident on the **same GPU** as the target (the H100 has the
  memory headroom — M3 measured 116 W / ~50 % active-power headroom; a 1B INT4
  draft is ~0.7 GB).
- Draft runs in its **own CUDA stream / green context** so draft generation and
  target verify do not false-serialize — this reuses the `cipher_rt_green_ctx`
  partitioning already in the v2 stack.
- Draft generates **k tokens** (k≈4–5) autoregressively; the target verifies
  them in **one batched forward** — M2 measured B=1→16 batched forward at
  1.13× the B=1 cost, so a k=5 verify is near-free in latency.

### 4.2 Verify path

Standard speculative-decoding semantics: the target runs one forward over the
k draft tokens, producing k+1 logit vectors. Accept the longest prefix where
`sample(target_logits[i]) == draft[i]`; on first reject, resample that position
from the target distribution and discard the rest. Acceptance rate drives the
speedup; tok/W win comes from amortizing k accepted tokens over one target
forward + one cheap draft pass.

### 4.3 Decision needed — Mistral-7B has no clean 1B sibling

Llama-3.2-1B is a clean draft for Llama-3.1-8B (shared tokenizer/family).
**Mistral-7B has no first-party 1B draft.** If the gate is Mistral-7B (§8
decision 1), the draft strategy must be one of: a generic small Mistral-class
draft, an EAGLE-style self-speculative head, or n-gram-only (the op31 path,
~0.59 match). This is §8 decision 2 — and it materially affects the achievable
spec-decode lift on Mistral.

---

## 5. Composition order

Literature order for inference acceleration: **PagedAttention → quantization →
speculative decode** (KV-memory substrate first so quant and spec have room;
spec last because it multiplies an already-optimized step).

CP 2.4 adapts this: **Marlin (quant) is already in place**, and PagedAttention
is a separate track (T4.6). So the CP 2.4 build/measure order is:

**Marlin → DVFS → speculative**, with each lever measured **alone** (matched
pair vs baseline) and then **cumulatively** (the scorecard's
1.62× → 2.84× → 2.96× staircase). This ordering also matches the dependency
chain: DVFS calibration depends on the post-Marlin workload character (§3.2),
and the spec-decode verify cost is measured against the post-Marlin target.

---

## 6. Workload + gate

Gate workload (pending §8 decision 1): **decode, B=1, 200-token generation**,
H100 — clock pinned for the literal-reproduction arm, free-running for the
envelope arm. Power via NVML `nvmlDeviceGetPowerUsage`, decode-window-only
averaging (the T4.3.x matched-pair discipline — *not* the loose
whole-run averaging the original scorecard used; this is a deliberate
methodology tightening, stated, not hidden).

**Gate criteria:**

| # | Criterion | PASS condition |
|---|---|---|
| (a) | Marlin lock-fix | CP 0.4/0.5 density sweep re-run: ~8.5 tok/s aggregate ceiling lifts; throughput scales with N |
| (b) | Per-lever lift, v2 dispatch | Marlin, DVFS, spec each measured alone — **n=5 matched pairs**, 95 % CI reported per lever |
| (c) | Composed lift | Marlin → +DVFS → +spec cumulative tok/W, **n=5 matched pairs**, 95 % CI; the composed number is the headline |
| (d) | Reproduction verdict | composed lift vs the 2.96× target — reproduce, or honest-gap per §7 |
| (e) | Anchors held | kmod / libcipher anchors + ABI + taint unchanged (§10) |

n=5 matched pairs per condition; report mean ± 95 % CI; never a point estimate.

---

## 7. Honest fallback — if 2.96× does not reproduce

The 2.96× is a Llama-3.1-8B number with a fragile DVFS tail. The memo commits,
**before building**, to the following honest accounting:

1. **Report each lever's measured contribution separately** — Marlin ×, DVFS ×,
   spec × — with CIs. The composed number is the product of what was actually
   measured, not asserted.
2. **Most-likely outcomes** (stated now so the result is not reframed later):
   - Marlin INT4 ≈ **1.5–1.7×** — the most robust lever; INT4 weight traffic
     reduction is workload-agnostic. Expected to reproduce.
   - Speculative ≈ **1.5–1.8×** *if* a good draft is available; **1.1–1.3×** if
     forced onto n-gram drafting (Mistral, no 1B sibling). This is the widest
     uncertainty band and the largest single contributor.
   - DVFS ≈ **+0–5 %** on Marlin-INT4 decode; possibly **0×** (disabled) if the
     pre-gate sweep (§3.3) shows neutral/negative.
   - Composed plausible range: **~2.4× – ~3.0×**. The 2.96× is at the optimistic
     end and assumes a strong draft model.
3. **The gap statement:** if the composed v2 number is below 2.96×, the report
   states (a) the v2 number with CI, (b) which lever underperformed vs its
   scorecard contribution, (c) whether the gap is workload (Mistral vs
   Llama-8B), draft-model quality, DVFS-disabled, or v2-dispatch overhead — each
   is separately measurable, so the gap is *attributed*, not hand-waved.
4. **A reproduction on the literal scorecard workload** (Llama-3.1-8B, 1200 MHz)
   is the cleanest way to isolate "does the v2 dispatch itself cost lift" from
   "does the workload differ" — which is exactly why §8 decision 1 option C
   (measure both) is recommended.

CP 2.4 is **not** declared closed on a number alone — it is closed on an honest,
attributed measurement of what the v2 dispatch delivers.

---

## 8. Open decisions — your adjudication

**Decision 1 — gate workload.** The 2.96× is Llama-3.1-8B; the brief also says
Mistral-7B.
- (A) Literal reproduction: gate on **Llama-3.1-8B + 1200 MHz pin** (scorecard
  workload). Mistral-7B becomes a supplementary envelope measurement.
- (B) Gate on **Mistral-7B**; drop the literal "2.96×", set the target from the
  pre-gate per-lever measurement; the gate becomes "composed v2 lift reproduces
  the per-lever contributions."
- (C) **Both** — Llama-3.1-8B is the literal-reproduction gate; Mistral-7B is a
  secondary gate at a re-derived target. Two numbers, fuller envelope, cleanly
  separates "v2-dispatch cost" from "workload portability."
- **Recommendation: (C).** It is the only option that can answer "does the v2
  dispatch reproduce 2.96×" *and* honor the Mistral-7B instruction. Cost: one
  extra workload's worth of runs.

**Decision 2 — draft model** (only bites if Mistral-7B is in scope):
- (i) Llama-3.2-1B draft + Llama-3.1-8B target — matches the scorecard exactly;
  clean if Decision 1 is (A) or (C).
- (ii) For Mistral-7B: a generic small Mistral-class draft, or EAGLE-style
  self-speculative head, or n-gram-only.
- **Recommendation:** (i) for the Llama arm; for the Mistral arm, start with a
  small Mistral-class draft and fall back to n-gram if none is clean — decide
  during the build's spec sub-phase, report which was used.

**Decision 3 — CP 2.1 addendum.** Update `CP_2_1_REPORT.md` row 30 (NCCL) to
DELETE with the `ncclTunerPlugin_v1/v2` evidence? One-line edit. (Recommend yes,
for inventory accuracy.)

---

## 9. Calendar and risk — honest scope

You estimated 1–2 weeks. The audit revises this:

| Sub-task | Estimate | Risk |
|---|---|---|
| Marlin per-shape registry + lock-fix | ~3 days | Low — well-scoped; CP 0.4/0.5 is the regression test |
| DVFS wire + pre-gate sweep + characterize | ~2 days | Low — already wired; mostly measurement |
| **Speculative decode build** | **~1.5–2 weeks** | **High** — built from zero; draft-model choice is new design; verify-path correctness is the hard part |
| Composition + n=5 matched-pair gate | ~3 days | Medium — measurement discipline, two workloads if Decision 1 = C |

**Realistic envelope: 2–3 weeks, not 1–2.** Speculative decode dominates and
carries real risk. This remains **one atomic STEP, one report at the end** —
but executed across multiple sessions; the report lands when all gate criteria
are measured. Stated now so the schedule is not a surprise.

---

## 10. Discipline — anchors held

**CP 2.4 is pure userspace.** The per-shape Marlin registry lives in
`libcipher_rt.so`; DVFS uses the **existing** `CIPHER_SET_CLOCK_MHZ` ioctl
(nr 10, shipped T4.3.2); speculative decode is userspace (draft + verify).

- **No kmod source change. No ABI change.** kmod **0.4.7**
  `2a69f9defd7730665e6b7f9d60e82b43` — untouched. ABI ioctl nrs untouched.
- Anchors `55ab8c0c` (kmod) / `86618c30` (libcipher_v2) — frozen, untouched.
- `libcipher_rt.so` will be rebuilt (current `4f5cf543…`); each rebuild records
  a fresh md5, anchors unmoved — per the campaign anchor rule.
- Pre-CP-2.4 `libcipher_rt.so` saved outside the build dir as a rollback point.
- Taint expected stable at 12288.

---

## 11. New artifacts CP 2.4 will produce (on approval)

- per-shape Marlin context registry — new TU in `cipher_rt_phase4/`
- speculative-decode actuator — new TU(s) in `cipher_rt_phase4/`
- DVFS pre-gate sweep harness + composed-gate harness
- `libcipher_rt.so` rebuilt (new working baseline md5 recorded)
- evidence under `cipher-fusion-evidence/cp_2_4/`: per-lever + composed
  matched-pair data, CP 0.4/0.5 lock-fix regression, `gate_result.json`,
  one report `CP_2_4_REPORT.md`

---

**No code, no harness, no rebuild until this memo is approved.** Please
adjudicate §8 decisions 1–3 in particular — decision 1 defines the gate.
