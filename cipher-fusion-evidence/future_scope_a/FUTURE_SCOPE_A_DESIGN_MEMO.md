# FUTURE_SCOPE / A — CIPHER + vLLM COMPOSED ARCHITECTURE — DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/architecture — **paperwork only**, no
source modified, no GPU, no anchor rotation. **Phase 1 of FUTURE_SCOPE/A.**
STOP for adjudication before Phase 2 (GPU work).

Anchors at start: kmod `008b3c66`, libcipher_rt `83afd1ca`, libcipher_v2
`cc0479b8`, cipher_kv_bridge `c04b0c39`. vLLM **0.21.0** installed isolated at
`/home/ubuntu/vllm_env`.

---

## HEADLINE

CP 5.5's 100-tenant benchmark must run on a production inference engine (vLLM),
not CIPHER's HF-eager harness. FUTURE_SCOPE/A designs and will verify how
CIPHER composes *beneath* vLLM. **vLLM is the engine; CIPHER is the substrate
beneath it.** The composed architecture is **N isolated vLLM instances, each
pinned to a CIPHER SM partition, sharing one physical weight copy, under one
power envelope.** CIPHER's composed value is **process/model consolidation +
driver actuators (weight sharing, SM partitioning, DVFS) — not batching**:
vLLM already does continuous batching intra-process, so CIPHER must not batch
*over* it. **The headline metric is DENSITY** (76 % memory saving measured,
~95 % asymptote — the 100-agent-per-H100 enabler). TPW, TPS and MFU are real
but bounded, and stated honestly below — no aspirational numbers.

---

## §0 — CIPHER FOUR-METRIC SCORECARD (honest, per-cell)

The campaign measures four composable metrics. Each cell here is **a verified
measurement or explicitly bounded** — corrected 2026-05-19 after a claims
audit removed three unsupported figures (see [[cipher-fusion-campaign]]).

| metric | verified status (current substrate) | mechanism | honest bound |
|---|---|---|---|
| **Density** | **76 % HBM saving, Mistral-7B N=4** (Track 2 SC6, 2026-05-19); ~95 % large-N asymptote (formula `N·W/(N·W+(N+1)·C)`) | cross-tenant weight sharing — CUDA VMM POSIX-fd refcount, one physical copy | same-model only; verified TinyLlama + Mistral-7B |
| **TPW** | cross-tenant batching, strict-baseline re-test 2026-05-19: **3.06× Mistral N=4 / 3.30× TinyLlama N=8 / 5.98× N=16** vs naive; **~1.8× vs vLLM at N=8** (Phase C 3-arm) | cross-tenant batching (launch-overhead amortization) — the single TPW lever that survived re-test; DVFS adds a power-only component | batching regime only; vs naive 3–6×, vs vLLM ~1.8× and a **power** win (vLLM ~277 W vs CIPHER ~147 W), not a throughput-multiplier win |
| **TPS** | aggregate batched **498 / 750 / 990 tok/s** at N=8/12/16 (TinyLlama); per-tenant **~62 tok/s** | density × per-tenant throughput held flat as N rises | aggregate is a **density** story; per-tenant ~62 tok/s ≈ **par with vLLM** (vLLM is a comparable-or-faster single-stream engine) |
| **MFU** | measured live via CP 3.3 hardware FLOP telemetry; decode MFU is **low for every serving stack** (~0.6 % at B=32 — decode is memory-bound) | CP 3.3 per-tenant FLOP counter; substrate is MFU-**neutral** | **no positive-lift claim.** CIPHER's MFU story = "measured honestly + substrate does not degrade it" — verify neutral-vs-vLLM in Phase 3 |

**Honest composite read:** CIPHER **lifts density clearly**, lifts **TPW in
the batching regime** (~1.8× vs vLLM, a power win), and is **neutral-to-par**
on MFU and per-tenant TPS. "Lifts every term" is **not** the claim — density
is the headline; the rest are real but bounded.

**Two measurement contexts — do not conflate:**
- **(A) CIPHER-native path** — CIPHER's own batched executor vs naive HF. This
  is where 3–6× TPW and 498–990 tok/s come from.
- **(B) CIPHER composed with vLLM** — *this memo's architecture.* Here batching
  belongs to **vLLM**; CIPHER contributes density + isolation + power. The
  substrate-attributable lift vs vLLM-alone is **~1.8× tok/W + density
  enablement** — the 3–6× native number does **not** transfer to context (B).

## §0.1 — PER-PRIMITIVE × PER-METRIC IMPACT MATRIX

Verified cells carry the measurement; unmeasured cells are **DEFERRED to the
Phase that measures them** — never filled with aspirational numbers. In the
**vLLM-composed** architecture, batching is vLLM's — so there is no CIPHER
"batching" primitive row here (§2d, dropped).

| CIPHER primitive | MFU | TPW | Density | TPS |
|---|---|---|---|---|
| **Weight sharing** (Track 2) | neutral (no FLOP change) | **not a TPW lever** — capacity, not bandwidth amortization | **76 % saving verified** (native); composed-with-vLLM DEFERRED → Phase 4 | enables higher N → indirect; composed DEFERRED → Phase 4 |
| **SM partitioning** (CP 5.4) | per-instance MFU bounding DEFERRED → Phase 5 | **not a TPW lever** ([[cipher-lift-framing]]) | enables dense N-partition packing; composed DEFERRED → Phase 5 | bounds per-instance TPS under contention; DEFERRED → Phase 5 |
| **Dynamic SM migration** (Track 3) | neutral | not a lever | ~70 % less POOL stranding under churn (native, verified) → composed DEFERRED → Phase 5 | neutral |
| **DVFS / VOLT** | neutral | **power component** — +55 % tok/W on small-model bandwidth-bound decode only; neutral/negative on 7B; composed-with-vLLM DEFERRED → Phase 3 | none | none |
| *(batching)* | — | — | — | **provided by vLLM, not CIPHER** in the composed architecture |

The matrix makes the honest shape explicit: **in the vLLM-composed
architecture almost every quantified composed-context cell is DEFERRED** —
because CIPHER+vLLM has not been measured yet. That is precisely what Phases
2–6 exist to fill. The verified numbers above are CIPHER-native-path; the
composed numbers are the deliverable of this workstream, not a precondition.

## §1 — vLLM TRANSPARENCY HYPOTHESIS — and where it breaks

**Hypothesis:** vLLM runs unmodified; libcipher_rt is injected via
`CUDA_INJECTION64_PATH` (CP 2.5 made the substrate LD_PRELOAD-free —
GOT-patching from `InitializeInjection2` at `cuInit`). The kmod arbitrates SM
allocation; vLLM never knows.

| CIPHER intercept | when it fires | does vLLM's hot path go through it? |
|---|---|---|
| green-context SM partition (libcipher_rt init) | once, at `cuInit` | **yes, if** the green ctx is current before vLLM creates its context / captures CUDA graphs (§4 R1) |
| VMM weight arena (`cuMemCreate`/`Import`) | model-load time | **NO** — vLLM loads weights through PyTorch's caching allocator (`cudaMalloc`), not VMM (§2 weight-sharing row) |
| Marlin matmul routing (`.symver` cuBLAS) | per-GEMM | **largely NO** — vLLM uses its own fused/paged kernels, not cuBLAS GEMM; graph replay bypasses per-call hooks. Marlin regresses at B=1 and is scoped to B≥8 — **not in the vLLM-composed path** |
| DVFS / clock (NVML + kmod ioctl) | external actuator | **yes** — fully outside vLLM, composes regardless |

**Honest conclusion:** the transparency that holds is **isolation and power**,
not speed and not weight sharing. SM partitioning composes (with an ordering
constraint); DVFS composes cleanly; weight sharing is **not transparent**
(§2); batching must **not** be composed at all.

## §2 — COMPOSITION ARCHITECTURE PER PRIMITIVE

### 2a — CIPHER + vLLM, single tenant (transparency floor)
vLLM under `CUDA_INJECTION64_PATH=libcipher_rt.so`, substrate observing only.
**Metric impact:** MFU — verify neutral (Phase 3); TPW — DVFS actuator
available (Phase 3); TPS — within ~5 % of vLLM-alone is the gate. Density —
N/A. If vLLM cannot run coherently under interception, nothing else matters.
**Verify Phase 2.** Risk: vLLM CUDA graphs + injection (§4 R1).

### 2b — CIPHER weight sharing across vLLM instances — NOT TRANSPARENT
N vLLM instances, same model, one physical weight copy (Track 2 arena). vLLM
loads weights via its loader → PyTorch tensors → `cudaMalloc`; CIPHER's arena
is VMM. For weights to live in the shared arena, vLLM must allocate them there
— requiring either a **PyTorch pluggable allocator** backed by
`cipher_kv_bridge`, scoped to weight-load, or a **vLLM model-loader hook**
rebinding weight tensors onto arena views (the SC3 `from_blob` pattern).
**Real integration work, not transparency** — the largest build item, and the
**density** lever. **Metric impact:** Density — the headline cell, DEFERRED →
Phase 4. MFU/TPW/TPS — neutral / not levers. **Build + verify Phase 4.**

### 2c — CIPHER SM arbitration across vLLM instances — composes, ordering-constrained
N vLLM instances, each `CIPHER_QOS_CLASS=partition`, each gets a green-context
SM slice; Track 3 DSM rebalances under churn. Composes **if** the green ctx is
established at `cuInit` **before** vLLM creates its context / captures graphs.
**Metric impact:** Density — enables dense N-partition packing; per-instance
MFU/TPS bounding under contention — DEFERRED → Phase 5. The **primary
multi-tenant composition.** **Verify Phase 5.**

### 2d — CIPHER POOL beneath vLLM — DROPPED
vLLM already does continuous batching intra-process. Stacking CIPHER's POOL
decode-step fusion on top is (i) redundant, (ii) bypassed by vLLM's CUDA
graphs, (iii) contradicted by Phase C — CIPHER-fusion-over-HF vs vLLM is
~1.75–1.84×, a *power* win, because vLLM's own batching already captures the
throughput. **The CIPHER-native 3–6× / 498–990 tok/s numbers are context (A)
and do not transfer to the vLLM-composed architecture.** CIPHER's multi-tenant
role with vLLM is §2c (N SM-isolated, weight-shared vLLM instances) — not
batching over vLLM. The POOL executor remains a CIPHER-native-path feature
only.

## §3 — PRODUCTION DEPLOYMENT ARCHITECTURE

```
   tenant 1            tenant 2                 tenant N
 ┌───────────┐      ┌───────────┐            ┌───────────┐
 │ vLLM 0.21 │      │ vLLM 0.21 │    ...      │ vLLM 0.21 │   ← each: own process,
 │ engine    │      │ engine    │            │ engine    │     paged-attn, continuous
 │ (TP=1)    │      │ (TP=1)    │            │ (TP=1)    │     batching, CUDA graphs
 └─────┬─────┘      └─────┬─────┘            └─────┬─────┘
       │ CUDA driver API (cuInit → InitializeInjection2)
       ▼                  ▼                        ▼
 ┌─────────────────────────────────────────────────────────┐
 │  libcipher_rt  (CUDA_INJECTION64_PATH, per vLLM process) │
 │  green-context SM partition  ·  weight-arena import      │
 └─────────────────────────────────────────────────────────┘
       │                  │                        │
 ┌─────────────────────────────────────────────────────────┐
 │  kmod cipher (008b3c66)                                  │
 │  CP 5.4 SM-group ledger · Track 2 weight-arena registry  │
 │  Track 3 DSM · /proc/cipher/{stats,arenas,migrations}    │
 └─────────────────────────────────────────────────────────┘
       │                                          ▲
       ▼                                          │ NVML clock-lock (DVFS)
 ┌─────────────────────────────────────────────────────────┐
 │  H100 — 132 SM (15×8-SM groups arbitrated), 80 GB HBM    │
 │  one shared weight copy in the VMM arena                 │
 └─────────────────────────────────────────────────────────┘
```

- **Startup sequence (load-bearing):** kmod loaded first; each vLLM instance
  launches with `CUDA_INJECTION64_PATH=libcipher_rt.so` + `CIPHER_QOS_CLASS=
  partition` + `CIPHER_SM_COUNT=<n>` — libcipher_rt allocates the green ctx at
  `cuInit`, **before** vLLM's engine initializes CUDA, so the partition is in
  place when vLLM captures graphs.
- **Tenant registration:** each vLLM process is a kmod ledger entry (pid,
  `grp_mask`); the do_exit reaper reclaims on crash.
- **Operator visibility:** `/proc/cipher/{stats,arenas,migrations}`.
- **vLLM worker-process note:** vLLM 0.21 (V1 engine) runs an EngineCore
  process; the injection env must reach the CUDA-owning process. Phase 2
  confirms which process that is (Phase C saw EngineCore zombies).
- **Failure modes:** vLLM crash → kmod do_exit reaper reclaims its SM groups +
  drops its arena participant slot; CIPHER kmod cannot be rmmod'd while vLLM
  holds green contexts (drain-first — documented).

## §4 — KNOWN INTEGRATION RISKS

| # | risk | assessment |
|---|---|---|
| R1 | vLLM CUDA graphs vs interception | graph replay bypasses per-call hooks; but CIPHER's composed primitives (SM partition, DVFS) are context/actuator-level — survive replay if set up pre-capture. Phase 2 tests graph **and** eager mode (Phase C forced `enforce_eager` at 8-concurrent). |
| R2 | weight allocation path | vLLM weights → torch caching allocator → `cudaMalloc`, not VMM → transparent weight sharing impossible without an allocator hook / load rebind (§2b). Largest build item. |
| R3 | vLLM worker / EngineCore process model | injection env must reach the CUDA-owning process; Phase 2 maps it. |
| R4 | NCCL | only at TP>1 / multi-GPU — out of v1 scope (TP=1 per instance). No risk in v1. |
| R5 | observability cost on vLLM's hot path | the `nvidia_unlocked_ioctl` kprobe fires per NVIDIA ioctl; Phase 3 measures it stays sub-% — and this is the **MFU-neutrality** check. |
| R6 | SM partition × vLLM tensor-parallel | TP and SM partitioning conflict on one GPU → v1 runs **TP=1 per instance**; partitions are the cross-instance axis. |
| R7 | speculative decoding | vLLM has its own; libcipher_v2 has its own — **do not stack.** Under vLLM, vLLM's spec decode runs; CIPHER's is a native-path-only feature (§8 Q4). |

## §5 — MEASUREMENT PLAN (Phases 2+, GPU work)

Each phase STOPs for adjudication. **Every arm is measured against vLLM-alone**
(naive HF is not a serious baseline — the 2026-05-19 strategic correction).
The plan **measures each of the four metrics honestly** — it does not aim to
hit a predetermined number.

- **Phase 2 — transparency.** vLLM single instance under injection — runs
  coherently, graph + eager? Output-correctness gate vs vLLM-alone.
- **Phase 3 — parity (MFU + TPW + TPS).** vLLM-alone vs vLLM-on-CIPHER, single
  tenant: tok/s within ~5 %; MFU **neutral** (CP 3.3 telemetry); telemetry
  overhead sub-%; DVFS actuator measured.
- **Phase 4 — weight sharing (Density).** Build §2b; N vLLM instances, one
  shared weight copy; bit-identical output + the **measured** memory saving.
- **Phase 5 — SM arbitration (per-instance bounding).** N SM-partitioned vLLM
  instances; ordering (R1), disjointness, DSM rebalance; per-instance MFU/TPS
  under contention.
- **Phase 6 — composed architecture.** End-to-end N SM-isolated, weight-shared
  vLLM instances under one DVFS envelope — what CP 5.5 measures.

## §6 — CP 5.5 READINESS CRITERIA

CP 5.5 (100 Nemotron Nano agents on one H100) may begin when the composed
architecture is **verified to run and to be measurable on all four metrics** —
not when it hits a predetermined claim:
1. vLLM transparency verified (Phase 2-3).
2. vLLM-on-CIPHER within ~5 % of vLLM-alone single-tenant; MFU neutral (Phase 3).
3. weight sharing verified across vLLM instances, saving **measured** (Phase 4).
4. SM arbitration verified across N instances (Phase 5).
5. composed architecture produces measurable values on Density / TPW / TPS /
   MFU vs vLLM-alone multi-tenant (Phase 6).

**CP 5.5 establishes the actual 100-agent numbers. It is not required to match
any predetermined claim** — the benchmark *is* the measurement.

## §7 — v1 BOUNDARIES

- Single-GPU H100; multi-GPU is v2.
- vLLM 0.21.0 the only engine verified; TGI / SGLang / TensorRT-LLM are v2.
- Same-model weight sharing only; heterogeneous-model arenas are v2.
- TP=1 per vLLM instance (R6); SM partitioning is the cross-instance axis.
- CIPHER's composed value is **consolidation + actuators**, not batching
  (§2d) — an architectural boundary, not a temporary limitation.

## §8 — OPEN QUESTIONS — decisions

| # | question | decision |
|---|---|---|
| Q1 | vLLM version | **Pinned 0.21.0** — installed, isolated venv. Frozen for FUTURE_SCOPE/A. |
| Q2 | model target | **Nemotron Nano is NOT on the pod** — flagged. Phases 2-5 use TinyLlama-1.1B + Mistral-7B; Nemotron Nano must be obtained before CP 5.5. |
| Q3 | continuous batching × CIPHER POOL | **Resolved: do not compose** — vLLM owns batching (§2d). |
| Q4 | speculative decoding | **Resolved: vLLM's own** under vLLM; libcipher_v2 spec decode is native-path-only (R7). |
| Q5 | vLLM tensor-parallel vs SM partitioning | **Resolved: TP=1 per instance** (R6). |
| Q6 | weight-sharing mechanism (§2b) | **Flagged → Phase 4** — pluggable-allocator vs load-rebind needs a vLLM-internals spike; not resolvable on paper. |

## ADJUDICATION ASK

**STOPPING — paperwork only, no source, no GPU, no anchor rotation.**
Decisions:
1. Accept the **four-metric scorecard (§0)** with honest per-cell numbers —
   density the headline; TPW ~1.8× vs vLLM (power win); TPS a density story;
   MFU measured-and-neutral, no lift claim.
2. Accept the **two-context distinction** — CIPHER-native (3–6×) vs
   CIPHER+vLLM-composed (~1.8× + density); the native numbers do not transfer.
3. Accept the **architecture** — N SM-isolated, weight-shared vLLM instances;
   §2d (POOL-under-vLLM) dropped; weight sharing is a real Phase 4 build.
4. Accept the **Phase 2-6 plan** and CP 5.5 readiness criteria ("measure each
   honestly").
5. Accept the **§8 decisions**, including Q2 (Nemotron Nano needed before CP 5.5).

On adjudication → Phase 2 (vLLM single-instance transparency verification —
GPU work).
