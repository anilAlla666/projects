# CP 5.4 — Step 1.3b' (POOL / executor binding) — PHASE A DESIGN MEMO

**Date:** 2026-05-19. **Status: orientation done — one non-trivial
architectural finding; STOP for adjudication before any code** (per the Step
1.3b' Phase A discipline). No source modified. Anchors unchanged.

---

## Orientation — the batch executor as it is

**Source:** `cipher-fusion-evidence/cp_5_6/phase_b/session1/cipher_batch_executor_gen.py`
— the executor `run_arm3.sh` launches. **Pure Python, 147 lines.** Form A: one
process holds a TinyLlama model; N clients submit decode requests over a Unix
socket; the executor fuses them into one `B=N` batched `generate()`.

Orientation answers to the three integration questions:

1. **Where to register as POOL.** The executor has a clean linear startup:
   `tenant_register.register("batch_exec")` (line 40) → model `.cuda()` (49) →
   socket accept (64–72) → warmup `generate()` (74–78) → 5 timed rounds
   (87–104) → teacher-forced gate (107–124). `CIPHER_CP54_ALLOCATE(POOL)` is a
   `/dev/cipher` ioctl — it can run any time after `tenant_register`. The
   green context needs CUDA initialized (`.cuda()`), so it must be built
   **after line 49 and before the warmup**, so warmup + all rounds run inside
   it.
2. **Streams.** The executor creates **no explicit CUDA streams** — every
   `generate()` / forward runs on PyTorch's default stream.
3. **Context.** No explicit context — PyTorch uses the device **primary
   context** (it `cuDevicePrimaryCtxRetain`s; it does not adopt an arbitrary
   pushed `CUcontext`). This is the same fact that forced libcipher_rt's
   per-launch `cuCtxSetCurrent` enforcement (T4.2.4c/d).

**Injection:** `tenant_register.py` sets `CUDA_INJECTION64_PATH=libcipher_v2.so`
(the minimal lib — tenant-register + launch-count only; no green-ctx code)
before `.cuda()`. The executor stays on `libcipher_v2` per the Q1/Option-B
decision.

**PyTorch green-context support is present.** torch is **2.11.0+cu130**;
`torch.cuda.green_contexts.SUPPORTED == True`; `torch.cuda.GreenContext` is a
usable beta API. Verified live: `GreenContext.create(16)` succeeds, `.Stream()`
returns a stream on it, `.set_context()/.pop_context()` work, a matmul runs
under it and is finite. So a thin Python client **can** create a green context
and run the executor's `generate()` under `gc.Stream()` — no libcipher_rt, no
CUPTI machinery needed. That is the good news for Option B.

## THE FINDING — `torch.cuda.GreenContext.create()` is count-only

```
torch.cuda.GreenContext.create(num_sms: int, device_id: int = 0)
```

The **only** creation API takes an SM **count**. There is no parameter for an
SM-resource set, a split-group index, or a `CUdevResource`. PyTorch internally
splits the device and picks the SMs itself. `GreenContext` also exposes **no
SM-introspection** method (`create`, `Stream`, `set_context`, `pop_context` —
that is all), so a caller cannot even read back which SMs it got.

**Why this matters.** CP 5.4's design rests on the kmod ledger being the single
source of truth: `ALLOCATE` returns a **`grp_mask`** — *which* 8-SM groups —
and the client must enact *exactly that placement*. The Step 1.3a probe proved
group g ↔ deterministic physical SMs precisely so this bridge is well-defined.
libcipher_rt's Step 1.3 green-ctx is grp_mask-precise (it selects the set-bit
groups). **`torch.cuda.GreenContext.create(num_sms)` cannot be grp_mask-precise
— it only knows a count.**

- **Test A (POOL alone)** — fine. Pool owns all 15 groups; `create(120)` gets a
  120-SM green context; "which 15 of 15" is moot.
- **Test B (POOL + PARTITION coexistence)** — **broken.** A PARTITION tenant
  (libcipher_rt, grp_mask-precise) takes, say, groups {0,1}; the kmod shrinks
  the pool ledger to groups {2..14}. The executor must run on **exactly
  {2..14}**. But `GreenContext.create(104)` gives PyTorch's *own* pick of 13
  groups — there is no reason that is {2..14}; if PyTorch picks the low prefix
  {0..12} (the natural split order), the executor's pool **overlaps the
  partition tenant on groups {0,1}**. The kmod ledger says "disjoint"; physical
  reality is not. CP 5.4's core invariant — *a partition tenant and the pool
  never share a group* (scope memo §1) — cannot be enforced through a
  count-only API.

Two further sub-issues compound it:
- **Pool resize → green-ctx recreation.** When a partition comes or goes the
  pool's SM count changes; a count-API green context would have to be
  *recreated* at the new size. Green contexts are "create once, never refresh"
  (a mid-run recreate strands existing streams). The executor is simple enough
  to re-stream between its 5 rounds, but this is a real design item, not free.
- **No disjointness check.** `GreenContext` exposes no way to read its SM set,
  so Test B's "verify disjoint from the executor's mask" cannot even be
  *measured* through the torch API — it would need an out-of-band `%smid`
  probe.

This is exactly the "launch path needs more than a thin client" risk the Step
1.3 spec pre-flagged and the Step 1.3b' Phase A stop-condition names.

## Options for adjudication

1. **Count-only pool, descope Test B's disjointness guarantee.** Executor uses
   `torch.cuda.GreenContext.create(num_sms)`. Step 1.3b' delivers Test A
   (POOL-alone, 120-SM green ctx), Test C (reaper — kmod-side, unaffected),
   Test D (regression — count-based confinement, fine). **Test B's
   disjoint POOL+PARTITION coexistence is deferred** — it needs grp_mask
   precision the torch API lacks. Note: Step 1.4's lift curve (executor on
   120/104/88/64 SMs) only needs *count*-based pool sizing, so the curve is
   **not** blocked by this. Cleanest, smallest, honest about the gap.
2. **Align the kmod allocation order to torch's split order.** Make the kmod
   allocate the POOL a low contiguous prefix [0..K-1] and PARTITIONs from the
   high end; if torch's `create(N)` deterministically picks the low N/8 groups,
   the two agree and disjointness holds. Cost: a kmod policy change (reopens
   the kmod — `cp54_claim`/`cp54_pool_shrink` scan directions); empirical
   verification of torch's pick order; **and** executor green-ctx recreation on
   every pool resize. More moving parts; couples CP 5.4 correctness to an
   undocumented PyTorch internal.
3. **grp_mask-precise green ctx for the executor.** Only libcipher_rt (or a new
   C extension replicating its driver-API green-ctx path) can build a green
   context over an arbitrary group set — and PyTorch will not *use* an
   externally-created green context (its `GreenContext` wraps only its own).
   This collapses back toward Option A (executor on libcipher_rt), which was
   rejected for Phase B methodology continuity. Not recommended.

## Recommendation

**Option 1.** It ships the POOL-bound executor (Test A/C/D) on the clean torch
native API with zero kmod churn and zero libcipher_rt churn, and it does **not**
block Step 1.4's lift curve. Test B (guaranteed-disjoint coexistence) is a real
capability gap, but closing it well needs either a kmod-policy change with
verified PyTorch-internal coupling (Option 2) or grp_mask-precise green
contexts (Option 3) — both are their own scoped work, not something to
improvise inside Step 1.3b'. Recommend Step 1.3b' = Option 1, and track
"disjoint POOL+PARTITION coexistence" as an explicit follow-on.

## Option 4 — wrap an externally-created `CUgreenCtx` — INVESTIGATED, REJECTED

Investigated per founder request (2026-05-19): can a thin client build a
grp_mask-precise `CUgreenCtx` via the CUDA driver API (exactly as
libcipher_rt's Step 1.3 path does — `cuDevSmResourceSplitByCount` → select
grp_mask groups → `cuDevResourceGenerateDesc` → `cuGreenCtxCreate`) and then
**wrap that handle in `torch.cuda.GreenContext`**, so the executor keeps
`libcipher_v2` yet still gets grp_mask-precise placement?

**Rejected — `torch.cuda.GreenContext` is count-only by C++ design.** The
authoritative evidence is the C++ header shipped with torch 2.11.0+cu130,
`torch/include/ATen/cuda/CUDAGreenContext.h`:

- The only public creator is `static create(uint32_t num_sms,
  std::optional<uint32_t> device_id)` — count-only.
- The only instance constructor, `GreenContext(uint32_t device_id,
  uint32_t num_sms)`, is **`private`** and also count-only. There is **no
  constructor — public or private — that takes a `CUgreenCtx`.**
- The `CUgreenCtx green_ctx_` member is **`private`**, set only by that
  count-based constructor.
- The pybind11 class `torch._C._CUDAGreenContext` exposes **no constructor to
  Python** (`_CUDAGreenContext() → TypeError: "No constructor defined!"`) and
  only four methods (`create`, `Stream`, `set_context`, `pop_context`) — no
  handle accessor or mutator.

There is no constructor, factory, or bound field through which an external
`CUgreenCtx` can enter torch's `GreenContext`. Reaching the private C++
`green_ctx_` from Python would require raw-memory ABI hackery on a beta class —
not a basis for a correctness-critical path. Probe steps 2–3 (build a handle
via ctypes, attempt the wrap) are moot: step 1 — the header — conclusively
shows there is nothing to wrap it with.

Nor does building a green context outside torch and pushing its `CUcontext`
current help: PyTorch binds to the device **primary context** and ignores a
pushed green-derived context (the T4.2.4c/d finding). Per-launch enforcement is
what libcipher_rt's CUPTI hook does — i.e. Option 3.

**Conclusion: Option 4 is not feasible.** Proceed to the Option 2 viability
probe (Appendix A).

---

## Appendix A — torch GreenContext SM-selection probe (2026-05-19)

**Question (Option 2's load-bearing assumption):** does
`torch.cuda.GreenContext.create(num_sms)` pick the **low-prefix** 8-SM groups
(groups 0..N-1 in the `cuDevSmResourceSplitByCount` ordering of
`PHASE_1_3A_PROBE.md`), deterministically?

**Method:** `cp54_torch_greenctx_probe.py` — for `num_sms` ∈ {8,16,24}, create a
torch green context, run a `%smid`-recording kernel (`load_inline`) on its
stream, read back the physical SM set used, compare to the kmod group sets;
4 runs each.

**Result — PASS, unambiguous:**

| `create()` | physical SMs used | = kmod groups | det. (4/4 runs) |
|---|---|---|---|
| `create(8)`  | {0,1,16,17,32,33,48,49} | group 0 | identical |
| `create(16)` | groups 0∪1 (16 SMs) | groups 0,1 | identical |
| `create(24)` | groups 0∪1∪2 (24 SMs) | groups 0,1,2 | identical |

`torch.cuda.GreenContext.create(8N)` uses **exactly the low N groups
(0..N-1)**, byte-identical across all 4 runs at every size, exact match to the
kmod / `cuDevSmResourceSplitByCount` ordering.

**Option 2 is viable.** And it needs **no kmod change for Step 1.3b' scope:**
the kmod's existing allocation policy already places the POOL low and
PARTITIONs high — `ALLOCATE(POOL)` claims all groups; a `PARTITION` shrinks the
pool from its **highest** index (`cp54_pool_shrink` scans g=N−1..0) and claims
those freed high groups. So with the executor registering POOL first (it does,
at startup): pool owns the low contiguous prefix [0..K−1], partitions take the
high end — and torch's `create(K·8)` picks exactly [0..K−1]. **Aligned →
disjoint.** Test B (POOL + one PARTITION) holds.

**Costs / documented dependencies of Option 2:**
1. **PyTorch-internal dependency.** Correctness rests on `create()`'s
   low-prefix pick — an undocumented behavior of a beta API ("may change in
   future releases"). Mitigation: the executor must **self-verify** its green
   context's SM set at startup (the `%smid` probe above, ~30 lines) and fail
   loud if torch's pick is ever not the expected low prefix — making the
   dependency self-checking, not silent.
2. **Executor green-ctx recreation on pool resize.** `create()` is count-based;
   when a partition arrives/leaves the pool's size changes, so the executor
   must detect the resize (kmod `QUERY` between rounds) and recreate its green
   context + stream at the new count. Feasible for this simple 5-round
   executor; real work, not free.
3. **Multi-partition free-order caveat.** If ≥2 partitions free out of LIFO
   order the kmod pool can become a non-contiguous, non-low-prefix set, which
   torch's low-prefix pick can no longer match. Out of Step 1.3b' scope (≤1
   partition); flag for Step 1.6 (mixed deployment) — needs kmod compaction or
   an accepted-fragmentation policy.

## Recommendation (post-probe)

**Option 2.** The probe settled its load-bearing question decisively, and for
Step 1.3b' scope it requires no kmod change — only executor-side work
(thin CP54 POOL client + torch GreenContext + self-verification + resize
recreation). It preserves the Test B disjointness invariant that Option 1
would have descoped. The PyTorch-internal dependency is real but is made safe
by the startup self-verification (cost 1). Recommend Step 1.3b' = Option 2.

**STOPPING HERE for adjudication — investigation only, no executor code
written, no source modified. Anchors unchanged: kmod `8d777dfb`, libcipher_rt
`ebc0baaa`, libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`.**
