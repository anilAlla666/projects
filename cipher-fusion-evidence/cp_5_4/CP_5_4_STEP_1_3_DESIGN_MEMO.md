# CP 5.4 — Step 1.3 (libcipher_rt client integration) — DESIGN MEMO

**Date:** 2026-05-18. **Status: orientation done — 4 architectural questions
for adjudication BEFORE any code.** No source modified. Anchor `a7ac8e97`
preserved (`cipher_rt_phase4/libcipher_rt.so.pre_cp5_4_step3` +
`cipher_rt_fallback/libcipher_rt.so.pre_cp5_4_step3`, both verified `a7ac8e97`).

---

## Substrate as read (not as assumed)

**Kmod CP 5.4 ABI** (`cipher_kmod/cipher_ioctl.h`, `cipher_cp54_sched.c`,
verified built + Steps 1.1/1.2):
- `CIPHER_CP54_ALLOCATE` `_IOWR('C',13, struct cipher_cp54_allocate)` —
  in `{qos_class, sm_count}`, out `{grp_mask_out u32 16-bit, grp_count_out}`.
- `CIPHER_CP54_FREE` `_IO('C',14)` — no payload, releases caller pid.
- `CIPHER_CP54_QUERY` `_IOR('C',15, struct cipher_cp54_query)` — out
  `{n_partitions, pool_grp_count, free_grp_count, my_grp_mask, my_qos_class}`.
- qos: PARTITION=0, SHARED=1, POOL=2. Keyed on `current->pid` (kernel LWP).

**Client green-ctx** (`cipher_rt_green_ctx.c`): one green ctx per process,
created lazily at the first `cuStreamCreate` CUPTI callback. Today picks ONE
8-SM group via `mix(getpid() ^ tenant_handle) % 16` — a hash-pick, no kmod
involvement. `cuDevSmResourceSplitByCount(MIN_SM=8)` → 16 groups; desc over
`groups[my_group_id]` (a single group); `cuGreenCtxCreate`; verifies SM count.

**Marlin already consumes green-ctx size.** `cipher_rt_marlin_engine.cpp`
reads `cipher_rt_green_ctx_sm_count()` to size its grid (CP 5.3 STEP 2A). When
CP 5.4 makes the green ctx variable-size, Marlin's grid tracks it for free —
but it also means a wrong CP 5.4 SM count silently mis-sizes Marlin. Composition
noted; not a blocker.

**ARB layer** (`cipher_rt_arbitrate.c`): calls nr-9 (now `-ENOSYS`). Its output
(`cipher_rt_arb_last_mask/_count`) has **zero consumers** outside its own TU;
`cipher_rt_arb_request_partition()` is called once in `cipher_rt_partition_-
router.c:220` with the return `(void)`-discarded. Retirement = delete the
`cipher_rt_arb_init()` call (`cipher_inject.c:37`), delete the
`cipher_rt_arb_request_partition()` call + comment (`partition_router.c:220`),
drop `cipher_rt_arbitrate.o` from `OBJS`. ~10 LOC removed, nothing reads the
output → **clean retirement, no deeper dependency** (Discipline-H gate: clear).

---

## Q1 — BLOCKER: which library does the Phase B executor load?

**ADJUDICATED 2026-05-18 — Option (c): descope item 3 to a new Step 1.3b.**
Rationale (user): Option (a) would retract the 3.69× Phase B baseline pending
re-measurement on libcipher_rt — breaks methodology continuity for no
architectural benefit. Option (c) keeps Step 1.3 bounded to libcipher_rt client
changes alone (Phase A SHARED + Phase B PARTITION single-tenant). Option (b)'s
"uncertain size" is unscoped work — not baked into Step 1.3.

**Revised CP 5.4 sub-step plan:**
| step | scope | status |
|---|---|---|
| 1.1 | kmod ledger build | DONE |
| 1.2 | kmod reload + isolation | DONE |
| 1.3 | libcipher_rt client — Phase A SHARED + Phase B PARTITION single-tenant | **CURRENT** |
| 1.3b | POOL / executor green-ctx binding | NEW — separate adjudication |
| 1.4 | confined-pool lift curve | pending 1.3b (1.4 gates on POOL) |
| 1.5–1.8 | churn / mixed-deployment / failure-modes / closure | as planned |

**Finding.** There are **two** injection libraries:
- `libcipher_v2.so` (anchor `86618c30`, 17 KB) — **minimal**: tenant register
  + CUPTI launch count. *No green-ctx, no CP 5.4 client, no Marlin.*
- `libcipher_rt.so` (anchor `a7ac8e97`, 151 KB) — full substrate: green-ctx,
  Marlin, attn, ARB, cuBLAS GOT-patch.

`CUDA_INJECTION64_PATH` usage across recent evidence: **126× `libcipher_v2.so`,
9× `libcipher_rt.so`.** The Phase B batch executor (`cipher_batch_executor_-
gen.py`, run via `run_arm3.sh`) loads **`libcipher_v2.so`** — confirmed in
`arm3_postreload_rep1/executor.log`:
`CUDA_INJECTION64_PATH=/home/ubuntu/libcipher_v2/libcipher_v2.so`.

**The problem.** Step 1.3 item 3 — "Phase B executor registers as POOL, binds
its kernel-launch stream to the pool's green context" — is **not implementable
as-is**: the executor's process has no green-ctx code. The CP 5.4 POOL owner
must run green-ctx + CP 5.4-client code. Step 1.3 item 1 puts that code in
`libcipher_rt.so`. So the executor must either load `libcipher_rt.so`, or gain
a CP 5.4 client by another route.

**Options:**
- **(a)** Switch the Phase B executor to `CUDA_INJECTION64_PATH=libcipher_rt.so`.
  Cost: it then also loads Marlin + attn + cuBLAS-GOT — the full substrate.
  Phase B's 3.69× was measured on the *minimal* lib, so this changes the
  measurement substrate; the 1.4 lift curve would re-baseline on libcipher_rt.
- **(b)** Keep the executor on `libcipher_v2.so`; give it a thin, explicit
  CP 5.4 POOL client (small new code — open `/dev/cipher`, `ALLOCATE(POOL)`,
  build a green ctx, bind the launch stream) rather than the whole runtime.
- **(c)** Descope item 3 from Step 1.3. Do only Phase A (SHARED) + Phase B
  (PARTITION) single-tenant green-ctx-via-kmod on `libcipher_rt.so`; the POOL /
  executor wiring becomes Step 1.3b or folds into 1.4 where the pool curve is
  measured anyway.

This decision determines whether Step 1.3 touches the executor at all, and
governs the Discipline stop-condition "executor launch-stream wiring is more
than a few-line change."

## Q2 — qos_class does NOT need a libcipher_v2 ABI change (scope cut)

Item 2 says "libcipher_v2 tenant registry: additive qos_class field." But the
kmod CP 5.4 ledger is **fully independent of `REGISTER_TENANT` (nr 1)**:
`cipher_cp54_ioctl_allocate` takes `qos_class` directly as an ALLOCATE input
and stores it in its own `cipher_cp54_allocs` table — there is no path from
REGISTER_TENANT into the CP 5.4 ledger.

Therefore qos_class need not enter the libcipher_v2 REGISTER_TENANT ABI at all.
It is simply an **ALLOCATE-time input** that libcipher_rt's green-ctx reads
from the environment (proposed: `CIPHER_QOS_CLASS` ∈ {partition,shared,pool}
default `shared`, `CIPHER_SM_COUNT` for PARTITION). **Consequence:
`libcipher_v2.so` (`86618c30`) does not change — no rebuild, no anchor
rotation.** Proposed: drop item 2's libcipher_v2 modification entirely; the
"declaration" is an env var consumed by libcipher_rt. Confirm.

## Q3 — split-order determinism was never verified (§2's load-bearing claim)

Scope memo §2 makes "kmod group g ↔ the same physical SMs in every process"
the bridge's load-bearing assumption and states "**Step 1.2 verifies** the
split is stable across processes." It did not: the Step 1.2 isolation test
(`cp54_isolation_test.c`) is pure-C `/dev/cipher` ioctl exercise — **0 CUDA
calls**, no `cuDevSmResourceSplitByCount`. Cross-process split-order
determinism is **unverified**.

If the split order is not stable across processes, kmod group g maps to
different SMs in process A vs B and green-context disjointness — the entire
design foundation — breaks. Proposed: Step 1.3 Phase A opens with a 2-process
CUDA probe that confirms `cuDevSmResourceSplitByCount` returns group g = the
same SM range in every process; this gates everything after it. Confirm this
belongs in 1.3 (it is the prerequisite the grp_mask bridge rests on).

## Q4 — ALLOCATE thread identity vs the per-LWP reaper

The kmod keys CP 5.4 on `current->pid` (LWP) and the do_exit reaper reclaims
per-LWP (verified Step 1.2). `green_ctx_ensure()` — which will issue ALLOCATE —
runs on whichever thread first trips the CUPTI `cuStreamCreate` callback. If
that is a transient thread that exits before the process, the reaper frees the
groups **while the green context is still live** (the green ctx is created once
and never destroyed). In practice it is the long-lived main thread (PyTorch
warmup), but that is not guaranteed.

Proposed v1: ALLOCATE on the injection-init thread (the `pthread_once`
`cipher_v2_init_body` thread, which is also where the legacy ARB fd was opened)
— a known process-lifetime thread — rather than the first-stream thread; or, if
simpler, accept the first-stream-thread assumption and **document it**. Low-risk
either way; flagged so the choice is explicit, not implicit.

---

## Proposed Step 1.3 plan (Q1 adjudicated; pending Q2–Q4)

1. Split-order determinism probe (Q3) — gates the rest.
2. `cipher_rt_green_ctx.c`: replace hash-pick with `ALLOCATE` → `grp_mask` →
   select set-bit group resources → `cuDevResourceGenerateDesc` over their
   union → `cuGreenCtxCreate`. Estimate **~40–60 LOC net** in green_ctx.c —
   **near the Discipline ">50 LOC → STOP" threshold**; will report exact count
   before rebuild.
3. qos_class via env (Q2) — no libcipher_v2 change.
4. ARB-poll retirement — ~10 LOC removed, clean (shown above).
5. ~~Executor / POOL wiring~~ — **descoped to Step 1.3b per Q1 adjudication.**

Anchors that rotate on full Step 1.3 PASS: `libcipher_rt` `a7ac8e97` → new.
`libcipher_v2` `86618c30` unchanged (per Q2). kmod `7f467de4` unchanged.

**Q1 ADJUDICATED. STOPPING for Q2–Q4 adjudication per design-memo-first
discipline — no source modified.**
