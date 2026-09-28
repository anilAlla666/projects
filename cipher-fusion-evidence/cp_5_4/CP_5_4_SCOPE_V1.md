# CP 5.4 — Per-Tenant Arbitration — SCOPE MEMO V1 (corrected)

**Date:** 2026-05-18. **Type:** design/scope, no GPU, no code. **Status:**
scoped for adjudication — no Step 1.1 build until adjudicated. Anchor
`a7ac8e97` unchanged by this memo. **Supersedes** `CP_5_4_SCOPE.md`
(`.superseded`, retained for audit).

---

## §0 — Why V1

The original memo's §8 under-sized the kmod work ("small extension") and
floated the scheduler as possibly a per-process libcipher_rt module.
Orientation on `cipher_rt_green_ctx.c` + the kmod `cipher_partition_allocator.c`
found two facts the original missed:

1. **The SM-allocation ledger is intrinsically cross-process.** libcipher_rt is
   LD_PRELOAD'd per tenant process — each process has its own static state, so
   it cannot maintain `Σ SMs ≤ 132` across tenants. The kmod is the only
   cross-process authority; `cipher_partition_allocator.c` already *is* that
   authority (lock-free per-PID slot array). The ledger **must be kmod-resident.**
2. **The slot→SM bridge does not exist.** `cipher_rt_green_ctx.c` hash-picks one
   fixed 8-SM group per process (`pid ^ tenant_handle`), with no kmod
   coordination; the kmod allocates *abstract* 4-SM slots with no SM identity.
   Its own comment defers the mapping ("T4.2.4c will add slot→SM-index
   mapping") — never built. That bridge is CP 5.4's load-bearing work.

Adjudicated direction: **Option A** — kmod-resident ledger + slot→SM bridge.
V1 scopes it. v1 functional scope is unchanged from the original (§1–2 there):
static per-session declared `qos_class`, single batch pool, dynamic/per-request
arbitration deferred to v2.

## §1 — Kmod ledger structure

CP 5.4 adds a **group-allocation ledger** to the kmod. It does **not** run
concurrently with the legacy 33×4-SM-slot allocator — the legacy allocation
path is deactivated (see "Legacy-allocator interlock" below); the legacy
*code* stays in place, the ABI nr stays defined.

**Why a new ledger, not an extension of the slot array:** CUDA green contexts
on H100 partition in **multiples of 8 SMs** (`cipher_rt_green_ctx.c`:
`cuDevSmResourceSplitByCount` with `MIN_SM=8` → 16 groups of 8 SMs, 128 SMs;
4 remainder). The legacy allocator's 4-SM slots cannot drive green-context
placement. CP 5.4 standardizes on the **8-SM group** as the allocation quantum
— the same 16-group split the green-ctx layer already performs.

```
CP 5.4 ledger:  16 groups × 8 SMs  (group g  ↔  the g-th 8-SM resource
                                     from cuDevSmResourceSplitByCount)
  state[g] : atomic_t   = 0 (free)  |  CIPHER_GRP_PACK(pid)
  per-allocation metadata (in cipher_pid_stats, extended):
    qos_class : u8   PARTITION | SHARED
    is_pool   : u8   1 = this PID is the batch-pool owner
    grp_mask  : u16  the groups this PID owns  (cache; state[] is truth)
```

- **Variable SM count** — an allocation is any subset of the 16 groups; a
  partition tenant gets `ceil(K/8)` groups, the pool gets the rest.
- **Batch pool** — one PID flagged `is_pool=1`. The pool **explicitly claims**
  every otherwise-free group (so the ledger is fully accounted and the
  `Σ groups = 16` invariant is checkable; the pool is *not* an implicit
  complement). It resizes by releasing/claiming groups as partitions come/go.
- **Invariant:** every group has exactly one owner or is free; a partition
  tenant + the pool never share a group → green contexts are disjoint by
  construction (CP 5.3 STEP 2A gate C already proved disjoint green contexts).
- **Concurrency:** per-group `atomic_cmpxchg` — the legacy allocator's lock-free
  model, extended (see §5).

The 4-SM remainder (132 − 128) is unallocatable under 8-SM granularity —
inherited from the existing green-ctx design's accepted 3% loss.

**Legacy-allocator interlock (correctness — not just ABI).** The legacy nr-9
allocator and the CP 5.4 group ledger address the **same 132 physical SMs**
(legacy 33×4 = 132; CP 5.4 16×8 = 128 of them). Lock-free atomicity holds only
*within* each allocator — there is no cross-allocator interlock, so two live
allocators could grant the same physical SMs to two tenants and break
green-context disjointness, the foundation of the whole design. Therefore the
CP 5.4 kmod **deactivates the legacy nr-9 allocation path**: nr 9 returns
`-ENOSYS` — the ABI nr stays *defined* (exactly the [[cipher-abi-rule]]
"reserved nr returns `-ENOSYS`" contract), but grants nothing. The §6 drain
makes this free: after the reload no CP-5.3-era tenant is alive to call nr 9,
and every new tenant uses the CP 5.4 path. **One allocator, one source of
truth** — "additive" here means new code added, not a second allocator running.

## §2 — Slot→SM bridge (the deferred T4.2.4c work)

The bridge makes a kmod allocation a **deterministic SM placement**, not an
abstract mask.

- **Representation across the ioctl boundary:** a `u16 grp_mask` — bit g set ⇒
  the PID owns 8-SM group g.
- **libcipher_rt side:** receives `grp_mask`; calls `cuDevSmResourceSplitByCount`
  (→ the 16 group resources, in a **stable order** for a fixed device — the
  determinism this bridge depends on; verified in Step 1.2); selects the
  resources for the set bits; `cuDevResourceGenerateDesc` over their union;
  `cuGreenCtxCreate`. A 24-SM partition = 3 groups; the batch pool with two
  8-SM partitions out = 14 groups = 112 SMs.
- **Determinism:** group g ↔ the g-th split resource ↔ a fixed SM range. No
  hashing. The kmod *decides* `grp_mask`; libcipher_rt *enacts* it. This
  replaces the current `pid ^ tenant_handle` hash-pick entirely.
- **Step 1.2 verifies** the split is stable across processes (group g = the same
  physical SMs in every process) — the bridge's load-bearing assumption.

## §3 — Ioctl surface (additive to `cipher_partition_allocator`)

Fresh additive nrs (next available, ≥ 10 — exact numbers assigned at build;
nr 9 and the legacy slot ioctls untouched). The original memo listed five
operations; they streamline to **three ioctls** — `allocate` subsumes
batch-executor registration (`qos_class=POOL`) and pool resize (a side effect
of any allocate/free):

| ioctl | in | out | notes |
|---|---|---|---|
| `CIPHER_CP54_ALLOCATE` | `qos_class` (PARTITION/SHARED/POOL), `sm_count` | `grp_mask`, `grp_count` | `POOL` ⇒ register caller as pool owner + grant residual — **singleton: a second `ALLOCATE(POOL)` while a pool is registered returns `-EEXIST`**; `PARTITION` ⇒ grant `ceil(sm_count/8)` groups, shrinking the pool first if needed; idempotent per PID |
| `CIPHER_CP54_FREE` | (caller pid) | `grp_mask` returned to pool | groups go free; pool reclaims on its next allocate/tick |
| `CIPHER_CP54_QUERY` | — | `{n_partitions, pool_grp_count, free_grp_count, my_grp_mask}` | read-only; for the scheduler client + diagnostics |

Pool **resize is internal** — not a separate ioctl: a `PARTITION` allocate that
needs groups triggers a pool shrink; a `FREE` lets the pool grow on its next
allocate or the 5 s `cipher_partition_tick`. (If adjudication prefers the
original 5-ioctl surface for explicitness, that is a cheap change — flagged.)

## §4 — libcipher_rt client side

`cipher_rt_green_ctx.c` reworked from "hash-pick one fixed 8-SM group" to
"kmod-driven variable-size green context":

- At green-ctx creation: `ioctl(CIPHER_CP54_QUERY)` (or the result cached from
  `ALLOCATE` at registration) → `grp_mask`.
- `cuDevSmResourceSplitByCount` → select the `grp_mask` resources →
  `cuDevResourceGenerateDesc` → `cuGreenCtxCreate`. Variable size.
- The `tenant_handle ^ pid` hash and the fixed `CIPHER_RT_GREEN_NUM_GROUPS`
  single-group path are removed.
- **`qos_class` declaration** — additive field on the libcipher_v2 tenant
  registration path; forwarded to the kmod so `ALLOCATE` routes
  partition-vs-pool. Default (undeclared) = `SHARED`.
- **Batch executor** — calls `ALLOCATE(qos_class=POOL)`, gets the residual
  groups, creates its green context over them, and **binds its kernel-launch
  stream to that green context** (the original §8 unverified-composition item —
  an explicit Step-1.3 task, not free).

## §5 — Race conditions and concurrency

The legacy allocator is "lock-free atomic, no global lock." CP 5.4 extends it —
with one honest caveat.

- **Per-group claim/release** — `atomic_cmpxchg(state[g], 0, PACK(pid))` /
  `cmpxchg(PACK(pid), 0)`. Lock-free, ABA-safe via the pid-packed word, exactly
  the legacy model. Extends cleanly.
- **The caveat — pool shrink + partition claim is not one atomic unit.** A
  `PARTITION` allocate that needs groups does: (1) shrink the pool (the pool PID
  releases its highest groups via cmpxchg), then (2) claim the freed groups.
  Steps (1) and (2) are each lock-free sequences but **not a single
  transaction** — a concurrent third tenant could claim a group freed in (1)
  before this caller's (2). Worst case: the caller under-fills and retries, or
  gets `-ENOSPC` — the legacy allocator is already request-driven and
  retry-friendly, so this degrades safely, not incorrectly.
**ADJUDICATED 2026-05-18 — Option (ii): a kmod mutex.** A lock-free+retry
model (option i) has an unbounded worst-case retry latency under sustained
churn; (ii) bounds it and does not rest correctness on "in practice the
ping-pong is unlikely."

**Implementation clarification (Step 1.1 build, adjudication-accepted).** The
mutex is taken for the **whole `ALLOCATE` operation**, not only the
pool-resize sub-step. `ALLOCATE` is a per-session cold path; splitting it into
"lock-free metadata/group claim + mutex'd pool-resize" reintroduces exactly
the race subtlety the mutex was chosen to avoid (the per-PID metadata-table
slot claim also needs serialisation). A whole-operation mutex on a cold path
is correctness-clean at no realistic perf cost. The **steady-state /
atomic-context paths stay lock-free**: `FREE`, the `do_exit`-reaper
`release()` (which *must* be lock-free — kprobe atomic context cannot sleep
on a mutex), and `QUERY` use only `atomic_cmpxchg` / atomic reads. The legacy
nr-9 path is untouched (and deactivated). This is the model built and
verified in `CP_5_4_STEP_1_1_BUILD_LOG.md`.

## §6 — Kmod reload strategy

A loaded `cipher_kmod` cannot be `rmmod`'d while any process holds `/dev/cipher`
open, and side-by-side (two kmods, same device major/name) is not possible. So:

- **Drain** — stop all CIPHER tenant processes, `rmmod cipher_kmod`, `insmod`
  the CP 5.4 kmod, tenants re-register. This is a **dev pod** with no production
  tenants; a drain is acceptable and fully under our control. (A
  hot-swap/live-patch path is out of scope for v1.0.)
- **Rollback** — the current `cipher_kmod.ko` (`e2f50452`) is preserved
  **outside the kbuild dir** (per [[cipher-kbuild-clean-wipes-ko]] — `kbuild
  clean` globs `*.ko`) as the fallback. If the CP 5.4 `insmod` fails or the
  isolation test (Step 1.2) fails, `rmmod` + `insmod` the preserved `e2f50452`
  and the pod is back to the CP 5.3-close state.
- Tenant re-registration after reload is the normal `tenant_register` path — no
  special handling; tenants started after the reload pick up the new ABI.

**Adjudication item** — confirm drain (recommended; dev pod) vs any requirement
to preserve a running workload across the swap. This is one of the three.

## §7 — Failure recovery — v1.0 robustness target

| failure | handling | v1.0 target |
|---|---|---|
| kmod `insmod` fails | rollback to `e2f50452` (§6) | full — no degraded state |
| Step 1.2 isolation test fails on new kmod | rollback; do not proceed to 1.3 | full |
| process crash mid-allocation | a group is atomically owned or not — no partial-group state; the kmod `do_exit` reaper (`cipher_partition_release_slots_only`, extended to the CP 5.4 group ledger) frees the crashed PID's groups | full — reuse the verified reaper path |
| batch-executor crash | the pool allocation frees; its groups go free; the next `ALLOCATE(POOL)` reclaims them. Batch-mode tenants see the pool disappear → fall back per §6 of the original memo (Phase B Step 4 eviction) | full |
| partition tenant crash | reaper frees its groups → pool reclaims on next tick | full |
| pool shrink under-fill race (§5) | caller retries / `-ENOSPC` | best-effort + retry — *acceptable for v1.0* |

**v1.0 robustness target:** full recovery for all crash/load-failure paths via
reuse of the verified `do_exit` reaper + the preserved-`.ko` rollback; the only
best-effort path is the §5 pool-resize race, which degrades to retry, never to
incorrect allocation. **Adjudication item** — confirm this target (full crash
recovery, best-effort resize race) is the right v1.0 line, or raise it.

## §8 — Step 1 sub-step plan

| step | scope | est. |
|---|---|---|
| 1.1 | Kmod ledger + slot→SM bridge + 3 ioctls (no libcipher_rt changes) | 1–2 d |
| 1.2 | Kmod test in isolation — `ALLOCATE`/`FREE`/resize cycles; verify split-order determinism; concurrent-requester stress | 0.5 d |
| 1.3 | libcipher_rt client — kmod-driven variable green ctx; `qos_class`; batch-executor pool registration + launch-stream binding | 1–2 d |
| 1.4 | **Confined-pool batch-lift curve** — the load-bearing measurement (§ below) | 0.5–1 d |
| 1.5 | Green-context churn cost — teardown/create microbench; is Marlin cubin recompiled per green-ctx | 0.5 d |
| 1.6 | Mixed-deployment verification — 2 partition + N pool tenants, per-tenant per-mode correctness | 0.5–1 d |
| 1.7 | Failure-mode coverage — crash paths, reload rollback, resize race | 0.5–1 d |

**Step 1.4** is the headline measurement. Measure delivered
substrate-attributable lift at pool sizes **128 / 112 / 96 / 64 SMs** (16 / 14 /
12 / 8 groups), 3 reps each, teacher-forced KL ≤ 0.1 per tenant. The curve
prices the opportunity cost of every group moved into a partition — exactly what
the arbitration policy needs.

**Baseline discipline:** the curve's top point is the **128-SM (16-group) pool**,
and the curve is reported as **delta from that 128-SM point** — *not* from
Phase B's 3.69×. Phase B's 3.69× was on the full 132 SMs; the 128-SM point is
already ~3% under it purely from the 8-SM-group granularity's 4-SM remainder,
*before* any confinement effect. Anchoring the curve to 3.69× would make the
top point look like an unexplained ~3.6× and send us chasing a non-finding. The
132-SM Phase B number is the separate no-arbitration reference, not the curve's
anchor.

**Stop-and-adjudicate** if the curve collapses sub-linearly past the 64-SM
point — that would mean `qos_class=isolated` costs more than v1 scoping assumes.

## §9 — Anchor rotation plan

| anchor | now | after Step 1 |
|---|---|---|
| kmod `cipher_kmod.ko` | `e2f50452` | rotates → preserve `e2f50452` **outside the kbuild dir** as the `.pre_cp5_4` fallback ([[cipher-kbuild-clean-wipes-ko]]) |
| libcipher_rt | `a7ac8e97` | rotates (Step 1.3) → preserve `libcipher_rt.so.pre_cp5_4` |
| libcipher_v2 | `86618c30` | gets an **additive** `qos_class` field — low-risk; rotates only if rebuilt. Preserve `86618c30` if so. |
| cipher_kv_bridge | `fca6843d` | unchanged (not touched by CP 5.4) |

`a7ac8e97` unchanged until Step 1.3; `e2f50452` unchanged until Step 1.1's
reload. All current anchors preserved per [[cipher-phase-discipline]].

## §10 — Time estimate (revised)

| phase | est. |
|---|---|
| this corrected scope memo | ~3–4 h (paperwork) |
| Step 1.1 kmod build | ~1–2 d |
| Step 1.2 kmod isolation test | ~0.5 d |
| Step 1.3 libcipher_rt integration | ~1–2 d |
| Step 1.4 batch-lift curve | ~0.5–1 d |
| Step 1.5 churn cost | ~0.5 d |
| Step 1.6 mixed-deployment | ~0.5–1 d |
| Step 1.7 failure modes | ~0.5–1 d |
| **total** | **~5–7 days** |

## §11 — Adjudication ask

Three decisions are required before Step 1.1 (the user named these):

1. **Concurrency model (§5)** — (i) lock-free + retry for the pool-resize race
   (consistent with the legacy allocator) vs (ii) a small kmod mutex around
   pool-resize only (simpler correctness, departs from "lock-free").
2. **Kmod reload strategy (§6)** — confirm drain (recommended, dev pod) vs a
   requirement to preserve a running workload across the swap.
3. **v1.0 failure-recovery target (§7)** — confirm full crash/load-failure
   recovery via the verified reaper + `.ko` rollback, with the §5 resize race
   as the one best-effort path.

Plus, lightweight: accept the §3 streamlined 3-ioctl surface (vs the original
5), and the §1 decision to add a new 8-SM-group ledger rather than extend the
legacy 4-SM-slot array.

No code until this memo is adjudicated.
