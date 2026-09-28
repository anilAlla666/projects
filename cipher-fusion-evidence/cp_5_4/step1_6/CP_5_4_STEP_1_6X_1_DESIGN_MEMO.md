# CP 5.4 — Step 1.6X-1 (POOL-disjointness-under-out-of-order-free) — DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no GPU, no
source modified, no measurement. STOP for adjudication before 1.6X-2 build.
Anchors unchanged: kmod `8d777dfb`, libcipher_rt `ebc0baaa`, libcipher_v2
`86618c30`, cipher_kv_bridge `fca6843d`.

---

## §0 — Headline: the mechanism the adjudication named is not the one that is
##      feasible — and the feasible one is smaller

Step 1.6X was authorized to *"build kmod compaction … reassign groups to keep
POOL on contiguous low prefix"* — finding D1 in the deferral audit. Orientation
on the live kmod source (`cipher_cp54_sched.c`, anchor `8d777dfb`) found:

- **True relocation-compaction — *moving a live partition's groups* — is not
  feasible in v1.** A PARTITION tenant's libcipher_rt green context is bound to
  *specific physical SMs* (its `grp_mask` groups, enacted at green-ctx
  creation). The kmod rewriting a partition's ledger entry does **not** move
  where that partition's kernels physically run. To actually relocate a
  partition, the tenant must **destroy and rebuild its green context** — but
  libcipher_rt's green context is **create-once-never-refresh** (Step 1.3), a
  mid-run rebuild strands in-flight streams, and there is **no kmod→tenant
  notification path** in the substrate. Relocation-compaction would reopen
  libcipher_rt (anchor `ebc0baaa` rotation), add a notification mechanism, and
  disrupt exactly the strong-isolation tenants that must stay stable. It is
  heavier than the rest of CP 5.4 combined and architecturally wrong for v1.
  **(Detailed as Option A, §3.)**

- **The user's stated fallback — an explicit-SM-mask POOL green context — is
  also not feasible**, and this is already settled: `torch.cuda.GreenContext`
  is **count-only by C++ design** (Step 1.3b' Option 4, conclusively rejected
  from the torch 2.11 header — no constructor takes a `CUgreenCtx`, and
  PyTorch ignores an externally-pushed green context). **(Option C, §3.)**

- **The feasible mechanism is a *constrained POOL grant* — Option B (§3,
  recommended).** It *reassigns nothing and moves no tenant.* It changes only
  *which* groups the kmod grants the POOL: the POOL is granted the **maximal
  contiguous low prefix `[0, K-1]`** with no partition in it, instead of *every*
  free group. Non-adjacent freed groups simply **stay free** until the gap
  below them closes — at which point the POOL reclaims them, still contiguous.
  The POOL invariant ("contiguous low prefix, disjoint from every partition")
  holds at every instant, with **zero partition disruption, zero libcipher_rt
  change, zero `cp54_pool.py` change.**

**This is a deliberate mechanism deviation from the adjudication wording** —
surfaced here per the D1 adjudication's own instruction (*"Surface which
approach is feasible before committing to timeline"*). Option B achieves the
**same invariant** the adjudication asked for ("POOL on contiguous low prefix")
by a different, smaller mechanism. The adjudicator can override the mechanism,
not just the timeline.

**Revised estimate (a finding, not a footnote).** The adjudication budgeted
1.6X at **1–2 days** for "compaction." Option B is a **~15–25 LOC kmod change**
in one function + isolation-suite extension + one GPU sanity: estimate
**~0.5 day build (1.6X-2) + ~0.5 day verification (1.6X-3)**, total **~1 day**,
not 1–2. The 1–2 day figure priced relocation-compaction, which §3 shows is
infeasible.

---

## §1 — The fragmentation, traced through the live kmod source

`cipher_cp54_sched.c` (`8d777dfb`), the relevant facts as read:

- **`cp54_claim(pid, n)`** (line 114) scans groups **low→high** (`g=0..14`),
  cmpxchg-claiming up to `n` free groups.
- **`cp54_pool_shrink(n)`** (line 141) releases the POOL's **highest-indexed**
  groups (`g=14..0`).
- **POOL `ALLOCATE`** (line 252, `case CIPHER_CP54_QOS_POOL`) calls
  **`cp54_claim(pid, CIPHER_CP54_NUM_GROUPS)`** — i.e. **"claim *every*
  currently-free group."** This is the fragmentation source.
- **PARTITION `ALLOCATE`** (line 270): claims free groups low→high; if short,
  `cp54_pool_shrink`s the POOL (frees its high groups) then claims those.
- **FREE / reaper** (`cp54_release_impl`, line 192): `cp54_release_groups(pid)`
  frees *every* group the dying PID owned — lock-free, anywhere in the array.

**The bug, concretely (OP-5 layout).** POOL registers first → owns groups
`[0..14]`. Five 16-SM PARTITIONs allocate; each `pool_shrink(2)` frees the
POOL's top 2 groups and the partition claims them:

```
  POOL {0,1,2,3,4}   P5{5,6}  P4{7,8}  P3{9,10}  P2{11,12}  P1{13,14}
```

All contiguous — disjoint by construction. Now **P3 frees `{9,10}`** (teardown,
or an early/crashed exit — *not* lockstep, which OP-5/OP-asym will not have).
The POOL's `check_resize()` re-calls `ALLOCATE(POOL)` → `cp54_claim(pid, 15)` →
claims *every* free group → claims `{9,10}`:

```
  POOL {0,1,2,3,4,9,10}  ← NON-CONTIGUOUS, NOT a low prefix
```

`cp54_pool.py` then does `torch.cuda.GreenContext.create(7×8=56)` →
torch's **count-only** API picks the low prefix `{0..6}` → **overlaps the live
partitions P5`{5,6}` and P4`{7,8}`.** The kmod ledger says "disjoint"; the
physical SMs are not. The POOL's `%smid` self-verify *raises* (it fails loud,
correctly — but a raise is a **STOP, not a recovery**: the run aborts).

This is invisible at N=2 lockstep (`b2_op5_b2a`, which passed) and **first bites
at OP-5 / OP-asym** — inside Step 1.6.

---

## §2 — Why only the POOL needs contiguity (and partitions do not)

The asymmetry is the key to why Option B is small:

- **PARTITION green contexts are `grp_mask`-precise.** libcipher_rt selects the
  exact set-bit groups from `cuDevSmResourceSplitByCount` (Step 1.3, scope memo
  §2). A partition can own an **arbitrary, even non-contiguous** group set and
  still place its green context exactly. Partitions need **no** contiguity.
- **The POOL green context is count-only.** `torch.cuda.GreenContext.create(N)`
  takes an SM *count* and deterministically picks the **low-prefix** `N/8`
  groups (Step 1.3b' Appendix-A probe, 4/4 runs). The POOL can *only* be a
  **contiguous low prefix `[0, K-1]`** — that is the entire constraint.

So the fix is **POOL-side only**: ensure the kmod never grants the POOL a
non-low-prefix-contiguous set. Partitions are left completely alone.

---

## §3 — The three mechanisms

### Option A — relocation-compaction (move live partitions) — NOT FEASIBLE v1

On a middle-partition free, the kmod migrates higher partitions down to close
the gap, so freed groups bubble to the POOL boundary.

**Why it fails:** moving partition P from groups `{9,10}` to `{11,12}` rewrites
P's ledger entry, but P's libcipher_rt green context still *physically runs on
`{9,10}`*. The ledger would lie. To make it true, P must rebuild its green
context — requiring (1) a **kmod→tenant notification path** (does not exist),
(2) **libcipher_rt green-ctx refresh** (today create-once-never-refresh — Step
1.3), (3) **in-flight stream draining** on the moved tenant. This reopens
libcipher_rt (anchor `ebc0baaa`), adds substantial new substrate machinery, and
disrupts the strong-isolation PARTITION tenants — the ones that must be most
stable. **Rejected for v1.** (A notification + refresh path is plausible v2
work, alongside dynamic arbitration D10 — see §8.)

### Option B — constrained POOL grant (recommended)

The POOL is granted the **maximal contiguous low prefix** with no partition in
it. Nothing is moved or reassigned; non-adjacent freed groups stay free until
reachable.

**The change** — replace `cp54_claim(pid, CIPHER_CP54_NUM_GROUPS)` in the POOL
case (line 266) with a new helper, sketch:

```c
/* POOL grant: the maximal contiguous low prefix [0, K-1] containing no
 * partition. K = lowest partition-owned group, or NUM_GROUPS if none.
 * Caller holds cipher_cp54_lock. */
static u32 cp54_pool_claim_low_prefix(pid_t pool)
{
    u32 pool_pack = CIPHER_CP54_GRP_PACK(pool);
    int g, K = CIPHER_CP54_NUM_GROUPS;

    /* (a) K = first group owned by someone other than the pool. */
    for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++) {
        int s = atomic_read(&cipher_cp54_groups[g]);
        if (s != 0 && s != (int)pool_pack) { K = g; break; }
    }
    /* (b) claim every free group in [0, K-1] for the pool. */
    for (g = 0; g < K; g++)
        atomic_cmpxchg(&cipher_cp54_groups[g], 0, (int)pool_pack);
    /* (c) release any pool-owned group at or above K (defensive — handles a
     *     partition that landed low after a shrink). */
    for (g = K; g < CIPHER_CP54_NUM_GROUPS; g++)
        atomic_cmpxchg(&cipher_cp54_groups[g], (int)pool_pack, 0);

    return cp54_mask_of(pool);
}
```

**Trace, OP-5 out-of-order free.** POOL`{0..4}`, P5`{5,6}`…P1`{13,14}`.
- P3 frees `{9,10}`. POOL `check_resize` → `cp54_pool_claim_low_prefix`: K=5
  (P5 is the lowest partition). POOL claims `[0..4]` (already owns them).
  `{9,10}` **stay free** (stranded). POOL = `{0..4}` — **contiguous ✓ disjoint
  ✓**, `create(40)` → torch `{0..4}` → self-verify PASS.
- P5 frees `{5,6}`. K=7 (P4). POOL claims `[0..6]` — picks up `{5,6}`. POOL =
  `{0..6}` contiguous ✓. `{9,10}` still stranded.
- P4 frees `{7,8}`. K=11 (P2). POOL claims `[0..10]` — picks up `{7,8,9,10}`.
  Stranded groups reclaimed the moment the gap closed.

The POOL is a contiguous low prefix **at every instant**, for **any** free
order. Partitions are never touched.

**Step 1.6 measurement impact: none.** Within a barrier'd OP run partitions are
co-resident for every round and free only at teardown; even at teardown Option
B keeps the POOL valid. 1.6B-4 measures exactly what it should.

**What B leaves on the table (name it explicitly).** Under *sustained*
partition churn, B can **transiently strand free capacity**: a long-lived low
partition at group 5 with everything above it freed leaves the POOL at `[0..4]`
even though 9 groups are physically idle. This is a **capacity-efficiency**
loss, never a correctness or disjointness loss. It is consistent with **D10
(dynamic per-request arbitration), which the user just ratified as deferred to
v2** — v1 is static per-session arbitration, so a static-allocation stranding
is in-scope-acceptable. Closing the stranding *too* is precisely Option A's
extra value, and Option A is v2 work. **B closes the Step 1.6 fragmentation
defect; A would additionally close sustained-churn capacity efficiency, which
v1 has already deferred.**

### Option C — explicit-SM-mask POOL green context — NOT FEASIBLE

The user's stated fallback: have the POOL build a `grp_mask`-precise green
context (raw driver path) so it survives fragmentation. **Already settled in
Step 1.3b' Option 4:** `torch.cuda.GreenContext` is **count-only by C++
design** — the torch 2.11 header has no constructor, public or private, that
takes a `CUgreenCtx`; the handle field is private; and PyTorch binds the device
**primary context**, ignoring an externally-pushed green context. Making the
POOL `grp_mask`-precise therefore requires the executor to abandon
`torch.cuda.GreenContext` for libcipher_rt's raw-driver + per-launch-CUPTI
enforcement path — i.e. **executor-on-libcipher_rt**, the Option (a) rejected
back in Step 1.3 Q1 for Phase B methodology continuity. Not a small change; not
recommended; and unnecessary, since Option B closes D1 without it.

---

## §4 — Atomicity & concurrency

`cp54_pool_claim_low_prefix` runs **inside `cipher_cp54_ioctl_allocate`, under
the existing `cipher_cp54_lock` mutex** (POOL `check_resize` issues an
`ALLOCATE(POOL)`). **No new lock, no new ordering.**

- **vs. another ALLOCATE** — serialized by the mutex. No interleave.
- **vs. the lock-free `do_exit` reaper** — the reaper does
  `atomic_cmpxchg(PACK(dying_pid) → 0)` per group; the helper does
  `atomic_cmpxchg(0 → PACK(pool))` (claim) and `atomic_cmpxchg(PACK(pool) →
  0)` (release). **Each cmpxchg requires its own specific expected value, so
  the two paths can never act on the same group transition** — the helper
  cannot claim a group still partition-owned (it would need to see `0`), and
  cannot be tricked into releasing a partition's group (it only releases groups
  reading `PACK(pool)`). If a low partition's reaper fires *during* the
  helper's K-scan, K may be computed stale-high → the POOL claims one group too
  many? No: a reaper *frees* a group (partition→free), which can only make K
  *larger* (that partition no longer blocks the prefix); the helper then
  under-claims at worst (sees the group still as the old partition during the
  scan, stops early) and **recovers on the next `check_resize`**. The failure
  mode is *transient under-grow*, never an incorrect grant — the identical
  "degrades to retry, never to incorrect allocation" property the scope memo §5
  already accepts for the pool-resize race.
- **`release ≥ K` branch (c)** — only ever cmpxchg's `PACK(pool)→0`, i.e.
  releases groups the POOL itself owns; cannot affect a partition.

No change to the FREE path, the reaper, QUERY, the PARTITION path, or the
metadata table.

---

## §5 — Coordination with libcipher_rt and `cp54_pool.py`: NONE required

- **libcipher_rt (PARTITION tenants):** untouched. Partitions are never moved,
  never notified, never rebuild. Anchor `ebc0baaa` does **not** rotate.
- **`cp54_pool.py` (POOL executor):** untouched. `check_resize()` already
  re-`ALLOCATE`s each round, rebuilds the green context when `grp_count`
  changes, and `%smid`-self-verifies the low-prefix placement (Step 1.3b').
  With Option B the kmod now returns a contiguous-low mask, so the *existing*
  self-verify **passes** where today it would **raise**. No code change — the
  fix makes already-written verification logic succeed.
- **libcipher_v2 / cipher_kv_bridge:** not in scope, untouched.

This "no-coordination" property is the core feasibility win of Option B over
Option A: the POOL is *already* the only churning party with a rebuild path;
constraining its grant needs no new cross-process machinery.

---

## §6 — Verification plan (1.6X-3) — and which steps need a GPU

**Two layers; only one needs a GPU.**

1. **Kmod-side correctness — NO GPU.** Extend the pure-C isolation suite
   `cp54_isolation_test.c` (the Step 1.2 harness, `/dev/cipher` ioctls only)
   with **Test 7 — out-of-order free**: register a POOL; `ALLOCATE` 5 PARTITION
   tenants (16 SM each — fork/`_exit` pattern, as Test 4); free them in a
   **randomized order**; after each free, re-`ALLOCATE(POOL)` and assert
   (a) the POOL mask is a **contiguous run starting at bit 0**
   (`mask & (mask+ (1<< K)) …` — i.e. `mask == (1<<popcount)-1` shifted from 0,
   so `mask+1` is a power of two), and (b) the POOL mask ∩ every live
   partition mask = ∅. Several randomized orders. This is the load-bearing
   correctness gate and runs entirely in the kmod — **no CUDA, no GPU**.
2. **POOL green-ctx self-verify under fragmentation — ONE GPU sanity.** Drive
   the existing `cp54_s16_orchestrator.py` with 5 PARTITION tenants + the POOL,
   and have partitions exit in non-FIFO order; confirm the POOL's `%smid`
   self-verify **PASSES** at each `check_resize` (it would *raise* on the
   unfixed kmod). Reuses Step 1.6B harnesses; ~1 GPU run.
3. **Regression — ONE GPU run.** The Step 1.2 / 1.3 W1 Phase-B regression
   (N=8 TinyLlama, ±3% mean-vs-mean) on the rotated kmod, to confirm the POOL
   grant change does not perturb the steady-state path.

1.6X-3 does **not** re-budget what Test 7 covers: kmod logic is proven without
a GPU; the GPU runs only confirm the torch-side composition.

---

## §7 — Anchors

| anchor | now | after 1.6X | fallback |
|---|---|---|---|
| kmod `cipher_kmod.ko` | `8d777dfb` | **rotates** (1.6X-2 build + 1.6X-3 verify) | preserve `8d777dfb` **outside the kbuild dir** as `cipher_kmod_fallback/cipher_kmod.ko.pre_cp5_4_step1_6x` ([[cipher-kbuild-clean-wipes-ko]]) |
| libcipher_rt | `ebc0baaa` | unchanged | — |
| libcipher_v2 | `86618c30` | unchanged | — |
| cipher_kv_bridge | `fca6843d` | unchanged | — |

The kmod reload at 1.6X-2 is a drain/`rmmod`/`insmod` cycle on a quiescent pod
(scope memo §6) — the same procedure as Step 1.2; rollback target is the
preserved `8d777dfb`.

---

## §8 — Estimate & relation to the revised Step 1.6 plan

| sub-phase | scope | GPU | est. |
|---|---|---|---|
| 1.6X-1 | this design memo | no | ~2–3 h (done) |
| 1.6X-2 | kmod build: `cp54_pool_claim_low_prefix`, replace line 266; rebuild; drain-reload; anchor rotation | reload only | ~0.5 d |
| 1.6X-3 | Test 7 (no GPU) + 1 GPU self-verify sanity + 1 GPU regression | yes (2 runs) | ~0.5 d |

**Total 1.6X ≈ 1 day**, vs the 1–2 days the adjudication budgeted for
relocation-compaction. The saving is real: Option B is a one-function kmod
change, not a new cross-process notification subsystem.

**v2 note.** Relocation-compaction (Option A) — closing the sustained-churn
capacity stranding §3 names — is natural v2 work, bundled with **D10 dynamic
per-request arbitration**: both need the same missing piece, a kmod→tenant
notification path + libcipher_rt green-ctx refresh. Recommend tracking "Option
A relocation-compaction" as a line item under the CP 5.4 v2 / dynamic-
arbitration workstream, not as orphaned debt.

---

## §9 — Adjudication ask

**STOPPING HERE for adjudication — no source modified, no build, no GPU.**

1. **Mechanism (§0/§3) — accept Option B (constrained POOL grant)** in place of
   the adjudication's "relocation-compaction" wording. B achieves the same
   invariant ("POOL contiguous low prefix, disjoint") with a ~15–25 LOC
   kmod-only change, zero partition disruption, zero libcipher_rt/`cp54_pool.py`
   change. Option A (true relocation) is infeasible for v1 (§3); Option C
   (explicit-mask POOL) is infeasible (Step 1.3b' Option 4).
2. **Accept the named v1 boundary (§3)** — Option B leaves *transient
   capacity stranding* under sustained partition churn (never a correctness
   loss); closing it is Option A / v2 work, consistent with the already-ratified
   D10 deferral.
3. **Accept the revised estimate** — 1.6X ≈ **1 day** (not 1–2), because the
   feasible mechanism is smaller than relocation-compaction.
4. **Accept the §6 verification plan** — kmod correctness via a no-GPU
   `cp54_isolation_test.c` Test 7 (randomized out-of-order free); one GPU
   self-verify sanity + one GPU regression.

On adjudication: proceed to **1.6X-2** (kmod build), then **1.6X-3** (verify),
then the revised Step 1.6 plan (1.6Y → 1.6B-3 → 1.6B-4 → 1.6P → 1.7 → 1.8).

No code until this memo is adjudicated.
