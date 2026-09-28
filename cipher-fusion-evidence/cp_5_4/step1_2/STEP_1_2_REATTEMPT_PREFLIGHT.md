# CP 5.4 Step 1.2 re-attempt — pre-flight verification

**Date:** 2026-05-18. Verifies the authorized fix (move `cipher_cp54_release`
above the `cipher_do_exit_pre` early-return) does not break the legacy reaper.

## 1. What the early-return guard is for

`cipher_probe.c` header comment (lines 170–174), verbatim:

> *do_exit reaper … Fires for every thread/task exit on the system, not just
> nvidia consumers, so the fast path must be O(1) for not-tracked PIDs.*

The guard `if (likely(!e)) return 0;` is a **performance fast-path**, not a
correctness gate. `do_exit` fires on every task exit system-wide; the
overwhelming common case is a process with no `cipher_pid_stats` entry, and the
guard makes that case O(1). A `cipher_pid_stats` entry exists only for a pid
that did one of: `REGISTER_TENANT`, a telemetry submit, or an observed NVIDIA
`nvidia_unlocked_ioctl` (the kprobe hot path creates entries for GPU consumers).

## 2. Why the placement was correct for the legacy reaper paths

The two reaper actions below the guard stay below it:

- **`hash_del_rcu` + `kfree_rcu(e)`** — *requires* `e`. Correctly below the
  guard by necessity.
- **`cipher_partition_release_slots_only(pid)`** — releases legacy nr-9 4-SM
  slots. Below-guard was correct *in practice*: a legacy partition tenant
  acquired slots via the nr-9 ioctl, and any process doing GPU work is an
  observed NVIDIA consumer → it always had a `cipher_pid_stats` entry → the
  guard never blocked the legacy slot release. (And nr 9 is now deactivated by
  CP 5.4 — no new legacy allocations exist at all.) **Left unchanged.**

## 3. Why the placement was wrong for CP 5.4

The CP 5.4 group ledger (`cipher_cp54_sched.c`) is an **independent
subsystem**. A process can `open(/dev/cipher)` and `CIPHER_CP54_ALLOCATE`
groups **without** ever calling `REGISTER_TENANT`, submitting telemetry, or
doing an NVIDIA ioctl — so it can hold CP 5.4 groups with **no
`cipher_pid_stats` entry**. Step 1.1 wired `cipher_cp54_release(pid)` *below*
the guard, so such a process leaks its groups on exit (Phase 2 Test 4). The
CP 5.4 ledger's crash-safety must not be coupled to the legacy
`cipher_pid_stats` lifecycle.

## 4. Why moving `cipher_cp54_release` above the guard is safe

- **Disjoint state.** `cipher_cp54_release` → `cp54_release_impl` touches
  *only* `cipher_cp54_groups`, `cipher_cp54_allocs`, `cipher_cp54_pool_pid`.
  It does not read or write `cipher_pid_stats`, the hashtable, the insert
  lock, or the legacy partition slots. Running it before vs after the guard
  cannot affect any legacy reaper path — order-independent.
- **Atomic-context safe.** Lock-free (`atomic_cmpxchg`, `READ_ONCE`/
  `WRITE_ONCE`) — no sleep — correct in the kprobe `pre_handler`.
- **Idempotent / no-op for non-owners.** For a pid owning no CP 5.4 groups it
  is a bounded scan (16 group reads + 64 metadata reads + 1 pool cmpxchg) that
  finds nothing — correct, side-effect-free.
- **Cost.** It now runs on every `do_exit`, adding ~bounded ~100 atomic
  reads/compares to each task exit. This is still O(1) in the comment's sense
  (bounded constant, not scaling with system load) and negligible against
  `do_exit`'s own cost (mm/fd/signal teardown). A global "any CP 5.4
  allocations live" fast-out is a possible future micro-opt — **not done here**;
  the authorized fix is the placement move, and the cost does not warrant it.

## Verdict

The fix — move `cipher_cp54_release(pid)` above the `if (likely(!e)) return 0`
guard — is safe: it does not touch what the guard protects, the legacy paths
stay correctly below the guard, and the per-exit cost is negligible. Proceed
with the source fix + rebuild + Step 1.2 re-attempt.
