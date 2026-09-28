# Phase 5 prep — kmod SM-partition allocator: exhaustion diagnostic (TRACKED)

**Date raised:** 2026-05-16. **Status:** tracked work item — **not a CP 2.4
blocker**. Diagnose during Phase 5 multi-tenancy prep, before the
substrate+actuator composition build.

---

## Observation

During the CP 2.4 Marlin-hang debugging session, a test A retest
(`test_a_density.py`, substrate ON, kmod 0.4.8) stalled at its first step
(N=1). The visible cause, at model-load time:

```
[cipher_v2] ARB: REQUEST_SM_PARTITION rc=-1 errno=28 No space left on device
```

The kmod `REQUEST_SM_PARTITION` ioctl (nr 9) returned `-ENOSPC` — the kmod's
SM-partition slot pool was **exhausted**. The libcipher_rt ARB-poll thread
then reissued every 30 s, failing `ENOSPC` each time (~134 reissues ≈ 67 min)
before the workload process ended.

**Context of the exhaustion:** the kmod had been loaded ~24 h (uptime
≈ 45 909 s at first observation, then a fresh window), and this debugging
session ran **25+ substrate-on workload processes** (smokes, diagnostic
repros, isolation tests, fix smokes, the instrumented run, the test A
retest). Each registers a tenant and requests an SM partition.
`/proc/cipher/stats` showed `reaped=100` — the `do_exit` reaper had freed 100
per-PID hashtable entries — but the workload still could not obtain a
partition, i.e. the **SM-partition slots specifically were not available**.

A kmod reload (`rmmod`/`insmod`, same 0.4.8 `e2f50452`) resets the allocator
state and clears the condition — this was done as session hygiene and the
substrate-on config then ran clean.

---

## Two suspected causes — to be distinguished in Phase 5 prep

1. **Reclaim leak.** SM-partition slots are not released on tenant/process
   exit. The `do_exit` reaper frees the per-PID hashtable entry but may not
   free the tenant's partition-slot reservation; or the 30 s idle-reclaim
   path (`cipher_partition_allocator.c` — "FCFS + quartile-rebalance + 30 s
   idle-reclaim") does not actually return slots to the free pool. If so,
   any long-lived deployment leaks slots until `ENOSPC` — a real bug.

2. **Pool sizing.** The slot pool is simply smaller than the cumulative
   distinct-tenant count a long session produces, and reclaim *is* working
   but is bounded/lagging. If so, the fix is sizing/policy, not a leak.

Distinguishing them needs a read of `cipher_partition_allocator.c` (pool
size, the free-on-exit path, the idle-reclaim implementation) plus a
controlled repro: N sequential substrate-on runs with `/proc` slot-occupancy
sampled between each. **Deliberately not done now** — see scope below.

---

## Scope

- **Not a CP 2.4 blocker.** The CP 2.4 gate is **single-tenant** (per-lever
  and composed lift, n=5 matched pairs, Llama + Mistral). A single fresh
  substrate-on process obtains its partition without issue; the exhaustion
  only appears after many cumulative runs. Session hygiene (periodic kmod
  reload) keeps CP 2.4 measurement clean.
- **Is a Phase 5 prerequisite.** Phase 5 multi-tenancy composes the substrate
  with actuators under sustained, many-tenant load — exactly the regime that
  surfaces this. Diagnose and fix (or codify a sizing/reclaim policy) before
  the Phase 5 multi-tenant build.
- Related Phase 5 constraint: `cp_2_4/PHASE_5_MARLIN_PARTITION_CONSTRAINT.md`
  (Marlin GEMM is structurally full-GPU — to be written with the Fix A docs).

**Action:** Phase 5 prep — read `cipher_partition_allocator.c`, run the
sequential-exhaustion repro, determine leak vs sizing, fix accordingly.
