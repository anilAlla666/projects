# T4.2.1 partition allocator — algorithm review (Concerns 1+2)

**Date:** 2026-05-13
**Scope:** `cipher_kmod/cipher_partition_allocator.c` (325 lines, staged in 0.4.1 build, not yet loaded).
**Status:** Pre-cutover review. No code changes made by this document.

## Concern 1.A — Quartile rebalance algorithm

### Pseudocode

```
function rebalance():
    # snapshot phase (RCU read)
    rcu_read_lock()
    loads = []
    for entry in cipher_pid_table:
        mask = READ_ONCE(entry.sm_partition_mask)
        if mask == 0: skip            # only consider tenants with slots
        loads.append((entry.pid, READ_ONCE(entry.launches_total)))
    rcu_read_unlock()

    # sort phase (process context)
    sort(loads, key=launches, desc=True)
    n = len(loads)

    # cap-trim phase (per-entry RCU + per-write spinlock)
    for i, (pid, launches) in enumerate(loads):
        quartile = (i * 4) // n           # 0..3
        cap = {0: 8, 1: 4, 2: 2, 3: 1}[quartile]
        rcu_read_lock()
        for e in bucket(pid):
            if e.pid != pid: continue
            cur = READ_ONCE(e.sm_partition_mask)
            if popcount(cur) > cap:
                spin_lock_irqsave(cipher_partition_lock)
                trimmed = pick_lowest_bits(cur, cap)
                WRITE_ONCE(e.sm_partition_mask, trimmed)
                WRITE_ONCE(e.sm_partition_count, popcount(trimmed))
                spin_unlock_irqrestore(cipher_partition_lock)
            break
        rcu_read_unlock()
```

### Window definition

**`launches_total` is the value the workload's CUPTI hook has accumulated since `REGISTER_TENANT`** (i.e., since the workload started). It is NOT a moving window, EMA, or per-rebalance delta.

**Implication:** a tenant that launched 1,000,000 kernels in the first 30 seconds and then went idle still ranks in the top quartile vs. a tenant that just started and only logged 50 launches. The rebalancer **does not adapt to recent activity**. This is a deliberate v1 simplification — easy to reason about, but biased toward old/historical activity.

**Phase 4.7+ refinement candidate:** track `launches_total - launches_at_last_rebalance` and use the delta. Or maintain an EMA. Not blocking T4.2.1.

### Cap-only adjustment, no growth

The rebalancer **only trims** tenants over their quartile cap. It does NOT proactively give more slots to under-allocated heavy tenants. That happens reactively: when a heavy tenant `cipher_partition_request`s again with `hint=8`, and the rebalance has trimmed light tenants, the freed slots become available.

This is "passive rebalance" — heavy tenants must keep re-requesting to claim slots freed by trimming. Suitable for the planned cipher_rt SM_PACKER / ARBITRATE workflow (Stage 2 thread polls + re-requests every 100 ms).

## Concern 1.B — Initial assignment algorithm: fairness under contention

### Algorithm (in `cipher_partition_request`)

```
hint = clamp(hint, 1, 8)
spin_lock_irqsave(cipher_partition_lock)
rcu_read_lock()
locate entry e for target_pid; if not found → -ESRCH

cur_mask = READ_ONCE(e.sm_partition_mask)
if cur_mask != 0 AND popcount(cur_mask) >= hint:
    # idempotent return — already have enough
    return cur_mask, popcount(cur_mask)

free_mask = (FULL_MASK & ~union(all_other_tenants_masks))
avail = popcount(free_mask)
grant = min(hint, avail)
if grant < 1:
    if cur_mask != 0: return cur_mask  # keep existing
    else:              return -ENOSPC

new_mask = cur_mask | pick_lowest(free_mask, grant)
WRITE_ONCE(e.sm_partition_mask, new_mask)
WRITE_ONCE(e.sm_partition_count, popcount(new_mask))
return new_mask
```

### 10-tenant × 4-each → 33 slots scenario

- Total demand: 10 × 4 = 40 slots
- Available: 33 slots
- Assignment in arrival order:
  - Tenants 1–8 get 4 slots each (32 used, 1 free)
  - Tenant 9 gets 1 slot (33 used, 0 free)
  - Tenant 10 gets `-ENOSPC` (cur_mask was 0)

**This is starvation-FCFS, not proportional fairness.** Old tenants who got there first hold their full allocation; late arrivals get whatever's left. There is no fairness pass.

**Mitigations available today:**
- The 5-second rebalance will trim old tenants if their `launches_total` drops them out of the top quartile. After ~5 s, late arrivals can re-request and likely get a slot.
- The 30-s idle reclaim frees slots from completely-idle tenants.

**Gap:** "many small tenants vs few big tenants" with all tenants busy = late tenants stay at 0 slots until rebalance trims early tenants. The 5-s rebalance latency creates a noticeable head-of-line for new tenants.

**Recommendation for T4.2 fairness lift (not blocking T4.2.1):** add a proportional-share pass on request when `avail < hint` — instead of giving 0 to the latest tenant, trim the top quartile by 1 slot each to make room.

## Concern 1.C — Race condition story

### Three concurrent actors

1. **Userspace ioctl call** (`cipher_dev_request_sm_partition`) — runs on caller's CPU, takes `cipher_partition_lock` with IRQs disabled.
2. **state_updater kthread** — invokes `cipher_partition_tick()` every 5 s, which calls `reap_idle()` and `rebalance()`. Both internally take `cipher_partition_lock`.
3. **do_exit reaper** (Phase 1 kprobe) — removes entries from `cipher_pid_table` on process exit. Uses RCU + the existing `cipher_pid_insert_lock`, NOT `cipher_partition_lock`.

### Race scenarios analyzed

| Race | Outcome |
|---|---|
| Tenant A requests while rebalance is trimming tenant A | Serialized: rebalance takes lock for write, request takes lock for read+write. One sees the other's result. No torn state. |
| Tenant A requests slots; concurrent rebalance gives those slots to tenant B | Impossible — rebalance only trims, never grants. A's request will compute free_mask after rebalance's writes complete, so A sees freed slots. |
| Tenant A exits via do_exit; concurrent rebalance reads A's entry | RCU read-side ensures A's `cipher_pid_stats` remains valid during the rebalance walk. After A's exit + RCU grace period, A is gone from subsequent walks. No use-after-free. |
| Two userspace tenants A and B request simultaneously | Spinlock serializes. The second waiter sees the first's writes in its `compute_free`. Order matters (FCFS in spinlock contention), but no slot is granted twice. |
| state_updater calls `tick()` while a request is in progress | Tick takes the same spinlock. Tick will wait for the request, or vice versa. No state corruption. |

### Memory ordering

- `WRITE_ONCE(e->sm_partition_mask, x)` is compiler-fence only. On x86-64, aligned 4-byte stores are torn-write-free.
- `READ_ONCE` likewise compiler-fence only.
- Cross-CPU visibility: the spinlock acquire/release provides the necessary memory barriers (`smp_mb__after_spinlock`).
- **Therefore: readers outside the lock (e.g., `cipher_get_current_tenant_snapshot`) see eventually-consistent values.** A reader might see an inconsistent (mask, count) pair if it reads between the two WRITE_ONCEs of a writer. This is acceptable per the contract — the snapshot has `snapshot_jiffies` for freshness, and the reader can verify count = popcount(mask) if it cares.

## Concern 1.D — Invariant: two tenants cannot hold the same slot

### Statement

For any time **t** at which `cipher_partition_lock` is not held by any thread:
> For all pairs of distinct cipher_pid_stats entries `e1, e2`:
> > `READ_ONCE(e1.sm_partition_mask) & READ_ONCE(e2.sm_partition_mask) == 0`

### Enforcement

The invariant is enforced by:

1. **Every write to `sm_partition_mask` happens while holding `cipher_partition_lock`**. The five write sites:
   - `cipher_partition_request` line 141
   - `cipher_partition_request` failure path (no write, just return)
   - `cipher_partition_release` line 163
   - `cipher_partition_reap_idle` line 197
   - `cipher_partition_rebalance` line 275
   
   All five take `spin_lock_irqsave(&cipher_partition_lock)` before the write.

2. **`cipher_partition_request`'s grant computation uses `compute_free`**, which `AND`s the complement of every other tenant's mask. Therefore the granted bits are disjoint from every existing assignment AT THE MOMENT OF COMPUTATION.

3. **Since `compute_free` runs under the same spinlock as the write**, no concurrent allocator can sneak in a write between `compute_free` and the WRITE_ONCE.

**Note:** `reap_idle` and `rebalance` only *clear* bits, never set them — so they cannot violate the invariant.

**Note:** `REGISTER_TENANT` does NOT touch `sm_partition_mask` — only the allocator does. New entries always start with mask=0 (cleared by `memset` in `cipher_pid_get_or_create`).

**This invariant is the safety basis for `cipher_rt`'s use of `STREAM_ATTR_LAUNCH_AFFINITY` per tenant — no two tenants can be scheduled onto the same physical SM partition simultaneously.**

## Concern 2 — Idle reclaim reawakening behavior

### Question: when a tenant idle > 30 s has its mask reclaimed to 0, what happens on its next kernel launch?

### Answer: **Option C — permanently parked.**

The tenant continues to exist in `cipher_pid_table` (do_exit reaper hasn't fired; the process is still alive, just idle on the GPU). Its mask is now 0.

When the tenant's next kernel launch happens:
- **libcipher_v2's CUPTI callback** increments `g_launches_total` and flushes via `CIPHER_SUBMIT_LAUNCH_STATS` ioctl. This **does not auto-rerequest** a partition; it only updates the per-tenant launches counter on the kernel side.
- **cipher_rt's PARTITION_ROUTER** (planned for T4.2.2 — not yet written) would observe the next kernel launch, read the cached snapshot, see `sm_partition_mask == 0`, and **need to issue a fresh CIPHER_REQUEST_SM_PARTITION ioctl** to recover.
- **If PARTITION_ROUTER is not active** (cipher_rt isn't loaded, or the actuator is disabled), the launch proceeds with mask=0 → no SM affinity → NVIDIA's stock scheduler decides. **Effectively equivalent to no-CIPHER baseline.**

### Code path verification

`cipher_state_updater` derives sm_partition_mask from `cipher_pid_stats.sm_partition_mask`. After idle reclaim sets this to 0, the snapshot reads back 0. PARTITION_ROUTER's Mode 3 cache will see 0. There is no auto-restore.

### Implications

1. **For the T4.2.1 standalone test**, the idle reclaim path is exercised, but reawakening must be tested by issuing a new `CIPHER_REQUEST_SM_PARTITION` ioctl. The test program `cipher_test_phase4_partition.c` already has a placeholder for this (Test 6 currently skipped with informational message).

2. **For T4.2.2** when PARTITION_ROUTER is written, it must include reawake logic: when reading mask=0 from snapshot AND the tenant has a registered tenant_id, re-issue the ioctl.

3. **For T4.2.1 cutover**, this behavior is correct as designed. PARTITION_ROUTER handles the reawake. Document Test 6 as "soak-only — requires daemon telemetry to drive last_telemetry_jiffies."

### Added test case (T4.2.1.A Test 8)

A new test will exercise:
1. Request 4 partitions → get mask M1
2. Don't issue any telemetry for 35 s (mask reaches idle reclaim threshold)
3. Verify mask is now 0 in `cipher_get_current_tenant_snapshot`
4. Issue a fresh request → get a (possibly different) mask M2

**Caveat:** Test 8 requires the daemon to NOT submit telemetry for the test tenant for 35 s. The daemon polls NVML which sees all CUDA processes, so as long as the test tenant doesn't issue CUDA launches, telemetry stays stale. Without CUDA the test runs purely on the ioctl interface — fine.

## Summary of recommendations before cutover

| Recommendation | Block cutover? | Rationale |
|---|---|---|
| Document the FCFS-starvation property of initial assignment | No | Working as designed for v1; documented for future fairness work |
| Add proportional-share fallback when `avail < hint` | No | T4.2 lift work; doesn't break correctness |
| Track recent-launches-delta instead of cumulative for rebalance | No | T4.7+ refinement |
| Auto-rerequest on launch after idle reclaim | No | PARTITION_ROUTER (T4.2.2) handles this |
| Add Test 8 for reclaim-then-reawaken via fresh ioctl | **Yes** | T4.2.1.A explicitly asks for this case |
| Replace hard-coded 132/33 with cipher_silicon.sm_count or module param | **Yes** | Concern 3 — required before live-load on non-H100-SXM5 pods |
| Multi-threaded contention perf test | **Yes** | Concern 4 — required to validate spinlock isn't a bottleneck |

Concerns 1+2 documented. Concerns 3, 4, 5 addressed in subsequent code changes + tests.
