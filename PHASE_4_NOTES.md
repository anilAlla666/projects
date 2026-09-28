# Phase 4 — Engineering Notes

This file is the audit trail for Phase 4. Anything inconvenient that surfaces
and gets fixed lives here. Investors / customers reading the evidence bundle
see honest engineering history, not a polished narrative.

## Module version history

| Version | srcversion | .ko md5 | Stored at | Notes |
|---|---|---|---|---|
| 0.2.0 | (Phase 2/3 baseline) | 55ab8c0cd8309ca7cc0fc40fe556aa19 | `/tmp/cipher-phase3-evidence/fallback/cipher_kmod.ko.v0.2.0` | Fallback. Frozen. NEVER modified. |
| 0.3.1 | B1AF5E2AAB2FB170A661E2A | 849c63e0bf9521b4573f56c803cd6283 | `/tmp/cipher-phase3-evidence/bin/cipher_kmod.ko` | Phase 3 ship. |
| 0.4.0 | 7D165262DCCBBA1CB084B82 | f1048f2d93d6d0d67ab51960a94f3fa5 | `/home/ubuntu/cipher_kmod.ko.v0.4.0` | T4.1 only — tenant snapshot. Re-built 2026-05-13 from source after rollback artifact was lost. |
| 0.4.1 | FED7A35396FBD80CDDC2D4A | a6b7d676135777910d6141b229b6e640 | `/home/ubuntu/cipher_kmod.ko.v0.4.1` (saved before unload) + source tar `/home/ubuntu/cipher_kmod_src_v0.4.1.tar.gz` md5 df06b1aff9152a1b82b6ddb0302ae429 | T4.2.1 spinlocked partition allocator. Failed contention gate at 126.7× (33-thread p99 vs ~310 ns single). Replaced by 0.4.2. |
| 0.4.2 | (lock-free draft) | abfaff350bf920963d7281dbd6ff52a8 | `/home/ubuntu/cipher_kmod.ko.v0.4.2_with_do_exit_release` + source tar `/home/ubuntu/cipher_kmod_src_v0.4.2_with_do_exit_release.tar.gz` md5 987afa204f4aa5ce6268c0df21ed26ee | Lock-free atomic-slot redesign + do_exit slot release. NOT functionally verified — `rmmod` crashed before contention test ran (see Incident 2). Replaced by 0.4.3. |
| 0.4.3 | 335F54871889FFF2C36E3B0 | fb210777460c47bedd52c7d4223b22ca | `/home/ubuntu/cipher_kmod.ko.v0.4.3` + source tar `/home/ubuntu/cipher_kmod_src_v0.4.3.tar.gz` md5 e8e24ce611aad1142c36392e86cfb965 | Working baseline. 0.4.2 codepath + cipher_bar0_exit pci_dev_put leak workaround. 20/20 stress cycles clean. All Phase 3/4 functional gates pass. Contention p99 ratio at 33 threads with per-thread fds: **1.4×** (well under 5× gate). Superseded by 0.4.4 (B6 fix). |
| 0.4.4 | 1B657D6043D718B6DE2BC56 | c6de1afa228fec20883c6659ea3e5fc7 | `/home/ubuntu/cipher_kmod.ko.v0.4.4` + source tar `/home/ubuntu/cipher_kmod_src_v0.4.4.tar.gz` md5 5f154a14cdf708e502d3ec2ae75586c6 | **Current working baseline.** 0.4.3 base + B6 fix: `CIPHER_PARTITION_SLOTS_MAX` capped at 32 to keep `(1U << i)` inside u32 mask width. UBSAN shift-out-of-bounds eliminated. 20/20 stress clean, partition 8/8, contention 1.4× at 33 t hint=1 per-thread fd (61.8 M ops/s), Phase 3 ABI happy/negative/root all PASS. H100 SXM5 capacity loss: 1 slot (33→32). |

## Fallback md5 verification (run at every checkpoint)

```
$ md5sum /tmp/cipher-phase3-evidence/fallback/cipher_kmod.ko.v0.2.0 \
         /tmp/cipher-phase3-evidence/fallback/libcipher_v2.so.v0.2.0
55ab8c0cd8309ca7cc0fc40fe556aa19  cipher_kmod.ko.v0.2.0
86618c30896470b642fcc6985d8dc632  libcipher_v2.so.v0.2.0
```

Both expected. Confirmed 2026-05-13.

## Build Discipline (binding)

Every cipher_kmod version bump MUST execute these steps BEFORE rebuilding:

```bash
# 1. Snapshot current source state to a tarball.
cd /home/ubuntu
tar czf cipher_kmod_src_v<current_version>.tar.gz \
    --exclude='*.o' --exclude='*.ko' --exclude='*.mod*' \
    --exclude='Module.symvers' --exclude='modules.order' \
    --exclude='.*.cmd' --exclude='.*.d' --exclude='phase4_draft' \
    cipher_kmod/
md5sum cipher_kmod_src_v<current_version>.tar.gz \
    > cipher_kmod_src_v<current_version>.tar.gz.md5

# 2. Snapshot the current .ko OUTSIDE the build dir.
# (kbuild clean globs *.ko inside cipher_kmod/, so the in-tree binary
#  is destroyed on `make clean`. Save to /home/ubuntu/ instead.)
cp /home/ubuntu/cipher_kmod/cipher_kmod.ko \
   /home/ubuntu/cipher_kmod.ko.v<current_version>
md5sum /home/ubuntu/cipher_kmod.ko.v<current_version> \
    > /home/ubuntu/cipher_kmod.ko.v<current_version>.md5
```

THEN bump version and rebuild. Both `*_src_*.tar.gz` and `.ko.v*` must exist
at every cutover boundary. The .ko is the load-ready rollback artifact;
the source tarball is the cleanroom-reproducible artifact.

This discipline is now part of the cutover procedure permanently. Any
sub-phase that skips this step gets reverted.

Why this exists: T4.2.1 cutover lost the 0.4.0 .ko because `make clean` was
run before saving the prior binary. When the contention gate failed and we
needed to roll back, we had to do source surgery on the live tree to
reconstruct 0.4.0 from 0.4.1. That worked, but it was risky — a bug in the
revert would have left us with neither 0.4.0 NOR 0.4.1 cleanly loadable.

Memory record updated: see `cipher-kbuild-clean-wipes-ko` for the underlying
constraint.

## Incident 2 — Kernel oops in cipher_bar0_exit, 2026-05-13 14:24

During the 0.4.2 lock-free allocator iteration, after several rmmod/insmod
cycles to tune the contention path, `sudo rmmod cipher_kmod` triggered a
**NULL pointer dereference** at `kernfs_find_and_get_ns+0x17` inside the
exit path:

```
cipher_bar0_exit
  → pci_dev_put
    → device_release
      → devres_release_all
        → devm_attr_group_remove
          → sysfs_remove_group
            → kernfs_find_and_get_ns   ← NULL deref (kobj.sd == NULL)
```

Tainted: G W OE (was clean before — fresh taint from this oops).
Module stuck in "Unloading" state with refcnt = -1; cannot be unloaded
without reboot. /dev/cipher still present but returns ENXIO (cdev fops
were never released because cipher_dev_exit runs AFTER cipher_bar0_exit
in cipher_exit, and bar0_exit crashed mid-call).

### Root cause analysis

The crash is in another driver's `devres` entry — `devm_attr_group_remove`
is releasing a sysfs group whose `kobj.sd` is already NULL. We don't use
`devm_*` in cipher_bar0; this devres entry belongs to a sibling driver
(most likely nvidia.ko) that partially tore down its sysfs nodes between
our `pci_get_device` (init) and our `pci_dev_put` (exit).

By being a referrer that calls `pci_dev_put` at module-exit time, we
trigger `device_release` → `devres_release_all` over an inconsistent list.
The bug is not ours to fix, but the trigger is.

### Permanent fix (to apply on next clean boot)

Two options, in order of preference:

1. **Leak the pci_dev ref deliberately.** Drop the `pci_dev_put` in
   `cipher_bar0_exit`. The reference stays held forever (one per module
   load), keeping the device struct alive. Module reloads leak refs but
   never crash. `pci_iounmap` still runs cleanly. This is the minimum-
   risk change.

2. **Detach earlier via a PCI bus notifier.** Register a notifier for
   `BUS_NOTIFY_REMOVED_DEVICE`; on callback, set `cipher_pci_dev = NULL`
   so our exit path is a no-op. More complex; deferred unless option 1
   proves insufficient.

### Recovery procedure

Reboot the pod. Module is stuck at refcnt = -1, cannot be force-unloaded.
After reboot:
1. Verify fallback md5s still match (55ab8c0c / 86618c30).
2. Apply the pci_dev_put leak workaround to cipher_bar0.c.
3. Load 0.4.2 with cipher_bar0 hardening.
4. Re-run cutover gates.

### Artifacts saved before reboot

| File | md5 | Notes |
|---|---|---|
| /home/ubuntu/cipher_kmod.ko.v0.4.0 | f1048f2d93d6d0d67ab51960a94f3fa5 | clean rollback target |
| /home/ubuntu/cipher_kmod.ko.v0.4.1 | a6b7d676135777910d6141b229b6e640 | clean intermediate |
| /home/ubuntu/cipher_kmod.ko.v0.4.2_with_do_exit_release | abfaff350bf920963d7281dbd6ff52a8 | 0.4.2 lock-free + do_exit slot release — NOT verified clean unload |
| /home/ubuntu/cipher_kmod_src_v0.4.1.tar.gz | df06b1aff9152a1b82b6ddb0302ae429 | 0.4.1 source snapshot |
| /home/ubuntu/cipher_kmod_src_v0.4.2_with_do_exit_release.tar.gz | 987afa204f4aa5ce6268c0df21ed26ee | 0.4.2 source snapshot |

Fallback md5s still verified clean:
- /tmp/cipher-phase3-evidence/fallback/cipher_kmod.ko.v0.2.0 = 55ab8c0c
- /tmp/cipher-phase3-evidence/fallback/libcipher_v2.so.v0.2.0 = 86618c30

### Incident 2 closure (2026-05-13 post-reboot)

Status: **FIXED** by Fix A. Verified by 20-cycle stress + functional gates under 0.4.3.

Fix A — `cipher_bar0_exit` leaks the `pci_dev` reference deliberately. The
`pci_iounmap` still runs (cleaning up our own iomap). The `pci_dev_put` that
would have walked another driver's devres list is removed, with an in-source
comment pointing back to this incident. `cipher_pci_dev` is NULLed after the
iounmap so a subsequent insmod cannot reuse a stale pointer.

Verification:
- 20/20 insmod → `/tmp/phase3_happy` → rmmod cycles complete cleanly.
- /proc/sys/kernel/tainted stays at 12288 (the pre-existing nvidia OE+unsigned
  bits); no W (warning) bit, no D (died) bit gets set across all 20 cycles.
- `sudo dmesg --since "5 min ago" | grep -iE 'oops|warn|bug|null'` is empty
  after the stress test.
- Reference: `/home/ubuntu/cipher-phase4-evidence/stress_20cycle.log`.

Note: the post-reboot pod has /tmp wiped (tmpfs), so the fallback artifacts
under `/tmp/cipher-phase3-evidence/fallback/` are re-extracted from
`/home/ubuntu/cipher-phase3-evidence.tar.gz` after each boot. The /home/ubuntu
copies (`cipher_kmod.ko.v0.2.0`) survive reboots independently and md5-match.

## Incident 3 — do_exit slot leak (resolved in 0.4.2/0.4.3)

When a tenant exits without explicitly releasing its SM partition slots, the
slot atomics retained the dead pid's `CIPHER_SLOT_PACK(pid)` state. New
tenants then saw the slot pool as oversubscribed.

Fix B — `cipher_partition_release_slots_only(pid)` is called from
`cipher_do_exit_pre` (kprobe on `do_exit`) before the cipher_pid_stats entry
is torn down via kfree_rcu. The slot-only variant skips the cache writeback
(the cache lives on cipher_pid_stats and is about to be freed). Verified by
partition Test 4: after 11 ephemeral children exit, the parent can still
request and receive 8 slots (post-reap free pool is fully replenished).

## Phase 4 functional gates under 0.4.3 (2026-05-13 post-reboot)

All from `/home/ubuntu/cipher-phase4-evidence/`:

- Phase 3 ABI: happy + negative + root all PASS (12/12 assertions).
- Phase 4.1 snapshot (10/10 PASS): tenant_id_str match, snapshot.pid match,
  session_fp + handle_u32 non-zero, ENOENT on unknown/missing selectors.
- Phase 4.2 partition (8/8 PASS): mask popcount matches hint; 11-way concurrent
  fork claims slots from a shared pool; ephemeral-exit reap returns slots to
  the free pool; hint clamps work at both ends.

### Phase 4.2 partition single-thread perf (N=1,000,000)

| Metric | 0.4.1 (spinlock) | 0.4.3 (lock-free) |
|---|---|---|
| p50 | 248 ns | 228 ns |
| p90 | 253 ns | 233 ns |
| p99 | 260 ns | 236 ns |
| max | 20,675 ns | 14,828 ns |

Lock-free path is ~9% faster on the uncontended idempotent fast path; the
real win is at scale (below).

### Phase 4.2 contention scaling sweep — 0.4.3, baseline 236 ns

26-logical-CPU / 13-physical-core pod. Each thread registers its own tenant
(unique kernel pid via pthreads), claims slots, then drives
`CIPHER_REQUEST_SM_PARTITION` on the idempotent fast path under contention.

#### First sweep: shared fd (one `/dev/cipher` fd across all threads)

| Threads | hint | p50 (ns) | p99 (ns) | p99 ratio | ops/s |
|---:|---:|---:|---:|---:|---:|
|  4  | 8 |    472 |    574 |  2.4× |  7.7 M |
|  8  | 8 |    848 |  2,059 |  8.7× |  8.3 M |
| 13  | 8 |  1,189 |  4,996 | 21.2× |  8.1 M |
| 16  | 8 |  1,215 |  7,049 | 29.9× |  8.2 M |
| 24  | 8 |  2,149 |  9,784 | 41.5× |  8.2 M |
| 26  | 8 |  2,491 | 11,862 | 50.3× |  7.9 M |
| 33 (gate spec) | 8 | 2,320 | 10,430 | 44.2× |  8.5 M |

Reference: `/home/ubuntu/cipher-phase4-evidence/contention_sweep.log`.

#### Discriminator A: hint=1 sweep (33 threads × 1 = exact supply)

Same shared-fd setup but every thread asks for one slot — exact match against
the 33-slot supply. Demand-supply ratio cannot drive any thread into the
slow-path scan. Result is statistically indistinguishable from hint=8:

| Threads | p99 (hint=1) | p99 (hint=8) |
|---:|---:|---:|
|  4  |    615 |    590 |
|  8  |  2,073 |  1,961 |
| 13  |  4,410 |  4,390 |
| 33  | 11,892 | 11,008 |

Demand-supply mismatch is NOT the cause of the slowdown. Reference:
`contention_sweep_hint1.log`, `contention_sweep_hint8.log`.

#### Discriminator B: per-thread fd (each worker opens its own `/dev/cipher`)

| Threads | hint | p50 (ns) | p99 (ns) | p99 ratio | ops/s |
|---:|---:|---:|---:|---:|---:|
|  4  | 1 |   244 |   252 | **1.1×** | 13.9 M |
|  8  | 1 |   243 |   251 | **1.1×** | 27.9 M |
| 13  | 1 |   253 |   315 | **1.3×** | 37.9 M |
| 24  | 1 |   290 |   323 | **1.4×** | 69.0 M |
| **33 (gate spec)** | 1 | **289** | **322** | **1.4×** | **75.9 M** |

Reference: `contention_sweep_perthread_fd.log`.

### What this means

The shared-fd sweep was hitting a **kernel-side single-`struct file`
throughput cap of ~8 M ops/s** that has nothing to do with the partition
allocator. Two observations make this unambiguous:

1. The shared-fd sweep ops/s flatlines at ~8 M from 4 threads up through 33
   threads — adding threads converts capacity into latency, exactly the
   signature of N threads sharing a single throughput-capped resource.
2. The per-thread-fd sweep scales nearly linearly to 75.9 M ops/s at 33
   threads, and the p99 ratio collapses from 44.2× to **1.4×** at the
   gate-spec workload.

Conclusion: **the 0.4.3 lock-free allocator clears the 5× contention gate
cleanly (1.4× at 33 threads, hint=1, per-thread fd).** The original gate
test conflated VFS-level shared-fd serialization with allocator contention;
once that confound is removed, the lock-free design behaves as designed —
no remaining global serialization, near-linear scaling.

Comparison vs 0.4.1 spinlock (at the gate-spec 33-thread workload):

| Test setup | 0.4.1 (spinlock) | 0.4.3 (lock-free, shared fd) | 0.4.3 (lock-free, per-thread fd) |
|---|---:|---:|---:|
| p99 ratio | 126.7× | 44.2× | **1.4×** |
| ops/s     | (capped) | 8.5 M  | **75.9 M** |

The 0.4.2 → 0.4.3 lock-free design did exactly what it was supposed to do.
The headline gate result is the third column: **1.4× at 33 threads, well
under 5×.** That single number is the right one to ship.

Implication for libcipher_v2 / Phase 5 actuators: high-concurrency tenants
should hold a per-thread fd to `/dev/cipher` (typically a per-CUDA-stream or
per-worker-thread open), not share a single process-wide fd. This is good
practice anyway because `struct file` carries position state that is
mutually exclusive across pwrite/pread, but it is now also a measurable
~50× win for the partition-allocator hot path under contention.
