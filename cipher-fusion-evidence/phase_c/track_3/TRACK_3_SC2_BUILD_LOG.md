# Track 3 SC2 — BUILD LOG

**Date:** 2026-05-19. **Scope:** kmod-side migration state machine (nrs 16-20).

## Source changes (4 files, as adjudicated)

| file | change |
|---|---|
| `cipher_ioctl.h` | +5 ioctl NRs 16-20, `struct cipher_cp54_migrate_poll`, `CIPHER_CP54_MIG_*` / `CIPHER_CP54_MIGOUT_*` defines — additive, no relayout |
| `cipher_cp54_sched.c` | RSVD encoding (OCC b31 / RSVD b30 / pid b0-29); per-tenant migration state in `struct cipher_cp54_alloc`; `cp54_release_groups` two-cmpxchg reaper sweep; `cp54_reserve` / `cp54_release_rsvd` / `cp54_commit_migration` / `cp54_abort_migration` / `cp54_abort_timeouts` / `cp54_eval_migration`; 5 ioctl entry points; FREE hooks `cp54_eval_migration(false)` |
| `cipher_dev.c` | +5 `case` labels in the ioctl dispatch switch |
| `cipher_internal.h` | +5 entry-point prototypes |

**Untouched:** all other kmod `.c`/`.h`, libcipher_rt, libcipher_v2,
cipher_kv_bridge. The legacy nr-9 allocator (`cipher_partition_allocator.c`)
has its own separate `CIPHER_SLOT_OCCUPIED` encoding — verified it does not
reference the CP54 group encoding, so the RSVD-bit narrowing is contained to
`cipher_cp54_sched.c` (no 4-file scope expansion).

## ACK_MIGRATE — deviation from the SC2 plan, flagged

The SC2 plan's struct table listed `ACK_MIGRATE` as a payload-less `_IO`.
**It is built as `_IOW(C,19,__u32)`** — the tenant passes `1`=commit /
`0`=NACK. This is a *necessary consequence of the adjudicated item-5
addition*: the `CIPHER_CP54_MIGOUT_ABORTED_TENANT_NACK` outcome can only be
produced if the tenant has a way to report a failed migration. A payload-less
ACK could signal success only. The migration-machine invariant is preserved
in refined form: **the kmod never unilaterally aborts a MIGRATING migration;
the tenant may abort its own** (it has authoritative knowledge — verify-
before-swap, SC3 — that it did not swap its green context, so old_mask is
still live). NR 19 stays `ACK_MIGRATE`; the 5-ioctl ABI at NRs 16-20 is intact.

## Build

```
make -C /lib/modules/6.8.0-1046-nvidia/build M=/home/ubuntu/cipher_kmod modules
```

- **Result: clean.** No warning or error from `cipher_cp54_sched.o`,
  `cipher_dev.o`, or MODPOST. (The "compiler differs" + "BTF skipped"
  notices are pre-existing environment notices, identical to every prior
  CIPHER kmod build — not SC2-introduced.)
- Built `cipher_kmod.ko` — md5 **`98da2d1fd906a1b58c978a6996021539`**
  (SC2 anchor candidate `98da2d1f`), srcversion `00A9A344426D9EB9C521EDE`,
  vermagic `6.8.0-1046-nvidia`.
- `nm` confirms all 5 new entry points exported: `cipher_cp54_ioctl_`
  `subscribe_migrate` / `poll_migrate` / `start_migrate` / `ack_migrate` /
  `compact_migrate`.

## Load + isolation regression

`rmmod cipher_kmod` (old `8d777dfb`, refcount 0) → `insmod cipher_kmod.ko`
(SC2 `98da2d1f`). Loaded OK; `/dev/cipher` present, mode 0666; dmesg clean
(kprobe + reaper attach, no warning/error).

**Existing 6 isolation tests — `cp54_isolation_test.c` (15-group):**

```
Test 1 legacy nr-9 deactivated         PASS
Test 2 ALLOCATE / FREE / QUERY         PASS (4 assertions)
Test 3 pool resize                     PASS (4 assertions)
Test 4 do_exit reaper                  PASS  ← two-cmpxchg sweep, reaper intact
Test 5 disjointness                    PASS (3 assertions)
Test 6 concurrent stress               PASS (2 assertions)
=== Phase A result: 15 PASS, 0 FAIL ===
```

**No NR 1-15 behaviour change.** The RSVD-bit encoding is byte-identical to
the pre-SC2 `GRP_PACK` value for any real pid (< 2²²), and `cp54_eval_migration`
is a side-effect-free scan when there are zero `migratable` tenants — which is
the isolation-test (and W1/W2/W3) condition. dmesg clean after the run.

## Next

SC2 unit tests for the NR 16-20 state machine + 5-scenario latency capture
(task #4), then the W1/W2/W3 smoke test (task #5).
