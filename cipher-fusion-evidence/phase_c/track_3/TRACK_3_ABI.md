# Track 3 — CIPHER `/dev/cipher` ABI MANIFEST

**Purpose:** the authoritative record of the ioctl ABI extension Track 3 adds
to `cipher_kmod`. Created at SC2 close (2026-05-19, item-2 PUSH).

**Binding rule ([[cipher-abi-rule]]):** `/dev/cipher` ioctl nrs are a stable
ABI. Once shipped, **nrs are never renumbered, reordered, collapsed, or
repurposed.** New ioctls take fresh nrs only. A future maintainer must not
"tidy" the numbering — a deployed userspace binary encodes these nrs.

---

## Full nr map (magic `'C'`, `cipher_ioctl.h`)

| nr | name | dir | added | status |
|----|------|-----|-------|--------|
| 1 | `CIPHER_REGISTER_TENANT` | `_IOW` | Phase 2 | live |
| 2 | `CIPHER_SNAPSHOT` | `_IOR` | reserved | `-ENOSYS` |
| 3 | `CIPHER_RESET` | `_IOW` | reserved | `-ENOSYS` |
| 4 | `CIPHER_GET_VERSION` | `_IOR` | reserved | `-ENOSYS` |
| 5 | `CIPHER_SUBMIT_GPU_STATE` | `_IOW` | Phase 3 T3 | live |
| 6 | `CIPHER_SUBMIT_PROCESS_UTIL` | `_IOW` | Phase 3 T3 | live |
| 7 | `CIPHER_SUBMIT_LAUNCH_STATS` | `_IOW` | Phase 3 T3 | live |
| 8 | `CIPHER_GET_TENANT_SNAPSHOT` | `_IOWR` | Phase 4.1 | live |
| 9 | `CIPHER_REQUEST_SM_PARTITION` | `_IOWR` | Phase 4.2 | **deactivated** (`-ENOSYS`; superseded by nr 13) |
| 10 | `CIPHER_SET_CLOCK_MHZ` | `_IOW` | Phase 4.3 | live |
| 11 | `CIPHER_SUBMIT_FLOP_SAMPLE` | `_IOW` | CP 3.3 | live |
| 12 | `CIPHER_QUERY_FLOPS` | `_IOWR` | CP 3.3 | live |
| 13 | `CIPHER_CP54_ALLOCATE` | `_IOWR` | CP 5.4 | live |
| 14 | `CIPHER_CP54_FREE` | `_IO` | CP 5.4 | live |
| 15 | `CIPHER_CP54_QUERY` | `_IOR` | CP 5.4 | live |
| **16** | **`CIPHER_CP54_SUBSCRIBE_MIGRATE`** | `_IOW` (`__u32`) | **Track 3 SC2** | **live** |
| **17** | **`CIPHER_CP54_POLL_MIGRATE`** | `_IOR` (`struct cipher_cp54_migrate_poll`) | **Track 3 SC2** | **live** |
| **18** | **`CIPHER_CP54_START_MIGRATE`** | `_IO` | **Track 3 SC2** | **live** |
| **19** | **`CIPHER_CP54_ACK_MIGRATE`** | `_IOW` (`__u32`) | **Track 3 SC2** | **live** |
| **20** | **`CIPHER_CP54_COMPACT_MIGRATE`** | `_IO` | **Track 3 SC2** | **live** |

Next free nr: **21**.

---

## Track 3 SC2 extension — detail (nrs 16-20)

The kmod-side Dynamic SM Migration state machine. Additive: with zero
`migratable` tenants the ledger behaves byte-identically to nrs 13/14/15 alone
(verified — SC2 regression smoke W1/W2/W3 PASS, `TRACK_3_SC2_REGRESSION.md`).

### nr 16 — `CIPHER_CP54_SUBSCRIBE_MIGRATE` — `_IOW(__u32)`
Payload `__u32`: `1` = the caller is `migratable`, `0` = `pinned` (default).
A tenant that never calls this is never proposed a migration.

### nr 17 — `CIPHER_CP54_POLL_MIGRATE` — `_IOR(struct cipher_cp54_migrate_poll)`
```c
struct cipher_cp54_migrate_poll {
    __u32 migrate_state;   /* CIPHER_CP54_MIG_{IDLE,PROPOSED,MIGRATING} */
    __u32 target_mask;     /* proposed new group mask; 0 unless PROPOSED+ */
    __u32 cur_mask;        /* caller's current owned 8-SM-group mask */
    __u32 last_outcome;    /* CIPHER_CP54_MIGOUT_* — last migration's outcome */
    __u32 reserved[4];
};
```
Lock-free read (QUERY discipline). `last_outcome` (item-5 addition) survives
until the next migration so a tenant can inspect why its last migration ended.

### nr 18 — `CIPHER_CP54_START_MIGRATE` — `_IO`
`PROPOSED → MIGRATING`. The tenant's commit of intent. `-EINVAL` if the caller
is not in `PROPOSED`.

### nr 19 — `CIPHER_CP54_ACK_MIGRATE` — `_IOW(__u32)`
Payload `__u32`: `1` = COMMIT (migration succeeded), `0` = tenant-NACK → ABORT
(the tenant reports its `migrate()` did not complete; it keeps its old groups).
`-EINVAL` if the caller is not in `MIGRATING`.

> **ABI note — deviation from the SC2 plan.** The SC2 plan listed nr 19 as a
> payload-less `_IO`. It ships as `_IOW(__u32)` because the adjudicated item-5
> addition (`CIPHER_CP54_MIGOUT_ABORTED_TENANT_NACK` outcome) requires a tenant
> path to report a *failed* migration — a payload-less ACK can signal success
> only. NR 19 is still `ACK_MIGRATE`; the 5-ioctl assignment 16-20 is intact.

### nr 20 — `CIPHER_CP54_COMPACT_MIGRATE` — `_IO`
Forces a compaction-evaluation pass (operator control / deterministic test
trigger). Bypasses the conservative size/sustain gates; still honours opt-in,
IDLE state, and the rate limit.

### Enums (`cipher_ioctl.h`)
```
CIPHER_CP54_MIG_IDLE 0  PROPOSED 1  MIGRATING 2
CIPHER_CP54_MIGOUT_NONE 0  COMMITTED 1  ABORTED_TIMEOUT 2
                       ABORTED_KMOD_REFUSED 3  ABORTED_TENANT_NACK 4
```
`ABORTED_KMOD_REFUSED` (3) is defined but not yet produced by SC2 — reserved
for a future kmod-side refusal path; the value is claimed so it is not reused.

---

## Kmod-internal note — not ABI

`cipher_cp54_sched.c` narrows the per-group packed pid from 31 to 30 bits
(bit 30 becomes the `RSVD` flag). This is the **group-state atomic encoding**,
internal to the kmod — userspace sees `pid_t` unchanged through every ioctl.
It is recorded here only so a maintainer does not mistake it for an ABI change.
