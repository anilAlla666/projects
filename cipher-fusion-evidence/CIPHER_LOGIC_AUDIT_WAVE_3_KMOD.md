# CIPHER Logic Audit — Wave 3: Kernel Module (`cipher_kmod` 0.4.8)

**Scope.** Deep logic-level audit of the kernel module that anchors the
CIPHER substrate, `/home/ubuntu/cipher_kmod/`. The on-disk source is
version **0.4.8** (`cipher_main.c` L128 `MODULE_VERSION("0.4.8")`,
codified-devnode anchor `e2f50452` per [[cipher-devnode-codified]]); the
CP 5.4 Step 1.2 and Track 2 SC5 work later rotated the published anchor
to `285d102e` and `008b3c66` in the closeouts, but the source on disk
corresponds to 0.4.8 with the CP 5.4 + Track 2 + Track 3 ioctls live.
Companion to Wave 1 (`CIPHER_LOGIC_AUDIT_WAVE_1_CLASSIFIER.md` — the
classifier brain in `cipher-may13-evidence`) and Wave 2
(`CIPHER_LOGIC_AUDIT_WAVE_2_ACTUATORS.md` — the userspace substrate
`cipher_rt_phase4` and the earlier `libcipher_v2`).

**Method.** Each file is audited under nine sections:
PURPOSE / PUBLIC SURFACE / CONTROL FLOW / STATE / CONCURRENCY /
DEPENDENCIES / LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED / CONTRIBUTION
TO SYSTEM / FUSION POINTS. Citations are file-relative line numbers.
Wave 3 adds four per-file emphases on top of the nine sections:
the file's state machine (if any), the invariants it enforces and
how, the ABI surface it lands (which ioctl `nr`s and their failure
modes), and behaviour under `do_exit` (what happens when a tenant
crashes mid-operation).

**Fusion classes** carried over from Wave 2:
PORT-AS-IS = compile with no logic change.
SHIM-REQUIRED = add adapter from classifier output, no behavioural change.
REFACTOR-REQUIRED = substantive code change.
VESTIGIAL = retired; not in OBJS.

**Build composition (`Kbuild` L7-20).** The module is built from 14
`.o` units. Listed in Kbuild order:
`cipher_main`, `cipher_probe`, `cipher_ioctl_decode`, `cipher_proc`,
`cipher_dev`, `cipher_bar0`, `cipher_tenant_snapshot`,
`cipher_state_updater`, `cipher_partition_allocator`,
`cipher_cp54_sched`, `cipher_weight_arena`, `cipher_clock`,
`cipher_kvdedup`, `cipher_flops`. The 15th source file
`probe_microbench.c` is a userspace microbench, deliberately
NOT included in Kbuild — verified by inspection (it builds via
the gcc one-liner in its header L10).

---

## The complete /dev/cipher ioctl NR table

Built from `cipher_ioctl.h` and cross-checked against the switch in
`cipher_dev.c::cipher_dev_unlocked_ioctl` (L251-304). Magic byte
`'C'` (`CIPHER_IOCTL_MAGIC`, L19). Each row is verified to dispatch.

| nr  | name                              | dir   | payload                              | dispatch (cipher_dev.c)         | handler                                                  | status                                       |
|-----|-----------------------------------|-------|--------------------------------------|---------------------------------|----------------------------------------------------------|----------------------------------------------|
| 1   | `CIPHER_REGISTER_TENANT`          | _IOW  | `struct cipher_register_tenant`      | L255 → L58                      | `cipher_dev_register_tenant`                             | live (Phase 2)                               |
| 2   | `CIPHER_SNAPSHOT`                 | _IOR  | `struct cipher_snapshot` (4 KiB)     | L297-300                        | `-ENOSYS`                                                | reserved (Phase 6) — ABI rule [[cipher-abi-rule]] |
| 3   | `CIPHER_RESET`                    | _IOW  | `__u32`                              | L297-300                        | `-ENOSYS`                                                | reserved (Phase 6)                           |
| 4   | `CIPHER_GET_VERSION`              | _IOR  | `__u32`                              | L297-300                        | `-ENOSYS`                                                | reserved (Phase 6)                           |
| 5   | `CIPHER_SUBMIT_GPU_STATE`         | _IOW  | `struct cipher_gpu_state`            | L257 → L162                     | `cipher_dev_submit_gpu_state`                            | live, CAP_SYS_ADMIN (Phase 3 T3)             |
| 6   | `CIPHER_SUBMIT_PROCESS_UTIL`      | _IOW  | `struct cipher_process_util`         | L259 → L196                     | `cipher_dev_submit_process_util`                         | live, CAP_SYS_ADMIN (Phase 3 T3)             |
| 7   | `CIPHER_SUBMIT_LAUNCH_STATS`      | _IOW  | `struct cipher_launch_stats`         | L261 → L225                     | `cipher_dev_submit_launch_stats`                         | live, anti-spoof (Phase 3 T3)                |
| 8   | `CIPHER_GET_TENANT_SNAPSHOT`      | _IOWR | `struct cipher_tenant_snapshot_query`| L263 → L118                     | `cipher_dev_get_tenant_snapshot`                         | live, unprivileged read (Phase 4.1 T4.1.7)   |
| 9   | `CIPHER_REQUEST_SM_PARTITION`     | _IOWR | `struct cipher_partition_request`    | L265 → L107                     | `cipher_dev_request_sm_partition` → `-ENOSYS`            | DEACTIVATED by CP 5.4 (see below)            |
| 10  | `CIPHER_SET_CLOCK_MHZ`            | _IOW  | `__u32`                              | L267 → cipher_clock.c L130     | `cipher_dev_set_clock_mhz`                               | live, unprivileged + bounds (T4.3.2)         |
| 11  | `CIPHER_SUBMIT_FLOP_SAMPLE`       | _IOW  | `struct cipher_flop_sample`          | L269 → cipher_flops.c L89      | `cipher_flops_submit`                                    | live, CAP_SYS_ADMIN (CP 3.3)                 |
| 12  | `CIPHER_QUERY_FLOPS`              | _IOWR | `struct cipher_flop_query`           | L271 → cipher_flops.c L178     | `cipher_flops_query`                                     | live, unprivileged (CP 3.3)                  |
| 13  | `CIPHER_CP54_ALLOCATE`            | _IOWR | `struct cipher_cp54_allocate`        | L273 → cp54_sched.c L550       | `cipher_cp54_ioctl_allocate`                             | live (CP 5.4)                                |
| 14  | `CIPHER_CP54_FREE`                | _IO   | (none)                               | L275 → cp54_sched.c L646       | `cipher_cp54_ioctl_free`                                 | live (CP 5.4)                                |
| 15  | `CIPHER_CP54_QUERY`               | _IOR  | `struct cipher_cp54_query`           | L277 → cp54_sched.c L662       | `cipher_cp54_ioctl_query`                                | live (CP 5.4)                                |
| 16  | `CIPHER_CP54_SUBSCRIBE_MIGRATE`   | _IOW  | `__u32` (1=migratable, 0=pinned)     | L279 → cp54_sched.c L705       | `cipher_cp54_ioctl_subscribe_migrate`                    | live (Track 3 SC2)                           |
| 17  | `CIPHER_CP54_POLL_MIGRATE`        | _IOR  | `struct cipher_cp54_migrate_poll`    | L281 → cp54_sched.c L726       | `cipher_cp54_ioctl_poll_migrate`                         | live (Track 3 SC2)                           |
| 18  | `CIPHER_CP54_START_MIGRATE`       | _IO   | (none)                               | L283 → cp54_sched.c L754       | `cipher_cp54_ioctl_start_migrate`                        | live (Track 3 SC2)                           |
| 19  | `CIPHER_CP54_ACK_MIGRATE`         | _IOW  | `__u32` (1=COMMIT, 0=NACK)           | L285 → cp54_sched.c L773       | `cipher_cp54_ioctl_ack_migrate`                          | live (Track 3 SC2)                           |
| 20  | `CIPHER_CP54_COMPACT_MIGRATE`     | _IO   | (none)                               | L287 → cp54_sched.c L797       | `cipher_cp54_ioctl_compact_migrate`                      | live (Track 3 SC2)                           |
| 21  | `CIPHER_ARENA_REGISTER`           | _IOWR | `struct cipher_arena_register`       | L289 → cipher_weight_arena.c L157| `cipher_wa_ioctl_register`                              | live (Track 2 SC5)                           |
| 22  | `CIPHER_ARENA_IMPORT`             | _IOWR | `struct cipher_arena_import`         | L291 → cipher_weight_arena.c L220| `cipher_wa_ioctl_import`                                | live (Track 2 SC5)                           |
| 23  | `CIPHER_ARENA_LEAVE`              | _IOW  | `__u32` (arena_id)                   | L293 → cipher_weight_arena.c L281| `cipher_wa_ioctl_leave`                                 | live (Track 2 SC5)                           |
| 24  | `CIPHER_ARENA_QUERY`              | _IOR  | `struct cipher_arena_query`          | L295 → cipher_weight_arena.c L309| `cipher_wa_ioctl_query`                                 | live (Track 2 SC5)                           |
| —   | default                           | —     | —                                    | L301-303                        | `-ENOTTY`                                                | strict unknown-cmd contract                  |

24 nrs assigned, 21 live, 3 reserved (`nr 2/3/4`), 1 deactivated
(`nr 9`). NRs 2/3/4 and 9 still occupy their stable slot per the
[[cipher-abi-rule]] ("additive only; reserved nrs return `-ENOSYS`;
new ioctls take fresh nrs") — they are defined ABI; the kmod
returns `-ENOSYS` for them so caller libraries can do feature
probes without an `-ENOTTY` ambiguity.

## The complete /dev/cipher_kvdedup ioctl NR table

Separate char device, separate magic (`'K'`, `cipher_kvdedup.h`
L18). Dispatched in `cipher_kvdedup.c::kvd_unlocked_ioctl`
(L361-376).

| nr | name                       | dir   | payload                          | handler          | status                       |
|----|----------------------------|-------|----------------------------------|------------------|------------------------------|
| 1  | `CIPHER_KVDEDUP_INIT`      | _IOR  | `struct cipher_kvdedup_init`     | `kvd_ioctl_init` | live, returns tenant_id      |
| 2  | `CIPHER_KVDEDUP_PUT`       | _IOWR | `struct cipher_kvdedup_put`      | `kvd_ioctl_put`  | live, HIT/MISS + force-new   |
| 3  | `CIPHER_KVDEDUP_CONFIRM`   | _IOW  | `struct cipher_kvdedup_confirm`  | `kvd_ioctl_confirm` | live, post-memcmp refcount  |
| 4  | `CIPHER_KVDEDUP_FREE`      | _IOW  | `struct cipher_kvdedup_free`     | `kvd_ioctl_free` | live, anti-spoof per-tenant  |
| 5  | `CIPHER_KVDEDUP_STATS`     | _IOR  | `struct cipher_kvdedup_stats`    | `kvd_ioctl_stats` | live, telemetry             |
| —  | default                    | —     | —                                | `-ENOTTY`        | strict                       |

---

## Section /home/ubuntu/cipher_kmod/cipher_main.c

### PURPOSE
Module init/exit + cross-TU global storage. Wires the dependency
order between the 14 subsystems and provides the
`MODULE_LICENSE/AUTHOR/DESCRIPTION/VERSION` metadata
(L125-128, version `"0.4.8"`).

### PUBLIC SURFACE
- `cipher_init` (L25, `__init`).
- `cipher_exit` (L103, `__exit`).
- Cross-TU storage (L16-23):
  - `spinlock_t cipher_pid_insert_lock` — guards `cipher_pid_table` writes.
  - `atomic64_t cipher_global_cmd_counts[26]` and `_errors[26]` —
    decoder slot counters + OTHER + TOTAL.
  - `atomic64_t cipher_alloc_failures`, `cipher_reaped_count`.
  - `u64 cipher_load_jiffies`.
  - `unsigned long cipher_hooked_addr` — set by probe init.
  - `struct cipher_gpu_state_kern cipher_gpu_state` — Phase 3 spine.

### CONTROL FLOW
`cipher_init` runs 11 ordered steps:

1. `cipher_decode_self_check()` (L29) — table sanity, fails module
   load on drift.
2. `spin_lock_init` × 2, `cipher_load_jiffies = get_jiffies_64()` (L35-37).
3. `cipher_partition_allocator_init()` (L41) — legacy nr-9 allocator;
   cannot fail.
4. `cipher_cp54_sched_init()` (L42) — CP 5.4 ledger; cannot fail.
5. `cipher_wa_init()` (L43) — weight-arena registry; cannot fail.
6. `cipher_flops_init()` (L44) — FLOP ring; cannot fail.
7. `cipher_dev_init()` (L46) — `/dev/cipher` chardev; fatal on fail.
8. `cipher_kvdedup_init()` (L52) — `/dev/cipher_kvdedup`; fatal on
   fail, unwind dev.
9. `cipher_bar0_init()` (L58) — special: `-ENODEV` (no NVIDIA PCI)
   is acceptable, the module loads with the BAR0 layer disabled
   (L63-64). Any other error unwinds.
10. `cipher_proc_init()` (L66) — `/proc/cipher/{stats,bar0_state,
    gpu_state,flops,migrations,arenas}`; fatal.
11. `cipher_probe_init()` (L72) — registers kprobes on
    `nvidia_unlocked_ioctl` (pre + kretprobe) AND on `do_exit`. Fatal.
12. `cipher_state_updater_init()` (L81) — derived-state kthread;
    BEST-EFFORT, kthread failure is a `pr_warn` only, module still loads
    (L83-86).

Banner at L88-89 logs the hooked address.

`cipher_exit` (L103-120) runs the **reverse** order with one
explicit constraint at L110-112: `cipher_cp54_sched_exit()` must come
**after** `cipher_probe_exit()` so that the do_exit kprobe is
unregistered first — otherwise a process exiting mid-`rmmod` could
fire `cipher_cp54_release()` against torn-down state. This
ordering is a hard invariant of the module shutdown.

### STATE
The cross-TU globals in PUBLIC SURFACE. No file-local state beyond
the `__init/__exit` boundary.

### CONCURRENCY
Single-threaded by construction: `__init`/`__exit` run with the
module lock held by `init_module`/`delete_module`. The order in
which subsystems are inited determines which globals are valid at
each step — e.g., `cipher_hooked_addr` is only valid after step 11.

### DEPENDENCIES
INBOUND: kernel module loader.
OUTBOUND: every other .c in the module.
ioctls: none directly.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- **Stale init banner** (L39): `"loading (Phase 4.2 T4.2.1 — lock-free
  SM partition allocator)"`. The current allocator is CP 5.4 (since
  nr 9 was deactivated). The banner predates CP 5.4 and was not
  updated. Cosmetic.
- `MODULE_DESCRIPTION` at L127 mentions
  "GPU-state spine + BAR0 reads + CP3.3 FLOP telemetry"; it does NOT
  mention CP 5.4 / Track 2 / Track 3. Cosmetic.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. Without `cipher_main.c` no other TU's init runs.

### FUSION POINTS
- (1) Classifier consumption: the classifier brain port (Wave 1)
  would insert ahead of `cipher_state_updater_init` to register
  itself as a derived-field producer at the 1 kHz cadence.
- (2) Fallback today: no classifier — `cipher_state_updater`
  computes thermal/power/voltage from `cipher_gpu_state` only.
- (3) Insertion point: a new `cipher_classify_init()` between L81
  and L86, after the kthread is up. Fusion class: REFACTOR
  (small) — one line.

### State machine: n/a (linear init/exit).

### Invariants
- Init runs to completion or returns `rc < 0` (the loader unloads).
- Exit always runs to completion; every `proc_remove`/`device_destroy`
  is NULL-guarded inside its TU.
- `cipher_load_jiffies` is set exactly once, at init, never updated.

### ABI surface: none (this TU exports no ioctls; it composes them).

### Concurrency under do_exit
`cipher_main.c` itself is not on the do_exit path; the
`cipher_do_exit_pre` handler lives in `cipher_probe.c`. The
ordering invariant in `cipher_exit` (L110-112) is the only
do_exit-related contract here.

---

## Section /home/ubuntu/cipher_kmod/cipher_internal.h

### PURPOSE
Cross-TU types, constants, and extern declarations. The kernel-
internal mirror of the userspace ABI in `cipher_ioctl.h`. Defines
the slot model (Option Z, L6-8), the per-PID stats record, the
internal tenant-snapshot struct, and every cross-TU function
prototype.

### PUBLIC SURFACE (header-only)
- **Decoder slot enum** (L41-66): 24 slot indices,
  `CIPHER_SLOT_RM_ALLOC_MEMORY=0` through
  `CIPHER_SLOT_WAIT_OPEN_COMPLETE=23`.
- `CIPHER_NV_IOCTL_COUNT=24`, `OTHER_SLOT=24`, `TOTAL_SLOT=25`,
  `CIPHER_GLOBAL_SLOTS=26` (L32-35).
- **Per-PID hashtable sizing**: `CIPHER_PID_HASH_BITS=10` →
  1024 buckets (L73-74).
- `struct cipher_gpu_state_kern` (L89-99): the Phase 3 GPU-state
  spine — single global, latest-wins, spinlock-protected.
- `struct cipher_pid_stats` (L103-165): 192 B atomic counters
  + Phase 3 telemetry + Phase 4 derived fields. Heavy struct (~600 B).
- `struct cipher_tenant_snapshot` (L176-232): assembled view
  used by Phase 4.1 lookups; layout-identical to the userspace
  `cipher_tenant_snapshot_user` in `cipher_ioctl.h` so the ioctl
  nr 8 handler can do a direct `memcpy` (verified at
  `cipher_dev.c` L146).
- All function prototypes for `_init/_exit`, `_ioctl_*`,
  `_proc_show` across the 14 TUs (L244-360).

### STATE
None — pure header.

### CONCURRENCY
None — pure header.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- L7 "Phase 1.5.1: NV_ESC_RM_* family added to the decoder" —
  historically correct; the decode table is now stable.
- L235 `extern struct hlist_head cipher_pid_table[CIPHER_PID_HASH_SIZE]`
  — the actual table lives in `cipher_probe.c` (L28 `DEFINE_HASHTABLE`),
  the storage for the locks lives in `cipher_main.c` (L16-23). The
  header is the single source of truth for both.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. Every other TU includes this header.

### FUSION POINTS
The cipher_pid_stats record (L103-165) is the substrate's
per-tenant kernel-side state. A classifier port would extend it
with classifier-decision fields; per the ABI rule those go after
the existing fields with a reserved tail (the userspace mirror
already has `__u32 reserved[16]` at the end of
`cipher_tenant_snapshot_user`, `cipher_ioctl.h` L178). Fusion class:
REFACTOR (small) — additive struct fields.

### State machine, invariants, ABI: definitional only.
### Concurrency under do_exit: not applicable.

---

## Section /home/ubuntu/cipher_kmod/cipher_ioctl.h

### PURPOSE
Public ABI surface — every userspace caller (`libcipher_rt`,
`libcipher_v2`, `cipher-gpustate`, `cipher_flopd`,
`cipher_rt_phase4`'s green-ctx and weight-arena clients,
`cipher_kv_bridge`, vLLM offload, the Phase 4.1 prometheus exporter)
includes this header. The contract is ABI-stable per the comment
block L4-7 and the [[cipher-abi-rule]].

### PUBLIC SURFACE (the ioctl NR table above documents every entry)
- `CIPHER_IOCTL_MAGIC='C'` (L19), `CIPHER_TENANT_ID_LEN=64` (L21).
- 24 ioctl macros (`_IO/_IOW/_IOR/_IOWR`), spread across
  five generations: Phase 2 (nr 1), Phase 6 reserved (2/3/4),
  Phase 3 T3 (5/6/7), Phase 4.1 T4.1.7 (8), Phase 4.2 T4.2.1 (9),
  Phase 4.3 T4.3.2 (10), CP 3.3 (11/12), CP 5.4 (13/14/15),
  Track 3 SC2 (16-20), Track 2 SC5 (21-24).
- 10 structs: `cipher_register_tenant`, `cipher_snapshot`,
  `cipher_gpu_state`, `cipher_process_util`, `cipher_launch_stats`,
  `cipher_tenant_snapshot_user`, `cipher_tenant_snapshot_query`,
  `cipher_partition_request`, `cipher_flop_sample`,
  `cipher_flop_tenant`, `cipher_flop_query`, `cipher_cp54_allocate`,
  `cipher_cp54_query`, `cipher_cp54_migrate_poll`,
  `cipher_arena_register`, `cipher_arena_import`,
  `cipher_arena_query`.
- Flags and constants:
  - `CIPHER_PARTITION_FLAG_FIT_HINT = 1<<0` (L228).
  - `CIPHER_PARTITION_FLAGS_ALL` (L229).
  - `CIPHER_CP54_QOS_PARTITION=0 / SHARED=1 / POOL=2` (L354-356).
  - `CIPHER_CP54_MIG_IDLE=0 / PROPOSED=1 / MIGRATING=2` (L421-423).
  - `CIPHER_CP54_MIGOUT_NONE=0 / COMMITTED=1 / ABORTED_TIMEOUT=2 /
     ABORTED_KMOD_REFUSED=3 / ABORTED_TENANT_NACK=4` (L430-434).
  - `CIPHER_FLOP_STATUS_LIVE=0 / STALE=1 / NO_SOURCE=2` (L312-314).
  - `CIPHER_FLOP_QUERY_MAX_TENANTS=64` (L316).
  - `CIPHER_WA_BLOB_MAX=65536`, `CIPHER_WA_MAX_ARENAS=16` (L468-469).

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The comment block at L9-12 enumerates the live nrs vs reserved;
  the dispatcher in `cipher_dev.c` (L297-300) honours this exactly.
- L390-393 is explicit about the [[cipher-abi-rule]]: "NRs 16-20 are
  PERMANENTLY assigned to these five ioctls; future maintainers
  must not collapse the numbering or repurpose them". Same for
  21-24 at L457-459. The discipline is enforced on the header
  side, not the dispatcher side.

### FUSION POINTS
- (1) Classifier outputs: extending the snapshot user-struct
  (L123-179) with classifier-decision fields fits the reserved
  tail (`__u32 reserved[16]` at L178), no new ioctl required.
- (2) New classifier action ioctls: would take new nrs starting
  at 25 — the ABI rule forbids re-using 2/3/4 even though they
  are reserved.

### State machine
The header documents one state machine in the comments: the
Track 3 SC2 migration FSM at L401-417 — `IDLE → PROPOSED → MIGRATING
→ {COMMIT, ABORT} → IDLE`. The implementation in
`cipher_cp54_sched.c` matches; see that file's section.

### Invariants
- Stable ABI: layout/`nr` is not changed once shipped.
- Tail `__u32 reserved[N]` on every struct (L67-68, L79, L91, L100,
  L178, L297, L308, L332, L365, L374, L441, L478, L489, L494)
  so additions land inside the same `sizeof`.
- Magic byte `'C'` for `/dev/cipher`; `'K'` for `/dev/cipher_kvdedup`
  (in `cipher_kvdedup.h`).

### Concurrency under do_exit: definitional.

---

## Section /home/ubuntu/cipher_kmod/cipher_kvdedup.h

### PURPOSE
Public ABI surface for the second char device, `/dev/cipher_kvdedup`,
which the cipher_rt KV layer uses for cross-tenant KV-page dedup
(T4.6.4). Separate magic (`'K'`, L18) so it does not collide with
`/dev/cipher` magic `'C'`.

### PUBLIC SURFACE
- `CIPHER_KVDEDUP_MAGIC='K'` (L18), `CIPHER_KVDEDUP_DEV_NAME="cipher_kvdedup"` (L19).
- 5 ioctl macros (L70-79): INIT, PUT, CONFIRM, FREE, STATS.
- 5 structs (L30-68) — sizes deliberately small (`cipher_kvdedup_init`
  is 8 B, `put` is 40 B).
- Caps: `CIPHER_KVDEDUP_MAX_PER_TENANT=8192` (L82, design memo C),
  `CIPHER_KVDEDUP_MAX_ENTRIES=2^20` (L84, design memo G).
- Result/flag constants:
  - `CIPHER_KVDEDUP_FLAG_FORCE_NEW=0x1`
  - `CIPHER_KVDEDUP_RESULT_MISS=0`, `_HIT=1`.

### CONTROL FLOW (documented two-phase contract)
PUT is a HIT-CANDIDATE handshake: on hash hit the kmod returns a
fresh fd to the existing entry (`candidate_fd`), userspace
memcmp-verifies (the hash is xxhash64 — collisions are rare but
possible), and CONFIRMs to bump the refcount. On hash miss the
kmod registers the producer's exported fd as the canonical entry.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The header explicitly says (L10-12) "Userspace computes the
  xxhash64 of the 2 MiB page and the memcmp-verify on a hit; the
  kmod owns the table, the refcounts, and the cuIpc POSIX-FD handles."
  This is consistent with the implementation; the kmod does
  not interpret page contents.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. The userspace `cipher_rt_kv_alloc.c` (Wave 2)
opens this device at injection-init.

---

## Section /home/ubuntu/cipher_kmod/cipher_ioctl_decode.c

### PURPOSE
Translates an `nvidia_unlocked_ioctl` `cmd` argument (which encodes
an NV_ESC frontend or NV_ESC_RM nr) to a kmod-internal slot index
0..23. Sorted-table + binary search. Self-checked at init.

### PUBLIC SURFACE
- `const struct cipher_nv_ioctl_def cipher_nv_ioctls[24]` (L27-80).
- `int cipher_decode_nv_ioctl_slot(unsigned int cmd)` — L89.
  Returns slot in 0..23 or -1.
- `int cipher_decode_self_check(void)` — L119. Returns 0 or -EINVAL.

### CONTROL FLOW
- L89-112: binary search on sorted ascending `nr`. `_IOC_TYPE` check
  filters non-NVIDIA-magic commands at L94 before the search runs.
- L119-138: linear walk verifying every slot has a name and that
  `nr` is strictly ascending. Called from `cipher_init` first
  (`cipher_main.c` L29) — module load aborts if the table drifts.

### STATE
The `cipher_nv_ioctls` table is `const` (L27); it is `static_assert`'d
to be exactly 24 entries at L82-83.

### CONCURRENCY
Read-only after init; lock-free.

### DEPENDENCIES
Called from `cipher_probe.c::cipher_kprobe_pre` (L88) and
`::cipher_kretprobe_handler` (L142) on the kernel ioctl hot path,
so the binary-search cost (~5 compares for 24 entries) is what every
observed NVIDIA ioctl pays.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The header `cipher_internal.h` L23-24 documents two families;
  the table at L27-80 splits the 24 entries between RM (`nr 0x27..0x5e`)
  and FRONTEND (`nr 200..218`). The split is logical not numerical
  (FRONTEND `nr` is decimal, RM `nr` is hex), but they don't overlap
  so the single sorted table works.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for telemetry — without this, ioctls would only land
in the OTHER bucket. Not load-bearing for kernel correctness; the
worst-case failure mode is mis-attribution.

### FUSION POINTS
- (1) Classifier consumption: the slot index could feed a per-slot
  classifier feature vector (which ioctls a tenant issues is a
  strong workload signal).
- (2) Insertion point: add a `cipher_classify_observe_ioctl(slot)`
  call at the top of `cipher_probe.c::cipher_kprobe_pre`. PORT-AS-IS.

### State machine: n/a.
### Invariants
- Table sorted ascending by `nr` (enforced at init L129-134).
- Slot enum positions match table designated-initialiser positions
  (enforced implicitly by the designated initialisers in
  `cipher_nv_ioctls[]`).
- `static_assert(ARRAY_SIZE == 24)` (L82) — catches table-add-without-
  enum-add drift at compile time.

### ABI: none directly; this is internal decoding.
### Concurrency under do_exit: n/a — pure functions.

---

## Section /home/ubuntu/cipher_kmod/cipher_main.c (continued)

Already covered above as part of the module-level audit. Module
load/unload ordering is the load-bearing invariant.

---

## Section /home/ubuntu/cipher_kmod/cipher_dev.c

### PURPOSE
`/dev/cipher` chardev — the public ABI front door. Implements
`open/release/unlocked_ioctl`, the cdev/device/class plumbing,
the `devnode` callback that codifies mode 0666 (kmod 0.4.8,
[[cipher-devnode-codified]]), and 7 ioctl handlers in-file: 4 Phase
2/3 telemetry handlers, the deactivated nr 9, the Phase 4.1 cross-
process snapshot query (nr 8), and the legacy register-tenant
(nr 1). The other 17 handlers (nrs 10/11/12, 13-20, 21-24) live in
their respective TUs but are dispatched here.

### PUBLIC SURFACE
- `cipher_dev_init` (L332), `cipher_dev_exit` (L381).
- `static const struct file_operations cipher_dev_fops` (L306-311):
  `.open = cipher_dev_open`, `.release = cipher_dev_release`,
  `.unlocked_ioctl = cipher_dev_unlocked_ioctl`. No `mmap`, no
  `read/write` — this device is ioctl-only.
- `cipher_devnode` (L325-330): emits mode 0666.
- Static handlers: `cipher_dev_register_tenant` (L58),
  `cipher_dev_request_sm_partition` (L107 — deactivated),
  `cipher_dev_get_tenant_snapshot` (L118),
  `cipher_dev_submit_gpu_state` (L162),
  `cipher_dev_submit_process_util` (L196),
  `cipher_dev_submit_launch_stats` (L225).
- `cipher_fnv64` (L48-56): FNV-1a 64-bit string hash used to derive
  `tenant_session_fp` from `tenant_id`.

### CONTROL FLOW
**open/release** (L37-45) are pass-throughs — no per-fd state
(unlike `/dev/cipher_kvdedup`).

**unlocked_ioctl** (L251-304) is a 21-case `switch` against the
`cmd` constant. The order in the switch matters only for code
readability (the kernel compiles this to a jump table on x86-64);
the dispatcher does no validation beyond the matching itself.

**REGISTER_TENANT (nr 1)** L58-96:
1. `copy_from_user(&payload)` (L66).
2. Anti-spoof: `payload.pid == current->pid && payload.tgid == current->tgid` (L70). `-EPERM` on mismatch.
3. NUL-terminate `tenant_id` (L74).
4. `cipher_pid_get_or_create` (L76) — looks up or inserts the entry. `-ENOMEM` on alloc failure.
5. Under `cipher_pid_insert_lock`: `strscpy` tenant_id (L85-87).
6. FNV-1a → `tenant_session_fp` and `(u32)fp` → `tenant_handle_u32` via WRITE_ONCE (L91-93).

**REQUEST_SM_PARTITION (nr 9)** L107-113: returns `-ENOSYS` after a
`pr_warn_once`. The handler is intentionally a no-op — both the
legacy 4-SM-slot allocator (`cipher_partition_allocator.c`) and
the CP 5.4 8-SM-group ledger (`cipher_cp54_sched.c`) address the
same 132 physical SMs; running both would double-grant. The ABI
nr is preserved (per [[cipher-abi-rule]]) but no longer grants.

**GET_TENANT_SNAPSHOT (nr 8)** L118-160:
1. `kmalloc(GFP_KERNEL)` for the query struct (~1 KiB, too big for stack).
2. `copy_from_user`.
3. Selector precedence: non-empty `target_tenant_id` wins over `target_pid` (L137-141). Both empty → NULL (not found).
4. Lookup under `rcu_read_lock`.
5. `memcpy(&q->snapshot, snap, sizeof(...))` — relies on layout-identity between the kernel struct and userspace struct, see `cipher_internal.h` L176 vs `cipher_ioctl.h` L123.
6. `copy_to_user`, `kfree`.

**SUBMIT_GPU_STATE (nr 5)** L162-194 — CAP_SYS_ADMIN-gated
(L166-167). Validates `timestamp_ns != 0`, `sm/mem_util_pct ≤ 100`,
`temp_c ≤ 200` (L172-180). Writes under `cipher_gpu_state.lock`
spinlock (L182-191), `latest-wins` semantics.

**SUBMIT_PROCESS_UTIL (nr 6)** L196-223 — CAP_SYS_ADMIN-gated.
Validates `sm/mem/enc/dec_util_pct ≤ 100`. Writes per-PID via
`cipher_pid_get_or_create` then `WRITE_ONCE` on individual
fields. No spoof check — the daemon names the target pid.

**SUBMIT_LAUNCH_STATS (nr 7)** L225-249 — NOT CAP_SYS_ADMIN-gated.
Anti-spoof: caller pid/tgid must match payload (L237-238).
Workloads' CUPTI counters end up here.

### STATE
File-static: `cipher_dev_major`, `cipher_cdev`, `cipher_class`,
`cipher_device` (L32-35). All set by `cipher_dev_init`, torn down
by `cipher_dev_exit`.

### CONCURRENCY
- `cipher_pid_insert_lock` serialises tenant-id writes against
  hashtable insert (L85-87). Comment at L80-84 notes that proc
  readers do a lockless memcpy and accept torn 64-byte reads —
  diagnostic-only, benign.
- `cipher_gpu_state.lock` serialises the 8-field GPU state
  multi-field write (L182-191) — comment at `cipher_internal.h`
  L86-88 explains: the 8-field memcpy is not naturally atomic, the
  lock is the only way to prevent torn cross-field reads.
- All other writes use `WRITE_ONCE` on single fields.

### DEPENDENCIES
INBOUND: every userspace caller of `/dev/cipher`.
OUTBOUND: handlers in `cipher_clock.c`, `cipher_flops.c`,
`cipher_cp54_sched.c`, `cipher_weight_arena.c`,
`cipher_tenant_snapshot.c`, `cipher_probe.c`.
The dispatch table is the canonical "what nr does what" map.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The deactivation of nr 9 (L100-113) is documented in the
  comment block; the handler returns `-ENOSYS` cleanly.
- `cipher_devnode` (L325-330) emits mode 0666 — this matches
  the trust-model comment block at `cipher_ioctl.h` (the
  "Trust model:" header in the SET_CLOCK_MHZ section L243-263)
  and `cipher_clock.c` L13-30: non-root ioctls are permitted
  because the device-node permission is the access gate, not
  per-ioctl `capable()` calls.
- The `devnode` callback was added in kmod 0.4.8 specifically
  to codify mode 0666 at every reload — replacing the
  post-insmod chmod that was used in earlier versions; see
  `cipher-devnode-codified`.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. The chardev is the only ioctl entrypoint.

### FUSION POINTS
- (1) Classifier brain port (Wave 1) would add a new nr 25 for a
  `CIPHER_CLASSIFY_*` family. The dispatcher (L251-304) gains one
  more case.
- (2) Per-handler insertion is unnecessary; the classifier sees
  the substrate via the existing snapshot ioctl (nr 8).
- Fusion class for the dispatcher: PORT-AS-IS + SHIM (one new
  case). For the static handlers: PORT-AS-IS.

### State machine
Per-fd: stateless (open/release are pass-throughs). The chardev
does not track per-fd state — the per-tenant hashtable in
`cipher_probe.c` is the only state.

### Invariants
- Magic byte `'C'` (the dispatch trusts `_IO/_IOR/_IOW/_IOWR` to
  carry it; the chardev only sees encoded `cmd` values from its
  own `unlocked_ioctl`).
- Anti-spoof: REGISTER_TENANT (L70) and SUBMIT_LAUNCH_STATS (L237)
  require caller PID/TGID match payload; SUBMIT_PROCESS_UTIL
  trusts CAP_SYS_ADMIN.
- Unknown `cmd` returns `-ENOTTY` (L302), reserved/known-disabled
  returns `-ENOSYS` (L297-300, L112). This distinction matters for
  feature-probing libraries.

### ABI surface
The full nr table from the top of the document. All 24 nrs land
in the L251-304 dispatcher.

### Concurrency under do_exit
The chardev does NOT register a per-fd `release` that does
substantial work (L42-45 is a no-op). The do_exit kprobe in
`cipher_probe.c` is what reclaims per-tenant state. This is the
cleanly-decoupled design: the chardev is `open`/`release`-light,
the per-PID hashtable is the substrate of truth, and the reaper
runs from `do_exit` regardless of whether the tenant ever opened
`/dev/cipher`.

---

## Section /home/ubuntu/cipher_kmod/cipher_probe.c

### PURPOSE
The Phase-1 observation surface. Registers a `kprobe` + `kretprobe`
on `nvidia_unlocked_ioctl` (to count ioctls + errors per slot per
tenant) and a `kprobe` on `do_exit` (to reap dead tenants from
the hashtable). Also owns the 1024-bucket RCU hashtable
`cipher_pid_table` and the `cipher_pid_get_or_create` helper used
by `cipher_dev.c`.

### PUBLIC SURFACE
- `int cipher_probe_init(void)` — L228.
- `void cipher_probe_exit(void)` — L259.
- `struct cipher_pid_stats *cipher_pid_get_or_create(pid_t, pid_t)` — L41.

### CONTROL FLOW
**kprobe pre-handler** (`cipher_kprobe_pre`, L78-111):
- Reads `cmd` from `regs->si` (x86_64 2nd arg, L85).
- `cipher_decode_nv_ioctl_slot(cmd)` → slot 0..23 or -1.
- Bumps global `cmd_counts[slot]` (or OTHER) and TOTAL (L91-95).
- `cipher_pid_get_or_create` (L97) — RCU lookup + GFP_ATOMIC
  insert on miss; `cipher_alloc_failures` counted on `kmalloc`
  fail (L52-54).
- On entry hit: refresh `comm`, bump per-slot atomic, bump total,
  stamp `last_seen_jiffies` (L99-107).
  **Modification A** (comment L99): `get_task_comm` is called on
  every observation, not just on insert — a ~16 B memcpy plus a
  brief `task->alloc_lock` per ioctl.

**kretprobe** (`cipher_kretprobe_entry` + `_handler`, L121-155):
- Entry stashes `(cmd, pid)` in `ri->data` (L121-128) because the
  trampoline may clobber `regs->si` by the time the return handler
  fires.
- Return-handler short-circuits on `ret >= 0` (L139-140) — success
  has nothing to do.
- On error, bumps the GLOBAL `_errors[slot]` + total errors and
  the per-PID `errors` (L142-153).

**do_exit reaper** (`cipher_do_exit_pre`, L184-221):
- **Step 1, unconditional:** `cipher_cp54_release(pid)` (L198).
  Comment at L189-197 is explicit about why: the CP 5.4 ledger is
  an independent subsystem — a process can hold CP 5.4 groups
  without an `cipher_pid_stats` entry (it need not be an observed
  NVIDIA consumer; it just needs to have called `ioctl(nr=13)`).
  The CP 5.4 release is lock-free (atomic_cmpxchg + WRITE_ONCE),
  so it is atomic-context safe.
- **Step 2, conditional:** RCU lookup; bail if miss (the common
  case for non-NVIDIA processes, L200-204).
- **Step 3, on hit:** `cipher_partition_release_slots_only(pid)`
  (L210) — releases legacy nr-9 4-SM-slot allocations.
- **Step 4:** under `cipher_pid_insert_lock`, re-check, `hash_del_rcu`,
  `kfree_rcu`, increment `cipher_reaped_count`.

### STATE
- `DEFINE_HASHTABLE(cipher_pid_table, 10)` — 1024 buckets (L28).
- 3 `struct kprobe`/`kretprobe` (L157-168, L223-226).
- The kretprobe carries `struct cipher_ret_data` (L116-119)
  through its `data_size`.

### CONCURRENCY
- The hashtable is RCU-walked by readers (`cipher_proc.c`,
  `cipher_state_updater.c`, `cipher_tenant_snapshot.c`,
  `cipher_flops.c`).
- Writers (insert: kprobe + REGISTER_TENANT; delete: do_exit reaper)
  serialise on `cipher_pid_insert_lock` (L65-74, L212-219).
- The hot-path kprobe pre-handler does an RCU lookup first and only
  falls back to the lock + `GFP_ATOMIC` `kmalloc` on miss (L45-49 in
  `get_or_create`). Double-check under lock (L67-72) handles the
  race window between lock-free lookup and lock acquisition.
- The `do_exit` kprobe runs in atomic context (kprobes do); this is
  why `cipher_cp54_release` and `cipher_partition_release_slots_only`
  must be lock-free.

### DEPENDENCIES
- INBOUND: `cipher_main.c::cipher_init` calls `_init`; `cipher_exit`
  calls `_exit`. `cipher_dev.c::cipher_dev_register_tenant` uses
  `cipher_pid_get_or_create`.
- OUTBOUND: `cipher_decode_nv_ioctl_slot` (`cipher_ioctl_decode.c`);
  `cipher_cp54_release` (`cipher_cp54_sched.c`);
  `cipher_partition_release_slots_only` (`cipher_partition_allocator.c`).
- Kernel: kprobes, kretprobes, do_exit symbol, RCU, hlist hashtable.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The reaper's two-step lock pattern (L200-204 fast path, then
  L212-219 under lock) is documented in the block comment L170-183.
  Comment + code match exactly.
- The unconditional `cipher_cp54_release` (L198) breaks the "rcu
  lookup miss → return immediately" fast path the comment block
  describes for the LEGACY reaper. The comment is correct
  historically; CP 5.4 added a leg in front of it that does not
  depend on `cipher_pid_stats`. Code at L189-198 documents the
  drift in-line; this is honest in-code documentation, not stale.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. Without `cipher_probe.c`:
- no per-tenant ioctl telemetry,
- no `cipher_pid_table` (other TUs would crash on `hash_for_each_rcu`),
- no `cipher_cp54_release` on tenant death → CP 5.4 group leaks,
- no `cipher_partition_release_slots_only` → legacy nr-9 slot leaks
  (academic now that nr 9 is `-ENOSYS`, but the storage still
  exists in the slot array).

### FUSION POINTS
- (1) Classifier observer: the kprobe pre-handler (L78-111) is the
  best place to feed the classifier with `(tenant, slot)` events.
  PORT-AS-IS + SHIM (one line at the bottom of the hot path).
- (2) Reaper extension: a classifier port that holds per-tenant
  GPU state must add its release-on-exit hook into `cipher_do_exit_pre`
  alongside the CP 5.4 release. PORT-AS-IS + SHIM.

### State machine
Per-tenant lifecycle: NEW (kmalloc + insert) → ALIVE (hot path increments)
→ DEAD (do_exit reaper deletes). One transition per direction; no
intermediate states.

### Invariants
- `cipher_pid_table` is RCU-safe: every reader uses
  `hash_for_each_*_rcu`, every writer holds `cipher_pid_insert_lock`
  for the insert/delete and uses `hash_add_rcu`/`hash_del_rcu`/
  `kfree_rcu`.
- Lookup miss in the slow path is re-checked under lock (L67-72)
  to prevent duplicate inserts in the get-or-create.
- The kprobe pre-handler succeeds *even if* the per-PID alloc fails:
  the global counters are bumped first (L91-95). This means global
  totals are always accurate; per-PID totals may under-count under
  GFP_ATOMIC pressure.
- The kretprobe handler short-circuits on `ret >= 0` (L139): success
  is silent, only errors increment per-slot error counters.

### ABI surface
None directly. This TU implements the substrate that the
`cipher_dev.c` and `cipher_proc.c` ABIs are built on.

### Concurrency under do_exit
This file *is* the do_exit handler. The two-step (CP 5.4 release,
then conditional `cipher_pid_stats` cleanup) at L184-221 is the
canonical pattern for any future per-tenant kmod resource.

---

## Section /home/ubuntu/cipher_kmod/cipher_proc.c

### PURPOSE
Six `/proc/cipher/*` seq_file emitters: `stats` (kprobe counters,
top-16 PIDs and TGIDs), `bar0_state` (delegates to
`cipher_bar0.c`), `gpu_state` (Phase 3 spine), `flops` (delegates
to `cipher_flops.c`), `migrations` (delegates to
`cipher_cp54_sched.c`), `arenas` (delegates to
`cipher_weight_arena.c`).

### PUBLIC SURFACE
- `cipher_proc_init` (L466), `cipher_proc_exit` (L558).
- `cipher_proc_show` (L141) — emits `/proc/cipher/stats`.
- `cipher_gpu_state_proc_show` (L380) — emits `/proc/cipher/gpu_state`.
- Four `proc_ops` structs for the four entries that this file
  owns; two delegated.

### CONTROL FLOW
**`cipher_proc_show`** (L141-354):
1. Header (kmod version, uptime, hooked addr, totals — L150-166).
2. `cipher_emit_family` × 2 (frontend, RM; L168-174). Each call
   walks `cipher_nv_ioctls[]` filtering by family and skips
   zero-count rows.
3. OTHER bucket (L176-181).
4. RCU walk #1: count PIDs (L184-186).
5. Allocate `rows = kmalloc_array(npids ?: 1, ...)` GFP_KERNEL.
6. RCU walk #2: copy out atomic counters per PID (L197-219).
7. `sort` by total descending; emit top-16 (L222-256).
8. Inner block: per-TGID rollup using a single-pass scan of
   the PID rows (L260-350). Tenant resolution rule: PID==TGID
   wins; worker threads only fill if main thread had no tenant
   (L304-320).
9. Free, return.

**`cipher_gpu_state_proc_show`** (L380-410):
- `spin_lock` on `cipher_gpu_state.lock` for a clean snapshot copy
  (L386-388).
- Bail "no sample yet" if `timestamp_jiffies == 0` (L390).
- Emit power_mW, temp_C, sm/mem_clock, sm/mem_util, fb_used, age.

### STATE
File-static `proc_dir_entry *` pointers (L28-34): the parent
directory and six entries. All NULL-guarded in init/exit unwind.

### CONCURRENCY
- Read-only proc file; readers don't mutate kmod state.
- RCU read sections cover the hashtable walks.
- The intermediate `rows`/`trows` arrays are stack-free of RCU
  liveness: data is copied OUT under RCU then sorted/emitted
  AFTER `rcu_read_unlock`.

### DEPENDENCIES
- Reads `cipher_nv_ioctls`, `cipher_global_cmd_counts/errors`,
  `cipher_alloc_failures`, `cipher_reaped_count`, `cipher_load_jiffies`,
  `cipher_pid_table`, `cipher_gpu_state`.
- Delegates BAR0/FLOPS/MIGRATIONS/ARENAS to other TUs.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The "Modification A" comment in `cipher_probe.c` L99 (refresh
  comm every observation) is the reason `cipher_proc_show` can show
  fresh comm strings without locking against
  `task_struct->alloc_lock` itself.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for operator visibility. Not load-bearing for
substrate correctness.

### FUSION POINTS
- (1) Classifier output visibility: a new `/proc/cipher/classify`
  emitter would slot in alongside `flops` and `arenas`.
  PORT-AS-IS + new entry.

### State machine: n/a (read-only).
### Invariants
- Each proc entry's `proc_remove` is NULL-guarded (L560-587).
- Init's unwind cascade (L484-555) correctly removes earlier-created
  entries on later-creation failure.
### ABI surface: read-only /proc files (file descriptors).
### Concurrency under do_exit
A process reading `/proc/cipher/stats` while dying continues to
hold the seq_file lock until userspace dies; the kernel cleans up
the seq_file on `release()`. No special handling here.

---

## Section /home/ubuntu/cipher_kmod/cipher_bar0.c

### PURPOSE
Read-only BAR0 register access. Maps the first NVIDIA PCI device's
BAR0 via `pci_iomap`, caches `PMC_BOOT_0/1`, exposes live
`PMC_INTR_0/1` and `PBUS_INTR_STATUS` via `/proc/cipher/bar0_state`.
The pre-eminent design constraint: **READ-ONLY**, no `iowrite32` or
`__raw_writel` anywhere in this file (enforced by use of
`const void __iomem *` in `cipher_bar0_rd32`, L57-60).

### PUBLIC SURFACE
- `int cipher_bar0_init(void)` — L62.
- `void cipher_bar0_exit(void)` — L126.
- `int cipher_bar0_proc_show(struct seq_file *, void *)` — L145.

### CONTROL FLOW
**init** (L62-124):
1. `pci_get_device(NVIDIA, ANY)` (L67). `-ENODEV` if no NVIDIA card
   — this is *acceptable* (caller in `cipher_main.c` L59-64
   tolerates `-ENODEV`).
2. Class check (L73-78): must be display-class (0x03xxxx).
3. Multi-GPU detection (L80-85): if a second NVIDIA device exists,
   `pr_warn` and proceed with the first.
4. `pci_resource_start/len(0)` (L87-95). `-EIO` if length is 0.
5. Size sanity (L96-100): warn if not 16 MiB (the H100 expected).
6. `pci_iomap(pdev, 0, 16 MiB)` (L102). `-EIO` on fail (releases pdev).
7. Cache `PMC_BOOT_0/1` via `ioread32` (L112-113).
8. `pr_info` the raw values (L118-121).

**exit** (L126-143):
- `pci_iounmap`. **DELIBERATELY DOES NOT** `pci_dev_put` the
  pdev reference (L132-141). The comment is explicit: this is
  [[cipher-incident-2-bar0_exit]] — calling `pci_dev_put` here
  races with another driver's `devres` teardown
  (`devm_attr_group_remove → sysfs_remove_group` on a `kobj` whose
  `.sd` is NULL), crashing in `kernfs_find_and_get_ns`. One leaked
  `pci_dev` ref per module load; the device is kept alive by
  `nvidia.ko` regardless.

**proc_show** (L145-174): emits the cached `PMC_BOOT_0/1` plus
three live reads of `PMC_INTR_0/1` and `PBUS_INTR_STATUS`.

### STATE
File-static (L51-55):
- `struct pci_dev *cipher_pci_dev` — leaked at exit; nulled.
- `void __iomem *cipher_bar0_base` — iounmap'd, then nulled.
- `u32 cipher_bar0_pmc_boot_0/1` — cached.
- `unsigned long cipher_bar0_addr` — physical addr (for `pr_info`).

### CONCURRENCY
Single-threaded init/exit. The proc-show path can race with
anything, but it's all `ioread32` of a `__iomem` BAR — the
kernel's MMIO read is atomic at the bus level.

### DEPENDENCIES
- Linux PCI / IO (`pci_get_device`, `pci_iomap`, `ioread32`).
- Caller in `cipher_main.c::cipher_init`.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- **Cosmetic drift**: `cipher_bar0_proc_show` emits
  `"cipher_bar0 v0.3.1"` (L154) while the module is at 0.4.8.
  Cosmetic; nothing depends on it.
- The `pci_dev_put` leak is documented in-code at L132-141 and in
  the memory anchor [[cipher-incident-2-bar0_exit]]. Honest drift.
- The H100 PMC_BOOT_1=0x00000000 finding (L120-121's `pr_info`)
  surfaces in the kmod log; the bare-metal explanation
  ([[cipher-pmc-boot-1-bare-metal]] — VGPU bits absent on bare metal)
  lives in the memory anchor, not in the kmod source.

### CONTRIBUTION TO SYSTEM
NOT load-bearing — the module continues to load with
`cipher_bar0_base = NULL` on `-ENODEV` (`cipher_main.c` L59-64).
The proc emitter then emits `"layer disabled (no NVIDIA GPU bound)"`
(L150-152). LOAD-BEARING for hardware-presence proof.

### FUSION POINTS
- (1) Classifier consumption: BAR0 PMC_INTR_* could feed the
  classifier with interrupt-rate features. PORT-AS-IS + SHIM.
- (2) Insertion point: a new `cipher_bar0_observe()` hook in
  `cipher_proc_show` is the simplest path; or a kthread polling
  `PMC_INTR_*` if a higher cadence is needed.

### State machine
Init transitions BOUND ↔ UNBOUND; that's it. `cipher_bar0_base`
NULL ↔ non-NULL is the state.

### Invariants
- **Read-only**: enforced at type level (`const void __iomem *`).
  No write helper exists in this file.
- Whitelist of safe offsets (L19-25): `PMC_BOOT_0=0x0`,
  `PMC_BOOT_1=0x4`, `PMC_INTR_0=0x100`, `PMC_INTR_1=0x104`,
  `PBUS_INTR_STATUS=0x1100`. The header comment forbids reads
  in PFIFO (0x800000+) or PRAMIN.
- `cipher_bar0_base == NULL` ⇒ proc emits "layer disabled".

### ABI surface: `/proc/cipher/bar0_state` (read-only file).
### Concurrency under do_exit: not on this path.

---

## Section /home/ubuntu/cipher_kmod/cipher_clock.c

### PURPOSE
Phase 4.3 T4.3.2 — SM clock actuation via `call_usermodehelper`.
Implements ioctl nr 10 (`CIPHER_SET_CLOCK_MHZ`). The user-process
VOLT actuator hits an NVML privilege boundary
(`nvmlDeviceSetGpuLockedClocks` returns `NVML_ERROR_NOT_SUPPORTED`
from a non-root injection context); this kmod-mediated path runs
`/usr/bin/nvidia-smi -i 0 -lgc <mhz>` (or `-rgc` on `mhz == 0`)
under `UMH_WAIT_PROC`, inheriting root regardless of caller.

### PUBLIC SURFACE
- `int cipher_dev_set_clock_mhz(unsigned long arg)` — L130.
  Called from `cipher_dev.c` L268 on ioctl nr 10.

### CONTROL FLOW
1. `copy_from_user(&mhz)` (L135).
2. Bounds check (L139): `mhz == 0` (unlock) bypasses range;
   otherwise `mhz ∈ [210, 1980]` (L53-54 `CIPHER_CLOCK_MIN/MAX`).
   Out of range → `-EINVAL` after `pr_warn`.
3. `cipher_clock_apply(mhz)` (L145) — see below.
4. On success, `pr_info` with caller pid + uid (L147-149) for
   after-the-fact attribution audit.

`cipher_clock_apply` (L85-117):
1. `mutex_lock_interruptible(&cipher_clock_lock)` — at most one
   in-flight call. `-ERESTARTSYS` if interrupted.
2. Branch on `mhz == 0`: argv_reset (`-rgc`) vs argv_set
   (`-lgc <mhz>`).
3. `run_nvidia_smi` → `call_usermodehelper` with `UMH_WAIT_PROC`
   (L72). Returns the umh exit code.
4. Mutex unlock. Map `rc != 0` → `-EIO` (L114-115).

### STATE
- `static DEFINE_MUTEX(cipher_clock_lock)` — L56. Module-lifetime.
- No per-fd or per-tenant state.

### CONCURRENCY
- `cipher_clock_lock` serialises all nr-10 callers. The mutex is
  interruptible so a CTRL-C during ~100-300 ms `nvidia-smi` spawn
  unblocks cleanly with `-ERESTARTSYS`.
- The usermode helper runs in its own task context; no kmod
  mutexes are held while it's running (the mutex covers the
  entire `call_usermodehelper` call, but `nvidia-smi` itself is
  a userspace process).

### DEPENDENCIES
- Linux `umh`, `mutex`.
- `/usr/bin/nvidia-smi` must exist at the absolute path used
  (L88). No fallback.
- INBOUND: `cipher_dev_unlocked_ioctl` (`cipher_dev.c` L268).

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The "Trust model" block at L12-30 explicitly justifies the
  absence of a `capable(CAP_SYS_ADMIN)` check. The access gate
  is the device-node mode 0666 (codified by
  `cipher_dev.c::cipher_devnode`).
- The clamp [210, 1980] MHz is documented at L24-26 and enforced
  at L139. The 0 = unlock special case is the only bypass
  (L138, "0 is the explicit reset/unlock — bypass range check").
- Latency claim "~100-300 ms" (L39) — empirically measured
  per [[cipher-t432-kmod-volt-ioctl]] (~100-300 ms is the spawn
  + NVML round-trip).

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for the VOLT actuator (Wave 2 `cipher_rt_volt.c`).
Without this ioctl, the non-root user-process VOLT path cannot
set clocks — the gap that T4.3.2 closes (the same +36% watts
reduction without sudo, per [[cipher-t432-kmod-volt-ioctl]]).

### FUSION POINTS
- (1) Classifier consumption: a classifier-driven dynamic clock
  retarget would call this ioctl on each band-change. The
  fixed 100-300 ms latency makes it unsuitable for per-launch
  retargeting (the volt.c block at L37-40 says so explicitly).
- (2) Per-launch retarget is a v1.5+ deferral (Wave 2 fusion
  summary calls VOLT REFACTOR-REQUIRED for per-launch).

### State machine
None per-ioctl; per-device state is "locked at N MHz" vs
"unlocked", lives in the GPU's firmware not in the kmod.

### Invariants
- `mhz == 0` OR `mhz ∈ [210, 1980]` (L139).
- At most one in-flight call (mutex L102).
- Audit log on success (L147-149).
- Per-ioctl CAP_SYS_ADMIN is **intentionally absent** — see L21-23
  and the trust-model block.

### ABI surface
ioctl nr 10, `_IOW`, `__u32` payload, returns 0 / `-EFAULT` /
`-EINVAL` / `-EIO` / `-ERESTARTSYS`.

### Concurrency under do_exit
A caller process dying while its `nvidia-smi` spawn is in-flight
will: (1) the mutex is held in the dying task; (2) `UMH_WAIT_PROC`
blocks the dying task; (3) on signal the task returns
`-ERESTARTSYS` to userspace, which is now dead. The kernel cleans
up the dying task's mutex hold via the normal task-exit path.
**There is no lingering effect on the clock** — `nvidia-smi`
itself runs as a fresh root-owned task, decoupled from the
caller's lifecycle.

---

## Section /home/ubuntu/cipher_kmod/cipher_flops.c

### PURPOSE
CP 3.3 continuous FLOP telemetry + per-tenant MFU. The GH100 SM
perfmon PRI register map is not in the open kernel tree, so this
TU cannot read hardware FLOP counters directly. The `cipher_flopd`
root daemon does the hardware read via CUPTI PM Sampling (sampling
`sm__pipe_tensor_cycles_active` — the dominant FLOP path for LLM
inference). It submits a device-wide sample to the kmod ~10 Hz via
`CIPHER_SUBMIT_FLOP_SAMPLE` (nr 11). The kmod owns the 256-entry
ring + per-tenant attributed-FLOPs derivation. Per-tenant FLOPs are
ATTRIBUTED (device × launch-share), not measured.

### PUBLIC SURFACE
- `int cipher_flops_init(void)` — L336.
- `void cipher_flops_exit(void)` — L347.
- `long cipher_flops_submit(unsigned long arg)` — L89 (nr 11).
- `long cipher_flops_query(unsigned long arg)` — L178 (nr 12).
- `int cipher_flops_proc_show(struct seq_file *, void *)` — L263.

### CONTROL FLOW
**SUBMIT (nr 11)** — L89-165:
1. `CAP_SYS_ADMIN` check (L98-99).
2. `copy_from_user`.
3. Validation: `timestamp_ns != 0`, `tensor_pipe_milli_pct ≤ 100000`,
   `sm_util_pct ≤ 100` (L104-109).
4. Build a `cipher_flop_ring_ent` and append under
   `cipher_flop_lock` spinlock (L123-129). Modular ring index
   advance; cap `cipher_flop_count` at `RING_SIZE=256`.
5. **Pass 1 (per-tenant launch-delta sum)** under RCU read lock
   (L135-142). Walks the entire `cipher_pid_table`.
6. **Pass 2 (per-tenant share + attribution)** under same RCU
   read lock (L143-161). `share_ppm = delta * 1e6 / total_delta`;
   `attributed = device_rate/1e6 * share_ppm`. WRITE_ONCE both
   `attributed_flops_per_s` and the new `launches_at_last_flop`
   baseline.

**QUERY (nr 12)** — L178-260: unprivileged read.
1. `kzalloc` the query struct (~5 KiB with 64 tenant rows).
2. `copy_from_user` to get `max_tenants`.
3. Clamp `max_tenants ≤ 64` (`CIPHER_FLOP_QUERY_MAX_TENANTS` L316).
4. Read latest ring entry (L203-218). Status:
   `NO_SOURCE` if ring empty, `STALE` if `age > 2*HZ` (L48),
   `LIVE` otherwise.
5. RCU walk: fill per-tenant rows up to `cap` (L224-251).
6. `copy_to_user`, free.

**proc_show** — L263-334: human-readable dump. Same structure as
QUERY but pretty-printed.

### STATE
- 256-entry static ring `cipher_flop_ring[]` (L46, L63).
- `cipher_flop_head` (next-write index), `cipher_flop_count`
  (populated, capped at 256), `cipher_flop_total_samples`
  (lifetime, no cap).
- `DEFINE_SPINLOCK(cipher_flop_lock)` (L62).

### CONCURRENCY
- Single-writer for SUBMIT (one `cipher_flopd` daemon by design),
  but the spinlock protects against any concurrent QUERY/proc
  reader doing a `cipher_flop_latest` read.
- Per-tenant `attributed_flops_per_s` and `launches_at_last_flop`
  are written ONLY in SUBMIT — single-writer because there is one
  daemon — and read by QUERY and proc.

### DEPENDENCIES
- `cipher_pid_table` (RCU walks).
- INBOUND: `cipher_dev.c` L270/272 dispatches nrs 11/12 here;
  `cipher_proc.c` L428 delegates `/proc/cipher/flops` here.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The "PEAK = 989 TFLOP/s" constant (L42) matches
  `cipher_metrics.py PEAK_FLOPS_SPEC` (per the comment L40-41).
- The "PM Sampling fits exactly one metric per single-pass config"
  finding from `cipher_ioctl.h` L277-282 is the reason
  `device_fp_flops_per_s` is reserved 0 — the daemon only
  samples the tensor pipe. This is the matrix in the
  measurement-of-opportunity gate [[cipher-t46-measurement]].

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for the MFU readout. Without this, no real-time
device-MFU number, and no per-tenant attribution.

### FUSION POINTS
- (1) Classifier consumption: per-tenant MFU is a load feature.
  PORT-AS-IS + SHIM (read via nr 12 or proc).
- (2) MFU-gated decisions are how Wave 1 + Wave 2 frame the
  "ship on MFU gates not TPW lift" discipline ([[cipher-lift-framing]]).

### State machine
Per-sample lifecycle inside the ring: WRITE_AT(head) → bump head
mod 256 → bump count (capped) → bump total_samples (unbounded).

### Invariants
- Ring size 256 (`CIPHER_FLOP_RING_SIZE` L46) — about 25 s of
  history at 10 Hz, enough for any liveness check.
- `STALE` threshold 2 × HZ (L48): sample older than 2 s is reported
  STALE.
- Attribution is launch-share proportional, not measured. The
  comment at L17-19 is explicit: "gate-sufficient for real-time
  MFU", not "true per-context measurement".

### ABI surface
ioctl nrs 11 (CAP_SYS_ADMIN, SUBMIT) and 12 (unprivileged, QUERY).
`/proc/cipher/flops`. Failure modes:
- nr 11: `-EPERM` (no CAP), `-EFAULT`, `-EINVAL`.
- nr 12: `-ENOMEM`, `-EFAULT`.

### Concurrency under do_exit
A tenant dying mid-attribution: the SUBMIT pass uses RCU read
lock; a `kfree_rcu`'d entry remains valid for the duration of
the RCU section. The next SUBMIT after the entry is gone simply
does not see it in the hashtable walk. No leaks.

---

## Section /home/ubuntu/cipher_kmod/cipher_cp54_sched.c

This is the largest TU in the module (894 lines). Spending the
budget it deserves: state machine, three migration ioctl flows,
conservation invariants under `do_exit`, the SC2 bit-30 reservation
encoding, the POOL low-prefix invariant, and the SC4 policy
parameters all get explicit treatment below.

### PURPOSE
CP 5.4 per-tenant SM-allocation arbitration ledger. The kmod-
resident cross-process SM allocator. CUDA green contexts on H100
partition in multiples of 8 SMs; this ledger allocates the device
as **15 × 8-SM groups** (120 of 132 SMs; the 12-SM remainder is
unallocatable — per [[cipher-cp54-15groups]] this is what
`cuDevSmResourceSplitByCount(minCount=8)` returns on H100 80GB
SXM5). Track 3 SC2 added the Dynamic SM Migration state machine
on top of the SC1 partition/pool model; Track 3 SC4 added the
module-parameter policy surface and the
`/proc/cipher/migrations` emitter.

### PUBLIC SURFACE
- `int cipher_cp54_sched_init(void)` — L867.
- `void cipher_cp54_sched_exit(void)` — L888.
- `void cipher_cp54_release(pid_t)` — L360 (do_exit reaper hook).
- Five ioctl handlers (nrs 13-15):
  - `cipher_cp54_ioctl_allocate(arg)` — L550 (nr 13).
  - `cipher_cp54_ioctl_free(arg)` — L646 (nr 14).
  - `cipher_cp54_ioctl_query(arg)` — L662 (nr 15).
- Five Track 3 SC2 migration ioctl handlers (nrs 16-20):
  - `cipher_cp54_ioctl_subscribe_migrate(arg)` — L705 (nr 16).
  - `cipher_cp54_ioctl_poll_migrate(arg)` — L726 (nr 17).
  - `cipher_cp54_ioctl_start_migrate(arg)` — L754 (nr 18).
  - `cipher_cp54_ioctl_ack_migrate(arg)` — L773 (nr 19).
  - `cipher_cp54_ioctl_compact_migrate(arg)` — L797 (nr 20).
- One /proc emitter:
  - `cipher_cp54_migration_proc_show(seq_file *, void *)` — L811.
- Four module parameters (mode 0644 — boot-time via insmod, runtime
  via sysfs write, L95-110):
  - `cipher_cp54_mig_gap_min_grps` (int, default 1).
  - `cipher_cp54_mig_sustain_ms` (uint, default 2000).
  - `cipher_cp54_mig_ratelimit_ms` (uint, default 10000).
  - `cipher_cp54_mig_verbose` (int, default 1).

### STATE
- **Group ledger** (L118):
  `atomic_t cipher_cp54_groups[15]` — the source of truth.
  Three-state per-group encoding (see "State machine: group cell"
  below).
- **Metadata table** (L123, L139):
  `struct cipher_cp54_alloc cipher_cp54_allocs[64]` — at most
  64 simultaneously-classified tenants. Each entry carries:
  `pid`, `qos_class`, `in_use`, plus the SC2 migration state
  (`migratable`, `migrate_state`, `last_outcome`, `target_mask`,
  `proposed_ns`, `last_migrate_ns`, `migration_count`).
- **Pool owner** (L156): `atomic_t cipher_cp54_pool_pid` — the
  singleton batch-pool owner PID; 0 = none.
- **Lock** (L158): `DEFINE_MUTEX(cipher_cp54_lock)`.
- **Migration gating** (L143): `static u64 cipher_cp54_gap_since_ns`
  — when the current POOL-unreachable gap was first observed.
- **Migration counters** (L147-151): five `atomic_t` for
  proposals, commits, abort_timeout, abort_nack, abort_refused
  (the last is reserved; the kmod never sets it currently — the
  field exists so the ABI carries the "kmod refused" outcome for
  future policies).

### CONTROL FLOW

**ALLOCATE (nr 13)** — `cipher_cp54_ioctl_allocate` (L550-644):

1. `copy_from_user(&p)` (L559).
2. Validate `p.qos_class ∈ {PARTITION, SHARED, POOL}` (L561-564).
3. `mutex_lock(&cipher_cp54_lock)`.
4. `cp54_meta_get_or_create(pid)` (L568) — find or create the
   metadata slot. `-ENOMEM` if all 64 slots are full.
5. Class-switching rule (L578-582): a tenant already PARTITION or
   POOL cannot switch class without FREE-ing first. SHARED → any
   is allowed (SHARED owns no groups).
6. Switch on `p.qos_class`:
   - **POOL** (L585-606): singleton check — `atomic_read(pool_pid)
     != 0 && != pid` → `-EEXIST`. Else `atomic_set(pool_pid, pid)`,
     classify as POOL, call `cp54_pool_claim_low_prefix(pid)`
     (L604) which walks groups 0..14 claiming each free or
     already-POOL-owned cell until the first PARTITION-owned or
     RSVD cell — the "maximal contiguous LOW PREFIX" invariant
     ([[cipher-cp54-step1-3b]] / SC5 §1 fragmentation finding).
   - **PARTITION** (L608-627): `need = ceil(sm_count/8)`, clamped
     to `[1, 15]`. Idempotent re-call: `have = popcount(cp54_mask_of(pid))`.
     If `have < need`: `cp54_claim` against free groups first; if
     still short, `cp54_pool_shrink(shortfall)` then re-claim.
     Final mask via `cp54_mask_of(pid)`; `-ENOSPC` if 0.
   - **SHARED** (L629-633): owns no groups; `mask = 0`.
7. Unlock; `copy_to_user(p.grp_mask_out, grp_count_out)`.

**FREE (nr 14)** — `cipher_cp54_ioctl_free` (L646-660):
- `cp54_release_impl(current->pid)` (L649) — lock-free.
- Then under mutex, `cp54_eval_migration(false)` (L656-658) —
  may trigger a compaction PROPOSE if the freed groups left a
  POOL-unreachable gap.

**QUERY (nr 15)** — `cipher_cp54_ioctl_query` (L662-698):
- Counts registered PARTITION-class tenants (L675-680).
- Pool group count, free group count.
- Caller's own mask and qos_class (sentinel `0xFFFFFFFF` for
  unregistered).
- All under `READ_ONCE` — diagnostic, torn read is benign.

**SUBSCRIBE_MIGRATE (nr 16)** — `cipher_cp54_ioctl_subscribe_migrate`
(L705-722):
- Get or create the caller's metadata slot.
- Set `migratable = val ? 1 : 0`.
- Tenants that never call this stay pinned (default 0). All-pinned
  workloads (the regression smoke tests) never PROPOSE — the
  `cp54_eval_migration` filter at L506-508 short-circuits.

**POLL_MIGRATE (nr 17)** — `cipher_cp54_ioctl_poll_migrate`
(L726-749):
- Read `cur_mask` lock-free via `cp54_mask_of(pid)`.
- If caller has a metadata slot: read `migrate_state`, `target_mask`,
  `last_outcome` lock-free with `READ_ONCE`. (The kernel guarantees
  no torn 32-bit reads on aligned u32, but `READ_ONCE` is the
  defensive style; a torn migrate_state at worst causes one more
  poll round.)

**START_MIGRATE (nr 18)** — `cipher_cp54_ioctl_start_migrate`
(L754-768):
- Under mutex: must be PROPOSED. `-EINVAL` otherwise. Sets state
  to MIGRATING.

**ACK_MIGRATE (nr 19)** — `cipher_cp54_ioctl_ack_migrate` (L773-792):
- `copy_from_user(&ok)`.
- Under mutex: must be MIGRATING. `-EINVAL` otherwise.
- `ok=1` → `cp54_commit_migration` (L787) — release the tenant's
  PACK groups, promote its RSVD groups to PACK, set
  `last_outcome=COMMITTED`, increment commit counter.
- `ok=0` → `cp54_abort_migration(NACK)` — release RSVDs, keep
  PACKs, `last_outcome=ABORTED_TENANT_NACK`.

**COMPACT_MIGRATE (nr 20)** — `cipher_cp54_ioctl_compact_migrate`
(L797-804): Under mutex, `cp54_eval_migration(true)` — `forced=true`
bypasses the gap-size + sustain-time gates but still respects
opt-in + IDLE + rate-limit (L491-497, then L506-512).

### State machine: per-group cell (3-state, atomic)

```
Cell value (u32, in atomic_t):

   0x00000000                       FREE
   0x80000000 | (pid & 0x3FFFFFFF)  OWNED by pid     (bit 31)
   0x40000000 | (pid & 0x3FFFFFFF)  RESERVED for pid (bit 30 — RSVD)
```

Encoded by `CIPHER_CP54_GRP_PACK(pid)` (L77-78) and `_GRP_RSVD(pid)`
(L79-80). The packed pid uses bits 0..29 (`GRP_PID_MASK 0x3FFFFFFF`,
L76).

**The 30-bit-pid choice** is the [[cipher-abi-rule]]-friendly trick
that earned its own anchor in the SC2 work. Linux's
`PID_MAX_LIMIT` is `2^22` (4.19 M), so 30 bits hold 1.07 B (256×
headroom). For any real PID < 2^22, `GRP_PACK` is byte-identical
to the pre-SC2 value (which used bit 31 + 31-bit pid; see L70-73
in-code documentation), so the SC1 ledger semantics are preserved
under the SC2 extension.

Cell transitions:

- FREE → OWNED: `atomic_cmpxchg(cell, 0, GRP_PACK(pid))`
  (the `cp54_claim` and `cp54_pool_claim_low_prefix` hot paths).
- FREE → RESERVED: `atomic_cmpxchg(cell, 0, GRP_RSVD(pid))`
  (`cp54_reserve` in PROPOSE; L235-245).
- OWNED → FREE: `atomic_cmpxchg(cell, GRP_PACK(pid), 0)`
  (`cp54_release_groups`, `cp54_pool_shrink`).
- RESERVED → FREE: `atomic_cmpxchg(cell, GRP_RSVD(pid), 0)`
  (`cp54_release_rsvd`, `cp54_abort_migration`'s `cp54_release_rsvd`
  call).
- RESERVED → OWNED: `atomic_cmpxchg(cell, GRP_RSVD(pid), GRP_PACK(pid))`
  (`cp54_commit_migration`'s second loop L391-392, after the
  first loop has cleared the old PACKs).

Every transition is a single atomic cmpxchg → lock-free, safe in
atomic context (the `do_exit` kprobe path).

### State machine: per-tenant migration

```
                  [kmod cp54_eval_migration]
                                |
   IDLE -----PROPOSE---------> PROPOSED
                                |  |
              [tenant START]----+  |  [timeout
                                |  |   (no START within 30s)]
                                v  v
              MIGRATING <-------+  +--> ABORT (timeout)
                  |
   +--------------+--------------+
   |                             |
[tenant ACK(ok=1)]      [tenant ACK(ok=0) NACK]
   |                             |
   v                             v
 COMMIT                        ABORT (tenant_nack)
   |                             |
   +-------> IDLE <--------------+
            (last_outcome set on every transition back to IDLE)
```

State constants (`cipher_ioctl.h` L421-423): `IDLE=0`, `PROPOSED=1`,
`MIGRATING=2`. `COMMIT` and `ABORT` are not states the tenant
sees — they are transient `cp54_{commit,abort}_migration` actions
that re-enter IDLE.

`last_outcome` (`cipher_ioctl.h` L430-434): `NONE=0`, `COMMITTED=1`,
`ABORTED_TIMEOUT=2`, `ABORTED_KMOD_REFUSED=3` (reserved),
`ABORTED_TENANT_NACK=4`. Set on every transition back to IDLE;
survives until the next migration.

The 30-second `MIG_TIMEOUT_NS` (L93) governs PROPOSED → ABORT.
The kmod **never unilaterally aborts a MIGRATING migration** —
only the tenant may (via ACK(0)). The comment at L416-419 is
explicit: "MIGRATING migrations are NOT timed out by the kmod
(SC1 §3d: safe-but-stuck; a hung MIGRATING tenant is reclaimed
only by the do_exit reaper)".

### The compaction algorithm — cp54_eval_migration (L445-546)

This is the core invention of Track 3 SC2 (Dynamic SM Migration).
A migratable PARTITION tenant blocking the POOL's contiguous low
prefix gets PROPOSEd a count-preserving move to the k highest-
indexed free groups.

1. Abort any expired PROPOSED migrations (L456 → `cp54_abort_timeouts`).
2. Scan the ledger:
   - `free_mask`: every cell with value 0 (L463-465).
   - `lowest_part_g`: lowest group index owned by a non-POOL
     OCC cell (L466-470).
3. `stranded_mask = free_mask & ~((1U << (lowest_part_g + 1)) - 1)`
   — free groups strictly above the lowest partition (L480).
   `stranded_count = hweight32(stranded_mask)`.
4. If `lowest_part_g >= 15` (no partition) or `stranded_count == 0`
   (no gap above): reset `gap_since_ns = 0`, return.
5. Otherwise stamp `gap_since_ns` if first time (L486-487).
6. **Policy gates** (skipped if `forced=true`, COMPACT_MIGRATE):
   - `stranded_count < gap_min_grps` → return.
   - `now - gap_since_ns < sustain_ms * 1e6` → return.
7. Identify the blocker: the partition owning `lowest_part_g`
   (L501-502). Skip if not migratable, not IDLE, or rate-limited
   (the last `< ratelimit_ms` ago).
8. Compute `own_mask = cp54_mask_of(cand_pid)`, `k = popcount`.
9. **Count-preserving target** (L519-526): scan groups 14..0,
   pick the k highest-indexed free groups into `new_mask`.
10. Sanity: `cnt == k`, `new_mask != own_mask`, `__ffs(new_mask)
    > lowest_part_g` (the move would actually free the blocker).
11. `cp54_reserve(cand_pid, new_mask)` — atomic_cmpxchg every bit
    to RSVD(cand_pid). On a short count (a free cell was stolen
    by a concurrent FREE), roll back via `cp54_release_rsvd` and
    return.
12. State transition to PROPOSED: stamp `target_mask`, `proposed_ns`,
    `last_migrate_ns`, bump proposal counter.

### COMMIT (cp54_commit_migration, L383-398)

Two-pass cmpxchg:
- Pass 1 (L389-390): every cell with value `GRP_PACK(pid)` → 0
  (release the old PACK groups atomically).
- Pass 2 (L391-392): every cell with value `GRP_RSVD(pid)` →
  `GRP_PACK(pid)` (promote the reservation).

After the two passes the tenant owns exactly `target_mask`, the
old groups are free, `migrate_state=IDLE`, `target_mask=0`,
`last_outcome=COMMITTED`, `migration_count++`, commit counter++.

The two-pass ordering matters under crash: a do_exit between the
two passes leaves cells in either state (the tenant's PACK is
cleared, but its RSVD is not yet promoted). The do_exit reaper
calls `cp54_release_groups(pid)` which clears BOTH PACK and RSVD
states in a single sweep (L196-217) — **so a tenant that crashes
mid-COMMIT leaks zero groups**, because the reaper's cmpxchg
loop matches whichever state survives.

### ABORT (cp54_abort_migration, L402-414)

`cp54_release_rsvd(pid)` (L222-229) — every cell with value
`GRP_RSVD(pid)` → 0. The tenant keeps its OWNED groups untouched
(the B-floor fallback). `last_outcome` records why.

### Conservation invariants

The ledger maintains five invariants:

I1. **Group ownership disjoint.** Each cell has exactly one owner
    or is FREE/RSVD. Atomic cmpxchg enforces this — claim
    succeeds only on 0→PACK transition, so no two tenants can
    simultaneously hold the same cell.

I2. **POOL singleton.** `cipher_cp54_pool_pid` is an `atomic_t`
    set by `atomic_set(pid)` in ALLOCATE(POOL) (L595) and cleared
    by `atomic_cmpxchg(pid, 0)` in release (L351 — only clear if
    still this dying pid). The `cur != 0 && cur != pid` check
    (L590) returns `-EEXIST` on a second registrant. Idempotent
    re-call by the same pid is allowed.

I3. **POOL contiguous low prefix.** Enforced by
    `cp54_pool_claim_low_prefix` (L275-294): the POOL only
    claims cells starting at index 0 and stops at the first
    partition-owned or RSVD cell. A free cell above a partition
    is POOL-unreachable until Dynamic SM Migration compacts it
    back ([[cipher-cp54-step1-3b]]).

I4. **Migration count-preserving.** `popcount(new_mask) ==
    popcount(old_mask)` is enforced by the target-mask construction
    (L519-526): the algorithm picks exactly k highest-indexed
    free groups.

I5. **PID encoded in 30 bits** (`GRP_PID_MASK`, L76). 256× headroom
    against `PID_MAX_LIMIT=2^22`. The encoding is kmod-internal;
    userspace sees `pid_t` unchanged through the ABI (the
    `cipher_cp54_*` structs all use raw `__u32 pid` fields, never
    the packed form).

### CONCURRENCY

- **ALLOCATE/SUBSCRIBE/START/ACK/COMPACT** take
  `cipher_cp54_lock` (mutex) for the whole operation.
  - The V1 scoping memo originally kept a "lock-free fast path"
    for non-resize allocates; this build takes the mutex for the
    whole ALLOCATE because the metadata-table slot claim also
    needs serialisation. The code self-discloses this drift
    at L26-31 — honest in-source documentation of the deviation
    from the memo, flagged for the Step 1.2 adjudication checkpoint.
- **FREE** invokes `cp54_release_impl` lock-free, then takes
  the mutex for `cp54_eval_migration(false)`.
- **POLL/QUERY** are lock-free atomic reads.
- **`cipher_cp54_release(pid)`** — the do_exit reaper hook — is
  **lock-free** (L342-357): only touches the dying pid's cells
  via cmpxchg + the dying pid's metadata slot via WRITE_ONCE. It
  is called from the kprobe atomic context (`cipher_probe.c` L198)
  and **must not** take the mutex.

The discipline that makes the lock-free reaper safe: every
metadata field a tenant writes is ALSO only ever written by that
tenant's own ALLOCATE/SUBSCRIBE/etc., which run under the mutex.
A mutex-held ALLOCATE cannot conflict with a reaper call from
the same dying pid (the task can't be both alive in `ALLOCATE`
and exiting in `do_exit`). A reaper for pid A and a mutex-held
ALLOCATE by pid B touch DIFFERENT metadata slots, so the
reaper's `WRITE_ONCE(in_use, 0)` is benign.

### DEPENDENCIES

- INBOUND:
  - `cipher_dev.c` L273-288 dispatches nrs 13-20.
  - `cipher_probe.c` L198 calls `cipher_cp54_release` on do_exit.
  - `cipher_proc.c` L442 delegates `/proc/cipher/migrations` to
    `cipher_cp54_migration_proc_show`.
- OUTBOUND: kernel mutex/atomic/bitops, `ktime_get_ns`, `printk`,
  `current->pid`. No CUDA driver dependency — the ledger is pure
  bookkeeping; libcipher_rt's green-ctx allocator enacts the
  placement (Wave 2 `cipher_rt_green_ctx.c`).

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED

- The "15 not 16" finding ([[cipher-cp54-15groups]]) is wired into
  the source at L59-62 and `CIPHER_CP54_NUM_GROUPS = 15` (L61).
- The conservative-default migration policy (L82-87): the code
  describes its own conservatism — "all-pinned regression workloads
  (W1/W2/W3) behave as if migration did not exist". Verified at
  L506-508 (a non-migratable blocker short-circuits) and L491-497
  (gap-size + sustain-time gates skipped only on `forced`).
- The "lock-free fast path" memo-vs-build drift (L24-31): code is
  honest about deviating from the memo's literal text.

### CONTRIBUTION TO SYSTEM

LOAD-BEARING. This file is the sole cross-process SM-allocation
authority for CP 5.4 and Track 3. The legacy nr-9 allocator is
deactivated; CP 5.4 is the source of truth. Track 3 DSM (Dynamic
SM Migration) per [[cipher-track3-dsm]] depends entirely on this
file's migration FSM.

### FUSION POINTS

- (1) Classifier consumption: a band-aware POOL/PARTITION sizer
  would read tenant snapshots (nr 8) for `slo_priority` and
  re-issue `ALLOCATE` with the appropriate `sm_count`. SHIM
  on the userspace side; no kmod change.
- (2) New migration policy: implement a policy callback that
  reads `cipher_pid_table` for tenant load (launches_total) and
  proposes a different blocker, not just the lowest-indexed
  partition. The structure of `cp54_eval_migration` makes this
  a localised refactor (~50 LOC); REFACTOR-REQUIRED (small).
- (3) Per-tenant SM-count classifier hint: extend the
  `cipher_tenant_snapshot_user` (`cipher_ioctl.h` L123-179) with
  a `recommended_sm_count` field in the reserved tail; the
  ALLOCATE call reads it on entry. SHIM.

### Invariants table (CP 5.4 + Track 3 SC2)

| # | Invariant                                          | Enforcement                                                                                                                       |
|---|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| 1 | Group cell ownership is exclusive                  | atomic_cmpxchg on every transition (claim/reserve/release)                                                                        |
| 2 | POOL is a singleton                                | `atomic_read(pool_pid) != 0 && != pid` returns `-EEXIST` (L590); cleared by `atomic_cmpxchg(pid, 0)` (L351) — concurrent-safe     |
| 3 | POOL holds a contiguous low prefix                 | `cp54_pool_claim_low_prefix` walks 0..N-1, stops at first non-self non-free cell (L275-294)                                       |
| 4 | Migration is count-preserving                      | target_mask construction picks exactly `k` highest free groups (L519-526); `cp54_reserve` returns short on race → roll back       |
| 5 | PID fits 30 bits                                   | `GRP_PID_MASK = 0x3FFFFFFF`; PID_MAX_LIMIT = 2^22 ⇒ 256× headroom (L70-73)                                                        |
| 6 | Tenant cannot change qos_class without FREE        | L578-582: PARTITION/POOL → other class returns `-EINVAL`                                                                          |
| 7 | Tenant crashing mid-migration leaks zero groups    | `cp54_release_groups` sweeps both PACK and RSVD in one pass (L196-217); lock-free; safe in do_exit atomic context                 |
| 8 | Kmod never unilaterally aborts MIGRATING            | `cp54_abort_timeouts` (L420-433) only touches PROPOSED, never MIGRATING                                                           |
| 9 | One migration proposal per `eval_migration` pass   | Only the lowest_part_g's owner is proposed; the function returns after one PROPOSE (L538-545)                                     |
|10 | Non-migratable tenants are never touched           | `if (!cand->migratable || ...) return` at L506-508                                                                                |

### ABI surface

ioctl nrs 13/14/15 (ALLOCATE/FREE/QUERY) and 16-20 (SUBSCRIBE_/POLL_/
START_/ACK_/COMPACT_MIGRATE). Failure modes per ioctl:

- **nr 13 ALLOCATE**: `-EFAULT` (copy), `-EINVAL` (bad qos_class or
  bad re-classify), `-ENOMEM` (metadata table full),
  `-EEXIST` (POOL already taken by another pid), `-ENOSPC`
  (PARTITION found 0 groups).
- **nr 14 FREE**: always 0 (no payload, idempotent).
- **nr 15 QUERY**: `-EFAULT` only.
- **nr 16 SUBSCRIBE_MIGRATE**: `-EFAULT`, `-ENOMEM`.
- **nr 17 POLL_MIGRATE**: `-EFAULT` only.
- **nr 18 START_MIGRATE**: `-EINVAL` (not PROPOSED).
- **nr 19 ACK_MIGRATE**: `-EFAULT`, `-EINVAL` (not MIGRATING).
- **nr 20 COMPACT_MIGRATE**: always 0.

### Concurrency under do_exit

`cipher_cp54_release(pid)` (L360-363) is the unconditional first
step of `cipher_do_exit_pre` (`cipher_probe.c` L198). It calls
`cp54_release_impl(pid)`:
1. `cp54_release_groups(pid)` (L347) — clears every cell whose
   value matches `GRP_PACK(pid)` OR `GRP_RSVD(pid)`. Lock-free.
2. `atomic_cmpxchg(pool_pid, pid, 0)` (L351) — only clears if
   still this pid.
3. Scan the metadata table, clear `in_use` for any slot whose
   `pid` matches (L352-356).

**Crash matrix:**

| Phase tenant crashes in       | After reaper completes                  |
|-------------------------------|------------------------------------------|
| IDLE (PARTITION/POOL)         | All groups freed; metadata cleared       |
| PROPOSED (before START)       | OWNED freed, RSVD freed; metadata cleared|
| MIGRATING (before ACK)        | OWNED freed, RSVD freed; metadata cleared|
| Mid-COMMIT (between two passes)| Whichever still in PACK or RSVD freed   |
| Post-ABORT, returning to IDLE | OWNED freed; metadata cleared            |

The `cp54_release_groups` two-state sweep (L196-217) is what makes
this safe — it cleans up both encodings in one pass. **Zero
group-leak under any tenant-crash phase.** This is the conservation
property that makes CP 5.4 robust enough to ship.

The metadata slot's `in_use` clear (L355) is `WRITE_ONCE`, not
under the mutex — but each PID only ever writes its own slot, so
there is no same-slot write race; a concurrent mutex-held
ALLOCATE by a different pid touches a different slot.

---

## Section /home/ubuntu/cipher_kmod/cipher_partition_allocator.c

### PURPOSE
Phase 4.2 T4.2.1 SM partition allocator. Lock-free `atomic_t`-slot
allocator partitioning H100's 132 SMs into 33 × 4-SM slots.
**Userspace path deactivated**: `cipher_dev.c` L107-113 returns
`-ENOSYS` for ioctl nr 9 because CP 5.4 superseded the
4-SM-slot model. But the file is still in Kbuild (L15), and
two of its functions are still live:
- `cipher_partition_release_slots_only(pid)` is called by the
  do_exit reaper (`cipher_probe.c` L210), reclaiming any slots
  the legacy allocator might still hold.
- `cipher_partition_tick()` is called every loop iteration by
  the state-updater kthread (`cipher_state_updater.c` L176), which
  internally gates to a 5-second rebalance + idle-reclaim cadence.

### PUBLIC SURFACE
- `int cipher_partition_allocator_init(void)` — L448.
- `void cipher_partition_allocator_exit(void)` — L476.
- `int cipher_partition_request(target_pid, hint, *out_mask, *out_count)` — L124, EXPORT_SYMBOL_GPL.
- `int cipher_partition_request_v2(target_pid, hint, flags, *, *)` — L147, EXPORT_SYMBOL_GPL.
- `int cipher_partition_release(target_pid)` — L260, EXPORT_SYMBOL_GPL.
- `void cipher_partition_release_slots_only(target_pid)` — L282, EXPORT_SYMBOL_GPL.
- `void cipher_partition_tick(void)` — L429.

### CONTROL FLOW
**REQUEST (legacy, was nr 9; now unreachable from userspace)** —
`cipher_partition_request_v2` L147-254:
- `hint_clamp ∈ [1, 8]` (`CIPHER_PARTITION_HINT_CAP=8`, L56), or
  0 with FIT_HINT (release-all).
- Idempotent fast path (skipped with FIT_HINT): read cached
  mask/count from `cipher_pid_stats`, return if cached_count ≥ hint
  (L183-199).
- First pass: count current allocations + claim free slots up to
  `hint_clamp` via `atomic_cmpxchg(0, my_state)` (L203-215). FCFS.
- FIT_HINT release pass: release highest-indexed slots back to
  free via `atomic_cmpxchg(my_state, 0)` (L222-236).
- Writeback cache: `WRITE_ONCE(sm_partition_mask/count)` if
  granted set changed (L239-240).
- Return: `-ENOSPC` if `have == 0 && hint_clamp > 0`.

**RELEASE** — `cipher_partition_release` L260-276: same
`cmpxchg(my_state, 0)` per slot, plus cache writeback. Returns
`-ESRCH` if released nothing.

**SLOT-ONLY RELEASE** (do_exit hot path) — L282-293: identical
cmpxchg loop, **no cache writeback** — the comment at L279-281
explains: the caller (do_exit reaper) is about to delete the
`cipher_pid_stats` entry, so cache writeback would be wasted work.

**TICK** (called from kthread loop) — `cipher_partition_tick`
L429-446: gated to `REBALANCE_INTERVAL_NS=5s` (L58). Each
gated firing runs:
- `cipher_partition_reap_idle` (L298): scan `cipher_pid_table`,
  release slots held by tenants idle > 30 s
  (`PARTITION_IDLE_NS=30s`, L57).
- `cipher_partition_rebalance` (L352): scan + sort by
  `launches_total` descending, then quartile-cap (top 25% → 8 slots,
  next 25% → 4, next → 2, bottom → 1). Over-allocated tenants
  release highest-indexed slots first.

### STATE
- `struct cipher_partition_slot cipher_slots[32]` (L91): each
  `____cacheline_aligned` (L90) to eliminate false sharing under
  heavy contention.
- `static int cipher_partition_slots` (L78): runtime slot count,
  derived from `sm_count` parameter at init.
- `static u64 cipher_last_rebalance_jiffies` (L79): gates the
  5 s rebalance.

### CONCURRENCY
- Slot array writes are all `atomic_cmpxchg` — lock-free.
- Cache writeback uses `WRITE_ONCE` under `rcu_read_lock`.
- The tick (called from the state-updater kthread) and the
  do_exit reaper can race against an in-flight (hypothetical)
  REQUEST call, but since REQUEST is `-ENOSYS`, the in-flight
  case doesn't exist any more. The tick still releases slots
  on idle/quartile-rebalance.

### DEPENDENCIES
- INBOUND:
  - `cipher_probe.c::cipher_do_exit_pre` L210 → `_release_slots_only`.
  - `cipher_state_updater.c::cipher_state_updater_fn` L176 → `_tick`.
  - `cipher_dev.c::cipher_dev_request_sm_partition` L107 (returns `-ENOSYS` before reaching this file).
- OUTBOUND: `cipher_pid_table` (RCU walk), `WRITE_ONCE` on
  `cipher_pid_stats::sm_partition_mask/count`.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- **STALE DOC**: L33-37 lists "Driven by: cipher_dev_request_sm_partition
  handler (T4.2.1 ioctl nr 9)" — but that handler now returns
  `-ENOSYS`. The allocator is reachable only via state_updater_tick
  + the do_exit reaper. Header comment is stale relative to CP 5.4.
- **B6 cap documented**: L59-66 the SLOTS_MAX=32 cap is well-
  documented (UBSAN shift-out-of-bounds + bit-0 aliasing). On
  H100 SXM5 with sm_count=132, slots=33 → clamped to 32 (L461-462),
  one slot of capacity sacrificed for ABI stability.
- **B10 FIT_HINT documented inline** at L132-146.
- **EXPORT_SYMBOL_GPL** on four public functions (L130, L255, L277,
  L293) — anticipated `cipher_rt_km` consumer per L36 ("future
  cipher_rt_km can call"). No consumer ships today.

### CONTRIBUTION TO SYSTEM
**Partial dependency.** The legacy allocator's public surface is
dead, but its housekeeping (tick + do_exit slot release) still
runs. Removing the file would not break userspace today (nr 9 is
already `-ENOSYS`), but the storage in `cipher_pid_stats`
(`sm_partition_mask`, `sm_partition_count`, L144-145) is read by
the snapshot at ioctl nr 8 and the `state_updater` kthread.

### FUSION POINTS

- (1) Classifier consumption: none.
- (2) Retirement: the right move is to remove the slot array
  + tick + EXPORT_SYMBOLs, but **keep** the cache writeback
  fields on `cipher_pid_stats` for CP 5.4 to write through —
  the snapshot ABI already exposes `sm_partition_mask/count`
  via ioctl nr 8.
- Fusion class: **REFACTOR-REQUIRED (small)** to fully retire,
  not VESTIGIAL — because the housekeeping functions are still
  called by live callers (state_updater, do_exit reaper). The
  advisor's prompt-cross-check on this is correct: the file is
  a "deactivated-public-surface allocator with live housekeeping".

### State machine
Per-slot cell: FREE (0) ↔ OWNED (`CIPHER_SLOT_OCCUPIED |
pid31`). Two-state, simpler than CP 5.4's three-state (no
reservation concept).

### Invariants
- Cell ABA-safe via packed state (L20-23 in-code documentation):
  release-then-reacquire from a different tenant produces a
  different state value.
- Slot count clamp 32 (L66, L461-462) — ABI stability invariant.
- Per-tenant cap 8 slots (`CIPHER_PARTITION_HINT_CAP=8`, L56).

### ABI surface
ioctl nr 9 (deactivated). The legacy callers that EXPORT_SYMBOL_GPL
anticipated never materialised — `cipher_rt_km` was not built.

### Concurrency under do_exit
`cipher_partition_release_slots_only` is the lock-free slot
release path (L282-293). Called from the kprobe atomic context
in `cipher_probe.c` L210. Skips the cache writeback because the
cipher_pid_stats entry is about to be `kfree_rcu`'d.

---

## Section /home/ubuntu/cipher_kmod/cipher_weight_arena.c

### PURPOSE
Track 2 SC5 weight-arena fd custodian. Implements ioctl nrs 21-24
(REGISTER/IMPORT/LEAVE/QUERY). Cross-tenant weight sharing: a
producer tenant exports a CUDA VMM POSIX-fd shareable handle; the
kmod takes an independent `struct file *` reference; consumer
tenants `IMPORT` to receive a dup'd fd plus an opaque metadata
blob (the SC3 layout manifest + SC4 fingerprint). Per
[[cipher-track2-weight-sharing]] this is the SC5 design: the kmod
NEVER calls the CUDA driver — the driver's POSIX-fd handles are
already cross-process reference-counted; the kmod's job is the
rendezvous/custodianship.

### PUBLIC SURFACE
- `void cipher_wa_init(void)` — L368.
- `void cipher_wa_exit(void)` — L379.
- `long cipher_wa_ioctl_register(arg)` — L157 (nr 21).
- `long cipher_wa_ioctl_import(arg)` — L220 (nr 22).
- `long cipher_wa_ioctl_leave(arg)` — L281 (nr 23).
- `long cipher_wa_ioctl_query(arg)` — L309 (nr 24).
- `int cipher_wa_proc_show(seq_file *, void *)` — L341.

### CONTROL FLOW

**REGISTER (nr 21)** — L157-214:
1. `copy_from_user`, `blob_len ≤ CIPHER_WA_BLOB_MAX = 65536` (L166-167).
2. `fget(p.fd)` (L169) — claim a ref on the producer's fd; `-EBADF`
   if invalid.
3. Under `cipher_wa_lock`: find a free arena slot (16 max, L174-178).
   `-ENOSPC` if full (drop fget ref).
4. `copy_from_user` the blob into the arena slot (L184-189). Note
   the `fput(f)` on the copy_from_user error path (L187) — proper
   cleanup.
5. Assign `arena_id = atomic_inc_return(next_id)` (L190).
6. Populate: producer_pid = current->pid, base, size, fdfile=f,
   blob_len, in_use=1 (L191-197).
7. Unlock. `copy_to_user` the arena_id back.
8. **Atomicity**: if the final `copy_to_user` fails (L201-209),
   roll back the registration under the lock — `wa_release` (L118-124)
   `fput`s the file and zeroes `in_use`. Prevents an
   orphaned-but-live-fd arena.

**IMPORT (nr 22)** — L220-277:
1. `copy_from_user`.
2. Under lock: find arena by id (`-ENOENT` if not found). Find a
   free consumer slot (max 64 per arena, L235-243). `-ENOSPC`
   if full.
3. **Ordering** (per the comment L216-219): reserve the fd number
   first via `get_unused_fd_flags(O_CLOEXEC)` (L244) — does
   NOT yet install. This way an abort before install just
   `put_unused_fd`s a reserved (uninstalled) number, no fd to
   revoke.
4. Copy out the blob if any. `-ENOSPC` if caller buffer too small,
   `-EFAULT` on copy error. Both error paths `put_unused_fd`.
5. **Commit**: claim the consumer slot
   (`a->consumers[slot] = current->pid`), `get_file(a->fdfile)`
   (extra ref), `fd_install(newfd, a->fdfile)` (L263-265).
6. Fill `p.fd_out/base/size/blob_len`, unlock, `copy_to_user`.
7. If the `copy_to_user` at L272 fails: the fd IS installed; the
   caller's bad arg pointer means they get the fd but no `p` —
   the file is owned by the caller process now and the kernel
   close-on-exec will eventually reap it. (The code returns
   `-EFAULT` but does NOT undo the install; the install is
   technically transferred ownership.)

**LEAVE (nr 23)** — L281-306: caller is producer or consumer.
Under lock, clear `producer_pid` if it matches caller, clear any
consumer slot whose pid matches. If no live participants remain,
`wa_release` — `fput` the kmod's file ref.

**QUERY (nr 24)** — L309-337: lock + walk + emit (read-only).

### STATE
- `struct cipher_wa_arena cipher_wa_arenas[16]` (L67): 16 max
  arenas. Each carries one producer pid, 64 consumer pids, base/size,
  one `struct file *`, and a 64 KiB blob.
- `DEFINE_MUTEX(cipher_wa_lock)` (L68).
- `atomic_t cipher_wa_next_id` (L69) — monotonic arena ID.
- `struct delayed_work cipher_wa_reap_work` (L70) — 5s periodic
  reaper.

### CONCURRENCY
- Mutex-serialised hot path (REGISTER/IMPORT/LEAVE — these touch
  multiple arena fields).
- The reaper (`cipher_wa_reap_fn`, L128-152) also takes the mutex.
- `fget/fput/get_file/fd_install` are fd-table operations on the
  CALLER's task struct; the kernel's own locking governs them
  (fdtable RCU + spin).

### DEPENDENCIES
- Kernel `file`/`fs` (`fget`, `fput`, `get_unused_fd_flags`,
  `fd_install`, `put_unused_fd`, `get_file`).
- Kernel `pid`/`sched` (`find_get_pid`, `get_pid_task`,
  `put_task_struct`).
- Kernel `workqueue` (`schedule_delayed_work`, `cancel_delayed_work_sync`).
- INBOUND: `cipher_dev.c` L289-296 dispatches nrs 21-24;
  `cipher_proc.c` L456 delegates `/proc/cipher/arenas`.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The "kmod never calls CUDA" design rule is documented at L9-15
  and enforced (no CUDA symbol use anywhere in this file).
- The "off-do_exit by design" rationale (L29-33) is the SC5
  design memo §e — arena reclamation is not latency-critical
  (the physical is held safely by the fd reference meanwhile),
  unlike CP 5.4 SM groups, and keeping SC5 off the delicate
  do_exit path is a deliberate isolation choice.
- 5 s reap cadence (`REAP_SECS=5`, L53), 16 arenas, 64 consumers
  per arena, 64 KiB metadata blob max — all `static_assert`-able
  ABI constants.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for Track 2 weight sharing (the headline ~76% GPU-mem
saving on Mistral-7B N=4 in [[cipher-track2-weight-sharing]]).
Without this TU, the producer's exported fd dies with the producer.

### FUSION POINTS

- (1) Classifier consumption: same-model tenant grouping is a
  Track 2 prerequisite; the classifier could co-locate same-model
  PIDs to maximise arena hit rate.
- (2) Fusion class: PORT-AS-IS (ABI is stable, behaviour is
  custodianship, no policy decisions live here).

### State machine
Per-arena: EMPTY (in_use=0) → REGISTERED (producer alive +
fdfile held) → SHARED (consumers attached) → REAPED (no live
participants, fput → in_use=0). Linear lifecycle; no migration.

### Invariants
- Each arena holds **exactly one** `struct file *` ref while
  `in_use` (L118-122 in `wa_release`: `if (a->fdfile) fput; a->fdfile = NULL`).
- 16 arenas max (`CIPHER_WA_MAX_ARENAS`, ABI L469).
- 64 consumers per arena (`CIPHER_WA_MAX_CONSUMERS`, L52).
- 64 KiB metadata blob max (`CIPHER_WA_BLOB_MAX`, ABI L468).
- ABI rule: arena_id is monotonic (never reused) — `atomic_inc_return`
  (L190). No id wrap-around within a module lifetime (32-bit overflow
  at ~4 G arenas, well outside any realistic scenario).
- A failed `copy_to_user` in REGISTER rolls back the registration
  (L201-209) — no orphaned arena.

### ABI surface

ioctl nrs 21-24. Failure modes:
- **nr 21 REGISTER**: `-EFAULT`, `-EINVAL` (blob_len too big),
  `-EBADF` (bad fd), `-ENOSPC` (arena table full).
- **nr 22 IMPORT**: `-EFAULT`, `-ENOENT` (no such arena),
  `-ENOSPC` (consumer slots full or caller buffer too small),
  fd-table error from `get_unused_fd_flags`.
- **nr 23 LEAVE**: `-EFAULT`, `-ENOENT`.
- **nr 24 QUERY**: `-EFAULT` only.

### Concurrency under do_exit
**Crucially, this TU is NOT on the do_exit kprobe path.** The
SC5 design memo §e chose the 5-second polling reaper (L126-152)
deliberately — arena reclamation is not latency-critical because
the CUDA driver itself holds the physical memory while ANY process
fd points to it. A producer crashing leaves its `producer_pid` set
in the arena; the next reaper tick sees it via `wa_pid_alive` (L86-101)
returning false (the pid is no longer a live task), clears the
pid; if no other participants remain, `wa_release` fputs the file.

The fact that the kmod's `struct file *` keeps the cuMem
shareable handle alive even after the producer exits is the key
design property — Track 2's SC5 §1 "probe-confirmed" line in the
header at L9-13 documents it. **Cross-tenant access survives
producer crash**.

---

## Section /home/ubuntu/cipher_kmod/cipher_kvdedup.c

### PURPOSE
T4.6.4 cross-tenant KV-page dedup. Implements
`/dev/cipher_kvdedup` (separate chardev, magic `'K'`). The dedup
refcount table lives in kernel memory; tenant processes ioctl in.
Userspace computes the xxhash64 of the 2 MiB page and the
memcmp-verify on a hit; the kmod owns the table, the refcounts,
and the cuIpc POSIX-FD handles (`struct file *`).

### PUBLIC SURFACE
- `int cipher_kvdedup_init(void)` — L400.
- `void cipher_kvdedup_exit(void)` — L455.
- `static const struct file_operations kvd_fops` (L378-383):
  per-fd `open` (L316), `release` (L337), `unlocked_ioctl` (L361).
- `kvd_devnode` (L393-398): emits mode 0666 (matches
  `cipher_dev.c`'s pattern).

### CONTROL FLOW

**open** (L316-332): allocate `struct kvd_tenant *t`, init the
tracked-list head, mint a per-fd `tenant_id = kvd_next_tenant++`
(under mutex), stash in `file->private_data`, bump global
`kvd_tenants_open`.

**release** (L337-359) — runs on close() AND on process death
(SIGKILL included; the kernel guarantees the release fop). For
every page this tenant still tracks, `kvd_entry_deref` (L89-104)
the entry. If refcount hits 0 the entry is unlinked, removed from
the xarray, the kmod's `struct file *` is `fput` (which releases
the cuIpc handle), the entry is freed, and `kvd_releases`
incremented.

**PUT (nr 2)** — L152-244, the most interesting handler:

1. `copy_from_user`, validate fd.
2. `fget(p.export_fd)` (L165) — one ref, transferred to the table
   on MISS, dropped on HIT/error.
3. Default-zero the output fields (L169-171).
4. Under `kvd_lock`:
5. Look up by content_hash (L176-177; skipped if FORCE_NEW).
6. **HIT path** (L179-194):
   - `get_unused_fd_flags(O_CLOEXEC)` — reserve a new fd.
   - `get_file(e->handle_file)`, `fd_install(newfd, e->handle_file)`.
   - Result = HIT, candidate_fd = newfd, pool_offset = e->pool_offset.
   - **Do NOT bump refcount** — that is CONFIRM's job (the
     two-phase handshake means userspace memcmp-verifies first).
   - `kvd_hits++`.
7. **MISS path** (L194-235):
   - `kzalloc` a new `kvd_entry`.
   - `xa_alloc` allocates the `pool_offset` in [1, 2^20]. Failure
     → `-ENOSPC` (global entry cap, `CIPHER_KVDEDUP_MAX_ENTRIES=2^20`).
   - Initialise the entry, link into the bucket chain.
   - `kvd_track_add(t, id)` — append to this tenant's tracked
     list. May fail with `-ENOSPC` (`MAX_PER_TENANT=8192`) — if so,
     fully roll back the registration.
   - `exp_file = NULL` (L227) — ref transferred to the entry.
   - Update stats (entries, virtual_refs, misses, forced).
   - Result = MISS, pool_offset = id.
8. Unlock. If `exp_file` is still non-NULL (HIT path or error)
   `fput` it.
9. `copy_to_user`.

**CONFIRM (nr 3)** — L246-268: under mutex, `xa_load(pool_offset)`.
`-ENOENT` if missing or refcount==0. `kvd_track_add` (anti-spoof
via the tracking list) then `refcount++`, `kvd_virtual_refs++`.

**FREE (nr 4)** — L270-291: under mutex, `kvd_track_remove`
(`-EPERM` if not in tenant's track list — anti-spoof: a tenant
may only free a page it holds). Then `kvd_entry_deref` (which
may reach refcount 0 and trigger the physical release).

**STATS (nr 5)** — L293-312: under mutex, snapshot the global
counters; `tenants_open` is read lock-free as an atomic.

### STATE
- `kvd_buckets[65536]` — hlist hashtable, allocated via `kvcalloc`
  in init (L405). `KVD_BUCKETS=65536` (L36).
- `DEFINE_XARRAY_ALLOC1(kvd_xa)` (L62) — pool_offset → entry.
- `DEFINE_MUTEX(kvd_lock)` (L60).
- `static u32 kvd_next_tenant = 1` (L63) — tenant_id minter.
- Global counters (L66-68): `entries`, `virtual_refs`, `puts`,
  `hits`, `misses`, `collisions_forced`, `refcount_releases`,
  `kvd_tenants_open` (atomic).
- Per-fd: `struct kvd_tenant *` in `file->private_data` (L53-58).
- Per-entry: `struct kvd_entry` (L38-45) with content_hash,
  handle_file, refcount, pool_offset, hlist_node.
- Per-tracked-page: `struct kvd_track` (L47-51) with pool_offset
  + list_node.

### CONCURRENCY
- Single mutex `kvd_lock` for the whole hot path. The comment at
  L15-17 justifies this: "dedup is a per-request cold path; the
  bucket walk measured 33 ns; the handler does sleepable work
  (kzalloc, fd_install, copy_to_user) which a mutex permits cleanly".
- The xarray has its own internal locking but is wrapped by the
  outer mutex for the dual-structure (hlist + xarray) consistency.
- `fput/get_file/fd_install` are fd-table operations; kernel
  serialises per task.

### DEPENDENCIES
- Kernel `fs/cdev/device/class`, `xarray`, `hashtable`, `mutex`,
  `slab`, `uaccess`, `file`.
- INBOUND: opened by `cipher_rt_kv_alloc.c` (Wave 2 L481, per
  the Wave 2 anchors table).

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The two-phase PUT/CONFIRM contract (header L21-25) matches code
  exactly: PUT does not bump refcount on HIT; CONFIRM does.
- `MAX_PER_TENANT=8192` (header L82), `MAX_ENTRIES=2^20` (L84) —
  enforced at L112 and L205 respectively.
- The `.owner = THIS_MODULE` (L379) ensures `rmmod` returns
  `EBUSY` while any fd is open — the documented kvdedup_exit
  L460-461 expects an empty table.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for the KV-dedup substrate primitive (CP 4.6.5/6 per
[[cipher-cp4656-closed]]). 100-proc dedup scales; Llama-3.1-8B
real-decode ceiling 13 tenants.

### FUSION POINTS
- (1) Classifier consumption: dedup hit rates per workload are a
  weight-share signal. PORT-AS-IS.

### State machine

Per-fd: NEW (open) → ACTIVE (PUT/CONFIRM/FREE) → RELEASED (release fop).

Per-entry: NEW (PUT MISS, refcount=1, in xarray and bucket chain)
→ SHARED (CONFIRM raises refcount) → DEREFED (FREE or tenant
release drops refcount) → COLLECTED (refcount reaches 0; entry
unlinked, xarray erased, file fput'd, freed).

### Invariants
- `refcount > 0` for every entry in the xarray (asserted at L91-92,
  WARN_ON_ONCE on zero — a defensive check).
- A tenant may only `FREE` a page it holds (`kvd_track_remove`
  returns `-EPERM` otherwise, L138).
- `MAX_PER_TENANT=8192` (per-tenant cap; design memo C).
- `MAX_ENTRIES=2^20` (global cap; design memo G).
- A tenant's `tracked` list is exactly the set of pages whose
  refcount this tenant contributes 1 to. Symmetric: every track
  in the list corresponds to one increment in the entry's
  refcount, removed on `kvd_track_remove`.

### ABI surface
ioctl nrs 1-5 on `/dev/cipher_kvdedup`, magic `'K'`. Failure modes:
- **nr 1 INIT**: `-EFAULT`.
- **nr 2 PUT**: `-EFAULT`, `-EBADF`, `-ENOMEM`, `-ENOSPC`
  (global cap or per-tenant cap).
- **nr 3 CONFIRM**: `-EFAULT`, `-ENOENT`, `-ENOSPC`.
- **nr 4 FREE**: `-EFAULT`, `-EPERM` (anti-spoof).
- **nr 5 STATS**: `-EFAULT`.

### Concurrency under do_exit
**Not via the cipher_kmod kprobe.** Reclamation happens via the
kernel's standard `release` fop on the `/dev/cipher_kvdedup` fd,
which the kernel guarantees runs on process death — including
`SIGKILL`. This is the design property the header calls out at
L7-8: "The kernel outlives tenant processes and is notified of
tenant death via the release fop — that is what makes cross-tenant
resource ownership defensible."

Crash matrix:

| State at crash                  | After kvd_release                                        |
|----------------------------------|----------------------------------------------------------|
| No PUT/CONFIRM activity          | tenant struct freed; no entries touched                  |
| After MISS PUT                   | tracked list has the new entry; release derefs it (entry collected if last ref)  |
| Between HIT PUT and CONFIRM      | the HIT candidate_fd was installed in the dying process; kernel reaps it; no track entry was added, so release does not deref |
| After CONFIRM (refcount bumped)  | tracked list has the entry; release derefs (the +1 from CONFIRM is removed)     |
| After FREE                       | track entry already removed; no double-free in release   |

The HIT-without-CONFIRM crash window is the only place where
refcounts could (in theory) leak — but they don't, because PUT
on HIT does NOT bump the refcount; the kmod's view of the entry
is unchanged. The dying process's HIT fd is reaped by the kernel
(fd close); since that fd held a `get_file` ref on
`e->handle_file`, that ref is dropped by the kernel — but the
kmod's own ref on `e->handle_file` (taken at MISS or transferred
from a previous PUT) is preserved. Net: zero leak.

---

## Section /home/ubuntu/cipher_kmod/cipher_state_updater.c

### PURPOSE
Phase 4.1 derived-state computation kthread. Runs at 1 kHz. For
each `cipher_pid_stats` entry, computes derived fields from
`cipher_gpu_state` (Phase 3 device-wide telemetry):
`thermal_headroom_pct`, `power_headroom_w`, `sustained_clock_mhz`
(95/5 EMA), `voltage_envelope_mv` (coarse linear model). Also
calls `cipher_partition_tick()` each loop (which internally
gates to 5 s rebalance).

### PUBLIC SURFACE
- `int cipher_state_updater_init(void)` — L185.
- `void cipher_state_updater_exit(void)` — L206.

### CONTROL FLOW
`cipher_state_updater_fn` (L139-183) is the kthread body:
1. Snapshot `cipher_gpu_state` under its spinlock (L153-155).
2. Skip if stale (`timestamp_jiffies == 0` or `> 1 s` old, L162-165).
3. RCU walk: per-entry `cipher_update_tenant_derived` (L168-172).
4. `cipher_partition_tick()` (L176).
5. `msleep_interruptible(1)` — 1 kHz cadence.
6. Loop until `kthread_should_stop()`.

### STATE
- `static struct task_struct *cipher_state_updater_task` (L43).
- `static atomic_t cipher_state_updater_running` (L44).

### CONCURRENCY
- The kthread is a single thread; uses RCU read-side for the
  hashtable walk.
- All writes to `cipher_pid_stats` fields are `WRITE_ONCE`;
  readers (`cipher_proc.c`, `cipher_tenant_snapshot.c`) use
  `READ_ONCE`.

### DEPENDENCIES
- INBOUND: `cipher_main.c::cipher_init` L81; `cipher_exit` L106.
- OUTBOUND: `cipher_pid_table`, `cipher_gpu_state`,
  `cipher_partition_tick`.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- **STALE HEADER COMMENT**: L5-7 says
  "STAGED: this file is NOT in Kbuild's cipher_kmod-y for
  cipher_kmod 0.3.1. Gets wired into the live build in T4.1.6
  when the module bumps to 0.4.0." But Kbuild L14 shows
  `cipher_state_updater.o` IS compiled in 0.4.8. Header comment
  is stale; the code is the truth.
- The 95/5 EMA coefficients (L36-37) are hard-coded.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for derived fields (thermal, power, voltage
envelope, sustained clock). `cipher_main.c` L81-86 tolerates
kthread init failure — derived fields stay zero, readers
tolerate.

### FUSION POINTS
- (1) Classifier hook: the kthread is the right place to invoke
  classifier-side per-tenant feature derivation. REFACTOR (small).

### State machine
Two-state task lifecycle: NOT_RUNNING ↔ RUNNING, atomic guard
via `cipher_state_updater_running` (L189) prevents double init.

### Invariants
- Single kthread (`atomic_xchg` guard at L189).
- 1 kHz cadence — `msleep_interruptible(1)`. Bound by `HZ`
  tick resolution: on this pod `CONFIG_HZ=1000`
  (`/boot/config-6.8.0-1046-nvidia`), so 1 ms `msleep` does
  resolve to ~1 ms / 1 kHz as the comment intends. On a
  HZ_250 kernel the cadence would degrade to ~250 Hz; the
  Wave-3 audit deployment runs HZ_1000 so this hypothetical
  does not apply here.
- All writes WRITE_ONCE single fields; no struct-wide locking.

### ABI surface
None directly.

### Concurrency under do_exit
The kthread does not run in the dying task's context; it cleans
up via `kthread_stop` in `cipher_state_updater_exit` (L208-209),
which signals the kthread to exit on its next `kthread_should_stop`
check (the cadence is 1 ms so worst-case unload delay is ~1 ms).

---

## Section /home/ubuntu/cipher_kmod/cipher_tenant_snapshot.c

### PURPOSE
Phase 4.1 contract implementation. Provides three lookup
functions consumed by ioctl nr 8, the prometheus exporter, and
in-kmod consumers (state updater, partition allocator). Plus
partition-mask write-back paths for nr-9-style writes (now
deactivated but the function still exists).

### PUBLIC SURFACE
- `cipher_get_current_tenant_snapshot(void)` — L124, EXPORT_SYMBOL_GPL.
- `cipher_get_tenant_snapshot_by_pid(pid_t)` — L140, EXPORT_SYMBOL_GPL.
- `cipher_get_tenant_snapshot_by_id(const char *)` — L157, EXPORT_SYMBOL_GPL.
- `cipher_enumerate_tenants(*out, max)` — L185, EXPORT_SYMBOL_GPL.
- `cipher_set_sm_partition_mask(pid_t, u32)` — L207.
- `cipher_get_sm_partition_mask(pid_t, u32 *)` — L224.

### CONTROL FLOW
**Per-CPU thread-local snapshot** (`DEFINE_PER_CPU`, L44). Each
read-side caller does:
1. `cipher_pid_lookup_rcu_local(pid)` (L112).
2. `this_cpu_ptr(&cipher_snapshot_tls)`.
3. `cipher_assemble_snapshot(out, e)` (L51-108) — copies fields
   from the cipher_pid_stats entry into the per-CPU TLS slot.
4. Return the per-CPU pointer.

The assembly is straight `READ_ONCE`s on individual fields
(L52-108) — no locking. Cross-field consistency is weak;
readers tolerate.

`cipher_enumerate_tenants` (L185-202) does the same per entry,
writing into the caller-provided array under one RCU section.

`cipher_set_sm_partition_mask` (L207-222): RCU-find then
`WRITE_ONCE(sm_partition_mask, mask)` + `WRITE_ONCE(sm_partition_count,
hweight32(mask))`. `-ESRCH` if not found.

### STATE
- `DEFINE_PER_CPU(struct cipher_tenant_snapshot, cipher_snapshot_tls)` (L44).
- No global state beyond per-CPU.

### CONCURRENCY
- Caller MUST hold `rcu_read_lock()` (the header at
  `cipher_internal.h` L288-291 documents this).
- The per-CPU TLS slot is reused across reads by tasks running
  on that CPU; comment L41-43 explains the preemption story (no
  preemption migration mid-read on PREEMPT_RCU; classic RCU gets
  the same via the implicit non-preempt section).

### DEPENDENCIES
- INBOUND: `cipher_dev.c::cipher_dev_get_tenant_snapshot` (nr 8) via
  `cipher_get_tenant_snapshot_by_pid/by_id`.
- OUTBOUND: `cipher_pid_table` (RCU walk).

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- Target "< 200 ns" for assembly + bucket walk (L20) — not
  measured in source, but the design is reasonable: ~30 field
  reads × ~3 ns/`READ_ONCE` = ~90 ns assembly, plus the bucket
  walk.
- The size comment at `cipher_internal.h` L174 mentions a
  `static_assert` "verifies at < 384 B" — but a `grep -n
  static_assert` over both files shows the only mention is the
  comment itself. The intended assert was not emitted. The
  layout claim (336 B target, 384 B ceiling) is therefore
  documentary only, not compile-time enforced. Minor; the
  layout is unchanged in practice because no edit has bumped
  the size.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for ioctl nr 8 and any kmod-internal consumer of
the per-tenant view.

### FUSION POINTS
- (1) Classifier consumption: this IS the cross-process per-tenant
  query API the classifier needs. PORT-AS-IS.

### State machine: per-call assembly; no persistent state.
### Invariants
- Caller holds RCU; returned pointer valid only within section.
- The per-CPU TLS slot's contents are valid only for the most
  recent assembly on that CPU.

### ABI surface
None directly; supports nr 8 via `cipher_dev.c`.

### Concurrency under do_exit
A dying tenant's `cipher_pid_stats` entry remains accessible
within an RCU section until `kfree_rcu` completes. A reader
that began its lookup before `hash_del_rcu` finishes its
RCU grace period sees the entry; a reader that begins after
does not. No torn snapshots.

---

## Section /home/ubuntu/cipher_kmod/probe_microbench.c

### PURPOSE
Userspace microbench. Tight loop calling
`NV_ESC_CHECK_VERSION_STR` through `/dev/nvidiactl` to measure
the kprobe pre-handler overhead. **NOT compiled into the
module** — verified by inspecting `Kbuild` L7-20 (not in the
`cipher_kmod-y` list). Builds standalone via the `gcc` one-liner
in the header (L10).

### PUBLIC SURFACE (userspace binary)
- `main(int argc, char **argv)` — L40. Optional argv[1] = iteration
  count (default 100000).

### CONTROL FLOW
1. Open `/dev/nvidiactl` (L56).
2. 200-iteration warmup (L66-67).
3. `clock_gettime(MONOTONIC)` → N iterations → `clock_gettime`
   (L69-73).
4. Print one JSON line: `iterations, total_ns, ns_per_ioctl` (L78).

### CONTRIBUTION TO SYSTEM
Diagnostic — not part of the running module. Used historically
for Phase 1.5.2 Workload D measurements (per the header L3).

### FUSION POINTS
None. Standalone bench.

### State, concurrency, ABI, do_exit: n/a — userspace single-threaded.

---

## Module load/unload ordering

Reading `cipher_main.c` L25-101 and L103-120 together produces
the canonical sequence:

**Load (cipher_init, L25-101):**

1. `cipher_decode_self_check` — abort if drift.
2. `spin_lock_init` × 2, set load_jiffies.
3. `cipher_partition_allocator_init` — legacy allocator (cannot fail).
4. `cipher_cp54_sched_init` — CP 5.4 ledger (cannot fail).
5. `cipher_wa_init` — Track 2 SC5 (cannot fail).
6. `cipher_flops_init` — CP 3.3 ring (cannot fail).
7. `cipher_dev_init` — /dev/cipher (fatal on fail).
8. `cipher_kvdedup_init` — /dev/cipher_kvdedup (fatal on fail).
9. `cipher_bar0_init` — `-ENODEV` is tolerated (no GPU); else fatal.
10. `cipher_proc_init` — /proc/cipher/* (fatal).
11. `cipher_probe_init` — register kprobes (fatal).
12. `cipher_state_updater_init` — kthread (best-effort).

**Unload (cipher_exit, L103-120):**

1. `cipher_state_updater_exit` — stops kthread.
2. `cipher_flops_exit`.
3. `cipher_partition_allocator_exit`.
4. `cipher_probe_exit` — unregisters kprobes; drains RCU; frees
   per-PID table (`synchronize_rcu` + `rcu_barrier`).
5. `cipher_cp54_sched_exit` — **must follow `cipher_probe_exit`**.
   The comment at L110-112 is explicit: a process exiting
   mid-`rmmod` could fire `cipher_cp54_release` against torn-down
   state if the order were reversed. **HARD ORDERING INVARIANT.**
6. `cipher_wa_exit` — cancels reaper, fputs held fds.
7. `cipher_proc_exit`.
8. `cipher_bar0_exit` — `pci_iounmap`, deliberately leaks
   `pci_dev` ref ([[cipher-incident-2-bar0_exit]]).
9. `cipher_kvdedup_exit`.
10. `cipher_dev_exit`.

The ordering rule: anything that the do_exit kprobe can call
(CP 5.4, partition allocator) must remain valid through
`cipher_probe_exit`. Conversely, anything that can be torn down
without affecting the kprobe (state_updater, flops, proc, BAR0,
kvdedup, weight-arena, dev) can be torn down before or after as
long as their own internal invariants hold.

---

## Specific anchors — verified or refuted

| Claim                                                                                                                                                | Verdict   | Cite                                                                                |
|------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|-------------------------------------------------------------------------------------|
| Module version 0.4.8                                                                                                                                  | VERIFIED  | cipher_main.c L128                                                                  |
| 14 .c files in cipher_kmod-y; probe_microbench.c NOT compiled into the module                                                                          | VERIFIED  | Kbuild L7-20 vs probe_microbench.c L10 (`gcc -O2`)                                  |
| /dev/cipher uses magic 'C'; /dev/cipher_kvdedup uses magic 'K'                                                                                         | VERIFIED  | cipher_ioctl.h L19; cipher_kvdedup.h L18                                            |
| Phase 6 reserved nrs 2/3/4 return -ENOSYS                                                                                                              | VERIFIED  | cipher_dev.c L297-300                                                               |
| Phase 4.2 nr 9 REQUEST_SM_PARTITION returns -ENOSYS (deactivated by CP 5.4)                                                                            | VERIFIED  | cipher_dev.c L107-113                                                               |
| 24 ABI nrs assigned (1, 2-4 reserved, 5-12, 13-15, 16-20, 21-24)                                                                                       | VERIFIED  | cipher_ioctl.h + cipher_dev.c dispatch                                              |
| CP 5.4 = 15 × 8-SM groups (not 16)                                                                                                                     | VERIFIED  | cipher_cp54_sched.c L59-62                                                          |
| CIPHER_CP54_MAX_ALLOCS = 64                                                                                                                            | VERIFIED  | cipher_cp54_sched.c L123                                                            |
| MIG_TIMEOUT_NS = 30s                                                                                                                                   | VERIFIED  | cipher_cp54_sched.c L93                                                             |
| Default policy: gap_min=1, sustain=2000ms, ratelimit=10000ms, verbose=1                                                                                | VERIFIED  | cipher_cp54_sched.c L95-98                                                          |
| Track 3 SC2 group encoding: bit31=OCC, bit30=RSVD, bits0..29=pid; PID_MAX_LIMIT 2^22 → 256× headroom                                                    | VERIFIED  | cipher_cp54_sched.c L70-80                                                          |
| `cp54_release_groups` clears BOTH PACK and RSVD in one sweep (do_exit-safe)                                                                            | VERIFIED  | cipher_cp54_sched.c L196-217                                                        |
| `cp54_commit_migration` is a two-pass cmpxchg (clear PACK then RSVD→PACK)                                                                              | VERIFIED  | cipher_cp54_sched.c L383-398                                                        |
| Kmod never aborts MIGRATING; only tenant via ACK(0) does                                                                                                | VERIFIED  | cipher_cp54_sched.c L416-419, L773-792                                              |
| cipher_cp54_release(pid) is lock-free, atomic-context safe                                                                                              | VERIFIED  | cipher_cp54_sched.c L342-363, called at cipher_probe.c L198                         |
| Legacy partition_allocator: SLOTS_MAX=32 (B6 cap), HINT_CAP=8, IDLE_NS=30s, REBALANCE=5s                                                                | VERIFIED  | cipher_partition_allocator.c L56-66                                                 |
| FIT_HINT (B10) implements release-down to N, including hint=0 release-all                                                                              | VERIFIED  | cipher_partition_allocator.c L147-254                                               |
| kvdedup: MAX_PER_TENANT=8192, MAX_ENTRIES=2^20, BUCKETS=65536                                                                                          | VERIFIED  | cipher_kvdedup.h L82-84, cipher_kvdedup.c L36                                       |
| kvdedup PUT/CONFIRM is a two-phase handshake; PUT does NOT bump refcount on HIT                                                                        | VERIFIED  | cipher_kvdedup.c L179-194 + L246-268                                                |
| kvdedup release fop runs on SIGKILL; derefs every tracked page                                                                                          | VERIFIED  | cipher_kvdedup.c L334-359                                                           |
| weight-arena: 16 arenas × 64 consumers × 64 KiB blob; off-do_exit reaper at 5s                                                                          | VERIFIED  | cipher_weight_arena.c L52-53, ABI L469, init L373                                   |
| weight-arena REGISTER rolls back on copy_to_user failure (no orphaned arena)                                                                            | VERIFIED  | cipher_weight_arena.c L201-209                                                       |
| weight-arena IMPORT reserves fd via get_unused_fd_flags before install                                                                                  | VERIFIED  | cipher_weight_arena.c L244, L263-265                                                |
| clock: [210, 1980] MHz, 0 = unlock, no per-ioctl CAP_SYS_ADMIN by design                                                                                | VERIFIED  | cipher_clock.c L53-54, L138-143                                                     |
| FLOP: RING_SIZE=256, STALE=2*HZ, PEAK=989 TFLOP/s, MFU_DIVISOR=9.89e9                                                                                  | VERIFIED  | cipher_flops.c L42-48                                                               |
| decode table: COUNT=24, OTHER_SLOT=24, TOTAL_SLOT=25                                                                                                    | VERIFIED  | cipher_internal.h L32-35                                                            |
| PID_HASH_BITS=10 (1024 buckets)                                                                                                                        | VERIFIED  | cipher_internal.h L73-74                                                            |
| BAR0 layer is READ-ONLY at the type level (`const void __iomem *`)                                                                                     | VERIFIED  | cipher_bar0.c L57-60                                                                |
| BAR0 exit deliberately leaks pci_dev ref (Incident 2)                                                                                                  | VERIFIED  | cipher_bar0.c L132-141                                                              |
| devnode codifies mode 0666 in cipher_dev and cipher_kvdedup                                                                                            | VERIFIED  | cipher_dev.c L325-330 + cipher_kvdedup.c L393-398                                   |
| Anti-spoof: REGISTER_TENANT, SUBMIT_LAUNCH_STATS verify caller pid/tgid                                                                                | VERIFIED  | cipher_dev.c L70, L237                                                              |
| CAP_SYS_ADMIN on SUBMIT_GPU_STATE, SUBMIT_PROCESS_UTIL, SUBMIT_FLOP_SAMPLE                                                                              | VERIFIED  | cipher_dev.c L166-167, L201-202; cipher_flops.c L98-99                              |
| Unload ordering: cp54_sched_exit MUST follow probe_exit                                                                                                | VERIFIED  | cipher_main.c L110-113                                                              |

## Drift / stale-comment register

1. **cipher_state_updater.c L5-7** — header says "STAGED: not in
   Kbuild for cipher_kmod 0.3.1" but Kbuild L14 has it. Stale
   comment, code is truth.
2. **cipher_partition_allocator.c L33-37** — driver list mentions
   `cipher_dev_request_sm_partition` which is now `-ENOSYS`.
   Stale relative to CP 5.4.
3. **cipher_bar0.c L154** — emits `v0.3.1` while module is at 0.4.8.
   Cosmetic.
4. **cipher_main.c L39** — init banner mentions "Phase 4.2 T4.2.1
   lock-free SM partition allocator"; CP 5.4 is the current.
   Cosmetic.
5. **cipher_cp54_sched.c L24-31** — code self-discloses memo-vs-build
   drift on the "lock-free fast path" for ALLOCATE. Honest in-source
   drift, flagged for Step 1.2 adjudication.
6. **cipher_kmod.mod.c** — auto-generated by kbuild, deliberately
   ignored per the prompt.
7. **MODULE_DESCRIPTION (cipher_main.c L127)** does not mention CP
   5.4, Track 2, or Track 3 — only "GPU-state spine + BAR0 reads +
   CP3.3 FLOP telemetry". Cosmetic.

## v1 fusion summary (Wave 3 kmod)

| File                              | Action               | Severity            | Rationale                                                                                  |
|-----------------------------------|----------------------|---------------------|--------------------------------------------------------------------------------------------|
| `cipher_main.c`                   | REFACTOR (small)     | minor               | One line for `cipher_classify_init` between state-updater and probe init; banner cosmetic. |
| `cipher_internal.h`               | REFACTOR (small)     | minor               | Add classifier fields to the snapshot reserved tail.                                       |
| `cipher_ioctl.h`                  | REFACTOR (small)     | minor               | Add nr 25+ for classifier ABI; preserve all existing nrs.                                  |
| `cipher_kvdedup.h`                | PORT-AS-IS           | none                | Stable ABI.                                                                                |
| `cipher_ioctl_decode.c`           | PORT-AS-IS           | none                | Decoder is workload-agnostic.                                                              |
| `cipher_dev.c`                    | PORT-AS-IS + SHIM    | minor               | One new switch case for nr 25+; existing handlers unchanged.                               |
| `cipher_probe.c`                  | PORT-AS-IS + SHIM    | minor               | One line in kprobe_pre to feed classifier; do_exit reaper extensible.                      |
| `cipher_proc.c`                   | PORT-AS-IS           | none                | Add a new proc entry alongside existing six.                                               |
| `cipher_bar0.c`                   | PORT-AS-IS           | none                | Read-only; pci_dev leak is documented and intentional.                                     |
| `cipher_clock.c`                  | PORT-AS-IS           | none                | nr 10 is stable; bounds + mutex + audit log all correct.                                   |
| `cipher_flops.c`                  | PORT-AS-IS           | none                | nrs 11/12 are stable; CP 3.3 telemetry is gate-sufficient.                                 |
| `cipher_cp54_sched.c`             | PORT-AS-IS           | none                | The arbitration substrate is sound; SC2/SC4 already operator-tunable.                      |
| `cipher_partition_allocator.c`    | REFACTOR (retire)    | minor (deferred)    | Userspace dead; housekeeping live. Retire cleanly when migrating state-updater hooks.       |
| `cipher_weight_arena.c`           | PORT-AS-IS           | none                | Custodian discipline; off-do_exit reaper.                                                  |
| `cipher_kvdedup.c`                | PORT-AS-IS           | none                | Two-phase handshake + release-fop reclamation is robust.                                   |
| `cipher_state_updater.c`          | REFACTOR (small)     | minor               | Add classifier-side per-tenant derivation hook in the 1 kHz loop.                          |
| `cipher_tenant_snapshot.c`        | PORT-AS-IS           | none                | This is already the per-tenant query API.                                                  |
| `probe_microbench.c`              | n/a                  | n/a                 | Userspace bench; not compiled.                                                             |

## The load-bearing fact (Wave 3)

The deployed kmod gives the classifier brain port:

- **A per-tenant kernel-side data structure** — `cipher_pid_stats`
  (cipher_internal.h L103-165) — 600 B, RCU-walkable, mutated
  by Phase 3 telemetry + Phase 4 actuators + identity bridge.
- **A cross-process query API** — ioctl nr 8 + the four
  `cipher_get_tenant_snapshot_*` functions
  (`cipher_tenant_snapshot.c`).
- **A do_exit reaper that releases per-tenant resources** —
  `cipher_probe.c::cipher_do_exit_pre` (L184-221) — CP 5.4 +
  partition slots + the cipher_pid_stats entry itself.
- **An arbitration substrate** — CP 5.4 ledger
  (`cipher_cp54_sched.c`) with PARTITION/SHARED/POOL classes,
  count-preserving Dynamic SM Migration, operator-tunable policy,
  conservation under crash.
- **Two privileged actuation paths** — VOLT clock-set (nr 10)
  and FLOP telemetry submission (nr 11).
- **Two custodian char devices** — `/dev/cipher` (24 nrs) and
  `/dev/cipher_kvdedup` (5 nrs), both 0666-codified via devnode
  callbacks.

What is missing for fusion:

- **No classifier observer** in the kprobe pre-handler — needs
  one line in `cipher_kprobe_pre` (`cipher_probe.c` L78-111).
- **No classifier output ioctl** — needs new nr 25+.
- **No per-tenant classifier-decision field** — needs additions
  in the snapshot reserved tail.
- **State-updater does not consult classifier** — needs one
  per-entry function call in the 1 kHz loop.

The honest engineering posture for v1: port the classifier brain
as a per-tenant observer that reads `cipher_pid_stats`/snapshots
and writes a `recommended_*` field tail; the kmod substrate is
already complete enough that no refactor of the existing
arbitration logic is required. The 30-bit-pid encoding gives 256×
PID headroom; the count-preserving DSM gives crash safety; the
custodial char-dev design pattern (weight-arena, kvdedup)
generalises to any future kernel-mediated cross-tenant resource.
