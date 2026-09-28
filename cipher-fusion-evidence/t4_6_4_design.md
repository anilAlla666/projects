# T4.6.4 — kmod-owned cross-tenant dedup — DESIGN MEMO

Pre-build memo. No C until approved. Based on the STEP 1 audit of
`cipher_kmod/` (cipher_dev.c, Kbuild, locking/memory conventions),
`cipher_rt_kv_alloc.c`, the April cuIpc probe, and the nvidia.ko
release-fop pattern.

## Scope (locked by STEP 1 calls A & B)

Move the T4.6.3 dedup refcount table from process-local `cipher_rt`
memory into `cipher_kmod`. New char device `/dev/cipher_kvdedup`,
per-fd tenant state, release-fop teardown, POSIX-FD cuIpc. **Dedup
stays a separate path — `slab_free` is NOT retrofitted** (call A;
the slab-merge question is deferred to T4.6.5). Userspace
`cipher_rt_kv_dedup_{put,free}` keep their signatures; their bodies
become ioctl shims.

## Call model — what the latency budget assumes

T4.6.4's latency budget assumes **model (a): the cold path** —
`cipher_rt_kv_dedup_put` is called **once per slab page at allocation
time**, not per-request and not per-decode-step. Pages allocate
seconds-to-minutes apart per tenant, so the ~400 µs hit-path cost
(`cuMemImportFromShareableHandle` + 1 DtoH + `memcmp`) is amortized to
nothing. This is why the mutex, the 1-2 syscall/page surface, and the
synchronous memcmp-verify are all acceptable.

T4.6.4 ships the *mechanism*; it does not commit to a call model.
**T4.6.5 decides which model it measures.** If T4.6.5 picks model (b)
— `dedup_put` on the live per-request / per-decode KV write path —
then ~400 µs/hit becomes a measurable TTFT / throughput tax and the
mechanism needs a latency pass (constant-time path, async import,
batched verify). That is a T4.6.5 finding, out of T4.6.4 scope. A
future reader: the numbers below assume (a).

## A. ioctl interface — `/dev/cipher_kvdedup`, magic `'K'`

`open()` allocates `struct kvdedup_tenant` into `file->private_data`
and assigns a monotonic `tenant_id`. 5 ioctls (fixed-layout structs,
explicit `__u32/__u64`, padded):

```
INIT    _IOR ('K',1, struct {u32 tenant_id; u32 _pad;})           -> tenant_id
PUT     _IOWR('K',2, struct {u64 content_hash;  /* xxh64, userspace-computed */
                             s32 export_fd;     /* cuMem POSIX fd of caller's page */
                             u32 flags;         /* bit0 FORCE_NEW */
                             u32 result;        /* out: 0 MISS_REGISTERED / 1 HIT_CANDIDATE */
                             u64 pool_offset;   /* out: entry id */
                             s32 candidate_fd;  /* out: HIT only, fresh fd to verify; else -1 */
                             u32 _pad;})
CONFIRM _IOW ('K',3, struct {u64 pool_offset;})  /* after a HIT memcmp matches */
FREE    _IOW ('K',4, struct {u64 pool_offset;})
STATS   _IOR ('K',5, struct cipher_kvdedup_stats)
```

Flow — **hashing and memcmp-verify stay in userspace** (prompt A):
- userspace `cipher_rt` `cuMemCreate`s a page, writes content, computes
  xxh64, `cuMemExportToShareableHandle(POSIX_FD)` → `export_fd`.
- `PUT`: kmod walks the bucket. **MISS** → `fget(export_fd)`, new entry
  refcount=1, append to tenant tracking list, return `MISS_REGISTERED`
  + `pool_offset`. **HIT** (hash match) → kmod does *not* take
  `export_fd`; it `fd_install`s a fresh `candidate_fd` to the existing
  entry's stored handle and returns `HIT_CANDIDATE` + `pool_offset`.
- userspace imports `candidate_fd`, `memcmp`-verifies vs its content.
  Match → `CONFIRM(pool_offset)` (kmod bumps refcount + tracks); frees
  its own redundant page. Mismatch (collision) → `PUT` again with
  `FORCE_NEW` (kmod skips the walk, registers fresh).
- `FREE(pool_offset)`: kmod verifies the offset is in **this fd's**
  tracking list (anti-spoof), removes it, decrements refcount, releases
  the handle (`fput`) at 0.

Syscall surface: **1 ioctl on a miss, 2 on a hit/collision** — minimal
given memcmp-verify is userspace-side; dedup is a per-request op (not
per-token), so 1-2 syscalls/page is not chatty. A read-only
`/proc/cipher/kvdedup_stats` mirrors `STATS` (indicators b/d read it).

## B. Locking — **mutex** (measured)

Bucket-walk microbench (`t4_6_4_bucketwalk_bench.c`, 65536 buckets,
150K entries, load 2.29, max chain 10): **32.9 ns/lookup**. The bare
critical section (walk + refcount + list op) is well under 1 µs — by
the prompt's ">10 µs ⇒ mutex" rule that alone permits a spinlock.

**Choice: mutex** (`DEFINE_MUTEX`, matches the kmod's existing
cold-path convention — `mutex_lock_interruptible` for the clock ioctl).
Reason: `PUT`/`FREE` are per-request cold paths whose handlers do
*sleepable* work — `kmalloc(GFP_KERNEL)`, `get_unused_fd_flags`,
`fd_install`, `copy_{from,to}_user`. A spinlock would force `GFP_ATOMIC`
+ a speculative-prealloc dance for zero benefit on a cold path. The
mutex lets the handler be written straightforwardly; sleepable work
stays *outside* the lock regardless (entry `kmalloc`'d before locking;
`copy_to_user`/`fd_install` after unlocking). Hold time under the
mutex ≈ the 33 ns walk + a few list ops.

## C. Per-tenant tracking list

`struct kvdedup_tenant` (in `file->private_data`) holds a linked list
of small nodes `{u64 pool_offset; list_head}` — one per page this fd
put-or-confirmed. **Bound N = 8192 pages/tenant** (16 GiB of KV;
covers 128K context: 32 layers × 2 roles × 128 pages). The 8193rd
registration returns **`-ENOSPC`**; userspace falls back to a
non-deduped local page (see F-3). Nodes are `kmalloc`'d (memory scales
with actual use, capped at N). `release()` walks this list and
decrements every entry — the teardown guarantee.

## D. ~~slab_free retrofit~~ — REMOVED

Per call A, dedup stays a separate path; `slab_free` owns only
exclusively-held plain slab pages and is **not** modified. The op #1
allocator unit test stays **14/14 unchanged** — indicator (c) is
"verify unchanged," not "evolve."

## E. cuIpc handle storage — POSIX-FD (verified path)

April probe used legacy `cuIpcGetMemHandle` (works on `cuMemAlloc`,
not VMM). T4.6.2 S1 verified the VMM path:
`cuMemExportToShareableHandle(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)`
returns a valid fd on this driver (580.105.08 / CUDA 13). FABRIC
untested / not needed (single node). **POSIX-FD chosen.**

The bucket entry stores `struct file *handle_file` — the kmod's
`fget` reference to the exporter's fd. Holding this `file*` keeps the
physical GPU allocation alive across the exporter's death (the
shareable handle is a reference to the allocation) — this is the
mechanism that makes the kmod the cross-tenant owner. On a HIT the
kmod hands a requester a fresh fd: `nfd = get_unused_fd_flags(O_CLOEXEC);
get_file(handle_file); fd_install(nfd, handle_file)`. The entry's
`fput(handle_file)` happens when refcount reaches 0. (Indicator (b)
empirically confirms the file* ref keeps B's mapping valid after A
dies.)

## F. Failure modes

Note: you said "five (minus the slab_free one)" — but section D (the
slab_free *retrofit*) is a memo section, not one of the six runtime
failure modes; none of the six below is slab-free-related. The
discipline says the six-mode analysis is not optional, so **all six
are kept**. If you meant a specific item dropped, name it.

1. **SIGKILL during the kmod mutex bucket walk.** Not a problem: the
   mutex is acquired *and* released within one ioctl syscall by the
   handler running in the tenant's own context. A thread executing a
   non-sleeping in-kernel critical section runs to a safe point;
   SIGKILL is delivered at a signal check, never mid-critical-section.
   The mutex is never "held by a dead process."
2. **SIGKILL during cuIpc import on a HIT.** The kmod gave a
   `candidate_fd` but the tenant died before `CONFIRM`. Refcount was
   *not* bumped (bump is on CONFIRM); the page is *not* in the dead
   tenant's tracking list (add is on CONFIRM/MISS-register). Kernel
   closes the dead fd → its `file*` ref drops, no leak. `release()`
   finds nothing to decrement. Correct, no action needed.
3. **Tracking-list overflow.** 8193rd page → `-ENOSPC`. Userspace
   fallback: `cipher_rt_kv_dedup_put` keeps the local page it created,
   does not register it, returns it un-deduped (`out_devptr` = the
   local page). Correctness preserved, dedup simply skipped for that
   page.
4. **kmod memory pressure.** `kmalloc` fails for a bucket entry or
   tracking node → `-ENOMEM` → same userspace fallback as F-3.
5. **Module unload with tenants holding mappings.** `file_operations`
   `.owner = THIS_MODULE` → the kernel auto-holds a module reference
   while any `/dev/cipher_kvdedup` fd is open → `rmmod` returns
   **`EBUSY`** by construction. No explicit `try_module_get` needed.
6. **Hash collision during PUT.** xxh64 matched but userspace `memcmp`
   differs → userspace re-issues `PUT` with `FORCE_NEW` → kmod skips
   the bucket walk and registers a fresh entry. Collision degrades to
   a missed dedup, never corruption.

## G. Security boundary / threat model

The `/dev/cipher_kvdedup` boundary is the cross-tenant security
boundary.

- **A cannot read B's content it doesn't already have.** Dedup only
  ever shares pages of *byte-identical* content, gated by A's own
  userspace `memcmp` against content A already possesses. A only
  receives a handle for a page whose content A already holds — sharing
  leaks nothing.
- **A cannot forge a decrement for B's pages.** `FREE`/`CONFIRM` verify
  `pool_offset` is in the *issuing fd's* tracking list (`-EPERM`
  otherwise) — mirrors `cipher_dev_register_tenant`'s `pid` anti-spoof.
- **A cannot exhaust kmod memory.** Per-tenant N=8192 cap + a global
  entry cap (`CIPHER_KVDEDUP_MAX_ENTRIES`, propose 2^20 ≈ 2 TiB of
  tracked KV); over cap → `-ENOSPC`.
- **Residual risk — timing side channel.** `PUT` latency differs
  hit vs miss; a tenant that already possesses a candidate page can
  `PUT` it and learn from the response whether another tenant has
  byte-identical content (e.g. a specific system prompt). This is a
  *confirmation oracle* (A must already hold the content to probe),
  not content exfiltration — but it is **a real residual risk.
  Accepted and documented; Phase 7 retires it** (constant-time PUT
  path or response-time noise). Not hand-waved.

## H. Module loading

- `/dev/cipher_kvdedup` is created by `class_create` + `device_create`
  (second device alongside `/dev/cipher`); **udev auto-creates the
  node** — no manual `mknod`.
- Permissions: cross-tenant → tighter than `/dev/cipher`'s 0666.
  Start **0660, group `cipher`**, set via an operator udev rule
  (operator policy).
- Reload: `rmmod` is blocked while any tenant fd is open (F-5), so a
  reload requires all tenants to disconnect first — there is no
  partial-state hazard. `module_exit` frees all dedup state;
  re-`insmod` starts clean. (Persisting state across reload while
  tenants are live is a Phase 7 item, not needed here.)

## Five binding indicators (Phase 4 closure gate)

(a) cross-process byte-correct — 15 T4.6.3 windows replayed across 2
processes + 1 explicit cross-process test, 16 runs, strict SHA-256.
(b) tenant teardown — graceful `close()`, `SIGKILL`, and 3-of-4
simultaneous `SIGKILL`; B's mappings stay byte-correct, no leaked
physical (via `/proc/cipher/kvdedup_stats`).
(c) **slab API unchanged** — op #1 unit test still **14/14**
(verification, not evolution — per call A).
(d) cross-process refcount integrity under churn — Mooncake toolagent
replayed by 4 tenant processes; after graceful exit `physical_pages==0`,
`refcount_releases==Σ misses`.
(e) module unload safety — `rmmod` with fds open → `EBUSY`; after
close → succeeds; re-`insmod` clean.

## LOC budget & anchors

- Budget: ~600 LOC kmod (`cipher_kmod/cipher_kvdedup.c` + Kbuild line +
  `/proc` entry), ~150 LOC userspace ioctl shim in
  `cipher_rt_kv_alloc.c`. **No slab_free retrofit** (call A) → the
  prompt's ~50 LOC line is dropped. Actuals stated in the report.
- Anchors (corrected per call B): `55ab8c0c` (fallback **kmod**
  snapshot) — frozen, the pre-T4.6.4 rollback point, unchanged forever.
  `86618c30` (fallback **libcipher_v2**) — frozen, libcipher_v2 not
  touched. The new working `cipher_kmod/cipher_kmod.ko` gets a fresh
  md5 recorded in the report as the **T4.6.4 working artifact** — a
  new baseline, not an evolution of `55ab8c0c`.

**Awaiting approval before STEP 3 (C).**
