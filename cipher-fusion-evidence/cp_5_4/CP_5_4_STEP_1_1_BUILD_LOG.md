# CP 5.4 — Step 1.1: kmod ledger + slot→SM bridge — BUILD LOG

**Date:** 2026-05-18. **Status:** built + verified — **build only, no kmod
reload** (reload is Step 1.2). Loaded kmod is still `e2f50452`.

---

## What Step 1.1 built

The kmod-resident CP 5.4 8-SM-group allocation ledger, per `CP_5_4_SCOPE_V1.md`.

| file | change |
|---|---|
| `cipher_cp54_sched.c` | **new** (~280 LOC) — the 16×8-SM-group ledger: atomic group array, per-PID metadata, ALLOCATE/FREE/QUERY ioctl handlers, `cipher_cp54_release()` reaper hook, init/exit |
| `cipher_ioctl.h` | **+** nrs 13/14/15 (`CIPHER_CP54_ALLOCATE`/`FREE`/`QUERY`), 2 structs, `CIPHER_CP54_QOS_*` — additive per [[cipher-abi-rule]] |
| `cipher_internal.h` | **+** 6 CP 5.4 function decls |
| `cipher_dev.c` | **+** 3 ioctl switch cases; **nr-9 `REQUEST_SM_PARTITION` deactivated** → `-ENOSYS` (legacy 4-SM allocator and CP 5.4 ledger address the same physical SMs — one allocator, one source of truth) |
| `cipher_probe.c` | **+** `cipher_cp54_release(pid)` in the `do_exit` reaper, beside the legacy slot release |
| `cipher_main.c` | **+** `cipher_cp54_sched_init()` / `_exit()` wiring |
| `Kbuild` | **+** `cipher_cp54_sched.o` |

The legacy `cipher_partition_allocator.c` is **byte-unchanged** — its code
stays, only its userspace entry (nr 9) is closed.

## Build result

`make` (kernel 6.8.0-1046-nvidia headers) — **clean**. `cipher_cp54_sched.o`
compiled with no warnings under `-Wall`; `cipher_kmod.ko` linked. (Benign
pre-existing notes: compiler-patch-level mismatch; BTF skipped — no `vmlinux`.)

Verification of the artifact:
- New `cipher_kmod.ko` md5 **`90103f82`** — the CP 5.4 in-flight artifact.
- `modinfo` valid.
- All CP 5.4 symbols linked in: `cipher_cp54_ioctl_{allocate,free,query}`,
  `cipher_cp54_release`, `cipher_cp54_sched_{init,exit}`, `cipher_cp54_groups`,
  `cipher_cp54_allocs`, `cipher_cp54_lock`, `cipher_cp54_pool_pid`.
- nr-9-deactivation + ledger-init log strings present in the `.ko`.
- **No reload** — `lsmod` shows the still-loaded kmod is the old `e2f50452`.

## Pre-reload review — three races found and fixed

A code review before the Step 1.2 reload caught three correctness issues —
fixed pre-reload (cheap) rather than as a post-reload kmod hot-swap (a full
`rmmod`/`insmod` cycle). First build was `4b536e41`; the fixed build is
`90103f82`.

1. **Module-exit teardown race.** `cipher_cp54_sched_exit()` was wired *before*
   `cipher_probe_exit()` — the `do_exit` kprobe (which calls
   `cipher_cp54_release()`) is unregistered only in `cipher_probe_exit`, so a
   process exiting mid-`rmmod` could fire the reaper against torn-down ledger
   state. Fixed: `cipher_cp54_sched_exit()` moved to *after* `cipher_probe_exit`
   in `cipher_main.c`.
2. **`cipher_cp54_pool_pid` clobber race.** The lock-free `release()` did
   `if (pool_pid == pid) pool_pid = 0` — a concurrent `ALLOCATE(POOL)`
   registering a new owner between the read and write would be clobbered to 0.
   Fixed: `cipher_cp54_pool_pid` is now `atomic_t`; `release()` uses
   `atomic_cmpxchg(pid → 0)` — clears only if still the dying pid.
3. **qos_class change via re-`ALLOCATE`.** A tenant already classified
   PARTITION/POOL re-calling `ALLOCATE` with a different class set new metadata
   without releasing the old class's groups. Fixed: a class change on an
   already-classified tenant returns `-EINVAL` ("FREE first"); SHARED→other and
   same-class idempotent re-calls still allowed.

Also added: `static_assert(CIPHER_CP54_NUM_GROUPS <= 32)` (the `grp_mask` is a
`u32`). Rebuild after the fixes: clean, no warnings.

## Anchors

- **Fallback preserved BEFORE the rebuild**, outside the kbuild dir per
  [[cipher-kbuild-clean-wipes-ko]]: `e2f50452` →
  `/home/ubuntu/cipher_kmod_fallback/cipher_kmod.ko.pre_cp5_4` (md5-verified).
- The new `90103f82` `.ko` is built but **not loaded**. The kmod anchor
  rotates only when Step 1.2 reloads it; until then `e2f50452` is live.
- `libcipher_rt` `a7ac8e97` unchanged (no libcipher_rt work in Step 1.1).

## Design notes / flags for the Step 1.2 adjudication checkpoint

1. **Concurrency — one cold-path mutex, not "resize-only".** The adjudication
   chose Option (ii) "small mutex around pool-resize only; steady-state
   allocate/free stays lock-free." As built: **`cipher_cp54_ioctl_allocate`
   takes `cipher_cp54_lock` for the whole operation**; `FREE`/`release` and
   `QUERY` are lock-free. Reason: the per-PID metadata-table slot claim also
   needs serialisation, and splitting "lock-free claim" from "mutex'd resize"
   reintroduces exactly the subtle reasoning Option (ii) was chosen to avoid.
   ALLOCATE is a cold path (per-session registration), so a whole-operation
   mutex is bounded and simple — the *intent* of Option (ii). `FREE`/`release`
   had to stay lock-free regardless: `release()` runs in the `do_exit` kprobe
   pre-handler (atomic context — cannot take a mutex). **This is a deliberate,
   documented deviation from the memo's literal wording — confirm or correct
   at the checkpoint.**
2. **Metadata table size** — `CIPHER_CP54_MAX_ALLOCS = 64`. Fine for CP 5.4
   Step 1 verification (2 partition + a handful of pool tenants); **too small
   for CP 5.5's 100-tenant soak** — flagged to revisit (hashtable or larger
   array) before CP 5.5.
3. The slot→SM *bridge proper* (`cuDevSmResourceSplitByCount` enactment) is
   libcipher_rt-side — Step 1.3. Step 1.1's kmod side is the `u16 grp_mask`
   the bridge consumes.

## Next — Step 1.2 (NOT done here)

Step 1.2 = kmod isolation test: `rmmod e2f50452` → `insmod 90103f82` →
exercise ALLOCATE/FREE/QUERY/resize cycles → verify; rollback to the preserved
`e2f50452` on any failure. **The reload is the load-bearing risk surface and is
held for explicit go-ahead** — it disrupts any running tenant and rotates the
kmod anchor. Per the adjudicated discipline, an adjudication checkpoint follows
Step 1.2 before Step 1.3 (libcipher_rt integration).
