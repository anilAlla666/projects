# Week 2 Step 5 — `/proc/cipher/classify_stats` Proc Node — RESULT

**Status: PASS.**

New kmod-side proc node `/proc/cipher/classify_stats` exposing classify-substrate counters. Step 5 ships the node with zero-valued atomic64 counters; Step 6 wires the userspace-to-kmod ioctl bridge that populates them from `libcipher_rt.so::cipher_rt_classify_observer_snapshot()`. Kmod reload clean, ABI invariants hold, CP 5.4 byte-identical.

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 5 of 7
**Anchors:**
  - Pre:  `f8572ecf` (`week-1-step-4-cb2-reserved-tail`)
  - Post: `bc48590e` (`week-2-step-5-classify-proc-node`)
  - cipher_kmod.ko md5: `09c6ded5...` → `f497b2ee...`

---

## A — Pre-edit verification — PASS

| signal | value | expected | match |
| --- | --- | --- | --- |
| HEAD | `f8572ecf` | `f8572ecf` | ✓ |
| working tree | clean | clean | ✓ |
| Pre-edit build rc | 0 | 0 | ✓ |
| Pre-edit md5 | `09c6ded5...` | `09c6ded5...` | ✓ |
| Pre-edit warnings | 1 | 1 | ✓ |

Kmod build is deterministic (unlike libcipher_rt.so's nvcc-non-determinism). md5 reproducibility check is reliable here.

---

## B — `cipher_proc.c` emit pattern (captured)

Six existing proc nodes follow the same pattern:

```
proc_create(name, 0444, dir, &fops);  /* mode 0444; read-only */
static const struct proc_ops fops = {
    .proc_open    = <name>_open,
    .proc_read    = seq_read,
    .proc_lseek   = seq_lseek,
    .proc_release = single_release,
};
static int <name>_open(struct inode *inode, struct file *file) {
    return single_open(file, <name>_show, NULL);
}
static int <name>_show(struct seq_file *m, void *v) { ... seq_printf(m, ...); ... }
```

Existing nodes: `stats`, `bar0_state`, `gpu_state`, `flops`, `migrations`, `arenas`. `flops` and `migrations` `_show` callbacks live in their owning subsystem files (`cipher_flops.c`, `cipher_cp54_sched.c`); `bar0_state`, `gpu_state`, `arenas` follow similar pattern. `stats` show is in `cipher_proc.c` itself.

`cipher_proc_init()` registers all 6 with full unrolled-cleanup pattern (each failure path unwinds all prior `proc_create`s + the parent dir). `cipher_proc_exit()` reverses, NULL-checking each entry.

---

## C — `classify_stats` node design

For Step 5 (stub), the show callback + counter storage live in `cipher_proc.c` directly (no separate subsystem .c needed; substrate is small and may evolve into its own .c file in Step 6 wiring).

Output format:

```
classify_stats:
  total:        <N>
  handled:      <N>
  passthrough:  <N>
  per_op_class:
    GEMM:               <N>     ← only non-zero classes printed
    ATTENTION:          <N>
    ...
    UNCLASSIFIED:       <N>     ← slot 15 (observer convention)
  (no producer wired yet — Step 6 will populate via ioctl bridge)   ← only printed if total==0
```

The "no producer wired yet" line is the Step 5 honesty signal — it disappears once Step 6's ioctl bridge starts pushing real counts.

---

## D — Implementation diff

### `cipher_internal.h` (+20 lines)

```c
#define CIPHER_PROC_CLASSIFY_STATS "classify_stats"   /* Week 2 Step 5 */

struct cipher_classify_stats {
    atomic64_t total;
    atomic64_t handled;
    atomic64_t passthrough;
    atomic64_t per_op_class[16];
};
extern struct cipher_classify_stats cipher_classify_stats;
```

### `cipher_main.c` (+4 lines)

```c
/* Week 2 Step 5 — classify-stats counters (BSS zero-init). */
struct cipher_classify_stats cipher_classify_stats;
```

### `cipher_proc.c` (+88 lines)

- New static `cipher_proc_classify_stats_entry` proc-dir-entry pointer
- `cipher_classify_op_names[16]` name table (matches `cipher::OpClass` enum + UNCLASSIFIED at slot 15 per observer bound-clamp convention)
- `cipher_classify_stats_proc_show()` — atomic64_read each counter, seq_printf formatted output, skip zero per-op-class entries, print "no producer wired yet" hint if total==0
- `cipher_proc_classify_stats_open()` + `_fops` matching existing pattern
- `proc_create()` registration in `cipher_proc_init()` with full unrolled-cleanup pattern (added 7th tier of cleanup)
- `proc_remove()` in `cipher_proc_exit()` (added before the existing arenas removal — LIFO order)

---

## E — Build + ABI invariants — PASS

| signal | value |
| --- | ---:|
| Build rc | 0 |
| Warning count | 1 → 1 (zero delta) |
| cipher_kmod.ko md5 | `09c6ded5...` → `f497b2ee...` |
| New nm symbols (8 new internal/static + 1 exported) | |
| Existing IOCTL nrs | unchanged (`diff` vs HEAD: clean) |
| Existing struct field offsets | unchanged (no existing struct modified) |
| Reserved ioctl nrs 2/3/4 | still `-ENOSYS` (path untouched) |

### New nm symbols

```
B cipher_classify_stats                              ← BSS counter storage
t cipher_classify_stats_proc_show                    ← show callback
t cipher_proc_classify_stats_open                    ← open callback
b cipher_proc_classify_stats_entry                   ← BSS entry pointer
r cipher_proc_classify_stats_fops                    ← rodata fops
t __pfx_cipher_classify_stats_proc_show              ← Linux 6.x prologue
t __pfx_cipher_proc_classify_stats_open              ← Linux 6.x prologue
```

`cipher_classify_stats` is the exported global (uppercase `B`); the rest are file-local statics (lowercase `t`/`b`/`r`).

---

## F — Regression smoke — PASS

### F.1 Existing /proc/cipher state (pre-reload)

```
/proc/cipher/: arenas bar0_state flops gpu_state migrations stats   (6 nodes)
```

### F.2 Kmod reload

```
$ sudo rmmod cipher_kmod && sudo insmod cipher_kmod.ko
rc=0
```

### F.3 Post-reload /proc/cipher state

```
/proc/cipher/: arenas bar0_state classify_stats flops gpu_state migrations stats   (7 nodes)
                                ^^^^^^^^^^^^^^^ NEW; mode 0444; readable by all
```

### F.4 `cat /proc/cipher/classify_stats`

```
classify_stats:
  total:        0
  handled:      0
  passthrough:  0
  per_op_class:
  (no producer wired yet — Step 6 will populate via ioctl bridge)
```

Honest zero state; the trailing hint line confirms Step 5 stub status.

### F.5 dmesg

```
[612907.089134] cipher_kmod: /dev/cipher ready (major=511, REGISTER_TENANT + GPU_STATE/PROCESS_UTIL/LAUNCH_STATS live)
[612907.089235] cipher_kmod: /dev/cipher_kvdedup ready (major=510, T4.6.4 cross-tenant KV dedup)
[612907.089259] cipher_bar0: bound to 0000:07:00.0 [10de:2330] BAR0=0x6002000000 size=16 MB
[612907.089261] cipher_bar0: PMC_BOOT_0=0x180000a1 (raw; decode deferred to Phase 6)
[612907.089262] cipher_bar0: PMC_BOOT_1=0x00000000 (raw; decode deferred to Phase 6)
[612907.095872] cipher_kmod: kprobes attached at 0xffffffffc05f6300; ...
[612907.095934] cipher_kmod: loaded ok; nvidia_unlocked_ioctl @ 0xffffffffc05f6300 hooked
[612907.095937] cipher_state_updater: kthread starting (cadence 1000 Hz)
```

Clean init. PMC_BOOT_1=0x00000000 is the H100 bare-metal reading per the `cipher-pmc-boot-1-bare-metal` memory entry — not a stuck-bus issue.

### F.6 CP 5.4 regression

```
=== Phase A result: 15 PASS, 0 FAIL ===
rc=0
```

Byte-identical to Step 4 baseline.

---

## G — Commit + tag — PASS

```
3 files changed, 112 insertions(+)
  cipher_internal.h | 20 ++++
  cipher_main.c     |  4 +++
  cipher_proc.c     | 88 +++++++++++++++++++++++++++++++

HEAD: bc48590efce0ee1cb790bc203b9f6c0e929ed7d7
Tag:  week-2-step-5-classify-proc-node
Pre tag: f8572ecf... (week-1-step-4-cb2-reserved-tail)
```

---

## Discipline notes

- Single commit. Tag chain on cipher_kmod: `week-1-step-4-cb2-reserved-tail` → `week-2-step-5-classify-proc-node`.
- Rollback: `git reset --hard week-1-step-4-cb2-reserved-tail`.
- Counters are kmod-side BSS-resident. **Stay zero until Step 6 wires the userspace-to-kmod ioctl bridge.** The "(no producer wired yet)" hint line in proc output documents this state.
- Per impedance check §3.5 Option (i): kmod-side counter storage chosen. Step 6 will design the ioctl signature (likely `CIPHER_PUSH_CLASSIFY_STATS` or similar; takes `struct cipher_rt_classify_observer_stats` payload).

---

## Anchors at close

- `cipher_kmod` HEAD: `f8572ecf` → `bc48590e` (tag `week-2-step-5-classify-proc-node`)
- `cipher_rt_phase4` HEAD: `7ee5b2a8` (unchanged after Step 4)
- `cipher-may13-evidence` HEAD: `fc8a9ae6` (unchanged)

Week 2 progress: **5/7 steps complete**. Step 6 (hot-path CLASSIFY+SENSE+ORACLE wiring + userspace-to-kmod ioctl bridge for classify_stats) is the integration payoff step. Step 7 is closeout.
