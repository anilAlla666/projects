# Track 3 — Dynamic SM Migration — SC3 DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no source
modified, no GPU, no measurement. STOP for adjudication before SC3-2 build.
**Predecessor:** SC2 closed (`TRACK_3_SC2_CLOSEOUT.md`, kmod `98da2d1f`).

SC3 builds the **tenant-side** migration primitive in libcipher_rt. SC2 proved
the kmod coordination is sub-µs — **SC3 is the load-bearing work**: it
determines whether a migration is operationally cheap (a green-context rebuild
the tenant barely feels) or expensive. SC3 rotates the libcipher_rt anchor
`ebc0baaa`; kmod `98da2d1f`, libcipher_v2 `86618c30`, cipher_kv_bridge
`fca6843d` are untouched.

---

## §0 — Recap: what SC3 plugs into

`cipher_rt_green_ctx.c` (`ebc0baaa`) already builds a per-process green context
from a kmod-granted group mask — `cipher_rt_green_ctx_ensure()`:
`CIPHER_CP54_ALLOCATE` (cached by `cipher_rt_green_ctx_cp54_init()`) →
`cuDevSmResourceSplitByCount(MIN_SM=8)` → select the grp_mask groups →
`cuDevResourceGenerateDesc` → `cuGreenCtxCreate` → `cuCtxFromGreenCtx` →
`cuGreenCtxGetDevResource` verify. The module-static green-ctx state is
`g_green_ctx` / `g_green_cuctx` / `g_green_sm_count` / `g_green_group_id`,
guarded by `g_green_lock`. Enforcement is per-launch:
`cipher_rt_green_ctx_make_current()` does `cuCtxSetCurrent(g_green_cuctx)` from
the CUPTI launch callback (T4.2.4d).

**SC3's migration is exactly: rebuild that state for a new mask, drain, verify,
and atomically swap the four `g_green_*` variables the launch callback reads.**
The mechanism already exists for *create*; SC3 generalises it to *re-create*.

Three SC1 facts still govern: migration is **count-preserving**
(`popcount(new_mask) == g_green_sm_count/8`), **opt-in** (default pinned), and
**B ⊆ A** (a tenant that never migrates is byte-identical to today).

---

## §1 — API: `cipher_rt_green_ctx_migrate(uint32_t new_mask)`

New libcipher_rt C entry point, exported (plain symbol):

```c
/* Migrate this process's green context to the kmod-granted new_mask.
 * COUNT-PRESERVING: rejects unless popcount(new_mask) == current group count.
 * Called by the tenant ON ITS OWN THREAD, at a safe point (between decode
 * rounds — no kernel the tenant cares about is in flight). Never from a
 * signal handler, never from a background thread.
 * Returns 0 on success (green ctx now on new_mask), <0 on any failure with
 * the OLD green context left intact and current. */
int cipher_rt_green_ctx_migrate(uint32_t new_mask);
```

The contract's load-bearing half is the **failure guarantee**: on *any* non-zero
return the old green context is untouched and still current — the tenant is
exactly where it was, and the handler (§3) reports a NACK so the kmod frees the
reservation. This is the B-floor: a failed migration costs latency, never
correctness.

---

## §2 — The drain / build / verify / swap / release sequence

All on the calling (tenant) thread, under `g_green_lock`:

1. **Precondition.** `popcount(new_mask) != g_green_sm_count/8` → return
   `-EINVAL` (count-preserving, client-side — the kmod also enforces this at
   PROPOSE; SC1 fact 1, checked both ends).
2. **Drain.** `cuCtxSynchronize()` on the green context — no kernel of the old
   green context is in flight. The tenant has *also* called
   `torch.cuda.synchronize()` before invoking migrate() (the handler does
   this, §3); the in-library drain is defensive. (Step 1.4C cross-stream-race
   discipline: sync before any context swap.)
3. **Build.** The exact `cipher_rt_green_ctx_ensure()` path, for `new_mask`:
   `cuDevSmResourceSplitByCount` → select `new_mask`'s set-bit group resources
   → `cuDevResourceGenerateDesc` → `cuGreenCtxCreate` → `cuCtxFromGreenCtx`,
   into **local** variables `new_ctx` / `new_cuctx` (not the live globals yet).
4. **Verify — before any swap (§2a).** Confirm the freshly built context
   actually has `new_mask`'s placement. If verify fails: destroy `new_ctx`,
   return `-EIO` — the old context never moved.
5. **Swap.** Atomically reassign `g_green_ctx` / `g_green_cuctx` /
   `g_green_sm_count` / `g_green_group_id` to the new context, and
   `cuCtxSetCurrent(new_cuctx)`. Because step 2 drained and the tenant is at a
   safe point, **no launch callback runs concurrently** — the CUPTI callback
   reads `g_green_cuctx` only while the tenant is launching, and the tenant is
   not. `g_green_lock` covers the reassignment for hygiene against a
   (one-shot, already-complete) `ensure()`.
6. **Release.** `cuGreenCtxDestroy(old_ctx)`. If destroy fails the migration is
   *already committed* (step 5 done) — **log and leak the old context**, never
   fail a committed migration ([[cipher-incident-2-bar0-exit]] discipline:
   leak a handle rather than unwind a completed state change).
7. Return 0. The handler then issues `ACK_MIGRATE(1)`.

### §2a — Verify mechanism — **adjudication item**

SC1 §3b/§3d named a `%smid` probe kernel. That is a real cost for libcipher_rt:
the library is C and ships no CUDA kernels — a `%smid` probe means embedding a
cubin or carrying NVRTC. Two options:

| option | what it checks | cost |
|---|---|---|
| **(a) structural** *(recommend)* | `cuGreenCtxGetDevResource` SM count == `8·popcount(new_mask)`, **and** the descriptor was generated from exactly `new_mask`'s group resources (build provenance). Group→SM placement is deterministic (Step 1.3a probe; the `GROUP_SMS` table is fixed hardware). | zero new deps — same call `ensure()` already makes |
| (b) `%smid` probe | launches a probe kernel, reads the live SM set, set-compares to `new_mask`'s groups | embed a cubin / NVRTC in libcipher_rt — a new build dependency |

**Recommendation: (a) structural verify inside libcipher_rt.** Building the
descriptor from `new_mask`'s exact group resources *is* the placement
guarantee on this deterministic split; the count check catches a driver
anomaly.

**ADJUDICATED (2026-05-19) — TWO-LAYER verification chain.** The verify is
documented and built as two layers, **both of which must pass** — if either
fails, the tenant NACKs (`ACK_MIGRATE(0)`):

| layer | where | when | checks |
|---|---|---|---|
| **L1 structural** | libcipher_rt C, inside the migrate primitive | immediate, pre-swap (§2 step 4) | `cuGreenCtxGetDevResource` SM count == `8·popcount(new_mask)` + descriptor built from exactly `new_mask`'s group resources |
| **L2 `%smid` probe** | Python tenant (`cp54_pool.py` / `cp54_s16_partition_tenant.py` — already exists) | per decode round, post-swap | launches the probe kernel, set-compares the observed SM set to `new_mask`'s groups |

L1 gates the swap (a failed L1 → no swap, old context intact, NACK). L2 is the
per-round running confirmation the tenant already does — a post-swap L2
mismatch is a NACK and a tenant-level fault surface. This layering is what
makes structural verify *safe*: libcipher_rt stays cubin-free, and the
authoritative placement check still runs every round at the tenant layer.

---

## §3 — Poll-and-migrate handler

**Not a background thread.** A background thread cannot know the tenant is at a
safe point (SC1 §3b). The handler runs **inline at the tenant's decode-round
boundary** — a point where the tenant already synchronizes. Two layers:

- **Poll + protocol — Python**, a thin helper `cipher_migrate.py`
  (`cp54_pool.py`-style): opens `/dev/cipher`, and once per decode round issues
  `POLL_MIGRATE` (ioctl nr 17) directly. On `PROPOSED` it drives
  `START_MIGRATE` (18) → the migrate primitive → `ACK_MIGRATE` (19, ok=1/0).
  Direct ioctls from Python need no libcipher_rt symbols — the proven
  `cp54_pool.py` `QUERY` pattern.
- **Migrate primitive — libcipher_rt C** (`cipher_rt_green_ctx_migrate`, §1/§2).
  It must run in the process's injected CUDA context where `g_green_*` lives.

**Reaching the C primitive from Python — adjudication item.** libcipher_rt is
loaded by the driver as the `CUDA_INJECTION64_PATH` library; the manifest's
operational invariant warns that a *separately* `ctypes.CDLL("libcipher_rt.so")`
gets a different instance whose static state is not the injected one's. The
fix: bind the **already-loaded** instance via `ctypes.CDLL(None)` (global
symbol namespace) or `dlopen(..., RTLD_NOLOAD)`, so
`cipher_rt_green_ctx_migrate` resolves to the injected instance and mutates the
*live* `g_green_*`. **SC3-2 must verify this empirically** (a probe: call a
trivial libcipher_rt accessor via `CDLL(None)` and confirm it sees the injected
state) before the handler is built on it. If `CDLL(None)` does not reach the
injected instance, the fallback is a tiny ioctl-like trigger libcipher_rt
itself watches — flagged, not designed here.

**Poll cadence (ADJUDICATED — explicit).** Per the SC2 finding and the
Item-2 downgrade: v1 is **poll-based, eventfd deferred to v2**. The exact
per-round sequence the tenant runs:

```
each decode round:
    decode()                              # the round's actual work
    st = POLL_MIGRATE                     # ioctl nr 17, sub-µs
    if st.migrate_state == PROPOSED:
        START_MIGRATE                     # ioctl nr 18
        rc = cipher_rt_green_ctx_migrate(st.target_mask)   # drain/build/L1/swap/release
        %smid probe (L2)                  # post-swap tenant verify
        ACK_MIGRATE(rc == 0 and L2 ok)    # ioctl nr 19: 1 commit / 0 NACK
    # -> next decode round
```

The decode loop is the clock — no fixed timer. Cadence ≈ one decode round
(~16 ms, W2 data) — far inside the ≤1/tenant/10 s migration rate.

**Migration latency budget (for SC6 to characterize empirically):**
```
worst-case tenant-observed migration latency =
      (time to the next decode-round boundary)      # poll cadence — ≤ 1 round
    + cuCtxSynchronize drain                         # tenant already at safe pt
    + cuGreenCtxCreate build (+ SplitByCount/GenerateDesc)
    + L1 structural verify (cuGreenCtxGetDevResource)
    + g_green_* pointer swap + cuCtxSetCurrent
    + cuGreenCtxDestroy release
    + L2 %smid probe
```
Prior: a green-ctx rebuild measured ~1.7 ms (Step 1.5). The kmod PROPOSE→COMMIT
coordination is sub-µs (SC2) and is *not* in this budget — the budget is
entirely tenant-side. SC6 measures each term on the real substrate.

---

## §4 — Opt-in plumbing — `CIPHER_MIGRATABLE`

Per SC1 item 8 (env var, **no libcipher_v2 change**). In
`cipher_rt_green_ctx_cp54_init()`, after a successful `CIPHER_CP54_ALLOCATE`
with `qos=partition`: if `getenv("CIPHER_MIGRATABLE")` is `1`/`true`, issue
`CIPHER_CP54_SUBSCRIBE_MIGRATE(1)` on the same `/dev/cipher` fd. Unset/`0` →
the tenant stays `pinned` — never subscribes, never polls, never migrates;
byte-identical to today (the no-regression guarantee, exercised by the
all-pinned W1/W2/W3 smoke). `CIPHER_MIGRATABLE` is meaningful only for
`qos=partition`; ignored for POOL/SHARED.

---

## §5 — Failure modes & the tenant-NACK path

`cipher_rt_green_ctx_migrate()` returns non-zero — old context intact — and the
handler issues `ACK_MIGRATE(0)` (→ kmod `ABORTED_TENANT_NACK`, frees the RSVD
reservation, tenant keeps `old_mask`):

| failure | detection | result |
|---|---|---|
| count mismatch | §2 step 1 | `-EINVAL`, no CUDA touched |
| `cuGreenCtxCreate` / `GenerateDesc` fails | §2 step 3 rc | `-ENOMEM`/`-EIO`, old intact |
| verify fails (§2a) | §2 step 4 | `-EIO`, `new_ctx` destroyed, old intact |
| drain error | §2 step 2 `cuCtxSynchronize` rc | `-EIO`, no swap |
| old-ctx destroy fails | §2 step 6 | migration **already committed** — log + leak, return **0** |
| tenant crash mid-migrate | — | kmod do_exit reaper (SC2 two-cmpxchg) frees PACK+RSVD |

**NACK conditions = every pre-swap failure above.** Once step 5 (swap)
completes the migration is committed and the handler ACKs success even if
step 6 leaks. There is no tenant-initiated abort *after* the swap — consistent
with the SC2 invariant (the kmod never aborts MIGRATING; the tenant NACKs only
a migration it did not complete, and "did not complete" means "did not swap").

**Drain has no hard timeout** in v1: the tenant calls migrate() at a safe point
where its own kernels are already drained, so `cuCtxSynchronize` returns
promptly. A pathological hang is a stuck tenant (SC1 §3d safe-but-stuck), not a
migrate() concern; a drain watchdog is v2.

---

## §6 — SC3-2 / SC3-3 / SC3-4 plan & regression discipline

| phase | scope | est. |
|---|---|---|
| **SC3-1** | this design memo | done |
| **SC3-2** | build: `cipher_rt_green_ctx_migrate()` (§1/§2); structural verify (§2a); `CIPHER_MIGRATABLE` plumbing (§4); `cipher_migrate.py` handler (§3); the `CDLL(None)` reachability probe. Anchor `ebc0baaa` → `.pre_sc3` **before any edit**. | ~2–3 d |
| **SC3-3** | verify: pre-SC3 baseline (W1/W2/W3 on `ebc0baaa` + kmod `98da2d1f`) → `TRACK_3_SC3_PRE_BASELINE.md`; unit tests — (1) a tenant migrates itself via the API, kernels run on the new SM set; (2) verify-before-swap fails injection (forced mismatch → tenant NACKs, old intact); (3) `CIPHER_MIGRATABLE` plumbing on/off; regression smoke W1/W2/W3 within ±3 %; dmesg clean. | ~1–2 d |
| **SC3-4** | closeout: libcipher_rt `ebc0baaa` → SC3 anchor; fallback preservation; `TRACK_3_SC3_CLOSEOUT.md`; update `TRACK_3_ANCHORS.md` + `cipher_anchors_manifest.txt` + memory. | ~0.5 d |

**Per-milestone regression discipline (matches SC2):** existing kmod isolation
15/15 still PASS; W1 within ±3 % mean-vs-mean of the SC3 pre-baseline; W2
clause1=clause2=0 + KL gates; W3 `SC2_PASS=True`; dmesg clean. Any regression →
**STOP**, rollback to `.pre_sc3` libcipher_rt + kmod `98da2d1f`, surface.

**Anchors:** `ebc0baaa` → `cipher_rt_fallback/libcipher_rt.so.pre_sc3` before
any edit; → SC3 anchor only after smoke PASS. kmod `98da2d1f`, libcipher_v2
`86618c30`, cipher_kv_bridge `fca6843d` unchanged in SC3.

---

## §7 — Adjudication ask

**STOPPING — no source modified, no build, no GPU.** Decisions:

1. **API & sequence (§1/§2)** — accept `cipher_rt_green_ctx_migrate(new_mask)`,
   the drain → build → verify → swap → release order, and the failure
   guarantee (any pre-swap failure leaves the old context intact + current).
2. **Verify mechanism (§2a)** — accept **structural verify** inside
   libcipher_rt (count + build provenance; `%smid` probe optional at the
   Python layer, no embedded cubin) — or require the embedded `%smid` probe.
3. **Handler (§3)** — accept the inline two-layer design (Python
   `cipher_migrate.py` for poll + protocol; libcipher_rt C for the primitive),
   poll every decode round, no background thread, eventfd deferred to v2.
4. **Reachability (§3)** — note that SC3-2 empirically verifies `CDLL(None)` /
   `RTLD_NOLOAD` reaches the injected libcipher_rt instance before the handler
   is built on it; a fallback trigger is flagged if it does not.
5. **Opt-in (§4)** — accept `CIPHER_MIGRATABLE` env var → `SUBSCRIBE_MIGRATE`
   in `cipher_rt_green_ctx_cp54_init()`, no libcipher_v2 change.
6. **Failure / NACK (§5)** — accept NACK on every pre-swap failure;
   commit-and-leak on a post-swap destroy failure; no drain watchdog in v1.

On adjudication: proceed to **SC3-2** (build), closing with the §6 regression
discipline and the libcipher_rt anchor rotation.
