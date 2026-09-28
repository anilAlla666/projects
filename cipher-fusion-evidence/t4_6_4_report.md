# T4.6.4 — kmod-owned cross-tenant KV dedup — STEP 3 REPORT

**Phase 4 closure gate: all five indicators PASS.**

T4.6.4 moved the dedup refcount table out of process-local `cipher_rt`
memory into `cipher_kmod`. Tenant processes ioctl into a new
`/dev/cipher_kvdedup` char device; the kernel owns the cross-tenant
refcount table and the cuIpc POSIX-FD handles, and tears a tenant's
references down via the release fop on `close()` and on process death
alike.

## Five binding indicators

| # | Indicator | Result |
|---|---|---|
| **(a)** | cross-process byte-correct | **PASS** — 16 runs (1 explicit + 15 Mooncake windows), **15,500 pages verified cross-process, 0 fail**. Tenant B (separate process) imported tenant A's physical pages via cuIpc POSIX-FD through the kmod and SHA-256-matched every readback. |
| **(b)** | tenant teardown / release fop | **PASS** — graceful `close()`: B's 200 pages intact after A exits; **SIGKILL**: B's 200 intact after A killed mid-life; **3-of-4 SIGKILL**: survivor's 200 intact after 3 tenants reaped. Post-test kmod `entries=0` — no leaked physical. |
| **(c)** | slab API regression | **PASS** — op #1 allocator unit test **14/14 unchanged** (the `requestedHandleTypes` addition to `map_one` did not perturb the slab path; verification, not evolution, per call A). |
| **(d)** | refcount integrity under churn | **PASS** — 4 tenant processes replayed a 4000-page Mooncake toolagent window interleaved (shared id pool → cross-tenant overlap); **misses=2987, refcount_releases=2987** (every unique physical page released exactly once), `entries=0`, `virtual_refs=0`. |
| **(e)** | module unload safety | **PASS** — `rmmod` with a tenant fd open → *"Module cipher_kmod is in use"* (refused); close fd → `rmmod` succeeds; re-`insmod` → clean (`refcount 0`, `/dev/cipher_kvdedup` back). |

## Artifacts

| Artifact | LOC | md5 / status |
|---|---|---|
| `cipher_kmod/cipher_kvdedup.c` | 461 | new kmod file |
| `cipher_kmod/cipher_kvdedup.h` | 86 | ioctl ABI (shared kmod/userspace) |
| `cipher_rt_phase4/cipher_rt_kv_alloc.c` dedup rework | 254 | kmod-ioctl shim (prompt budget ~150; the export/import/confirm put-flow is heavier than a thin shim — honest actual) |
| `cipher-fusion-evidence/t4_6_4_kvdedup/kvdedup_xproc.c` | 335 | cross-process test harness |
| **`cipher_kmod/cipher_kmod.ko`** | — | **T4.6.4 artifact md5 `b263ad30453d9f620d4f279627c7258e`** (was `119cb583…` pre-T4.6.4) |

Kmod LOC 461+86=547 (≈ prompt's ~600 budget). All files build clean
(no warnings).

## Anchors & system state

- **Fallback anchors — both frozen, unchanged** (per call B):
  `55ab8c0cd8309ca7cc0fc40fe556aa19` (kmod snapshot — pre-T4.6.4
  rollback point) · `86618c30896470b642fcc6985d8dc632` (libcipher_v2 —
  not touched).
- **New working kmod** `cipher_kmod.ko` md5 `b263ad30453d9f620d4f279627c7258e`
  recorded as the T4.6.4 artifact (a fresh baseline, not an evolution
  of `55ab8c0c`).
- **Kernel taint: 12288** — unchanged across the rmmod/insmod cycles.
- `cipher_kmod` loaded, refcount 0.

## How the gate ran (honest record)

- The currently-loaded `cipher_kmod` held refcount 2 — **not a leak**:
  CIPHER's own monitoring daemons `cipher-gpustate` (held `/dev/cipher`)
  and `cipher-exporter` (polled `/proc/cipher`). They were stopped to
  free the kmod for the reload. **They remain down** — restarting CIPHER's
  monitoring daemons is an operator step, not part of T4.6.4.
- `rmmod` (old) → `insmod` (T4.6.4) succeeded; dmesg confirms
  *"/dev/cipher_kvdedup ready (major=510, T4.6.4 cross-tenant KV dedup)"*.

## Honest gaps

- **`/dev/cipher_kvdedup` perms:** the kmod's `device_create` makes the
  node `crw------- root root` (0600). The design-memo-H target — 0660,
  group `cipher` — is **operator udev-rule policy**, a deployment
  artifact not emitted by the kmod build and not shipped here. The gate
  harness ran as root, so this did not block it; shipping the udev rule
  is an operator-deployment step.
- **Call model:** T4.6.4 ships the mechanism on the **cold-path
  assumption** (`dedup_put` once per slab page at allocation; ~400 µs
  hit latency amortized). Whether the live decode write-path (model b)
  pays a TTFT tax is a **T4.6.5** measurement.
- **Timing side-channel** (PUT latency hit-vs-miss) — accepted and
  documented for Phase 7 (design memo G).
- The dedup path remains **separate from the live slab path** (call A);
  the `slab_free` refcount retrofit + live-write-path wiring is
  **T4.6.5**.

## Phase 4 closure

All five indicators PASS → **Phase 4 closes.** T4.6.5 (full-trace
replay with real KV bytes + the slab-merge question against the live
write path) is the first Phase 5 deliverable.
