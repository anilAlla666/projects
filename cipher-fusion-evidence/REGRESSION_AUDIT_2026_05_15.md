# Regression Audit — 2026-05-15

**Trigger.** The CP 2.4 DVFS envelope sweep could not run: the VOLT actuator
failed to lock the GPU clock (`VOLT: DEGRADED`). Root question — did CP 3.3
(kmod rebuild 0.4.6→0.4.7, `b263ad30…` → `2a69f9de…`) regress the kmod?

**Scope.** Four checks against the **current** kmod `2a69f9defd7730665e6b7f9d60e82b43`
and the current libcipher_rt. Each check PASS/FAIL; for any FAIL, the causing
change. Audit ordered *before* any further CP 2.4 work, per instruction.

**Anchors at audit time:** kmod `2a69f9de…` (CP 3.3, 0.4.7); T4.6.4 kmod was
`b263ad30…`; libcipher_v2 anchor `86618c30…` — unchanged; taint 12288.

---

## Check 1 — T4.6.4's 5 binding indicators vs current kmod — **PASS (5/5)**

Harness `cipher-fusion-evidence/t4_6_4_kvdedup/kvdedup_xproc` (md5
`23544f1e…`) run as root against `/dev/cipher_kvdedup` on kmod `2a69f9de`.
Module-unload indicator (e) run as a separate rmmod/insmod sequence.

| # | Indicator | Result vs kmod `2a69f9de` | Verdict |
|---|---|---|---|
| (a) | cross-process byte-correct | 16 runs, **15 500 pages verified cross-process, 0 fail** | **PASS** |
| (b) | tenant teardown / release fop | graceful + SIGKILL + 3-of-4 SIGKILL — all survivors **200/200**; post-test kmod `entries=0 virtual_refs=0` | **PASS** |
| (c) | slab / allocator unit test | **14/14** — see Check 3 | **PASS** |
| (d) | refcount integrity under churn | 4-tenant interleaved 4000-page window: **misses=2987 releases=2987**, `entries=0 virtual_refs=0` | **PASS** |
| (e) | module-unload safety | rmmod with fd open → *"Module cipher_kmod is in use"* (refused); fd closed → rmmod succeeds; re-insmod → clean (`refcount 0`, md5 `2a69f9de` intact, taint 12288, `/dev` nodes back) | **PASS** |

**Cross-tenant KV dedup is NOT regressed by CP 3.3.** Every T4.6.4 closure
indicator reproduces on the current kmod, byte-identical numbers where the
T4.6.4 report gave them (15 500 / 0 fail; misses=releases=2987).

---

## Check 2 — T4.5.1 `.symver` cuBLAS capture vs current libcipher_rt — **PASS**

CP 3.3 was a **kmod-only** change — it did not rebuild `libcipher_rt.so`. The
only edit to libcipher_rt since T4.5 is the CP 2.4 Marlin per-stream registry
(`cipher_rt_marlin_engine.cpp`). Check 2 therefore verifies the CP 2.4 build
(`a0d6cdda…`) did not perturb the `.symver` substrate.

- **Structural:** `nm -D libcipher_rt.so` →
  `cublasGemmEx@libcublas.so.13` present, aliased to
  `cipher_rt_cublasGemmEx_impl`. `cublas_version.map` unchanged.
  `cipher_rt_cublas_shim.c` mtime `2026-05-14 16:58` — predates all CP 2.4
  work; the CP 2.4 Marlin edit did not touch the cuBLAS shim.
- **Runtime:** CP 2.4 test A already exercised the shim end-to-end —
  `MATMUL: calls=687375 handled=666675 passthrough=20700` — 687 K cuBLAS
  calls intercepted with 0 errors. The literal T4.5 "155/155" is a
  forward-specific count; the substrate's capture mechanism is intact and
  runtime-confirmed at far larger scale.

**The `.symver` interception is NOT regressed.**

---

## Check 3 — Op #1 allocator unit test — **PASS (14/14)**

`cipher-phase4-evidence/t4_6_2_kv_alloc/alloc_unit_test` (md5 `dd857483…`),
standalone. All 14 checks pass — `=== ALL PASS ===`: alloc_init, slab_create,
cuMemset/readback, slab_ensure growth, page_info + tag match, unmanaged-ptr
rejection, multi-slab, slab_free + VA reuse, all-pages-released. The KV
allocator (userspace `cipher_rt_kv_alloc.c`) is unaffected.

---

## Check 4 — `/dev/cipher` mode, T4.3 era vs now — the DVFS-access FAIL, root-caused

This is the one real FAIL: the DVFS VOLT actuator cannot reach its actuation
path as a non-root process.

**Observed now.** `/dev/cipher` and `/dev/cipher_kvdedup` are both
`crw------- root root` (**0600**). VOLT's NVML clock-lock returns
`NOT_SUPPORTED` for a user process (it did at T4.3 too); its fallback — the
kmod `CIPHER_SET_CLOCK_MHZ` ioctl — needs to `open("/dev/cipher")`, which a
non-root process now cannot. Result: `VOLT: DEGRADED`.

**T4.3 era.** The T4.3 envelope run logs show
`VOLT: kmod ioctl path available (CIPHER_SET_CLOCK_MHZ)` →
`VOLT: ACTIVE — locked GPU clock … via path=kmod-ioctl`. So at T4.3,
`/dev/cipher` *was* openable by the non-root workload.

**What actually changed — and what did NOT.** The kmod source did **not**
regress. `cipher_dev.c` creates the node with `device_create` and **no
`devnode` mode callback** — so the node has *always* defaulted to 0600. The
comment at `cipher_dev.c:360` is explicit:

> *"The /dev/cipher node defaults to mode 0600 root:root via udev. For
> Phase 2 testing … we relax via chmod after insmod."*

i.e. **0666 was never in the kmod** — it was always a manual `chmod` applied
after `insmod`. CP 3.3 rebuilt and **reloaded** the kmod (the legitimate,
necessary reload to ship 0.4.7 with the FLOP-telemetry ioctls). The reload
recreated `/dev/cipher` at its always-default 0600, and the manual chmod that
T4.3 relied on was not re-applied. This was demonstrated **live** in this
audit: indicator (e)'s re-insmod brought `/dev/cipher` back at `crw-------`.

**Causal statement (the FAIL's cause).** Not a CP 3.3 *code* bug — CP 3.3's
cipher_dev.c device-node logic is functionally identical to T4.6.4's. The
cause is: **CP 3.3 reloaded the kmod, and the device-node mode was never
codified in the kmod source — it lived only in a manual post-insmod chmod —
so the reload reverted to 0600 and the relax was lost.** A latent fragility
(uncodified permission) exposed by a legitimate reload.

**Verdict: FAIL** — DVFS non-root actuation is blocked. Cause: uncodified
device-node mode + CP 3.3 kmod reload. Not a functional kmod regression.

---

## Summary

| Check | Subject | Verdict |
|---|---|---|
| 1 | T4.6.4 cross-tenant KV dedup (5 indicators) | **PASS 5/5** |
| 2 | T4.5.1 `.symver` cuBLAS capture | **PASS** |
| 3 | Op #1 allocator unit test | **PASS 14/14** |
| 4 | `/dev/cipher` access for DVFS actuation | **FAIL** — uncodified 0666, lost on CP 3.3 kmod reload |

**CP 3.3 did not regress any kmod functionality.** All KV-dedup, allocator,
and substrate behavior is intact. The DVFS blocker is the single FAIL, and its
cause is a *pre-existing* gap — the kmod never codified the `/dev/cipher`
permission — surfaced by CP 3.3's necessary kmod reload. The fix belongs in
the kmod (codify the device mode so it survives reloads), not in a runtime
chmod.

---

## Decision needed — the right fix for `/dev/cipher` access

Per instruction, the fix is decided *after* this audit. Three options:

**Option 1 — codify the device mode in the kmod (proper fix).**
Add a `devnode` callback to the `cipher_class` (and `kvd_class`) that sets the
mode at node creation, so it is deliberate and survives every reload. Sub-choice:
- 1a — codify **0666**: restores T4.3 behavior; any process can issue
  `/dev/cipher` ioctls, *including `CIPHER_SET_CLOCK_MHZ`* — a tenant could
  throttle the whole GPU. This is the footgun CP 3.3's 0600 instinct avoided.
- 1b — codify **0600** deliberately: makes the current state intentional and
  reload-stable; non-root actuation still needs another path (Option 2 or 3).
Requires a kmod change → re-verify the CP 3.3 gate still passes.

**Option 2 — add a non-root-capable actuation path.**
A udev rule shipped with the kmod, or a narrow capability (e.g. a dedicated
`/dev/cipher_clock` node at 0666 exposing *only* `CIPHER_SET_CLOCK_MHZ`, with
the full ioctl surface staying 0600). More design; cleanest separation of the
privileged clock control from the rest of the ABI.

**Option 3 — `/dev/cipher` stays root-only; run the DVFS harness as root.**
No kmod change. Matches CP 3.3's deliberate posture (`cipher_flopd` runs as
root; CP 3.3 report: *"a udev rule for unprivileged consumers is operator
deployment policy"*). The CP 2.4 DVFS sweep runs the workload under `sudo` —
clock actuation is genuinely an operator privilege. Matched-pair Δtok/W is
privilege-invariant, so running both arms as root does not bias the result.

**Recommendation.** Clock actuation *is* a privileged operation — CP 3.3's
0600 instinct was correct, and Option 1a re-introduces a real footgun. The
honest fix is **Option 3 for CP 2.4's measurement** (run the DVFS harness as
root — the legitimate privilege level for clock control, consistent with the
`cipher_flopd` pattern) **plus a small Option-1b kmod hardening**: add a
`devnode` callback that *explicitly* sets 0600 with a source comment, so the
permission is a deliberate, reload-stable decision rather than an accidental
default — and delete the stale `cipher_dev.c:360` "relax via chmod" comment
that documents a workflow that no longer holds. That is "resolved in the
kmod": the kmod owns its device mode, the DVFS path uses root as designed,
and no ACL is silently weakened.

Awaiting adjudication on the fix before resuming CP 2.4 (DVFS sweep).

---

## Artifacts

| Artifact | md5 / note |
|---|---|
| `t4_6_4_kvdedup/kvdedup_xproc` | `23544f1e7819e2350b82e1b56ffb0d1e` |
| `t4_6_2_kv_alloc/alloc_unit_test` | `dd8574835f3caad1e26c218e4247379b` |
| current kmod | `2a69f9defd7730665e6b7f9d60e82b43` (0.4.7) |
| libcipher_rt (CP 2.4 build) | `a0d6cddacb116f2d51ad1d1f867ef564` |

**Anchors held throughout:** kmod `2a69f9de` (unchanged — the (e) rmmod/insmod
cycle re-loaded the identical .ko), libcipher_v2 `86618c30`, kmod anchor
`55ab8c0c`, taint 12288. No source modified during this audit.

---

## Re-verification — 2026-05-15 (independent re-run, "anchors check first")

Every check below was re-executed first-hand this session against the live
system before any decision/build work. Results reproduce the body verbatim.

| Check | Re-run evidence | Verdict |
|---|---|---|
| 1 | `kvdedup_xproc` (md5 `23544f1e`) re-run **as root** vs kmod `2a69f9de`: (a) 16 runs, cross-process verified **ok=15500 fail=0**; (b) graceful + SIGKILL + 3-of-4 all **200/200**, post `entries=0 virtual_refs=0`; (d) 4-tenant churn **misses=releases=2987**, `entries=0`. Exit 0. (c)=Check 3. (e) corroborated live: nodes mtime 17:37, kmod `2a69f9de` loaded, taint 12288, `/dev` nodes present. | **PASS 5/5** |
| 2 | `nm -D cipher_rt_phase4/libcipher_rt.so` (md5 `a0d6cdda`) → `cublasGemmEx@libcublas.so.13` aliased to `cipher_rt_cublasGemmEx_impl` (both at `0x8940`). | **PASS** |
| 3 | `alloc_unit_test` (md5 `dd857483`) → 14 `ok:` lines, `=== ALL PASS ===`. | **PASS 14/14** |
| 4 | `/dev/cipher` live = `crw------- root root` (0600). `cipher_dev.c` re-read: no `devnode` callback; `device_create` at line 322 sets no mode; the line-360 "relax via chmod after insmod" comment is present verbatim. Root cause stands. | **FAIL** (uncodified mode) |

Conclusion unchanged: CP 3.3 regressed **no kmod functionality**; the single
FAIL is the uncodified `/dev/cipher` mode surfaced by CP 3.3's legitimate kmod
reload. md5 anchors re-confirmed: kmod `2a69f9de`, libcipher_rt `a0d6cdda`,
`kvdedup_xproc` `23544f1e`, `alloc_unit_test` `dd857483`. No source modified.

---

## Decision-section revision — 2026-05-15 (correction; supersedes §"Decision needed")

Re-reading `cipher_clock.c` and `cipher_ioctl.h` during the anchor re-check
surfaced a fact the original §"Decision needed" did **not** account for. The
recommendation there (Option 3 + Option 1b) rests on a wrong premise and is
**withdrawn pending adjudication**.

**The premise that was wrong.** Check 4 said *"0666 was never in the kmod."*
Only half true. `device_create` sets no mode — but the kmod's **ABI design is
0666**, documented explicitly:
- `cipher_ioctl.h:249` — *"Trust model: /dev/cipher is 0666 (matches existing
  CIPHER ABI pattern …)."*
- `cipher_clock.c:12-23` — *"The access gate is the device-node permission,
  not a per-ioctl CAP check … Adding a per-ioctl CAP_SYS_ADMIN check here
  would defeat the design intent (the very gap T4.3.2 closes is non-root
  user-process → privileged actuation)."*
- `cipher_dev_set_clock_mhz` (cipher_clock.c:125-146) has **no `capable()`
  check** — deliberately. Its safety story is the `[210,1980]` MHz bounds
  clamp + per-call uid-audit log, *not* node-permission exclusivity.

So 0666 is not an accident — it is the kmod's stated contract, and T4.3.2 was
built on it (memory `cipher-t432-kmod-volt-ioctl.md`: "closes non-root gap").

**Why Option 1b (codify 0600) is incoherent as written.** Codifying 0600
*without also* adding a CAP check leaves `ioctl.h:249` and `cipher_clock.c:13`
**lying** (header says 0666, node says 0600) and silently breaks T4.3.2's
design. 0600 is only coherent if SET_CLOCK_MHZ *also* gets a `capable()` gate
and the trust-model comments are rewritten — i.e. a deliberate reversal of a
shipped feature.

**Separately — a stale comment, a wart to fix under either decision.**
`cipher_clock.c:123` says *"CAP_SYS_ADMIN required"* — directly contradicted by
its own handler (no `capable()` call) and by the trust-model block 20 lines
above. Misleading security documentation; correct it either way.

**The real fork (replaces the original 3 options):**

- **Posture A — keep T4.3.2's design.** Codify **0666** via a `devnode`
  callback on `cipher_class` *and* `kvd_class`; fix the stale `:123` comment;
  CP 2.4 DVFS harness runs **non-root** as T4.3.2 intended. Kmod source,
  docstrings, and behavior all align; the fragile manual chmod becomes a
  deliberate, reload-stable, source-of-truth decision. (This is *not* the
  rejected shell-trap chmod — it is the same mode, codified in the kmod with
  an explicit trust-model rationale.)
- **Posture B — reverse T4.3.2.** Add `capable(CAP_SYS_ADMIN)` to
  `cipher_dev_set_clock_mhz`; codify **0600** via the `devnode` callback;
  rewrite the `cipher_clock.c:12-23` / `ioctl.h:249-254` trust-model blocks;
  retire the t432 "non-root actuation" claim; run the CP 2.4 DVFS harness
  **as root** (the original Option 3). A coherent, deliberate reversal of a
  shipped feature.

Both are "resolved in the kmod, not a shell trap." Both need a kmod rebuild
(0.4.7 → 0.4.8) + CP 3.3 gate re-verification. The choice is a **security-
posture decision** — is non-root clock actuation intended (A) or wrong (B)? —
and is the user's to make. **No kmod source edited; awaiting adjudication.**

---

## CLOSING ADDENDUM — 2026-05-15: Posture A built + re-verified (kmod 0.4.8)

**Adjudication.** User selected **Posture A** — keep T4.3.2's design; codify
0666 in the kmod; non-root DVFS actuation is intended. Built per the user's
6-item verification plan.

### Kmod diff — 0.4.7 (`2a69f9de`) → 0.4.8 (`e2f50452`)

Four hunks, four files. Pure kmod, **no ABI change** (no ioctl nrs touched —
the additive-only rule is trivially held); no userspace touched.

| File | Change |
|---|---|
| `cipher_dev.c` | New `cipher_devnode()` callback returning `*mode = 0666`; wired `cipher_class->devnode = cipher_devnode` *before* `device_create()` (devtmpfs consults `->devnode` at node creation). Trailing comment block (was: *"defaults to 0600 … we relax via chmod after insmod"*) rewritten to state the mode is now codified — the stale chmod workflow is gone. |
| `cipher_kvdedup.c` | New `kvd_devnode()` callback (`*mode = 0666`); wired `kvd_class->devnode` before `device_create()`. |
| `cipher_clock.c` | nr-10 handler docstring: the **stale `"CAP_SYS_ADMIN required."` line** (contradicted by the handler — no `capable()` call — and by the trust-model block above it) corrected to describe the actual node-perm-gated, no-CAP, bounds-clamp + uid-audit design. No behavior change — comment only. |
| `cipher_main.c` | `MODULE_VERSION` `0.4.7` → `0.4.8`. |

The two CAP_SYS_ADMIN-gated ioctls (`SUBMIT_GPU_STATE`, `SUBMIT_PROCESS_UTIL`)
keep their in-handler `capable()` checks — independent of, and unaffected by,
the node mode. Out-of-scope observation: `cipher_proc.c:148` prints a hardcoded
`"cipher_kmod 0.4.5"` banner — pre-existing version drift (already wrong at
0.4.7), **not** touched here; flagged for a future cleanup.

### Verification — user's 6-item plan, all PASS

| # | Item | Result |
|---|---|---|
| 1 | Pre-edit rollback anchor | `cp_2_4/kmod_0.4.7_pre_devnode.ko` = `2a69f9de…` (verified, outside build dir) |
| 2 | Post-build md5 | `e2f50452f668859a96b1e25a2cba4e10`, `MODULE_VERSION=0.4.8`; clean build (only the benign env compiler-mismatch warning) |
| 3 | CP 3.3 gate vs 0.4.8 | **all 4 PASS** — (a) ring=256; (b) 3/3 tenants MFU>0; (c1) r_util **0.9793** (≥0.95); (c2) ratio **1.175** (0.80–1.20). Reproduces CP 3.3 (0.9811 / 1.169) within noise. Evidence: `cp_3_3/reverify_0_4_8/` |
| 4 | Regression checks 1/3 vs 0.4.8 | Check 1 — `kvdedup_xproc` (a) 15500/0, (b) 200/200×3 `entries=0`, (d) misses=releases=2987 → **PASS**. Check 3 — `alloc_unit_test` **14/14 ALL PASS**. Indicator (e) = the 0.4.7→0.4.8 rmmod/insmod itself: clean (`rmmod OK` / `insmod OK`, foreground — no job-control trip this time). |
| 5 | Node modes after fresh insmod | `/dev/cipher` and `/dev/cipher_kvdedup` both `crw-rw-rw- root root` (0666) — codified, **no chmod used**. taint 12288. |
| 6 | Documentation | this addendum |

### Artifacts

| Artifact | md5 / note |
|---|---|
| kmod 0.4.8 (new campaign anchor) | `e2f50452f668859a96b1e25a2cba4e10` |
| `cipher_kmod.ko.v0.4.8` (fallback, outside build dir) | `e2f50452…` |
| `cipher_kmod_src_v0.4.8.tar.gz` | `874f2f4c77038550e8feb3a664b2c9e9` |
| `cp_2_4/kmod_0.4.7_pre_devnode.ko` (rollback) | `2a69f9defd7730665e6b7f9d60e82b43` |
| `cp_3_3/reverify_0_4_8/{gate_run.log,flop_gate_result.json,flop_gate_corr.csv,flopd.log}` | 0.4.8 CP 3.3 gate evidence (CP 3.3 originals untouched — `flop_gate_result.json` still `67e4e650`) |

**Anchors.** libcipher_v2 `86618c30` untouched; libcipher_rt `a0d6cdda`
untouched (pure kmod change). kmod anchor `55ab8c0c` frozen. **New campaign
kmod anchor: 0.4.8 `e2f50452` — supersedes 0.4.7 `2a69f9de` going forward.**
taint 12288 throughout.

### Status

`/dev/cipher` access is **resolved in the kmod** — the device mode is now
codified, deliberate, and reload-stable. CP 3.3 fully re-verified on 0.4.8; no
regression. **CP 2.4 DVFS envelope sweep is unblocked** — VOLT's
`CIPHER_SET_CLOCK_MHZ` fallback can now `open("/dev/cipher")` from the non-root
harness, as T4.3.2 designed. The DVFS sweep runs non-root.
