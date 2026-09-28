# CP 5.4 Step 1.2 — pre-reload NR9 caller audit

**Date:** 2026-05-18. **Purpose:** verify *empirically*, before the Step 1.2
kmod reload, the V1 scope memo §1 assertion that deactivating legacy ioctl
nr 9 (`CIPHER_REQUEST_SM_PARTITION`) is safe. No GPU, no kmod change, no
anchor touched.

**Verdict: nr-9 callers DO exist** (in `libcipher_rt` itself) — so the V1 §1
"drain makes it free" wording is wrong — **but the deactivation is
nonetheless SAFE**: the callers degrade gracefully on any nonzero return, and
nr-9 failure is *already* an observed-and-tolerated condition on the live
substrate. Recommendation: Step 1.2 reload may proceed; retire the caller in
Step 1.3.

---

## §1 — Source audit: nr-9 callers

`grep` over `/home/ubuntu/cipher_rt_phase4/` and `/home/ubuntu/libcipher_v2/`:

- **Caller found: `cipher_rt_arbitrate.c`** — the T4.2.3/B7 ARBITRATE module.
  It defines a local mirror `_IOWR('C', 9, struct cipher_arb_request_local)`
  (== `CIPHER_REQUEST_SM_PARTITION`) and issues it from **two sites**:
  - `cipher_rt_arb_request_partition()` — one-shot, cold-path, at first stream
    observation.
  - `cipher_rt_arb_reissue_fair_share()` — the 30 s **ARB-poll thread**
    (FIT_HINT fair-share re-evaluation).
- `cipher_rt_arbitrate.o` **is linked into the live `libcipher_rt.so`**
  (`a7ac8e97`) — confirmed by `nm -D`: `cipher_rt_arb_init`,
  `cipher_rt_arb_request_partition`, `cipher_rt_arb_close` all exported.
- **No other caller.** `cipher_rt_partition_router.c` explicitly does *not*
  call nr 9 (a source comment confirms). No harness / test / vLLM script
  issues a `/dev/cipher` ioctl directly — every workload reaches nr 9 only
  *indirectly*, via the ARB module inside the LD_PRELOAD'd substrate. (The
  `grep` hits for "REQUEST_SM_PARTITION" outside the substrate are all `.md`
  audit documents, not code.)

**This contradicts V1 §1's "after the reload no tenant is alive to call nr 9,
and every new tenant uses the CP 5.4 path."** The drain clears *live
allocations*, but the `libcipher_rt.so` binary itself contains an nr-9 caller
— so every workload that LD_PRELOADs the current substrate will call nr 9
until `libcipher_rt` is reworked (Step 1.3).

## §2 — Severity: the callers tolerate failure (deactivation is benign)

Both call sites were read in full (`cipher_rt_arbitrate.c`):

- They branch on a **generic `rc != 0`**, not on a specific errno. On failure:
  - the ARB-poll thread logs (`"ARB-poll: FIT_HINT reissue rc=… errno=…"`) and
    **continues** — its return value is `(void)`-discarded by the poll loop;
  - the one-shot logs and **returns mask 0**.
  So `-ENOSYS` is handled identically to any other nonzero return.
- **The ARBITRATE mask does not drive SM placement.** `cipher_rt_green_ctx.c`
  hash-picks its 8-SM group (`pid ^ tenant_handle`) and its own comment states
  the ARBITRATE mask popcount is "unused in v1." A mask of 0 from a failed
  nr 9 changes nothing about green-context creation.
- **Empirical proof it is already tolerated:** the CP 5.3 STEP 2B Arm-C run
  *this session* — current kmod `e2f50452` + current `libcipher_rt`
  `a7ac8e97` — logged `[cipher_v2] ARB-poll: FIT_HINT reissue rc=-1 errno=28`
  (nr 9 returned **-ENOSPC**) and the workload ran correctly (0.380
  acceptance, coherent decode). nr-9 failure is already a live, benign
  condition; `-ENOSYS` post-reload is the same class of event.

## §3 — Live-state audit

| check | result |
|---|---|
| `lsmod` cipher_kmod | loaded, **refcount 0** — no process holds `/dev/cipher` |
| tenant processes (`ps`) | none — no python/vLLM/CIPHER workload running (only system daemons) |
| open fds on `/dev/cipher` (`lsof`) | none |
| `/proc/cipher/` | present (`stats`/`flops`/`gpu_state`/`bar0_state`); `reaped=5346` — all past tenants exited; no live tenant holds any partition allocation |

Live state is **clean** — the drain (of live allocations) is trivially
already done; nothing is running.

## §4 — Verdict and recommendation

- nr-9 callers **exist** (`cipher_rt_arbitrate.c` in `libcipher_rt.so`) — the
  V1 §1 "no callers after drain" claim is **incorrect** and should be
  corrected (the safety argument is §2, not "no callers").
- The deactivation is **safe anyway**: the callers degrade gracefully on any
  nonzero `rc`, the ARBITRATE mask is not load-bearing for SM placement, and
  nr-9 failure is already observed-and-tolerated on the live substrate.
- The user's pre-registered branch ("if callers found → STOP and adjudicate:
  keep nr 9 active *or* update libcipher_rt to nr 13") — the audit surfaces a
  **third, cleaner option**: **proceed with the deactivation** (callers
  tolerate `-ENOSYS`), and **retire the ARB module in Step 1.3**. The 30 s
  ARB-poll thread is itself *superseded* by the CP 5.4 scheduler — its
  fair-share quartile logic is exactly what CP 5.4 replaces. Post-reload it
  would harmlessly log one `-ENOSYS` every 30 s per process until Step 1.3.

**Recommendation — Step 1.2 reload is SAFE w.r.t. nr 9.** Follow-ups:
1. Correct V1 §1's wording (the safety is "callers degrade gracefully," not
   "no callers").
2. Step 1.3 (libcipher_rt rework) must **retire the ARB-poll thread** (or
   migrate it to `CIPHER_CP54_ALLOCATE` nr 13) — it is dead, superseded code
   post-reload.

## §5 — Adjudication ask

Confirm Step 1.2 (kmod reload) may proceed on the §2/§4 basis — nr-9 callers
exist but tolerate the `-ENOSYS` deactivation — **or** elect to keep nr 9
active / rework `libcipher_rt` first. The audit recommends proceeding.
