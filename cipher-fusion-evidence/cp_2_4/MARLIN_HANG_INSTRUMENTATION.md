# CP 2.4 — Marlin hang: instrumented confirmation

**Date:** 2026-05-16. **Status:** root cause **CONFIRMED by direct evidence.**

This is the diagnostic artifact for the `CIPHER_MARLIN=on` hang that surfaced
once kmod 0.4.8 restored `/dev/cipher`. Prior context:
`PROGRESS.md` (DVFS section + correction), and the two failed fixes —
substrate gate (`55e2e323`) and the primary-context pin (`caae0bb8`, fix (i)).

---

## 1. Instrumentation added

Diagnostic build of `libcipher_rt.so` — md5 **`b1a3424c07f8d0ebb36135f3d9130884`**
— built from the `caae0bb8` source (a0d6cdda + fix (i)) plus logging in
`cipher_rt_marlin_engine.cpp`. **Diagnostic only — does not ship.**

- `marlin_instr_launch()` helper — logs, for every Marlin engine kernel
  launch: monotonic-ns timestamp, kernel descriptor (name + K/N/M shape),
  grid (x,y,z), block (x,y,z), shared-mem bytes, stream pointer, the **current
  CUDA context** (`cuCtxGetCurrent`), and the `cipher_rt_in_marlin_quant` flag.
- Calls at all four engine launch sites: `quant.transpose`, `quant.scales`,
  `quant.pack` (the three NVRTC quant kernels) and `marlin.gemm` (the GEMM).
- `cipher_rt_marlin_engine_dispatch` and `cipher_rt_marlin_engine_quantize_repack`
  ENTER/EXIT logging.

Smoke: `envelope_driver.py`, Mistral-7B B=1 decode, `CIPHER_MARLIN=on`,
substrate ON (production config), kmod 0.4.8, log captured to file.

---

## 2. Captured log fragment — the smoking gun

```
GREEN: green context bound to group 2 with 8 SMs (CUcontext=0x587b98689cf0)
...
MARLIN-INSTR quantize_repack ENTER W=0x7d80b2000000 K=4096 N=4096
MARLIN-INSTR ... launch=[quant.transpose K=4096 N=4096] grid=(128,512,1) ... ctx=0x587b95ed80b0 in_marlin_quant=1
MARLIN-INSTR ... launch=[quant.scales   K=4096 N=4096] grid=(32,32,1)   ... ctx=0x587b95ed80b0 in_marlin_quant=1
MARLIN-INSTR ... launch=[quant.pack     K=4096 N=4096] grid=(64,512,1)  ... ctx=0x587b95ed80b0 in_marlin_quant=1
MARLIN-INSTR quantize_repack EXIT  W=0x7d80b2000000 rc=0
MARLIN: quantized W=0x7d80b2000000 (K=4096 N=4096) — total 1 weights
MARLIN-INSTR dispatch ENTER M=1 N=4096 K=4096 G=128 stream=(nil)
MARLIN-INSTR ... launch=[marlin.gemm M=1 N=4096 K=4096] grid=(132,1,1) block=(256,1,1) shmem=98304 stream=(nil) ctx=0x587b98689cf0 in_marlin_quant=0
MARLIN-INSTR dispatch EXIT  rc=0 M=1 N=4096 K=4096
   ... (weight 2: same pattern — quant kernels ctx=0x587b95ed80b0, marlin.gemm grid=(132,1,1) ctx=0x587b98689cf0) ...
MARLIN-INSTR quantize_repack ENTER W=0x7d7fc3000000 K=4096 N=1024
MARLIN-INSTR ... launch=[quant.transpose K=4096 N=1024] ... ctx=0x587b95ed80b0 in_marlin_quant=1
MARLIN-INSTR ... launch=[quant.scales   K=4096 N=1024] ... ctx=0x587b95ed80b0 in_marlin_quant=1
MARLIN-INSTR ... launch=[quant.pack     K=4096 N=1024] ... ctx=0x587b95ed80b0 in_marlin_quant=1
   <<< HANG — no quantize_repack EXIT; GPU pegs 100% and stays >>>
```

**Read it against the user's decision tree:**

| # | Question | Answer from the log |
|---|---|---|
| 1 | last Marlin **GEMM** launch before the hang | `marlin.gemm` (weight 2), `grid=(132,1,1)` |
| 2 | grid dimensions of that kernel | **(132,1,1)** — 132 blocks, block (256,1,1), shmem 98304 (96 KiB) |
| 3 | SM count of the context it launched in | `ctx=0x587b98689cf0` == the green context — **8 SMs** |
| 4 | quant kernel or GEMM kernel? | the **GEMM kernel** (`marlin.gemm`, via `marlin_gemm_launch`) |
| 5 | launch → hang | GEMMs at monotonic t≈218746.732–.744 s; GPU pegged 100 % indefinitely thereafter |

The quant kernels launch in `ctx=0x587b95ed80b0` (`in_marlin_quant=1`) — **the
primary context. Fix (i)'s `PrimaryCtxGuard` works** — the quant path is
correctly pinned. The hang is **not** the quant path.

---

## 3. Confirmed hypothesis

**The Marlin GEMM kernel — `grid=(132,1,1)`, persistent-style with inter-CTA
`locks` split-K coordination — is launched into the green context, which has
only 8 SMs.** 132 blocks × 96 KiB shared mem ⇒ ~16 blocks resident (2/SM × 8);
116 cannot be scheduled. Marlin's inter-CTA split-K protocol requires its CTAs
co-resident — resident CTAs spin waiting on `locks` signals from CTAs that can
never be scheduled. **Inter-CTA deadlock. GPU pegs at 100 % forever.**

The GEMMs are launched async (`stream=(nil)`, on the green context's NULL
stream); `dispatch EXIT rc=0` — the launch returns. The hang then surfaces at
the **next device-wide synchronizing call** — weight 3's `quantize_repack`
`cudaMemcpy`/`cudaFree` — which waits on the wedged GPU. The hang's *location*
(weight 3 quant path) is a downstream symptom; the *cause* is the green-context
GEMM launches (weights 1 and 2).

**Why it appeared only at kmod 0.4.8:** pre-fix, `/dev/cipher` was EPERM →
CUPTI never subscribed → its launch callback never ran → the green context was
never made current (`ctx_swaps_to_green=0`) → the Marlin GEMM ran in the
primary context on all 132 SMs. test A passed for that reason. The kmod fix
(correct, adjudicated) activated CUPTI → the T4.2.4d green-enforcement →
the GEMM now lands in the 8-SM green context.

This is an **architecture-level incompatibility**, not a coding bug: Marlin's
GEMM kernel structurally assumes the full SM array (`grid = SM count`, all CTAs
co-resident); green-context SM partitioning gives it 8. They cannot compose
as built.

---

## 4. Candidate fix shapes

**Fix A — extend the primary-context pin to the GEMM dispatch (recommended).**
Wrap `cipher_rt_marlin_engine_dispatch` in the same `PrimaryCtxGuard` already
used for the quant path. The Marlin GEMM then launches in the primary context
→ `grid=132` on all 132 SMs → all CTAs co-resident → no deadlock. Same
mechanism as fix (i), scope extended from "quant path" to "all Marlin engine
GPU work." Small. Restores test A. Implementation note: the GEMM is meant to
launch on the caller's stream — observed `stream=(nil)` here; if a non-NULL
caller stream bound to the green context is ever passed, the pin must also
re-home the launch onto a primary-context stream (else stream/context
mismatch). Consequence: Marlin always runs full-GPU, never SM-partitioned.

**Fix B — `grid` = the current context's SM count (not always 132).**
`marlin_gemm_launch` would query the active partition (8 in a green context)
and launch `grid=8`. HIGH RISK: Marlin's split-K factor and `locks` usage are
tied to the grid size; `grid≠SM-count` is unverified for numerical
correctness. Needs deep validation before trust.

**Fix C — architectural statement (pairs with A).**
Marlin's GEMM kernel and green-context SM partitioning are mutually exclusive.
CP 2.4 is single-tenant → Marlin runs full-GPU (Fix A), correct and desired.
Phase 5 multi-tenant (substrate + actuators composed) must either exclude
Marlin-routed tenants from green partitioning, or invest in a partition-aware
Marlin kernel rewrite (large). Tracked as a Phase 5 constraint.

**Recommendation:** Fix A for CP 2.4 (unblocks the DVFS sweep and the Marlin
sub-task closure; small; consistent with the adjudicated fix (i)) + record
Fix C as the Phase 5 architectural constraint. Fix decision is the user's.

---

## 5. Artifacts / anchors

| Build | md5 | role |
|---|---|---|
| libcipher_rt original | `a0d6cddacb116f2d51ad1d1f867ef564` | test A baseline; rollback |
| substrate-gate diag | `55e2e3233c3b92506d1dd2f573b852c4` | failed fix — diagnostic only |
| primary-ctx pin, fix (i) | `caae0bb88650c1121b9f92b1eda00e22` | failed fix (quant path pinned; GEMM not) — artifact |
| **instrumented diag** | `b1a3424c07f8d0ebb36135f3d9130884` | this confirmation build — **not a campaign anchor** |

Held: kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`. Forward libcipher_rt
build = `a0d6cdda` + the real fix (Fix A, pending adjudication). No CP 2.4
report. `MARLIN_HANG_ROOT_CAUSE.md` to be written once the fix ships.
