# CP 5.4 Step 1.3a — split-order determinism probe — REPORT

**Date:** 2026-05-18. **Probe verdict: split-order determinism PASS (clean).**
**But the probe surfaced a non-trivial mismatch — HARD STOP before Phase 1.3b
per global discipline ("any phase surfaces unexpected behavior → STOP and
adjudicate"). No libcipher_rt source modified. Anchor `a7ac8e97` unchanged,
both fallback copies intact.**

---

## Purpose

Verify scope memo §2's load-bearing assumption — *"kmod group g ↔ the same
physical SMs in every process"* — which the grp_mask bridge rests on and which
Step 1.2 was claimed to (but did not) verify.

## Method

`cp54_splitorder_probe.cu` (throwaway, ~140 LOC, `nvcc -arch=sm_90`). Forks two
processes **before any CUDA init**; each independently:
- `cuInit` → device 0 → `cuDeviceGetDevResource(SM)`
- `cuDevSmResourceSplitByCount(minCount=8, useFlags=0)` — identical call to
  `cipher_rt_green_ctx.c`
- per group: (1) FNV-1a hash of the raw `CUdevResource` struct (groups[] memset
  to 0 first, so padding cannot cause a false mismatch); (2) **ground truth** —
  green ctx restricted to that group, launch `4096×64` grid reading `%smid`,
  collect the set of physical SM ids the partition actually used.

Parent diffs the two children's per-group output.

## Result — determinism: PASS

`diff probe_proc_0.txt probe_proc_1.txt` → **byte-identical.** For all 15
groups, both the `CUdevResource` struct hash **and** the ground-truth `%smid`
set match across the two processes. Split order is process-deterministic — the
grp_mask bridge's load-bearing assumption holds.

```
device_sm_count 132
nb_groups 15 remaining_sm 12
g  0  smCount 8  structHash deb1d50ca0145bf8  smids 0,1,16,17,32,33,48,49
g  1  smCount 8  structHash bef7098a30e8c36a  smids 2,3,18,19,34,35,50,51
 ...  (g 2..7 — low cluster, smids 0..63)
g  7  smCount 8  structHash 29e5fc82a4a3194b  smids 14,15,30,31,46,47,62,63
g  8  smCount 8  structHash ffbe035067b633f7  smids 64,65,78,79,92,93,106,107
 ...  (g 9..14 — high cluster, smids 64..119)
g 14  smCount 8  structHash 5c08af5b7c3efbc6  smids 76,77,90,91,104,105,118,119
```
(Both processes produced exactly the above. Full files: `probe_proc_0.txt`,
`probe_proc_1.txt`.)

## THE FINDING — the H100 yields 15 groups, not 16

`cuDevSmResourceSplitByCount(minCount=8)` on this H100 80GB SXM5 returns
**`nb_groups = 15`, `remaining = 12` SMs** — 15×8 + 12 = 132. It does **not**
return 16 groups + 4 remainder.

The hardware does not pack into 16 equal 8-SM groups: the split is **8 groups
over physical SMs 0–63, then 7 groups over SMs 64–119**, with SMs 120–131 (12)
unallocatable — almost certainly a GPC/TPC clustering constraint. 16×8 = 128
contiguous 8-SM groups is simply not achievable on this part.

**This contradicts the substrate as built/specified:**
- Scope memo §1/§2: *"132 SMs split into 16 groups of 8 SMs … 4 remainder."*
- Kmod `cipher_cp54_sched.c`: **`CIPHER_CP54_NUM_GROUPS = 16`** — the CP 5.4
  ledger (anchor `7f467de4`, built Steps 1.1–1.2) has **16 group slots**.
- Client `cipher_rt_green_ctx.c`: `CIPHER_RT_GREEN_NUM_GROUPS = 16` (a comment
  + an array size; the live code already uses the API's returned count, so the
  *existing* a7ac8e97 hash-pick is unaffected — it picks among the real 15).

**Why it matters.** The kmod ledger has a **phantom group 15** with no hardware
backing. The kmod will hand it out:
- `ALLOCATE(POOL)` does `cp54_claim(pid, 16)` → POOL owns groups 0–15, mask
  `0xFFFF`. Bit 15 is real in the ledger, nonexistent in hardware.
- A `PARTITION` summing >15 groups, or a near-full ledger, likewise reaches
  group 15.
- `QUERY` counts (`free_grp_count`, `pool_grp_count`) are all 16-based.

A client that receives a `grp_mask` with bit 15 set, then calls
`cuDevSmResourceSplitByCount` (→ only 15 groups, indices 0–14), would index
`groups[15]` out of bounds → invalid green context.

**Scope of impact on Step 1.3 as descoped.** Step 1.3 = SHARED + *single*
PARTITION tenant. A lone PARTITION asking ≤120 SMs gets low-indexed groups
(0,1,2,…) — all real; the phantom group 15 is only reached by POOL (Step 1.3b′)
or a near-full multi-tenant ledger (Step 1.4). So Step 1.3's *test path* would
not hit the phantom — **but the kmod ledger is nonetheless objectively wrong**,
and silently shipping a `NUM_GROUPS` that overstates the hardware is exactly the
class of latent error the per-phase STOP gate exists to catch.

## Verdict & adjudication ask

- **Probe determinism check: PASS** — clean, unambiguous (struct hash + `%smid`
  ground truth agree across processes). §2's assumption is verified.
- **Surfaced issue (non-trivial):** kmod `CIPHER_CP54_NUM_GROUPS = 16` vs. the
  hardware's 15. Fixing it is a **kmod source change → rebuild → reload** —
  i.e. it reopens Step 1.1/1.2, outside Step 1.3's libcipher_rt-only scope.

**HARD STOP before Phase 1.3b.** No libcipher_rt modification. Anchor
`a7ac8e97` unchanged; `7f467de4` kmod loaded and unchanged. Options for
adjudication:

1. **Correct the kmod to 15** — `CIPHER_CP54_NUM_GROUPS = 15` (one constant),
   rebuild kmod, re-run the Step 1.2 isolation suite (its "all 16 free"
   assertions become "all 15"), rotate the kmod anchor. Clean but reopens the
   kmod and is a mini Step 1.1/1.2 cycle.
2. **Make the kmod hardware-aware** — kmod queries/accepts the real group count
   rather than hard-coding it. Larger change; arguably correct long-term;
   heavier than v1 warrants.
3. **Proceed with Step 1.3 (SHARED + single PARTITION) on the 16-group kmod
   as-is**, since Step 1.3's descoped test path provably never reaches group
   15, and fold the kmod 16→15 correction into Step 1.3b′ (POOL — which *is*
   where the phantom first bites). Documents the constraint, defers the kmod
   touch to when it is on the critical path anyway.

Recommendation: **Option 3** for tonight — it keeps Step 1.3 within its
adjudicated libcipher_rt-only scope and does not reopen the kmod at hour 18+;
the phantom group is unreachable on the Step 1.3 path and Step 1.3b′ cannot
proceed without confronting it regardless. But this is a substrate-correctness
call — surfacing for the user, not deciding it.
