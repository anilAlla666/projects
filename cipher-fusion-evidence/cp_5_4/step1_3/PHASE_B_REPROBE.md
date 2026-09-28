# CP 5.4 Step 1.3 — Phase B: Q3 probe re-run on corrected kmod — REPORT

**Date:** 2026-05-18. **Verdict: PASS.** Split-order determinism re-confirmed
on the 15-group kmod (`8d777dfb`). Group count 15. The authoritative
ground-truth signal (`%smid` physical placement) is byte-identical to Phase
1.3a. One secondary observation — `structHash` cross-run variance — is
documented below, explained, and shown non-load-bearing; it is **not** a gate
failure (it is the exact case the probe's `%smid` disambiguator was built for).

---

## Method

Re-ran the unchanged Phase 1.3a probe binary (`cp54_splitorder_probe`) on the
corrected kmod. (The probe calls `cuDevSmResourceSplitByCount` directly — it is
CUDA-driver-side and does not depend on the kmod; re-running it confirms the
hardware split is unchanged and re-validates determinism.) Phase 1.3a outputs
preserved as `probe_1_3a_proc_{0,1}.txt`.

## Result

| check | result |
|---|---|
| group count (`nb_groups`) | **15** (+ 12-SM remainder) ✓ |
| determinism — `diff probe_proc_0 probe_proc_1` (within-run, 2 concurrent procs) | **byte-identical** ✓ |
| `%smid` ground truth vs Phase 1.3a | **byte-identical** — group g → same physical SMs ✓ |
| `structHash` vs Phase 1.3a | **differs** (see below) |

The `%smid` set for every group is identical to Phase 1.3a — group 0 →
{0,1,16,17,32,33,48,49}, … group 14 → {76,77,90,91,104,105,118,119}. Physical
SM placement per group index is stable across processes **and** across runs.

## The `structHash` cross-run observation — explained, non-load-bearing

The FNV hash of the raw `CUdevResource` struct bytes is **identical between the
two concurrently-forked processes within a run** (the `proc_0 ≡ proc_1` diff is
empty in both the 1.3a run and this run) but **differs between the two separate
runs** (1.3a vs Phase B).

**Cause.** `CUdevResource` is an opaque struct that embeds a process-private
address/handle. `fork()` duplicates the parent's address space, so two forked
children share one ASLR layout → identical struct bytes. Two *separate* `exec`s
of the probe binary get independent ASLR → the embedded address differs → the
struct hash differs. The 8 lower-bit `sm.smCount` and the partition itself are
unaffected.

**Why it is not a gate failure and not an ambiguity.**
- This is exactly the case the Phase 1.3a design anticipated, verbatim: *"the
  opaque bytes might include a process-local handle, making memcmp give a false
  FAIL even when placement is identical"* — which is precisely why the probe
  carries a `%smid` ground-truth check, and `PHASE_1_3A_PROBE.md` designated
  `%smid` the **authoritative** signal and `structHash` a non-authoritative
  corroborating one.
- The authoritative signal (`%smid`) is unambiguous: identical across all four
  process-instances spanning both runs. There is no ambiguity to resolve.
- The grp_mask bridge **cannot** be affected: the kmod transports a `grp_mask`
  (a bitmask of integers), never a `CUdevResource` struct. Each client process
  calls `cuDevSmResourceSplitByCount` itself and selects groups by index. No
  struct ever crosses a process boundary or is cached across a process
  lifetime. Group *index* g → same physical SMs (the `%smid` proof) is the only
  property the bridge needs, and it holds.
- Concurrent tenants — the real use case — get identical structs anyway
  (fork-shared ASLR). The variance is purely across separate program runs,
  which the bridge never compares.

## Verdict

- Determinism: **PASS** (within-run byte-identical; `%smid` placement identical
  to Phase 1.3a).
- Group count: **15** — matches the corrected kmod.
- `structHash` cross-run variance: a documented, fully-explained, non-load-
  bearing observation — the probe's designed-in `%smid` disambiguator resolves
  it; the authoritative criterion passes unambiguously.

**Phase B PASS → proceeding to Phase C (libcipher_rt client build).** Anchors
unchanged this phase (kmod `8d777dfb`, libcipher_rt `a7ac8e97`).
