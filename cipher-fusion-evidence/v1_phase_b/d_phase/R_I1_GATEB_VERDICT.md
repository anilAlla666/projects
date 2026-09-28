# R-I1 SHIELD via SM-PARTITION — Gate B VERDICT: **OUTCOME 3 (NOT-ENFORCED), 1-H100 ceiling**

**Date:** 2026-05-29. bf16, clean diag2 harness (`d8_ri1_gateb.py`), CP54 green-ctx reused (not
rebuilt). **Anchor NOT rotated.** The disjoint SM-partition gave the latency victim **zero**
isolation from a saturating neighbor — the pre-registered OUTCOME 3. Measured, not inferred.

## Gate B — victim wall p99 (ms), GPU-bound probe vs saturating 8192³ aggressor

| condition | victim p50 | victim p99 | note |
|---|---|---|---|
| C1 victim SOLO, full GPU | 0.033 | **0.049** | floor |
| C2 victim SOLO, 8-SM partition | 0.097 | **0.111** | `cipher-t424e` absolute-SM-cost ref (~2.3×) |
| C3 victim + aggressor, NO partition | 1.908 | **1.930** | contention baseline (39.5× vs C1) |
| C4 victim + aggressor, **DISJOINT partition** | 1.942 | **1.959** | **40.1× vs C1 — NO improvement over C3** |

- **Disjoint partitions CONFIRMED:** victim mask `0x1000` (8 SMs), aggressor mask `0xfff` (96 SMs),
  A∩B=∅ = True; both green ctxs engaged (`init=1`, sm_count 8 / 96). This is not a setup miss.
- **Aggressor confined to its OWN execution but not isolated from the victim:** aggressor matmul/s
  dropped 612 (no partition) → 494 (96-SM partition) ≈ 96/132 — so the green mask shapes the
  aggressor's own SM usage. **Yet C4 victim p99 (1.959) ≈ C3 (1.930)** — the partition delivered no
  tail relief. Empirical confinement **REFUTED** (per the required measurement: C4 ≉ C2-solo).

## Root cause — cross-process time-slicing without MPS (the `cipher-t424c` failure, understood)

Green-context CU masks partition SMs **within a single process's context**. Across **separate
processes** (two CUDA contexts) on a bare GPU **without CUDA MPS**, the driver **time-slices the
whole GPU** — it hands all 132 SMs to one context per scheduling quantum (~1.9 ms here, exactly the
observed p99). So the victim's op waits a full aggressor quantum for its turn **regardless of
disjoint CU masks** — the masks shape *which SMs each context uses during its slice*, not *whether
the contexts run concurrently*. Green-ctx alone therefore cannot provide cross-tenant isolation;
the 40× tail is the time-slice quantum, untouched by partitioning. (Consistent with
`cipher-t424c` partition-not-enforced and `cipher-t424e` "isolation is variance reduction, not p99
reduction" — now with the precise mechanism: no cross-process concurrency without MPS/MIG.)

## Verdict + disposition (STOP for Anil)

**OUTCOME 3 — NOT-ENFORCED on bare green-ctx → 1-H100 cross-tenant isolation CEILING.** Neither
D.8 throttle-band (frequency-reducer, can't isolate the tail — `D8_CLOSE_REPORT` §3.5) nor R-I1
bare green-ctx SM-partition (defeated by cross-process time-slicing) bounds a latency tenant's p99
under a saturating neighbor on a single shared-context H100. The 40× noisy-neighbor p99 gap is
**real and remains open**; the lever is not in CIPHER's user-space substrate alone.

- **Anchor NOT rotated.** KL=0 moot (no close; the partition alters only which SMs run, not output).
- **No fakes, no tuning.** Disjoint masks + engaged partitions + aggressor-work-held were all
  measured; the isolation simply did not materialize.

**Measured vs inferred (kept distinct):** *Measured + load-bearing:* disjoint partitions → ZERO
isolation (C4≈C3, both ≈40×; C2-solo=0.111 ms, so if the partition worked C4 would sit near C2, not
C3). *Inferred (consistent, not directly instrumented):* the ~1.9 ms tail = the cross-process
time-slice quantum. The verdict rides on the measured null, not the quantum explanation.

**Escalation (pre-registered OUTCOME-3 path — all options need a design-memo + validation first):**
1. **CUDA MPS — the likely lever, but a HYPOTHESIS, not an expectation.** MPS lets clients share one
   context and run **concurrently**; whether that yields **p99 isolation** (vs mere throughput
   sharing), and whether the existing CP54 green-ctx **composes** with MPS or needs new wiring /
   `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` provisioning, are **both unverified**. Testable on this H100
   (start `nvidia-cuda-mps-control`, re-run this exact Gate B, same 3 outcomes). Do not claim it
   works until measured — same standard we applied to the throttle.
2. **MIG** — hardware partitions, true isolation, BUT **caps at 7 instances on an H100** → cannot
   give 100 agents partitions. Usable only for ≤7 large tenants; not a Goal-1-scale lever.
3. **Multi-GPU** — sidesteps co-tenancy; not single-H100.

⇒ **For Goal-1's 100-agent target, MPS is effectively the ONLY scalable cross-tenant isolation
lever** (MIG ≤7, multi-GPU sidesteps). So the entire single-H100 isolation story now hinges on an
**unvalidated MPS+green-ctx composition** — worth stating plainly, and worth validating before V.1
relies on it.

**The actual fork for Anil (don't bury it):**
- **(A) Validate MPS now** (proactive; one run: `nvidia-cuda-mps-control` + re-run this Gate B).
- **(B) Defer to the V.1 soak** and escalate only if the p99 gap manifests under realistic load.
  Rationale: the 40× came from a **synthetic continuous FLOP-flood** worst case; D.8 §4 found the
  H100 **self-arbitrates at Goal-1's realistic ~2-active-of-100-at-2%-duty**, so it is genuinely
  open whether this gap even bites in the real soak. If it doesn't, MPS may be unnecessary.

Both are legitimate. If (A): pre-scoped to **ONE run** (no harness spiral — v1/v2/v2b/diag already
done). This is Anil's call.

Artifacts: `ri1_gateb_result.json`, `ri1_gateb.log`, `d8_ri1_gateb.py`.
