# Extended 3-Arm Benchmark — Operator-Realistic Operating Points

**Date:** 2026-05-18. WL01 (TinyLlama-1.1B), greedy, 128 tok/prompt, 5 prompts.
Single H100. Anchor `a7ac8e97`. Status: **OP1 complete — STOP / ADJUDICATE
before OP2–4.**

## OP1 — N=8 tenants × B=2 (effective fused batch 16)

3 arms, 2 reps each. Arm 1 = naive HF (8 procs × B=2); Arm 2 = vLLM 0.21.0
intra-process (8 instances, `max_num_seqs=2`, `enforce_eager`); Arm 3 = CIPHER
cross-process fusion (16 streams → one B=16 batch).

| arm | rep 1 | rep 2 | **mean tok/W** | tok/s | power | correctness |
|---|---|---|---|---|---|---|
| Arm 1 — naive HF B=2 | 1.485 | 1.344 | **1.415** | ~249 | ~176 W | teacher-forced KL ≈ 0 ✓ |
| Arm 2 — vLLM × B=2 | 3.602 | 3.605 | **3.603** | ~1001 | ~278 W | exact-match 99.96–100% ✓ |
| Arm 3 — CIPHER eff-16 | 6.693 | 6.569 | **6.631** | ~987 | ~149 W | teacher-forced KL_max 3.1e-5 ✓ |

**Lifts:** Arm3/Arm2 = **1.84×**, Arm3/Arm1 = 4.69×, Arm2/Arm1 = 2.55×.
(No stop condition triggered — Arm3/Arm2 = 1.84 > 1.5; Arm 2 ran after a
clean-GPU retry, see Notes.)

## The result: the ratio is flat — this is the *plateau*, not the compounding

| operating point | Arm 3 / Arm 2 |
|---|---|
| floor — N=8 × B=1 (eff batch 8) | 1.75× |
| **OP1 — N=8 × B=2 (eff batch 16)** | **1.84×** |

The hypothesis behind the extended benchmark was that N=8×B=1 is "the floor"
and CIPHER's advantage **compounds** with effective batch. **OP1 does not
support that.** B=1→B=2 moves Arm3/Arm2 only 1.75→1.84 (+5%, barely above
measurement noise).

**Why — vLLM rides the batch curve too.** At B=1, vLLM had nothing to batch
intra-process (Arm 2 floor = 1.91). At B=2 vLLM's continuous batching engages,
and Arm 2 jumps **1.91 → 3.60 (1.88×)**. CIPHER's Arm 3 over the same step
goes **3.35 → 6.63 (1.98×)**. Both arms roughly double; the *ratio* is
preserved. CIPHER fuses to a bigger effective batch (16) than vLLM's
per-instance batch (2) — but on TinyLlama's tok/W-vs-batch curve both are
still climbing, and vLLM captures the batching benefit per instance just as
CIPHER does across instances. The cross-process-fusion *advantage* does not
widen.

Extrapolating the +0.09-per-B-doubling drift: OP3 (eff-32) ≈ 1.93×, OP4
(eff-64) ≈ 2.0× — **the trajectory does not reach the 2.5× the brief set as
the "compounding" threshold.**

## What CIPHER's ~1.8× actually is — power consolidation (confirmed again)

At OP1, Arm 2 (vLLM) has marginally *higher* aggregate throughput than Arm 3
(1001 vs 987 tok/s) — but draws **~278 W vs CIPHER's ~149 W**. The 1.84× is
≈ power-ratio (278/149 = 1.87×) × throughput-ratio (987/1001 = 0.99). It is
the **same power-consolidation win** identified at the floor: one CIPHER
process vs eight vLLM engine processes. It is stable across B because process
count (1 vs 8) and therefore the power gap is B-independent, while throughput
scales with B in both arms.

This is the outcome the brief named: *"if the lift plateaus at ~1.5–2× — the
substrate's advantage is the power-consolidation win; the Peak-XV pitch
becomes architectural + DVFS-power-efficiency, not a scaling-batching
advantage."* OP1 (and the floor) put the plateau at **~1.8×**.

## Carried confound (unchanged from the floor)

Arm 3 is CIPHER's cross-process fusion on an **HF-`generate()` executor**, not
"CIPHER substrate over vLLM" (`FUTURE_SCOPE/A`, not built). The 1.84× mixes a
batching gain with an HF-vs-vLLM engine deficit. Note the direction: CIPHER's
~1.8× win is achieved *despite* its executor using the weaker engine — the
power-consolidation effect is strong enough to overcome the engine gap. The
clean architectural number still needs `FUTURE_SCOPE/A`.

## Notes

- **vLLM 8-instance operational fragility (a finding in itself).** OP1 Arm 2's
  first attempt failed 6/8 — leaked `VLLM::EngineCore` zombie subprocesses
  from prior runs (vLLM v1 does not reap them on parent exit) had filled
  ~74 GB. After killing zombies by PID and adding pre/post cleanup to the
  harness, the retry ran clean 8/8. Running one engine per tenant does not
  co-tenant reliably on a single GPU — which supports the consolidation
  argument.
- **Arm 1 variance.** Naive-HF B=2 reps spread ~10% (1.34–1.49); one rep also
  had 7/8 tenants fail. 8-concurrent HF at B=2 is itself noisy/fragile. The
  key ratio (Arm3/Arm2) does not use Arm 1 and is solid (Arm 2 reps agree to
  0.1%, Arm 3 to 2%).

## Recommendation — adjudicate before OP2–4

OP1 already answers the benchmark's core question: **CIPHER's
cross-process-fusion advantage does not compound with per-tenant batch — it
is a stable ~1.8× power-consolidation win.** Running OP2 (N=4×B=4), OP3
(N=8×B=4, eff-32) and OP4 (N=8×B=8, eff-64) — ~3 GPU-hours — would refine the
curve, but the floor + OP1 trend (1.75 → 1.84) does not project to ≥2.5×.

Two paths to adjudicate:
1. **Accept the plateau read now.** The defensible-Thursday claim is: CIPHER
   delivers a stable ~1.8× tok/W advantage over the realistic vLLM-per-tenant
   alternative, via process consolidation (power) — *not* a batch-scaling
   advantage. Reframe the pitch as architectural + power-efficiency, per the
   brief's own plateau branch.
2. **Run OP3 only** (the highest single-GPU effective-batch point, eff-32) as
   one confirming measurement rather than the full OP2–4 sequence — if OP3
   Arm3/Arm2 is still ~1.8–2.0×, the plateau is confirmed; if it jumps toward
   2.5×, re-open the compounding hypothesis.

Recommended: path 2 if a confirming point is wanted cheaply, else path 1.
The compounding hypothesis is, on the OP1 evidence, not holding.

## Artefacts

`phase_c/arm1_op1_rep{1,2}/`, `arm2_op1_rep{1,2}/`,
`phase_b/session1/arm3_op1_rep{1,2}/`. OP2–4 sections pending adjudication.
