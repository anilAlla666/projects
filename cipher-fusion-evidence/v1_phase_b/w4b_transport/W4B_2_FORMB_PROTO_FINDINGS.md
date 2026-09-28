# W.4b.2 — Form B′ cross-tenant batching PROTOTYPE — FINDINGS

**Date:** 2026-05-28. **Anchors (UNCHANGED — no substrate/kmod touch):**
cipher_rt_phase4 `8b5e928` (w4a-pool-eligibility, libcipher_rt.so md5 `9fe23143`);
cipher_kmod `02fc2d1` (0.6.6, NR 30/31). This step is **all userspace harness**
(Python + cuIPC + ctypes into the existing audited guard); the W.4a guard and the
co-residence kmod are reused verbatim.

**Scope adjudication (Anil, 2026-05-28).** "continue" → adopt memo §8 dispositions
(a) Form B′ + (b) warmup deployment-match + (c) the 7-item plan. The item-3
architecture fork (memo-literal tenant-owned-KV + cuIPC-mapped peer-KV attention
*cannot* use HF's batched forward — HF reallocates KV each step) was surfaced and
adjudicated to **option (i) executor-owned-KV**: the executor holds the batched
KV and runs HF's B=N forward over cuIPC-gathered per-token embeds. This measures
the **activation-transport + rendezvous** half of R-2. The cuIPC-mapped peer-KV
attention (the R-4 surface) is carved to **W.4b.2b**.

## 0. What was built (userspace only)

| file | role |
|---|---|
| `formb_ipc.py` | cuIPC activation channel (cipher_kv_bridge VMM arenas, SCM_RIGHTS fd-pass, persistent per-token reuse) + JSON control signals |
| `formb_cohort.py` | raw NR 31 cohort QUERY (caller_fp=0, query-only) + ctypes into the **audited** `cipher_rt_pool_partition` (no second copy of the guard) |
| `formb_warmup_fp.py` | one-time INJECTED warmup → executor's deployment-match W.6 fp (memo §2 #2a) |
| `formb_tenant.py` | INJECTED tenant: warm→register fp; per-token embed→cuIPC ship→barrier→recv logit row→sample; solo fallback on gate-deny; teacher-forced KL |
| `formb_executor.py` | VANILLA executor: gate → batched HF B=N forward over cuIPC-gathered embeds (executor-owned KV) → scatter logit rows |
| `formb_baselines.py` / `run_formb_proto.sh` / `formb_measure.sh` / `formb_smoke_ipc.py` | baselines, harness, measurement, cuIPC round-trip smoke |

cuIPC round-trip smoke (`formb_smoke_ipc.py`): **PASS** (per-token overwrite +
re-read across processes is coherent).

## 1. Gate — both directions PASS

| MODE | tenants | cohort partition (audited guard) | outcome |
|---|---|---|---|
| **positive** | 2× TinyLlama | `eligible=1 n_in_group=2 distinct_rejected=1*` matched_rows=[0,1] | **batched B=2 ran** |
| **negative** | TinyLlama + Llama-3.2-1B | `eligible=0 n_in_group=1 distinct_rejected=2` peer_tgids=[tenant0] | **no gather; both solo** |

executor_fp `0xe45c45a8a8533381` (TinyLlama). tenant fps: TinyLlama
`0xe45c45a8a8533381` (match), Llama-3.2-1B `0x718645ea2fcbb581` (distinct).
*The positive `distinct_rejected=1` is a stale warmup-process entry (registered at
the pre-stabilization fp `0xc37e…`, exited, lingers ≤30 s before kmod prune);
filtered out by the hellos-intersection so it does not affect `matched_rows`.

## 2. Correctness — PASS (exact token match)

**Primary gate: exact greedy token match — 64/64 for BOTH tenants.** The
cross-process batched-B=2 decode produces a byte-identical token sequence to each
tenant's single-tenant solo B=1 decode. cuIPC fp16 activation transport is
lossless.

Supporting (teacher-forced per-step logit-KL, gold-trajectory): tenant0
`kl_mean=5.0e-6 kl_max=4.9e-5`; tenant1 `kl_mean=6.7e-6 kl_max=1.17e-4`. The
**mean** is well under the CP 5.6 5e-5 figure; the **max** (sequence tail) exceeds
it because this is an *incremental-decode-KV* comparison — B=2 vs B=1 GEMM
reduction-order differs and accumulates in the executor-owned KV over 64 steps.
This is **argmax-stable** (tokens lock 64/64) and is a *stricter* test than the
smoke's single-full-forward teacher-forcing. Not a correctness defect.

## 3. Negative control — the Form-B value proof (which Form A cannot express)

tenant_1 = a genuinely different model (Llama-3.2-1B). The **audited W.4a guard
refused to coalesce it**: `distinct_rejected≥1`, `eligible=0`, no activation
gather. Both tenants ran solo and produced correct output (TinyLlama tenant
byte-identical to its positive-run output; Llama tenant its own correct tokens).
This is the Memory #11 catastrophic guard doing **real admission work** on real,
independent model processes — the test a one-model Form-A executor structurally
cannot pose.

## 4. Plumbing gap (N=2, per-token-barrier granularity), TinyLlama, GEN=64

| | tok/s | mean W | tok/W |
|---|---|---|---|
| cross-process batched B=2 | 18.3 | 100.7 | **0.182** |
| in-process B=2 ceiling | 44.0 | 83.3 | 0.528 |
| naive 2-concurrent | 43.4 | 91.7 | 0.473 |

- **plumbing gap = cross/ceiling tok/W = 0.345×** (cross-process reaches 34.5% of
  the in-process B=2 ceiling). Per-step cost ≈ 109 ms cross-process vs 45 ms
  in-process ⇒ **~64 ms/step of rendezvous overhead** (2 tenants × socket
  round-trip + cuIPC write-sync + read, plus redundant `torch.cuda.synchronize`).
- **substrate-attributable = cross/naive tok/W = 0.385×** (below 1.0).

**Load-bearing caveat — N=2 is below the lever regime.** The **in-process lever at
N=2 is only 1.115×** (ceiling/naive) on this eager manual decode loop. CP 5.6 §1
measured 1.88× at N=2 historically; this harness is ~3× slower per step
(Python/launch-overhead-bound, not GPU-launch-bound), so the batching lever is
barely present *in-process* at N=2. **The lever lives at N=8–16** (CP 5.6 §1:
~3.8× / ~7× substrate-attributable). So `0.385×` must **not** be read as "lever
eaten" — at N=2 in this harness there is almost no lever present to eat. The
prototype cannot, by construction, decide the engagement-gate (3.69× at N=8).

## 5. What this prototype does NOT measure (carry-forward, do not misread as tested)

- **R-3 peers' residual shim cost — NOT exercised.** Under option (i) the tenants
  are thin embed+sample shells; they run no model-body GEMMs per token. The
  100.7 W cross-process draw reflects 3 processes (executor + 2 thin tenants)
  idling, not the memo's peers-running-attention scenario.
- **R-4 cuIPC-mapped peer-KV attention — deferred to W.4b.2b** (net-new; the
  memo-literal Form B′). Here the KV is executor-owned.
- **N=8 engagement gate, Mistral (E.7), the scheduler/heterogeneous lengths/crash**
  — all out of prototype scope (W.4b.4 / W.4b.5).

## 6. R-2 disposition — engagement-gate UNDECIDED (NOT-MET not triggered)

The memo's pre-committed NOT-MET branch fires "if the gather + peers' residual
shim eats the lever … W.4b closes NOT-MET **on the engagement gate**" — i.e. at
N=8/3.69×. **That precondition does not apply at N=2 in this harness** (no lever
present). So this prototype neither inflates to PASS nor deflates to a premature
NOT-MET. Honest disposition:

- **Architecture + safety pre-conditions: PROVEN.** Real per-process tenants with
  real fingerprints; audited guard admits same-fp / rejects distinct-fp; lossless
  cross-process activation transport; exact-token correctness through the batched
  path.
- **The engagement-gate question is undecided here.** The actionable result is the
  **plumbing-cost-per-step (~64 ms at N=2 barrier granularity)**, dominated by the
  per-token rendezvous + redundant syncs — the exact cost **W.4b.4 must amortize**
  (batched signaling, sync elision, pipelined/CUDA-graph rendezvous) before the
  N=8 gate run (W.4b.5) can clear 3.69×.

## 7. Verdict + adjudication (next checkpoint)

**Prototype VERDICT: PASS on its two primary deliverables** — (i) the cross-process
plumbing gap is measured (0.345× of ceiling at N=2 barrier granularity, ~64 ms/step
overhead) and (ii) the negative control proves the guard does real work. Correctness
(exact token match) and the safety guard are proven. No HARD STOP; no substrate
touched.

Awaiting adjudication:
- **(a) [recommended]** Proceed → W.4b.3 (first-coalesce fail → `mark_blocked` →
  ratio auto-disable) → W.4b.4 (scheduler + **amortized rendezvous** + N=4) →
  W.4b.5 (N=8 engagement gate + close report). The prototype proved the
  safety/transport pre-conditions; the per-step plumbing cost is the next target.
- **(b)** Close W.4b NOT-MET now — *not warranted*: the NOT-MET precondition
  (lever eaten at N=8) does not apply at N=2 in this harness.
- **(c)** Insert an intermediate measurement (option-(i) prototype at N=8 with no
  real scheduler) before W.4b.3 — informative but scope-creeps the prototype and
  pre-empts W.4b.4's amortization work.

Sub-steps after this remain as memo §4: W.4b.2b (R-4 peer-KV attention) /
W.4b.3 / W.4b.4 / W.4b.5. Mistral E.7 gates W.4b.5 only.
