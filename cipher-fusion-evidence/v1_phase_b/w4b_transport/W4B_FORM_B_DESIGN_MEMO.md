# W.4b — cross-tenant POOL transport — Form B DESIGN MEMO

**Date:** 2026-05-28. **Type:** design, no substrate code (memo → approve →
build, per [[cipher-phase-discipline]]). **Decision recorded this session
(Anil, AskUserQuestion):** W.4b proceeds as **Form B** (per-process model
tenants, real fingerprints, the Memory #11 guard does real work) — *not* the
Form A re-port the entry note defaulted to. Within Form B, the engagement-gate
arithmetic (footnote A) selects the **B′ dedicated-vanilla-executor** variant
over the pure §2.2B elected-tenant form.
**Anchors (UNCHANGED):** cipher_rt_phase4 `8b5e928` (w4a-pool-eligibility),
libcipher_rt.so md5 `9fe23143`; cipher_kmod `02fc2d1` (0.6.6, NR 30/31
co-residence registry). **Predecessor:** W.4b.1 smoke CLOSED (`b41ba2f`).

## 0. Why Form B — the W.4b.1 finding that retired Form A

W.4b.1 smoke surfaced one load-bearing fact and one load-bearing observation:

1. **Injecting libcipher_rt into the batched executor costs 2.6×**
   (`W4B_1_SMOKE_FINDINGS.md`): the executor's batched GEMMs must NOT pay the
   per-GEMM cuBLAS-shim tax. Decided.
2. **In Form A the POOL eligibility gate is structurally ceremony.** A Form A
   executor holds exactly ONE model; every batched row runs on those weights by
   construction, so a same-fingerprint check protects against *nothing
   observable* — there is no second model present to mis-coalesce. The W.4a
   Memory #11 guard would gate nothing, and the engagement-gate tok/W would
   merely reproduce the CP 5.6 batching number — not a *new*
   substrate-attributable result.

The entry note's **R-W.4b.2** anticipated this ("revisit at W.4b.2 if cohort
coordination forces it"). It is now forced. **Form B is where the substrate
earns its keep:** each tenant is its own GPU process running the model, so each
produces a *real* W.6 sub-B fingerprint, the eligibility gate does *real*
admission work, and the Memory #11 guard protects against a *real* threat
(coalescing process A's activations with process B's *different* weights →
silent cross-model corruption).

## 1. Architecture — Form B′ (dedicated-vanilla executor, real peer tenants)

```
  tenant_0  injected · own ctx · own weights · own KV · green-partition · real W.6 fp ─┐
  tenant_1  injected · own ctx · own weights · own KV · green-partition · real W.6 fp ─┤ same-fp
  ...                                                                                  ─┘  group
        each intercepts its per-token model.forward (CP 5.6 §2.3, transparent)
                                   │  ship hidden-state via cuIPC, block, recv logit row
                                   ▼
  EXECUTOR  ── VANILLA (no shim tax) · holds a weight copy of the deployed model
            gather peer hidden-states via cuIPC → [N, hidden]
            run weight-bound GEMMs B=N (q/k/v/o proj + MLP up/gate/down)        ← the lift
            per-sequence attention: row i attends row i's OWN KV (cuIPC-mapped peer KV)
            scatter N logit rows back to originating processes via cuIPC + completion
```

- **Peers are real, independent GPU processes** running the model (own CUDA
  context, own weight copy, own KV, own green partition from CP 5.4). Each is
  injected, so each computes a *real* W.6 sub-B fingerprint and registers into
  the cohort (NR 30/31). **This is the user's choice criterion** — the gate has
  real work because there genuinely are N independent model processes.
- **The executor is a dedicated VANILLA process.** It holds one weight copy of
  the deployed model and runs the batched projection/MLP GEMMs (the lift,
  CP 5.6 §1) with **zero shim tax**. It is the compute engine, not a tenant.
- **Batched:** weight-bound projection/MLP GEMMs (M=1→M=N is the entire measured
  lift). **Per-sequence:** attention — each row attends its own KV of its own
  length, so heterogeneous lengths do not break the GEMM batch (CP 5.6 §2.4).
- **Barrier substrate-side (discipline (j)):** gather/scatter + per-token
  rendezvous live in libcipher_rt-side IPC + cuIPC, not a vLLM hook; tenant code
  unmodified (transparent interception, §2.3).

**Why not the pure §2.2B elected-tenant form.** In §2.2B the executor is itself
an injected tenant, so its batched GEMMs pay the 2.6× shim tax — which drops the
N=8 substrate-attributable ratio to ~1.45× (footnote A), missing the 3.69× gate
badly. B′ moves the heavy batched GEMMs onto a vanilla process while keeping the
peers real. *Who* runs the executor (dedicated vanilla vs elected tenant) is
**decoupled** from the Form-B substrate value (peers being real GPU processes
with real fingerprints); the elected form is deferred unless R-3 retest shows
the shim tax is non-fatal.

## 2. The eligibility gate under B′ — two real checks

On each batch-formation (once per token-group, not per GEMM) the executor:

1. **Peer-homogeneity** (fast pre-filter): `cipher_rt_pool_partition` over the
   live NR 31 cohort → the same-fingerprint coalesce-group. **Only peers sharing
   one nonzero W.6 fp are gathered.** Distinct-fp peers NEVER join (Memory #11
   structural guard); fp==0 (unwarmed) peers run solo.
2. **Executor-deployment-match:** verify the homogeneous peer fp matches what the
   executor's *own loaded weights* would produce. Two ways — **adjudicate:**
   - **(a) [recommended]** Executor does a one-time ≥64-GEMM warmup *off the hot
     path* (vanilla executor briefly self-injects only for warmup, or links the
     pure fingerprint accumulator) to materialize a real W.6 fp, then runs the
     decode loop vanilla. Keeps the substrate fingerprint definition unified;
     deployment-match is a direct fp compare.
   - **(b)** Executor keys on `hf_config_hash` (G10) instead. This is *not* the
     W.6 fp, so deployment-match leans on the first-coalesce **correctness
     backstop** (below) as the load-bearing guard; the fp partition then only
     catches peer↔peer mismatches.
3. **First-coalesce correctness backstop** (the transport-site Memory #11
   guard): `cipher_rt_pool_correctness_check` compares each peer's batched-path
   logits to that peer's single-tenant reference (tol 0.01). Any failure →
   `cipher_rt_pool_mark_blocked(fp,K,N,dtype)` → that group falls back to solo
   thereafter (W.4b.3) + ratio auto-disable (>25% per-tenant / >50% session).

The W.4a substrate (`cipher_rt_pool.[ch]`, anchor `8b5e928`) is reused unchanged
for the decision + block + check primitives. **No second copy of the
catastrophic guard is written.** Self host-tgid (W.4a debt #2/#3): NR 31 keys on
`current->tgid` (host PID; bare-metal == no namespace on this pod) and cuIPC
handles carry the originating tgid, so the executor maps each gathered peer to
its cohort entry by host tgid — no new ABI.

## 3. Substrate primitives (CP 5.6 §3) — reuse map

| Primitive | cuIPC mechanism | integration |
|---|---|---|
| Cross-process activation gather ([N,hidden] staging) | T4.6 / `cipher_kv_bridge` page-pool — **verified on this pod** | **new** (per-token gather) |
| Per-tenant KV isolation (cuIPC-mapped peer KV) | CP 5.1 / Week-5 KV cuIPC — **verified** | **new** — Week-5 is *intra-tenant prefix* dedup; *cross-tenant peer-KV gather into per-sequence attention* is new HF-attention work |
| Logit scatter (cuIPC out buffer + completion signal) | — | **new** (small) |
| Batch-formation scheduler (bounded-wait admission) | — | **new** — W.4b.4 |

The cuIPC *mechanism* (cross-process GPU memory) is verified; the
HF-attention-over-mapped-peer-KV *integration* is the genuinely new, highest-risk
work (R-4).

## 4. Sub-step sequence (atomic; adjudicate between each, W.x cadence)

- **W.4b.2 — B′ PROTOTYPE** (2 injected tenants + 1 vanilla executor,
  barrier-sync, equal-length, no scheduler). The make-or-break. 7-item plan §5.
  Primary deliverables: **(i)** the measured cross-process plumbing gap vs the
  in-process B=2 ceiling, and **(ii)** the **negative control** that proves the
  guard does real work (TinyLlama + a *different* model ⇒ partition rejects ⇒
  both run solo) — the test Form A cannot express.
- **W.4b.3 — eligibility gate + failure→block→disable** hardened (`mark_blocked`
  on first-coalesce fail + ratio auto-disable). Memory #11 surface fully wired.
- **W.4b.4 — scheduler + heterogeneous lengths + crash tolerance + scale N=4**
  (bounded-wait admission, unequal KV lengths, kmod `do_exit` reaper
  subscription for tenant-crash drop). Elected-executor variant revisited here
  *iff* R-3 shows the shim tax is non-fatal.
- **W.4b.5 — engagement gate on the GATED substrate** — N=8 TinyLlama 3.69× /
  N=4 Mistral 3.26× substrate-attributable (vs naive N-concurrent), each
  correctness-gated + `W4B_TRANSPORT_CLOSE_REPORT.md`.

## 5. W.4b.2 prototype — 7-item plan (ready to execute on approval)

1. **Harness** (`run_formb_proto.sh`): two TinyLlama **tenant** processes
   (injected, `CUDA_INJECTION64_PATH=libcipher_rt.so`, each warms ≥64 GEMMs so
   its W.6 fp is nonzero, each registers NR 30/31) + one TinyLlama **executor**
   process (vanilla; deployment-match fp per §2 #2(a)). No election in the
   prototype — the executor is dedicated.
2. **cuIPC activation channel** (`formb_ipc.py`): each tenant exports a
   [1,hidden] hidden-state buffer + a [1,vocab] logit-return buffer; executor
   imports them. Reuse `cipher_kv_bridge` cuIPC export/import. The existing Unix
   socket (`batch_ipc.py`) carries cuIPC handles + per-token ready/done signals;
   tensors move via cuIPC (never over the socket).
3. **Per-token barrier + elected decode step** (`formb_executor.py`): custom
   decode loop — gather [h0; h1]=[2,hidden], run projection/MLP GEMMs B=2,
   per-sequence attention (row i ← tenant i's cuIPC-mapped KV), scatter logit
   rows back. Each tenant's intercepted forward blocks on the barrier per token.
4. **Eligibility gate wired** via `cipher_rt_pool_partition` over the NR 31
   cohort at batch-formation; executor gathers a tenant ONLY if same-fp +
   deployment-match.
5. **Negative control [mandatory gate]:** re-run with tenant_1 = *different*
   model (Qwen2-0.5B or Phi-2). Assert `partition` rejects
   (`distinct_rejected`≥1, `eligible`=0) ⇒ no gather ⇒ both run solo ⇒ no
   corruption. This is the Form-B value proof.
6. **Correctness gate:** teacher-forced per-row logit-KL for BOTH tenants vs
   their single-tenant FP16 gold (CP 5.6 gate); pass ⇒ KL ≤ ~5e-5.
7. **Plumbing-gap measurement:** cross-process batched-B=2 tok/W vs (a) naive
   2-concurrent (substrate-attributable ratio) and (b) in-process B=2 ceiling
   (the gap). Write `W4B_2_FORMB_PROTO_FINDINGS.md`.

## 6. Risks at entry

- **R-1 (HIGH — Memory #11 catastrophic surface, now REAL).** Form B is the
  first time genuinely-different model processes can be coalesced. A fingerprint
  collision or gate-bypass silently corrupts a tenant. Mitigation: audited
  `cipher_rt_pool_partition` is the sole admission gate; first-coalesce
  correctness backstop + `mark_blocked`; the §5 #5 negative control is a
  *mandatory* gate.
- **R-2 (HIGH — the plumbing gap is the bet).** No longer a 2.6× executor cliff
  (B′ runs the executor vanilla). It is now the **per-token cross-process
  cuIPC gather/scatter cost** plus the **peers' residual shim cost** (R-3). If
  the combined gap pushes N=8 below 3.6× substrate-attributable, the gate is not
  met. **Pre-committed disposition (no-inflation discipline,
  [[cipher-f1-fullgpu-marlin-broken]]):** if the gap eats the lever, **W.4b
  closes NOT-MET on the engagement gate** — the safe-coalescing guarantee
  (Memory #11 guard, proven by the negative control) + Track-2 capacity are
  *findings to report*, **not** a redefined headline.
- **R-3 (peers' residual shim tax).** Peers stay injected (for real fps +
  transparent interception), but they no longer run the batched projections —
  only attention + the rerouting hand-off. Their shim cost is much smaller than
  a full B=1 forward but nonzero; §5 #7 measures it as the residual gap above the
  vanilla in-process ceiling.
- **R-4 (HIGH — HF attention over cuIPC-mapped peer KV).** Splicing a peer's
  cuIPC-mapped KV into a per-sequence attention sub-step needs a hand-written
  decode loop reaching into model internals. The cuIPC mechanism is verified;
  this *integration* is new. Mitigation: prototype on TinyLlama (simplest MHA)
  before Mistral GQA.
- **R-5 (Mistral env, E.7).** N=4 Mistral is half the engagement gate; prior
  sessions logged a Mistral env-block. **Disposition:** E.7 is NOT on the
  W.4b.2 prototype path (TinyLlama-only); it gates **W.4b.5** (the Mistral half).
  Resolve E.7 in parallel with W.4b.3/W.4b.4 so it does not block the prototype.

## 7. Scope + estimate

Per CP 5.6 §6: prototype 1–2 wk, MVP N=4 3–4 wk, production N=16 ~6 wk total.
W.4b.2 (prototype) is the next atomic step and the highest-information one — it
either proves the cross-process lever survives the plumbing gap (proceed to the
scheduler) or it does not (R-2 NOT-MET disposition). **Nothing past W.4b.2 is
built until its plumbing-gap number is adjudicated.**

## 8. Status

Memo COMPLETE; no substrate code written; anchors unchanged. Awaiting
adjudication of **(a)** Form B′ architecture as scoped, **(b)** the §2 #2
deployment-match choice (recommended (a) warmup), and **(c)** the W.4b.2
prototype 7-item plan — before build. Related: [[cipher-cp56-closed]] (transport
re-ported), [[w4-pool-transport-frontier]] (frontier), [[cipher-phase-discipline]],
[[cipher-proceed-not-ask]].

---

**Footnote A — design history (why B′, not §2.2B).** Engagement gate = 3.69×
substrate-attributable at N=8 ≡ the in-process *vanilla* ceiling at B=8
(CP 5.6 §1; naive-concurrent ≈ 1.0 tok/W). An *injected* executor pays the
W.4b.1 2.6× shim tax on its batched GEMMs → 3.78/2.6 ≈ **1.45× at N=8** (~2.8× at
N=16) — pure §2.2B arithmetically misses the gate at the stated N. Hence the
heavy batched GEMMs run on a vanilla executor (B′); peers stay injected so the
eligibility gate keeps real work. This refinement was a pre-build correction,
which is the design-memo→approve cadence working as intended.
