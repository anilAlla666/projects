# V.1 — CP 5.5 stock-customer-mix soak — DESIGN MEMO

**Date:** 2026-05-29. **Status:** DESIGN MEMO — STOP for Anil approval before build
(design-memo → approve → build). The v1 **capstone**: all 5 Memory #1 goals validated
end-to-end on a stock-config mixed-workload soak. **Anchors UNCHANGED** at entry
(cipher_rt_phase4 `8b5e928`/`9fe23143`, cipher_kmod `02fc2d1`, cipher_kv_bridge
`5a3db034`; cipher-fusion-evidence HEAD `d412963`, `w4b-close`→`dbb7dd9`).

## §0 — Provenance

Frontier reached V.1 after: substrate+safety+density delivered (W.1/W.2/W.3/W.5/W.6/W.4a),
W.4b CLOSED on branch A, FWD-1 (52ms = DynamicCache prototype artifact; vLLM static),
W.8 de-prioritized, W.7 NCCL plugin-correctness validated (engagement v1.x-deferred). Anil
accepted the W.7 close → **V.1 (CP 5.5 soak)** 2026-05-29. V.1 = audit `:517`. ED ~3 (the
soak is short; the deliverable is the 5-goal engagement gate report).

## §1 — The 5 goal-gates + per-goal readiness (the pre-flight, item 1)

V.1 closes only if **ALL 5 interim gates pass on STOCK config with ZERO customer-env
intervention** (Memory #19 product-engagement-gate: it must work on a plain `vllm serve`,
no LD_PRELOAD gymnastics, no hand-tuning). Per goal: the interim gate, the substep that
provides the capability, and the **open readiness question** the pre-flight must answer.

| Goal | Interim gate (audit `:527`) | Provider (closed) | OPEN readiness question |
|---|---|---|---|
| **1 density** | ≥10 concurrent agents/H100 sustained, **KL=0** | W.4 POOL / cross-tenant batching; vLLM KV (CP 5.1) | Does ≥10-agent same-model batching hold KL=0 on stock vLLM? (branch-A density delivered; confirm at 10 on the agent-inference class) |
| **2 tok/W** | ≥1.5× on weighted mix vs vanilla | VOLT (W.1) auto-activate | Does VOLT **auto-activate** on stock `vllm serve` and lift the *weighted mix* ≥1.5×? (VOLT lift is regime-specific — Mem #`cipher-t43-envelope`; weighted-mix number is the risk) |
| **3 MFU** | ≥70% weighted-mean MFU | CUPTI MFU (CP 3.3) + actuators | Is weighted-mean MFU ≥70% on stock mix? (persistent-kernel for 85% is post-v1 / DE-PRIORITIZED per FWD-1 — 70% interim is the bar) |
| **4 Koopman** | handled_count>0 on long-context prefill, KL preserved | W.3 Koopman calibration | Does Koopman **auto-fire** (handled>0) on the RAG/long-context class on stock config, KL-preserving? (Goal 4 was historically the weakest — confirm it fires at all) |
| **5 auto-profile** | cipher-platform reports correct profile + activated capability set <30s of `vllm serve`, no customer env | K.1 classifier + cipher-platform status | Is the classifier (#5, flagged KEYSTONE-GAP in the audit) wired enough to report the correct profile + capability set within 30s? **Highest-risk gate.** |

**The pre-flight (item 1) resolves each open question BEFORE the soak** — if a gate's
capability does not auto-activate on stock config, that is a **finding** (V.1 surfaces it;
the goal is *not* silently failed or hand-forced). 1-GPU runnability: all 5 gates are
serving/inference-class (NOT multi-GPU training) → runnable on this single H100.

## §2 — Workload mix + stock-config soak (1-GPU representative)

Mix (audit `:521`): **30 agent-inference + 30 continuous-batched-serving + 20
single-tenant streaming + 10 batch-inference + 10 RAG**; mixed dtypes (bf16, fp16, AWQ
INT4); mixed models (Llama-3-8B, Mistral-7B, TinyLlama, Qwen-2.5). On a single H100 the
"100 workloads" is run as a **representative weighted soak** (the classes time-share /
right-sized to 80 GB — e.g. small models for the high-count agent class, one large model
for batch/RAG), NOT 100 simultaneous full 8B models (which won't fit). **The weighting
must match the mix proportions** so the tok/W and MFU numbers are the *weighted* gate, and
the scaling/right-sizing is **logged, not silently capped** (Mem #`cipher-lift-framing`
honesty). Driver: stock `vllm serve` per model + the cipher-platform auto-activation path;
no per-workload hand-tuning.

## §3 — Execution plan (7 atomic items; approve before item 1)

1. **Readiness pre-flight (§1).** For each of the 5 gates, confirm on a stock `vllm serve`:
   the capability auto-activates (VOLT lock, Koopman fire, classifier profile report, POOL
   batching, MFU counter) with zero customer-env intervention. Output a 5-row readiness
   table. **Any non-auto-activating capability → surface as a V.1 finding (don't hand-force).**
2. **Soak harness.** Build the weighted mixed-workload driver (§2) over stock `vllm serve`;
   right-size to 80 GB with the mix proportions preserved; log all scaling decisions.
3. **Goal 1 + Goal 5 gates.** ≥10 concurrent agents at KL=0 (density + correctness, Mem
   #11); cipher-platform profile + capability set reported <30s of serve start, no env.
4. **Goal 2 + Goal 3 gates.** Weighted-mix tok/W ≥1.5× vs vanilla; weighted-mean MFU ≥70%.
   (tok/W vs a vanilla `vllm serve` baseline on the identical mix.)
5. **Goal 4 gate.** Koopman handled_count>0 on the RAG/long-context-prefill class with KL
   preserved (Mem #11).
6. **Write `V1_CP55_SOAK_REPORT.md`** — 5-goal gate table (PASS/FAIL each, with the stock-
   config + zero-intervention attestation), the readiness findings, and the honest
   scaling log. Goal status: 5/5 = v1 goals delivered end-to-end; <5/5 = name exactly
   which goal(s) and why (hardware / not-auto-activating / regime).
7. **Commit** (author `Anil`, no co-author; measurement only → anchors UNCHANGED unless a
   readiness gap forces a substrate fix, which is a fresh substrate step per Mem #16).
   Tag at close only if Anil directs. **STOP for Anil.**

## §4 — Decision gate

- **PASS (V.1 closes, v1 ships on all 5 goals):** all 5 interim gates pass on stock config
  with zero customer-env intervention (Goal 1 KL=0 @≥10 agents; Goal 2 ≥1.5× tok/W; Goal 3
  ≥70% MFU; Goal 4 Koopman handled>0 KL-preserved; Goal 5 profile+caps <30s).
- **PARTIAL:** name exactly which goal(s) miss and why (a not-auto-activating capability,
  a regime limit, or a 1-GPU/hardware constraint). No goal is faked or hand-forced to pass.
- **HARD STOP (Mem #11):** any KL≠0 on the density/Koopman correctness gates.

## §5 — Discipline guarantees

- **Stock-config, zero-intervention (Mem #19):** every gate measured on a plain
  `vllm serve` + the cipher-platform auto-activation path; no per-workload hand-tuning,
  no customer-env edits. If a capability needs intervention to fire, the goal does not pass.
- **Correctness (Mem #11):** KL=0 on density; KL-preserved on Koopman; HARD STOP otherwise.
- **Anchors (Mem #16):** measurement only → UNCHANGED; a readiness-gap substrate fix is a
  fresh substrate step (backfill + gate), not folded silently into V.1.
- **Honesty (Mem #`cipher-lift-framing`):** the 1-GPU right-sizing of the "100-workload"
  mix is logged, not silently capped; interim gates (10 agents / 1.5× / 70%) are reported
  as interim, with the post-v1 targets (100 agents / 2× / 85%) named.

## §6 — One-line ask

Approve V.1 (the 5-goal stock-mix soak, items 1–7, starting with the §1 readiness
pre-flight) — or adjust the gate bands / the 1-GPU right-sizing approach / the workload
weighting. No code until approval; STOP at the §4 5/5 verdict for your v1-ship adjudication.
