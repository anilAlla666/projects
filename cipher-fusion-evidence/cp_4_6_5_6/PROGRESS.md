# CP 4.6.5+6 — build progress (atomic STEP across sessions)

Close Phase 4.6: T4.6.5 (real-trace dedup validation gate) + T4.6.6 (multi-
tenant integration at the decision-tree-selected count). One atomic STEP;
one `CP_4_6_5_6_REPORT.md` at close. This file is the durable cross-session
checkpoint — not the report.

Anchors held throughout — no substrate code changes: kmod 0.4.8 `e2f50452`
(srcversion `E427CAFA4E94D548233DC7A`), libcipher_v2 `86618c30`, libcipher_rt
`c2c5d313`.

## Adjudicated parameters (user, 2026-05-16)

- D2: cuIpc-imported shared weights; fp16 gate / INT4 stretch; Mistral-7B; 32K.
- D3: T4.6.5 measures cold-path call model for the rate gate; live-path
  characterised as a T4.6.6 finding.
- D4: p99 ≤ 2× the §3b roofline-predicted mean.
- D5: **decision-tree-selected tenant count** — T4.6.6 validates at whatever
  count the measured `m` supports (dataset predicts ~30 at 32K), not a fixed 50.
- F1 accepted + extended: T4.6.5 measures `m` at **25K, 32K, AND 137K**
  resident context — full curve.

## Build STEP — 8 items

1. **Design memo — DONE.** `CP_4_6_5_6_DESIGN_MEMO.md`, revised with the
   dataset-grounded §2 (m per-context table), §3a consequence, §4 per-context
   curve + steady-state-resident tree application.
2. **Trace prep — DONE (dataset).** GitHub `callanjfox/kv-cache-tester`
   cloned → `/home/ubuntu/kv-cache-tester/`. 739 traces, 59,204 requests,
   `hash_id_scope: "local"`, `block_size: 64`. Characterised: shared prefix
   (sys+tool) median 14.9K; first-req input median 24.8K; last-req median
   137.5K. Replay/normalisation harness — built as part of item 3.
3. **T4.6.5 dedup measurement — DONE.** `t465_dedup_measure.py` →
   `dedup_measurement.json`. Structural canonical-shared-prefix model
   (forced by `hash_id_scope: local`), n=5 windowed. Measured `m`:
   **25K → 2.35× (L3); 32K → 1.81× (L1-only); 137K → 1.12× (L1-only).**
   All within the pre-registered prediction band. Tier crossover at ~28.6K.
4. **Decision artifact — DONE.** `T465_DECISION_ARTIFACT.md`. Tree fires:
   32K gate regime → **L1-only → ~30-tenant target** for T4.6.6 (per D5,
   decision-tree-selected, not fixed 50). Substrate-achievement (real==sim)
   validation folded into T4.6.6 item-6 dedup-at-scale instrumentation.
### T4.6.6 restructured — Option A (user-adjudicated 2026-05-16)

HBM blocker surfaced: 30 real-decode tenants @ 32K fp16 = 134 GB ≫ 80 GB
(dedup substrate not wired into live PyTorch KV cache — audit §2). Option A:
- **Real concurrent decode: 16 tenants @ 32K fp16** (HBM ceiling 78 GB) —
  per-tenant tok/s, p99, HBM, arbitrator-under-load.
- **Substrate dedup-at-scale: 30 processes** through /dev/cipher_kvdedup
  (scale T4.6.4 2→30, trace-derived pages) — cuIpc scaling + achieved m vs
  T4.6.5 1.81×.
- **Isolation: 30-process kvdedup harness.**
- Report: 30 = substrate-validated; 16 = real-decode-measured; the gap =
  wiring dedup into live KV cache (integration step beyond this CP).

### BLOCKER surfaced 2026-05-16 — Mistral-7B sliding-window confounds T4.6.6

`t466_decode_load.py` ran 16 tenants @ 32K OK (no OOM) — but the result is
confounded: **Mistral-7B-v0.1 has `sliding_window: 4096`**. At 32K context
its resident KV is capped at 4096 tokens (~0.5 GB), NOT the 4 GB/32K the §3a
capacity math assumed. Measured per-tenant tok/s 38, p99 27 ms, torch HBM
28.8 GB vs nvidia-smi peak 75.8 GB (prefill-peak, not decode steady-state) —
all reflect a 4K-window regime, not 32K-resident-KV. **The §3a capacity
arithmetic + decision-artifact tenant counts assumed full-attention KV.**
Product implication: the KV-dedup substrate is load-bearing only for
FULL-attention long-context models (KV grows with context); for
sliding-window models KV is bounded and density is already high.
→ adjudication needed: re-run decode arm with a full-attention model
(Llama-3.1-8B on pod, GQA 8×128 — matches §3a geometry). D2c (Mistral-7B)
revisited. AWAITING user.

### CONSOLIDATED DIRECTION (user, 2026-05-16) — 3-arm T4.6.6

Campaign target = 100 tenants/H100, two phases. **Phase 4.6 (this CP) ships
the substrate primitive at 100-proc scale.** Phase 5 (Q3 2026, 10–14 wk) =
integration (KV offload + partition-aware Marlin/Song Han + arbitration +
live-decode wiring) → 100 real-decode tenants.
- **Arm 1** — substrate validation at 100-proc scale (model-free): cuIpc
  dedup-at-scale through /dev/cipher_kvdedup, lightweight client procs.
  Measure cuIpc scaling 2→100, ioctl throughput, achieved m vs 1.81×,
  per-proc RSS, isolation. Pass = honest measured ceiling; don't stop at 30.
- **Arm 2** — real-decode demo: Llama-3.1-8B fp16 @ 32K, 16 tenants (HBM
  ceiling). Mistral → appendix ("early validation run"). NOT the campaign
  target — current real-decode capacity pre-Phase-5.
- **Arm 3** — isolation on the 100-proc harness.
Report §1–§8 structure specified (market / substrate / T4.6.5 curve /
substrate-primitive validation / decode demo / arithmetic gap to 100 /
architecture-coverage footnote / Phase 4.6 close + Phase 5 handoff).

5–8. **T4.6.6 — measurements + report — ALL DONE.** Verified numbers
below are from the on-disk JSON artifacts, not prose.
   - **Arm 1 (item 6) — DONE — PASS.** `t466_dedup_scale` at **100 procs:
     100/100 put-ok**, no scaling ceiling. 6400 puts → kmod 3430 distinct
     physical pages (exact: 30 shared collapsed + 3400 unique) → **achieved
     m = 1.866×**. per-proc RSS **107.7 MiB**. `dedup_put` throughput
     **508 puts/s** (slowest proc, 12.59 s/64 puts — Finding 4a: 100-way
     concurrency serialises the kmod lock + cuIpc-import; cf. ~3348 puts/s
     at the 4-proc smoke). `t466_dedup_scale_result.json`.
   - **Arm 2 (item 6) — DONE.** Llama-3.1-8B fp16 32K. First run OOM'd at
     6 tenants — caching-allocator fragmentation + 8.4 GB full-prompt-logits
     transient. **Fixed in source** (`expandable_segments:True`,
     `logits_to_keep=1`). Re-run: **13 tenants resident** (directive est.
     16). Per-tenant **2.93 tok/s**, p99 ITL **352.5 ms**, peak HBM 73.7 GB.
     D4 gate FAILS as specified (p99 > 2× §3b roofline) — characterised:
     B=1 decode is launch-latency-bound; p99/median=1.036 (tight).
     `t466_decode_llama_result.json`.
   - **Arm 3 (item 7) — DONE — PASS.** `t466_isolation`: 3/3 unauthorized
     foreign-pointer probes rejected, **0 violations**.
     `t466_isolation_result.json`.
   - **Item 8 — DONE.** `CP_4_6_5_6_REPORT.md` written, §0–§8 + Arm 3 +
     appendices. **Phase 4.6 CLOSED. Awaiting adjudication.**

## Pre-registered prediction (engineering-marvel discipline — before measurement)

Cross-tenant `m = 1/(1−f)`, `f` = shared-prefix fraction. Dataset-grounded:
| context | `f` | predicted `m` | tier |
|---|---|---|---|
| 25K | 0.60 | 2.5× | L3 |
| 32K | 0.47 | 1.9× | borderline L1/L3 |
| 137K | 0.11 | 1.1× | L1-only |
`m` is a measurable LOWER BOUND — `local` hash scope hides codebase-overlap
dedup. Decision tree fires mechanically on measured steady-state `m`.

## Session log

### 2026-05-16 — STEP started; items 1–2 done, item 3 in progress.

### 2026-05-16 (cont.) — items 5–8 DONE; STEP COMPLETE.

3-arm T4.6.6 executed; `CP_4_6_5_6_REPORT.md` written. Two
measurement/estimate deltas surfaced honestly in the report: (i) real-decode
ceiling 13 tenants vs the directive's 16 estimate (≈12 GB CUDA/green-ctx/
workspace overhead the naïve `(80−15)/4` ignores); (ii) D4 latency gate fails
as specified because its §3b reference is a bandwidth roofline B=1 decode
cannot reach — p99/median is 1.036 (tight). Both deltas concern the Phase-5
real-decode integration layer, not the substrate primitive this CP closes.
Anchors held — no substrate code changes. Phase 4.6 CLOSED.