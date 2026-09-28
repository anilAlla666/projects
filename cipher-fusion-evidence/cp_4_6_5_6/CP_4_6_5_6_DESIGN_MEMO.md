# CP 4.6.5+6 — Phase 4.6 multi-tenant substrate close — DESIGN MEMO

**Date:** 2026-05-16. **Status:** pre-build memo — no code, no traces, no
load test until §6 decisions are adjudicated. Per Phase-4+ discipline this
memo opens with mechanism characterization and bound calculation; the build
then validates *within* those bounds and the report states what is bounded
vs what is measured.

**Scope.** Close Phase 4.6 by executing the two remaining tasks:
- **T4.6.5** — validation gate: replay 739 anonymized Claude Code agentic
  traces through the dedup substrate, measure cross-tenant block dedup rate,
  decide the deployment tier per the pre-specified decision tree.
- **T4.6.6** — 50–100 concurrent tenant integration: real decode workloads,
  measure per-tenant tok/s, p99 latency, isolation, HBM utilisation,
  cross-tenant dedup at scale.

**No substrate code changes.** T4.6.1–4 shipped the mechanism (attn substrate,
KV page allocator, L1 in-process dedup, L3 kmod cross-process dedup). CP
4.6.5+6 is validation + scale-up of existing infrastructure. Anchors held:
kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`, libcipher_rt `c2c5d313`.

---

## 1. Mechanism characterization

**What the substrate does.** The KV-dedup substrate (T4.6.3 L1 + T4.6.4 L3)
content-addresses KV-cache pages: a CIPHER page = N consecutive 512-token
blocks; identical block-id tuple ⇒ content-identical page. The kmod
(`cipher_kvdedup.c`, `/dev/cipher_kvdedup`) owns a cross-tenant refcount table
and cuIpc POSIX-FD handles; when tenant B `put`s a page whose content hash
matches a page tenant A already registered, B imports A's physical page
instead of allocating its own. **The lever is HBM capacity**: M tenants that
share a page store it once, not M times.

**Two things the mechanism does NOT do** (bounds on the claim):
1. It does not reduce per-token compute or HBM *bandwidth* — a deduped page
   is still read once per decode step by each tenant that uses it. Dedup is a
   **density** lever (tenants-per-GPU), not a throughput lever.
2. T4.6.4 shipped the *cold-path* call model only (`dedup_put` once per slab
   page at allocation, ~400 µs/hit amortised to nothing). The live per-decode
   call model was explicitly deferred — `t4_6_4_design.md` §"Call model":
   *"T4.6.5 decides which model it measures."* This memo picks it — §6 D3.

**Hit-path cost (matters at scale).** The dedup hit path does a synchronous
`cuMemImportFromShareableHandle` + DtoH copy + full-page `memcmp`-verify. At
T4.6.4's cold-path call model that ~400 µs is amortised to nothing. But the
memcmp-verify is *in the hit path* — if §6 D3 selects the live (model b) call
model, that cost compounds per request across 50–100 tenants and becomes a
measurable tax. §6 D3's choice changes not just what T4.6.5 measures but what
the substrate *costs* at T4.6.6 scale; the report must separate the two.

**The scaling question T4.6.6 answers.** 50–100 tenants on one 80 GB GPU is a
capacity problem before it is a throughput problem (see §3 roofline). The
substrate is the thing that makes the capacity arithmetic close — or does
not. T4.6.5 measures whether the dedup rate on *realistic* traces is high
enough to matter; T4.6.6 measures whether the density actually holds under
concurrent load with isolation intact.

---

## 2. T4.6.5 — dedup-rate prediction model (BEFORE measurement)

**The quantity.** Per-page duplicate fraction `r` = (cross-tenant hits) /
(total pages). Effective HBM capacity multiplier `m = 1/(1 − r)`. The
pre-specified decision tree is stated in `m`:

| Measured `m` | Tier | Tenant target |
|---|---|---|
| `m < 2×` | L1-only (in-process dedup) | 20–30 tenants/GPU |
| `2× ≤ m ≤ 10×` | L3 build (kmod cross-process — already shipped) | 50–100 tenants |
| `m > 10×` | full RadixAttention (prefix-tree KV) | 100–200+ tenants |

**Prediction model.** `r` is bounded by the cross-tenant *shared-prefix
fraction* `f`: at scale every tenant's shared-prefix pages are duplicates of
tenant 1's, so `r → f` and `m → 1/(1−f)`. For the 739 Claude Code agentic
traces, `f` is driven by three structural sources, all identical across
sessions: (i) the Claude Code system prompt, (ii) the tool-definition
schema, (iii) common project context (CLAUDE.md, repo boilerplate).

**Bound — anchored to the T4.6.3 measured data** (`sim_results.json`,
Mooncake-derived traces, the only cross-tenant dedup numbers measured so far):

| Trace flavour | measured `m` (N=2 pages) | character |
|---|---|---|
| conversation | 1.49× | generic chat, low shared prefix |
| toolagent | 2.03× | tool-using, moderate shared prefix |
| synthetic | 2.64× | constructed high-overlap |

**Prediction — DATASET-GROUNDED (revised after fetching the 739 traces).**
The traces are now on the pod (`kv-cache-tester/traces/`, GitHub
`callanjfox/kv-cache-tester`). Measured trace characteristics (all 739,
computed directly):

- shared prefix = `system_tokens + tool_tokens`: **median 14,929** tokens
  (sys ~2.8K + tool ~12.2K) — and it is *genuinely identical cross-tenant*
  (same Claude Code system prompt + tool schema for every session).
- context grows over a conversation: first-request input **median 24.8K**,
  last-request input **median 137.5K**.
- `hash_id_scope: "local"` — hash_ids are per-conversation; **cross-tenant
  dedup cannot be read from hash_id overlap**, only from the structurally
  shared system+tool prefix. So T4.6.5's number is a **measurable lower
  bound** on production cross-tenant dedup — it cannot capture codebase /
  shared-file overlap (agentic coders on the same repo), which `local` scope
  hides. Frame the result accordingly.

Cross-tenant `m = 1/(1−f)`, `f` = shared-prefix fraction of resident context.
`f` is **sharply context-dependent** — the 14.9K prefix is most of a short
context and a sliver of a long one:

| resident context | shared `f` | `m` | tier the tree selects |
|---|---|---|---|
| 25K (median first request) | 0.60 | **2.5×** | L3 (2–10×) |
| 32K (the §3 gate regime) | 0.47 | **1.9×** | borderline L1 / L3 |
| 64K | 0.23 | **1.3×** | L1-only |
| 137K (median last request) | 0.11 | **1.1×** | L1-only |

**The honest pre-registration:** the L1/L3 decision boundary (`m = 2×`) falls
**squarely inside the regime the gate runs in** — between the 25K first-
request scale and the 137K steady-state scale. The prediction is therefore
explicitly *"the tier depends on the context regime measured"* — which is why
the §4 measurement reports `m` as a **per-context curve**, and the decision
artifact applies the tree at the **steady-state HBM-resident `m`** (§4, the
quantity that actually sets tenants-per-GPU). We do not pre-judge the tier:
the pre-specified tree picks it from the measured number, mechanically. If
the steady-state `m` lands `<2×`, that is the honest finding — L1-only,
20–30 tenants — and the dataset's lower-bound nature (codebase overlap
unmeasured) is the stated caveat, not a reframe.

---

## 3. T4.6.6 — per-tenant tok/s roofline (BEFORE load test)

**Hardware bound (this pod):** H100 80 GB HBM3, 132 SMs, HBM bandwidth
**3.35 TB/s**, 80 GB capacity. B=1 decode is **memory-bandwidth-bound** — per
decode step a tenant reads its model weights + its KV cache once.

### 3a. Capacity roofline — does 50–100 tenants even fit?

Per-tenant copies of a 7B fp16 model do **not** fit: 50 × 14 GB = 700 GB ≫
80 GB. **The arithmetic only closes with shared model weights** — one
resident copy, all tenants attach (§6 D2). With shared weights:

- Weights: one 7B fp16 copy = **14 GB** (or ~4 GB INT4 via the Marlin path).
- KV budget: 80 − 14 ≈ **66 GB** for all tenants' KV cache.
- KV/token (Mistral-7B: 32 layers, 8 KV heads, 128 dim, fp16) =
  2·32·8·128·2 B = **128 KB/token**.
- **Context regime — 32K.** At 8 K context KV/tenant = 1 GB and 66 tenants
  fit with *no dedup at all* — the substrate would be unnecessary for the 50
  target. The regime where the substrate is load-bearing, and where campaign
  memory places the dedup moat ("32K+ context"), is **32K**: KV/tenant =
  32768 × 128 KB = **4 GB**.
- Naïve fit at 32K: 66 / 4 = **16 tenants** — far below the 50 target.
- **Dedup — honest model.** Effective KV/tenant is not `KV/m` by fiat; it is
  `(1−f)·KV + f·KV/N`, where `f` is the shared-page fraction and `N` the
  tenant count. At N≥50 the shared term `f·KV/N` amortises to ~nothing, so
  effective KV/tenant ≈ `(1−f)·KV = KV/m`. Tenants-that-fit = 66 / (4/m) =
  **16·m**.
- ⇒ tenants-that-fit = **16·m**: **50 tenants needs `m ≥ ~3.1×`; 100 needs
  `m ≥ ~6.3×`.**
- **Dataset-grounded reality (§2 table):** at 32K the 739-trace prediction is
  `m ≈ 1.9×` ⇒ **~30 tenants** fit — *below* the 50 gate. 50 tenants at 32K
  needs `m ≥ 3.1×`, which the dataset's measurable (system+tool-prefix-only)
  dedup does **not** reach. Honest consequence, stated now: **on this dataset,
  at 32K, the substrate supports ~30 tenants, not 50.** Reaching 50 requires
  one of — (i) shorter resident context (25K → `m≈2.5×` → ~40, still short);
  (ii) the codebase-overlap cross-tenant dedup the `local`-scope dataset
  cannot measure (real but unprovable here); (iii) INT4 weights freeing HBM
  (§3b, D2b) — 4 GB weights ⇒ 76 GB KV budget ⇒ 19·m tenants ⇒ `m≥2.6×` for
  50. The two tasks stay coupled by arithmetic: **T4.6.5's measured `m` sets
  what T4.6.6 can honestly claim** — and the pre-specified decision tree, not
  this memo, picks the tier.

### 3b. Throughput roofline — per-tenant tok/s

Tenants time-share the GPU (132 SMs cannot be space-partitioned 50–100 ways).
Aggregate decode steps/s ≈ HBM_bw / (bytes read per step). Per step the
serviced tenant reads weights + its KV (32K context):

- bytes/step ≈ 14 GB (weights) + 4 GB (32K KV) = **18 GB**
- aggregate ≈ 3.35 TB/s ÷ 18 GB ≈ **186 steps/s** (ideal; ~0.6 efficiency ⇒
  **~112 steps/s** realistic)
- per-tenant tok/s = aggregate / N: **50 tenants → ~2.2 tok/s**;
  **100 tenants → ~1.1 tok/s** (realistic-efficiency roofline).

Note dedup does **not** move this roofline — a deduped page is still read
per decode step by each tenant using it (§1). Dedup buys capacity (§3a), not
bandwidth.

**This is the load-bearing honesty in the memo:** at 50–100 tenants per-tenant
tok/s is **single-digit by the roofline** — the GPU's aggregate bandwidth is
fixed and divided N ways. T4.6.6's pass criterion must be set against *this
roofline*, not against single-tenant ~50 tok/s. The substrate's win is
**density** (tenants-per-GPU) and **isolation** (bounded p99), not per-tenant
speed. Anyone reading the report must see the roofline first.

Weight-sharing INT4 (Marlin, 4 GB weights) lifts the roofline materially:
bytes/step ≈ 4 + 4 = **8 GB** ⇒ ~419 ideal / ~251 realistic steps/s ⇒
50 tenants → **~5 tok/s**, and it frees ~10 GB of HBM for more KV (more
tenants). Whether T4.6.6 runs fp16 or INT4 weights is §6 D2.

---

## 4. Pass criteria

**T4.6.5 (dedup gate)** — mechanical, no judgement:
- Replay completes; `dedup_measurement.json` carries cross-tenant `m` as a
  **per-context curve** (`m` vs resident-context, since §2 shows `m` drifts
  2.5×→1.1× over a conversation), each point with n=5 windowed CI.
- The decision tree is applied at the **steady-state HBM-resident `m`** —
  the `m` of the KV actually resident across tenants at peak, which is the
  quantity that sets tenants-per-GPU (§3a). The per-context curve is the
  supporting evidence; the steady-state scalar is what the tree consumes.
- `m` is reported as a **measurable lower bound** — `hash_id_scope: "local"`
  means codebase/shared-file cross-tenant overlap is invisible to this
  dataset (§2); the artifact states this explicitly.
- Decision artifact selects the tier the tree dictates for the measured
  steady-state `m`.
- PASS = the measurement ran clean and the artifact is internally consistent
  (real==sim cross-check holds, as in T4.6.3). The *tier outcome itself is
  not pass/fail* — it is the finding.

**T4.6.6 (scale gate):**
- (a) **50 tenants sustained** — 50 concurrent tenant processes complete real
  decode workloads with **0 crashes / 0 hangs** over the run.
- (b) **p99 within roofline** — per-tenant p99 inter-token latency ≤ **2× the
  §3b roofline-predicted mean** for the tenant count run. (Proposed budget —
  §6 D4. Anchoring to the roofline, not an absolute ms number, is the
  engineering-marvel-honest form: the GPU's bandwidth is fixed; the test is
  whether arbitration/substrate overhead stays within a 2× envelope of the
  physics, not whether it beats physics.)
- (c) **0 isolation violations** — no tenant reads another tenant's KV
  content except via a *verified* dedup hit; the cross-tenant isolation
  security test (build item 7) finds no leakage. This is a hard gate.
- (d) **HBM headroom** — peak HBM utilisation reported; 50 tenants must fit
  with the dedup-enabled footprint and not OOM.
- 100 tenants: **stretch** — run it, report how far it gets; not a gate.

---

## 5. Build STEP — 8 items

1. **Design memo** (this document) — mechanism + bounds + pass criteria.
2. **Trace prep** — obtain the 739 Claude Code agentic traces (§6 D1),
   normalise to the `{hash_ids}` JSONL schema the T4.6.3 harness consumes,
   build/extend the replay harness to drive them cross-tenant.
3. **T4.6.5 dedup measurement run** — replay through the substrate, n=5
   windowed, real==sim cross-check; emit `dedup_measurement.json`. ~3 h.
4. **Decision artifact** — apply the §2 decision tree to the measured `m`;
   emit the tier decision with evidence.
5. **T4.6.6 harness** — 50–100 concurrent tenant processes, shared-weight
   model (§6 D2), real decode workloads, per-tenant tok/s + p99 sampling,
   HBM + dedup-at-scale instrumentation.
6. **Load test + measurement** — 50-tenant gated run; 100-tenant stretch.
   ~6–8 h.
7. **Cross-tenant isolation verification** — security test: confirm no
   tenant can read another's KV bytes except via a verified-hit dedup; probe
   forced-hash-collision and post-teardown access paths.
8. **Report** — `CP_4_6_5_6_REPORT.md`: bounded vs measured, both gates,
   the tier decision, Phase 4.6 close recommendation.

---

## 6. Open decisions — adjudication needed before build

**D1 — Source of the 739 traces — RESOLVED 2026-05-16.** Found and fetched:
GitHub `callanjfox/kv-cache-tester`, cloned to `/home/ubuntu/kv-cache-tester/`.
`traces/` = 739 `trace_NNNN.json`, 59,204 requests, `hash_id_scope: "local"`,
`block_size: 64`. Real Claude Code sessions via claude-code-proxy (WEKA
Augmented Memory Grid research). The repo also ships `trace_replay_tester.py`
+ `cache_rate_tester.py` — its own replay/cache tooling, usable as a
methodology reference. **The fetch surfaced the §2 dataset-grounded
revision** (local hash scope ⇒ cross-tenant dedup is the system+tool prefix
only, a measurable lower bound; `m` per-context table). No remaining D1
blocker.

**D2 — T4.6.6 tenant model.** Per-tenant model copies do not fit HBM (§3a).
Four sub-choices, all load-bearing for the item-5 harness:
- **D2a — how weights are shared.** (i) one server process, N decode
  streams/threads — simplest, but no process isolation, weakens the §4(c)
  isolation gate; (ii) NVIDIA MPS — shared CUDA context across separate
  processes; (iii) cuIpc-imported weight buffers — separate processes, one
  exporter, the *same mechanism the KV substrate already uses*. **Recommend
  (iii)** — it keeps tenants as real processes (the isolation gate stays
  meaningful) and reuses the substrate's own cuIpc path. The harness cannot
  be written until this is fixed.
- **D2b — dtype.** fp16 weights (14 GB, roofline ~2.2 tok/s @50) vs
  INT4/Marlin (4 GB, roofline ~5 tok/s @50, frees ~10 GB HBM). **Recommend
  fp16 for the gate run** (no Marlin-regime confound), INT4 as a stretch arm.
- **D2c — model.** **Mistral-7B** (the CP 2.4/2.5 reference) — or a smaller
  model if shared-weights + 50× 32K-KV is tight.
- **D2d — context length.** §3a shows 8K makes the substrate unnecessary for
  the 50 target; **32K** is the regime where dedup is load-bearing and where
  campaign memory places the moat. **Recommend 32K** as the gate regime.
  Confirm — this choice sets the §3a/§3b rooflines and the §4 pass bar.

**D3 — Dedup call model measured by T4.6.5.** T4.6.4 deferred this here.
Recommendation: T4.6.5 measures **model (a) cold-path** for the dedup-*rate*
gate (matches the T4.6.3/4 method, the rate is call-model-independent), and
**separately characterises model (b)** — the ~400 µs/hit live-path tax — as a
T4.6.6 latency finding, not a T4.6.5 gate. Confirm.

**D4 — p99 budget.** Proposed: p99 inter-token latency ≤ 2× the §3b
roofline-predicted mean for the tenant count. Alternative: an absolute ms
number. Recommend the roofline-relative form (honest to the physics).
Confirm the 2× multiplier.

**D5 — 50 vs 100.** Gate at **50 sustained**; run 100 as a reported stretch,
not a gate. Confirm. *Note — §3a now shows the 739-trace dataset predicts
~30 tenants at 32K (`m≈1.9×`); the 50 gate may not be met on this dataset.
The decision tree picks the tier from the measured `m` regardless; if it
lands L1-only, T4.6.6 honestly validates at the supported tenant count
(20–30) rather than forcing 50. Confirm you accept a decision-tree-driven
tenant target rather than a fixed 50.*

### Follow-on items — flagged, not CP-blocking

- **F1 — context-length sub-arms for T4.6.5.** Measuring `m` at both 25K and
  32K resident context (not just 32K) doubles the tier-boundary information
  for ~2× the dedup-run cost (~3 h → ~5 h). Offered as a user choice — not
  unilaterally taken.
- **F2 — the codebase-overlap gap.** This dataset (`local` hash scope) can
  only measure the system+tool-prefix component of cross-tenant dedup. The
  larger production component — agentic coders sharing repo files / libraries
  — is real but unmeasurable here. Closing that gap needs an augmented
  harness (synthetic shared-file overlay) — a genuine scope addition, a
  candidate next-CP follow-on, **not** part of CP 4.6.5+6.

### libcipher_v2 anchor decision (queued parallel, low-priority)

Per the user's queue: `cc0479b8` vs `86618c30` diff analysis + re-anchor
decision (~1–2 h, any session). Audit §3 already determined `cc0479b8` is the
Phase-3 Task-5 CUPTI build; the remaining work is the formal re-anchor. Not
part of this CP's critical path.

---

## 7. Calendar

| Sub-task | Estimate |
|---|---|
| Trace prep (item 2) — contingent on D1 | 1–3 h |
| T4.6.5 dedup run + decision artifact (items 3–4) | ~3 h |
| T4.6.6 harness (item 5) | 2–4 h |
| Load test 50 + 100 (item 6) | 6–8 h |
| Isolation security test (item 7) | 1–2 h |
| Report (item 8) | 1–2 h |

**Envelope: ~2 days of build + measurement**, GPU-bound by the T4.6.6 load
test. One atomic STEP; one `CP_4_6_5_6_REPORT.md` at close. Closing this CP
closes Phase 4.6 (T4.6.1–6 all done) and strengthens the May 28 Ditlev demo
(real-trace dedup number + a 50–100-tenant density result).

---

## 8. Anchors

Held through the STEP — **no substrate code changes**: kmod 0.4.8 `e2f50452`
(srcversion `E427CAFA4E94D548233DC7A`), libcipher_v2 `86618c30`, libcipher_rt
`c2c5d313`. Build artifacts of the STEP are harnesses and measurement data,
not substrate changes. If the isolation test (item 7) or the live-call-model
characterisation (D3) surfaces a substrate defect, that is a **finding** —
reported, and any fix is a separate adjudicated STEP, not absorbed here.
