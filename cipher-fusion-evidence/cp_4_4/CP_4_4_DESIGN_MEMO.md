# CP 4.4 — L2 persistent weight pin — DESIGN MEMO

**Date:** 2026-05-16. **Status:** **CP 4.4 CLOSED at memo — §6 adjudicated to
path 6.2 (skip the build).** This memo is the canonical CP-4.4 artifact; no
code, no build, no measurement was performed. Per Phase-4+ engineering-marvel
discipline (carried from CP 4.6.5+6) the memo opens with mechanism
characterization and bound calculation; here the bound (§1) was decisive
enough to close the CP without a build. Closure note: §9.

**This memo's headline finding is unusual and must be read first.** The
working-set bound (§1) shows the L2 persistent-weight-pin lever **cannot earn
material throughput lift in the CP-relevant regime** (Mistral-7B / Llama-3.1-8B,
B=1 decode). The 31.25 MiB hardware persisting-L2 budget is **0.2–0.9 % of the
per-token weight working set**. The pre-registered prediction band (§3) is
therefore a **measured null** — +0.1 % to +0.6 %, inside n=5 measurement
noise. This is not a pessimistic framing; it is the bound. §6 puts three
honest paths to the user before any build week is spent.

**Scope.** Migrate the canonical L2-weight-pin actuator (`cipher_l2_persist.cu`,
"SUSTAIN" / "F3" in the may13 canonical-op tree) into the `cipher_rt_phase4`
production runtime as `cipher_rt_l2_persist.{c,h}`, wire it into the cuBLAS
GEMM dispatch path, and validate whether it adds throughput lift on top of the
CP 2.4 composed stack (Marlin INT4 + DVFS + spec decode, 3.617× tok/W
[3.591, 3.642]).

**Anchors held:** kmod 0.4.8 (srcversion `E427CAFA4E94D548233DC7A`, reference
build `e2f50452`), libcipher_v2 `86618c30`, libcipher_rt `c2c5d313`. This CP
*would* rebuild libcipher_rt with `cipher_rt_l2_persist.o` linked in → a new
libcipher_rt anchor; `c2c5d313` is preserved as the rollback point and the new
anchor recorded only on a successful close.

---

## 1. Mechanism characterization + working-set bound

### 1.1 What the lever does

`cudaAccessPolicyWindow` (set via `cudaStreamSetAttribute`,
`cudaStreamAttributeAccessPolicyWindow`) tags a contiguous device-memory range
with `cudaAccessPropertyPersisting`. Lines in that range resist L2 eviction:
once resident they stay in L2 across kernel launches until explicitly unpinned
or displaced. A read that hits L2 is served at ~10 TB/s; a read that misses to
HBM3 is served at 3.35 TB/s. **Bit-exact:** the lever changes *where in the
memory hierarchy* a weight is read from, never its value — same fp16/INT4
bytes, different placement.

### 1.2 The hardware budget — measured on this pod

Live query, this H100 (`cudaDeviceGetAttribute`):

| Attribute | Value |
|---|---|
| `cudaDevAttrMaxPersistingL2CacheSize` | **32,768,000 B = 31.25 MiB** |
| `cudaDevAttrMaxAccessPolicyWindowSize` | 134,217,728 B = 128 MiB |
| L2 cache total (H100) | 50 MiB |

The access-policy *window* may span up to 128 MiB of address range, but at
most **31.25 MiB stays persistent** (the `hitRatio` controls which fraction of
a larger window persists). **31.25 MiB is a hardware cap — `S ≤ 31.25 MiB` is
not negotiable in software.**

### 1.3 The weight working set — Mistral-7B and Llama-3.1-8B

Both models share transformer geometry (`hidden 4096`, `intermediate 14336`,
`32 layers`, `32 heads`, `8 KV heads`, `head_dim 128`). Per-layer weight
tensors:

| Tensor | Params | fp16 | INT4 (≈0.5 B/param) | Fits 31.25 MiB? |
|---|---|---|---|---|
| q_proj | 16.78 M | 33.55 MB | 8.39 MB | fp16 **no** (1.07×over) · INT4 yes |
| k_proj | 4.19 M | 8.39 MB | 2.10 MB | both yes |
| v_proj | 4.19 M | 8.39 MB | 2.10 MB | both yes |
| o_proj | 16.78 M | 33.55 MB | 8.39 MB | fp16 **no** · INT4 yes |
| gate_proj | 58.72 M | 117.44 MB | 29.36 MB | fp16 **no** (3.6×over) · INT4 yes (barely) |
| up_proj | 58.72 M | 117.44 MB | 29.36 MB | fp16 **no** · INT4 yes (barely) |
| down_proj | 58.72 M | 117.44 MB | 29.36 MB | fp16 **no** · INT4 yes (barely) |
| **per layer** | **218.1 M** | **436.2 MB** | **≈109 MB** | — |

Whole-model weight working set `W` (read once per token in B=1 decode):

| Model | dtype | `W` | lm_head note |
|---|---|---|---|
| Mistral-7B | fp16 | **14.5 GB** | lm_head 32000×4096 = 0.26 GB fp16 |
| Mistral-7B | INT4 (Marlin) | **≈3.7 GB** | lm_head/embed stay fp16 (+0.5 GB) |
| Llama-3.1-8B | fp16 | **16.06 GB** | lm_head 128256×4096 = **1.05 GB** fp16 (128K vocab) |

### 1.4 The bound — `S/W`, the pinnable fraction

A B=1 decode step reads the **entire** weight set `W` exactly once (224 weight
GEMMs across 32 layers; no weight is re-read within a token). The persisting
window protects at most `S = 31.25 MiB` of that traffic across the decode
loop. Between consecutive tokens the full `W` is streamed — L2 (50 MiB) is
flushed ≈290–320× per token — so only the explicitly-pinned `S` survives.

| Regime | `W` | `S/W` (pinnable fraction) |
|---|---|---|
| Mistral-7B fp16, B=1 decode | 14.5 GB | **0.22 %** |
| Mistral-7B INT4, B=1 decode | 3.7 GB | **0.85 %** |
| Llama-3.1-8B fp16, B=1 decode | 16.06 GB | **0.20 %** |

**The canonical actuator was never a transformer-decode lever.**
`cipher_l2_persist.h` states its design target verbatim: *"CIPHER's LNN
weights (<1 MB each × 3 LNNs = <3 MB total)"*, `CIPHER_L2_PERSIST_MAX_BYTES =
4 MB`. It was built for a regime where the **entire hot weight set fits in
L2** (3 MB ≪ 31.25 MiB). Migrating it to 7B-class transformer decode moves it
from a "whole set fits" regime to one **100–460× over budget**. The lever is
sound; the regime is not.

### 1.5 Where the lever *does* earn — reference point

`cudaAccessPolicyWindow` weight-pinning earns material lift only when the hot,
repeatedly-read weight set `W ≤ ~31–50 MiB`:

- **W ≤ 31 MiB** — entire hot set pins; every weight read hits L2. This is the
  canonical LNN regime (3 MB). Corresponds to models ≤ ~16 M params fp16 /
  ~62 M params INT4.
- **Prefill (32K context)** — even when a weight fits, prefill is
  *compute*-bound, not weight-bandwidth-bound (a q_proj prefill GEMM is ~1
  TFLOP of compute against ~10 µs of weight reads); pinning the weight does
  not move the bottleneck. ~null.
- **Spec-decode draft model** — a 0.2–1 B-param draft is 0.4–2 GB fp16, still
  12–60× over the 31.25 MiB budget; `S/W` = 1.5–8 % bounds lift to +1–5 %, and
  B=1 draft decode carries the same launch-latency dominance measured in
  CP 4.6.5+6 (2.6× gap to the bandwidth roofline), which eats most of it.

No CP-4.4-relevant production regime (7B+ agentic decode) is in the
lever's earning band.

---

## 2. L2 hit-rate → throughput lift model

B=1 decode is HBM-bandwidth-bound: time-per-token ≈ `W / BW`. With a pinned
set `S`, the pinned bytes read at `BW_L2`, the rest at `BW_HBM`:

```
T_unpinned = W / BW_HBM
T_pinned   = (W − S)/BW_HBM + S/BW_L2
speedup    = T_unpinned / T_pinned
           = 1 / [ 1 − (1 − BW_HBM/BW_L2) · (S/W) ]
```

With `BW_HBM = 3.35 TB/s`, `BW_L2 ≈ 10 TB/s` ⇒ `BW_HBM/BW_L2 = 0.335`,
factor `(1 − 0.335) = 0.665`:

```
speedup = 1 / [ 1 − 0.665 · (S/W) ]
```

| Regime | `S/W` | predicted speedup | lift |
|---|---|---|---|
| Mistral-7B fp16 | 0.00224 | 1.00149× | **+0.15 %** |
| Mistral-7B INT4 | 0.00846 | 1.00566× | **+0.57 %** |
| Llama-3.1-8B fp16 | 0.00204 | 1.00136× | **+0.14 %** |

This is a **bound, not an aspiration** — and it is an *upper* bound: it credits
the pinned fraction with the full 3× bandwidth ratio and assumes decode is
purely weight-bandwidth-bound. CP 4.6.5+6 measured B=1 decode running 2.6×
*below* its bandwidth roofline (launch-latency-bound) — so the realised lift is
plausibly **even smaller** than the table, because the bottleneck is partly not
bandwidth at all.

---

## 3. Pre-registered prediction band

Per engineering-marvel discipline, registered **before** measurement:

| Regime | predicted point | **pre-registered band** | interpretation |
|---|---|---|---|
| Mistral-7B Marlin INT4, B=1 decode | +0.57 % | **1.000 – 1.008×** | measured null |
| Llama-3.1-8B fp16, B=1 decode | +0.14 % | **1.000 – 1.003×** | measured null |

The Mistral INT4 row **assumes §6-D1 resolves to in-Marlin hooking** — a
shim-level pin contributes 0 % to Marlin-substituted GEMMs (they read their own
packed INT4 buffer, not `call->B`), collapsing that row toward the fp16 figure.

Both bands sit **inside** the n=5 matched-pair measurement noise of the CP 2.4
rig (asserted at ~±1 % on tok/s — if 6.1 is chosen, the first measurement task
establishes the rig's baseline-of-baseline variance; should it exceed ~1 %, the
band is simply "below this rig's resolution" — the same null, framed sharper). The honest pre-registration is therefore: **on top of
CP 2.4's 3.617× tok/W baseline, the L2 weight-pin lever is predicted to land
at 3.59–3.65× — i.e. statistically indistinguishable from the baseline.** Any
measured "lift" outside ±1 % would indicate a *measurement artifact* (clock
drift, thermal, or a confound), not a real L2 effect, and would itself be a
finding to investigate.

The lever **does not earn** in either CP-relevant regime. It would earn only
in the §1.5 small-`W` band, which is out of CP-4.4 scope.

---

## 4. Measurement methodology (if the build proceeds)

- **L2 hit rate — measured, not inferred.** CUPTI metric
  `lts__t_sectors_op_read_lookup_hit.sum` (and `.miss`), captured per decode
  step, pre-pin baseline vs post-pin. The mechanism check is *local*: hit rate
  **on reads to the pinned 31.25 MiB region** must rise. A global hit-rate
  delta will be ~null (31.25 MiB of 14 GB) — the measurement must isolate the
  pinned range, or it will (correctly) show nothing.
- **Throughput.** Mistral-7B Marlin INT4, B=1 decode, the CP 2.4 varied-prompt
  set, **n=5 matched pairs** (pin / no-pin interleaved to cancel drift),
  VOLT@1000 MHz held for consistency with CP 2.4.
- **Composed gate run.** Four arms: `vanilla` / `marlin` / `marlin+L2pin` /
  `all-on` (marlin + L2pin + DVFS + spec). tok/s and tok/W each.
- **Correctness.** Bit-exact output vs the unpinned baseline on a
  representative GEMM workload + cosine-similarity check (hard gate, §5).

---

## 5. Pass criteria (pre-specified)

1. **Bit-exact output** vs unpinned baseline — **hard gate.** L2 placement must
   not perturb a single bit. Any deviation fails the CP.
2. **Mechanism validation** — CUPTI L2 read-hit rate on the pinned region rises
   post-pin. This validates the actuator *works* even though §3 predicts the
   throughput consequence is null.
3. **Throughput within the pre-registered band** (§3) — i.e. **confirmed null**:
   measured tok/s lift in 1.000–1.008× (Mistral INT4). "Pass" here means the
   measurement *agrees with the bound*; it does **not** mean a speedup. A
   measured lift materially outside the band fails engineering-marvel
   validation (the model would be wrong — investigate before claiming).
4. **Composition** — L2 pin composes with the CP 2.4 stack. Trivially true for
   a ~1.00× factor; the real composition question is the §6 Marlin-interaction
   subtlety, not multiplicativity.

---

## 6. Open decisions — adjudication needed before any build

**The §1 bound is decisive and pre-empts the build.** The directive's premise
("pin the hot weights → 3× faster weight reads → tok/s rises") holds only when
the hot set fits the budget; at 7B it does not, by 100–460×. Three honest
paths — **the user picks one before item 2 begins:**

### 6.1 — Build as a characterized null
Execute items 2–8. Outcome: actuator migrated and wired, bit-exact confirmed,
CUPTI shows the mechanism works on the pinned region, throughput measured at
the predicted ~null. Phase 4.4 closes as *"L2-persist actuator shipped into
the production runtime; measured null in the 7B B=1 decode regime; the lever's
earning regime (W ≤ 31 MiB) is documented and out of current scope."* This is
an honest close and it does add a production-runtime module + a measured
data-point, but it spends ~1 build week to confirm a bound already known.

### 6.2 — Skip the build
The §1–§3 bound is sufficient; no measurement adds decision-relevant
information. Phase 4.4 closes on this memo as *"analysed; lever does not earn
in the production regime; not migrated."* Frees the ~1 week for CP 4.7
(fusion+agentic). Risk: Phase 4.4 closes without a shipped artifact — the
campaign's CP-count framing may want a migrated module regardless.

### 6.3 — Re-scope CP 4.4 to a regime where the lever earns
Redefine the CP around the §1.5 earning band. Candidate re-scopes (each is new
memo scope, not a tweak): (a) L2-pin the **spec-decode draft model** if a
draft ≤ ~250 M-param INT4 is in the CP 2.4 stack — bounded +1–5 %, partly
launch-latency-eaten; (b) re-purpose persisting-L2 for **KV-page metadata or a
hot attention sub-tensor** rather than weights — a different working set, not
this actuator. Both need their own bound calculation.

**Recommendation:** **6.1**, with eyes open. It honours the directive's
explicit instruction to build, ships a production module, and produces a
measured null that *closes the question with data* rather than analysis alone
— consistent with how CP 4.6.5+6 surfaced its 13-tenant and D4 findings. But
6.1 must be entered knowing the result: this is a **null-confirmation CP**, and
the report (item 8) will lead with that. If the campaign would rather spend the
week on CP 4.7, **6.2** is fully defensible on the bound. **6.3** only if there
is a concrete small-`W` use case to anchor it.

### Secondary decisions (if 6.1 or 6.3 is chosen)

- **D1 — Marlin-composition hook point.** The cuBLAS shim sees the *fp16* B
  operand (`call->B`). A Marlin-substituted GEMM reads its own *packed INT4*
  weight buffer, not `call->B`. To pin the weights Marlin actually reads, the
  L2 window must be set **inside the Marlin actuator** (over Marlin's packed
  buffer), not at the shim level. Decision: hook L2-pin inside
  `cipher_rt_marlin_actuator.c`, or accept that shim-level pinning only covers
  the non-Marlin (fp16-passthrough) GEMMs. This is the user-flagged
  "Marlin/L2 interaction" finding, surfaced *before* the build.
- **D2 — Llama-3.1-8B fp16 in scope?** §3 registers a band for it; the
  directive says "if scoped in." Recommend: include it as a second measured
  arm — cheap (same harness) and it doubles the regime coverage of the null.
- **D3 — audit-finding handling.** The canonical `cipher_l2_persist_apply()`
  sets one `cudaAccessPolicyWindow` per registered tensor, but the
  access-policy window is **singular per stream** — each `cudaStreamSetAttribute`
  overwrites the previous, so the N-tensor loop only ever pins the *last*
  tensor. The production port must set the window **per-GEMM in the dispatch
  hook**, not via a one-time `apply()`. Note: this is a real correctness bug in
  the canonical code, but **it does not change the §1 bound** — `S ≤ 31.25 MiB`
  is the hardware cap; the bug bounds *which* 31.25 MiB is pinned per op, not
  how much.

---

## 7. Build STEP — 8 items (items 2–8 gated on §6)

1. **Design memo** — this document. Hold for adjudication.
2. **Audit `cipher_l2_persist.cu`** — locate (done: `cipher-may13-evidence/src/`),
   verify it compiles under CUDA 13, verify the `cudaAccessPolicyWindow` path
   on H100, confirm the §6-D3 singular-window finding.
3. **Migrate to `cipher_rt_phase4`** — create `cipher_rt_l2_persist.{c,h}` on
   the `cipher_rt_*` convention; register in the `cipher_inject.c` init chain
   (after `cipher_rt_matmul_dispatch_init()`); hook the per-GEMM window-set
   into the matmul dispatch path (§6-D1/D3 decide shim-level vs inside-Marlin);
   Makefile additions per the CP 2.5 build structure.
4. **Correctness validation** — bit-exact + cosine-similarity vs unpinned
   baseline on a representative GEMM. Hard gate.
5. **L2 hit-rate measurement** — CUPTI `lts__t_sectors_op_read_lookup_hit.sum`
   pre/post-pin; verify the mechanism on the pinned region; document the delta.
6. **Throughput measurement** — Mistral-7B Marlin INT4 B=1 decode, varied-prompt
   set, n=5 matched pairs, VOLT@1000 MHz; four-arm composed gate run.
7. **Composition verification** — confirm multiplicative composition with the
   CP 2.4 stack; update the composed scorecard.
8. **Report** — `CP_4_4_REPORT.md`, CP 2.4 / 2.5 / 4.6.5+6 register, leading
   with bounded-vs-measured and the honest regime verdict.

**Per the directive: hold for adjudication before items 6–8 launch** — and,
per campaign discipline (design-memo→approve→build) and this memo's
premise-challenging bound, **hold now, after item 1, before item 2.**

---

## 8. Calendar & anchors

| Sub-task | Estimate |
|---|---|
| Items 2–3 (audit + migration + wiring) | 2–3 days |
| Items 4–5 (correctness + CUPTI hit-rate) | 1 day |
| Items 6–7 (throughput + composition) | 1–2 days |
| Item 8 (report) | 0.5 day |

**Envelope ≈ 1 week** if §6 selects 6.1. 6.2 closes on this memo (~0 further).
6.3 resets the calendar with a fresh memo.

**Anchors:** kmod 0.4.8 (`E427CAFA4E94D548233DC7A` / `e2f50452`), libcipher_v2
`86618c30`, libcipher_rt `c2c5d313` (rollback point; new anchor recorded only
on successful close with `cipher_rt_l2_persist.o` linked).

**Concurrent housekeeping (not CP-blocking):** libcipher_v2 anchor decision
(`cc0479b8` vs `86618c30` diff) — 1–2 h, any session.

---

---

## 9. Closure note — CP 4.4 STEP closed at memo (§6 path 6.2)

CP 4.4 STEP closed at memo. The L2 weight pin lever does not earn at
production transformer scale (B=1 decode working set 118–524× over L2
persistent budget). The canonical `cipher_l2_persist` actuator was designed
for LNN regimes where total weights fit L2 (<3 MB); the migration to
`cipher_rt_phase4` is deferred indefinitely. Phase 4.4 closes via documented
characterization rather than measured null.

This is the bound-first discipline operating as designed — the wrong CP killed
before build, ~1 week of GPU saved, calendar pulls in for CP 4.7.

**Items 2–8: not executed (path 6.2).** Anchors unchanged — no rebuild, no new
module linked: kmod `e2f50452`, libcipher_v2 `86618c30`, libcipher_rt
`c2c5d313`. The L2-weight-pin actuator is **not** in `cipher_rt_phase4` and is
deferred indefinitely unless a small-`W` regime (memo §1.5: hot weight set
≤ ~31 MiB) surfaces in Phase 5.

**CP 4.4 CLOSED 2026-05-16. The marvel is the bound that prevented the build,
not the lever that did not earn.**
