# Frequency-aware residency — hot≠big re-run: the earlier "marginal" was the adversarial configuration draw. When the hot mass FITS in budget, cold-miss halves and LFU-DA gives a consistent modest edge.

**Date:** 2026-06-01. **Type:** measurement re-run (NO source change — LFU-DA already shipped default-OFF, tag
`pager-freq-residency-size-bound`). Harness: `pager_freq_policy_sweep_hotsmall.py` (= `pager_freq_policy_sweep.py`
with the **MODELS list reordered**, small models first). **Anchors UNCHANGED: deployed `1f305ce6`, staging
`2edba0d2`.** Correctness carried (KL=0 across swap 5/5 under both policies, re-confirmed).

**The change (one line):** `MODELS=[TinyLlama-1.1B, Llama-3.2-1B, Mistral-7B, Qwen2-7B, Llama-3.1-8B]` — the SMALL
models are now the Zipfian-hot ranks (`(z-1)%M` maps rank→list index, so index0=hottest), the BIG models the cold
tail. Everything else identical (budget 18 GiB, seed, burstiness sweep, RNG call order). freq-by-index is fixed by
the RNG `sel` sequence; reordering only relabels which model carries which frequency + reserve.

## Result — cold-miss LFU-DA vs LRU, hot≠big vs the earlier hot≡big draw
| burstiness | hot≡big LRU | hot≡big LFU-DA | **hot≠big LRU** | **hot≠big LFU-DA** | static floor |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 36% | 36% | 15% | **12%** | 8% |
| 0.5 | 34% | 31% | 19% | **12%** | 8% |
| 0.2 | 33% | 31% | 18% | **14%** | 8% |

per-model cold-miss/req (b=1.0), reserves TinyLla 1.37 / Llama3.2 2.35 / Mistral 5.01 / Qwen2 7.85 / Llama3.1 7.95 GiB:
- **LRU:** TinyLla 5/66, Llama3.2 3/36, Mistral 3/25, Qwen2 6/24, Llama3.1 7/9
- **LFU-DA:** TinyLla **1/66** (0/69 at b=0.5 — a perfect pin), Llama3.2 2/36, Mistral 4/25, Qwen2 6/24, Llama3.1 7/9

Latency also collapsed (fewer/smaller cold pulls): p50 ~400→~120ms, p99 ~1800→~830ms (still eager/queue-confounded).

## Finding 1 — the driver is "the HOT MASS FITS in budget," not "hot≠big" per se
Reordering halved cold-miss **for BOTH policies** (33–36% → 12–19%). The cause is not the label "hot≠big" — it is
that the high-frequency traffic now **fits**: the freq-optimal resident set {TinyLla, Llama3.2, Mistral, Qwen2} =
16.58 GiB serves **92%** of traffic and fits in 18 GiB; only Llama-3.1 (8%, biggest) spills → static floor **8%**. In
the hot≡big draw the 3 hottest were the 3 big models (78% of traffic needs 20.8 GiB > 18) → it did NOT fit → static
floor 16% and both policies stuck ~2× above it. **Generalizable product statement: when the hot working set fits in
budget, cold-miss is low (~12–17%) and LFU-DA gives a consistent modest edge; when it does not (hot≡big), the regime
is capacity-bound and no eviction policy helps.** "hot≠big" is just one way to make the hot mass fit.

## Finding 2 — LFU-DA beats LRU on all three draws (a clean but MODEST win), closing ~half the gap to the floor
LFU-DA wins every row: 15→12, 19→12, 18→14 (3–7 absolute pts, ~20–37% relative). **These three rows are three
INDEPENDENT `sel` sequences** (not a burstiness curve — the single-thread in-order server serves in request order
regardless of arrival timing, so burstiness has no causal effect on residency; `gamma(shape=b)` rejection-sampling
just consumes a burstiness-dependent number of uniforms before `zipf`, shifting the stream → realized fractions
41/22/16/15/6 vs 43/18/17/14/9 vs 41/19/16/15/9). So LFU-DA winning all three = **robustness across independent
draws**, stronger than a single curve. Sized against the floor: LRU sits ~7–11 pts above the 8% static floor,
LFU-DA ~4 — **LFU-DA closes roughly half the LRU→floor gap.**
- **Why the win is MODEST, not a blowout (honest):** the very-hot small models (TinyLla 41% + Llama3.2 22% = 61% of
  traffic in just 3.72 GiB) are so frequently hit they stay recently-used → **already largely resident under LRU too**
  (TinyLla only 5/66 misses under LRU). LFU-DA's edge is pinning them *harder* (5→1) and not churning them when the
  big cold tail intrudes. The residual under LFU-DA is dominated by the cold-ish big models (Qwen2 6 + Llama3.1 7),
  which is the correct, cheap sacrifice (now only ~23% of traffic vs 78% in hot≡big).
- (Do not over-read the identical-looking final-resident snapshots — that is the trailing in-order KL-swap loop,
  not a policy signal.)

## Verdict
**The advisor's flag is confirmed: the earlier "marginal at M=5" was the adversarial hot≡big CONFIGURATION DRAW, not
a property of the policy.** Under a hot-set-fits configuration the frequency lever delivers — LFU-DA beats LRU
consistently across independent draws and closes ~half the gap to the static floor — and, more importantly, the
pager's cold-miss in this regime is low (~12–17%, near the 8% floor) for both policies. **This proves the lever's
value is real and configuration-dependent; it keeps LFU-DA as a worth-shipping gated policy (it helps whenever the
hot mass fits) and shows the pager's realistic common case is good — *under the hot-set-fits assumption.***

**Honesty guard (binding):** this demonstrates the configuration-dependence; it does **not** assert that real agent
fleets *are* hot-set-fits. Whether a production fleet's hot working set fits the budget is the SAME workload question
the M=15 multi-distinct-model scale test would answer (unmeasurable on this 5-model pod). No source change; default-OFF,
LRU fallback intact. **STOP for the ship/scale decision.**
[[cipher-freq-residency-build]], [[cipher-zipfian-k-fleet-quality]], [[cipher-pcie-rootcause]].
