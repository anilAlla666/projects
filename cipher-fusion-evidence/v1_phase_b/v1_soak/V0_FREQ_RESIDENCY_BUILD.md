# Frequency-aware residency (LFU-DA): correct + gated, but aggregate cold-miss is MARGINAL at M=5 — the residual is SIZE-CAPACITY-bound, not recency churn

**Date:** 2026-06-01. **Type:** BUILD (residency-manager change to `cipher_rt_pager.c`/`.h`) + measurement.
**Source:** `cipher_rt_pager.c` (+64/-9), `cipher_rt_pager.h` (+9). Harness: `pager_freq_policy_sweep.py`.
**Staging `.so`:** `9c318ac6` → `2edba0d2`. **Deployed `1f305ce6` UNCHANGED.** Default-OFF, additive, OFF byte-identical
(proven below). D.8 strays (`cublas_shim`/`fairness`) excluded.

**One line:** built the requested frequency-aware eviction policy (LFU with Dynamic Aging). It is correct, coherence-
preserving, and an LRU-byte-identical fallback when off. It does exactly the one thing a frequency policy *can* do at
M=5 — **pin the single hottest model** (Mistral cold-miss 11→3) — but it does **NOT** reduce aggregate cold-miss
(36→36 / 34→31 / 33→31, a deterministic 0–3 pts), because at M=5 the binding constraint is a **size-capacity
(knapsack) wall, not recency churn**: the 3 hottest models are *also* the 3 big ones and cannot co-reside. This is
the "NO SCOPE-DOWN → DIAGNOSE why" outcome — diagnosed, not papered.

## What was built (`mgr_make_room`, default-OFF, env-gated)
LFU with Dynamic Aging (LFU-DA): evict the min-`key` RESIDENT/ref==0 region, `key = freq + age`, where `age` is the
key of the last-evicted victim — a monotone floor (under `g_mgr_mu`) that lets a newly-popular model overtake a
stale-hot one (curing pure-LFU's no-aging pin; ≡ LFU on a stationary workload). `CIPHER_PAGER_POLICY=lfu|lfuda`
selects it; **unset/anything-else = LRU (the byte-identical fallback)**, fixed at `mgr_init`.
- `pg_region` gains `freq` + `key`; `serve_demand` bumps both (HIT fast-path + MISS page-in path) alongside
  `last_used`. The `g_mgr_age` read on the HIT path is a **benign-racy aligned-64 load exactly like `last_used`** —
  the policy value is a heuristic, **never a correctness input** (`cipher_rt_pager.c:319-340`).
- **Coherence is untouched:** victim *choice* ⊥ the eviction-during-use invariant — any `ref==0`/RESIDENT region is
  safe to evict, so the state/ref/`cuCtxSynchronize` machinery is unaffected by which region is selected.

## GATE 1 — correctness + GATE 2 — coherence non-regression (run with `CIPHER_PAGER_POLICY=lfuda`)
- **GATE-A under LFU-DA: PASS** — 6 regions / budget 3, **13,287 evictions × 37,945 serve_demands across 8 threads
  → 0 corrupt, 0 cross-region, 0 fault, 0 wasted, resident == 12 MiB, freeHBM drift 4 MiB.** Eviction-during-use
  coherence holds under the new victim selection.
- **NEGATIVE CONTROL livelock (`-DCIPHER_RESIDENCY_BREAK_LIVELOCK`) under lfuda: DETECTED** (wasted>0).
- **NEGATIVE CONTROL deadlock (`-DCIPHER_RESIDENCY_BREAK_LOCKORDER`) under lfuda: DETECTED** (watchdog fired, exit 3).
- **GATE-A under LRU (default): PASS** (13,180 evicts, 0 corrupt) — unchanged. `test_pager` GATE1/GATE2: PASS.
- **INT4 co-residence non-regression:** the sweep's **KL=0 across swap 5/5** (5 distinct 4-bit incl. the 3 big 7-8B)
  ran **under `CIPHER_PAGER_POLICY=lfuda`** — so the bit-identical-serve gate holds under the policy being shipped.

## GATE 3 — cold-miss MEASURED, LFU-DA vs LRU (M=5 distinct 4-bit, hot-set budget 18 GiB, matched seeds)
reserves(GiB): Mistral 5.01, Qwen2 7.85, Llama-3.1 7.95, TinyLlama 1.37, Llama-3.2-1B 2.35.

| burstiness | LRU cold-miss | LFU-DA cold-miss | per-model cold-miss/req shift (b=1.0) |
|---:|---:|---:|---|
| 1.0 | 36% | 36% | Mistral **11→3** ✓, Qwen2 15→16, Llama3.1 16→17, TinyLla 10→12, Llama3.2 6→9 |
| 0.5 | 34% | 31% | — |
| 0.2 | 33% | 31% | — |

**LRU reproduces the tagged 36/34/33% EXACTLY** (cold-miss is deterministic in (seed, policy) — the LRU-byte-identical
fallback gate). LFU-DA's 34→31 / 33→31 are **real but marginal** (deterministic, not noise). The per-model breakdown
is the discriminator: **LFU-DA pins the hottest model cleanly (Mistral 11→3 — the frequency effect working as
designed), but the aggregate is a wash** because pinning Mistral's 5 GiB leaves *less* room for the others, which then
churn more.

## DIAGNOSIS (binding — "don't accept LRU's 33%"): the residual is SIZE-CAPACITY-bound, not policy-bound
- The 3 hottest models (Mistral 41% + Qwen2 ~20% + Llama3.1 16% = **~78% of traffic**) are *also the 3 big ones*:
  **5.01 + 7.85 + 7.95 = 20.8 GiB > 18 GiB budget.** Any two big models barely fit, so **one big hot model is
  perpetually cold regardless of eviction policy.** Frequency-awareness cannot fix a problem that is about bytes.
- **Static knapsack floor (offline calc):** the served-traffic-maximizing resident subset in 18 GiB is
  {Mistral, Qwen2, TinyLlama, Llama3.2} = 16.58 GiB, serving 84% → sacrifice Llama3.1 → **16% static floor.** The
  **dynamic floor is HIGHER** (every request to a non-resident model pages it in, displacing another — you can't keep
  a requested model evicted), which is why both policies sit at ~31–36%, ~2× the static floor.
- **The lever that would attack this is SIZE-aware, not frequency-aware:** GDSF (greedy-dual-size-frequency; evict by
  `freq/size`) would keep the best freq-per-byte set and stop the two big rank-2/3 models from thrashing. **But that
  is a DIFFERENT policy axis than the frequency-aware policy requested here, and its M=5 payoff is genuinely uncertain
  — the residual is the two BIG rank-2/3 models thrashing (Qwen2 16/36, Llama3.1 17/25), and serving the non-resident
  one is FORCED (every request pages its model in), so the dynamic page-in floor bounds ANY policy. GDSF would likely
  recover ~2-3 pts by stopping the big-model thrash, NOT approach the 16% static floor.** Surfaced as a scope option
  for Anil — not built this turn.
- **"hot≡big" here is a CONFIGURATION DRAW, not a property of frequency-aware policies (do not over-generalize).**
  The harness maps Zipfian rank to list index (`(z-1)%M` over `MODELS=[Mistral, Qwen2, Llama-3.1, TinyLlama,
  Llama-3.2]`), and the 3 big models happen to be listed first — so the hottest are big *by ordering coincidence*,
  independent of size. Had the 2 small models been listed first (hot≠big), LFU-DA would likely **win cleanly** (pin
  3-4 small hot models cheaply, sacrifice the big cold ones). Same flavor of artifact as the earlier 63%-cold
  budget-forced result: a property of the SETUP, not the mechanism. **This strengthens "keep it, gated" — the lever's
  value appears whenever hot≠big or sizes are homogeneous; it is marginal only in this pod's adversarial-for-frequency
  draw.** (A hot≠big re-run is cheap + on-pod if Anil wants it shown empirically — a decision input, not done now.)

## GATE 4 — OFF byte-identical + LRU-fallback byte-identical (PROVEN)
- The pre-edit pager source rebuilds to **md5 `9c318ac6` — bit-identical to the prior tagged staging `.so`**
  (deterministic build; prior staging reproduced exactly from HEAD).
- **Only exported-symbol delta** old(`9c318ac6`)→new(`2edba0d2`) is the **added** `cipher_pager_mgr_freq` diagnostic
  accessor — nothing removed, no signature changed. Pager is not wired into `cipher_v2_init_body` → OFF unchanged by
  construction.
- **Marlin gate smoke is bit-identical on both `.so`s:** `marlin_active=1`, handled 0→8 eager, eager-vs-bf16
  `max_rel=1.148e-01` (identical fingerprint), fired-during-capture True — on **both** old and new.
- **LRU-fallback byte-identical:** LRU sweep reproduces 36/34/33% exactly (above).

## Verdict
**Frequency-aware residency is CORRECT, COHERENCE-PRESERVING, and a clean LRU fallback — worth keeping (it is the
right policy in the hot≠big regime). It is NOT a cold-miss win at M=5.** The result is *conclusion-reinforcing*, not a
null: the cheapest on-pod lever isolated the one thing policy can do (pin the hottest model) and proved the residual
cold-miss is **capacity-bound** — so the binding question remains **M=15 scale, not policy sophistication** (and M=15
is unmeasurable on this 5-model pod). The diagnosed next lever is SIZE-aware eviction (GDSF), a different axis, left
to the ship/scale decision. **STOP for the ship/scale decision.** Anchors unchanged.
[[cipher-zipfian-k-fleet-quality]], [[cipher-pcie-rootcause]], [[cipher-pager-int4-build]].
