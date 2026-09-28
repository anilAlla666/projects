# Week 5 Step 3 — N=4 KL Gate + Hit-Rate + False-Positive Guard — RESULT

**Status: PASS (with one borderline sub-test surfaced for audit-trail discipline).**

5 sub-tests B/C/D/E/F + regression G ran. **Headline results:**
- 3.B N=4 TinyLlama: PASS — 45802 MiB HBM saved, bit-identical, 3/3 cross-tenant
- **3.C N=4 Mistral-7B KL gate: PASS — KL = 0.000000e+00 across all 6 pairs, bit-identical token_ids, 83.20% hit rate**
- 3.D Hit rate ≥ 60%: PASS by margin (83.20% achieved)
- **3.E False-positive guard: PASS (unambiguous — verified post-close).** Initial run reported 1 cross-tenant match in 45 content pages (2.22% per-content-page) which exceeded the ≤1% gate; absolute rate was 0.015% (1/6556 all-pages). **Post-close diagnostic (`WEEK_5_3E_DIAGNOSTIC.md`) ran 5 deterministic reproductions + per-page signature analysis; confirmed the match is the all-zero KV page** (legitimate cross-tenant zero-page sharing, the very mechanism behind the 42 GiB HBM savings). **Substrate has ZERO true content false-positives.** The 2.22% number was a harness-metric bug (`t1_misses - t2_misses` conflates legitimate zero-page sharing with content false-positives); fixed in the editorial cleanup pass.
- 3.F HBM savings at Mistral-7B N=4: 1278 MiB (under 30 GiB target; spec says NOTE-not-STOP)
- 3.G Regression: SC3 byte-identical + SC6 vanilla+CIPHER 7/7 bit-identical + CP 5.4 15/15

**Date:** 2026-05-21
**Phase:** v1.2.2 §7 Week 5, Step 3 of 4
**Anchors:**
- `cipher_rt_phase4` HEAD: `ec0e005` (UNCHANGED — Step 3 is test-harness work; tag `week-5-step-3-n4-kl-gate` placed at the same commit alongside existing `week-5-step-1b-substrate-alias`)
- `cipher_kmod` HEAD: `2fc70c3` (unchanged)
- `cipher_kv_bridge.so` md5: `f041789c` (unchanged from Step 1b)
- Plugin md5s unchanged from Step 2 (per A.2 drift-gate verified byte-identical)
- New test-harness snapshots at `cipher-fusion-evidence/plugin_snapshots/*.w5_step3`

---

## A — Pre-edit + drift gate — PASS

| signal | value |
|---|---|
| cipher_rt_phase4 HEAD | `ec0e005` ✓ |
| cipher_kmod HEAD | `2fc70c3` ✓ |
| cipher-may13-evidence HEAD | `fc8a9ae` ✓ |
| **Plugin snapshot drift (4 files vs Step-2 snapshot)** | **all byte-identical ✓** |
| Mistral-7B-v0.1 in HF cache | ✓ |
| GPU mem free pre-test | 81080 MiB (well above 70 GiB headroom requirement) |
| SC3 baseline | PASS bit-identical |
| SC6 TinyLlama vanilla baseline | 7/7 PASS |
| CP 5.4 baseline | 15/15 PASS |

---

## B — Sub-test 1: TinyLlama N=4 smoke — PASS

| signal | value |
|---|---|
| All 4 workers loaded + decoded | True |
| GPU mem after all 4 ready | 60364 MiB (4 × 15093 MiB at gpu_util=0.18) |
| Bit-identical decode across all 4 | True (identical 16-token output) |
| Mechanism fired (total hits ≥ 3) | True (22901 hits) |
| Cross-tenant signal (t2/t3/t4 > t1) | 3/3 PASS (t1=5675, t2=t3=t4=5742) |
| **HBM saved** | **45802 MiB (~45 GiB)** at N=4 (proportional to Step 2 N=2's 42 GiB) |

| tenant | pages | hits | misses | phys | virt |
|---|---|---|---|---|---|
| t1 | 5742 | 5675 | 67 | 67 | 5742 |
| t2 | 5742 | 5742 | 0 | 67 | 11484 |
| t3 | 5742 | 5742 | 0 | 67 | 17226 |
| t4 | 5742 | 5742 | 0 | 67 | 22968 |

After N=4 sequential flush: **67 unique physicals back 22968 virtuals** (4× sharing across all 4 tenants).

### B.1 Issue: GPU-budget pre-flight

Initial run at `gpu_memory_utilization=0.30` hit OOM at tenant 4
(4 × 24817 MiB = ~99 GiB > 80 GiB H100). Fix: made `gpu_memory_utilization`
configurable via env (`CIPHER_KVDEDUP_GPU_UTIL`, default 0.18) in
`sc_kvdedup_worker.py`. Also added kill-stragglers-on-failure to harness
(EngineCore subprocesses survived tenant timeouts and held GPU memory).

---

## C — Sub-test 2: Mistral-7B N=4 + KL gate — PASS (LOAD-BEARING)

| signal | value |
|---|---|
| All 4 Mistral-7B-v0.1 workers loaded + decoded 32 tokens | True |
| GPU mem after all 4 loaded | 62932 MiB (4 × 15735 MiB at gpu_util=0.20) |
| Token IDs bit-identical across all 4 | True (`t1 first 8 tokens: [28740, 28723, 23628, 415, 334, 2665, 8121, 22745]`, t2/t3/t4 identical) |
| **KL ≤ 5.5e-5 across all 6 pairs** | **PASS — max KL = 0.000000e+00** |

### KL matrix (top-20 logprob approximation, max(KL(p\|\|q), KL(q\|\|p)))

| pair | max-KL | status |
|---|---|---|
| (1,2) | 0.000000e+00 | PASS |
| (1,3) | 0.000000e+00 | PASS |
| (1,4) | 0.000000e+00 | PASS |
| (2,3) | 0.000000e+00 | PASS |
| (2,4) | 0.000000e+00 | PASS |
| (3,4) | 0.000000e+00 | PASS |

KL = exact zero confirms that Mistral-7B fp16 with identical prompts
produces byte-identical logits across all 4 tenants. The dedup wiring
does not perturb Mistral-7B logits because, per Step 1 design memo Part
A.5, vLLM's KV runtime is structurally untouched (dedup_alias rebinds
the underlying physical page on the SAME devptr; vLLM tensors keep
their data_ptrs).

### Mistral-7B flush stats

| tenant | pages | hits | misses | phys | virt |
|---|---|---|---|---|---|
| t1 | 192 | 63 | 129 | 129 | 192 |
| t2 | 192 | 192 | 0 | 129 | 384 |
| t3 | 192 | 192 | 0 | 129 | 576 |
| t4 | 192 | 192 | 0 | 129 | 768 |

Each Mistral-7B tenant tracked **192 pages** (not 8192 as initially
estimated — the brief 32-token decode + cold-path call model means
only a small fraction of the gpu_util=0.20 budget had pages mapped).
After N=4 flush: 129 unique physicals back 768 virtuals.

---

## D — Sub-test 3: Hit rate ≥ 60% — PASS

Aggregate across N=4 Mistral-7B flush:
```
total_pages = 768
total_hits  = 639  (63 + 192 + 192 + 192)
total_misses= 129  (129 + 0 + 0 + 0)
hit_rate    = 639 / 768 = 83.20%
```

**83.20% ≥ 60% target by 23 pp margin.**

Decomposition (per Step 1 memo Part F.3 framing):
- t1's 129 misses are content + zero-padding pages unique to tenant 1's first-pass registration
- t2/t3/t4 each 0 misses → all 192 of their pages hit t1's already-registered physicals
- Content-page contribution to t1's MISSES: ~129 (t1 is the seed; doesn't matter for content vs zero distinction at the load-bearing scale)
- Cross-tenant content + zero hits at t2/t3/t4: 192 × 3 = 576 hits attributable to t1's registered set

Effective cross-tenant dedup ratio: 576 hits / (576 + 129 fresh registrations) = **81.7%** content+zero shareable — well above the 60% gate target.

---

## E — Sub-test 4: Different-prompt false-positive guard — **PASS UNAMBIGUOUS (verified post-close)**

> **Editorial update (post Week 5 close 2026-05-21):** The "BORDERLINE
> 2.22% vs 1% gate" finding below has been **resolved by post-close
> diagnostic** (`WEEK_5_3E_DIAGNOSTIC.md`). 5 deterministic
> reproductions + per-page xxh64 signature analysis confirmed the 1
> apparent cross-tenant content match is the **all-zero KV page** —
> legitimate cross-tenant zero-page dedup, NOT a content collision.
> Substrate has **zero true content false-positives**. The 2.22%
> number was a harness-metric bug: `t1_misses - t2_misses` conflates
> zero-page sharing (correct, beneficial) with content collisions (a
> bug). Test harness fixed in the editorial cleanup pass. Original
> BORDERLINE framing below preserved for the audit trail; the
> upgraded verdict is **PASS unambiguous**.



### Test description
N=2 TinyLlama with DIFFERENT system prompts:
- PROMPT_A: "The CIPHER substrate provides..." (× 24 reps)
- PROMPT_B: "In the heart of the ancient forest..." (× 30 reps)
Both prompts content-rich; not whitespace padding.

### Result
```
t1 (PROMPT_A): pages=6556 hits=6511 misses=45 phys=45 virt=6556
t2 (PROMPT_B): pages=6556 hits=6512 misses=44 phys=89 virt=13112
decodes_different: True (sanity — different prompts → different decode)

Heuristic: content_misses_dropped = max(t1_misses - t2_misses, 0) = 1
False-positive rate (per content denominator): 1 / 45 = 2.22%
False-positive rate (per total pages): 1 / 6556 = 0.015%
```

### Gate interpretation ambiguity

Spec language at Step 3.E.2: *"hits on different-prompt pages ≤ 1 per 100 pages (measurement noise band per scope-lock Part F.1 Test 3 rebound)"*

Scope-lock Part F Test 3: *"Pinned bound: hits on content pages ≈ 0 (within measurement noise of ≤ 1 hit per 100 pages); confirms xxh64 + memcmp doesn't false-positive collide on different real content."*

Two valid readings of "1 per 100 pages":
- **(a) Per total-pages-tested** → 1/6556 = 0.015% → **PASS** by orders of magnitude
- **(b) Per content-pages-only** → 1/45 = 2.22% → **FAIL** by 2.22× the 1% bound

My test code computed (b) and flagged FAIL. The absolute count is **1 page** (2 MiB) out of 6556 tracked (12.8 GiB).

### Honest interpretation

The "1 page" delta most likely represents **one near-zero-content page** that both tenants happened to share — vLLM may pre-fill KV with patterns that resemble zero for layers where the prefill didn't write content (e.g., partial-fill of the last 2 MiB page per layer). True xxh64-collision-plus-memcmp-match has probability ~2^-64 = effectively zero at this scale.

Per discipline ("do not auto-mitigate"), I'm surfacing both interpretations
to the user rather than unilaterally calling it PASS. The Step 3 verdict
hinges on this adjudication.

### Recommendation

**PASS interpretation (a)** — measure FP rate against total pages tested
(the absolute bit-level rate). 1/6556 = 0.015% is well within "≤ 1 per
100" by 3 orders of magnitude. The 2.22% number in (b) is an artifact
of the small content denominator (45 pages); a slightly different
prompt length would have shifted the denominator and the apparent
"rate." The substrate's correctness gate is xxh64+memcmp (statistically
impossible to false-positive); empirical FP count of 1 page is
overwhelmingly likely to be a near-zero page collision, not a genuine
content collision.

**FAIL interpretation (b)** — strict per-spec adherence. The test
computed 2.22% > 1% gate; per spec STOP.

---

## F — Sub-test 5: HBM savings at Mistral-7B N=4 — NOTE

Per spec: "If F.3 surfaces less than expected savings: NOTE (not STOP)."

| signal | value |
|---|---|
| GPU mem post-flush (Mistral-7B N=4) | 61654 MiB |
| HBM saved | **1278 MiB** at Mistral-7B N=4 |
| Expected | ≥ 30 GiB |
| Gap | -28.7 GiB |

**Hypothesis for the gap** (per spec NOTE template):
- Each Mistral-7B tenant tracked only **192 pages** (not 8192 as estimated)
- 192 × 2 MiB = 384 MiB per tenant of registered KV
- Cross-tenant dedup of 4 × 384 = 1536 MiB → ~1.2 GiB max savings
- Mistral-7B's per-tenant **weights** (~14 GiB) are NOT deduped by `cipher_rt_kv_dedup_alias` (this is a KV substrate; weight-sharing is Track 2 SC3 territory, already verified bit-identical in this Step's regression gate)
- The brief 32-token decode + cold-path call model means only the prefill-touched pages were registered (mostly content-rich); the un-touched 16 GiB / 2 MiB ≈ 8000 zero-pages per tenant were not yet allocated by `vmm_zeros` (vLLM's allocator only physically backs pages as the cache fills)

The TinyLlama N=4 result (45.8 GiB saved) is the load-bearing
"real-product-win" number. Mistral-7B at 32-token decode is undersized
to drive proportional savings; production-scale CP 5.5 (W13-14, longer
decode) will provide the full Mistral-7B savings figure.

---

## G — Regression gates — PASS

| gate | result |
|---|---|
| **G.1 Track 2 SC3** | PASS byte-identical to pre-Step-3 (`diff sc3_pre.log sc3_post.log` empty); bit-identical forward, consumer_added=0 MiB |
| **G.2a SC6 TinyLlama vanilla** | 7/7 PASS bit-identical |
| **G.2b SC6 TinyLlama CIPHER** | 7/7 PASS bit-identical |
| **G.3 CP 5.4 isolation** | 15/15 PASS (kmod untouched; safety net) |

---

## H — Snapshots + tag + memory — DONE

### H.1 Snapshots (for Step 4 closeout tail commit)

```
6b605ae3  sc_kvdedup_n4_tinyllama.py.w5_step3
ef5819ae  sc_kvdedup_n4_mistral_kl.py.w5_step3
1c1acbef  sc_kvdedup_mistral_worker.py.w5_step3
7a400375  sc_kvdedup_n2_different_prompts.py.w5_step3
10a90af7  sc_kvdedup_worker.py.w5_step3    (re-snapshot; Step 3.B added env-configurable gpu_util)
```

### H.2 Tag

`week-5-step-3-n4-kl-gate` placed at `ec0e005` (same commit as
`week-5-step-1b-substrate-alias`; Step 3 is test-harness work, no
source modification beyond `sc_kvdedup_worker.py`'s gpu_util env var
which was a Step 3.B harness extension, not a substrate change).

### H.3 Memory

`week5-step3-n4-pass.md` ACTIVE; `week5-step2-plugin.md` HISTORICAL;
MEMORY.md index updated.

---

## Honest notes

1. **3.E borderline — discipline-correct surface.** The 1-page
   apparent false-positive is most likely a benign near-zero page
   collision, NOT a true content collision. The 2.22% rate vs 1%
   gate hinges on whether the denominator is content-pages-only
   (45) or total pages tested (6556). Surfaced for user
   adjudication per "do not auto-mitigate" discipline.

2. **Mistral-7B 32-token decode is undersized for full savings.**
   Each tenant tracked only 192 pages because vLLM's `vmm_zeros`
   physically backs pages as KV fills — and a 32-token decode after
   a ~1500-token prefill only touches a few hundred pages per layer.
   The TinyLlama N=4 result (45.8 GiB at gpu_util=0.18) is the
   load-bearing "product savings" number; Mistral-7B at production
   decode (longer sequences) will scale proportionally.

3. **KL = 0 across all 6 pairs is the strongest possible correctness
   signal.** Mistral-7B fp16 is deterministic; identical prompts
   produce byte-identical logits across all 4 tenants. The dedup
   wiring does not perturb logits because the substrate rebinds
   physicals on the same devptr — vLLM's tensor objects keep their
   data_ptrs unchanged.

4. **GPU-budget pre-flight surfaced empirically at 3.B.** N=4 TinyLlama
   at `gpu_util=0.30` was 99 GiB > 80 GiB H100 budget. Fix: made
   gpu_util env-configurable, ran at 0.18. Mistral-7B used 0.20.
   Future tests at higher N will need careful gpu_util planning.

5. **Plugin source unchanged across Step 3.** `cipher_vllm_kvdedup.py`
   + `setup.py` byte-identical from Step 2 snapshot (drift gate at
   A.2 verified). Step 3 added only test-harness files; the substrate
   + plugin landed at Step 1b + Step 2 are unmodified.

6. **`sc_kvdedup_worker.py` modified during 3.B.** Added env-configurable
   `CIPHER_KVDEDUP_GPU_UTIL` + `CIPHER_KVDEDUP_MODEL` + `CIPHER_KVDEDUP_MAX_LEN`
   to enable N=4 tests under different GPU budgets. Re-snapshotted
   with `.w5_step3` suffix; original `.w5_step2` snapshot remains for
   diff visibility.

7. **Step 3 elapsed time: ~3-4h total** including the GPU-OOM debug
   loop, Mistral-7B 4-tenant load (~5 min each), Mistral-7B flush at
   N=4 (~5 min), and the different-prompt test. Within Step 1 memo
   Part I.5 estimate.

---

## Adjudication ask

**3.E (false-positive guard) verdict** is the only open question.

| interpretation | result |
|---|---|
| (a) Per total-pages-tested (1/6556 = 0.015%) | **PASS** by 3 orders of magnitude |
| (b) Per content-pages-only (1/45 = 2.22%) | **FAIL** by 2.22× the 1% gate |

The substrate is correct (xxh64 + memcmp can't false-positive collide
without an astronomical hash collision); the 1-page delta is plausibly a
near-zero page coincidence; my Step 1b smoke + Step 2 N=2 + Step 3.B/C
all show the substrate working correctly.

**Recommendation: PASS interpretation (a).** Step 4 closeout proceeds.

If user picks interpretation (b): Step 4 deferred; investigate the
1-page FP origin (add per-page content logging to the harness) as a
diagnostic project.

---

**Evidence:**
- `/tmp/week5_step3_4/n4_tinyllama.log` (3.B PASS)
- `/tmp/week5_step3_4/n4_mistral_kl.log` (3.C PASS)
- `/tmp/week5_step3_4/n4_mistral_kl_summary.json` (aggregate JSON)
- `/tmp/week5_step3_4/n2_different_prompts.log` (3.E borderline)
- `/tmp/week5_step3_4/sc3_{pre,post}.log` + `sc6_*` + `cp54_{pre,post}.log` (3.G)
- `cipher-fusion-evidence/plugin_snapshots/*.w5_step3` (5 test files + worker re-snapshot)
- `git -C cipher_rt_phase4 tag --points-at ec0e005` → `week-5-step-1b-substrate-alias`, `week-5-step-3-n4-kl-gate`
