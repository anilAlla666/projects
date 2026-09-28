# Week 5 — 3.E False-Positive Diagnostic — RESULT

**Headline VERDICT: SHARED-CONTENT (specifically zero-page overlap).**
The "1 cross-tenant content match" flagged by 3.E is not a content
collision at all — it's the **single all-zero KV page** that both
tenants legitimately share. The harness metric `t1_misses - t2_misses`
counts t1's once-registered zero-page as a "false positive" because
t2 hits it without needing to re-register. **The substrate is
behaving correctly.** Action: amend the test interpretation; no fix
to substrate or plugin needed.

**Date:** 2026-05-21 (post Week 5 close)
**Substrate state:** UNCHANGED — diagnostic was read-only (one
temporary plugin patch, reverted after analysis).
**Final plugin md5:** `cd8c826f` (matches `plugin_snapshots/cipher_vllm_kvdedup.py.w5_step2`).

---

## A — Pre-edit + GPU clear — PASS

| signal | value |
|---|---|
| cipher_rt_phase4 HEAD | `ec0e005` (week-5-complete + week-5-step-3-n4-kl-gate + week-5-step-1b-substrate-alias) |
| cipher_kmod HEAD | `2fc70c3` (week-5-complete; unchanged from Week 4) |
| cipher-may13-evidence HEAD | `fc8a9ae` (week-5-complete; unchanged from Week 1) |
| Plugin drift check | both files (cipher_vllm_kvdedup.py + sc_kvdedup_n2_different_prompts.py) byte-identical to Step 2/3 snapshots ✓ |
| `cipher_kv_bridge.so` md5 | `f041789c` (unchanged from Step 1b) ✓ |
| GPU mem pre-diag | 0 MiB ✓ |

---

## B — Diagnostic (a): sampling reruns — **NOT SAMPLING**

5 independent reruns of `sc_kvdedup_n2_different_prompts.py`, GPU
cleared between each.

| run | pages | t1 misses | t2 misses | cross-tenant content matches (harness heuristic) | content-FP rate (heuristic) |
|---|---|---|---|---|---|
| 1 | 6556 | 45 | 44 | 1 | 2.22% |
| 2 | 6556 | 45 | 44 | 1 | 2.22% |
| 3 | 6556 | 45 | 44 | 1 | 2.22% |
| 4 | 6556 | 45 | 44 | 1 | 2.22% |
| 5 | 6556 | 45 | 44 | 1 | 2.22% |

**All 5 runs produced byte-identical output.** The 1-page delta is
**deterministically reproducible**, not sampling noise. Rules out
Verdict 1 (SAMPLING ARTIFACT).

Logs: `/tmp/week5_3e_diagnostic/run{1,2,3,4,5}.log`

---

## C — Diagnostic (b): content inspection — DEFINITIVE

### Patch applied (later reverted)

Added env-gated per-page signature dump to `cipher_vllm_plugin/cipher_vllm_kvdedup.py`:
- New env `CIPHER_KVDEDUP_DIAG_DUMP=<dir>`
- Before each `cipher_rt_kv_dedup_alias` call, DtoH-copy the 2 MiB
  page via libcuda ctypes, compute xxh64 via libxxhash ctypes, log
  `(page_idx, va, was_deduped, xxh64, first_64_bytes_hex, last_64_bytes_hex, is_zero_marker)`
  to `<dir>/t<TENANT>_pages.tsv`
- Patch size: ~50 LOC (ctypes binding + dump loop + cleanup)

Ran 3.E once with `CIPHER_KVDEDUP_DIAG_DUMP=/tmp/3e_diag` → both
tenants' per-page TSVs written (1.9 MB each; 6556 rows per tenant).

### Analysis (`/tmp/week5_3e_diagnostic/analysis2.log`)

```
t1 registered hashes: 45 unique
t2 registered hashes: 44 unique
shared registered hashes (t1∩t2 reg): 0          ← ZERO cross-tenant content collisions
t1-registered hashes that t2 hit at least once: 1 ← THIS is the "1 false positive"

Top hash hit by t2:
  xxh=eb8a7322f88e23db  count=6512  in_t1_registered=True
  is_zero(marker)=1  first64=00000000000000000000000000000000...

Distribution of t2-hit-count per t1-registered hash:
  xxh=eb8a7322f88e23db  t2_hit_count=6512  (the ALL-ZERO page; first/last 64 bytes all zeros)
  (other 44 t1-registered hashes: 0 hits from t2)
```

### Interpretation

**The "1 cross-tenant content match" is the zero-page.**

- t1 PUT 6556 pages → 45 are unique-content; 44 are PROMPT_A-specific
  content pages, **1 is the all-zero KV page** that vLLM uses for
  uninitialized KV slots.
- t1's other 6511 pages are also all-zero (vLLM pre-allocates 16 GiB
  KV; only ~44 pages contain prefill content; ~6512 are unfilled
  zero pages). These all dedup against t1's own registered zero-page
  → 6511 hits within-tenant.
- t2 PUT 6556 pages → 44 are unique-content (PROMPT_B-specific), **0**
  shared with t1's content. The other 6512 are t2's own zero-pages →
  all 6512 dedup against t1's already-registered zero-page
  (`xxh=eb8a7322f88e23db`).

The arithmetic identity:
```
t1_unique = 44 (content) + 1 (zero) = 45
t2_unique = 44 (content) + 0 (zero, already registered by t1) = 44
delta     = 1
```

**This delta is the EXPECTED, CORRECT, BENEFICIAL behavior of
cross-tenant zero-page dedup.** Not a false positive. Not a substrate
bug. Not a hash collision.

### Why the harness misclassifies it

The harness computes:
```python
content_misses_dropped = max(t1_misses - t2_misses, 0) = 1
false_positive_rate = content_misses_dropped / max(t1_misses, 1) = 1/45 = 2.22%
```

`t1_misses - t2_misses` was intended to measure "of t2's content
pages, how many false-positive matched t1's content." But it actually
measures "how many fewer NEW physicals did t2 register than t1." If
t1 registered the zero page first, t2 doesn't need to register it
again — that legitimately drops t2's miss count by 1.

The fix is the harness metric, not the substrate. Correct
formulation:
```
real_cross_tenant_content_hits = (t2 hits on t1-registered hashes
                                  EXCLUDING the all-zero hash)
                               = 6512 - 6512 = 0
```

i.e., **zero true cross-tenant content false-positives.**

---

## D — Diagnostic (c): kmod-side memcmp_failures — NOT NEEDED

Diagnostic (b) gave a definitive answer (Verdict 2 SHARED-CONTENT,
specifically zero-page). No need to investigate kmod-side hash
collision behavior; xxh64+memcmp is operating correctly — the 1 shared
hash IS a legitimate match (both tenants' all-zero pages are
byte-identical, so xxh64 matches AND memcmp matches).

---

## E — Verdict

**Verdict 2: SHARED-CONTENT (zero-page overlap; not content collision).**

| classification | applies? | evidence |
|---|---|---|
| Verdict 1: SAMPLING ARTIFACT | **NO** | 5/5 runs byte-identical |
| Verdict 2: SHARED-CONTENT | **YES** | `shared registered hashes = 0`; the one "shared" hash is the all-zero page |
| Verdict 3: HARNESS MISLABEL | partially | the harness metric is misleading at this scenario, but it's measuring exactly what it claims to measure — just that what it measures (`t1_misses - t2_misses`) doesn't isolate "content false positives" from "zero-page sharing" |
| Verdict 4: GENUINE COLLISION | **NO** | `shared registered hashes (t1∩t2 reg): 0` → no content hash matched between tenants except the zero-page (which is byte-identical, not a collision) |

### Substrate health

**The `cipher_rt_kv_dedup_alias` substrate primitive is correct.**
xxh64 + memcmp_verify is operating exactly as designed:
- Same content (the all-zero page) produces the same xxh64 →
  HIT-and-memcmp-confirmed → atomic rebind onto shared physical
- Different content produces different xxh64 → MISS → register new
  physical

The 3.E harness's "false positive" finding is a measurement-language
issue, not a correctness issue.

### Action items

1. **Update WEEK_5_STEP_3_RESULT.md 3.E section** with this diagnostic
   finding. The "borderline PASS under absolute-rate interpretation"
   framing is unnecessary — there is no false positive at all; the
   harness metric was misleading. (Out of this diagnostic's scope to
   land the result-doc edit; flagged for Week 6 entry preflight or
   any future cipher-fusion-evidence editorial pass.)

2. **The `sc_kvdedup_n2_different_prompts.py` test harness should be
   updated** to use a clean metric for content false-positive rate:
   ```python
   # Old (misleading):
   content_misses_dropped = max(t1_misses - t2_misses, 0)
   false_positive_rate = content_misses_dropped / max(t1_misses, 1)

   # New (correct): count t2 hits whose xxh64 matches a t1-registered
   # hash whose content is NOT all-zero. Requires the diagnostic
   # dump infrastructure to be promoted from env-gated to permanent;
   # or simpler: compute the all-zero-page hash up front and exclude
   # it from the cross-tenant counter.
   ```
   Out of this diagnostic's scope; flagged for Week 6+ test-harness
   improvement.

3. **No substrate or plugin change needed.** The diagnostic patch was
   reverted; plugin md5 restored to `cd8c826f` (Step 2 anchor).

---

## F — Cleanup — DONE

| action | result |
|---|---|
| Reverted `cipher_vllm_plugin/cipher_vllm_kvdedup.py` to Step 2 state | md5 restored to `cd8c826f` ✓ |
| Plugin diff vs `plugin_snapshots/cipher_vllm_kvdedup.py.w5_step2` | byte-identical ✓ |
| `cipher_kv_bridge.so` md5 | `f041789c` (unchanged — substrate untouched throughout) ✓ |
| Diagnostic dumps at `/tmp/3e_diag/` | retained for audit trail (3.8 MB total: t1_pages.tsv + t2_pages.tsv) |
| Diagnostic patch backup at `/tmp/week5_3e_diagnostic/cipher_vllm_kvdedup.py.pre_diag` | retained |

---

## Honest notes

1. **5/5 runs ruled out sampling.** Determinism alone wasn't sufficient
   to identify the cause — both real-collision and shared-content
   produce deterministic results — but it WAS sufficient to rule out
   sampling. Diagnostic (b) page-signature dumps were necessary for
   final attribution.

2. **The diagnostic patch ran cleanly.** ~50 LOC ctypes binding to
   libcuda's `cuMemcpyDtoH_v2` + libxxhash's `XXH64`. Patch + analysis
   completed in ~30 min total. Reverted cleanly afterwards.

3. **`is_zero(marker)` heuristic in the dump was misleading.** It only
   checked first/last 512 bytes. All 45 of t1's "registered" pages
   had `is_zero=1` per this marker, but only ONE was actually all-zero
   (xxh `eb8a7322f88e23db`); the other 44 had content in the middle
   of the 2 MiB page but happened to have zero borders. (The xxh64
   hash distinguishes them correctly; the heuristic is just for
   coarse classification.)

4. **The zero-page-sharing IS a real product benefit.** vLLM's
   pre-allocated zero KV at gpu_util=0.20 is ~16 GiB × 6556/45 ≈
   13 GiB per tenant. Cross-tenant deduplication of these zero pages
   is the bulk of the 42-45 GiB HBM savings observed at Step 2/3.

5. **Week 5 closeout's "3.E borderline PASS under absolute-rate"
   framing is now obsolete.** The correct framing is: **3.E is
   unambiguous PASS — zero true cross-tenant content false-positives.**

---

## Updated Week-5 audit-trail position

The Week-5 closeout (`WEEK_5_STEP_4_CLOSEOUT.md` §5 finding #9) said
3.E was "auto-mode adjudication PASS under absolute-rate interpretation";
this diagnostic upgrades that to **PASS unambiguous: substrate has
ZERO true cross-tenant content false-positives between different
prompts; the harness metric `t1_misses - t2_misses` measures
zero-page sharing, not content collisions.**

Week 5 result and tags stand. No re-tagging needed.

---

**Evidence:**
- `/tmp/week5_3e_diagnostic/run{1,2,3,4,5}.log` (Diagnostic a — 5
  determinism-confirming reruns)
- `/tmp/week5_3e_diagnostic/run_dumped.log` (Diagnostic b — single
  run with signature dump)
- `/tmp/3e_diag/{t1,t2}_pages.tsv` (per-page signatures: page_idx, va,
  was_deduped, xxh64, first/last 64 bytes hex, is_zero marker; 6556
  rows × 2 tenants = 13112 entries)
- `/tmp/week5_3e_diagnostic/analyze.py` + `analyze2.py` + `analysis.log`
  + `analysis2.log` (decoder + analysis output)
- `/tmp/week5_3e_diagnostic/cipher_vllm_kvdedup.py.pre_diag` (pre-patch
  backup for revert)
- Final plugin md5: `cd8c826f` (matches Step 2 snapshot;
  diagnostic patch reverted)
