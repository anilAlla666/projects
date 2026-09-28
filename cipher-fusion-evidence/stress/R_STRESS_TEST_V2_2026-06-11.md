# STRESS TEST v2 — CORRECTED after the panel caught confounds in v1 (2026-06-11)

The v1 stress test (`R_STRESS_TEST_2026-06-11.md`) was itself confounded — a 4-lens adversarial panel found 10
material methodology problems. The biggest two **changed conclusions**, so v1's spec-decode verdict is RETRACTED.
This v2 re-measures with the fixes: **multi-trial (median + range), true saturation (NREQ=192 — fp16@700 now draws
665 W, vs v1's under-saturated 524 W), iso-power configs, iso-cudagraph spec comparison (both arms eager), and
prefix-caching OFF.** Anchor `2edba0d2` unchanged; substrate not loaded (0/14); power reset.

## CORRECTION 1 — Spec-decode does NOT "break under load." It HELPS at every load. (v1 conclusion RETRACTED.)
v1 reported spec collapsing to 0.61× at NREQ=128. **That was a cudagraph confound:** v1 gave the spec arm a
truncated capture set (`[8,16,24,32]`) while the base arm used vLLM's default (up to 128), so at high batch the
spec run fell back to eager while the base kept cudagraph. The panel caught this. Re-measured **iso-cudagraph (both
eager), prefix-cache off, 3 trials**:

| concurrency (NREQ) | fp8 tok/s | fp8+spec tok/s | ratio |
|---|---|---|---|
| 1 (single user) | 64.8 | 414.6 | **6.40× HELPS** |
| 16 (moderate) | 586.3 | 1194.5 | **2.04× HELPS** |
| 128 (saturated) | 4980.6 | 6957.7 | **1.40× HELPS** |

**Spec-decode's compute genuinely helps throughput at all load levels** (more so at low load). v1's "single-user
latency feature only / hurts a loaded server" was WRONG — an artifact I introduced. **Caveat (real, not a
confound):** this is the eager comparison. In production you'd run the base with cudagraph (faster); composing
spec-decode *with* cudagraph at high concurrency is genuinely harder (the verify batch is K+1× larger, so capture
sizes balloon) — that deployment friction is real, but it is NOT "spec compute hurts." The honest line: spec-decode
helps throughput across loads; making it co-exist with cudagraph at scale is an engineering task, not a wall.

## CORRECTION 2 — TPW is ~1.96×, not 2.0–2.11×, under true saturation. (v1/microbench were optimistic.)
Multi-trial, truly power-saturated (fp16@700 draws 665 W), iso-power. tok/W vs fp16@700 (23.53):

| config | tok/s (median, range) | power | tok/W | **× baseline** |
|---|---|---|---|---|
| fp16 @ 700W (baseline) | 15612 (15494–15642) | 665W | 23.53 | 1.00× |
| fp16 @ 300W (DVFS, lossless) | 9879 | 299W | 33.02 | **1.40×** |
| **fp8 @ 300W** | 13715 | 297W | 46.21 | **1.96×** |
| fp8 @ 250W | 11445 | 249W | 45.97 | 1.95× |
| fp8 @ 200W | 8025 | 199W | 40.29 | 1.71× |
| fp8 @ 700W | 18823 | 592W | 31.82 | 1.35× |
| gptq @ 300W | 10982 | 297W | 36.93 | 1.57× |
| gptq @ 700W | 14728 | 562W | 26.22 | 1.11× |

- **TPW peaks at ~1.96× (fp8@300/250) — just under 2×, NOT the 2.11× v1 reported or the 2.00× microbench.** Those
  were under-saturated (v1's GPU drew 524 W, not power-bound). Under true saturation the fp16 baseline is more
  efficient per watt, so the headroom is ~1.96×. Variance is tight (±1 %).
- DVFS-lossless = **1.40×** (robust, matches prior 1.37–1.39×).

## CORRECTION 3 — the "gptq goes compute-bound, drops below fp16" mechanism was a cross-power artifact (panel).
v1's "0.90×" compared gptq@300W vs fp16@700W (a 400 W gap). **Iso-power, it's nuanced, not "compute-bound":**
gptq@300 (10982) is **1.11× FASTER** than fp16@300 (9879); but gptq@700 (14728) is **0.94× slower** than fp16@700
(15612). So 4-bit helps when power-constrained, hurts when not — the clean dequant-compute-bound story is dropped.
Either way **FP8 dominates both**: fp8@700 (18823) = 1.21× fp16 throughput at iso-power, and best tok/W.

## CORRECTION 4 — Quality is a PROXY, downgrade "4-bit unacceptable."
PPL (185 teacher-forced tokens, 3 passages, single run — **small sample, no CI**): fp16 3.393, FP8 +0.38%, 4-bit
+4.46%. Directionally FP8 ≪ 4-bit on quality cost, but **+4.46 % on 185 in-distribution tokens does not establish
downstream failure** — a real eval (MMLU/GSM8K/HumanEval, hundreds–thousands of tokens) is needed before calling
4-bit "unacceptable." (Also noted: fp16 baseline casts the bf16-native model to fp16.) Conclusion softened to "PPL
proxy suggests 4-bit degrades ~10× more than FP8; needs downstream confirmation."

## Net after correction
| claim | v1 said | **v2 (rigorous)** | holds? |
|---|---|---|---|
| TPW peak | 2.11× | **1.96×** (fp8@300, saturated, multi-trial) | ⚠️ just UNDER 2× |
| DVFS lossless | 1.37× | 1.40× | ✅ |
| Spec-decode under load | "breaks, 0.61×" | **HELPS, 1.40× @N128 / 6.40× @N1** (iso-cudagraph) | ✅ (v1 RETRACTED) |
| FP8 vs fp16 throughput (iso-700W) | — | 1.21× | ✅ |
| Quality | "4-bit unacceptable" | proxy: +0.38% / +4.46%, needs downstream eval | ⚠️ proxy only |

**Honest bottom line:** TPW is **~1.96×** under real saturation (not 2×); spec-decode **helps at all loads** (my
"breaks" was my own confound); FP8 is the right precision on every axis. Two of my prior headline numbers moved —
one down (TPW 2.11→1.96×), one reversed (spec-decode). That's what stress-testing-the-stress-test is for.

## Remaining gaps (unchanged, still real)
Substrate not loaded (0/14 — vLLM-native physics); batch-submit not Poisson arrival (no p99-latency-under-SLO);
no minutes-long thermal soak (runs are 3–5 s); single GPU; short-to-mid context (no KV-dominated long-context where
FP8's weight-byte lever shrinks); spec measured eager-vs-eager (production cudagraph composition untested); quality
is PPL-proxy not downstream-task accuracy.

## Integrity
Anchor `2edba0d2` entry==exit; substrate not loaded (`anchor_loaded:false` ×14); power reset to 700 W; no leftover
processes. v2 = 3 trials/config, median reported with range; NREQ=192 (saturated, fp16 draws 665 W); iso-power +
iso-cudagraph (eager) + prefix-cache-off fixes applied per the panel. v1 report kept and labeled superseded.
