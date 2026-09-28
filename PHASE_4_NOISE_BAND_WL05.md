# Phase 4 — WL05 same-condition noise band (binding evidence standard)

**Measured:** 2026-05-14 02:58–03:30 UTC, same-condition back-to-back.

## Methodology

Three sequential 600 s WL05 multi-tenant ×8 runs under identical
conditions:
- **Library actually loaded by child processes:** `libcipher_v2.so`
  (Phase 3 substrate). The runner was launched with
  `CIPHER_INJECTION_OVERRIDE=/home/ubuntu/libcipher_rt.so.v0.2.0_T4_2_3`,
  but `run_baseline_wl05.sh` had a hardcoded export that bypassed it.
  Discovered 2026-05-14 03:50 UTC after the B7 verify run (also via
  this script). Confirmed by per-child `/tmp/wl05_t*.log` showing only
  `[cipher_v2]` init lines (no PR/ARB/SMP markers).
- The same-condition noise band number below is therefore the noise
  band for **libcipher_v2 + cipher_kmod 0.4.4**, NOT for libcipher_rt.
  This is still a usable noise band — it bounds the variance of the
  measurement infrastructure itself across an identical-window. The
  cross-library noise question is to be answered separately under the
  corrected script.
- **Original intent (libcipher_rt.so.v0.2.0_T4_2_3, md5
  bc51b9d6f6827ccdd90ad2d5ab09e3ff):** captured in the runner command
  but not actually exercised by the child processes.
- **Kernel module:** cipher_kmod 0.4.4 (srcversion
  `1B657D6043D718B6DE2BC56`, md5 `c6de1afa228fec20883c6659ea3e5fc7`)
- **Daemons:** same `cipher-gpustate` + `cipher-exporter` instances
  across all three runs (no restart between runs)
- **Wall-clock window:** 02:58–03:30 UTC, contiguous (no breaks > 60s)
- **WL05 driver:** `cipher_workloads/measurement/run_baseline_wl05.sh
  --duration 600`

Each run measures aggregate device throughput and power across the 8
TinyLlama child tenants.

## Raw measurements

| Run | wall start | aggregate MFU% | tok/s | watts_avg | TPW |
|---|---|---:|---:|---:|---:|
| 1 | 02:58 UTC | 100.0 | 281.57 | 200.64 | 1.4034 |
| 2 | 03:09 UTC | 100.0 | 277.95 | 199.57 | 1.3927 |
| 3 | 03:19 UTC | 100.0 | 277.75 | 199.53 | 1.3920 |

## Computed band

| Metric | Mean | Stddev | Max-dev | Max-dev % | CV % |
|---|---:|---:|---:|---:|---:|
| TPW | 1.3960 | 0.0064 | 0.0074 | ±0.53% | 0.46% |
| tok/s | 279.09 | (~) | 2.48 | ±0.89% | (~) |
| watts_avg | 199.91 | (~) | 0.73 | ±0.36% | (~) |

Raw stats JSON: `/home/ubuntu/cipher-phase4-evidence/wl05_noise_band_stats.json`.

## Outcome A — tight band

Max-deviation on TPW is **±0.53%**, well under the ≤1% threshold for
"tight" in the brief. The CIPHER substrate under same-condition
measurement is highly stable. Same-condition tok/s spread is also tight
(±0.89%), and watts_avg is essentially flat (±0.36%) — the GPU's
power-cap regulation is the dominant noise floor here, not algorithmic
variance.

## Binding evidence standard for future T4.2.x WL05 lift claims

Per advisor (binding pre-design consultation): the rule is NOT a binary
"beat ±0.53%". A 1-sigma threshold has 32% false-negatives. The honest
three-band interpretation:

| Δ TPW vs same-condition reference | Verdict |
|---|---|
| `|Δ| ≤ 0.53%` (one max-deviation) | **Neutral within measured noise** — cannot claim lift |
| `0.53% < |Δ| ≤ 1.06%` (1×–2× band) | **Ambiguous** — requires same-condition repeat measurement of the build under test before claiming |
| `|Δ| > 1.06%` (>2× band) | **Genuine signal** — claim "lift" (or "regression") with confidence |

This standard binds the next sub-phase (T4.2.4 GREEN_CTX). When T4.2.4
ships, the lift claim against WL05 needs a TPW Δ > 1.06% against a
same-condition T4.2.3 reference to be called lift; anything less is
either neutral or ambiguous, NOT lift.

## Important caveat: day-to-day drift dominates same-condition noise

Today's WL05 mean (1.396 TPW) is **4.3% below** yesterday afternoon's
T4.0.9.D baseline (1.459 TPW). It is also below all of yesterday
evening's runs (1.437–1.462). Combined with the WL14 bisect result
(B8 closed: pod drift, not kmod), the day-to-day pod-state drift on
this pod is on the order of **3–5% TPW**, which is **6–10× the
same-condition noise band** measured here.

**Implication:** the T4.0.9.D static baselines from 2026-05-13 afternoon
are not directly comparable to today's measurements. All future
comparisons must be **matched-pair** (baseline + build measured in the
same wall-clock window, on the same pod-state). The 1.06%
genuine-signal threshold applies to matched-pair Δ, not to Δ vs the
static T4.0.9.D row.

PHASE_4_BACKLOG.md B9 (new) opened for this finding.

## Retrospective on yesterday evening's three-run spread

Yesterday's STATUS_OVERNIGHT_2026-05-13.md described the spread of
1.459 / 1.462 / 1.437 across three different library+kmod conditions as
a "three-run spread across mixed conditions." It quoted ±0.9% as the
spread.

Today's same-condition band is ±0.53% TPW. The 0.4% gap between yesterday's
mixed-condition spread and today's same-condition band is small — the
yesterday-evening runs happened to cluster more by-chance than
expected. The advisor's caution last night ("a real noise band needs
≥2 runs at the same condition") was correct in principle; in this
particular case, yesterday's mixed-condition spread was within ~50% of
the real same-condition band, so the framing-creep cost was small. The
lesson stands: don't promote mixed-condition spreads to noise bands.

## Artifacts

- Per-run JSONs: `/home/ubuntu/cipher-phase4-evidence/wl05_noise_run{1,2,3}.json`
- Stats JSON: `/home/ubuntu/cipher-phase4-evidence/wl05_noise_band_stats.json`
- Runner log: `/home/ubuntu/cipher-phase4-evidence/wl05_noise_runner.log`
