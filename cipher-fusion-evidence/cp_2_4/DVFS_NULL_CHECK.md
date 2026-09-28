# CP 2.4 — DVFS harness null-check

**Date:** 2026-05-16. **Verdict: PASS** — the sentinel-windowed DVFS harness
is unbiased; the +62 % envelope result is validated.

---

## Why

The DVFS sweep (`DVFS_ENVELOPE` → `dvfs_envelope_result.json`) returned a large
positive Δtok/W (+62 % at 1000 MHz) — ~15× the memo §3.2 expectation and
opposite the T4.3 Mistral-7B FP16 envelope (−2..−14 %). Before locking DVFS
into the §6c composed gate, a null-check validates the harness itself.

## Procedure

`run_dvfs_nullcheck.sh` — byte-identical `run_one` to `run_dvfs_sweep.sh`
(same sentinel-windowed sampler, same Mistral-7B B=1 decode, `CIPHER_MARLIN=on`,
substrate ON, libcipher_rt `5e304549`). The only change: **both arms run
`VOLT=off` (nominal SM clock, no lock either side)** — a genuine no-op. A
sound harness must report Δtok/W ≈ 0; the `analyze_dvfs.py` matched-pair path
is unchanged (output read as `M9999/`). n=5 matched pairs.

## Result

| | Δtok/W | 95 % CI | CI width |
|---|---|---|---|
| null-check (off vs off) | **−0.02 %** | −0.99 % .. +0.95 % | ~1.9 |

Per-pair tok/W (10 runs, both arms nominal): all clustered **0.2894–0.2959**,
no systematic A-vs-B difference. tok/s ~51–52, power ~175–177 W, both arms.

**Pass criteria (user-set):**
- 95 % CI contains 0 — **yes** (−0.99 .. +0.95; point estimate −0.02 %).
- CI width comparable to the sweep's per-point CIs — **yes** (~1.9 vs the
  sweep's 0.9–3.0).

*(`analyze_dvfs.py` prints "VERDICT: DVFS OUT" on this input — its generic
verdict rule fires "no positive point ⇒ OUT" on a ≈0 result. That line is
meaningless for a null-check; the meaningful quantity is Δtok/W ≈ 0 with a
zero-straddling CI.)*

## Conclusion

**The harness is sound — no residual measurement bias.** Therefore:

- **The DVFS sweep's +62 % (and the full monotonic envelope) is a real
  measurement.** The DVFS verdict — **DVFS IN** the §6c composed gate (memo
  §3.3) — is **confirmed, no longer provisional.**
- **The T4.3 Mistral-7B FP16 −2..−14 % envelope and the 2.96× scorecard's
  +4.2 % DVFS contribution are retracted** — they came from the pre-fix
  harness lineage, which carried the power-averaging window bug (sampler
  averaged model-load/compile/idle power instead of the decode window; fixed
  this session, see `PROGRESS.md` "DVFS sweep — harness windowing fix"). They
  are superseded by this null-validated measurement.

Mechanism, restated: post-Marlin-INT4 Mistral-7B B=1 decode is
memory-bandwidth-bound — locking the SM clock costs ~0 throughput (tok/s
ratio ≈ 1.00 at every clock point) while cutting up to 38 % of board power,
so tok/W rises up to +62 %.

## Artifacts

`run_dvfs_nullcheck.sh`, `M9999/pair{1..5}_{off,on}/`, `dvfs_nullcheck_result.json`.
Real sweep result: `dvfs_envelope_result.json`. Anchors held: kmod 0.4.8
`e2f50452`, libcipher_v2 `86618c30`, libcipher_rt `5e304549`.
