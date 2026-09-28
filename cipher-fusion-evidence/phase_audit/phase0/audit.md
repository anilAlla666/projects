# CIPHER Phase 0 — Evidence Audit (CP 0.1 – CP 0.6)

**Audit date:** 2026-05-15
**Auditor scope:** read-only. No code written, no workloads run. Evidence gathered
via `ls`/`stat`/`md5sum`/`wc -l`/`grep`/`cat` only.
**Canonical reference:** the "canonical Phase 0-9 plan" CP 0.1–CP 0.6. No copy of
that plan document exists on the pod; it is the external rubric this audit scores
against. No git repos exist anywhere in the evidence trees — file timestamps
(`ls -l`/`stat`) are the dating evidence throughout.

---

## 1. CP-by-CP status table

| CP | One-line scope | STATUS |
|----|----------------|--------|
| CP 0.1 | TinyLlama smoke test | **SUPERSEDED** (by `smoke_marlin_tinyllama`) |
| CP 0.2 | D1 Mistral-7B single-tenant calibration | **PARTIAL** (receipt produced; runner reports `OVERALL: FAIL` — `agg_tps` 36.39 outside the [60,250] calibration band) |
| CP 0.3 | D2 four-tenant dry run | **SHIPPED** (dry-run-specific gates — fairness, 4/4 tenants — PASS; harness-ceiling FAILs are documented and out of dry-run scope) |
| CP 0.4 | Test A continuous decode sweep (N=4…128) | **NOT DONE** |
| CP 0.5 | Test B agentic burst sweep (N=8…192) | **NOT DONE** |
| CP 0.6 | Final report (md + JSON, both stories, MFU/HFU/TPW/density per step, $/M tok power-only) | **NOT DONE** |

**Headline:** Phase 0 was built as a four-step density-sweep program
(D1 calibration → D2 dry run → Test A sweep → Test B sweep → final report). The
harness for all of it was built and verified. D1 and D2 ran and produced receipts.
The runner script then **explicitly STOPs after D2** ("STOP — waiting for explicit
go on the full sweep") and the two sweeps plus the final report were never executed.
A diagnostic report dated the same day records *why* the sweeps were deferred and
states the next session moved on to "Phase 1 — `cipher_kmod` skeleton". Phase 0 was
abandoned at 2 of 4 measurement steps, by design and with the drift documented.

---

## Evidence location note

Two unrelated things both call themselves "Phase 0" on this pod:

1. `cipher-may13-evidence/CLAUDE.md` has a section **"Phase 0 — Hardware
   measurements (2026-04-28)"** — clock/power sweeps, `cudaGetDeviceProperties`
   profiling, M1–M4 GEMM baseline. That is a *hardware-characterization* Phase 0
   from a 12-stage silicon-build numbering. It is **not** the canonical CP 0.1–0.6
   plan (no TinyLlama smoke, no Mistral calibration, no tenant sweeps) and is not
   scored here.

2. The canonical CP 0.1–0.6 plan (TinyLlama smoke → Mistral calibration →
   four-tenant dry run → decode sweep → burst sweep → final report) is implemented
   by the **density-sweep harness** in
   `/home/ubuntu/cipher-may13-evidence/stress2/`. The naming is a one-to-one match:
   the runner is literally named `density_d1d2.py`, its docstring reads
   *"D1: single-tenant 60s continuous decode (calibration) / D2: 4-tenant 60s
   continuous decode (multi-thread dry run) / STOP after D2"*, and the harness
   metrics emit `mfu_pct`, `hfu_pct`, `tpw`, and `dollar_per_M_tok_power_only`
   — the exact CP 0.6 metric set. This audit scores against that tree.

The `cipher_workloads/` WL01–WL24 baseline harness (Phase 4.0.9, JSONs dated
2026-05-13/14) is a *different* measurement axis — 24 workload archetypes × 1
baseline each — and is **not** a Phase 0 substitute. It is noted only where it
could be naively mistaken for one (CP 0.1, CP 0.3, CP 0.4).

---

## CP 0.1 — TinyLlama smoke test — **SUPERSEDED**

**Original gate intent:** a minimal smoke run confirming TinyLlama loads and
generates coherently under the CIPHER substrate before any heavier measurement.

**Substitute work:** `smoke_marlin_tinyllama` — a TinyLlama-1.1B-Chat-v1.0 smoke
test that loads the model under `libcipher_rt.so`, Marlin-patches every INT4-eligible
linear, generates 80 tokens, and gates on compression count + Marlin counter +
output coherence.

**Evidence on disk:**

| Artifact | Path | md5 | lines |
|---|---|---|---|
| Smoke driver | `/home/ubuntu/cipher-may13-evidence/stress2/smoke_marlin_tinyllama.py` | `5eb50fafa6cf0c3c9d5dbf5ba56d3a2c` | (10834 bytes) |
| Smoke run log | `/home/ubuntu/cipher-may13-evidence/stress2/smoke_marlin_tinyllama.log` | `d86bcd044c65844764a70a326e68e6b6` | 70 |

**Indicator / numeric results (from the log):**
- `torch=2.7.0 cuda=True`; model loaded in 0.61 s.
- `n_compressed = 155` linears Marlin-patched, `skipped = 0`.
- `Counters.marlin = 12246` (Marlin GEMM kernel fired).
- `gen tokens = 80/80`; `tps = 71.85`; `coherent = True`.
- PASS/FAIL block: `n_compressed > 0 : PASS`, `Counters.marlin > 0 : PASS`,
  `text coherent : PASS`.

**Equivalence argument:** the canonical CP 0.1 is "TinyLlama smoke test" — the
weakest gate in Phase 0, asking only that TinyLlama runs and produces sane output.
`smoke_marlin_tinyllama` does exactly that: it is a TinyLlama run, it is explicitly
labelled `[SMOKE]`, it is short (80 tokens), and it gates on coherence. It does
more than the canonical asks (it also exercises the Marlin INT4 path) but nothing
less; the canonical intent — "confirm TinyLlama works under CIPHER before the real
measurement" — is fully met. Marked SUPERSEDED rather than SHIPPED only because the
canonical plan's literal artifact (a plan-named "CP 0.1 smoke") does not exist; an
equivalent-or-stronger smoke does. If the auditor prefers, this is a borderline
SHIPPED.

**Reproducibility command:**
```bash
cd /home/ubuntu/cipher-may13-evidence/stress2 && python3 smoke_marlin_tinyllama.py
```
(Caveat: the script and logs reference `/workspace/...` paths; on this pod real
source lives under `/home/ubuntu`, so `ROOT`/`libcipher_rt.so` paths may need
adjustment — see the may13 tree's own path notes.)

---

## CP 0.2 — D1 Mistral-7B single-tenant calibration — **PARTIAL**

**Gate intent:** a single-tenant Mistral-7B continuous-decode run that calibrates
the density harness and establishes the N=1 reference point (tps, MFU/HFU, TPW,
$/M tok, memory).

**Evidence on disk:**

| Artifact | Path | md5 |
|---|---|---|
| D1/D2 runner | `/home/ubuntu/cipher-may13-evidence/stress2/density_d1d2.py` | `91da210c5eb416af24727d3d56d92bce` (153 lines) |
| Density harness | `/home/ubuntu/cipher-may13-evidence/stress2/density_harness.py` | `d5187d09bea005cae7a4e3e53ab8da30` (429 lines) |
| Metrics collector | `/home/ubuntu/cipher-may13-evidence/stress2/cipher_metrics.py` | `e80d21fae3107bf01a81601cf8c72b31` (376 lines) |
| Run log (D1+D2) | `/home/ubuntu/cipher-may13-evidence/stress2/density_d1d2.log` | `7b1eb3fe7c4e6155b94594111c80e021` (119 lines) |
| **D1 receipt JSON** | `/home/ubuntu/cipher-may13-evidence/cipher_metrics_receipts/d1_single_60s.json` | `f5a07be95eae82e0f48b45638a18ce85` (57364 bytes) |

**Numeric results (D1, from receipt `aggregate` block + log lines 61–76):**
- Model: Mistral-7B-v0.1 fp16 + Marlin INT4; 225 linears compressed.
- `n_tenants = 1`, `duration = 60.0 s`, `n_samples = 365`, `total_tokens = 2184`.
- `agg_tps = 36.39`; `achieved_TFLOPS = 0.5219`.
- `mfu_pct = 0.0528 %` (/989), `hfu_pct = 0.0791 %` (/660 this-pod ceiling).
- `tpw = 0.2323 tok/J`; `dollar_per_M_tok_power_only = $0.119554`.
- `mean_power_w = 156.6`, `mean_clock_mhz = 1980`, `peak_mem_mib = 39096` (38.18 GB).
- per-token latency: p50 27.4 ms, p95 28.8 ms, p99 29.7 ms.

**Gate status — why PARTIAL, not SHIPPED.** The D1 run completed and left a full
receipt on disk with every canonical metric (MFU/HFU/TPW/$/M-tok) populated — that
much is real and is the case *for* counting it as progress. But the runner's own
PASS/FAIL block prints **`OVERALL: FAIL`**, with `[FAIL] D1: agg_tps in [60,250]
(agg_tps=36.39)` and `[FAIL] D1: peak_mem < 20 GB (38.18 GB)`. The canonical CP is a
*calibration* — its job is to produce a trustworthy single-tenant reference — and the
harness's own throughput sanity band failed by ~1.6×. The harness's
`DIAGNOSTIC_REPORT_2026_05_13.md` explains both FAILs as known prototype-harness
limits (36 tps vs the ~119 tps May-2 figure = eager-mode launch ceiling + ~27%
Marlin-lock overhead; the 38 GB peak = `orig.weight` deliberately retained because
Marlin's M≥128 prefill falls back to `F.linear`). That explanation is honest and
makes the *number* interpretable — but it does not make the calibration band pass.
A calibrated reference whose own runner declares `FAIL` is a reference with a flagged
anomaly, not a clean calibration. **PARTIAL.**

**Specific gaps and how to close each:**
- *Gap 1 — `agg_tps` 36.39 outside the [60,250] band.* Eager-mode launch ceiling +
  Marlin-lock overhead. Close by re-running the calibration on a graph-capable /
  kmod-backed harness, or by consciously re-baselining the band to the prototype
  harness's real range and re-labelling the result a "prototype-ceiling calibration."
- *Gap 2 — `peak_mem` 38.18 GB over the 20 GB band.* `orig.weight` retained for the
  prefill fallback. Close by either freeing `orig.weight` once a Marlin parallel-M
  prefill kernel exists, or re-baselining the memory band to account for the
  documented dual-weight cost.

**Reproducibility command:**
```bash
cd /home/ubuntu/cipher-may13-evidence/stress2 && python3 density_d1d2.py
# runs D1 then D2; receipts written to /tmp/cipher_metrics/{d1_single_60s,d2_n4_60s}.json
```

---

## CP 0.3 — D2 four-tenant dry run — **SHIPPED**

**Gate intent:** a four-tenant Mistral-7B continuous-decode dry run verifying the
multi-tenant harness runs cleanly, all tenants produce tokens, and fairness holds —
the correctness gate before the full N-sweep.

**Evidence on disk:**

| Artifact | Path | md5 |
|---|---|---|
| **D2 receipt JSON** | `/home/ubuntu/cipher-may13-evidence/cipher_metrics_receipts/d2_n4_60s.json` | `132e70fe9fb146d98ccc19b108476f5f` (50113 bytes) |
| Run log (D1+D2) | `/home/ubuntu/cipher-may13-evidence/stress2/density_d1d2.log` | `7b1eb3fe7c4e6155b94594111c80e021` (119 lines) |

(Runner / harness / metrics-collector files and md5s as listed under CP 0.2.)

**Numeric results (D2, from log lines 82–97):**
- `n_tenants = 4` (exactly four — matches the canonical "four-tenant"),
  `duration = 60.1 s`, `n_samples = 308`, `total_tokens = 1027`.
- `agg_tps = 17.10`; per-tenant tps 4.24 … 4.31; `fairness_ratio = 1.0152`.
- `achieved_TFLOPS = 0.2450`; `mfu_pct = 0.0248 %`; `hfu_pct = 0.0371 %`.
- `tpw = 0.1279 tok/J`; `dollar_per_M_tok_power_only = $0.217252`.
- `mean_power_w = 133.6`, `peak_mem_mib = 39680` (38.75 GB).
- PASS/FAIL: `[PASS] D2: fairness < 1.10 (1.0152)`,
  `[PASS] D2: all 4 tenants produced tokens (4/4)`.

**Gate status.** D2 ran with exactly four tenants, all four produced tokens, the
sampler captured 308 clean samples, and fairness (1.0152) passed. The two `[FAIL]`
flags it prints — `agg_tps > 1.5× D1` and `peak_mem < 22 GB` — are the same
prototype-harness limits documented for D1 (continuous-decode density does *not*
scale on this eager-mode harness; this is the central finding of
`HARNESS_LIMITATIONS.md` §3, recorded as "by design once understood"). A "dry run"
gates on *the harness running cleanly across four tenants with all tenants alive and
fair* — not on a throughput target. That gate is met and the receipt is on disk.
SHIPPED.

**Reproducibility command:** same as CP 0.2 (`density_d1d2.py` runs D1 then D2).

**Why this is not the WL05 8-tenant run.** `cipher_workloads/drivers/wl05_multitenant_x8.sh`
spawns 8 TinyLlama processes, not 4 Mistral-7B threads, on a different harness
(Phase 4.0.9) with no $/M-tok or HFU metric. It is a different artifact answering a
different question and is **not** the CP 0.3 evidence. The genuine CP 0.3 evidence
(`d2_n4_60s.json`, 4-tenant Mistral) exists, so no substitution argument is needed.

---

## CP 0.4 — Test A continuous decode sweep (N=4…128) — **NOT DONE**

**Gate intent:** a continuous-decode tenant sweep across N=4…128, producing
Story-A numbers (MFU/HFU/TPW/density per N step).

**Status:** the sweep was **never executed.** This is stated explicitly, by the
project's own contemporaneous record:

> `cipher-may13-evidence/stress2/DIAGNOSTIC_REPORT_2026_05_13.md`, line 9:
> *"The density sweep (continuous N∈{4,8,16,24,32,48,64,96,128} + burst
> N∈{8,16,32,64,96,128,192}) was **not executed**."*

The `density_d1d2.py` runner ends with `print("STOP — waiting for explicit go on
the full sweep.")` (line ~144) and the diagnostic report confirms the next session
began Phase 1 (`cipher_kmod` skeleton) instead.

**What exists vs. what is missing:**
- *Exists:* the harness function `run_step_continuous(...)` in `density_harness.py`
  (lines 346–385) — the machinery to run any single N step. The capability is built.
- *Missing:* any sweep driver that calls it across N=4…128; any per-N receipts
  beyond `d1_single_60s.json` (N=1) and `d2_n4_60s.json` (N=4); any Story-A
  aggregated output. `ls /tmp/cipher_metrics/` and `cipher_metrics_receipts/`
  contain only the D1 and D2 receipts — no `*_n8_*`, `*_n16_*`, … files exist.

**Caveat on the data point that does exist.** D2's N=4 receipt is technically the
N=4 step of this sweep. But CP 0.4 asks for a *sweep* (N=4…128, multiple steps,
Story-A numbers per step). One point is not a sweep. NOT DONE.

**How to close:** write a sweep driver that loops `run_step_continuous` over
N∈{4,8,16,24,32,48,64,96,128}, collect per-N receipts, aggregate into a Story-A
table. **But note** — `HARNESS_LIMITATIONS.md` §3 warns that continuous-decode
density on this eager-mode harness is launch-overhead-bound, not silicon-bound, so
the numbers would reflect the prototype ceiling, not CIPHER's capacity. Closing CP
0.4 honestly requires either the kmod-backed harness or an explicit "prototype
ceiling" label on the result.

---

## CP 0.5 — Test B agentic burst sweep (N=8…192) — **NOT DONE**

**Gate intent:** an agentic-burst tenant sweep across N=8…192, producing Story-B
numbers (MFU/HFU/TPW/density per N step).

**Status:** never executed. Same evidence as CP 0.4 —
`DIAGNOSTIC_REPORT_2026_05_13.md` line 9 names the burst sweep
"N∈{8,16,32,64,96,128,192}" explicitly as part of what "was **not executed**."

**What exists vs. what is missing:**
- *Exists:* the harness function `run_step_burst(...)` and worker
  `tenant_burst_worker(...)` in `density_harness.py` (lines 296–344, 388–429) —
  the bursty agentic measurement machinery is fully built.
- *Missing:* any burst-sweep driver; any burst receipts at all. Unlike CP 0.4,
  CP 0.5 has **zero** data points on disk — not even an N=8 step. The burst path
  was built but never run once.

**How to close:** write a burst-sweep driver looping `run_step_burst` over
N∈{8,16,32,64,96,128,192}; collect per-N burst receipts; aggregate into Story-B.
`HARNESS_LIMITATIONS.md` §3 notes the burst framing is the one that *does* hold on
this harness (idle-fraction density scales) — but also (§7) that the FAIRNESS SHM
caps at 64 tenants, so N=128/192 burst steps would have partial fairness
observability. Both should be addressed before the result is published.

---

## CP 0.6 — Final report (md + JSON, both stories, MFU/HFU/TPW/density per step, $/M tok power-only) — **NOT DONE**

**Gate intent:** a final Phase-0 report (markdown + JSON) presenting both Story-A
(continuous) and Story-B (burst), with MFU/HFU/TPW/density per N step and a
power-only $/M-tok figure.

**Status:** never produced. It cannot have been — it depends on CP 0.4 and CP 0.5,
both of which are NOT DONE. A final report covering "both stories per step" is
impossible when neither sweep ran.

**Verification performed:**
- `grep -rln "HFU|$/M tok|Story-A|Story-B|Story A|Story B"` across `/home/ubuntu`
  returned **no Phase-0 report** — only library source files and unrelated cache
  dirs (the literal strings "Story-A"/"Story-B" appear nowhere in any report).
- The only Phase-0 report-like documents that exist are *diagnostic* documents
  explaining why the sweeps were deferred — not result reports:
  - `cipher-may13-evidence/stress2/DIAGNOSTIC_REPORT_2026_05_13.md`
  - `cipher-may13-evidence/stress2/HARNESS_LIMITATIONS.md`
  These are honest "why we stopped" memos, not the CP 0.6 deliverable.
- The CP 0.6 *metrics* do exist in machine-readable form — every harness receipt
  (`d1_single_60s.json`, `d2_n4_60s.json`) carries `mfu_pct`, `hfu_pct`, `tpw`,
  `dollar_per_M_tok_power_only`. So the metric *plumbing* for CP 0.6 is proven
  working. What is missing is (a) the per-step data across the two sweeps and
  (b) the assembled md+JSON report.

**How to close:** complete CP 0.4 and CP 0.5, then aggregate all per-N receipts
into a markdown report + a combined JSON, with Story-A and Story-B sections, an
MFU/HFU/TPW/density column per N step, and the power-only $/M-tok figure (already
emitted per-receipt as `dollar_per_M_tok_power_only`).

---

## Summary of drift (honest accounting)

Phase 0 was a coherent, well-instrumented program: a multi-tenant density-sweep
harness (`density_harness.py`, 429 lines) with a real metrics collector emitting the
full canonical metric set, plus dedicated continuous (`run_step_continuous`) and
burst (`run_step_burst`) paths. The team built the *entire mechanism* and verified
it end-to-end on the first two steps:

- **CP 0.1 / 0.3 cleared** — TinyLlama smoke passed; the D2 dry run met its
  dry-run-specific gates (fairness, 4/4 tenants alive) and left a receipt on disk.
- **CP 0.2 partially cleared** — D1 ran and produced a full receipt, but the runner
  declared `OVERALL: FAIL`: `agg_tps` 36.39 fell outside the [60,250] calibration
  band (documented eager-mode launch ceiling). A calibrated reference with a flagged
  anomaly, not a clean calibration.
- **CP 0.4 / 0.5 / 0.6 did not** — the runner deliberately STOPped after D2
  ("waiting for explicit go on the full sweep"), and the project's own diagnostic
  report records that the explicit go never came: the sweeps were judged to be
  measuring a prototype-harness ceiling rather than CIPHER's intrinsic capacity, and
  the next session pivoted to "Phase 1 — `cipher_kmod` skeleton."

So the drift is not sloppiness — it is a deliberate, *documented* mid-phase pivot.
Phase 0 was 50% executed (2 of 4 measurement steps) and then abandoned in favour of
kernel-module work that the team believed was a prerequisite for honest sweep
numbers. The canonical Phase 0 — as a completed program with a final report — never
shipped. What shipped is a verified harness and a calibration pair (N=1, N=4), with
an explicit, honest record of why the rest was deferred.
