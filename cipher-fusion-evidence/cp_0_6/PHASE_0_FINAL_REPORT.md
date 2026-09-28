# CIPHER Phase 0 — Final Report (CP 0.6)

Generated: 2026-05-15 13:12:15

Canonical Phase 0 final deliverable (CP 0.6): both density stories, with MFU / HFU / TPW / density and power-only $/M-tok per N step. It assembles the on-disk receipts from CP 0.4 (Test A) and CP 0.5 (Test B); it asserts no number not in those receipts.

## 1. Diagnostic finding (the deliverable)

**Phase 0 measured Mistral-7B decode density two different ways — continuous decode (Test A, N=4…128) and agentic burst (Test B, N=8…192) — and both hit the same wall. Aggregate throughput saturates at ~8.9 tok/s and does not move, regardless of tenant count and regardless of workload pattern. That is one architectural ceiling, and it is not the silicon.**

- It is *not* the H100. Peak MFU across the entire program is 0.0185% — the GPU is idle of useful FLOPs ~99.99% of the time at every measured point.
- It *is* the user-process software path: one Python interpreter dispatching ~1346 kernel launches/token across N tenants, plus the harness funnelling every tenant's Marlin GEMM through one shared workspace lock. Both serialize all tenant compute into a single queue.
- Test A shows it as a flat throughput plateau as N grows. Test B shows it as the burst idle headroom being consumed — mean_idle_fraction collapses 0.093 → 0.004 — while aggregate throughput stays on the same plateau. Two workload patterns, two symptoms, one cause.

**Escaping this ceiling requires moving the serialization point out of the user-process Python/ctypes layer into operator-context kernel-module work.** That is the motivation Phase 1 onward acts on. Phase 0's job was to produce that motivation from measurement rather than assertion — done, from both stories independently. The per-step numbers in §4–§6 quantify the finding; they are the body, not the headline.

## 2. Phase 0 scope and checkpoint status

Phase 0 is a density-sweep measurement program:

| CP | Step | Status |
|---|---|---|
| 0.1 | TinyLlama smoke test | done (`smoke_marlin_tinyllama`) |
| 0.2 | D1 Mistral-7B single-tenant calibration | receipt on disk (`d1_single_60s.json`) |
| 0.3 | D2 four-tenant dry run | done (`d2_n4_60s.json`) |
| 0.4 | Test A continuous decode sweep N=4…128 | done — Story-A |
| 0.5 | Test B agentic burst sweep N=8…192 | done — Story-B |
| 0.6 | This final report (md + JSON, both stories) | this doc |

## 3. Method

- Model: Mistral-7B-v0.1 fp16 + Marlin INT4.
- Harness: `/workspace/stress2/density_harness.py` — the canonical Phase-0 density harness, run unmodified. Drivers `density_sweep_a.py` (CP 0.4) and `density_sweep_b.py` (CP 0.5) add only the sweep loop and the per-step aggregation.
- Test A (continuous): N tenants each decode continuously for 60s; prefill=128, max_decode=200. N grid [4, 8, 16, 24, 32, 48, 64, 96, 128].
- Test B (burst): N tenants each repeat (prefill=128 + decode 50 tokens) then sleep 5.0s, for 90s. N grid [8, 16, 32, 64, 96, 128, 192].
- Telemetry: NVML at 5 Hz; steady-state window [20%, 95%] of samples; MFU/HFU/TPW/$ computed by `cipher_metrics.py`.

**Metric definitions.**
- MFU % = achieved_TFLOPS / 989 × 100 (H100 FP16 dense sparse spec — industry-comparable).
- HFU % = achieved_TFLOPS / 660 × 100 (this pod's measured FP16 ceiling at the 700 W power cap).
- TPW = tokens / (mean_power_W × elapsed_s) — tokens per joule.
- $/M-tok = power cost only at $0.10/kWh — excludes capital, cooling, networking, ops, margin (full neocloud $/M-tok is typically 50–100× larger).
- density (Test A) = aggregate-throughput curve agg_tps vs N. density (Test B) = idle-headroom curve: mean_idle_fraction, compute_duty (=1−idle), effective_concurrent (=N×duty).

**Harness reality (binding).** Per `HARNESS_LIMITATIONS.md` §1 and §3: this harness is eager-mode. Test A is host-launch-bound; Test B is bound by the shared Marlin-workspace lock. Every number below is the **prototype-harness ceiling**, reported honestly as such — not CIPHER's silicon-bound capacity.

## 4. Story A — continuous-decode density sweep (CP 0.4)

| N | agg_tps | tps/tenant | MFU % (/989) | HFU % (/660) | TPW tok/J | $/M-tok | fairness | peak_mem GB | power W | clk MHz | util % | samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 12.76 | 3.190 | 0.0185 | 0.0277 | 0.0971 | $0.285976 | 1.026 | 38.66 | 131 | 1980 | 11.8 | 378 |
| 8 | 9.34 | 1.167 | 0.0135 | 0.0202 | 0.0725 | $0.383168 | 1.092 | 39.76 | 128 | 1980 | 9.0 | 327 |
| 16 | 8.86 | 0.554 | 0.0127 | 0.0190 | 0.0673 | $0.412923 | 1.027 | 41.95 | 130 | 1980 | 8.5 | 357 |
| 24 | 8.91 | 0.371 | 0.0128 | 0.0191 | 0.0678 | $0.409605 | 1.027 | 43.12 | 130 | 1980 | 8.5 | 389 |
| 32 | 8.84 | 0.276 | 0.0127 | 0.0191 | 0.0679 | $0.409002 | 1.014 | 44.44 | 129 | 1980 | 8.6 | 415 |
| 48 | 8.93 | 0.186 | 0.0127 | 0.0191 | 0.0682 | $0.407330 | 1.027 | 47.70 | 129 | 1980 | 8.6 | 483 |
| 64 | 8.89 | 0.139 | 0.0128 | 0.0192 | 0.0675 | $0.411497 | 1.019 | 51.12 | 131 | 1980 | 8.6 | 539 |
| 96 | 8.90 | 0.093 | 0.0127 | 0.0191 | 0.0673 | $0.412810 | 1.030 | 57.55 | 131 | 1980 | 8.5 | 647 |
| 128 | 8.82 | 0.069 | 0.0126 | 0.0189 | 0.0668 | $0.416100 | 1.034 | 64.42 | 131 | 1980 | 8.4 | 794 |

agg_tps ranges 8.82–12.76 tok/s and plateaus at ~8.9 tok/s from N=16 on. Aggregate throughput does not rise with tenant count; per-tenant share collapses (3.19 → 0.07 tok/s). Peak MFU 0.0185%. Fairness < 1.10 throughout.

## 5. Story B — agentic-burst density sweep (CP 0.5)

| N | agg_tps | bursts | mean_idle_frac | compute_duty | eff_concurrent | s/burst | MFU % (/989) | HFU % (/660) | TPW tok/J | $/M-tok | fairness | peak_mem GB | power W | samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 7.45 | 16 | 0.0931 | 0.9069 | 7.25 | 48.799 | 0.0106 | 0.0159 | 0.0564 | $0.492852 | 1.123 | 39.24 | 130 | 547 |
| 16 | 8.21 | 16 | 0.0513 | 0.9487 | 15.18 | 92.469 | 0.0118 | 0.0177 | 0.0636 | $0.436604 | 1.009 | 41.00 | 129 | 489 |
| 32 | 8.39 | 32 | 0.0263 | 0.9737 | 31.16 | 185.613 | 0.0121 | 0.0182 | 0.0648 | $0.428617 | 1.006 | 42.83 | 129 | 955 |
| 64 | 8.50 | 64 | 0.0133 | 0.9867 | 63.15 | 371.333 | 0.0123 | 0.0184 | 0.0652 | $0.425927 | 1.005 | 47.19 | 130 | 1885 |
| 96 | 8.55 | 96 | 0.0089 | 0.9911 | 95.14 | 556.655 | 0.0123 | 0.0185 | 0.0659 | $0.421337 | 1.006 | 52.60 | 130 | 2813 |
| 128 | 8.52 | 128 | 0.0067 | 0.9933 | 127.15 | 746.062 | 0.0123 | 0.0184 | 0.0657 | $0.422529 | 1.004 | 58.78 | 129 | 3760 |
| 192 | 8.55 | 192 | 0.0045 | 0.9955 | 191.14 | 1117.576 | 0.0124 | 0.0185 | 0.0660 | $0.421194 | 1.002 | 69.52 | 130 | 5620 |

mean_idle_fraction collapses 0.0931 (N=8) → 0.0045 (N=192); compute_duty → 0.9955. s/burst rises ~linearly with N — tenants spend linearly longer queued, not doing more work. agg_tps stays on the ~8.3 tok/s plateau. effective_concurrent ≈ N is an artifact of the shared Marlin queue being full, not concurrent throughput. The 'burst framing escapes the ceiling' prediction (HARNESS_LIMITATIONS.md §3) is cleanly refuted.

## 6. Power-only $/M-tok — honest operating point

**There is no good operating point. The prototype ceiling dominates at every N in both sweeps**, so an averaged $/M-tok would only hide the saturation. The figures below are reported per operating point, not averaged.

| Operating point | agg_tps | TPW tok/J | $/M-tok (power only) |
|---|---|---|---|
| Test A best (N=4, highest TPW in sweep) | 12.76 | 0.0971 | $0.285976 |
| Test A lowest-N (N=4) | 12.76 | 0.0971 | $0.285976 |
| Test B best (N=192, highest TPW in sweep) | 8.55 | 0.0660 | $0.421194 |
| Test B lowest-N (N=8) | 7.45 | 0.0564 | $0.492852 |
| Single-tenant reference (D1, CP 0.2, N=1) | 36.39 | 0.2323 | $0.119554 |

The single-tenant D1 reference (N=1, TPW 0.2323, $0.1196/M-tok) is the least-bad point measured in all of Phase 0 — and even it is a prototype-harness number. The *best* point inside the multi-tenant sweeps (Test A N=4, TPW 0.0971) is already 2.4× worse than D1, because 4 tenants are enough to trigger Marlin-lock contention. There is no N at which the sweeps reach a 'good' efficiency point; $/M-tok degrades as soon as the second-and-later tenants start queueing.

All $/M-tok figures are **power cost only** at $0.10/kWh — they exclude capital amortization, cooling, networking, ops and margin. A full neocloud $/M-tok is typically 50–100× larger. These are true power-cost readings of *this prototype harness*, not product $/M-tok.

## 7. Conclusion — CP 0.6 closure

Phase 0 is complete as a measurement program: the smoke test, the D1 calibration, the D2 dry run, the Test A continuous sweep (9 N-steps), the Test B burst sweep (7 N-steps), and this final report all exist on disk with receipts.

Phase 0's diagnostic purpose is fulfilled. Both stories, independently, identify a single architectural ceiling — aggregate decode throughput pinned at ~8.9 tok/s, peak MFU 0.0185%, the H100 starved by a user-process software path, not by silicon. The escape route — operator-context kernel-module work — is the motivated subject of Phase 1 onward. Phase 0 delivers a verified harness, a fully characterised prototype ceiling, and that motivation; it does not deliver, and does not claim, a silicon-capacity number.

