# CIPHER RESULTS SNAPSHOT — 2026-06-10

**For the Nebius design-partner conversation. Read this header first.**

Everything in this document is one of exactly two kinds, and the table never blurs them:

- **(A) MEASURED THIS RUN (2026-06-10)** — fresh, dispatch-boundary-legal, on real workloads on this H100.
  No production CIPHER substrate (`libcipher_rt.so` / `cipher_kv_bridge.so`) was loaded into vLLM, no mux engaged,
  no graph mutation. R-pillar detectors ran as scratch LD_PRELOAD cuBLAS-interception shims rebuilt in `snapshot/`
  (ldd: libc only; `/dev/cipher` never opened — proof in `NO_SUBSTRATE_PROOF.txt`).
- **(B) QUOTED FROM PRIOR VALIDATION** — cited verbatim to the source report + JSON artifact, **not re-measured**.
  The cross-tenant mux tok/W and the SLO/MFU-with-CIPHER numbers live here **deliberately**: the mux is compiled
  into the substrate and needs a vLLM-cooperating host — the walled, partner-gated measurement (see the mux row).
  Faking a fresh run of those would have been easy and is exactly what this snapshot refuses to do.

Anchor `libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — **unchanged, entry == exit**
(full 184-file fork-1 manifest diff clean: `fork1_md5_entry.txt` == `fork1_md5_exit.txt`).

---

## Definitions (one line each)

- **MFU** = achieved model-FLOPs/s ÷ H100 SXM peak dense BF16/FP16 **989.5 TFLOP/s** (NVIDIA H100 datasheet, no sparsity).
  Decode MFU is *expected* to be low — decode is memory-bound; that is the honest physics, not a defect.
- **MBU** = achieved HBM bytes/s ÷ H100 SXM HBM3 peak **3.35 TB/s** (same datasheet). The meaningful decode-side number.
- **TPW** = tokens ÷ NVML energy (J): `power.draw` sampled @50 ms, trapezoid-integrated over the measured generate
  window (the validated method, `rc_ab/ab_driver.py`). The NVML hardware energy counter
  (`nvmlDeviceGetTotalEnergyConsumption`) over the same window is reported as a cross-check; on short (3–4 s) windows
  the trapezoid reads 1.5–8.7 % below the counter (sampling misses power ramps) — both values are in the JSONs.

**Analytic models (so Nebius can recompute every cell):**

- Model: `mistralai/Mistral-7B-v0.1`, fp16, vLLM 0.20.2 **eager** (`enforce_eager=True`). Eager-host caveat carried
  throughout: cudagraph is **2.05×** eager on this workload (B row "eager-vs-cudagraph").
- `N_params = 7,241,732,096` (exact: h=4096, L=32, kv_heads=8, inter=14336, vocab=32000).
- Decode FLOPs/token = `2·N_params` = 14.483 GFLOP (linear + lm_head matmuls; attention FLOPs excluded at short
  context ⇒ MFU is a slight lower bound).
- Decode bytes/step = `N_params·2 B` = 14.483 GB (fp16 weight stream; KV-cache traffic excluded ⇒ **MBU is a lower
  bound**, labelled MBU-LB). MBU-LB = (tok/s ÷ B) · 14.483 GB ÷ 3.35 TB/s.
- Prefill FLOPs = `2·N_params·T_prompt` (quadratic attention term excluded, <4 % at 2048).
- LoRA train FLOPs/token = `4·N_params` (fwd 2N + input-grad 2N; base frozen ⇒ no weight-grad GEMMs; r=16 adapter <1 %).
- Clock: locked 1980 MHz (driver delivers 1830 ceiling); lock re-applied in-process before every measured window
  (see Session integrity §3); **achieved clock + SwPowerCap recorded per run** in each JSON.

---

## RESULTS TABLE

Category column: **A = measured this run (2026-06-10, JSON in `snapshot/`)** | **B = quoted from prior (source file:line)**.

| # | Metric | Value | Cat. | Evidence |
|---|--------|-------|------|----------|
| 1 | Decode **B=1** (latency regime), eager | **80.6 tok/s · MFU 0.118 % · MBU-LB 34.9 % · TPW 0.322 tok/J** (counter-XCheck 0.299) · 255 W avg · clock 1830 pinned | **A** | `a1_decode_B1.json` |
| 2 | Decode **B=64** (throughput regime), eager | **4698.2 tok/s · MFU 6.88 % · MBU-LB 31.7 % · TPW 14.46 tok/J** (13.29) · 332 W · clock 1830 (brief SwPowerCap dips to 1515) | **A** | `a1_decode_B64.json` |
| 3 | **Prefill** 8×2048 ×10 reps (compute-bound) | **41,427 prompt-tok/s · MFU 60.6 % · TPW 65.7 prompt-tok/J** (60.0) · 636 W · SwPowerCap active, clock median 1440 | **A** | `a1_prefill_B8.json` |
| 4 | **LoRA train step** (r=16, B=1, seq=512, fp16, wi1 recipe) | **0.1399 s/step (3652 tok/s) · MFU 10.7 % (4N model) · TPW 11.32 tok/J** (counter 11.15) · STABLE (0 NaN, loss 0.122→0.014) · clock 1980 pinned | **A** | `a1_train_B1.json` |
| 5 | **R.A detector marginal overhead** (B=8 eager, vs same-session clean 629.2 tok/s) | **+4.28 % @N=45** (602.3 tok/s) · **+12.62 % @N=8** (549.8 tok/s) — prior: +3.1 % / +12.0 % | **A** | `a2_summary.json`, `a2_ra/*` |
| 6 | **R.A clean baseline (re-measured)** | **629.2 tok/s — identical to the prior 629.2** (after plugin decontamination, §Session integrity) | **A** | `a2_ra/clean_bench4.txt` |
| 7 | **R.A 0-FP** | **0 detections / 10,240 GEMM checks** across 4 clean runs (640 + 3200 + 4224 + 2176); `max_clean_residual` exactly **0** everywhere (T=0 bit-identical recompute) | **A** | `a2_ra/n{45,8}_counts.json`, `a2_rb/clean_N8_counts.json`, `a2_rc/clean_sdc_counts_p0.json` |
| 8 | **R.A detection latency** (persistent bit-14 SDC, onset step 20) | **4 steps @N=8** (det @24) · **25 steps @N=45** (det @45) — equals prior (4 / 25); both ≤N | **A** | `a2_rb/persist_N{8,45}_events.jsonl` |
| 9 | **R.B quarantine** (batch mode, B=1) | Persistent SDC **detected before serve** (det wall-time < generate-window end, verified from timestamps) · **0 false quarantine** (0/4224) · transient **between checks: MISSED** (0 det on 1 flip — the honest ~1/N wall) · transient on-check: caught (positive control) | **A** | `a2_summary.json`, `a2_rb/*` |
| 10 | **R.C per-GPU verdict** | Inject → **DEGRADED** (onset det step 64, confirmed 72; matches prior "64det/72conf" — note the configured onset 64 sits on the N=8 check grid, so step-64 detection is by construction; the measured content is detection at the first post-onset check, +N confirm, 8 clean pre-onset checks, persistence 9); descriptor `SDC-pid==NVML-pid` (`VLLM::EngineCore`), ECC/XID deltas 0 (software flip ⇒ NVML correctly counter-clean) · Clean → **HEALTHY**, 0/2176 | **A** | `a2_rc/{inject,clean}_verdict.json` |
| 11 | **Cross-tenant mux tok/W** — Mistral-7B, N=4 tenants | **3.06×** substrate-attributable (batched 0.886 vs naive-4-concurrent 0.290 tok/W; KL-gated) | **B** | `cp_5_6/TPW_RETEST_2026_05_19.md:49` + `cp_5_6/phase_b/session1/tpw_retest_result.json` |
| 12 | **Cross-tenant mux tok/W** — TinyLlama-1.1B, N=8 tenants | **3.30×** substrate-attributable (3.535 vs 1.073 tok/W) | **B** | `cp_5_6/TPW_RETEST_2026_05_19.md:52` + same JSON |
| 13 | **DVFS clock governance tok/W** | **+57.28 % ± 0.32 %** (95 % CI +56.65…+57.90, n=5 matched pairs, z≈180). **Scope caveat (from source + envelope):** B=1 TinyLlama-class memory-bound decode; on Mistral-7B the envelope shows **−2 % to −15 %** — not a universal lift | **B** | `/home/ubuntu/PHASE_4_T4_3_2_REPORT.md:17,128` + `cipher-phase4-evidence/t4_3_2_npairs/summary.json`; envelope: `phase_audit/non_canonical/audit.md:188–197` |
| 14 | **W6 single-instance baseline, B=1** (MLPerf-v5.1-aligned; vLLM 0.21.0 **cudagraph**, BF16, 1024-tok agentic prompt) | **163.3 tok/s · 0.249 % decode MFU · 59.57 % prefill MFU** (vanilla, cache-OFF) | **B** | `WEEK_6_OPTION_1_REDO.md:68` + `week6/bench_v2/results/20260523_mistral7b_vanilla_…_nocache/result.json` |
| 15 | **W6 single-instance baseline, B=64** | **4156.1 tok/s · 6.292 % decode MFU · 62.06 % prefill MFU** (vanilla; the oft-quoted 4155.2 is the CIPHER arm, −0.02 %, n.s.) | **B** | `WEEK_6_OPTION_1_REDO.md:72,106,176` + same results dir |
| 16 | **Eager vs cudagraph host tax** | **629.2 vs 1289.0 tok/s ⇒ +51.2 % tax (cudagraph 2.05×)** | **B** | `R_A_REAL_VLLM_2026-06-08.md:70–77` + `ra_realvllm/tokps_result.json` |
| 17 | **Mux SLO-goodput / MFU-with-CIPHER under load** | **NOT MEASURED — partner-gated.** The mux lives in the compiled substrate and needs a vLLM-cooperating host (wall); single-tenant SLO baseline ≥0.98 attainment exists, but mux SLO was never measured and is **deliberately not faked here** | **B (gap, disclosed)** | `wi4_memo/RPILLAR_GOODPUT_CLOSEOUT_MEMO_2026-06-10.md:22`; mechanism: `wi2_servinggoodput/WI2_SERVING_GOODPUT_2026-06-09.md:60–72,107–109` |

**Reason-not-re-run (rows 11/12/17, one line):** mux engagement requires a vLLM-cooperating host — the walled,
partner-gated measurement per the R-pillar close-out memo; re-deriving it would load the substrate into vLLM and
cross the dispatch-boundary line this snapshot exists to hold. (Note: the Nebius-titled memo
`v1_phase_b/DEVANG_NEBIUS_UPDATE.md` does not itself carry the mux sentence; the close-out memo above is the
citable source.)

---

## Cross-checks the partner can do (fresh A vs quoted B)

- **B=1 decode:** fresh eager 80.6 tok/s × 2.05 (row 16 cudagraph factor — measured at B=8/64-tok decode, so this is a heuristic extrapolation across batch and prompt length) ≈ **165 ≈ 163.3** (row 14, cudagraph config). Consistent.
- **Prefill MFU:** fresh 60.6 % (2048-tok prompts) vs W6 59.57 % (1024-tok) — same compute-bound regime.
- **B=64 decode:** fresh eager 4698 vs W6 4156 — fresh uses 32-tok prompts (short context ⇒ smaller KV/attention work
  per step) vs W6's 1024-tok agentic prompts; direction and magnitude expected.
- **R.A baseline:** fresh clean 629.2 == prior 629.2 (row 6/16, same config two days apart).
- **R.B drivers:** four of five B=1 runs at 66.3–67.7 tok/s vs rc_ab's 67.55 on 2026-06-09 — consistent; one outlier
  (persist_N45 at 75.5, a with-detector run outrunning clean) is B=1 eager launch-bound run-to-run jitter, flagged
  rather than folded into the range. No table number depends on it.
- **W6 peak nuance:** W6 used 989.0 TFLOP/s as peak; this snapshot uses 989.5 — a 0.05 % definitional difference, ignorable.

## Pre-registration vs measured (PREREG.md written before any measurement)

| Pre-reg band | Measured | Verdict |
|---|---|---|
| Decode B=1: 60–130 tok/s, MFU 0.09–0.19 % | 80.6, 0.118 % | in band |
| Decode B=64: 2000–4500 tok/s, MFU 3.0–6.6 %, MBU 14–31 %, TPW 5–14 | 4698, 6.88 %, 31.7 %, 14.46 | **slightly above band ×4** (all four metrics) — favorable-side surprise; mechanism: short-context config + the band was set before decontamination was understood |
| Prefill: MFU 50–65 %, 25–45 k tok/s, 30–80 tok/J | 60.6 %, 41.4 k, 65.7 | in band |
| Train MFU 15–45 % | **10.7 % (4N model)** | the band was pre-registered against a generic **~6N** fwd+bwd model; under that registered model the same step is **16.1 % — in band**. The report deliberately uses the more defensible frozen-base **4N** model (no base weight-grad GEMMs under LoRA), which yields 10.7 %. The "miss" is a model change, disclosed here, not a measurement shortfall |
| R.A: N=45 2–5 %, N=8 8–14 % | 4.28 %, 12.62 % | in band; prior values 3.1 %/12.0 %. Each config is a single best-of-2 measurement; the fresh-vs-prior deltas (±1.2 pp) are consistent with run-to-run scatter but this session did not run repeats to establish a scatter estimate |
| Task brief said "~6 % @N=8" | 12.62 % | **the ~6 % figure matches no prior source** (sources: 12.0 % @B=8, 12.6 % @B=1); flagged and corrected, not absorbed |
| R.A latency ≤N, R.B 0-false + between-miss, R.C DEGRADED/HEALTHY | all exactly reproduced | in band |

## Session integrity events (disclosed, not buried)

1. **Substrate auto-load caught and neutralized (material).** First-pass vLLM runs were contaminated: the
   `cipher_vllm_kv` / `cipher_vllm_kvdedup` plugins auto-load via `vllm.general_plugins` entry points
   (`easy-install.pth` → `/home/ubuntu/cipher_vllm_plugin`) and pull the cipher_v2 substrate into the engine —
   detected mid-session via `[cipher_v2]` stderr lines and the scratch shim never seeing a GEMM. Every vLLM
   measurement was re-taken with `VLLM_PLUGINS=""` (the same guard the validated `rc_pergpu/rc_monitor.py` already
   used). Proof: 3545 cipher_v2 lines in a pre-guard run (positive control) vs **0 in all 14 measured runs**
   (`NO_SUBSTRATE_PROOF.txt`). This also resolved an apparent ~1.9× eager-decode regression: decontaminated clean
   returned **629.2 tok/s — the prior value to the decimal**. The torch-only train run was never contaminated.
2. **DeepGEMM init crash** (deep_gemm not installed; fp8 warmup path) → `VLLM_USE_DEEP_GEMM=0`, identical to the
   validated prior harness (`rc_pergpu/rc_monitor.py:159`).
3. **Clock-lock release during vLLM engine init** (driver 580.105.08): `-lgc 1980` is observed to drop during init;
   fixed by re-applying the lock in-process immediately before each measured window; achieved clock is recorded
   per run (decode/train pinned 1830/1980; prefill SwPowerCap-limited ~1440 median — reported, not hidden).
   Contaminated early B=1 runs at 1005 MHz were superseded by the re-measurements.
4. **TPW method cross-check:** trapezoid (validated, primary) vs hardware energy counter gap −1.5 % (train, long
   window) to −8.7 % (3–4 s windows). Both values are in every JSON; TPW rows quote trapezoid with counter in parentheses.
5. **Superseded artifacts:** the contaminated first-pass JSONs/stderr were overwritten by the re-measurements
   (values preserved in the session transcript: B=1 42.4/42.8 tok/s @~1005 MHz, B=64 2554.7, prefill 40,744,
   clean benches 332.9–341.2). Two pre-guard stderr files are retained as positive controls
   (`a2_ra/clean_bench3.err`, `a2_ra/clean_bench2.err`).
6. **Environment notes:** a pre-existing read-only telemetry daemon (`cipher-exporter.py`, running since May 18,
   parses clocks, sets nothing) and the `cipher_kmod` (MIG-watch params only) were present throughout — neither
   touches GPU clocks nor loads into vLLM; the in-process measurements are insulated by (1)/(4) of the proof file.
   A `.git` metadata refresh (index + empty blob, no content objects, nothing committed) occurred under
   `cipher-fusion-evidence/.git` at 06:02:58 during the verification panel — content-neutral; no tracked file
   outside `snapshot/` changed (md5 manifest diff-clean).

## Discipline checklist

- Anchor md5 `2edba0d2…` entry == exit; full fork-1 manifest (184 files) diff-clean; **no fork-1 dir modified**
  (all scratch work in `snapshot/`, shim counts redirected away from fork-1 paths).
- Shims: `ldd` = libc only — **zero cipher entries in link deps** (the binaries do contain compiled-in literal
  strings: `/dev/cipher` in ab_shim's env-gated, never-enabled heartbeat, plus default counts paths — scoped
  precisely in `NO_SUBSTRATE_PROOF.txt`). Freshly compiled in `snapshot/` from copied sources; resulting binaries
  are **byte-identical (md5) to the validated fork-1 builds** (deterministic gcc build) — provenance-equivalent to
  the validated detectors. `det_shim` has no `/dev/cipher` code at all; `ab_shim` heartbeat gated off
  (`RV_HEARTBEAT` never set); R.C monitor cohort query **disabled** in the scratch copy ⇒ no process opened
  `/dev/cipher` (R.C descriptor therefore exercises SDC-pid==NVML-pid only; the full SDC==NVML==W6 triple is a
  citation to `R_C_PERGPU_FLEET_2026-06-09.md`, not a fresh claim).
- Counts-file mechanism: det_shim's counts-path env (`RV_OUT`) collides with the driver's token-count env and
  defaults to a fork-1 path; the a2_ra runs executed with cwd=`a2_ra/` and `RV_OUT=64`, so counts landed in
  `a2_ra/64` (renamed `n{45,8}_counts.json`). Fork-1's own `ra_realvllm/{48,64}` files carry identical counter
  values because the workload is deterministic — they are the prior session's artifacts, not cross-writes (the
  md5 manifest proves no fork-1 byte changed).
- Single-GPU exclusivity verified before runs; 0 leftover compute processes after; clock relocked 1980 per window;
  `-rgc` at session exit; children reaped.
- No retracted number appears anywhere in this snapshot (checked against: 3.617×, 14×, universal 85 % MFU,
  2.96× tok/W stress2, 0.127 % MFU, 87→83 tok/s, 7.43×).

## Verification panel (mandatory adversarial pass — 4 read-only dimensions, full record in `verification_panel.json`)

**Result: 0 material findings; 12 minor/note findings — every one applied or disclosed above.**

| Dim | Checks passed | Findings → action |
|---|---|---|
| 1 NUMBERS | 27 (every cat-A cell recomputed from its JSON: MFU/MBU/TPW/marginals/latencies/0-FP sum; every cat-B number opened verbatim at file:line; retracted-number scan clean) | ×4-not-×3 pre-reg surprise + missing MBU band → **fixed**; train 6N-vs-4N band model mismatch → **disclosed in pre-reg table**; train counter TPW missing → **added (11.15)**; R.C proof audited monitor stderr not driver stderr → **proof regenerated, 16 files, drivers 0**; 75.5 tok/s outlier folded into a range → **flagged explicitly**; 2.05× factor extrapolated across batch → **caveat added** |
| 2 CATEGORY | 13 (no B-as-fresh anywhere; ldd re-run; 0 cipher_v2 re-verified on spot-checks; /dev/cipher gating re-read in sources; mux not engaged; contaminated first-pass numbers absent from table) | "zero cipher strings" overbroad → **rescoped to link deps, string constants disclosed**; R.C driver stderr coverage → **fixed in proof**; train band model → **disclosed** |
| 3 HONESTY | 21 (decode-MFU-low framed as physics; eager caveat present; mux row explicit not-measured/not-faked; DVFS scope caveat verified against the envelope audit; MBU lower-bound labelled; contamination + TPW-gap disclosed; "629.2 identical" and "pre-serve" claims verified true) | R.C "equals prior" partly by construction (onset on check grid) → **wording fixed in row 10**; "~1 pp scatter" asserted not established → **reworded**; superseded artifacts not retained → **disclosed (§5)** |
| 4 INTEGRITY | 19 (md5 manifest diff re-run clean; anchor re-hashed = `2edba0d2…`; 5 random fork-1 files re-hashed fresh = match; no fork-1 mtime after session start; 0 compute procs; scratch confined to snapshot/) | `.git` metadata write during panel → **disclosed (§6)**; "rebuilt" unsubstantiated → **strengthened: byte-identical md5 to validated fork-1 builds**; RV_OUT fork-1-default footgun → **mechanism documented**; pre-existing cipher-exporter daemon → **disclosed (§6)** |

Exit state (after panel): fork-1 manifest re-diffed clean, anchor md5 `2edba0d2136f8ede4713d90a8f7cd55f` re-verified,
0 leftover compute processes, GPU clocks reset (`-rgc`) at session exit.
