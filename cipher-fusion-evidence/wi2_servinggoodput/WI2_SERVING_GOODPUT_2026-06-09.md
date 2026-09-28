# WI-2 — SERVING-GOODPUT / SLO-ATTAINMENT (the mux lever)

**Date:** 2026-06-09. **Goodput definition: SERVING goodput** (DistServe) = request rate meeting BOTH TTFT and TPOT
SLOs, per GPU. A correct-but-slow request scores zero. The relevant CIPHER lever is the cross-tenant **MUX** (NOT the
SDC detector). **Stack:** real vLLM 0.20.2 OpenAI server, Mistral-7B-v0.1 fp16, `enforce_eager`, one H100, streaming;
Poisson arrivals; per-request TTFT (time-to-first-token) + TPOT (mean inter-token) measured client-side.
**Discipline:** anchor `2edba0d2…` md5 **entry==exit**; scratch code in `wi2_servinggoodput/` only; substrate NOT loaded
(`CUDA_INJECTION64_PATH` unset, `VLLM_PLUGINS=""`) ⇒ this is the honest **WITHOUT-mux** baseline; server reaped; clock
1980 → `-rgc` at loop exit. **Pre-reg:** `wi2_servinggoodput/PREREG.md`. **Evidence:** `wi2_loadgen.py, wi2_lam*.json,
wi2_sustained.json, vllm_serve.log`.

---

## BOTTOM LINE
**WITHOUT-mux baseline is measured; the MUX comparison WALLS (production substrate).** On one eager H100, single-tenant
Mistral-7B serving is **latency-comfortable, not latency-bound** at chatbot SLOs: TTFT p90 ≤ 76 ms and TPOT ≈ 14 ms/token
(~72 tok/s/req) stay far inside both the relaxed (TTFT 1.0 s / TPOT 0.05 s) and strict (DistServe TTFT 0.25 s / TPOT
0.1 s) SLOs across the **entire tested offered range (λ up to 64): strict-SLO attainment never fell below 0.98** (0.988 at
the two highest loads λ=32/64; 0.995 sustained).
**I did NOT reach the SLO knee** — latency was never the binding constraint in the tested range — so **no sustained
capacity ceiling is established.** *(CORRECTION 2026-06-10, caught by the WI-4 memo panel: this report's prose floor
"never fell below 0.988" was inconsistent with its own table — strict attainment is 0.98 at λ=1–16 (49/50 per run),
0.988 at λ=32/64 (79/80), 0.995 sustained (199/200). Prose corrected to the true floor ≥0.98 throughout; the table
and JSONs were always correct; no conclusion changes. The VERIFICATION PANEL section below retains its original
wording as historical record.)* The achieved-rate figures (≈22–26 req/s) are **transient**: each run is an arrival
ramp + a fixed ~2.4 s drain tail, so achieved=n/wall is run-length-dependent (λ=64/n=80 → 22.0 < λ=40/n=200 → 25.6 —
a true ceiling cannot fall as offered load rises). The honest measured result is therefore the **latency
characterization** (attainment ≥0.98 up to ≈26 req/s achieved in-burst), NOT a goodput ceiling number; finding the
knee needs a sustained steady-state soak with rising λ (not run — bounded scope, see Bounds). The **cross-tenant MUX
cannot be measured as SLO-goodput here**: it lives
in the compiled substrate (`libcipher_rt.so` / `cipher_kv_bridge.so`), not a scratch-engageable module; engaging it
requires loading the production substrate into the serving process (crosses the substrate line) and, per the R.C/Path-1
findings, a vLLM-cooperating plugin (external in-graph hosting is walled). **WALL-WITH-MECHANISM** recorded; the
validated **3.06–3.30×** is cited as what it is — a **tok/W / density** number — **NOT** an SLO-goodput uplift.

## WITHOUT-MUX BASELINE — single-tenant SLO-goodput (`wi2_lam*.json`, `wi2_sustained.json`)
| offered λ (req/s) | achieved req/s | TTFT p50/p90 (s) | TPOT p50/p90 (s) | strict-SLO attainment | strict SLO-goodput (req/s) |
|---|---|---|---|---|---|
| 1  | 0.85  | 0.036/0.044 | 0.0123/0.0124 | 0.98 | 0.83 |
| 4  | 3.23  | 0.039/0.044 | 0.0124/0.0125 | 0.98 | 3.17 |
| 8  | 5.83  | 0.039/0.046 | 0.0125/0.0127 | 0.98 | 5.72 |
| 16 | 9.51  | 0.043/0.051 | 0.0126/0.0127 | 0.98 | 9.32 |
| 32 | 16.83 | 0.048/0.059 | 0.0133/0.0136 | 0.988 | 16.62 |
| 64 | 21.99 | 0.055/0.073 | 0.0141/0.0155 | 0.988 | 21.72 |
| **40 (sustained, n=200)** | **25.59** | 0.052/0.076 | 0.0136/0.0138 | **0.995** | **25.46** |

- **Latency is not the binding constraint in the tested range:** vLLM continuous batching admits arrivals fast (TTFT
  p90 ≤ 76 ms ≪ 250 ms strict) and batched decode holds TPOT ≈ 14 ms ≪ 100 ms, so strict-SLO attainment stays ≥0.98
  through λ=64 (0.988 at λ=32/64). **No SLO knee was reached and no sustained capacity ceiling is established** — the achieved-rate numbers
  are transient (drain-tail-diluted, run-length-dependent: λ=64/n=80 gave 22.0 < λ=40/n=200 gave 25.6, which a real
  ceiling cannot do). The SLO knee was never reached (attainment ≥0.98 throughout). Finding the knee requires a
  steady-state soak with rising λ until attainment drops — not run here.
- The ~0.5–2.0% strict-SLO misses are the slowest-TTFT tail (a few requests above 250 ms under burst), not TPOT.
- **Implication for the mux:** since single-tenant latency is not the binding constraint at these SLOs, the mux's value
  is NOT
  reducing single-tenant latency — it is **DENSITY** (serving more tenants' SLO-load per GPU), which is exactly the
  tok/W lever (3.06–3.30×). That density is an SLO-goodput-per-GPU multiplier *in principle*, but measuring it requires
  the production mux (below).

## MUX — WALL-WITH-MECHANISM (what is missing, precisely)
The CIPHER cross-tenant mux (intelligent interleaving/coalescing of multiple tenants' requests onto one GPU's
execution) is implemented in the **compiled substrate** — `grep` finds no scratch-engageable Python mux; the logic is in
`libcipher_rt.so` and `cipher_kv_bridge.so`. To measure its SLO-goodput I would have to:
1. **Load the production substrate into the serving process** — forbidden here (substrate line; would also contaminate
   the WITHOUT baseline), and
2. **Host it inside vLLM's execution** — per R.C Path-1, external in-graph injection into vLLM's (cudagraph) execution
   is rejected by vLLM; the mux needs a vLLM-cooperating plugin/scheduler hook that is not wired in this scratch harness.
A scratch eager-vLLM harness can therefore only do **naive co-residence** (independent vLLM instances sharing the GPU),
which is *not* the intelligent mux and would only show contention, not the mux's coalescing benefit. So the **WITH-mux
SLO-goodput is NOT measured.** The validated **3.06× (Mistral N=4) / 3.30× (TinyLlama N=8)** is a prior **tok/W /
density** result and is **NOT** reported as an SLO-goodput number (the WI-2 definition). No SLO-attainment uplift is
fabricated for the mux.

## NAMED BOUNDS
- **Mux SLO-goodput WALLED** (production substrate + vLLM-plugin needed); only WITHOUT-mux baseline measured.
- **Eager-only:** no cudagraph (would lower TTFT further — the baseline is conservative on latency but already inside
  SLO); one GPU; one model.
- **No capacity ceiling established (transient runs):** all runs are arrival-ramp + ~2.4 s drain tail, so achieved=n/wall
  is run-length-dependent and the ≈22–26 req/s figures do NOT measure sustained capacity. The SLO knee was never reached
  (attainment ≥0.98 throughout). A steady-state soak with rising λ until attainment drops is the missing measurement
  (bounded scope, not run).
- **3.06–3.30× is tok/W/density, not SLO-goodput** — explicitly not conflated. (3.06× Mistral N=4 / 3.30× TinyLlama N=8.)

## FORK-1 INTEGRITY
Anchor `libcipher_rt.so`: entry `2edba0d2…` == exit `2edba0d2…` ✅. New work only in `wi2_servinggoodput/`. Substrate not
loaded into the server. Server + EngineCore worker reaped (GPU 81 GB free at exit).

## VERIFICATION PANEL (4-dimension adversarial pass, read-only)
**NUMBERS — clean** (every SLO-table cell matches wi2_lam*.json / wi2_sustained.json; attainment integer-count
consistent; goodput=meeting/wall confirmed in code; 3.06/3.30× kept in tok/W lane; no retracted number). **HONESTY —
clean** (SERVING goodput labeled, not conflated; mux WALL precise; no fabricated WITH-mux number; bounds stated).
**INTEGRITY — clean** (anchor 2edba0d2 entry==exit; no fork-1 dir touched; substrate not in the serve log; server +
EngineCore reaped, GPU free). **LOGIC — 1 material + minors, FIXED:**
- **[MATERIAL → FIXED] No capacity ceiling was actually measured.** The "≈25.5 req/s/GPU ceiling" was invalid: all runs
  are transients (drain-tail dilutes short runs — λ=64/n=80→22.0 < λ=40/n=200→25.6, impossible for a real ceiling), and
  strict-SLO attainment never fell below 0.988, so the SLO knee was never reached. **Fix:** retracted the ceiling claim;
  restated as "latency is not the binding constraint up to λ=64 (attainment ≥0.988); no knee reached; achieved-rate is
  transient and does not establish sustained capacity; a steady-state soak is the missing measurement (not run)."
- **[minor → FIXED]** strict-SLO miss range 0.5–1.2% → 0.5–2.0% (matches the attainment column).
- **[minor → FIXED]** 3.06–3.30× now per-model attributed in the bounds (3.06× Mistral N=4 / 3.30× TinyLlama N=8).
- **[nit] clock** still reads locked 1980 at panel time — reset (`-rgc`) occurs at LOOP exit (after WI-4), as stated.

## HONEST BOTTOM LINE
Under the SERVING-goodput (SLO) definition: single-tenant Mistral-7B on one eager H100 keeps TTFT/TPOT deep inside
chatbot SLOs (strict-SLO attainment ≥0.98; 0.988 at λ=32/64, 0.995 sustained) up to at least ≈26 req/s achieved in-burst — **latency is not the binding
constraint, and the SLO knee / sustained capacity ceiling was NOT reached in the tested range** (transient runs; a
steady-state soak is the missing measurement). CIPHER's mux is a **density** lever (tok/W 3.06× Mistral / 3.30×
TinyLlama) whose SLO-goodput uplift **could not be measured** here — the mux is compiled into the substrate and needs a
vLLM-cooperating host (WALL-WITH-MECHANISM) — and the 3.06–3.30× is kept strictly in its density/tok-W lane, never
relabeled as SLO-goodput.
