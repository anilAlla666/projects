# WI-2 SERVING-GOODPUT / SLO-ATTAINMENT — RULE-4 PRE-REGISTRATION (before measurement)

**Goodput definition used: SERVING goodput** (DistServe) = request rate meeting BOTH TTFT and TPOT SLOs at a stated
attainment, per GPU. A correct-but-slow request scores ZERO. This is the definition under which CIPHER's cross-tenant
MUX (not the SDC detector) is the relevant lever.

## SLO definitions (sourced)
DistServe / production chatbot SLOs: TTFT (time-to-first-token) and TPOT (time-per-output-token). DistServe uses
e.g. TTFT 0.25 s and TPOT 0.1 s for chatbots at strict attainment (90%). For a 7B model on one eager H100 (no
cudagraph, no disaggregation) those strict numbers may be unmet even at low load, so I will (a) report the raw TTFT/TPOT
distributions, (b) compute SLO-attainment against BOTH the strict DistServe SLOs AND a relaxed 7B-eager SLO (TTFT 1.0 s,
TPOT 0.05 s ≈ 20 tok/s/req), stating both. Goodput = (fraction of requests meeting both SLOs) × offered rate, per GPU.

## Plan
Single-tenant baseline: real vLLM (eager) under Poisson arrivals at rate λ; stream each request; record per-request
TTFT + TPOT; sweep λ; SLO-goodput(λ) = attainment(λ)·λ. Find the goodput knee (where attainment falls as λ saturates).
MUX comparison: CIPHER cross-tenant multiplexing WITH vs WITHOUT.

## PRE-REGISTERED EXPECTATIONS
1. **Single-tenant baseline:** attainment ≈ 1.0 at low λ, falls as λ → saturation (queueing inflates TTFT first, then
   TPOT); SLO-goodput rises then plateaus/drops. Expect the knee well below raw max throughput.
2. **MUX:** the CIPHER mux should raise SLO-meeting req/s PER GPU by serving multiple tenants' interleaved load on one
   GPU that would otherwise need separate GPUs (its validated value is a tok/W / density lever, 3.06–3.30×).
3. **WALL RISK (pre-registered):** the production mux actuator lives in the CIPHER engine/substrate (`libcipher_rt.so` +
   its scheduler). Engaging it requires loading the production substrate into the serving process, which (a) crosses the
   substrate line for this scratch harness and (b) per the R.C/Path-1 findings needs a vLLM-cooperating plugin (external
   in-graph hosting is walled). A scratch eager-vLLM harness can only do NAIVE co-residence (independent instances
   sharing the GPU), which is NOT the intelligent mux. **If so, WI-2 WALLS on the mux SLO measurement:** record
   WALL-WITH-MECHANISM (exactly what the mux needs that isn't wired), deliver the measurable single-tenant SLO baseline
   (+ naive co-residence if informative), and cite the validated 3.06–3.30× explicitly as a **tok/W / density** number,
   **NOT** an SLO-goodput number. Do NOT fabricate an SLO-attainment uplift for the mux.

## REAL vs MODELED / WALLED
REAL: single-tenant TTFT/TPOT distributions + SLO-goodput(λ); naive co-residence if run. WALLED (if it walls): the
CIPHER mux SLO uplift (production engine needed). The 3.06–3.30× is a prior tok/W result, NOT measured here as SLO.
