---
name: cipher-cp24-closed
description: "CP 2.4 atomic STEP complete — Marlin+DVFS+n-gram-spec composed at 3.617x tok/W on Mistral-7B; report landed, awaiting adjudication. Llama model-draft 1.75x NOT MET (deferred Phase 5)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 12f5b2f5-2ee5-4989-a2c4-7a1258935d0d
---

CP 2.4 (the decisive Phase-2 CP — perf-actuator payload on the v2 dispatch)
**atomic STEP COMPLETE + CLOSED + adjudicated 2026-05-16** (13/23 → 14/23,
honest partial); `CP_2_4_REPORT.md` landed (md5 `573c50d6`). Next: CP 2.5
(Phase 2 close → 15/23).

**Composed gate result (criterion c — PASS):** one clean 15-invocation run
(`cp_2_4/composed.runlog` md5 `19fe3af1`, `composed_result.json` `9b126710`),
Mistral-7B B=1, n=5 matched pairs, 3 arms (vanilla / Marlin / all-on):
- **all-on/vanilla tok/W 3.617× [3.591, 3.642]**, tok/s 1.795× [1.777, 1.813]
- Marlin/vanilla tok/W 1.307× (UNDER scorecard 1.62× — B=1 is Marlin's weak
  regime, designed B≥8); also serves as criterion (b)'s Marlin per-lever.
- composition clean multiplicative (1.090 × 1.647 = 1.7955× = composed tok/s);
  all-on/marlin tok/s 1.647× reproduces standalone Mistral n-gram 1.639×.
- VOLT verified: all-on watts.csv clock = 1005 MHz, vanilla/marlin 1980.
- 3.617× tok/W exceeds the old 2.96× Llama scorecard, but the lever mix
  differs (Marlin under, DVFS dominant) — attributed in report §4, not
  reframed. DVFS over-delivers (+62% tok/W, null-validated).

**Gate scorecard:** (a) PASS Marlin lock-fix · (b) PASS per-lever · (c) PASS
composed · **(d) NOT MET** — Llama model-draft 1.75× primary; the model-draft
arm is correct in isolation (accept 0.490) but acceptance collapses to 0.036
under the live substrate (finding 3b, Marlin-INT4 two-model interaction),
deferred to Phase 5 · (e) PASS anchors held.

**Spec arms:** both n-gram, greedy-decode regime — Mistral 1.639× [1.602,
1.677], Llama 1.597× [1.514, 1.680] (n=25; one transient outlier pair4/p0
0.690× → excl-sensitivity n=24 1.635× [1.595, 1.675]).

Anchors held: kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`, libcipher_rt
`5e304549`, taint 12288 — no ABI change.

Linked: [[cipher-fusion-campaign]], [[cipher-marlin-primary-ctx-pin]],
[[cipher-t43-envelope]] (DVFS envelope superseded by the null-validated
sweep), [[cipher-lift-framing]].
