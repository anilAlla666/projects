export const meta = {
  name: 'cipher-verify-3q',
  description: 'Read-only code-cited verification of 3 CIPHER claims: 745 TFLOPS provenance, MFU actuator counter truth, all-three-together regime composition',
  phases: [
    { title: 'Investigate', detail: '3 deep code+artifact investigators (745 TFLOPS, MFU counters, regime composition)' },
    { title: 'Verify', detail: '3 adversarial cross-checks' },
  ],
}

const CTX = `
READ-ONLY verification of the CIPHER codebase. Root = /home/ubuntu. DO NOT edit/move/delete. Read, grep, git-log only. Cite file:line for EVERY claim. Each question is allowed to come back "no/negative" — blunt honesty over a flattering answer (this project has a documented honesty arc; overclaims are the failure mode).

GROUND TRUTH already established:
- Shipped substrate = cipher_rt_phase4/libcipher_rt.so md5 2edba0d2 (anchor). OBJS = cipher_rt_phase4/Makefile:55-107.
- The .so emits CLASSIFY telemetry lines (grep "cipher_v2] CLASSIFY") with real counters: marlin_bf16_obs / marlin_bf16_sub (Marlin), mach_intercepts / mach_sub (Machete), koop_bf16_obs (Koopman observe), marlin_engage / volt_engage / multi flags. FP8 substitution shows as handled=N in V0A_JSON lines.
- Measurement artifacts live mostly under cipher-fusion-evidence/ (cp_3_3, cp_0_6, v1_phase_b/...) and cipher-may13-evidence/.
- Inventory verdicts: FP8 actuator live (real cublasGemmEx E4M3 substitution); Marlin live (capture-safe, density not delivered); Koopman handled=0 even when registered (koopman_graph.txt:275 — cipher-fusion-evidence/v1_phase_b/results/b6_9_8_5b_3/); Machete intercept-only, g_machete_substituted never incremented (cipher_rt_machete_intercept.c:42,140-142).
- The 24 MFU-pillar files (from the inventory): cipher_rt_fp8_actuator.c, cipher_rt_fp8_engine.cpp, cipher_rt_fp8.h, cipher_rt_marlin_actuator.c, cipher_rt_marlin_engine.cpp, cipher_rt_marlin_kernel_src.cpp/.h, cipher_rt_marlin.h, cipher_rt_marlin_perms.h, cipher_rt_machete_intercept.c, cipher_rt_matmul_dispatch.c/.h, cipher_rt_cublas_shim.c, cipher_rt_cublaslt_layout.c/.h, cipher_rt_cublaslt_variants.c, cipher_rt_koopman_engine.cpp, cipher_rt_koopman.h, cipher_rt_attn_6pattern.c, cipher_rt_attn_dispatch.cpp/.h, cipher_rt_attn_test_actuator.c (all in cipher_rt_phase4/).
- Product layer: cipher_product.py (router), cipher_v2_build/pkgroot/etc/cipher/platform.json (regimes agent/density/compute -> handlers + inject env), cipher_v2_build/src/cipher_platform/. Env gates: CIPHER_FP8, CIPHER_MARLIN, CIPHER_VOLT, CIPHER_KOOPMAN, CIPHER_RT_DISABLE_AUTO_INIT, CUDA_INJECTION64_PATH.
`;

phase('Investigate')

const Q1_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    exists_as_figure: { type: 'boolean', description: 'does the string 745 TFLOPS appear anywhere' },
    exists_as_real_measurement_artifact: { type: 'boolean', description: 'is there a measurement JSON/log with value ~745 produced by a committed script (not just narrative prose)' },
    citations: { type: 'array', items: { type: 'string' }, description: 'file:line of every 745 occurrence that matters' },
    producing_script: { type: 'string', description: 'the script that produced it, or NONE-FOUND' },
    date: { type: 'string' },
    workload: { type: 'string', description: 'model, prefill or decode, batch' },
    single_gpu: { type: 'string', enum: ['yes','no','unknown'] },
    closest_real_artifact: { type: 'string', description: 'the nearest actual committed measurement (value + file:line + script)' },
    blunt_answer: { type: 'string' },
  },
  required: ['exists_as_figure','exists_as_real_measurement_artifact','citations','producing_script','blunt_answer'],
};

const q1 = agent(CTX + `

QUESTION 1 — THE 745 TFLOPS NUMBER. Determine whether "745 TFLOPS" is a real, file-cited MEASUREMENT (a number with the script that produced it) or just a narrative figure.

Seed findings (verify + go deeper):
- 745 appears in PLANNING/SUMMARY prose: cipher-fusion-evidence/CIPHER_REENGINEERING_PLAN.md:971 ("CP 3.3 measured 745 TFLOPS ... at 59.8% MFU; device sum 887; NVML 916. ... The 745 figure was likely a brief uncapped burst or proxy-counter over-read") and CIPHER_PLAN_EXECUTIVE_SUMMARY.md:28,132,134,135,311 (self-caveated as burst/over-read; defensible ceiling ~660 TFLOPS/67% at 700W vs 989 H100 bf16 peak).
- The actual cp_3_3 flop-gate JSON artifacts report DIFFERENT values: cipher-fusion-evidence/cp_3_3/flop_gate_result.json (analytical_tflops ~784.4) and cp_3_3/reverify_0_4_8/flop_gate_result.json (analytical 779.79 / sum_attributed 888.58 / device_phaseB 888.58). So 745 is NOT literally in those JSONs.
- The V.0 unified bf16/fp8 sweep (cipher-fusion-evidence/v1_phase_b/v1_soak/v0_unified/v0_phaseA.txt V0A_JSON) shows the LATER real single-GPU numbers: bf16 best batch=12 tflops=521.8 mfu=69.5; fp8 best batch=12 tflops=644.5 mfu=67.2 — nowhere near 745.

YOUR JOB:
1. Find the EXACT script + artifact that produced 745 (look in cp_3_3/, cp_0_6/, any flop_gate*.py / mfu*.py / *flop*.py script, capture logs, *.json with achieved/attributed tflops ~745). Does any committed measurement output literally ~745?
2. If 745 only lives in narrative prose attributed to "CP 3.3" without a reproducing artifact of that exact value, say so plainly.
3. Pin down the workload (model? prefill/decode? batch?), date (git log / file mtime), and whether single-GPU.
4. Give the closest REAL committed measurement and its value.
Blunt verdict: is "745 TFLOPS" a real file-cited measurement, or a self-caveated narrative number?`,
  { label: 'q1-745-tflops', phase: 'Investigate', model: 'opus', schema: Q1_SCHEMA });

const Q2_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    files: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        path: { type: 'string' },
        actuator_entry: { type: 'string', description: 'file:line of the actuator entry / register / dispatch fn' },
        counter: { type: 'string', description: 'the counter name + value from a real artifact (e.g. marlin_bf16_sub=24427), or n/a' },
        status: { type: 'string', enum: ['live-substituting','intercept-only','inert','support-no-counter'] },
        evidence: { type: 'string', description: 'file:line for the counter value + the actuator' },
      },
      required: ['path','actuator_entry','status','evidence'],
    }},
    real_levers: { type: 'array', items: { type: 'string' } },
    inert_scaffolding: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['files','real_levers','inert_scaffolding','summary'],
};

const q2 = agent(CTX + `

QUESTION 2 — THE MFU ACTUATOR TRUTH. For every one of the 24 MFU-pillar files, give: (a) file:line of the actuator entry point, and (b) its LIVE status from the ACTUAL counter value found in a real artifact. Classify each: live-substituting / intercept-only / inert / support-no-counter (header or kernel-source-blob with no runtime counter).

For the counters, dig into real artifacts (do not just read code):
- FP8: V0A_JSON handled=N lines in cipher-fusion-evidence/v1_phase_b/v1_soak/v0_unified/v0_phaseA.txt (fp8 handled up to 29025/34875). Also cipher_rt_fp8_actuator.c counter (g_calls_handled / cipher_rt_fp8_handled).
- Marlin: CLASSIFY lines marlin_bf16_obs / marlin_bf16_sub. Find a capture where marlin_bf16_sub is large (e.g. cipher-fusion-evidence/v1_phase_b/k1_close_gate/captures/P3m_fp16b.log marlin_bf16_sub up to 24427) AND one where it is 0 (v0_phaseA marlin_bf16_sub=0) — report both so the gating is clear. Code counter: cipher_rt_marlin_actuator.c g_calls_handled (:326).
- Koopman: koop_bf16_obs in CLASSIFY (observe>0) but calls_handled=0 (koopman_graph.txt:275 in cipher-fusion-evidence/v1_phase_b/results/b6_9_8_5b_3/). Confirm handled=0.
- Machete: mach_intercepts / mach_sub (both 0 in all CLASSIFY lines seen). Code: cipher_rt_machete_intercept.c g_machete_substituted never incremented.
- Attn (attn_6pattern, attn_dispatch, attn_test_actuator): CLASSIFY attn_p1/p2/p5/p6 counters — are these substitutions or just pattern-observations? Determine live vs observe-only.
- cublas_shim / cublaslt_* / matmul_dispatch: these are the intercept+dispatch substrate, not actuators with their own substitution counter — classify as intercept-only or support, and note whether they actually route (the GOT-patch intercept count).

Deliver one row per file. Then list which MFU files are REAL LEVERS (actually substitute work) vs INERT SCAFFOLDING.`,
  { label: 'q2-mfu-counters', phase: 'Investigate', model: 'opus', schema: Q2_SCHEMA });

const Q3_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    substrate_can_engage_all_three: { type: 'boolean', description: 'can the .so classifier set marlin_engage+volt_engage+multi all=1 on one workload in one process' },
    substrate_all_three_actually_substitute: { type: 'boolean', description: 'in that same single process/run, do MFU(marlin/fp8) AND TPW(volt) AND multitenant all ACTUALLY fire (not just engage flags)' },
    product_fires_all_three_together: { type: 'boolean', description: 'does the shipped V.2 product (cipher_product.py/platform.json) run them in ONE config on ONE workload, or split into separate regimes' },
    env_gating: { type: 'array', items: { type: 'string' }, description: 'file:line of the per-regime inject env that turns each pillar on/off' },
    compose_evidence: { type: 'string', description: 'file:line of any single run/capture where all three co-fire' },
    nocompose_evidence: { type: 'string', description: 'file:line proving separate regimes / DISABLE_AUTO_INIT in engine' },
    plus57_provenance: { type: 'string', description: 'where does TPW +57% come from + what it measures' },
    x306_provenance: { type: 'string', description: 'where does the cross-tenant 3.06x come from + what it measures' },
    blunt_answer: { type: 'string', description: 'architecturally one-config-all-three-firing, or one-product-three-separate-regimes?' },
  },
  required: ['substrate_can_engage_all_three','product_fires_all_three_together','env_gating','blunt_answer'],
};

const q3 = agent(CTX + `

QUESTION 3 — THE "ALL THREE TOGETHER" RECONCILIATION. Can MFU (FP8/Marlin), TPW (DVFS +57% AND cross-tenant 3.06x), and the multi-tenant engine fire in ONE configuration on ONE workload? Or are they separate regimes? Answer the literal question: is "ship all three firing together" architecturally TRUE, or is it "one product containing three regimes that fire separately"?

Evidence to chase:
- COMPOSE side: cipher-fusion-evidence/v1_phase_b/k1_close_gate/captures/P3m_fp16b.log has CLASSIFY lines with marlin_engage=1 AND volt_engage=1 AND multi=1 SIMULTANEOUSLY and marlin_bf16_sub>0 (e.g. obs=24605 sub=24427). Verify: is this ONE injected .so process where MFU(marlin substituting) + TPW(volt engaged) + multitenant(multi=1) all truly co-fire? What config/script produced P3m_fp16b? Is volt_engage=1 actually changing clocks there, or just a flag?
- NO-COMPOSE side: the inventory found the V.2 PRODUCT splits regimes — cipher_product.py runs agent/density/compute as separate isolated subprocesses; the engine handler (cipher_inc4.py:6-8) sets CIPHER_RT_DISABLE_AUTO_INIT=1 so the .so compute stack (FP8/Marlin/volt) is OFF in the engine path; platform.json injects different env per regime (compute: CIPHER_FP8=1; agent: CIPHER_RT_DISABLE_AUTO_INIT=1, CIPHER_VOLT=0). Confirm the per-regime env mutual-exclusivity with file:line.
- The TPW numbers: find where "+57%" and "3.06x" (cross-tenant) come from. grep for 57, 3.06, 3.0x, "cross-tenant", tok/W, tokens/W, DVFS. What exactly does each measure, on what workload, and were they measured in the SAME run as MFU substitution or in isolation?

Reconcile precisely: distinguish (A) the SUBSTRATE/.so CAN classify-and-engage all three on one workload [if true, cite it] from (B) the shipped PRODUCT actually runs them together vs as separate regimes [cite the router + env gating]. Then give the blunt one-sentence verdict.`,
  { label: 'q3-compose', phase: 'Investigate', model: 'opus', schema: Q3_SCHEMA });

const inv = await parallel([() => q1, () => q2, () => q3]);
const [r1, r2, r3] = inv;
log('Investigation done: q1=' + (r1 ? 'ok' : 'null') + ' q2=' + (r2 ? 'ok' : 'null') + ' q3=' + (r3 ? 'ok' : 'null'));

phase('Verify')

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    question: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED','REFUTED','PARTIAL','UNCERTAIN'] },
    evidence: { type: 'string', description: 'file:line proof' },
    correction: { type: 'string' },
  },
  required: ['question','verdict','evidence'],
};

const c1 = r1 ? JSON.stringify(r1).slice(0, 1100) : 'n/a';
const c2 = r2 ? JSON.stringify(r2).slice(0, 1100) : 'n/a';
const c3 = r3 ? JSON.stringify(r3).slice(0, 1400) : 'n/a';

const verifyTasks = [
  'Q1 VERIFY: The investigator claims about 745 TFLOPS: ' + c1 + ' . ADVERSARIALLY re-check: independently grep for any committed measurement script/JSON/log that outputs a value within +/-5 of 745 (achieved or attributed TFLOPS) on a single GPU. If you find a real reproducing artifact, REFUTE the narrative-only finding and cite it. If you cannot, CONFIRM that 745 is a self-caveated narrative figure with no reproducing committed artifact at that exact value. Be specific about the closest real number.',
  'Q2 VERIFY: The MFU-counter classification: ' + c2 + ' . ADVERSARIALLY re-check the two most consequential calls: (a) is Koopman truly handled=0 in EVERY committed run, or is there ANY artifact with koopman calls_handled>0 outside a hand-seeded test? (b) is Marlin genuinely live-substituting (marlin_bf16_sub>0 in a real run) AND is FP8 genuinely live (handled>0)? Cite the counter file:line. Flag any MFU file the investigator mislabeled live-vs-inert.',
  'Q3 VERIFY: The compose/no-compose finding: ' + c3 + ' . ADVERSARIALLY re-check the literal question. (a) Confirm P3m_fp16b.log is ONE .so process where MFU substitution (marlin_bf16_sub>0) + volt_engage=1 + multi=1 genuinely co-occur, and say whether volt_engage=1 there actually actuated DVFS or was just a flag with no clock change. (b) Confirm the shipped product (cipher_product.py + platform.json) runs FP8/Marlin/volt/engine in MUTUALLY EXCLUSIVE regime subprocesses (cite the inject env per regime + DISABLE_AUTO_INIT in the engine). (c) Verify the +57% and 3.06x were NOT measured in the same run as MFU substitution. Final: is "all three firing together on one workload" architecturally true at the substrate level but NOT how the product is packaged? Give the precise blunt verdict.',
];

const verdicts = (await parallel(verifyTasks.map((t, i) => () =>
  agent(CTX + `

ADVERSARIAL VERIFICATION. Default to skepticism; cite file:line.

` + t,
    { label: 'verify-q' + (i + 1), phase: 'Verify', model: 'opus', schema: VERDICT_SCHEMA })
))).filter(Boolean);

return { q1: r1, q2: r2, q3: r3, verdicts: verdicts };
