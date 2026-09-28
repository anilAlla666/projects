export const meta = {
  name: 'cipher-pillar-inventory',
  description: 'Read-only pillar inventory of the CIPHER codebase (MFU/TPW/multi-tenant/shared/dead) with entanglement + selftest analysis',
  phases: [
    { title: 'Classify', detail: 'per-file pillar classification across live source trees' },
    { title: 'Analyze', detail: 'entanglement chokepoints, kv/accelerate path, selftest dependency' },
    { title: 'Verify', detail: 'adversarially verify high-stakes DEAD + ENTANGLED claims' },
  ],
}

const CTX = `
You are doing a READ-ONLY pillar inventory of the CIPHER codebase. Working dir root = /home/ubuntu.
DO NOT edit, move, or delete anything. Read, grep, git-log only.

PILLARS:
- MFU  = max-throughput / compute-bound single-GPU actuators (FP8 GEMM substitution, int4 Marlin, matmul/cublas substitution, attn compute).
- TPW  = tokens-per-watt / energy (DVFS host-side clock-set = "volt"; kmod clock control).
- MULTITENANT = multi-model co-residence and serving: pager residency, coresidence registry, tenant tracking, fairness, SM partitioning (partition_router / green_ctx / sm_packer), cross-tenant batching/coalescing engine, KV dedup across tenants, classify/route-by-regime, model swap.
- SHARED = serves >=2 non-dead pillars (intercept surface: dlsym/got_patch/cuinit/cupti/inject; dispatch; matmul_dispatch chokepoint; counter/ring telemetry; pager when it feeds >1).
- DEAD = scaffolded-never-fired / retired / abandoned / inert translation-unit.

MECHANICAL SHARED RULE: count the non-dead pillars a file serves. >=2 => SHARED. exactly 1 => that single pillar. 0 (never actuates) => DEAD.

TWO-LEVEL DEAD RULE (every DEAD row MUST state which level + cite file:line):
(a) NOT LINKED — not in the shipped libcipher_rt.so OBJS list (cipher_rt_phase4/Makefile ~L55-107), OR explicitly retired, OR a .pre_*/.bak backup, OR unbuilt. Cite Makefile:line or filename.
(b) LINKED-BUT-NEVER-ACTUATES — compiled into the .so but handled=0 / registered=0 / gated-off-at-runtime / inert-TU / no caller. Cite the gate var or the absent-caller evidence.
IN-OBJS != FIRES. A file can be linked yet DEAD.

GROUND TRUTH (verify against the code; do not blindly trust):
- Shipped substrate = cipher_rt_phase4/libcipher_rt.so, md5 2edba0d2 (the project anchor). The OBJS list in cipher_rt_phase4/Makefile (~L55-107) is the definitive compiled-in set.
- cipher_rt_arbitrate.o RETIRED, .c kept unbuilt (Makefile:52-54); arbitrate.o absent on disk = confirmed not-linked.
- may13_* family (src/may13/*, include/may13/*) = the LEGACY single-tenant CIPHER actuator layer (Koopman/EDMD/LNN/liquid-state/green-ctx/carbon/fairness/topology/comply/etc). Makefile (~L285-294) flags several as "inert TUs / no runtime effect". The product regimes inject CIPHER_RT_DISABLE_AUTO_INIT=1 which disables the may13 auto-init path.
- Koopman (cipher_rt_koopman_engine + may13 edmd/edmd_live/koopman_runtime): does NOT compose into the engine, handled=0/registered=0, capture-illegal (cudaHostAlloc edmd_live.cpp:171). MFU-adjacent but dead-in-engine.
- Marlin int4 (cipher_rt_marlin_*): capture-safe, real substitution, but density NOT delivered. Live MFU/density actuator.
- FP8 (cipher_rt_fp8_*): live MFU actuator, real cublasGemmEx E4M3 substitution, modest gain (1.05x).
- DVFS/volt (cipher_rt_volt): live TPW actuator, host-side clock-set between waves.
- Per the V.0 scorecard: SM-packer FIRED (counter 61278 => LIVE); green-ctx/audit/ring/sense/NCCL-tuner were reported DORMANT — VERIFY against code (registered? called?).
- accelerate path (cli.py cmd_accelerate) injects CIPHER_KV_ALLOC=1 + CIPHER_KVDEDUP=1 (the one real request->tokens path under a customer vLLM). NOTE: cipher_rt_kv_alloc.c is NOT in the shipped OBJS (kv_alloc.o exists on disk but unlinked) => CIPHER_KV_ALLOC may be a no-op in the deployed .so; the real effect may be CIPHER_KVDEDUP via the kmod (cipher_kvdedup.c) + the cipher_vllm_plugin.
- ENTANGLED = a path you cannot change for one pillar without affecting another (e.g. Marlin[density] and FP8[MFU] both routing through matmul_dispatch/cublas_shim; the engine keeping the GPU fed serving MFU+TPW+density; cross-tenant batching).

Use cipher_rt_phase4 git log + in-tree .pre_*/.bak backups + Makefile retired/inert markers as primary 'abandoned' evidence.
`;

const ROWS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    rows: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        path: { type: 'string', description: 'path relative to /home/ubuntu' },
        role: { type: 'string', description: 'one-line: what this file does' },
        pillars: { type: 'array', items: { type: 'string', enum: ['MFU','TPW','MULTITENANT','SHARED','DEAD'] } },
        linked_in_so: { type: 'string', enum: ['yes','no','n/a'], description: 'in shipped libcipher_rt.so OBJS? n/a for non-.so files' },
        entangled: { type: 'boolean' },
        entangled_detail: { type: 'string', description: 'if entangled: shared path + file:line, else empty' },
        safe_to_touch: { type: 'string', enum: ['yes','no','qualified'] },
        safe_to_touch_reason: { type: 'string' },
        dead_level: { type: 'string', enum: ['not-linked','linked-inert','none'], description: 'for DEAD rows which level; none if not dead' },
        evidence: { type: 'string', description: 'file:line citations for DEAD/ENTANGLED/key claims' },
      },
      required: ['path','role','pillars','linked_in_so','entangled','safe_to_touch','dead_level','evidence'],
    }},
    batch_notes: { type: 'string' },
  },
  required: ['rows'],
};

phase('Classify')

const classifyBatches = [
  { label: 'rt-compute-MFU', files: 'cipher_rt_phase4: cipher_rt_fp8_actuator.c, cipher_rt_fp8_engine.cpp, cipher_rt_fp8.h, cipher_rt_marlin_actuator.c, cipher_rt_marlin_engine.cpp, cipher_rt_marlin_kernel_src.cpp, cipher_rt_marlin_kernel_src.h, cipher_rt_marlin.h, cipher_rt_marlin_perms.h, cipher_rt_machete_intercept.c, cipher_rt_matmul_dispatch.c, cipher_rt_matmul_dispatch.h, cipher_rt_cublas_shim.c, cipher_rt_cublaslt_layout.c, cipher_rt_cublaslt_layout.h, cipher_rt_cublaslt_variants.c, cipher_rt_koopman_engine.cpp, cipher_rt_koopman.h, cipher_rt_attn_6pattern.c, cipher_rt_attn_dispatch.cpp, cipher_rt_attn_dispatch.h, cipher_rt_attn_test_actuator.c' },
  { label: 'rt-multitenant', files: 'cipher_rt_phase4: cipher_rt_pager.c, cipher_rt_pager.h, cipher_rt_coresidence.c, cipher_rt_coresidence.h, cipher_rt_tenant.cpp, cipher_rt_tenant.h, cipher_tenant.c, cipher_rt_fairness.c, cipher_rt_fairness.h, cipher_rt_partition_router.c, cipher_rt_partition_router.h, cipher_rt_green_ctx.c, cipher_rt_green_ctx.h, cipher_rt_sm_packer.c, cipher_rt_sm_packer.h, cipher_rt_arbitrate.c, cipher_rt_arbitrate.h, cipher_rt_classify_observer.c, cipher_rt_classify_observer.h, cipher_rt_classify_substrate.cpp, cipher_rt_classify_substrate.h' },
  { label: 'rt-kv-telemetry', files: 'cipher_rt_phase4: cipher_rt_kv_alloc.c, cipher_rt_kv_alloc.h, cipher_rt_pool.c, cipher_rt_pool.h, cipher_kv_cache.py, cipher_kv_bridge.cpp, cipher_rt_volt.c (TPW), cipher_rt_volt.h, cipher_rt_counter_dump.c, cipher_rt_counter_dump.h, cipher_rt_ring_write.c, cipher_rt_ring_write.h, cipher_rt_tc_probe.c, cipher_rt_tc_probe.h, cipher_rt_geom_capture.c, cipher_rt_geom_capture.h, cipher_rt_sense_transition.c, cipher_rt_sense_transition.h, cipher_rt_oracle_bridge.cpp, cipher_rt_oracle_bridge.h, cipher_rt_remember_consumer.cpp, cipher_rt_remember_consumer.h, cipher_stream_resolver.c, cipher_stream_resolver.h' },
  { label: 'rt-shared-intercept', files: 'cipher_rt_phase4: cipher_inject.c, cipher_cupti.c, cipher_rt_dlsym_hook.c, cipher_rt_dlsym_hook.h, cipher_rt_got_patch.c, cipher_rt_got_patch.h, cipher_rt_cuinit_hook.c, cipher_rt_dispatch.cpp, cipher_rt_dispatch.h, cipher_rt_intercept.h, cipher_rt_commit.c, cipher_rt_commit.h, cipher_rt_audit.c, cipher_rt_audit.h, cipher_v2_internal.h, src/cipher_workload_detect.cpp, include/cipher_workload_detect.h, src/cipher_may13_harness.cpp, src/cipher_may13_stubs.cpp, src/may13_intercept/cipher_graph_inspect.cpp, src/may13_intercept/cipher_intercept_cudart.cpp, src/may13_intercept/cipher_intercept_stubs.cpp' },
  { label: 'rt-may13-math', files: 'cipher_rt_phase4/src/may13: cipher_edmd.cpp, cipher_edmd_live.cpp, cipher_koopman_runtime.cpp, cipher_lnn.cpp, cipher_liquid_state.cu, cipher_block_sub_kernel.cu, cipher_l2_persist.cu, cipher_green_ctx.cu, cipher_structural_lookup.cpp, cipher_recipes.cpp, cipher_kernel_table.cpp, cipher_runtime.cpp. Also skim include/may13/*.h (randsvd, param_recovery, predict, attn_koopman, koopman_runtime, edmd, lnn, liquid_state). LEGACY actuator-math layer. For each: linked in OBJS? actuates at runtime or inert TU?' },
  { label: 'rt-may13-orch', files: 'cipher_rt_phase4/src/may13: cipher_dispatch.cpp, cipher_oracle.cpp, cipher_sense.cpp, cipher_telemetry.cpp, cipher_continuity.cpp, cipher_loop.cpp, cipher_pipeline.cpp, cipher_pulse.cpp, cipher_trace.cpp, cipher_receipt.cpp, cipher_carbon.cpp, cipher_fairness.cpp, cipher_fairness_shm.cpp, cipher_guard.cpp, cipher_determinism.cpp, cipher_topology.cpp, cipher_comply.cpp. LEGACY orchestration/telemetry/governance layer. For each: linked? actuates or inert? cipher_carbon/cipher_fairness relate to TPW/multitenant conceptually but check if they actually fire.' },
  { label: 'rt-probes-tests', files: 'cipher_rt_phase4 python+test sources: cipher_spec_decode.py, d1_kl_probe.py, hang_bisect.py, pillar_driver.py, spec_llama_smoke.py, spec_measure_driver.py, spec_smoke.py, spec_varied_driver.py, spec_verify.py, test_cipher_spec_decode.py, test_spec_generate.py, tf_gate_driver.py, test_pager.c, test_residency.c, test_step3_b0_producer.cpp, test_step3_b1_consumer.cpp, test_step3_c_lmhead_validate.cpp, test_step3_c_lmhead_validate.py. Dev probes/drivers/tests (NOT shipped in the .so or deb). Classify by which pillar each probes; set linked_in_so=no for all. Most are DEAD-style one-off probes OR pillar-specific test harnesses.' },
  { label: 'kmod', files: 'cipher_kmod/*.c + *.h (all 25): cipher_audit_chain.c, cipher_bar0.c, cipher_clock.c (TPW?), cipher_coresidence_registry.c, cipher_cp54_sched.c, cipher_dev.c, cipher_fairness_ledger.c/.h, cipher_flops.c, cipher_internal.h, cipher_ioctl.h, cipher_ioctl_decode.c, cipher_kvdedup.c/.h, cipher_main.c, cipher_model_registry.c, cipher_probe.c, cipher_proc.c, cipher_state_updater.c, cipher_stream_registry.c/.h, cipher_tenant_snapshot.c, cipher_weight_arena.c, probe_microbench.c, cipher_kmod.mod.c (autogen). The kmod ships in the deb via DKMS 0.7.0. Map each feature to a pillar (clock=TPW; coresidence/kvdedup/fairness/sched/model_registry/tenant_snapshot/weight_arena=MULTITENANT; dev/proc/main/bar0/audit/ioctl_decode=SHARED plumbing). Flag probe_microbench.c and cipher_kmod.mod.c specifically.' },
  { label: 'orchestration', files: 'libcipher_v2/{cipher_inject.c, cipher_tenant.c, cipher_cupti.c, cipher_v2_internal.h, Makefile} (the libcipher_v2.so CUPTI injection lib, also bundled in deb) AND cipher_v2_build/src/cipher_platform/{__init__,cli,config,gates,isolate,report,router}.py AND cipher_v2_build/pkgroot/etc/cipher/platform.json AND cipher_v2_build/pkgroot/usr/bin/cipher AND cipher_v2_build/build_deb.sh. Orchestration/product layer. router.py dispatches by regime to handlers. Classify whether each file is pillar-agnostic SHARED (router/cli/config/report/isolate/inject/cupti) or pillar-specific (gates encodes which gates run; platform.json maps regimes->handlers/inject).' },
  { label: 'handlers', files: 'The 10 shipped deb handlers. Home-level: cipher_inc4.py (multi-model engine handler), v0_phaseA_maxmfu.py (MFU), v0_phaseC_nf4_cofire.py (density/multitenant), cipher_engine.py, cipher_engine_batched.py, cipher_product_report.py. cipher_v2_mock/: smoke_agent_scaled.py (agent/multitenant quick), smoke_density_real.py (density quick), mock_handler.py (compute quick), off_byte_identical_real.py (OFF control gate). Classify each by pillar. CRITICAL per memory: handlers run FIXED demo traces + os._exit, NOT a serving runtime. Note which regime (agent/density/compute in platform.json) each maps to and which are quick_handlers used by selftest.' },
];

const triageBatch = agent(CTX + `

TASK: Directory-level TRIAGE of the non-canonical CIPHER dirs to decide LIVE vs ARCHIVAL/DEAD. The LIVE codebase feeding the deb is classified separately (cipher_rt_phase4, cipher_kmod, libcipher_v2, cipher_platform, the 10 handlers). Your job: for EVERY other cipher* dir under /home/ubuntu, decide archival-vs-live and spot-check.

LIVE TEST (cite which applies): LIVE if it (1) feeds the deb / builds a shipped artifact, (2) is imported by a shipped handler, or (3) is the accelerate target. Otherwise ARCHIVAL (snapshot/fallback/rollback/phase-evidence/experiment).

SPOT-CHECK THESE (read a few files before calling archival):
- cipher-fusion-evidence (318 src files; V.0/V.1/V.2 working dir; does build_deb.sh or any handler import from it?)
- cipher_vllm_plugin (11 files; the accelerate/KVDEDUP target under customer vLLM: cipher_vllm_kv.py, cipher_vllm_kvdedup.py; LIKELY LIVE)
- cipher_workloads (34 files), cipher_exporter, cipher_gpustate, cipher_measurement, cipher_v2_mock

ARCHIVAL CANDIDATES (confirm + cite reason): cipher-phase1-evidence, cipher-phase1.5-evidence, cipher-phase2-evidence, cipher-phase4-evidence, cipher-phase4-cutover-evidence, cipher-may13-evidence, cipher-baselines, cipher_kmod_phase2_snapshot, cipher_kmod_rollback_workspace, cipher_kmod_fallback, cipher_ko_fallback, cipher_rt_fallback, cipher_rt_phase4_draft, cipher_phase4_tests, libcipher_v2 (live build dir or dup?).

Return one row PER DIRECTORY (dir path as path, file-count + verdict in role, pillars=[DEAD] for archival or served pillar(s) for live, dead_level=not-linked for archival). In batch_notes: list any LIVE files found inside an otherwise-archival dir, and confirm whether cipher_vllm_plugin is the real accelerate/KVDEDUP path.`,
  { label: 'triage-dirs', phase: 'Classify', model: 'sonnet', schema: ROWS_SCHEMA });

const classifyThunks = classifyBatches.map(b => () =>
  agent(CTX + `

TASK: Classify each of these CIPHER source files into pillar(s). Read each file and grep the cipher_rt_phase4 tree for callers / registration / gating env vars to decide linked-vs-fires. Apply the mechanical SHARED rule and two-level DEAD rule. Cite file:line for every DEAD and ENTANGLED row. Be precise about linked_in_so (check the OBJS list) and dead_level.

FILES:
` + b.files,
    { label: b.label, phase: 'Classify', model: 'sonnet', schema: ROWS_SCHEMA }));

const ENTANGLE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    chokepoints: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        path: { type: 'string' },
        symbol: { type: 'string', description: 'the shared function/dispatch symbol' },
        pillars_sharing: { type: 'array', items: { type: 'string' } },
        why: { type: 'string', description: 'how the pillars share this path' },
        cannot_change_without: { type: 'string', description: 'what breaks in other pillars if you touch it' },
        evidence: { type: 'string', description: 'file:line of the shared calls' },
      },
      required: ['path','pillars_sharing','why','evidence'],
    }},
    summary: { type: 'string' },
  },
  required: ['chokepoints'],
};

const entangleAgent = agent(CTX + `

TASK: ENTANGLEMENT call-graph pass. Entanglement is a property BETWEEN files — derive it from the call graph, not per-file self-reports. For each chokepoint, read it and grep the cipher_rt_phase4 tree to determine WHICH pillars route through it and what would break in other pillars if you changed it for one pillar.

CHOKEPOINTS (read + trace callers/callees):
- cipher_rt_matmul_dispatch.c/.h (do FP8[MFU], Marlin[MFU/density], bf16 passthrough all route here?)
- cipher_rt_cublas_shim.c + cipher_rt_cublaslt_variants.c/_layout.c (GEMM interception shared by MFU + density)
- cipher_rt_dlsym_hook.c + cipher_rt_got_patch.c (symbol-interception substrate every actuator depends on)
- cipher_rt_partition_router.c (SM partitioning: multitenant isolation AND MFU/TPW packing?)
- cipher_rt_classify_substrate.cpp + cipher_rt_classify_observer.c (regime classification feeding the router)
- cipher_rt_pager.c (multi-model residency: density AND throughput?)
- cipher_rt_dispatch.cpp (central dispatch)
- cipher_rt_cupti.c / cipher_inject.c (injection entrypoint shared by all)
- engine handler cipher_inc4.py + cipher_engine.py / cipher_engine_batched.py (does the engine that keeps the GPU fed serve MFU+TPW+density simultaneously via cross-tenant batching/coalescing = the canonical MFU<->multitenant entanglement?)

Report each genuine shared path with file:line. Expected biggest findings: the matmul/cublas dispatch chain and the engine-keeps-GPU-fed.`,
  { label: 'entanglement-pass', phase: 'Analyze', model: 'opus', schema: ENTANGLE_SCHEMA });

const KV_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    kv_alloc_verdict: { type: 'string', description: 'pillar + live/dead status of cipher_rt_kv_alloc.c in the SHIPPED .so' },
    kv_alloc_in_shipped_so: { type: 'boolean' },
    kvdedup_path: { type: 'string', description: 'where CIPHER_KVDEDUP is actually consumed (kmod? vllm plugin? rt?)' },
    accelerate_real_effect: { type: 'string', description: 'in the deployed product, what does the accelerate command actually do?' },
    env_consumers: { type: 'array', items: { type: 'string' }, description: 'file:line where CIPHER_KV_ALLOC / CIPHER_KVDEDUP are read' },
    pillar: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['kv_alloc_verdict','kv_alloc_in_shipped_so','kvdedup_path','accelerate_real_effect'],
};

const kvAgent = agent(CTX + `

TASK: Resolve the kv_alloc / accelerate REAL SERVING PATH (the one real request->tokens path). Trace exactly where CIPHER_KV_ALLOC and CIPHER_KVDEDUP are consumed and what the accelerate command actually does in the DEPLOYED product.

Investigate:
1. cipher_rt_phase4/cipher_rt_kv_alloc.c reads CIPHER_KV_ALLOC. Is cipher_rt_kv_alloc.o in the shipped libcipher_rt.so OBJS list (cipher_rt_phase4/Makefile L55-107)? kv_alloc.o exists on disk but I did NOT see it in OBJS — confirm. If NOT linked, CIPHER_KV_ALLOC is a NO-OP in the shipped .so. There is a cipher_rt_kv_alloc.c.pre_phase_c_sc2 backup — note it.
2. cipher_kmod/cipher_kvdedup.c/.h reads CIPHER_KVDEDUP. Ships via DKMS. Is this the real KV-dedup mechanism?
3. cipher_vllm_plugin/ (cipher_vllm_kv.py, cipher_vllm_kvdedup.py): is THIS the accelerate target running under a customer vLLM?
4. cli.py cmd_accelerate (cipher_v2_build/src/cipher_platform/cli.py ~L55-63): sets CUDA_INJECTION64_PATH=so_path + accelerate_env. What does it exec and which mechanism actually engages?

Deliver a crisp verdict: is cipher_rt_kv_alloc.c LIVE or DEAD in the shipped product, what pillar, and what the accelerate command truly does.`,
  { label: 'kv-accelerate-path', phase: 'Analyze', model: 'opus', schema: KV_SCHEMA });

const SELFTEST_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    depends_on_multitenant: { type: 'boolean' },
    chain: { type: 'string', description: 'cli -> router -> handlers -> gates chain with file:line' },
    gates_are_multitenant: { type: 'string', description: 'CERTAIN part: which gate criteria are multi-tenant' },
    substrate_exercised: { type: 'string', description: 'HANDLER-DEPENDENT part: do smoke handlers exercise the real multitenant substrate, or run demo traces with may13 disabled?' },
    mfu_exercised: { type: 'boolean' },
    tpw_exercised: { type: 'boolean' },
    explanation: { type: 'string' },
    citations: { type: 'array', items: { type: 'string' } },
  },
  required: ['depends_on_multitenant','gates_are_multitenant','substrate_exercised','mfu_exercised','tpw_exercised','explanation'],
};

const selftestAgent = agent(CTX + `

TASK: Definitively answer "does the green selftest (cipher selftest --quick) depend on multi-tenant code?" Trace the full chain and separate CERTAIN from HANDLER-DEPENDENT.

Read:
- cipher_v2_build/src/cipher_platform/cli.py cmd_selftest (~L28-43): the --quick manifest = [{regime:agent, agents:12, bmax:4}, {regime:density}] + off_byte_identical gate.
- cipher_v2_build/src/cipher_platform/router.py dispatch/run_manifest: agent->quick_handler, density->quick_handler.
- cipher_v2_build/pkgroot/etc/cipher/platform.json (regimes->handlers/quick_handlers/inject): agent quick=smoke_agent_scaled.py (inject CIPHER_RT_DISABLE_AUTO_INIT=1, CIPHER_VOLT=0), density quick=smoke_density_real.py.
- cipher_v2_build/src/cipher_platform/gates.py evaluate() + off_byte_identical().
- The quick handlers: cipher_v2_mock/smoke_agent_scaled.py, cipher_v2_mock/smoke_density_real.py, cipher_v2_mock/off_byte_identical_real.py.

Determine:
1. CERTAIN: selftest PASS criteria = OFF-byte-identical + agent(FAULT=0 + misroute) + density(KL=0). The agent + density regimes ARE the multi-tenant/co-residence pillar; compute(MFU) is NOT in the manifest and CIPHER_VOLT=0 forces TPW off. So gates-are-multitenant = ?
2. HANDLER-DEPENDENT: do smoke_agent_scaled.py / smoke_density_real.py actually exercise the multi-tenant SUBSTRATE (real co-residence / pager / kmod), or run fixed demo traces with may13 auto-init disabled? Read them and say honestly.
3. Set mfu_exercised / tpw_exercised (expected both false).`,
  { label: 'selftest-dependency', phase: 'Analyze', model: 'opus', schema: SELFTEST_SCHEMA });

const phase1 = await parallel([
  ...classifyThunks,
  () => triageBatch,
  () => entangleAgent,
  () => kvAgent,
  () => selftestAgent,
]);

const nC = classifyThunks.length;
const classifyResults = phase1.slice(0, nC).filter(Boolean);
const triage = phase1[nC];
const entanglement = phase1[nC + 1];
const kv = phase1[nC + 2];
const selftest = phase1[nC + 3];

const allRows = classifyResults.flatMap(r => (r && r.rows) ? r.rows : []);
if (triage && triage.rows) allRows.push(...triage.rows);
log('Classified ' + allRows.length + ' files/dirs across ' + classifyResults.length + ' batches + triage');

phase('Verify')

const kvCtx = kv ? JSON.stringify(kv).slice(0, 800) : 'n/a';
const entCtx = (entanglement && entanglement.chokepoints) ? JSON.stringify(entanglement.chokepoints.slice(0, 3)).slice(0, 900) : 'n/a';

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    claim: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED','REFUTED','PARTIAL','UNCERTAIN'] },
    evidence: { type: 'string', description: 'file:line proof' },
    correction: { type: 'string', description: 'if REFUTED/PARTIAL, the corrected classification' },
  },
  required: ['claim','verdict','evidence'],
};

const verifyClaims = [
  'The may13_* legacy family (cipher_rt_phase4/src/may13/*) is DEAD-in-product: either inert TUs (Makefile ~L285-294 no-runtime-effect) or disabled by CIPHER_RT_DISABLE_AUTO_INIT=1 in the product regimes. VERIFY: (a) read the Makefile inert-TU comments, (b) grep for the may13 auto-init/static-registration and how CIPHER_RT_DISABLE_AUTO_INIT gates it, (c) check whether ANY may13 symbol is reached at runtime in the agent/density product regimes. Distinguish linked-inert from linked-and-fires-only-in-legacy-single-tenant-path.',
  'Koopman (cipher_rt_koopman_engine.cpp + src/may13/cipher_edmd*.cpp + cipher_koopman_runtime.cpp) NEVER ACTUATES in the engine: handled=0/registered=0, capture-illegal (cudaHostAlloc in edmd_live.cpp). VERIFY the handled=0/registered=0 path and find the cudaHostAlloc capture-illegal call (cite file:line). Is it MFU-DEAD or only-legacy-live?',
  'cipher_rt_arbitrate.c is DEAD/not-linked: retired per Makefile:52-54, arbitrate.o absent on disk, .c unbuilt (+ a cipher_rt_arbitrate.c.pre_B7 backup). VERIFY and confirm no runtime path reaches it.',
  'Per the V.0 scorecard these were DORMANT: green-ctx (cipher_rt_green_ctx.c), audit (cipher_rt_audit.c), ring-write (cipher_rt_ring_write.c), sense-transition (cipher_rt_sense_transition.c). VERIFY against actual code: registered/called at runtime in the product path, or dormant? Give CONFIRMED-dormant or REFUTED-live with file:line for each. ALSO verify the converse: cipher_rt_sm_packer.c FIRES (scorecard counter 61278) => LIVE not dead.',
  'cipher_rt_kv_alloc.c is DEAD in the SHIPPED .so (kv_alloc.o NOT in OBJS L55-107 even though kv_alloc.o exists on disk), so CIPHER_KV_ALLOC is a no-op in the deployed product; the real accelerate KV mechanism is CIPHER_KVDEDUP via cipher_kmod/cipher_kvdedup.c + cipher_vllm_plugin. VERIFY whether kv_alloc.o is truly absent from OBJS and whether any OTHER linked object reads CIPHER_KV_ALLOC.',
  'ENTANGLEMENT: cipher_rt_matmul_dispatch.c + cipher_rt_cublas_shim.c form a single GEMM-interception chokepoint through which BOTH FP8[MFU] and Marlin[MFU/density] (and bf16 passthrough) route — cannot change for one pillar without affecting the others. VERIFY by tracing fp8_actuator and marlin_actuator into matmul_dispatch (cite file:line of the shared dispatch).',
  'ENTANGLEMENT: the engine that keeps the GPU fed (cipher_inc4.py / cipher_engine*.py + cipher_rt_pager.c) is the shared substrate where multitenant co-residence, MFU throughput, and density meet (cross-tenant batching/coalescing). VERIFY the engine role: does the same batching/residency path serve throughput AND multi-model co-residence? cite file:line.',
];

const verdicts = (await parallel(verifyClaims.map((c, i) => () =>
  agent(CTX + `

ADVERSARIAL VERIFICATION. Try to REFUTE the following claim by reading the actual code. Default to skepticism — if you cannot find positive proof, say UNCERTAIN. Cite file:line.

CROSS-CHECK CONTEXT from the analysis phase:
- kv-accelerate finding: ` + kvCtx + `
- entanglement finding (first chokepoints): ` + entCtx + `

CLAIM #` + (i + 1) + `:
` + c,
    { label: 'verify-' + (i + 1), phase: 'Verify', model: 'opus', schema: VERDICT_SCHEMA })
)).filter(Boolean);

return {
  rows: allRows,
  entanglement: entanglement,
  kv: kv,
  selftest: selftest,
  triage_notes: triage ? triage.batch_notes : '',
  verdicts: verdicts,
  batch_notes: classifyResults.map(r => r && r.batch_notes).filter(Boolean),
};
