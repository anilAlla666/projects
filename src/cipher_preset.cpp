// CIPHER profile selector — one env knob to enable a curated op set.
//
// Reads CIPHER_PROFILE first; falls back to CIPHER_PRESET (legacy alias).
//
// Profiles:
//   production — customer-shipping default. All safety / audit / billing /
//                policy ops on. Heavy observers OFF (WEIGHT_SHARE,
//                SPECULATE_CHECK, GRAPH_WARMUP, PIPELINE_STAGES). FP8
//                auto-detected.
//   safety     — compliance-only: GUARD + DETERMINISM + RECEIPT + COMPLY +
//                DVFS + LOOP + THERMAL_FEEDBACK.
//   perf       — substitution + caching only: GRAPH_WARMUP + PREDICT +
//                KV_REDIRECT + FUSION_KERNELS + MARLIN + SUBSTITUTE_V2.
//   full       — kitchen sink: every op CIPHER ships.
//
// Operator-set env vars always win (setenv called with overwrite=0).

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
std::atomic<int> g_initialized{0};

inline void on(const char* k) { ::setenv(k, "on", 0); }

void apply_perf() {
    on("CIPHER_GRAPH_WARMUP");
    on("CIPHER_PREDICT");
    on("CIPHER_PREDICT_PREALLOC");
    on("CIPHER_KV_REDIRECT");
    on("CIPHER_FUSION_KERNELS");
    on("CIPHER_MARLIN_SHIM");
    on("CIPHER_SUBSTITUTE_V2");
}

void apply_safety() {
    on("CIPHER_GUARD");
    on("CIPHER_DETERMINISM");
    on("CIPHER_RECEIPT");
    on("CIPHER_COMPLY");
    on("CIPHER_LOOP");
    on("CIPHER_DVFS");
    on("CIPHER_THERMAL_FEEDBACK");
    on("CIPHER_FAIRNESS");
    on("CIPHER_CARBON");
    on("CIPHER_THERMOSTAT");
}

void apply_telemetry() {
    on("CIPHER_NCCL_TUNER");
    on("CIPHER_TOPOLOGY");
    on("CIPHER_PIPELINE_STAGES");
    on("CIPHER_WEIGHT_SHARE");
    on("CIPHER_SENSE");
    on("CIPHER_SHIELD");
    on("CIPHER_ARBITRATE");
    on("CIPHER_CONTINUITY");
    on("CIPHER_SPECULATE_CHECK");
}

void apply_diagnostics() {
    on("CIPHER_FLOW_RECORDER");
    on("CIPHER_FLOW_MATCH");
    on("CIPHER_FLOW_SUBSTITUTE");
}
} // namespace

extern "C" int cipher_preset_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel)) return 0;
    // CIPHER_PROFILE is the customer-facing knob; CIPHER_PRESET is a
    // legacy alias retained for backward compatibility.
    const char* p = std::getenv("CIPHER_PROFILE");
    if (!p || !*p) p = std::getenv("CIPHER_PRESET");
    if (!p || !*p) return 0;
    if (std::strcmp(p, "perf") == 0) {
        apply_perf();
        std::fprintf(stderr, "[CIPHER PRESET] = perf (fusion + marlin + graph + predict)\n");
        return 1;
    }
    if (std::strcmp(p, "safety") == 0) {
        apply_safety();
        std::fprintf(stderr, "[CIPHER PRESET] = safety (guard + determinism + receipt + comply + dvfs)\n");
        return 1;
    }
    if (std::strcmp(p, "production") == 0) {
        // V1 GA — neocloud-shipping default. Every actuator + observer
        // we trust on every workload class. Excludes Koopman / KV-redirect
        // V3 / attention-FSM substitution (deferred to "experimental"
        // until M3 Phase 0 evidence proves them on real workloads).
        // Operator-set CIPHER_* env vars always win (setenv overwrite=0).
        on("CIPHER_GUARD");
        on("CIPHER_RECEIPT");
        on("CIPHER_COMPLY");
        on("CIPHER_AUDIT");                  // tamper-evident event log
        on("CIPHER_DETERMINISM");
        on("CIPHER_NCCL_TUNER");
        on("CIPHER_NCCL_V4");                // production NCCL tuner (M1.T1 verified)
        on("CIPHER_TOPOLOGY");
        on("CIPHER_DVFS");
        on("CIPHER_LOOP");
        on("CIPHER_PREDICT");
        on("CIPHER_PREDICT_PREALLOC");
        on("CIPHER_KV_REDIRECT");            // V1 (memcpy-only path is safe)
        on("CIPHER_FAIRNESS");
        on("CIPHER_ARBITRATE");              // intra-GPU SM rebalance (M1.T4 distributor)
        on("CIPHER_CARBON");
        on("CIPHER_THERMOSTAT");
        on("CIPHER_FUSION_KERNELS");
        on("CIPHER_FLOW_RECORD");            // pointer-flow recorder
        on("CIPHER_FLOW_MATCH");             // RMSNorm anchor matcher
        on("CIPHER_FLOW_SUBSTITUTE");        // dispatches fused when X/W/Y available
        on("CIPHER_MARLIN_SHIM");
        on("CIPHER_SUBSTITUTE_V2");
        on("CIPHER_GRAPH_WARMUP");
        on("CIPHER_SENSE");
        on("CIPHER_SHIELD");
        on("CIPHER_PERSIST_ENGINE");
        on("CIPHER_PULSE");
        on("CIPHER_SUSTAIN");
        on("CIPHER_HIBERNATE");
        on("CIPHER_TRACE");                  // OTel-compatible event spans
        on("CIPHER_PIPELINE");               // graph-capture coordination
        on("CIPHER_CONTINUITY");             // session checkpoint bookkeeping
        on("CIPHER_WORKLOAD_DETECT");        // 24-class auto-classifier
        on("CIPHER_THERMAL_FEEDBACK");       // NVML poll thread
        on("CIPHER_PARTITION_ROUTER");       // L2 partition steering
        on("CIPHER_VMM");                    // VMM pool for swap
        // FP8 auto-detected (auto-disables on torch with native FP8).
        ::setenv("CIPHER_FP8_COMPUTE", "auto", 0);
        std::fprintf(stderr, "[CIPHER PROFILE] = production "
            "(v1 GA: all proven actuators + full observability + "
            "billing/policy/audit; experimental Koopman/KV-V3/attn-FSM OFF)\n");
        return 1;
    }
    if (std::strcmp(p, "cloud") == 0) {
        // Neocloud-tenant profile: production + multi-tenant + signed billing.
        // Same as "production" but with explicit reminders that
        // CIPHER_TENANT_ID, CIPHER_POLICY_PATH, and CIPHER_RECEIPT_KEY
        // should be set by the orchestrator.
        if (!std::getenv("CIPHER_TENANT_ID")) {
            std::fprintf(stderr, "[CIPHER PROFILE=cloud] WARN: "
                "CIPHER_TENANT_ID is unset — per-tenant policy and billing "
                "won't be properly attributed.\n");
        }
        if (!std::getenv("CIPHER_POLICY_PATH")) {
            ::setenv("CIPHER_POLICY_PATH", "/etc/cipher/policy.json", 0);
        }
        // Fall through to production (recursive call would reset
        // g_initialized; just inline the same env setup).
        on("CIPHER_GUARD"); on("CIPHER_RECEIPT"); on("CIPHER_COMPLY");
        on("CIPHER_AUDIT"); on("CIPHER_DETERMINISM"); on("CIPHER_NCCL_TUNER");
        on("CIPHER_NCCL_V4"); on("CIPHER_TOPOLOGY"); on("CIPHER_DVFS");
        on("CIPHER_LOOP"); on("CIPHER_PREDICT"); on("CIPHER_PREDICT_PREALLOC");
        on("CIPHER_KV_REDIRECT"); on("CIPHER_FAIRNESS"); on("CIPHER_ARBITRATE");
        on("CIPHER_CARBON"); on("CIPHER_THERMOSTAT"); on("CIPHER_FUSION_KERNELS");
        on("CIPHER_FLOW_RECORD"); on("CIPHER_FLOW_MATCH"); on("CIPHER_FLOW_SUBSTITUTE");
        on("CIPHER_MARLIN_SHIM"); on("CIPHER_SUBSTITUTE_V2"); on("CIPHER_GRAPH_WARMUP");
        on("CIPHER_SENSE"); on("CIPHER_SHIELD"); on("CIPHER_PERSIST_ENGINE");
        on("CIPHER_PULSE"); on("CIPHER_SUSTAIN"); on("CIPHER_HIBERNATE");
        on("CIPHER_TRACE"); on("CIPHER_PIPELINE"); on("CIPHER_CONTINUITY");
        on("CIPHER_WORKLOAD_DETECT"); on("CIPHER_THERMAL_FEEDBACK");
        on("CIPHER_PARTITION_ROUTER"); on("CIPHER_VMM");
        on("CIPHER_DISTRIBUTOR");           // multi-tenant auto-distribution
        ::setenv("CIPHER_FP8_COMPUTE", "auto", 0);
        // Cloud-specific: enable Prometheus metrics endpoint by default
        if (!std::getenv("CIPHER_METRICS_PORT")) {
            ::setenv("CIPHER_METRICS_PORT", "9091", 0);
        }
        std::fprintf(stderr, "[CIPHER PROFILE] = cloud "
            "(production + per-tenant policy + Prometheus metrics on :9091)\n");
        return 1;
    }
    if (std::strcmp(p, "experimental") == 0) {
        // production + Koopman / KV-redirect V3 / attention-FSM /
        // driver-level deref. For design partners validating the v2
        // research bets. NOT for production deployments — these paths
        // have known correctness gaps documented in source.
        // Inline the production set:
        on("CIPHER_GUARD"); on("CIPHER_RECEIPT"); on("CIPHER_COMPLY");
        on("CIPHER_AUDIT"); on("CIPHER_DETERMINISM"); on("CIPHER_NCCL_TUNER");
        on("CIPHER_NCCL_V4"); on("CIPHER_TOPOLOGY"); on("CIPHER_DVFS");
        on("CIPHER_LOOP"); on("CIPHER_PREDICT"); on("CIPHER_PREDICT_PREALLOC");
        on("CIPHER_KV_REDIRECT"); on("CIPHER_FAIRNESS"); on("CIPHER_ARBITRATE");
        on("CIPHER_CARBON"); on("CIPHER_THERMOSTAT"); on("CIPHER_FUSION_KERNELS");
        on("CIPHER_FLOW_RECORD"); on("CIPHER_FLOW_MATCH"); on("CIPHER_FLOW_SUBSTITUTE");
        on("CIPHER_MARLIN_SHIM"); on("CIPHER_SUBSTITUTE_V2"); on("CIPHER_GRAPH_WARMUP");
        on("CIPHER_SENSE"); on("CIPHER_SHIELD"); on("CIPHER_PERSIST_ENGINE");
        on("CIPHER_PULSE"); on("CIPHER_SUSTAIN"); on("CIPHER_HIBERNATE");
        on("CIPHER_TRACE"); on("CIPHER_PIPELINE"); on("CIPHER_CONTINUITY");
        on("CIPHER_WORKLOAD_DETECT"); on("CIPHER_THERMAL_FEEDBACK");
        on("CIPHER_PARTITION_ROUTER"); on("CIPHER_VMM");
        on("CIPHER_DISTRIBUTOR");           // multi-tenant auto-distribution
        ::setenv("CIPHER_FP8_COMPUTE", "auto", 0);
        // Experimental adds:
        on("CIPHER_EDMD_LIVE");              // per-shape EDMD calibration
        on("CIPHER_EDMD_HOOK");              // wires EDMD into cuBLAS shim
        on("CIPHER_KV_COMPRESS");            // 2-bit KV compression (V3)
        on("CIPHER_ATTN_KOOPMAN");           // attention FSM
        on("CIPHER_FLOW_USE_ATEN_DEREF");    // libtorch ABI deref (M1.T2 partial)
        on("CIPHER_GRAPH_INSPECT");          // graph capture introspection
        // CIPHER_USE_CACHE=0 keeps Koopman cache safe (content-hash mandatory)
        ::setenv("CIPHER_USE_CACHE", "0", 0);
        // Default Koopman quality gate at 0.999 — research can lower it
        ::setenv("CIPHER_KOOPMAN_MIN_ENERGY", "0.999", 0);
        std::fprintf(stderr, "[CIPHER PROFILE] = experimental "
            "(production + Koopman + KV-V3 + attn-FSM + libtorch deref; "
            "research-grade — known correctness gaps, not for production)\n");
        return 1;
    }
    if (std::strcmp(p, "full") == 0) {
        apply_perf();
        apply_safety();
        apply_telemetry();
        apply_diagnostics();
        std::fprintf(stderr, "[CIPHER PRESET] = full (every op CIPHER ships)\n");
        return 1;
    }
    std::fprintf(stderr, "[CIPHER PRESET] unknown value '%s' — ignored. "
                         "Valid: perf | safety | production | cloud | "
                         "experimental | full\n", p);
    return 0;
}

// Run at the LOWEST priority (= earliest constructor) so that the setenv
// calls finish before any per-op constructor reads its env knob.
__attribute__((constructor(101)))
static void cipher_preset_autoinit() { cipher_preset_init(); }
