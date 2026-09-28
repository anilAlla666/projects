/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_workload_detect.cpp -- K.1 Workload Classifier implementation.
 *
 * Architecture (per V1_CAPABILITY_AUDIT Section D K.1):
 *
 *   Hot-path observe_* functions accumulate signals into a lock-free
 *   signal aggregator. Per-fn-pointer kernel-name cache resolves cuFuncGetName
 *   exactly once per unique fn ptr; subsequent observations reuse the cached
 *   name-class bitfield (Memory #25 <10ns budget).
 *
 *   Re-classification runs every CIPHER_WORKLOAD_RECLASSIFY_INTERVAL=100
 *   observe_launch invocations (probabilistic trigger from the hot path)
 *   OR on significant state change (NCCL first seen, capture begin first
 *   seen, multi-process first detected). Re-classification reads /proc
 *   for sibling-process count -- off the hot path.
 *
 *   Memory #9 framework-agnostic: signals are driver-level only -- kernel
 *   name fragments, cuBLAS GEMM shapes/dtypes, cudaMalloc sizes, capture
 *   events, libnccl presence. No PyTorch / transformers / vLLM type
 *   introspection.
 *
 *   Memory #20 ABI-additive: existing dispatch paths unchanged. observe_*
 *   functions ADDED at existing intercept sites; nothing existing is
 *   modified.
 *
 *   K.1 scope discipline (advisor #2 from K.1 design call): A1
 *   AGENTIC_MULTI_TENANT does NOT require same_model_across_tenants in
 *   K.1 -- NR 27 REGISTER_MODEL is gated off in CDI patch (B.6''.9.1
 *   crash) and there is no other observable same-model signal at this
 *   substrate layer. same_model_across_tenants flag is always false in
 *   K.1; future substep with W.6 NR 27 userspace fix will populate it.
 *
 *   K.1 Step 6-8 fix wave (Bugs #1 + #2 + decision-tree reorder, Bug #3 deferred to W.6):
 *
 *   Bug #1: long_context_detected was derived from cuBLAS max_m / max_n.
 *     Llama-3 intermediate_size=14336 triggers max_k=14336 every forward
 *     pass regardless of prompt length -- classifier wrongly routed
 *     Llama-3 short-prompt vLLM to C1_RAG_LONG_CONTEXT. Fix: rename to
 *     wide_matmul_detected (semantic honesty about what is observed) and
 *     derive a NEW long_context_detected from kv_cache_alloc_pattern AND
 *     max_n > LONG_CTX_SEQ_THRESH (32K).
 *
 *   Bug #2: has_nccl fired on libnccl-loaded-but-unused. vLLM 0.21 loads
 *     libnccl as DT_NEEDED even at TP=1 (single-GPU inference). Fix: drop
 *     nccl_lib_loaded from has_nccl; only count actual ncclAllReduce /
 *     ReduceScatter / etc kernel-name observations. Library presence
 *     remains an observable but is no longer a classification gate.
 *
 *   Decision tree reorder: A2_CONTINUOUS_BATCHED now has priority over
 *     A1_AGENTIC_MULTI_TENANT. has_capture (cuGraphLaunch observed) wins
 *     because vLLM defaults to cudagraph capture in 0.21+, and that
 *     dominates the multi_tenant false positive caused by EngineCore
 *     worker subprocess pollution.
 *
 *   Bug #3 deferred to W.6: multi_tenant detection via /proc walk has two
 *     known false positives:
 *       (a) EngineCore worker subprocess (single-tenant vLLM but
 *           libcipher_rt mapped in both main + worker -> count >= 2)
 *       (b) Container-namespaced /proc cannot see sibling containers
 *           (false negative for 4-container multi-tenant deployment)
 *     The architecturally correct fix is a kmod-side process registry
 *     keyed off CIPHER_REGISTER_MODEL (NR 27), which is gated off in
 *     CDI patch due to B.6''.9.1 userspace crash. W.6 substep closes
 *     both: NR 27 userspace fix + process registry usage. In K.1,
 *     multi_tenant is a SOFT signal; A1 also requires !has_capture so
 *     graphs-enabled vLLM workloads route to A2 not A1.
 */

#include "cipher_workload_detect.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <mutex>
#include <unordered_set>
#include <thread>
#include <chrono>
#include <pthread.h>

extern "C" {
#include "cipher_v2_internal.h"   /* cipher_log */
#include "cipher_rt_volt.h"       /* W.1: cipher_rt_volt_classifier_poll */

extern "C" int cipher_rt_machete_intercept_init(void);  /* W.2 Sub-step 7 retry */
extern "C" int cipher_rt_attn_6pattern_init(void);       /* W.5 retry */
extern "C" unsigned long cipher_rt_attn_p1_intercepts(void);
extern "C" unsigned long cipher_rt_attn_p2_intercepts(void);
extern "C" unsigned long cipher_rt_attn_p5_intercepts(void);
extern "C" unsigned long cipher_rt_attn_p6_intercepts(void);
extern "C" uint64_t cipher_workload_model_fingerprint(void);  /* W.6 sub-B fwd decl */
#include "cipher_rt_coresidence.h"                            /* W.6 sub-C cohort client */
#include "cipher_rt_pool.h"                                    /* W.4a coalesce-eligibility */
extern "C" unsigned long cipher_rt_marlin_calls_bf16_observed(void);
extern "C" unsigned long cipher_rt_marlin_calls_bf16_substituted(void);
extern "C" unsigned long cipher_rt_machete_intercepts(void);
extern "C" unsigned long cipher_rt_machete_substituted(void);
extern "C" unsigned long cipher_rt_koopman_bf16_observed(void);  /* W.3 SS1 */
extern "C" unsigned long cipher_rt_koopman_bf16_max_k(void);
extern "C" unsigned long cipher_rt_koopman_bf16_max_n(void);
}

namespace {

/* === Constants === */

constexpr uint32_t RECLASSIFY_INTERVAL = 100u;
constexpr uint32_t FN_CACHE_SLOTS      = 1024u;   /* open-addressing; ~5ns p50 probe */
constexpr uint32_t FN_CACHE_PROBE_MAX  = 4u;
constexpr uint64_t LARGE_ALLOC_BYTES   = 1ULL << 30;   /* >= 1 GiB single = weights */
constexpr uint64_t SMALL_ALLOC_BYTES   = 1ULL << 20;   /* < 1 MiB = KV cache or scratch */
constexpr uint64_t MEDIUM_ALLOC_MIN_BYTES = 1ULL << 18; /* 256 KiB lower edge for KV-block alloc class */
constexpr uint64_t MEDIUM_ALLOC_MAX_BYTES = 1ULL << 24; /* 16 MiB upper edge */
constexpr uint32_t WIDE_MATMUL_DIM_THRESH = 8192u;     /* renamed from LONG_CTX_DIM_THRESH (Bug #1) */
constexpr uint32_t LONG_CTX_SEQ_THRESH    = 32768u;    /* max_n threshold for derived long_ctx */
constexpr uint64_t KV_CACHE_PATTERN_THRESH = 100ULL;   /* medium-alloc count to fire pattern */
constexpr uint32_t LARGE_BATCH_THRESH  = 32u;
constexpr uint64_t IDLE_OBS_PER_SEC_THRESH = 50ULL;   /* K.1.6: <50 obs/s => idle */
constexpr uint64_t IDLE_MIN_ELAPSED_NS = 5ULL * 1000000000ULL; /* 5s grace period */
/* K.1.6 v2 recent-idle gate: if no observe in last IDLE_INACTIVITY_NS, force
 * UNKNOWN. Catches warmup-burst-then-idle (E2/E3) where lifetime-rate gate
 * misses because the cumulative obs/s sits above threshold thanks to burst. */
constexpr uint64_t IDLE_INACTIVITY_NS = 2ULL * 1000000000ULL; /* 2s sliding window */
/* K.1.6 v3 sustained-activity guard: don't downgrade a workload whose active
 * span (last_observe - first_observe) exceeds MIN_SUSTAINED_ACTIVE_NS. A real
 * inference workload runs sustained for tens of seconds; E2/E3-style "load +
 * brief burst" only span ~10-15s. 20s threshold separates them empirically.
 *
 * v6 (2026-05-27) replaces span-based guard with KNOWN-classify count guard:
 * empirically E3 emits 14 KNOWN classifies during load+1tok, P3 (5-iter inf)
 * emits 576. Span alone overlaps (E3 ~13s vs P3 ~15s), but emission count
 * separates cleanly. Threshold 50 = ~5s of sustained inference at the 100-
 * launch periodic-classify interval. */
constexpr uint64_t MIN_HIGH_CONF_CLASSIFY_COUNT = 50ULL;

/* K.1.5 Step 0 Bug #5 fix v4: registration-time host_ptr -> device_name map.
 * Open-addressed, linear-probe, lock-free.
 *
 * v4-iter1 used 4096 slots; vLLM-AWQ workload registered ~31K unique kernels
 * (template specializations across cuBLAS+cuDNN+Flash+pytorch+vllm_C). Map
 * filled at 4096 -> 26981 full_skips. v4-iter2 bumps to 65536 slots (~512KB)
 * to fit the empirical registration count + headroom; linear-probe distance
 * increased to 16 for higher load factor tolerance. */
constexpr uint32_t REG_MAP_SLOTS = 65536u;
constexpr uint32_t REG_MAP_PROBE_MAX = 16u;

/* Kernel-name-class bitfield (set on first observation of an unique fn ptr). */
constexpr uint32_t KCB_NONE       = 0u;
constexpr uint32_t KCB_MACHETE    = 1u << 0;   /* AWQ/GPTQ via Machete kernels */
constexpr uint32_t KCB_AWQ        = 1u << 1;   /* awq_* kernel names */
constexpr uint32_t KCB_GPTQ_MARLIN= 1u << 2;   /* gptq_marlin / Marlin AWQ kernels */
constexpr uint32_t KCB_FLASH_ATTN = 1u << 3;   /* flash_attn / fmha / SDPA fwd */
constexpr uint32_t KCB_RMS_NORM   = 1u << 4;
constexpr uint32_t KCB_SILU       = 1u << 5;
constexpr uint32_t KCB_GELU       = 1u << 6;
constexpr uint32_t KCB_NCCL       = 1u << 7;   /* ncclAllReduce / all_reduce */
constexpr uint32_t KCB_BACKWARD   = 1u << 8;   /* _bwd / backward / gradient */
constexpr uint32_t KCB_OPTIMIZER  = 1u << 9;   /* Adam / SGD / optimizer */
constexpr uint32_t KCB_REDUCE     = 1u << 10;
constexpr uint32_t KCB_RESOLVED   = 1u << 31;  /* sentinel: cache entry is filled */

/* CUDA driver typedefs (resolved lazily; do not pull cuda.h). */
typedef int (*pf_cuFuncGetName)(const char **, void *);
typedef int (*pf_cuFuncGetAttribute)(int *, int, void *);
typedef int (*pf_cuKernelGetName)(const char **, void *);  /* CUDA 12.4+ alternative API */

/* === Per-fn-pointer cache (open addressing, 1024 slots, 4-probe) === */

struct FnCacheEntry {
    std::atomic<const void *> fn;       /* nullptr = empty */
    std::atomic<uint32_t>     bits;     /* KCB_* mask; KCB_RESOLVED set when ready */
};

alignas(64) FnCacheEntry g_fn_cache[FN_CACHE_SLOTS];

/* K.1.5 Step 0 registration map: host_ptr -> device_name (process-lifetime
 * stable pointer; no strdup). Lock-free open-addressing; linear probe up
 * to REG_MAP_PROBE_MAX slots. */
struct RegMapEntry {
    std::atomic<const void *> host_ptr;   /* nullptr = empty slot */
    std::atomic<const char *> name;
};
alignas(64) RegMapEntry g_reg_map[REG_MAP_SLOTS];
std::atomic<uint64_t> g_reg_map_inserts{0};
std::atomic<uint64_t> g_reg_map_full_skips{0};
std::atomic<uint64_t> g_reg_map_calls{0};
std::atomic<uint64_t> g_reg_map_dup_returns{0};

/* K.1.5 Step 1.5: 4 per-site provenance sets. Populated at registration time;
 * read at unresolved-log site (cold path). std::unordered_set + mutex
 * because hot-path access is rare (only on negative-cache poisoning, which
 * is once per unique unresolvable fn). ~30K entries max per workload. */
static std::mutex g_provenance_mu;
static std::unordered_set<const void *> g_prov_cuda_register_fn;
static std::unordered_set<const void *> g_prov_cu_module_get_fn;
static std::unordered_set<const void *> g_prov_cu_library_get;
static std::unordered_set<const void *> g_prov_cu_kernel_get_fn;
std::atomic<int> g_log_registered_inited{0};
std::atomic<int> g_log_registered_on{0};

static inline uint32_t reg_map_hash_slot(const void *host_ptr) {
    uint64_t h = (uintptr_t)host_ptr * 0x9E3779B97F4A7C15ULL;
    return (uint32_t)(h >> 32) & (REG_MAP_SLOTS - 1);
}

/* Look up name for a host fn ptr; nullptr if not registered. */
static inline const char *reg_map_lookup(const void *host_ptr) {
    if (!host_ptr) return nullptr;
    uint32_t slot = reg_map_hash_slot(host_ptr);
    for (uint32_t i = 0; i < REG_MAP_PROBE_MAX; ++i) {
        RegMapEntry &e = g_reg_map[(slot + i) & (REG_MAP_SLOTS - 1)];
        const void *seen = e.host_ptr.load(std::memory_order_acquire);
        if (seen == host_ptr) return e.name.load(std::memory_order_relaxed);
        if (seen == nullptr) return nullptr;  /* empty slot = not registered */
    }
    return nullptr;
}

/* === Signal aggregator === */

struct WorkloadSignals {
    /* Launch-level counters */
    std::atomic<uint64_t> total_observations;
    std::atomic<uint64_t> total_classifications;
    std::atomic<uint64_t> launches_since_classify;

    /* Kernel-class counters (incremented by observe_launch on cache hit) */
    std::atomic<uint64_t> machete_count;
    std::atomic<uint64_t> awq_count;
    std::atomic<uint64_t> gptq_marlin_count;
    std::atomic<uint64_t> flash_attn_count;
    std::atomic<uint64_t> nccl_count;
    std::atomic<uint64_t> backward_count;
    std::atomic<uint64_t> optimizer_count;
    std::atomic<uint64_t> rms_norm_count;
    std::atomic<uint64_t> silu_gelu_count;

    /* GEMM shape stats (cuBLAS observe) */
    std::atomic<uint64_t> gemm_total;
    std::atomic<uint64_t> gemm_b1_count;          /* n (batch in PyTorch row-major cuBLAS view) <= 1 */
    std::atomic<uint64_t> gemm_large_batch_count; /* n > LARGE_BATCH_THRESH */
    std::atomic<uint32_t> max_m;
    std::atomic<uint32_t> max_n;
    std::atomic<uint32_t> max_k;
    std::atomic<uint64_t> gemm_int4_dtype_count;  /* Atype/Btype in {3,4,5,6} INT8 hint */
    /* W.1 (2026-05-27): small-batch GEMM telemetry (n<=32 cuBLAS view).
     * Decode workloads dominate this counter; prefill rarely uses small n.
     * Used together with gemm_b1_count for the bw_bound_regime decision. */
    std::atomic<uint64_t> gemm_small_count;
    /* W.1 v3 (2026-05-27): prefill-shape GEMM count (n>4096 cuBLAS view).
     * vLLM's startup profiling spikes max_n to 16384 for any model, so a
     * lifetime max_n > 4096 gate misfires. Ratio gemm_prefill / gemm_total
     * > 10% catches workloads where prefill dominates (N1) while ignoring
     * the one-shot profiling spike on decode workloads (P1/P2 et al). */
    std::atomic<uint64_t> gemm_prefill_count;
    /* W.2 (2026-05-27): bf16 GEMM count. CUDA_R_16BF == 14. Drives the
     * bf16_weights_detected derived signal (ratio > 10% of GEMMs). */
    std::atomic<uint64_t> gemm_bf16_count;

    /* Allocation stats */
    std::atomic<uint64_t> alloc_total_bytes;
    std::atomic<uint64_t> alloc_large_count;     /* >= LARGE_ALLOC_BYTES (weights class) */
    std::atomic<uint64_t> alloc_small_count;     /* < SMALL_ALLOC_BYTES */
    std::atomic<uint64_t> alloc_medium_count;    /* [MEDIUM_ALLOC_MIN..MAX] KV-block class */
    std::atomic<uint64_t> alloc_call_count;

    /* Event flags (set-once style; relaxed-atomic boolean) */
    std::atomic<int> capture_seen;
    std::atomic<int> nccl_lib_loaded;
    std::atomic<int> initialized;

    /* Process-count estimate (sampled at classify time, not hot path) */
    std::atomic<uint32_t> concurrent_processes_estimate;

    /* Latched profile (RCU-style snapshot). Two slots; even index = current. */
    cipher_workload_profile profile_a;
    cipher_workload_profile profile_b;
    std::atomic<int> profile_current;     /* 0 = profile_a, 1 = profile_b */

    /* K.1.5 Bug #5 diagnostic: disambiguate why lookup_fn_bits returns 0.
     * Pre-18-capture targeted probe per advisor 2026-05-27. Counters are
     * monotone; never reset. Probe via cipher_workload_fn_cache_* exports. */
    std::atomic<uint64_t> fn_cache_hits;                 /* returned non-empty bits (cached or fresh) */
    std::atomic<uint64_t> fn_cache_resolved_no_match;    /* getname OK, classify returned 0 */
    std::atomic<uint64_t> fn_cache_resolution_failed;    /* getname returned err OR null name */
    std::atomic<uint64_t> fn_cache_probe_overflow;       /* 4-slot probe distance exceeded */
    std::atomic<uint64_t> fn_cache_resolved_via_reg;     /* K.1.5 Step 0: registration-map hit */
    std::atomic<uint64_t> fn_cache_negative_cached;      /* K.1.5 Step 0 Sub-step 1: cached failed resolution */
    std::atomic<uint64_t> fn_cache_recovered_via_retry;  /* K.1.5 Step 1.5: retry chain recovered a name */
    std::atomic<uint64_t> fn_cache_resolved_via_cu_kernel_get_name; /* K.1.5 Step 1.6: primary CUkernel path */
    /* K.1.6 (2026-05-27) Sub-step 1.1: idle-rate gate. Compute
     * observations_per_second by storing first_obs_ns at startup and
     * dividing total_observations by elapsed wall time at classify
     * boundary. If rate < IDLE_THRESH, classifier emits UNKNOWN
     * regardless of other signals (fixes E2 false-positive A4 on
     * model-loading-only workloads per K.1.5 Section F). */
    std::atomic<uint64_t> first_obs_ns;
    /* K.1.6 Sub-step 1.1 v2 (2026-05-27): tracks LAST observe to detect
     * recent idleness. The lifetime-rate gate (first_obs_ns) catches steady
     * low activity but misses warmup-burst-then-idle patterns. Updated on
     * every observe_launch (relaxed store; hot path budget). Read at
     * classify-time AND from the cipher_workload_atexit_classify destructor
     * to demote class to UNKNOWN if last activity > IDLE_INACTIVITY_NS. */
    std::atomic<uint64_t> last_observe_ns;
    /* K.1.6 v6 (2026-05-27) high-confidence emission counter: increments on
     * each classify that emits a KNOWN class with confidence > 500.  Guards
     * the recent-idle downgrade against over-firing on real workloads (P3
     * accumulates 576 emissions before idle; E3 only 14). */
    std::atomic<uint64_t> high_conf_emits;
};

alignas(64) WorkloadSignals g_signals;

/* === Resolved cuFuncGetName ============================================== */

std::atomic<pf_cuFuncGetName> g_cuFuncGetName{nullptr};
std::atomic<pf_cuKernelGetName> g_cuKernelGetName{nullptr};      /* Step 1.5 retry */
std::atomic<pf_cuFuncGetAttribute> g_cuFuncGetAttribute{nullptr}; /* Step 1.5 validity probe */
std::atomic<int> g_cu_funcname_resolve_attempted{0};

bool resolve_cu_funcname(void) {
    /* K.1.5 advisor outcome (a) fix 2026-05-27 v2: prior code set
     * g_cu_funcname_resolve_attempted on first call regardless of success.
     * init-time resolve attempt happens BEFORE libcuda is fully loaded ->
     * first attempt fails -> flag set -> ALL subsequent attempts early-out
     * even after libcuda becomes available. Bug #5 fix v1 changed dlsym
     * path but kept the flag bug; fix v2 removes the negative-cache flag.
     * Each call retries dlsym(RTLD_DEFAULT) until SUCCESS (then caches).
     * Per-launch cost when unresolved: ~100-1000ns dlsym call; once
     * resolved: ~2ns atomic load. For workloads with many launches the
     * pre-resolution cost is amortized over thousands of launches. */
    if (g_cuFuncGetName.load(std::memory_order_relaxed) != nullptr) return true;
    /* Marker still set for diagnostic visibility, but no longer gates retry. */
    g_cu_funcname_resolve_attempted.store(1, std::memory_order_relaxed);

    /* Diagnostic: log each resolve attempt to surface which is failing. */
    static std::atomic<int> g_resolve_diag_count{0};
    int diag_n = g_resolve_diag_count.fetch_add(1, std::memory_order_relaxed);
    int diag_on = (diag_n < 3);  /* log first 3 attempts then quiet */

    void *sym = dlsym(RTLD_DEFAULT, "cuFuncGetName");
    if (diag_on) cipher_log("RESOLVE-DIAG-%d: dlsym(RTLD_DEFAULT,cuFuncGetName)=%p", diag_n, sym);
    if (!sym) {
        sym = dlsym(RTLD_NEXT, "cuFuncGetName");
        if (diag_on) cipher_log("RESOLVE-DIAG-%d: dlsym(RTLD_NEXT,cuFuncGetName)=%p", diag_n, sym);
    }
    if (!sym) {
        void *h1 = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (diag_on) cipher_log("RESOLVE-DIAG-%d: dlopen('libcuda.so.1',NOLOAD)=%p", diag_n, h1);
        if (h1) sym = dlsym(h1, "cuFuncGetName");
        if (diag_on && h1) cipher_log("RESOLVE-DIAG-%d: dlsym(h1,cuFuncGetName)=%p", diag_n, sym);
    }
    if (!sym) {
        void *h2 = dlopen("libcuda.so.1", RTLD_LAZY);
        if (diag_on) cipher_log("RESOLVE-DIAG-%d: dlopen('libcuda.so.1',LAZY)=%p errno=%s", diag_n, h2, dlerror());
        if (h2) sym = dlsym(h2, "cuFuncGetName");
        if (diag_on && h2) cipher_log("RESOLVE-DIAG-%d: dlsym(h2,cuFuncGetName)=%p", diag_n, sym);
    }
    if (!sym) {
        /* Last resort: known absolute paths inside vllm/vllm-openai:v0.21.0 */
        const char *paths[] = {
            "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcuda.so.1",
            "/usr/local/cuda-13.0/compat/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            NULL
        };
        for (int i = 0; paths[i] && !sym; ++i) {
            void *h3 = dlopen(paths[i], RTLD_LAZY);
            if (diag_on) cipher_log("RESOLVE-DIAG-%d: dlopen('%s')=%p", diag_n, paths[i], h3);
            if (h3) sym = dlsym(h3, "cuFuncGetName");
        }
        if (!sym) return false;
    }
    g_cuFuncGetName.store((pf_cuFuncGetName)sym, std::memory_order_release);
    cipher_log("WORKLOAD-CLASSIFIER: cuFuncGetName resolved at %p", sym);
    return true;
}

/* === Kernel-name -> KCB classification =================================== */

uint32_t classify_kernel_name(const char *name) {
    if (!name || !name[0]) return 0;
    uint32_t bits = 0;

    /* Order matters slightly: cheaper checks first; cumulative OR ensures
     * multi-tagged names (e.g. "machete_awq_") get all relevant bits. */
    if (strstr(name, "machete") || strstr(name, "Machete") ||
        strstr(name, "MacheteCollective")) bits |= KCB_MACHETE;
    if (strstr(name, "awq")) bits |= KCB_AWQ;
    if (strstr(name, "gptq") || strstr(name, "Marlin")) bits |= KCB_GPTQ_MARLIN;
    if (strstr(name, "flash") || strstr(name, "fmha") || strstr(name, "FMHA") ||
        strstr(name, "fwd_kernel_inner") || strstr(name, "sdp_fwd")) bits |= KCB_FLASH_ATTN;
    if (strstr(name, "rms_norm") || strstr(name, "RMSNorm")) bits |= KCB_RMS_NORM;
    if (strstr(name, "silu")) bits |= KCB_SILU;
    if (strstr(name, "gelu") || strstr(name, "GELU")) bits |= KCB_GELU;
    /* K.1.5 Step 1.6 Sub-step 3 spec-required pattern: rotary embedding kernels
     * (vllm::rotary_embedding_kernel etc). Reuses KCB_RMS_NORM bit semantically
     * because both are positional/normalization signal kernels in transformer
     * inference; K.1.6 may split into a dedicated bit if needed. */
    if (strstr(name, "rotary") || strstr(name, "Rotary")) bits |= KCB_RMS_NORM;
    /* K.1.6 (2026-05-27) Sub-step 1.2 pattern expansion per Table 4 empirical
     * kernel-name evidence from K.1.5 14-cell capture. Reuse existing bits
     * semantically; no new KCB_ bits to keep classify_kernel_name output
     * compatible with all downstream decision logic. */
    if (strstr(name, "act_and_mul") || strstr(name, "act_and_mul_kernel"))
        bits |= KCB_SILU;  /* vllm::act_and_mul_kernel uses silu/swiglu internally */
    if (strstr(name, "fused_add_rms_norm")) bits |= KCB_RMS_NORM;
    if (strstr(name, "TopPSamplingFromProb") ||
        strstr(name, "RadixTopKMaskLogitsKernel") ||
        strstr(name, "flashinfer8sampling") ||
        strstr(name, "flashinfer::sampling"))
        bits |= KCB_REDUCE;  /* sampling kernels are reduction-class */
    if (strstr(name, "nccl") || strstr(name, "all_reduce") ||
        strstr(name, "AllReduce") || strstr(name, "ReduceScatter") ||
        strstr(name, "AllGather")) bits |= KCB_NCCL;
    if (strstr(name, "_bwd") || strstr(name, "backward") ||
        strstr(name, "_gradient")) bits |= KCB_BACKWARD;
    if (strstr(name, "optimizer") || strstr(name, "Adam") ||
        strstr(name, "AdamW") || strstr(name, "SGD")) bits |= KCB_OPTIMIZER;
    if (strstr(name, "reduce") || strstr(name, "Reduce")) bits |= KCB_REDUCE;

    return bits;
}

/* K.1.5 Step 1: full-event diagnostic instrumentation. env-gated via
 * CIPHER_K1_DIAGNOSTIC; default OFF preserves <10ns hot-path budget (one
 * atomic load + branch-not-taken). When ON, all 5 observe_* functions emit
 * K1-DIAG-* lines + per-1000-launches snapshot + K1-DIAG-UNRESOLVED on
 * negative-cache poison (first 100 unique fn ptrs).
 *
 * Per advisor K.1.5 Step 0 framing: the empirical fn-ptr+name characterization
 * lives here (Step 1), not in the substrate (Step 0). */
static std::atomic<int> g_diag_inited{0};
static std::atomic<int> g_diag_on{0};
static std::atomic<uint64_t> g_diag_unresolved_logged{0};
constexpr uint64_t DIAG_UNRESOLVED_LOG_CAP = 100ULL;
constexpr uint64_t DIAG_SNAPSHOT_INTERVAL = 1000ULL;

static inline int diag_on(void) {
    if (!g_diag_inited.load(std::memory_order_acquire)) {
        int already = g_diag_inited.exchange(1, std::memory_order_acq_rel);
        if (!already) {
            const char *e = getenv("CIPHER_K1_DIAGNOSTIC");
            int on = (e && e[0] == '1') ? 1 : 0;
            g_diag_on.store(on, std::memory_order_release);
            if (on) {
                cipher_log("K1-DIAG: instrumentation ENABLED (snapshot every %llu launches; unresolved cap %llu)",
                           (unsigned long long)DIAG_SNAPSHOT_INTERVAL,
                           (unsigned long long)DIAG_UNRESOLVED_LOG_CAP);
            }
        }
    }
    return g_diag_on.load(std::memory_order_relaxed);
}

static inline uint64_t diag_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Per-event log helpers; only called when diag_on() returned true at caller.
 * cipher_log uses fprintf to stderr (microseconds). Not in hot-path budget. */
static void diag_log_launch(const void *fn, uint32_t grid_total, uint32_t block_total,
                            const void *stream, uint32_t classified) {
    /* Lookup name for diagnostic-mode log only (side-effect-free peek). */
    const char *name = reg_map_lookup(fn);
    if (!name && g_cuFuncGetName.load(std::memory_order_relaxed)) {
        pf_cuFuncGetName g = g_cuFuncGetName.load(std::memory_order_relaxed);
        if (g) (void)g(&name, (void *)fn);
    }
    cipher_log("K1-DIAG-LAUNCH ts=%lu fn=%p name=%s grid=%u block=%u stream=%p classified=0x%x",
               (unsigned long)diag_now_ns(), fn,
               name ? name : "(unresolved)",
               grid_total, block_total, stream, classified);
}

static void diag_log_gemm(int m, int n, int k, int Atype, int Btype, int Ctype, const void *stream) {
    cipher_log("K1-DIAG-GEMM ts=%lu M=%d N=%d K=%d Atype=%d Btype=%d Ctype=%d stream=%p",
               (unsigned long)diag_now_ns(), m, n, k, Atype, Btype, Ctype, stream);
}

static void diag_log_alloc(size_t bytes, const void *ptr) {
    cipher_log("K1-DIAG-ALLOC ts=%lu bytes=%zu ptr=%p",
               (unsigned long)diag_now_ns(), bytes, ptr);
}

static void diag_log_capture_begin(void) {
    cipher_log("K1-DIAG-CAPTURE ts=%lu", (unsigned long)diag_now_ns());
}

static void diag_log_nccl(void) {
    cipher_log("K1-DIAG-NCCL ts=%lu", (unsigned long)diag_now_ns());
}

static void diag_log_unresolved(const void *fn) {
    /* Cap dedup: first DIAG_UNRESOLVED_LOG_CAP unique fns get a log line.
     * Counter incremented even after cap to surface accurate count. */
    uint64_t n = g_diag_unresolved_logged.fetch_add(1, std::memory_order_relaxed);
    if (n < DIAG_UNRESOLVED_LOG_CAP) {
        cipher_log("K1-DIAG-UNRESOLVED #%lu fn=%p attempted_via=reg_map,cuFuncGetName",
                   (unsigned long)n, fn);
    }
}

/* K.1.5 Step 1.5: detailed unresolved logging — CBID + provenance.
 * Called only when negative-cache poisoning fires (once per unique fn). */
static void diag_log_unresolved_cbid(const void *fn, uint32_t cbid) {
    uint64_t n = g_diag_unresolved_logged.load(std::memory_order_relaxed);
    if (n <= DIAG_UNRESOLVED_LOG_CAP) {
        cipher_log("K1-DIAG-UNRESOLVED-CBID #%lu fn=%p cbid=%u",
                   (unsigned long)(n - 1), fn, cbid);
    }
}

static void diag_log_unresolved_prov(const void *fn, uint32_t cbid) {
    uint64_t n = g_diag_unresolved_logged.load(std::memory_order_relaxed);
    if (n > DIAG_UNRESOLVED_LOG_CAP) return;
    int in_reg = 0, via_module = 0, via_library = 0, via_kernel = 0, via_cuda_reg = 0;
    {
        std::lock_guard<std::mutex> lk(g_provenance_mu);
        via_cuda_reg = g_prov_cuda_register_fn.count(fn) ? 1 : 0;
        via_module   = g_prov_cu_module_get_fn.count(fn) ? 1 : 0;
        via_library  = g_prov_cu_library_get.count(fn) ? 1 : 0;
        via_kernel   = g_prov_cu_kernel_get_fn.count(fn) ? 1 : 0;
    }
    in_reg = (reg_map_lookup(fn) != nullptr) ? 1 : 0;
    cipher_log("K1-DIAG-UNRESOLVED-PROV #%lu fn=%p in_reg_map=%d "
               "via_cuda_reg=%d via_module_get=%d via_library_get=%d via_kernel_get=%d cbid=%u",
               (unsigned long)(n - 1), fn, in_reg,
               via_cuda_reg, via_module, via_library, via_kernel, cbid);
}

static void diag_log_unresolved_retry(const void *fn, const char *retry_path, const char *name) {
    uint64_t n = g_diag_unresolved_logged.load(std::memory_order_relaxed);
    if (n > DIAG_UNRESOLVED_LOG_CAP) return;
    cipher_log("K1-DIAG-UNRESOLVED-RETRY fn=%p retry_path=%s name=%s",
               fn, retry_path, name ? name : "(null)");
}

static void diag_log_snapshot(uint64_t obs);  /* defined after WorkloadSignals visible */

/* K.1.5 Bug #5 diagnostic: env-gated once-per-unique-name logging of kernel
 * names that resolved but didn't match any classify_kernel_name pattern.
 * Static atomic init; env read once. */
static std::atomic<int> g_log_unmatched_inited2{0};
static std::atomic<int> g_log_unmatched_on2{0};

static inline int log_unmatched_on(void) {
    if (!g_log_unmatched_inited2.load(std::memory_order_relaxed)) {
        int already = g_log_unmatched_inited2.exchange(1, std::memory_order_relaxed);
        if (!already) {
            const char *e = getenv("CIPHER_K1_LOG_UNMATCHED");
            int on = (e && e[0] == '1') ? 1 : 0;
            g_log_unmatched_on2.store(on, std::memory_order_relaxed);
        }
    }
    return g_log_unmatched_on2.load(std::memory_order_relaxed);
}

/* Lazy cache lookup. Returns the bit mask for fn (KCB_RESOLVED set if known).
 * Returns 0 if cache miss AND cuFuncGetName unavailable (degrade gracefully).
 *
 * K.1.5 Bug #5 diagnostic counters: distinguish three distinct fail modes
 * that all return 0 — resolve_failed vs resolved_no_match vs probe_overflow
 * per advisor 2026-05-27. cache_hits increments on every successful return. */
uint32_t lookup_fn_bits(const void *fn, uint32_t cbid = 0) {
    if (!fn) return 0;
    uint64_t h = (uintptr_t)fn * 0x9E3779B97F4A7C15ULL;
    uint32_t slot = (uint32_t)(h >> 32) & (FN_CACHE_SLOTS - 1);

    /* Linear probe up to FN_CACHE_PROBE_MAX slots. */
    for (uint32_t i = 0; i < FN_CACHE_PROBE_MAX; ++i) {
        FnCacheEntry &e = g_fn_cache[(slot + i) & (FN_CACHE_SLOTS - 1)];
        const void *seen = e.fn.load(std::memory_order_acquire);
        if (seen == fn) {
            uint32_t bits = e.bits.load(std::memory_order_relaxed);
            /* K.1.5 Step 0 Sub-step 1: distinguish actual class bits from
             * the bare KCB_RESOLVED marker (which indicates negative-cache
             * poison from prior failed resolution). Only count as a hit
             * when actual classification bits are present. */
            if (bits & ~KCB_RESOLVED)
                g_signals.fn_cache_hits.fetch_add(1, std::memory_order_relaxed);
            else
                g_signals.fn_cache_resolved_no_match.fetch_add(1, std::memory_order_relaxed);
            return bits;
        }
        if (seen == nullptr) {
            /* K.1.5 Step 1.6 primary resolution chain (H4 verdict applied):
             *   1. reg_map_lookup  — __cudaRegisterFunction host_ptr map
             *   2. cuKernelGetName — CUDA 12.4+ unified handle (PyTorch 2.x
             *      + vLLM dominant path; covered 50/50 unique unresolved
             *      kernels in Step 1.5 retry chain. Promoted to primary.)
             *   3. cuFuncGetName   — classic CUfunction (legacy paths)
             * Retry chain below still runs on failure of all three, but
             * with cuKernelGetName promoted to primary it should not
             * recover any more entries (counter should be 0 going forward). */
            const char *name = reg_map_lookup(fn);
            int from_reg = (name != nullptr);
            int from_cu_kernel = 0;
            int rc = name ? 0 : -1;

            if (!name) {
                /* Primary slot 2: cuKernelGetName (Step 1.6 promotion). */
                pf_cuKernelGetName gk = g_cuKernelGetName.load(std::memory_order_relaxed);
                if (!gk) {
                    void *sym = dlsym(RTLD_DEFAULT, "cuKernelGetName");
                    if (sym) {
                        gk = (pf_cuKernelGetName)sym;
                        g_cuKernelGetName.store(gk, std::memory_order_release);
                    }
                }
                if (gk && gk(&name, (void *)fn) == 0 && name) {
                    from_cu_kernel = 1;
                    rc = 0;
                }
            }
            if (!name) {
                /* Slot 3: classic cuFuncGetName for CUfunction handles. */
                if (!resolve_cu_funcname()) {
                    g_signals.fn_cache_resolution_failed.fetch_add(1, std::memory_order_relaxed);
                    return 0;
                }
                pf_cuFuncGetName getname = g_cuFuncGetName.load(std::memory_order_relaxed);
                if (getname) rc = getname(&name, (void *)fn);
            }
            if (rc == 0 && name) {
                uint32_t classified = classify_kernel_name(name);
                uint32_t bits = classified | KCB_RESOLVED;
                if (from_reg) {
                    g_signals.fn_cache_resolved_via_reg.fetch_add(1, std::memory_order_relaxed);
                } else if (from_cu_kernel) {
                    g_signals.fn_cache_resolved_via_cu_kernel_get_name.fetch_add(1, std::memory_order_relaxed);
                }
                /* Try to claim the empty slot; if another thread beat us,
                 * fall through to next probe iter. */
                const void *expected = nullptr;
                if (e.fn.compare_exchange_strong(expected, fn,
                        std::memory_order_release, std::memory_order_acquire)) {
                    e.bits.store(bits, std::memory_order_relaxed);
                    if (classified) {
                        g_signals.fn_cache_hits.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        g_signals.fn_cache_resolved_no_match.fetch_add(1, std::memory_order_relaxed);
                        /* K.1.5 once-per-unique-name dump of unclassified
                         * kernel names; gated by env. New cache entry = first
                         * time we see this fn ptr, so log fires once per fn. */
                        if (log_unmatched_on()) {
                            cipher_log("CLASSIFY-UNMATCHED: %s", name);
                        }
                    }
                    return bits;
                }
                /* Lost the race; reload and check if it's our fn. */
                if (e.fn.load(std::memory_order_acquire) == fn) {
                    uint32_t b2 = e.bits.load(std::memory_order_relaxed);
                    if (b2 & ~KCB_RESOLVED) g_signals.fn_cache_hits.fetch_add(1, std::memory_order_relaxed);
                    else g_signals.fn_cache_resolved_no_match.fetch_add(1, std::memory_order_relaxed);
                    return b2;
                }
                /* Different fn took the slot; continue probing. */
                continue;
            }
            /* K.1.5 Step 0 Sub-step 1: cache-poison failed resolution.
             * Without this, every launch of an unresolvable kernel pays the
             * full resolve cost (~100-1000ns dlsym + cuFuncGetName roundtrip).
             * With this, the second-and-subsequent launches hit the cache
             * empty-bits path and increment fn_cache_resolved_no_match (cheap).
             *
             * We mark KCB_RESOLVED but with no classification bits, so the
             * existing cache-hit path returns bits == KCB_RESOLVED -> visible
             * as "resolved but classify_kernel_name didn't match." For
             * counters: first attempt counts as fn_cache_resolution_failed
             * (this branch) AND fn_cache_negative_cached (new) so we can
             * measure how many of the "fail" total were converted to cheap
             * cache hits going forward. */
            /* K.1.5 Step 1.5 alternative-resolution retry chain. BEFORE
             * negative-cache CAS, try three alternative resolution paths.
             * Any success short-circuits to positive cache fill. */
            const char *retry_name = nullptr;
            const char *retry_path = nullptr;

            /* retry_a: reg_map again (memory-order race detection). */
            retry_name = reg_map_lookup(fn);
            if (retry_name) retry_path = "reg_map_retry";

            /* retry_b: cuKernelGetName (CUDA 12.4+ alt API). */
            if (!retry_name) {
                pf_cuKernelGetName g = g_cuKernelGetName.load(std::memory_order_relaxed);
                if (!g) {
                    void *sym = dlsym(RTLD_DEFAULT, "cuKernelGetName");
                    if (sym) {
                        g = (pf_cuKernelGetName)sym;
                        g_cuKernelGetName.store(g, std::memory_order_release);
                    }
                }
                if (g) {
                    const char *n = nullptr;
                    if (g(&n, (void *)fn) == 0 && n) {
                        retry_name = n;
                        retry_path = "cuKernelGetName";
                    }
                }
            }

            /* retry_c: cuFuncGetAttribute as validity probe (does NOT resolve
             * a name; tells us whether fn is a valid CUfunction at all).
             * Negative result here means fn is not a CUfunction handle — it
             * may be a host-side __device_stub__ or a stale/invalid pointer. */
            int fn_is_valid_cufunction = 0;
            if (!retry_name) {
                pf_cuFuncGetAttribute g = g_cuFuncGetAttribute.load(std::memory_order_relaxed);
                if (!g) {
                    void *sym = dlsym(RTLD_DEFAULT, "cuFuncGetAttribute");
                    if (sym) {
                        g = (pf_cuFuncGetAttribute)sym;
                        g_cuFuncGetAttribute.store(g, std::memory_order_release);
                    }
                }
                if (g) {
                    int unused = 0;
                    /* CU_FUNC_ATTRIBUTE_NUM_REGS = 4 (always queryable). */
                    if (g(&unused, 4, (void *)fn) == 0) {
                        fn_is_valid_cufunction = 1;
                    }
                }
            }

            if (retry_name) {
                /* Recovery success: classify + cache as POSITIVE not negative. */
                uint32_t r_classified = classify_kernel_name(retry_name);
                uint32_t r_bits = r_classified | KCB_RESOLVED;
                const void *expected_r = nullptr;
                if (e.fn.compare_exchange_strong(expected_r, fn,
                        std::memory_order_release, std::memory_order_acquire)) {
                    e.bits.store(r_bits, std::memory_order_relaxed);
                }
                g_signals.fn_cache_recovered_via_retry.fetch_add(1, std::memory_order_relaxed);
                if (r_classified)
                    g_signals.fn_cache_hits.fetch_add(1, std::memory_order_relaxed);
                else
                    g_signals.fn_cache_resolved_no_match.fetch_add(1, std::memory_order_relaxed);
                if (diag_on()) diag_log_unresolved_retry(fn, retry_path, retry_name);
                return r_bits;
            }

            /* All retries failed — negative-cache poison as before. */
            uint32_t bits = KCB_RESOLVED;  /* tried, no class */
            const void *expected2 = nullptr;
            int was_first = 0;
            if (e.fn.compare_exchange_strong(expected2, fn,
                    std::memory_order_release, std::memory_order_acquire)) {
                e.bits.store(bits, std::memory_order_relaxed);
                g_signals.fn_cache_negative_cached.fetch_add(1, std::memory_order_relaxed);
                was_first = 1;
            }
            g_signals.fn_cache_resolution_failed.fetch_add(1, std::memory_order_relaxed);
            /* K.1.5 Step 1 unresolved-residue characterization: log fn ptr the
             * FIRST time we negative-cache it. Subsequent launches hit the
             * cached entry and don't reach this branch. Capped at
             * DIAG_UNRESOLVED_LOG_CAP.
             *
             * K.1.5 Step 1.5 H1/H2/H3 disambiguator: also log CBID +
             * provenance for each unresolved fn. Used by Sub-step 4 verdict. */
            if (was_first && diag_on()) {
                diag_log_unresolved(fn);
                diag_log_unresolved_cbid(fn, cbid);
                diag_log_unresolved_prov(fn, cbid);
                if (!fn_is_valid_cufunction) {
                    cipher_log("K1-DIAG-UNRESOLVED-NOTVALIDCUFUNC fn=%p "
                               "(cuFuncGetAttribute rejected -> host stub or stale ptr)", fn);
                }
            }
            return 0;
        }
    }
    g_signals.fn_cache_probe_overflow.fetch_add(1, std::memory_order_relaxed);
    return 0;  /* probe distance exceeded; treat as unclassified */
}

/* === Process count via /proc =============================================
 * Counts sibling processes (different pid) that have libcipher_rt.so mapped.
 * O(N) over /proc; called only at classify time, not per-launch. */

uint32_t count_concurrent_cipher_processes(void) {
    uint32_t count = 0;
    DIR *d = opendir("/proc");
    if (!d) return 1;
    pid_t self_pid = getpid();
    char path[256], buf[4096];
    struct dirent *de;
    while ((de = readdir(d)) != nullptr) {
        if (de->d_name[0] < '0' || de->d_name[0] > '9') continue;
        pid_t pid = (pid_t)atoi(de->d_name);
        if (pid == self_pid) {
            /* Self counts. */
            ++count;
            continue;
        }
        snprintf(path, sizeof(path), "/proc/%d/maps", pid);
        int fd = open(path, O_RDONLY);
        if (fd < 0) continue;
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n <= 0) continue;
        buf[n] = '\0';
        if (strstr(buf, "libcipher_rt.so")) ++count;
    }
    closedir(d);
    return count > 0 ? count : 1;
}

/* === Time ============================================================== */

uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* === Classification decision table ====================================== */

void apply_capability_flags(cipher_workload_profile &p) {
    /* Reset all flags; will set per workload class. */
    p.volt_engage = 0; p.marlin_engage = 0; p.koopman_engage = 0;
    p.kv_dedup_engage = 0; p.weight_arena_engage = 0; p.sm_packer_engage = 0;
    p.per_tenant_routing = 0; p.pool_substrate = 0; p.flash_attn_intercept = 0;
    p.nccl_tuner = 0; p.persistent_kernel = 0; p.speculative_decode = 0;
    p.graph_engine = 0; p.flop_telemetry = 0; p.audit_chain = 0;

    /* W.1 (2026-05-27): VOLT engagement is regime-gated, not class-gated.
     * Per cipher-t43-envelope: +57% tok/W on bandwidth-bound decode but -14%
     * on compute-bound prefill / training. The volt_engage decision uses
     * the new bw_bound_regime_detected + large_prefill_detected signals
     * derived in classify_internal. Customer env CIPHER_VOLT=1/0 still
     * overrides (handled at the actuator-side env-gate per K.2 pattern). */
    bool bw_bound = p.bw_bound_regime_detected != 0;
    bool large_prefill = p.large_prefill_detected != 0;

    switch (p.workload_class) {
    case CIPHER_WL_A1_AGENTIC_MULTI_TENANT:
        p.volt_engage = (bw_bound && !large_prefill) ? 1 : 0;
        /* W.2 (2026-05-27): marlin_engage extended — fires on int4 (Machete
         * intercept path) OR bf16 (RTN INT4 dtype-cast path). The bf16 path
         * carries calibration risk; the actuator-side gate handles correctness
         * via the per-substitution KL check. */
        p.marlin_engage = p.int4_weights_detected || p.bf16_weights_detected;
        p.koopman_engage = p.long_context_detected;
        p.kv_dedup_engage = 1;
        p.weight_arena_engage = p.same_model_across_tenants;
        p.sm_packer_engage = 1;
        p.per_tenant_routing = 1;
        p.pool_substrate = 1;
        p.flash_attn_intercept = 1;
        p.flop_telemetry = 1;
        p.audit_chain = 1;
        break;
    case CIPHER_WL_A2_CONTINUOUS_BATCHED:
        p.volt_engage = (bw_bound && !large_prefill) ? 1 : 0;
        /* W.2 (2026-05-27): marlin_engage extended — fires on int4 (Machete
         * intercept path) OR bf16 (RTN INT4 dtype-cast path). The bf16 path
         * carries calibration risk; the actuator-side gate handles correctness
         * via the per-substitution KL check. */
        p.marlin_engage = p.int4_weights_detected || p.bf16_weights_detected;
        p.koopman_engage = p.long_context_detected;
        p.per_tenant_routing = 1;
        p.flash_attn_intercept = 1;
        p.graph_engine = 1;
        p.flop_telemetry = 1;
        break;
    case CIPHER_WL_A3_SINGLE_TENANT_STREAM:
        /* A3 = single-tenant decode, always bw-bound BY DEFINITION.
         * large_prefill flag still vetoes if a prefill-mode workload was
         * misclassified as A3 (defensive). */
        p.volt_engage = (!large_prefill) ? 1 : 0;
        /* W.2 (2026-05-27): marlin_engage extended — fires on int4 (Machete
         * intercept path) OR bf16 (RTN INT4 dtype-cast path). The bf16 path
         * carries calibration risk; the actuator-side gate handles correctness
         * via the per-substitution KL check. */
        p.marlin_engage = p.int4_weights_detected || p.bf16_weights_detected;
        p.koopman_engage = p.long_context_detected;
        p.flash_attn_intercept = 1;
        break;
    case CIPHER_WL_A4_BATCH_INFERENCE:
        /* A4 spans large-batch decode AND batched prefill.  bw_bound
         * separates them empirically. */
        p.volt_engage = (bw_bound && !large_prefill) ? 1 : 0;
        /* W.2 (2026-05-27): marlin_engage extended — fires on int4 (Machete
         * intercept path) OR bf16 (RTN INT4 dtype-cast path). The bf16 path
         * carries calibration risk; the actuator-side gate handles correctness
         * via the per-substitution KL check. */
        p.marlin_engage = p.int4_weights_detected || p.bf16_weights_detected;
        p.persistent_kernel = 1;
        p.flash_attn_intercept = 1;
        break;
    case CIPHER_WL_B1_PRE_TRAINING:
        /* Explicit deny: training is compute-bound, would regress
         * (cipher-t43-envelope -14% case). */
        p.volt_engage = 0;
        p.nccl_tuner = 1;
        p.flop_telemetry = 1;
        p.audit_chain = 1;
        /* W.2 (2026-05-27): marlin_engage extended — fires on int4 (Machete
         * intercept path) OR bf16 (RTN INT4 dtype-cast path). The bf16 path
         * carries calibration risk; the actuator-side gate handles correctness
         * via the per-substitution KL check. */
        p.marlin_engage = p.int4_weights_detected || p.bf16_weights_detected;
        break;
    case CIPHER_WL_B2_FINE_TUNING:
        /* Explicit deny: same reason as B1. */
        p.volt_engage = 0;
        p.nccl_tuner = 1;
        p.weight_arena_engage = 1;       /* LoRA shared backbone case */
        p.flop_telemetry = 1;
        break;
    case CIPHER_WL_C1_RAG_LONG_CONTEXT:
        p.volt_engage = (bw_bound && !large_prefill) ? 1 : 0;
        p.koopman_engage = 1;
        p.kv_dedup_engage = 1;
        p.flash_attn_intercept = 1;
        break;
    case CIPHER_WL_UNKNOWN:
    default:
        /* W.1 v2 (2026-05-27): UNKNOWN now defaults VOLT-off (safe default).
         * Was VOLT-only in K.1; flipped because UNKNOWN can include early-
         * warmup compute-bound work where +57%/-14% regime is ambiguous. */
        p.volt_engage = 0;
        break;
    }
}

cipher_workload_class_t classify_internal(cipher_workload_profile &out) {
    out = {};
    out.classified_at_ns = monotonic_ns();
    out.observations_at_classify =
        (uint32_t)g_signals.total_observations.load(std::memory_order_relaxed);

    /* Aggregate signals. */
    uint64_t machete   = g_signals.machete_count.load(std::memory_order_relaxed);
    uint64_t awq       = g_signals.awq_count.load(std::memory_order_relaxed);
    uint64_t gptq      = g_signals.gptq_marlin_count.load(std::memory_order_relaxed);
    uint64_t flash     = g_signals.flash_attn_count.load(std::memory_order_relaxed);
    uint64_t nccl_k    = g_signals.nccl_count.load(std::memory_order_relaxed);
    uint64_t bwd       = g_signals.backward_count.load(std::memory_order_relaxed);
    uint64_t opt       = g_signals.optimizer_count.load(std::memory_order_relaxed);
    int      nccl_lib  = g_signals.nccl_lib_loaded.load(std::memory_order_relaxed);
    int      capture   = g_signals.capture_seen.load(std::memory_order_relaxed);

    uint64_t gemm_tot  = g_signals.gemm_total.load(std::memory_order_relaxed);
    uint64_t gemm_b1   = g_signals.gemm_b1_count.load(std::memory_order_relaxed);
    uint64_t gemm_lg   = g_signals.gemm_large_batch_count.load(std::memory_order_relaxed);
    uint32_t max_m     = g_signals.max_m.load(std::memory_order_relaxed);
    uint32_t max_n     = g_signals.max_n.load(std::memory_order_relaxed);
    uint32_t max_k     = g_signals.max_k.load(std::memory_order_relaxed);

    uint64_t alloc_lg  = g_signals.alloc_large_count.load(std::memory_order_relaxed);
    uint64_t alloc_med = g_signals.alloc_medium_count.load(std::memory_order_relaxed);

    /* Process count: /proc-maps scan kept as a secondary diagnostic only. */
    uint32_t procs = count_concurrent_cipher_processes();
    g_signals.concurrent_processes_estimate.store(procs, std::memory_order_relaxed);

    /* W.6 sub-C: authoritative GPU co-residence from the kmod cohort registry.
     * Heartbeat-or-insert this process's substrate model fingerprint (W.6
     * sub-B) into the kmod table and read back the number of distinct LIVE
     * tenant processes on this GPU (incl. self). This replaces the /proc-maps
     * heuristic as the multi_tenant gate. The client throttles the underlying
     * ioctl to ~1/sec, so calling it every classify is cheap. */
    uint64_t model_fp   = cipher_workload_model_fingerprint();
    struct cipher_cohort_peer pool_peers[CIPHER_POOL_MAX];
    uint32_t pool_ntotal = 0;
    uint32_t co_resident = cipher_rt_coresidence_update(model_fp, pool_peers,
                                                        CIPHER_POOL_MAX, &pool_ntotal);
    out.co_resident_count = co_resident;
    /* W.4a: partition co-resident tenants into same-fingerprint coalesce-groups
     * (DECISION-ONLY; transport is the sequenced W.4b CP 5.6 re-port). The
     * Memory #11 guard lives in cipher_rt_pool_partition: distinct/zero
     * fingerprints NEVER group. ~1/sec cadence (rides the cohort query). */
    {
        uint32_t pn = pool_ntotal < CIPHER_POOL_MAX ? pool_ntotal : CIPHER_POOL_MAX;
        cipher_rt_pool_update(model_fp, pool_peers, pn);
    }

    bool has_machete  = (machete + awq + gptq) > 0;
    /* K.1 Step 7 (Bug #2 fix): drop nccl_lib_loaded from has_nccl. libnccl
     * loaded as DT_NEEDED by vLLM 0.21 even at TP=1 inference; only actual
     * KCB_NCCL kernel-name observations are a reliable signal. nccl_lib
     * remains a separate observable diagnostic but is not a class gate. */
    bool has_nccl     = (nccl_k > 5);
    bool has_backward = (bwd > 10) || (opt > 0);
    bool has_capture  = (capture != 0);
    bool batch1_dom   = (gemm_tot > 50) &&
                        (gemm_b1 * 4 > gemm_tot * 3);  /* > 75% of GEMMs B<=1 */
    bool large_batch  = (gemm_lg > 10);
    /* K.1 Step 6 (Bug #1 fix): split signals honestly.
     *   wide_matmul_detected = raw cuBLAS dim observation (model architecture)
     *   kv_cache_alloc_pattern_detected = cudaMalloc count in KV-block class
     *   long_ctx_detected (DERIVED) = pattern AND seq-len threshold
     * Llama-3 wide intermediate dims trigger wide_matmul but NOT long_ctx. */
    bool wide_matmul  = (max_m > WIDE_MATMUL_DIM_THRESH) ||
                        (max_n > WIDE_MATMUL_DIM_THRESH * 2);
    bool kv_pattern   = (alloc_med > KV_CACHE_PATTERN_THRESH);
    bool long_ctx     = kv_pattern && (max_n > LONG_CTX_SEQ_THRESH);
    bool multi_tenant = (co_resident >= 2);  /* W.6 sub-C authoritative (kmod cohort) */
    bool weights_seen = (alloc_lg > 0);

    uint32_t confidence = 500;     /* neutral; bumped per certainty */

    /* K.1.6 (2026-05-27) Sub-step 1.1 IDLE GATE: if observations_per_second
     * < IDLE_OBS_PER_SEC_THRESH (50/s), workload is in load/idle/teardown
     * phase, not active inference. Emit UNKNOWN with low confidence to
     * avoid false-positive A4 on E2-style "model load then sleep" cells.
     * The 5s grace period prevents false-idle on slow-warmup workloads. */
    {
        uint64_t first_ns = g_signals.first_obs_ns.load(std::memory_order_relaxed);
        if (first_ns > 0) {
            uint64_t now = diag_now_ns();
            uint64_t elapsed_ns = (now > first_ns) ? (now - first_ns) : 0;
            if (elapsed_ns > IDLE_MIN_ELAPSED_NS) {
                uint64_t total_obs = g_signals.total_observations.load(std::memory_order_relaxed);
                uint64_t obs_per_sec = (total_obs * 1000000000ULL) / elapsed_ns;
                if (obs_per_sec < IDLE_OBS_PER_SEC_THRESH) {
                    out.workload_class = CIPHER_WL_UNKNOWN;
                    out.classification_confidence = 250;
                    out.training_workload_detected = 0;
                    out.multi_tenant_detected = multi_tenant ? 1 : 0;
                    out.same_model_across_tenants = 0;
                    apply_capability_flags(out);
                    return out.workload_class;
                }
            }
        }
        /* K.1.6 v7: hysteresis gate moved out of classify_internal — see
         * cipher_workload_classify() at the publish boundary. Keeping the
         * decision tree's raw output here so high_conf_emits counts honest
         * "would-be" KNOWN classifications. */
    }

    /* Decision tree (audit Section B mapping; K.1 Step 8 reorder: A2 priority
     * over A1 because cuStreamBeginCapture / cuGraphLaunch dominates the
     * multi_tenant false positive caused by EngineCore worker subprocess —
     * graphs-enabled inference is A2_CONTINUOUS_BATCHED, period).
     *
     * K.1.6 (2026-05-27) Sub-step 1.3 A2/A4 collapse note: per K.1.5 Table 5
     * empirical analysis, A2_CONTINUOUS_BATCHED and A4_BATCH_INFERENCE are
     * INDISTINGUISHABLE at driver level on vLLM 0.21 default (cudagraph_mode
     * =NONE; capture_seen=0 in all 12 inference captures). The A2 branch
     * below is preserved for ABI but unreachable on stock vLLM 0.21; both
     * batched-decode and large-batch-prefill workloads route to A4. K.1.6
     * downstream consumers treat A2 and A4 as the same actuator-engagement
     * subset (A_INFERENCE_BATCHED semantic). A2 reachability returns when
     * vLLM 0.22+ promotes cudagraph_mode=FULL or when stream-count signal
     * lands at W.4 (POOL executor batching). */
    if (has_nccl && has_backward) {
        out.workload_class = CIPHER_WL_B1_PRE_TRAINING;
        confidence = 880;
    } else if (has_backward) {
        out.workload_class = CIPHER_WL_B2_FINE_TUNING;
        confidence = 800;
    } else if (has_capture && !batch1_dom && gemm_tot > 100) {
        /* Step 8 promoted above A1: graphs in use = continuous batched serving
         * (vLLM 0.21+ default). Defangs the multi_tenant false positive. */
        out.workload_class = CIPHER_WL_A2_CONTINUOUS_BATCHED;
        confidence = 800;
    } else if (multi_tenant && batch1_dom && !has_capture) {
        /* Step 8 added !has_capture: A1 only fires for agent-decode WITHOUT
         * graph mode. Multi-tenant signal itself is SOFT (Bug #3 deferred to
         * W.6 NR 27 process-registry); classification still benefits from it
         * as a corroborating signal when present. */
        out.workload_class = CIPHER_WL_A1_AGENTIC_MULTI_TENANT;
        confidence = 820;
    } else if (long_ctx && !batch1_dom && (flash > 0 || gemm_tot > 50)) {
        /* Step 6: long_ctx is now the DERIVED signal (kv_pattern AND seq).
         * Llama-3 short-prompt vLLM will NOT trigger this because either
         * alloc_med stays below KV_CACHE_PATTERN_THRESH (PagedAttention does
         * a single big alloc) OR max_n stays below LONG_CTX_SEQ_THRESH. */
        out.workload_class = CIPHER_WL_C1_RAG_LONG_CONTEXT;
        confidence = 740;
    } else if (batch1_dom) {
        out.workload_class = CIPHER_WL_A3_SINGLE_TENANT_STREAM;
        confidence = 820;
    } else if (large_batch) {
        out.workload_class = CIPHER_WL_A4_BATCH_INFERENCE;
        confidence = 720;
    } else if (weights_seen && gemm_tot < 50) {
        /* Insufficient launch signal yet; keep UNKNOWN with low confidence. */
        out.workload_class = CIPHER_WL_UNKNOWN;
        confidence = 350;
    } else {
        out.workload_class = CIPHER_WL_UNKNOWN;
        confidence = 200;
    }

    /* Conditional signals — publish all observed signal flags honestly. */
    out.int4_weights_detected     = has_machete ? 1 : 0;
    out.long_context_detected     = long_ctx ? 1 : 0;     /* DERIVED (Step 6) */
    out.wide_matmul_detected      = wide_matmul ? 1 : 0;  /* raw, observability only */
    out.kv_cache_alloc_pattern_detected = kv_pattern ? 1 : 0;
    out.training_workload_detected= (has_backward || has_nccl) ? 1 : 0; /* uses fixed has_nccl */
    out.multi_tenant_detected     = multi_tenant ? 1 : 0; /* W.6 sub-C: authoritative kmod cohort (co_resident>=2) */
    out.same_model_across_tenants = 0;     /* W.4 derives from per-peer fingerprints (see file header) */

    /* W.1 (2026-05-27): bandwidth-bound regime derivation. Drives volt_engage
     * via apply_capability_flags. Two signals:
     *
     *   bw_bound_regime_detected = > 50% of GEMMs ran at n<=32 (small-batch
     *     decode-dominant). Captures both B=1 decode (P2/P3) and B=8 decode
     *     (P1). cipher-t43-envelope calibration: B=1->1000MHz / B=8->1600MHz
     *     both bandwidth-bound. B=32 is the inflection — at >32 the workload
     *     becomes compute-bound and VOLT regresses.
     *
     *   large_prefill_detected = max_n > 4096. Mistral-7B B=1 ctx=32K prefill
     *     hits max_n=32768. Decode workloads never observe max_n > LARGE_BATCH_THRESH
     *     (which is 32) — anything > 4096 is unambiguous prefill / sequence-pack.
     *
     * Together these gate VOLT engagement per cipher-t43-envelope: +57% tok/W
     * on bandwidth-bound decode (good) vs -14% on compute-bound prefill (bad).
     * Engaging only on bw_bound_regime + NOT engaging on large_prefill avoids
     * the -14% regression case. v1 used gemm_b1 (n<=1) and missed P1 B=8; v2
     * widens to gemm_small (n<=32). */
    uint64_t gemm_small = g_signals.gemm_small_count.load(std::memory_order_relaxed);
    uint64_t gemm_prefill = g_signals.gemm_prefill_count.load(std::memory_order_relaxed);
    uint64_t gemm_bf16 = g_signals.gemm_bf16_count.load(std::memory_order_relaxed);
    bool bw_bound = (gemm_tot > 0) && (gemm_small * 2 > gemm_tot);  /* >50% n<=32 */
    /* W.1 v3: ratio-based prefill detector. vLLM profiling spikes max_n to
     * 16384 even on decode workloads, so the lifetime-max_n threshold
     * (max_n>4096) misfires. Use ratio: >10% of GEMMs in prefill-shape
     * regime means the actual workload is prefill-heavy. */
    bool large_prefill = (gemm_tot > 0) && (gemm_prefill * 10 > gemm_tot);
    /* W.2: bf16 weight detection. >10% bf16-Atype GEMMs => bf16 model. */
    bool bf16_weights = (gemm_tot > 0) && (gemm_bf16 * 10 > gemm_tot);
    out.bw_bound_regime_detected = bw_bound ? 1 : 0;
    out.large_prefill_detected   = large_prefill ? 1 : 0;
    out.bf16_weights_detected    = bf16_weights ? 1 : 0;

    out.classification_confidence = confidence;

    apply_capability_flags(out);
    return out.workload_class;
}

const char *class_names[] = {
    "UNKNOWN",
    "A1_AGENTIC_MULTI_TENANT",
    "A2_CONTINUOUS_BATCHED",
    "A3_SINGLE_TENANT_STREAM",
    "A4_BATCH_INFERENCE",
    "B1_PRE_TRAINING",
    "B2_FINE_TUNING",
    "C1_RAG_LONG_CONTEXT",
};

/* K.1.5 Step 1 per-1000-launches WorkloadSignals snapshot. Dumps every
 * relevant aggregator field for empirical Tables 1-5 analysis. */
static void diag_log_snapshot(uint64_t obs) {
    cipher_log("K1-DIAG-SNAPSHOT ts=%lu obs=%lu "
               "gemm_total=%lu gemm_b1=%lu gemm_lg=%lu max_m=%u max_n=%u max_k=%u "
               "machete=%lu awq=%lu gptq=%lu flash=%lu rms=%lu silu_gelu=%lu "
               "bwd=%lu opt=%lu nccl_k=%lu "
               "capture=%d nccl_lib=%d "
               "alloc_total_bytes=%lu alloc_large=%lu alloc_medium=%lu alloc_small=%lu alloc_calls=%lu "
               "procs=%u "
               "fn_cache hits=%lu via_reg=%lu nomatch=%lu fail=%lu negcache=%lu overflow=%lu "
               "reg_map calls=%lu inserts=%lu dups=%lu full_skips=%lu",
               (unsigned long)diag_now_ns(),
               (unsigned long)obs,
               (unsigned long)g_signals.gemm_total.load(std::memory_order_relaxed),
               (unsigned long)g_signals.gemm_b1_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.gemm_large_batch_count.load(std::memory_order_relaxed),
               g_signals.max_m.load(std::memory_order_relaxed),
               g_signals.max_n.load(std::memory_order_relaxed),
               g_signals.max_k.load(std::memory_order_relaxed),
               (unsigned long)g_signals.machete_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.awq_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.gptq_marlin_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.flash_attn_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.rms_norm_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.silu_gelu_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.backward_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.optimizer_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.nccl_count.load(std::memory_order_relaxed),
               g_signals.capture_seen.load(std::memory_order_relaxed),
               g_signals.nccl_lib_loaded.load(std::memory_order_relaxed),
               (unsigned long)g_signals.alloc_total_bytes.load(std::memory_order_relaxed),
               (unsigned long)g_signals.alloc_large_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.alloc_medium_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.alloc_small_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.alloc_call_count.load(std::memory_order_relaxed),
               g_signals.concurrent_processes_estimate.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_hits.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_resolved_via_reg.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_resolved_no_match.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_resolution_failed.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_negative_cached.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_probe_overflow.load(std::memory_order_relaxed),
               (unsigned long)g_reg_map_calls.load(std::memory_order_relaxed),
               (unsigned long)g_reg_map_inserts.load(std::memory_order_relaxed),
               (unsigned long)g_reg_map_dup_returns.load(std::memory_order_relaxed),
               (unsigned long)g_reg_map_full_skips.load(std::memory_order_relaxed));
}

void emit_classify_log(const cipher_workload_profile &p) {
    /* W.6 sub-C: read the cached co-residence snapshot (fp=0 => query-only,
     * returns the throttle cache; no extra ioctl) and render the live peer
     * fingerprints. W.4 POOL consumes (tgid, model_fingerprint) per peer. */
    struct cipher_cohort_peer peers[8];
    uint32_t n_total = 0;
    uint32_t got = cipher_rt_coresidence_update(0, peers, 8, &n_total);
    (void)got;
    char peerbuf[256];
    {
        int off = 0;
        uint32_t shown = n_total < 8 ? n_total : 8;
        for (uint32_t i = 0; i < shown && off < (int)sizeof(peerbuf) - 32; i++)
            off += snprintf(peerbuf + off, sizeof(peerbuf) - off,
                            "%s%u:0x%016lx", i ? "," : "",
                            peers[i].tgid, (unsigned long)peers[i].model_fingerprint);
        if (n_total > shown)
            snprintf(peerbuf + off, sizeof(peerbuf) - off, ",+%u", n_total - shown);
        if (n_total == 0)
            snprintf(peerbuf, sizeof(peerbuf), "none");
    }
    cipher_log("CLASSIFY-CORES: co_resident=%u multi=%u peers=[%s]",
               p.co_resident_count, p.multi_tenant_detected, peerbuf);
    /* W.4a coalesce-eligibility (decision-only; transport=W.4b). */
    cipher_log("CLASSIFY-POOL: eligible_groups=%lu solo=%lu distinct_fp_rejected=%lu "
               "group_size=%u blocked=%lu",
               (unsigned long)cipher_rt_pool_eligible_groups(),
               (unsigned long)cipher_rt_pool_solo(),
               (unsigned long)cipher_rt_pool_distinct_fp_rejected(),
               cipher_rt_pool_current_group_size(),
               (unsigned long)cipher_rt_pool_blocked_by_correctness());

    /* K.1.5 Bug #5 diagnostic: include fn_cache counter values in CLASSIFY
     * log so worker-subprocess counter values are visible without
     * cross-process IPC. Removed at K.1 close. */
    cipher_log("CLASSIFY: workload=%s confidence=%u obs=%u "
               "[int4=%u bf16=%u long_ctx=%u training=%u multi=%u marlin_engage=%u] "
               "[volt_engage=%u bw_bound=%u large_prefill=%u gemm_small=%lu gemm_bf16=%lu gemm_total=%lu max_n=%u] "
               "[marlin_bf16_obs=%lu marlin_bf16_sub=%lu mach_intercepts=%lu mach_sub=%lu koop_bf16_obs=%lu koop_max_k=%lu koop_max_n=%lu] "
               "[attn_p1=%lu attn_p2=%lu attn_p5=%lu attn_p6=%lu] "
               "[model_fp=0x%016lx] "
               "[fn_cache hits=%lu via_reg=%lu via_cu_kernel=%lu nomatch=%lu fail=%lu negcache=%lu recovered=%lu overflow=%lu] "
               "[reg_map calls=%lu inserts=%lu dups=%lu full_skips=%lu]",
               class_names[(unsigned)p.workload_class],
               p.classification_confidence,
               p.observations_at_classify,
               p.int4_weights_detected,
               p.bf16_weights_detected,
               p.long_context_detected,
               p.training_workload_detected,
               p.multi_tenant_detected,
               p.marlin_engage,
               (unsigned)p.volt_engage,
               (unsigned)p.bw_bound_regime_detected,
               (unsigned)p.large_prefill_detected,
               (unsigned long)g_signals.gemm_small_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.gemm_bf16_count.load(std::memory_order_relaxed),
               (unsigned long)g_signals.gemm_total.load(std::memory_order_relaxed),
               (unsigned)g_signals.max_n.load(std::memory_order_relaxed),
               cipher_rt_marlin_calls_bf16_observed(),
               cipher_rt_marlin_calls_bf16_substituted(),
               cipher_rt_machete_intercepts(),
               cipher_rt_machete_substituted(),
               cipher_rt_koopman_bf16_observed(),
               cipher_rt_koopman_bf16_max_k(),
               cipher_rt_koopman_bf16_max_n(),
               cipher_rt_attn_p1_intercepts(),
               cipher_rt_attn_p2_intercepts(),
               cipher_rt_attn_p5_intercepts(),
               cipher_rt_attn_p6_intercepts(),
               cipher_workload_model_fingerprint(),
               (unsigned long)g_signals.fn_cache_hits.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_resolved_via_reg.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_resolved_via_cu_kernel_get_name.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_resolved_no_match.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_resolution_failed.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_negative_cached.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_recovered_via_retry.load(std::memory_order_relaxed),
               (unsigned long)g_signals.fn_cache_probe_overflow.load(std::memory_order_relaxed),
               (unsigned long)g_reg_map_calls.load(std::memory_order_relaxed),
               (unsigned long)g_reg_map_inserts.load(std::memory_order_relaxed),
               (unsigned long)g_reg_map_dup_returns.load(std::memory_order_relaxed),
               (unsigned long)g_reg_map_full_skips.load(std::memory_order_relaxed));
}

void publish_profile(const cipher_workload_profile &p) {
    int cur = g_signals.profile_current.load(std::memory_order_relaxed);
    if (cur == 0) {
        g_signals.profile_b = p;
        g_signals.profile_current.store(1, std::memory_order_release);
    } else {
        g_signals.profile_a = p;
        g_signals.profile_current.store(0, std::memory_order_release);
    }
}

}  /* anonymous namespace */

/* === Public API ========================================================= */

extern "C" int cipher_workload_detect_init(void)
{
    int already = g_signals.initialized.exchange(1, std::memory_order_acquire);
    if (already) return 0;

    /* Init both profile slots to UNKNOWN with VOLT-only conservative default. */
    cipher_workload_profile init_p = {};
    init_p.workload_class = CIPHER_WL_UNKNOWN;
    init_p.classification_confidence = 0;
    init_p.classified_at_ns = monotonic_ns();
    apply_capability_flags(init_p);
    g_signals.profile_a = init_p;
    g_signals.profile_b = init_p;
    g_signals.profile_current.store(0, std::memory_order_release);

    /* Try to resolve cuFuncGetName early so the first hot-path miss is not
     * the resolution path. Non-fatal if libcuda isn't loaded yet. */
    (void)resolve_cu_funcname();

    /* K.1.6 v4: background classifier timer thread.
     *
     * Why a thread and not atexit: vLLM EngineCore subprocess exits via
     * os._exit() (POSIX _exit) which BYPASSES atexit handlers. The thread
     * fires classify every 1s when observe is active.  classify_internal's
     * recent-idle gate (active_span < 20s + since_last > 2s) promotes to
     * UNKNOWN for E2/E3-style warmup-only cells while leaving real workloads
     * (active_span > 20s) untouched.
     *
     * v5 (2026-05-27) fix: vLLM 0.21 EngineCore is spawned via fork() (no
     * exec); detached threads do NOT survive fork — only the calling thread
     * continues in the child. We register pthread_atfork so the EngineCore
     * child re-spawns its own classifier timer.
     *
     * Cost: 1 std::thread, 1s sleep + 1 atomic load + at most 1 classify per
     * second. Pure CPU work; no CUDA calls. Detached so process exit doesn't
     * block on it. */
    auto spawn_classifier_timer = [](){
        std::thread([](){
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (!g_signals.initialized.load(std::memory_order_acquire)) continue;
                uint64_t obs = g_signals.total_observations.load(std::memory_order_relaxed);
                if (obs > 0) {
                    (void)cipher_workload_classify();
                    /* W.1 (2026-05-27): drive classifier-side VOLT engagement.
                     * cipher_rt_volt_classifier_poll() is a no-op unless VOLT
                     * is in ARMED mode (env unset / auto). Pure CPU work on
                     * the fast path; only the first transition ARMED -> ACTIVE
                     * actually touches NVML. */
                    cipher_rt_volt_classifier_poll();
                }
            }
        }).detach();
    };
    spawn_classifier_timer();
    /* pthread_atfork in-child handler captures the spawn closure via a
     * static thunk (atfork expects a C function ptr, no closure). */
    static auto s_after_fork_in_child = spawn_classifier_timer;
    pthread_atfork(nullptr, nullptr, [](){
        s_after_fork_in_child();
    });

    cipher_log("WORKLOAD-CLASSIFIER: init complete (init class=UNKNOWN/VOLT-only)");
    return 0;
}

extern "C" void cipher_workload_observe_launch_v2(
    const void *fn, uint32_t grid_total, uint32_t block_total, void *stream, uint32_t cbid)
{
    uint64_t obs = g_signals.total_observations.fetch_add(1, std::memory_order_relaxed) + 1;
    /* K.1.6 Sub-step 1.1 idle-gate: record first observation timestamp so
     * classify_internal can compute observations_per_second. Atomic CAS so
     * only the FIRST observer fills the field; subsequent calls are cheap. */
    uint64_t now_ns = diag_now_ns();
    if (obs == 1) {
        uint64_t expected = 0;
        g_signals.first_obs_ns.compare_exchange_strong(expected, now_ns,
            std::memory_order_release, std::memory_order_relaxed);
    }
    /* K.1.6 v2 recent-idle gate: relaxed store every launch; cheap (no CAS). */
    g_signals.last_observe_ns.store(now_ns, std::memory_order_relaxed);
    /* W.2 (2026-05-27) Sub-step 7: Machete intercept retry hook. Once per
     * 1024 observed launches, attempt to arm the Machete GOT-patch. vLLM
     * lazy-loads _C.abi3.so when AWQ Machete is first used, so dlsym at
     * cuInit-time often fails. The retry tick catches it once loaded.
     * When armed, the function early-returns (single atomic load). */
    if ((obs & 1023) == 0) {
        (void)cipher_rt_machete_intercept_init();
        /* W.5 (2026-05-28): 6-pattern attention intercept retry. vLLM
         * lazy-loads _vllm_fa3_C / _C / _flashmla_C; arm once visible. */
        (void)cipher_rt_attn_6pattern_init();
    }

    uint32_t bits = lookup_fn_bits(fn, cbid);

    /* K.1.5 Step 1 per-event diagnostic logging. Default OFF path is one
     * atomic load + branch-not-taken (~2-3ns); inside <10ns Memory #25 budget. */
    if (diag_on()) {
        diag_log_launch(fn, grid_total, block_total, stream, bits);
        if ((obs % DIAG_SNAPSHOT_INTERVAL) == 0) {
            diag_log_snapshot(obs);
        }
    }
    if (bits) {
        if (bits & KCB_MACHETE)
            g_signals.machete_count.fetch_add(1, std::memory_order_relaxed);
        if (bits & KCB_AWQ)
            g_signals.awq_count.fetch_add(1, std::memory_order_relaxed);
        if (bits & KCB_GPTQ_MARLIN)
            g_signals.gptq_marlin_count.fetch_add(1, std::memory_order_relaxed);
        if (bits & KCB_FLASH_ATTN)
            g_signals.flash_attn_count.fetch_add(1, std::memory_order_relaxed);
        if (bits & KCB_NCCL)
            g_signals.nccl_count.fetch_add(1, std::memory_order_relaxed);
        if (bits & KCB_BACKWARD)
            g_signals.backward_count.fetch_add(1, std::memory_order_relaxed);
        if (bits & KCB_OPTIMIZER)
            g_signals.optimizer_count.fetch_add(1, std::memory_order_relaxed);
        if (bits & KCB_RMS_NORM)
            g_signals.rms_norm_count.fetch_add(1, std::memory_order_relaxed);
        if (bits & (KCB_SILU | KCB_GELU))
            g_signals.silu_gelu_count.fetch_add(1, std::memory_order_relaxed);
    }

    /* Probabilistic reclassify trigger. */
    uint64_t since = g_signals.launches_since_classify.fetch_add(1, std::memory_order_relaxed) + 1;
    if (since >= RECLASSIFY_INTERVAL) {
        g_signals.launches_since_classify.store(0, std::memory_order_relaxed);
        (void)cipher_workload_classify();
    }
}

/* K.1.5 Step 1.5: ABI-additive v1 -> v2 redirect with cbid=0 sentinel. */
extern "C" void cipher_workload_observe_launch(
    const void *fn, uint32_t grid_total, uint32_t block_total, void *stream)
{
    cipher_workload_observe_launch_v2(fn, grid_total, block_total, stream, 0u);
}

/* K.1.5 Step 1.5 provenance tracking API. Called from 4 registration sites
 * (cuModuleGetFunction, cuLibraryGetKernel, cuKernelGetFunction,
 * __cudaRegisterFunction). Cold path; mutex-guarded set insert. */
extern "C" void cipher_workload_observe_register_site(const void *fn, uint32_t site_id)
{
    if (!fn) return;
    std::lock_guard<std::mutex> lk(g_provenance_mu);
    switch (site_id) {
    case CIPHER_WL_REG_SITE_CUDA_REGISTER_FN:
        g_prov_cuda_register_fn.insert(fn); break;
    case CIPHER_WL_REG_SITE_CU_MODULE_GET_FN:
        g_prov_cu_module_get_fn.insert(fn); break;
    case CIPHER_WL_REG_SITE_CU_LIBRARY_GET:
        g_prov_cu_library_get.insert(fn); break;
    case CIPHER_WL_REG_SITE_CU_KERNEL_GET_FN:
        g_prov_cu_kernel_get_fn.insert(fn); break;
    default: break;
    }
}

extern "C" uint64_t cipher_workload_fn_cache_recovered_via_retry(void)
{ return g_signals.fn_cache_recovered_via_retry.load(std::memory_order_relaxed); }

extern "C" uint64_t cipher_workload_fn_cache_resolved_via_cu_kernel_get_name(void)
{ return g_signals.fn_cache_resolved_via_cu_kernel_get_name.load(std::memory_order_relaxed); }

extern "C" void cipher_workload_observe_gemm(
    int m, int n, int k,
    int Atype, int Btype, int Ctype,
    const void *stream)
{
    (void)Ctype; (void)stream;

    /* PyTorch row-major cuBLAS convention: A=weight (M=out_features),
     * B=activation (N=batch), so N is the batch dim per
     * cipher_rt_marlin_actuator.c:81 and cipher_dispatch.cpp:229. */
    g_signals.gemm_total.fetch_add(1, std::memory_order_relaxed);
    if (n <= 1)
        g_signals.gemm_b1_count.fetch_add(1, std::memory_order_relaxed);
    /* W.1: small-batch decode signal. n<=32 cuBLAS view = PyTorch batch<=32. */
    if (n > 0 && n <= 32)
        g_signals.gemm_small_count.fetch_add(1, std::memory_order_relaxed);
    if (n > (int)LARGE_BATCH_THRESH)
        g_signals.gemm_large_batch_count.fetch_add(1, std::memory_order_relaxed);
    /* W.1 v3 (2026-05-27): prefill-shape GEMM signal (n>4096 = sequence-pack).
     * Ratio-based to avoid the lifetime-max_n contamination from vLLM's
     * one-shot profiling spike at startup. */
    if (n > 4096)
        g_signals.gemm_prefill_count.fetch_add(1, std::memory_order_relaxed);
    /* W.2 (2026-05-27): bf16 GEMM count for the bf16_weights_detected signal.
     * CUDA_R_16BF == 14 in cudaDataType_t. Increments once per bf16 cuBLAS
     * call regardless of which operand is bf16; ratio threshold > 10% of
     * total GEMMs in classify_internal. */
    if (Atype == 14 /* CUDA_R_16BF */)
        g_signals.gemm_bf16_count.fetch_add(1, std::memory_order_relaxed);

    /* Track max dims. Lossy update is fine (signals are heuristic). */
    uint32_t um = (uint32_t)(m > 0 ? m : 0);
    uint32_t un = (uint32_t)(n > 0 ? n : 0);
    uint32_t uk = (uint32_t)(k > 0 ? k : 0);
    uint32_t cur_m = g_signals.max_m.load(std::memory_order_relaxed);
    if (um > cur_m) g_signals.max_m.store(um, std::memory_order_relaxed);
    uint32_t cur_n = g_signals.max_n.load(std::memory_order_relaxed);
    if (un > cur_n) g_signals.max_n.store(un, std::memory_order_relaxed);
    uint32_t cur_k = g_signals.max_k.load(std::memory_order_relaxed);
    if (uk > cur_k) g_signals.max_k.store(uk, std::memory_order_relaxed);

    /* INT4/INT8 hint via cudaDataType: CUDA_R_8I=3, CUDA_R_8U=8 etc.
     * Kept as weak signal; primary INT4 detection is kernel-name (Machete). */
    if (Atype == 3 || Btype == 3) {
        g_signals.gemm_int4_dtype_count.fetch_add(1, std::memory_order_relaxed);
    }

    /* K.1.5 Step 1 per-event diagnostic. */
    if (diag_on()) diag_log_gemm(m, n, k, Atype, Btype, Ctype, stream);
}

extern "C" void cipher_workload_observe_alloc(size_t bytes, const void *device_ptr)
{
    (void)device_ptr;
    g_signals.alloc_call_count.fetch_add(1, std::memory_order_relaxed);
    g_signals.alloc_total_bytes.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
    if (bytes >= LARGE_ALLOC_BYTES) {
        g_signals.alloc_large_count.fetch_add(1, std::memory_order_relaxed);
    } else if (bytes < SMALL_ALLOC_BYTES) {
        g_signals.alloc_small_count.fetch_add(1, std::memory_order_relaxed);
    }
    /* K.1 Step 6 (Bug #1 fix): KV-block alloc class signal. Allocations
     * sized [256 KiB, 16 MiB] are characteristic of PagedAttention KV
     * block-table sub-allocations + activation scratch. NOT exclusively
     * KV (false positives are possible on activation scratch) but a
     * strong signal class. Cumulative count; classify_internal applies
     * the > KV_CACHE_PATTERN_THRESH threshold. */
    if (bytes >= MEDIUM_ALLOC_MIN_BYTES && bytes <= MEDIUM_ALLOC_MAX_BYTES) {
        g_signals.alloc_medium_count.fetch_add(1, std::memory_order_relaxed);
    }

    /* K.1.5 Step 1 per-event diagnostic. */
    if (diag_on()) diag_log_alloc(bytes, device_ptr);
}

extern "C" void cipher_workload_observe_capture_begin(void)
{
    g_signals.capture_seen.store(1, std::memory_order_relaxed);
    if (diag_on()) diag_log_capture_begin();
}

extern "C" void cipher_workload_observe_nccl_present(void)
{
    /* K.1.5 Step 1.6 Sub-step 2: dedup NCCL diagnostic log. The auto-repatch
     * path walks all DSOs on each fatbin registration -> ~900 NCCL log fires
     * per 60s run if libnccl is mapped. Set-once log via CAS on the
     * nccl_lib_loaded flag transition 0 -> 1; subsequent calls keep the flag
     * set (correctness preserved) but suppress duplicate log emit. */
    int prev = g_signals.nccl_lib_loaded.exchange(1, std::memory_order_relaxed);
    if (prev == 0 && diag_on()) diag_log_nccl();
}

extern "C" cipher_workload_class_t cipher_workload_classify(void)
{
    cipher_workload_profile p;
    cipher_workload_class_t cls = classify_internal(p);
    /* K.1.6 v7 hysteresis: emission is provisional until MIN_HIGH_CONF
     * classifications have accumulated. Counts the would-be high-conf
     * KNOWN classifies; before the threshold the class publishes as
     * UNKNOWN. Separates short warmup bursts (E2=9, E3=14 high-conf
     * classifies; below 50 → never crosses → UNKNOWN) from real workloads
     * (P1=900, P3=576; cross → KNOWN). Replaces the v2-v6 recent-idle gate
     * with a simpler always-on hysteresis. */
    if (cls != CIPHER_WL_UNKNOWN && p.classification_confidence > 500) {
        uint64_t new_count = g_signals.high_conf_emits.fetch_add(1, std::memory_order_relaxed) + 1;
        if (new_count < MIN_HIGH_CONF_CLASSIFY_COUNT) {
            p.workload_class = CIPHER_WL_UNKNOWN;
            p.classification_confidence = 250;
            apply_capability_flags(p);
            cls = CIPHER_WL_UNKNOWN;
        }
    }
    publish_profile(p);
    g_signals.total_classifications.fetch_add(1, std::memory_order_relaxed);
    emit_classify_log(p);
    return cls;
}

extern "C" const struct cipher_workload_profile *cipher_workload_profile_get(void)
{
    int cur = g_signals.profile_current.load(std::memory_order_acquire);
    return cur == 0 ? &g_signals.profile_a : &g_signals.profile_b;
}

extern "C" uint32_t cipher_workload_class_current(void)
{
    return (uint32_t)cipher_workload_profile_get()->workload_class;
}

extern "C" uint64_t cipher_workload_observations_count(void)
{
    return g_signals.total_observations.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_workload_classifications_count(void)
{
    return g_signals.total_classifications.load(std::memory_order_relaxed);
}

extern "C" uint32_t cipher_workload_confidence_current(void)
{
    return cipher_workload_profile_get()->classification_confidence;
}

extern "C" const char *cipher_workload_class_name(cipher_workload_class_t cls)
{
    unsigned idx = (unsigned)cls;
    if (idx >= sizeof(class_names)/sizeof(class_names[0])) return "INVALID";
    return class_names[idx];
}

/* W.6 sub-B (2026-05-28): substrate-side per-process model fingerprint.
 *
 * Replaces the crashing NR 27 plugin model_uuid path as the W.4 POOL
 * cross-tenant model-identity source (Anil adjudication: bypass the plugin;
 * model identity moves substrate-side per Memory #30).
 *
 * The fingerprint is a stable 64-bit hash of the model's dimensional GEMM
 * signature observed at the cuBLAS dispatch boundary:
 *   - max_k : largest cuBLAS K (input feature) seen — captures the
 *             intermediate_size (FFN) which is model-architecture-specific
 *             (Llama-3-8B 14336, Mistral-7B 14336, TinyLlama 5632, ...).
 *   - max_m : largest cuBLAS M (output feature) seen — captures vocab_size
 *             at the LM head (Llama-3 128256, Mistral 32768) which
 *             discriminates same-intermediate models.
 *   - bf16/int4 dtype bit — distinguishes a bf16 deploy from a quantized one.
 *
 * Two processes running the SAME model observe the SAME (max_k, max_m, dtype)
 * and compute an identical fingerprint → W.4 may legally coalesce their
 * GEMMs. Different models differ in at least one dimension → distinct
 * fingerprint → no coalesce. This is observation-only (Memory #30 substrate-
 * line; no application coupling, no kmod NR 27 plugin path). Stable only
 * after enough GEMMs observed to have seen the LM head + FFN shapes — the
 * accessor returns 0 (UNKNOWN) until gemm_total crosses a small warmup. */
extern "C" uint64_t cipher_workload_model_fingerprint(void)
{
    uint64_t gemm_tot = g_signals.gemm_total.load(std::memory_order_relaxed);
    if (gemm_tot < 64) return 0;   /* warmup: LM head / FFN not yet observed */
    uint64_t mk  = (uint64_t)g_signals.max_k.load(std::memory_order_relaxed);
    uint64_t mm  = (uint64_t)g_signals.max_m.load(std::memory_order_relaxed);
    uint64_t bf16 = g_signals.gemm_bf16_count.load(std::memory_order_relaxed) ? 1u : 0u;
    uint64_t int4 = g_signals.gemm_int4_dtype_count.load(std::memory_order_relaxed) ? 2u : 0u;
    /* FNV-style mix; stable across processes for identical (mk,mm,dtype). */
    uint64_t h = 1469598103934665603ULL;
    h = (h ^ mk)  * 1099511628211ULL;
    h = (h ^ mm)  * 1099511628211ULL;
    h = (h ^ (bf16 | int4)) * 1099511628211ULL;
    return h ? h : 1;   /* never collide with 0 = UNKNOWN sentinel */
}

/* K.1.5 Step 0 Bug #5 fix v4: registration-time map population.
 * Called from cipher_intercept_cudart.cpp:1786 __cudaRegisterFunction shim.
 * Insert host_ptr -> name; name pointer stored directly (registration string
 * is process-lifetime stable in cubin metadata). Open-addressing, linear
 * probe; full map (>= REG_MAP_PROBE_MAX collisions) drops silently after
 * incrementing g_reg_map_full_skips. */
extern "C" void cipher_workload_register_kernel(const void *host_ptr, const char *name)
{
    if (!host_ptr || !name) return;
    g_reg_map_calls.fetch_add(1, std::memory_order_relaxed);

    /* env-gated first-N log (init once). */
    if (!g_log_registered_inited.load(std::memory_order_relaxed)) {
        int already = g_log_registered_inited.exchange(1, std::memory_order_relaxed);
        if (!already) {
            const char *e = getenv("CIPHER_K1_LOG_REGISTERED");
            g_log_registered_on.store((e && e[0] == '1') ? 1 : 0,
                                       std::memory_order_relaxed);
        }
    }

    uint32_t slot = reg_map_hash_slot(host_ptr);
    for (uint32_t i = 0; i < REG_MAP_PROBE_MAX; ++i) {
        RegMapEntry &e = g_reg_map[(slot + i) & (REG_MAP_SLOTS - 1)];
        const void *seen = e.host_ptr.load(std::memory_order_acquire);
        if (seen == host_ptr) {
            /* Already registered (probably re-registration on module reload). */
            g_reg_map_dup_returns.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (seen == nullptr) {
            const void *expected = nullptr;
            if (e.host_ptr.compare_exchange_strong(expected, host_ptr,
                    std::memory_order_release, std::memory_order_acquire)) {
                e.name.store(name, std::memory_order_release);
                uint64_t n = g_reg_map_inserts.fetch_add(1, std::memory_order_relaxed);
                if (g_log_registered_on.load(std::memory_order_relaxed) && n < 100) {
                    cipher_log("CLASSIFY-REGISTERED: host=%p name=%s", host_ptr, name);
                }
                return;
            }
            /* Lost the race; continue probing. */
            continue;
        }
    }
    g_reg_map_full_skips.fetch_add(1, std::memory_order_relaxed);
}

extern "C" uint64_t cipher_workload_fn_cache_resolved_via_registration(void)
{ return g_signals.fn_cache_resolved_via_reg.load(std::memory_order_relaxed); }

extern "C" uint64_t cipher_workload_fn_cache_negative_cached(void)
{ return g_signals.fn_cache_negative_cached.load(std::memory_order_relaxed); }

/* K.1.5 Bug #5 diagnostic accessors per advisor 2026-05-27. */
extern "C" uint64_t cipher_workload_fn_cache_hits(void)
{ return g_signals.fn_cache_hits.load(std::memory_order_relaxed); }

extern "C" uint64_t cipher_workload_fn_cache_resolved_no_match(void)
{ return g_signals.fn_cache_resolved_no_match.load(std::memory_order_relaxed); }

extern "C" uint64_t cipher_workload_fn_cache_resolution_failed(void)
{ return g_signals.fn_cache_resolution_failed.load(std::memory_order_relaxed); }

extern "C" uint64_t cipher_workload_fn_cache_probe_overflow(void)
{ return g_signals.fn_cache_probe_overflow.load(std::memory_order_relaxed); }

/* K.1.5 Bug #5 diagnostic: print one log line per unique kernel name that
 * resolved but didn't match any classify_kernel_name pattern. Enabled via
 * CIPHER_K1_LOG_UNMATCHED env var (default off, so production is quiet).
 * The print itself happens inside lookup_fn_bits only on resolved_no_match
 * path. The mechanism is a separate small bloom-style dedup so we log each
 * unique name only once (not per-launch). */
static std::atomic<int> g_log_unmatched_inited{0};
static std::atomic<int> g_log_unmatched_enabled{0};

extern "C" int cipher_workload_log_unmatched_enabled(void)
{
    if (!g_log_unmatched_inited.load(std::memory_order_relaxed)) {
        int already = g_log_unmatched_inited.exchange(1, std::memory_order_relaxed);
        if (!already) {
            const char *e = getenv("CIPHER_K1_LOG_UNMATCHED");
            int on = (e && e[0] == '1') ? 1 : 0;
            g_log_unmatched_enabled.store(on, std::memory_order_relaxed);
        }
    }
    return g_log_unmatched_enabled.load(std::memory_order_relaxed);
}
