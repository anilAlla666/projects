/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_workload_detect.h -- K.1 Workload Classifier public API.
 *
 * Architectural keystone for CIPHER v1 workload-adaptive product per the
 * V1_CAPABILITY_AUDIT_AND_PRODUCT_ARCHITECTURE.md re-anchor: customer
 * installs .deb -> Classifier detects workload type from driver-level
 * signals -> Profile Activator engages capability subset -> all 5
 * Memory #1 goals deliver on this workload class.
 *
 * K.1 scope: build the Classifier substrate; observe at intercept points;
 * log profile; export diagnostic accessors. K.1 does NOT wire actuators
 * to read CipherProfile flags -- that is K.2 + W-series scope per
 * audit Section D substep sequence.
 *
 * Discipline:
 *   - Memory #9 framework-agnostic: only driver-level signals; no
 *     vLLM-specific, no transformers-specific logic.
 *   - Memory #20 ABI-additive: existing dispatch paths unchanged; new
 *     observe() calls added at intercept points.
 *   - Memory #25 per-launch budget <10ns: hot-path observe_launch() uses
 *     per-fn-pointer kernel-name cache (lazy cuFuncGetName once per
 *     unique fn ptr; cached bitfield thereafter).
 */
#ifndef CIPHER_WORKLOAD_DETECT_H
#define CIPHER_WORKLOAD_DETECT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Workload classes per audit Section B (7 customer-surface classes + UNKNOWN). */
typedef enum cipher_workload_class {
    CIPHER_WL_UNKNOWN                = 0,
    CIPHER_WL_A1_AGENTIC_MULTI_TENANT= 1,
    CIPHER_WL_A2_CONTINUOUS_BATCHED  = 2,
    CIPHER_WL_A3_SINGLE_TENANT_STREAM= 3,
    CIPHER_WL_A4_BATCH_INFERENCE     = 4,
    CIPHER_WL_B1_PRE_TRAINING        = 5,
    CIPHER_WL_B2_FINE_TUNING         = 6,
    CIPHER_WL_C1_RAG_LONG_CONTEXT    = 7,
} cipher_workload_class_t;

/* CipherProfile: capability-engagement decision struct emitted by the
 * classifier per audit Section A capability #5. Read by K.2 + W-series
 * actuator wiring; ignored in K.1 itself. */
struct cipher_workload_profile {
    cipher_workload_class_t workload_class;
    uint64_t classified_at_ns;
    uint32_t classification_confidence;   /* 0-1000 scaled int */
    uint32_t observations_at_classify;

    /* Actuator engagement flags (decision-table driven). */
    uint8_t volt_engage;
    uint8_t marlin_engage;
    uint8_t koopman_engage;
    uint8_t kv_dedup_engage;
    uint8_t weight_arena_engage;
    uint8_t sm_packer_engage;
    uint8_t per_tenant_routing;
    uint8_t pool_substrate;
    uint8_t flash_attn_intercept;
    uint8_t nccl_tuner;
    uint8_t persistent_kernel;
    uint8_t speculative_decode;
    uint8_t graph_engine;
    uint8_t flop_telemetry;
    uint8_t audit_chain;

    /* Workload-specific conditional signals (observable, not derived). */
    uint8_t int4_weights_detected;
    uint8_t long_context_detected;           /* derived: kv_cache_pattern AND max_n > 32K */
    uint8_t training_workload_detected;
    uint8_t multi_tenant_detected;            /* W.6 sub-C: authoritative = (co_resident_count >= 2) from kmod cohort */
    uint8_t same_model_across_tenants;        /* K.1 always 0; W.4 derives from per-peer model fingerprints */

    /* K.1 Step 6 (Bug #1 fix): semantically-honest signal split. The prior
     * "long_context_detected" was unconditionally derived from cuBLAS GEMM
     * M/N dims, which means Llama-3-class models (intermediate_size=14336)
     * triggered long_ctx on every forward regardless of prompt length.
     * That field is now derived from kv_cache_alloc_pattern AND max_n > 32K
     * threshold. The raw "wide matmul observed" signal is kept here as a
     * separate observable, not used as a C1 classification gate. */
    uint8_t wide_matmul_detected;            /* raw: cuBLAS max_m > 8K or max_n > 16K */
    uint8_t kv_cache_alloc_pattern_detected; /* cudaMalloc count in [256KB,16MB] > 100 */

    /* W.1 (2026-05-27): bandwidth-bound regime signals driving volt_engage.
     * Both ABI-additive (consume the _pad byte and one reserved[] slot).
     * Producer: classify_internal. Consumer: apply_capability_flags +
     * VOLT actuator gate per Memory #26 brain-muscle. */
    uint8_t bw_bound_regime_detected;        /* gemm_b1_ratio > 0.5 (decode-dominant) */
    uint8_t large_prefill_detected;          /* max_n > 4096 (compute-bound prefill) */
    /* W.2 (2026-05-27): bf16 weight detection. Mirrors int4_weights_detected.
     * Producer: cuBLAS observe path (Atype == CUDA_R_16BF). Consumer:
     * apply_capability_flags extends marlin_engage to fire on bf16 OR int4. */
    uint8_t bf16_weights_detected;           /* >= 10% of GEMMs use Atype=CUDA_R_16BF */
    uint8_t _pad_w2;

    /* W.6 sub-C (2026-05-28): authoritative GPU co-residence count from the
     * kmod cohort registry (cipher_rt_coresidence_update -> CIPHER_COHORT_QUERY
     * nr 31). Distinct LIVE tenant processes on this GPU, INCLUDING self
     * (>= 1 once warmed up). multi_tenant_detected is now derived from this
     * (>= 2) rather than the K.1-era /proc-maps scan. Carved from reserved[]
     * (ABI size unchanged). W.4 POOL reads this + cipher_workload_model_
     * fingerprint() per peer for legal cross-tenant coalescing groups. */
    uint32_t co_resident_count;
    uint32_t _pad_w6c;
    uint64_t reserved[7];                    /* W-series ABI-additive */
};

/* === Init / lifecycle ==================================================== */

/* Init the workload classifier. Idempotent. Called from cipher_inject.c
 * cipher_v2_init_body() after cipher_v2_cupti_init(). Returns 0 on success. */
int cipher_workload_detect_init(void);

/* === Hot-path observation API ============================================ */

/* Per-launch observation. Called from CUPTI kernel launch callback after
 * cipher_rt_smp_observe. Hot-path budget <10ns p50 (Memory #25). The fn
 * pointer is cached -> name-class bitfield on first miss only. */
void cipher_workload_observe_launch(
    const void *fn,
    uint32_t grid_total,
    uint32_t block_total,
    void     *stream);

/* K.1.5 Step 1.5 ABI-additive v2: per-launch observe with CUPTI callback ID.
 * v1 retained for ABI stability; v2 enables empirical CBID classification
 * for the H1/H2/H3 unresolved-residue disambiguator. cbid is the CUPTI
 * CUpti_CallbackId at the callback site (13 = cuLaunchKernel, 211 =
 * cuLaunchKernelEx, etc.). Pass 0 if unknown. */
void cipher_workload_observe_launch_v2(
    const void *fn,
    uint32_t grid_total,
    uint32_t block_total,
    void     *stream,
    uint32_t cbid);

/* K.1.5 Step 1.5 registration-site provenance tracking. Called from the
 * 4 registration shims (cuModuleGetFunction, cuLibraryGetKernel,
 * cuKernelGetFunction, __cudaRegisterFunction) to mark which path(s)
 * saw a given fn pointer at registration time. Used by the unresolved
 * disambiguator to bucket H1 (CBID gap) vs H2 (no registration path
 * ever saw it) vs H3 (registration succeeded but lookup raced/missed). */
enum cipher_workload_reg_site {
    CIPHER_WL_REG_SITE_NONE              = 0,
    CIPHER_WL_REG_SITE_CUDA_REGISTER_FN  = 1,  /* __cudaRegisterFunction */
    CIPHER_WL_REG_SITE_CU_MODULE_GET_FN  = 2,  /* cuModuleGetFunction */
    CIPHER_WL_REG_SITE_CU_LIBRARY_GET    = 3,  /* cuLibraryGetKernel */
    CIPHER_WL_REG_SITE_CU_KERNEL_GET_FN  = 4,  /* cuKernelGetFunction */
};
void cipher_workload_observe_register_site(const void *fn, uint32_t site_id);

/* K.1.5 Step 1.5 retry-recovery counter accessor (new counter populated
 * when the negative-cache poison branch's retry chain recovers a name). */
uint64_t cipher_workload_fn_cache_recovered_via_retry(void);

/* Per-cuBLAS-GEMM observation. Called from cipher_rt_cublas_shim.c after
 * the cipher_dbg log line, before cipher_rt_matmul_dispatch. Cheap
 * (~30ns; cuBLAS path is microseconds). */
void cipher_workload_observe_gemm(
    int m, int n, int k,
    int Atype, int Btype, int Ctype,
    const void *stream);

/* Per-cudaMalloc observation. Called from cudaMalloc shim after the real
 * allocation returns. Cheap (allocation is microseconds anyway). */
void cipher_workload_observe_alloc(size_t bytes, const void *device_ptr);

/* cuStreamBeginCapture observation. Called once per capture begin from
 * the existing intercept hook. Marks capture_seen flag. */
void cipher_workload_observe_capture_begin(void);

/* NCCL library load observation (auto-repatch path already enumerates
 * loaded sonames; we sniff libnccl SONAMEs on the dlopen callback path).
 * Idempotent — sets nccl_lib_loaded flag. */
void cipher_workload_observe_nccl_present(void);

/* K.1.5 Step 0 Bug #5 fix v4: host_ptr -> device_name registration map.
 * Called from cipher_intercept_cudart.cpp:1786 __cudaRegisterFunction shim
 * at module-load time. PyTorch + vLLM launch kernels via cudaLaunchKernel
 * (runtime API), passing host_fn (a __device_stub__ pointer). cuFuncGetName
 * cannot resolve host pointers — only driver-API CUfunction handles. This
 * registration map captures the host_ptr -> device_name mapping at the
 * point CUDA registers the kernel, so lookup_fn_bits can resolve names
 * for runtime-API-launched kernels too.
 *
 * name is the deviceName arg from __cudaRegisterFunction — stable for
 * process lifetime (embedded in cubin metadata while module is loaded);
 * we store the pointer directly, no strdup. */
void cipher_workload_register_kernel(const void *host_ptr, const char *name);

/* K.1.5 Step 0 diagnostic: separate counter for registration-map coverage.
 * Distinguishes "resolved via map" from "resolved via cuFuncGetName". */
uint64_t cipher_workload_fn_cache_resolved_via_registration(void);

/* === Classification trigger (re-run every N observations) ================ */

/* Re-evaluate classification using accumulated signals. Called from
 * cipher_workload_observe_launch() probabilistically (every 100 obs)
 * and from external scripts via diagnostic accessor.
 * Returns the freshly-classified class. */
cipher_workload_class_t cipher_workload_classify(void);

/* === Diagnostic accessors (ABI-additive exports) ========================= */

/* Snapshot the current profile. Caller-owned copy; safe to read from any
 * thread (atomic-snapshot via lock-free RCU-style swap). */
const struct cipher_workload_profile *cipher_workload_profile_get(void);

/* Lightweight scalar accessors for python ctypes probes (Step 4). */
uint32_t cipher_workload_class_current(void);
uint64_t cipher_workload_observations_count(void);
uint64_t cipher_workload_classifications_count(void);
uint32_t cipher_workload_confidence_current(void);

/* Returns a string name for a class (static storage; do not free). */
const char *cipher_workload_class_name(cipher_workload_class_t cls);

/* W.6 sub-B (2026-05-28): substrate-side per-process model fingerprint.
 * Stable 64-bit hash of the model's dimensional GEMM signature
 * (max_k, max_m, dtype). Two processes running the same model compute the
 * same value; W.4 POOL compares fingerprints cross-tenant to decide legal
 * GEMM coalescing WITHOUT the crashing NR 27 plugin model_uuid path.
 * Returns 0 (UNKNOWN) until enough GEMMs observed (warmup). */
uint64_t cipher_workload_model_fingerprint(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_WORKLOAD_DETECT_H */
