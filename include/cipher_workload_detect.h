// =============================================================================
// CIPHER — workload detector
// Observes kernel-launch + GEMM-shape stream at the driver boundary, classifies
// the workload type and exposes a CipherProfile of which optimizations are
// expected to help. Cheap observers always run; heavy substituters can read
// the profile to skip themselves on workloads they hurt (e.g. don't FP8 in
// EMBEDDINGS where the GEMMs are too small).
// =============================================================================
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CIPHER_WL_UNKNOWN = 0,
    CIPHER_WL_LLM_DECODE_SINGLE,
    CIPHER_WL_LLM_DECODE_BATCHED,
    CIPHER_WL_LLM_PREFILL,
    CIPHER_WL_LLM_SERVING_VLLM,
    CIPHER_WL_MULTI_TENANT,
    CIPHER_WL_EMBEDDINGS,
    CIPHER_WL_LORA_FINETUNING,
    CIPHER_WL_DIFFUSION,
    CIPHER_WL_SPEECH,
    CIPHER_WL_SPECULATIVE_DECODING,
    CIPHER_WL_AGENTIC,
    CIPHER_WL_BATCH_PROCESSING,
    CIPHER_WL_LONG_CONTEXT,
    CIPHER_WL_TORCH_COMPILE,
    CIPHER_WL_MOE_MODELS,
    CIPHER_WL_PREFIX_CACHING,
    CIPHER_WL_TRAINING_FULL,
    CIPHER_WL_MULTI_GPU_TP,
    CIPHER_WL_VISION,
    CIPHER_WL_MULTIMODAL,
    CIPHER_WL_CODE_GENERATION,
    CIPHER_WL_RAG_PIPELINE,
    CIPHER_WL_MODEL_SWITCH,
    CIPHER_WL_QUANTIZED_NATIVE,
    CIPHER_WL__COUNT
} CipherWorkloadType;

typedef struct {
    int koopman_o1;
    int fp8_compute;
    int substitute_v2;
    int fuse_rmsnorm;
    int fuse_swiglu;
    int persist_engine;
    int thermostat;
    int thermostat_lock_low;
    int fairness;
    int arbitrate;
    int graph_engine;
    int nccl_tuner;
    int flow_record;
    int flow_match;
    int flow_substitute;
    int carbon;
    int receipt;
    int comply;
    int trace;
    int validate;
    int adapt;
    int remember;
    int speculate;
    int predict;
    int determinism;
    int continuity;
    int loop_detect;
    int audit;
    int guard;
    int topology;
    int pipeline;
} CipherProfile;

void               cipher_workload_observe(const char* kernel_name,
                                           int M, int N, int K,
                                           int n_contexts);
CipherWorkloadType cipher_workload_classify(void);
CipherProfile      cipher_workload_profile(void);
const char*        cipher_workload_name(CipherWorkloadType t);
void               cipher_workload_report(void);
int                cipher_workload_enabled(void);

#ifdef __cplusplus
}
#endif
