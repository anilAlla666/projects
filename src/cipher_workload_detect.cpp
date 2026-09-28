// =============================================================================
// CIPHER — workload detector implementation
// Lock-free atomic counters keep the hot path under ~50ns. Classification
// happens lazily (every 100th observation) on the calling thread; the result
// is stored in a single atomic int. Reads of classify()/profile() are O(1).
// =============================================================================
#include "cipher_workload_detect.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

// ---------------------------------------------------------------------------
// Env gate (mirror of the existing env-once-init pattern)
// ---------------------------------------------------------------------------
extern "C" int cipher_workload_enabled(void) {
    static const bool s_on = []() {
        const char* env = getenv("CIPHER_WORKLOAD_DETECT");
        return !env || !*env || env[0] != '0';  // default ON
    }();
    return s_on ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Hot-path counters (no locks)
// ---------------------------------------------------------------------------
static std::atomic<uint64_t> g_obs_total{0};
static std::atomic<uint64_t> g_gemm_count{0};
// Most-recent GEMM K dimension — used by the driver-fusion path as a
// reasonable hidden_dim estimate (the K dim of an attention/FFN GEMM
// equals the model's hidden_size for nn.Linear-class projections).
static std::atomic<int>      g_last_gemm_k{0};
static std::atomic<uint64_t> g_attention_kernels{0};
static std::atomic<uint64_t> g_rmsnorm_kernels{0};
static std::atomic<uint64_t> g_silu_kernels{0};
static std::atomic<uint64_t> g_conv_kernels{0};
static std::atomic<uint64_t> g_nccl_kernels{0};
static std::atomic<uint64_t> g_backward_kernels{0};
static std::atomic<uint64_t> g_graph_capture_seen{0};
static std::atomic<uint64_t> g_speech_kernels{0};
static std::atomic<uint64_t> g_dequant_kernels{0};

// Shape histogram — a few coarse buckets are enough for classification
static std::atomic<uint64_t> g_n_eq_1{0};         // single-stream decode
static std::atomic<uint64_t> g_n_2_64{0};         // batched decode
static std::atomic<uint64_t> g_n_65_511{0};       // medium / encoder
static std::atomic<uint64_t> g_n_512_plus{0};     // prefill

static std::atomic<int>      g_max_n_contexts{1};
static std::atomic<int>      g_cached_class{(int)CIPHER_WL_UNKNOWN};

// Process counter — set externally via cipher_workload_observe(.., n_contexts)
// or auto-derived from /proc scan if 0 is passed.

// ---------------------------------------------------------------------------
// Cheap process-count probe — scans /proc for sibling python processes that
// have libcipher_hook.so mapped. Cached; refreshed every 1000 observations.
// ---------------------------------------------------------------------------
static int probe_cipher_process_count() {
    DIR* d = opendir("/proc");
    if (!d) return 1;
    int count = 0;
    struct dirent* e;
    while ((e = readdir(d))) {
        if (e->d_type != DT_DIR) continue;
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char map_path[256];
        snprintf(map_path, sizeof(map_path), "/proc/%s/maps", e->d_name);
        FILE* f = fopen(map_path, "r");
        if (!f) continue;
        char line[1024];
        bool has_hook = false;
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "libcipher_hook.so")) { has_hook = true; break; }
        }
        fclose(f);
        if (has_hook) count++;
    }
    closedir(d);
    return count > 0 ? count : 1;
}

// ---------------------------------------------------------------------------
// Substring check on (possibly null/short) kernel name
// ---------------------------------------------------------------------------
static bool name_has(const char* name, const char* needle) {
    if (!name || !needle) return false;
    return strstr(name, needle) != nullptr;
}

// ---------------------------------------------------------------------------
// Re-classify based on accumulated stats. Cheap; runs only every Nth call.
// ---------------------------------------------------------------------------
static CipherWorkloadType reclassify_locked() {
    uint64_t total       = g_obs_total.load(std::memory_order_relaxed);
    uint64_t attn        = g_attention_kernels.load(std::memory_order_relaxed);
    uint64_t conv        = g_conv_kernels.load(std::memory_order_relaxed);
    uint64_t nccl        = g_nccl_kernels.load(std::memory_order_relaxed);
    uint64_t bwd         = g_backward_kernels.load(std::memory_order_relaxed);
    uint64_t graph       = g_graph_capture_seen.load(std::memory_order_relaxed);
    uint64_t speech      = g_speech_kernels.load(std::memory_order_relaxed);
    uint64_t dequant     = g_dequant_kernels.load(std::memory_order_relaxed);
    uint64_t n1          = g_n_eq_1.load(std::memory_order_relaxed);
    uint64_t n_small     = g_n_2_64.load(std::memory_order_relaxed);
    uint64_t n_med       = g_n_65_511.load(std::memory_order_relaxed);
    uint64_t n_large     = g_n_512_plus.load(std::memory_order_relaxed);
    int      ctxs        = g_max_n_contexts.load(std::memory_order_relaxed);
    uint64_t gemms       = g_gemm_count.load(std::memory_order_relaxed);

    if (total < 50) return CIPHER_WL_UNKNOWN;

    // Strong signals first
    if (bwd > total / 100) {
        return (nccl > 0) ? CIPHER_WL_TRAINING_FULL : CIPHER_WL_LORA_FINETUNING;
    }
    if (speech > 0)                        return CIPHER_WL_SPEECH;
    if (conv > total / 50)                 return CIPHER_WL_DIFFUSION;
    if (dequant > total / 50)              return CIPHER_WL_QUANTIZED_NATIVE;
    if (graph > 0 && attn > 0)             return CIPHER_WL_LLM_SERVING_VLLM;
    if (ctxs >= 2)                         return CIPHER_WL_MULTI_TENANT;
    if (nccl > 0)                          return CIPHER_WL_MULTI_GPU_TP;
    if (n_large > gemms / 4)               return CIPHER_WL_LLM_PREFILL;
    if (n_small > gemms / 4)               return CIPHER_WL_LLM_DECODE_BATCHED;
    if (n1 > gemms / 4)                    return CIPHER_WL_LLM_DECODE_SINGLE;
    if (n_med > gemms / 4 && attn > 0)     return CIPHER_WL_VISION;
    if (gemms > 0 && attn == 0 && total < 5000) return CIPHER_WL_EMBEDDINGS;
    return CIPHER_WL_LLM_DECODE_BATCHED;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
extern "C" void cipher_workload_observe(const char* kernel_name,
                                        int M, int N, int K,
                                        int n_contexts)
{
    if (!cipher_workload_enabled()) return;

    cipher_op_inc(OP_WORKLOAD_OBSERVE);

    uint64_t seq = g_obs_total.fetch_add(1, std::memory_order_relaxed);

    if (M > 0 && K > 0) {
        g_gemm_count.fetch_add(1, std::memory_order_relaxed);
        if (N == 1)             g_n_eq_1.fetch_add(1, std::memory_order_relaxed);
        else if (N <= 64)       g_n_2_64.fetch_add(1, std::memory_order_relaxed);
        else if (N <= 511)      g_n_65_511.fetch_add(1, std::memory_order_relaxed);
        else                    g_n_512_plus.fetch_add(1, std::memory_order_relaxed);
        // Track the most-recent K dim — used by driver-fusion to learn
        // hidden_size at runtime. nn.Linear's transposed-A convention
        // makes K = input_dim = hidden_size for attention/FFN projections.
        g_last_gemm_k.store(K, std::memory_order_relaxed);
    }

    if (kernel_name) {
        if (name_has(kernel_name, "flash_") || name_has(kernel_name, "fmha")
            || name_has(kernel_name, "attention") || name_has(kernel_name, "MemEffAttention"))
            g_attention_kernels.fetch_add(1, std::memory_order_relaxed);
        if (name_has(kernel_name, "rms_norm") || name_has(kernel_name, "MeanOps")
            || name_has(kernel_name, "RMSNorm"))
            g_rmsnorm_kernels.fetch_add(1, std::memory_order_relaxed);
        if (name_has(kernel_name, "silu") || name_has(kernel_name, "swiglu")
            || name_has(kernel_name, "SiLU"))
            g_silu_kernels.fetch_add(1, std::memory_order_relaxed);
        if (name_has(kernel_name, "conv2d") || name_has(kernel_name, "implicit_gemm")
            || name_has(kernel_name, "wgrad"))
            g_conv_kernels.fetch_add(1, std::memory_order_relaxed);
        if (name_has(kernel_name, "nccl"))
            g_nccl_kernels.fetch_add(1, std::memory_order_relaxed);
        if (name_has(kernel_name, "backward") || name_has(kernel_name, "bwd"))
            g_backward_kernels.fetch_add(1, std::memory_order_relaxed);
        if (name_has(kernel_name, "whisper") || name_has(kernel_name, "wav2vec"))
            g_speech_kernels.fetch_add(1, std::memory_order_relaxed);
        if (name_has(kernel_name, "dequant") || name_has(kernel_name, "awq")
            || name_has(kernel_name, "gptq") || name_has(kernel_name, "marlin"))
            g_dequant_kernels.fetch_add(1, std::memory_order_relaxed);
    }

    if (n_contexts > g_max_n_contexts.load(std::memory_order_relaxed))
        g_max_n_contexts.store(n_contexts, std::memory_order_relaxed);

    // Periodic refresh: process count + reclassification
    if ((seq & 1023) == 0) {
        int probed = probe_cipher_process_count();
        if (probed > g_max_n_contexts.load(std::memory_order_relaxed))
            g_max_n_contexts.store(probed, std::memory_order_relaxed);
    }
    if ((seq % 100) == 0) {
        g_cached_class.store((int)reclassify_locked(), std::memory_order_relaxed);
    }
}

extern "C" void cipher_workload_observe_kernel(const char* kernel_name) {
    cipher_workload_observe(kernel_name, 0, 0, 0, 0);
}

extern "C" int cipher_workload_last_gemm_k(void) {
    return g_last_gemm_k.load(std::memory_order_relaxed);
}

extern "C" CipherWorkloadType cipher_workload_classify(void) {
    return (CipherWorkloadType)g_cached_class.load(std::memory_order_relaxed);
}

extern "C" CipherProfile cipher_workload_profile(void) {
    CipherProfile p;
    memset(&p, 0, sizeof(p));
    CipherWorkloadType t = cipher_workload_classify();

    // Defaults that are safe everywhere
    p.persist_engine = 1;
    p.thermostat     = 1;
    p.carbon         = 1;
    p.receipt        = 1;
    p.audit          = 1;
    p.guard          = 1;
    p.predict        = 1;
    p.continuity     = 1;
    p.comply         = 1;
    p.trace          = 1;
    p.flow_record    = 1;
    p.loop_detect    = 1;
    p.validate       = 1;

    switch (t) {
    case CIPHER_WL_LLM_DECODE_SINGLE:
        p.koopman_o1 = 1; p.fp8_compute = 1; p.thermostat_lock_low = 1;
        p.flow_match = 1; p.flow_substitute = 1; p.fuse_rmsnorm = 1;
        p.speculate = 1; p.adapt = 1; p.remember = 1;
        break;
    case CIPHER_WL_LLM_DECODE_BATCHED:
        p.koopman_o1 = 1; p.fp8_compute = 1; p.thermostat_lock_low = 1;
        p.flow_match = 1; p.flow_substitute = 1; p.fuse_rmsnorm = 1; p.fuse_swiglu = 1;
        p.speculate = 1; p.adapt = 1; p.remember = 1;
        break;
    case CIPHER_WL_LLM_PREFILL:
        p.koopman_o1 = 1; p.fp8_compute = 1; p.substitute_v2 = 1;
        p.flow_match = 1; p.flow_substitute = 1; p.fuse_rmsnorm = 1;
        break;
    case CIPHER_WL_LLM_SERVING_VLLM:
        p.koopman_o1 = 1; p.fp8_compute = 1; p.thermostat = 1;
        p.flow_match = 1; p.flow_substitute = 1; p.graph_engine = 1;
        p.speculate = 1;
        break;
    case CIPHER_WL_MULTI_TENANT:
        p.koopman_o1 = 1; p.fp8_compute = 1; p.substitute_v2 = 1;
        p.thermostat = 1; p.fairness = 1; p.arbitrate = 1;
        p.flow_match = 1; p.flow_substitute = 1; p.fuse_rmsnorm = 1;
        p.speculate = 1; p.adapt = 1; p.remember = 1;
        p.topology = 1; p.pipeline = 1;
        break;
    case CIPHER_WL_EMBEDDINGS:
        // small shapes — overhead exceeds benefit; persist only
        break;
    case CIPHER_WL_LORA_FINETUNING:
        // training: skip lossy substitution
        break;
    case CIPHER_WL_TRAINING_FULL:
        p.nccl_tuner = 1; p.topology = 1;
        break;
    case CIPHER_WL_MULTI_GPU_TP:
        p.koopman_o1 = 1; p.fp8_compute = 1; p.nccl_tuner = 1;
        p.topology = 1; p.flow_match = 1; p.fuse_rmsnorm = 1;
        break;
    case CIPHER_WL_DIFFUSION:
        // sensitive to FP8 quant artifacts
        break;
    case CIPHER_WL_SPEECH:
        p.koopman_o1 = 1; p.fp8_compute = 1;
        break;
    case CIPHER_WL_VISION:
    case CIPHER_WL_MULTIMODAL:
        p.koopman_o1 = 1; p.fp8_compute = 1; p.fuse_rmsnorm = 1;
        break;
    case CIPHER_WL_QUANTIZED_NATIVE:
        // model already quantized; skip FP8
        p.koopman_o1 = 1;
        break;
    case CIPHER_WL_AGENTIC:
    case CIPHER_WL_BATCH_PROCESSING:
    case CIPHER_WL_CODE_GENERATION:
    case CIPHER_WL_LONG_CONTEXT:
    case CIPHER_WL_PREFIX_CACHING:
    case CIPHER_WL_RAG_PIPELINE:
    case CIPHER_WL_MODEL_SWITCH:
    case CIPHER_WL_TORCH_COMPILE:
    case CIPHER_WL_SPECULATIVE_DECODING:
    case CIPHER_WL_MOE_MODELS:
        p.koopman_o1 = 1; p.fp8_compute = 1; p.flow_match = 1;
        p.flow_substitute = 1; p.fuse_rmsnorm = 1;
        break;
    case CIPHER_WL_UNKNOWN:
    default:
        // Conservative defaults — let the existing env-gated paths run unchanged
        p.fp8_compute = 1; p.flow_match = 1; p.flow_substitute = 1;
        break;
    }
    return p;
}

extern "C" const char* cipher_workload_name(CipherWorkloadType t) {
    switch (t) {
    case CIPHER_WL_UNKNOWN:               return "UNKNOWN";
    case CIPHER_WL_LLM_DECODE_SINGLE:     return "LLM_DECODE_SINGLE";
    case CIPHER_WL_LLM_DECODE_BATCHED:    return "LLM_DECODE_BATCHED";
    case CIPHER_WL_LLM_PREFILL:           return "LLM_PREFILL";
    case CIPHER_WL_LLM_SERVING_VLLM:      return "LLM_SERVING_VLLM";
    case CIPHER_WL_MULTI_TENANT:          return "MULTI_TENANT";
    case CIPHER_WL_EMBEDDINGS:            return "EMBEDDINGS";
    case CIPHER_WL_LORA_FINETUNING:       return "LORA_FINETUNING";
    case CIPHER_WL_DIFFUSION:             return "DIFFUSION";
    case CIPHER_WL_SPEECH:                return "SPEECH";
    case CIPHER_WL_SPECULATIVE_DECODING:  return "SPECULATIVE_DECODING";
    case CIPHER_WL_AGENTIC:               return "AGENTIC";
    case CIPHER_WL_BATCH_PROCESSING:      return "BATCH_PROCESSING";
    case CIPHER_WL_LONG_CONTEXT:          return "LONG_CONTEXT";
    case CIPHER_WL_TORCH_COMPILE:         return "TORCH_COMPILE";
    case CIPHER_WL_MOE_MODELS:            return "MOE_MODELS";
    case CIPHER_WL_PREFIX_CACHING:        return "PREFIX_CACHING";
    case CIPHER_WL_TRAINING_FULL:         return "TRAINING_FULL";
    case CIPHER_WL_MULTI_GPU_TP:          return "MULTI_GPU_TP";
    case CIPHER_WL_VISION:                return "VISION";
    case CIPHER_WL_MULTIMODAL:            return "MULTIMODAL";
    case CIPHER_WL_CODE_GENERATION:       return "CODE_GENERATION";
    case CIPHER_WL_RAG_PIPELINE:          return "RAG_PIPELINE";
    case CIPHER_WL_MODEL_SWITCH:          return "MODEL_SWITCH";
    case CIPHER_WL_QUANTIZED_NATIVE:      return "QUANTIZED_NATIVE";
    default:                              return "UNKNOWN";
    }
}

extern "C" void cipher_workload_report(void) {
    CipherWorkloadType t = cipher_workload_classify();
    fprintf(stderr,
        "[CIPHER WORKLOAD] type=%s pid=%d obs=%llu gemms=%llu attn=%llu "
        "rmsnorm=%llu silu=%llu conv=%llu nccl=%llu bwd=%llu graph=%llu "
        "ctxs=%d N=1:%llu N<=64:%llu N<=511:%llu N>=512:%llu\n",
        cipher_workload_name(t), (int)getpid(),
        (unsigned long long)g_obs_total.load(),
        (unsigned long long)g_gemm_count.load(),
        (unsigned long long)g_attention_kernels.load(),
        (unsigned long long)g_rmsnorm_kernels.load(),
        (unsigned long long)g_silu_kernels.load(),
        (unsigned long long)g_conv_kernels.load(),
        (unsigned long long)g_nccl_kernels.load(),
        (unsigned long long)g_backward_kernels.load(),
        (unsigned long long)g_graph_capture_seen.load(),
        g_max_n_contexts.load(),
        (unsigned long long)g_n_eq_1.load(),
        (unsigned long long)g_n_2_64.load(),
        (unsigned long long)g_n_65_511.load(),
        (unsigned long long)g_n_512_plus.load());
}

// atexit auto-report when CIPHER_WORKLOAD_REPORT is set
static void cipher_workload_atexit() {
    const char* env = getenv("CIPHER_WORKLOAD_REPORT");
    if (env && *env && env[0] != '0') {
        cipher_workload_report();
    }
}

__attribute__((constructor))
static void cipher_workload_init() {
    atexit(cipher_workload_atexit);
}
