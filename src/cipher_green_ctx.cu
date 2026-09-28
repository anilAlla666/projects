// CPU stub redirect
#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#else
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#  include <cuda.h>
#endif
// =============================================================================
// CIPHER — F2: Green Context Allocation Implementation
// cipher_green_ctx.cu
//
// CUDA 12.8 Green Context API sequence:
//   cuDeviceGetDevResource → cuDevSmResourceSplitByCount →
//   cuDevResourceGenerateDesc → cuGreenCtxCreate →
//   cuCtxFromGreenCtx → cuGreenCtxStreamCreate
// =============================================================================

#include "cipher_green_ctx.h"
#include <stdio.h>
#include <string.h>

#define CIPHER_GREEN_CTX_MIN_DRIVER 12040

#define CIPHER_CUDA_CHECK(call) \
    do { \
        CUresult _r = (call); \
        if (_r != CUDA_SUCCESS) { \
            const char* _s = NULL; \
            cuGetErrorString(_r, &_s); \
            fprintf(stderr, "[CIPHER F2] CUDA error at %s:%d — %s (%d)\n", \
                    __FILE__, __LINE__, _s ? _s : "unknown", _r); \
            return _r; \
        } \
    } while(0)

static int get_driver_version(void) {
    int version = 0;
    cuDriverGetVersion(&version);
    return version;
}

// ---------------------------------------------------------------------------
// Green Context initialization — CUDA 12.4+ path
// ---------------------------------------------------------------------------

static CUresult init_green_ctx_path(CipherGreenCtxState* state,
                                    CUdevice device)
{
#ifdef CIPHER_CPU_STUB
    (void)state; (void)device;
    return CUDA_SUCCESS;
#else
    // Step 1: Query total SM count
    int total_sms_attr = 0;
    CIPHER_CUDA_CHECK((CUresult)cuDeviceGetAttribute(
        &total_sms_attr,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        device));
    state->total_sms = (uint32_t)total_sms_attr;

    if (state->total_sms < CIPHER_MIN_SM_TOTAL) {
        fprintf(stderr, "[CIPHER F2] Device has only %d SMs — "
                        "minimum %d required. Using fallback.\n",
                state->total_sms, CIPHER_MIN_SM_TOTAL);
        state->fallback_mode = true;
        return CUDA_SUCCESS;
    }

    // Step 2: Get device SM resource
    CUdevResource totalSMs;
    memset(&totalSMs, 0, sizeof(totalSMs));
    CIPHER_CUDA_CHECK(cuDeviceGetDevResource(device, &totalSMs, CU_DEV_RESOURCE_TYPE_SM));

    // Step 3: Split SMs — CIPHER gets CIPHER_SM_COUNT, rest for user workload
    CUdevResource cipherSMs;
    CUdevResource userSMs;
    unsigned int nbGroups = 0;
    memset(&cipherSMs, 0, sizeof(cipherSMs));
    memset(&userSMs, 0, sizeof(userSMs));

    CIPHER_CUDA_CHECK(cuDevSmResourceSplitByCount(
        &cipherSMs,
        &nbGroups,
        &totalSMs,
        &userSMs,
        /* useFlags = */ 0,
        /* minCount = */ CIPHER_SM_COUNT));

    state->cipher_sms   = (int)nbGroups;
    state->workload_sms  = state->total_sms - (int)nbGroups;

    fprintf(stderr, "[CIPHER F2] SM split: %d CIPHER + %d workload (of %d total)\n",
            state->cipher_sms, state->workload_sms, state->total_sms);

    // Step 4: Generate resource descriptor from CIPHER's SM partition
    CUdevResourceDesc cipherDesc = NULL;
    CIPHER_CUDA_CHECK(cuDevResourceGenerateDesc(&cipherDesc, &cipherSMs, 1));

    // Step 5: Create Green Contexts — one per LNN layer
    const char* layer_names[CIPHER_CTX_COUNT] = {
        "Layer3-Substitutor", "Layer2-Orchestrator", "Layer1-Generator"
    };

    for (int i = 0; i < CIPHER_CTX_COUNT; i++) {
        CIPHER_CUDA_CHECK(cuGreenCtxCreate(
            &state->green_ctx[i],
            cipherDesc,
            device,
            CU_GREEN_CTX_DEFAULT_STREAM));

        // Step 6: Get usable CUcontext from green context
        CIPHER_CUDA_CHECK(cuCtxFromGreenCtx(
            &state->ctx[i],
            state->green_ctx[i]));

        // Step 7: Create dedicated stream on this green context
        CUstream cuStream = NULL;
        CIPHER_CUDA_CHECK(cuGreenCtxStreamCreate(
            &cuStream,
            state->green_ctx[i],
            /* flags = */ 0,
            /* priority = */ -1));
        state->stream[i] = (cudaStream_t)cuStream;

        fprintf(stderr, "[CIPHER F2] Green Context [%d] %s: %d SMs allocated\n",
                i, layer_names[i], state->cipher_sms);
    }

    return CUDA_SUCCESS;
#endif  // !CIPHER_CPU_STUB
}

// ---------------------------------------------------------------------------
// Fallback path — no Green Context support
// ---------------------------------------------------------------------------

static CUresult init_fallback_path(CipherGreenCtxState* state) {
    CUcontext current_ctx;
    CIPHER_CUDA_CHECK(cuCtxGetCurrent(&current_ctx));

    int total_sms;
    CUdevice device;
    CIPHER_CUDA_CHECK(cuCtxGetDevice(&device));
    cuDeviceGetAttribute(&total_sms,
                         CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                         device);

    state->total_sms    = total_sms;
    state->cipher_sms   = 0;
    state->workload_sms = total_sms;

    for (int i = 0; i < CIPHER_CTX_COUNT; i++) {
        state->green_ctx[i] = NULL;
        state->ctx[i]       = current_ctx;

        cudaError_t err = cudaStreamCreateWithPriority(
            &state->stream[i],
            cudaStreamNonBlocking,
            -1);
        if (err != cudaSuccess) return CUDA_ERROR_UNKNOWN;
    }

    fprintf(stderr,
        "[CIPHER F2] Fallback mode: sharing default context. "
        "Green Ctx requires CUDA driver >= 12040 (have %d).\n",
        get_driver_version());

    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public: cipher_green_ctx_init
// ---------------------------------------------------------------------------

CUresult cipher_green_ctx_init(CipherGreenCtxState* state,
                                int device_ordinal)
{
    memset(state, 0, sizeof(*state));

    CIPHER_CUDA_CHECK(cuDeviceGet(&state->device, device_ordinal));

    int driver_ver = get_driver_version();
    if (driver_ver >= CIPHER_GREEN_CTX_MIN_DRIVER) {
        state->fallback_mode = false;
        CUresult r = init_green_ctx_path(state, state->device);
        if (r != CUDA_SUCCESS) {
            fprintf(stderr, "[CIPHER F2] Green Ctx init failed (%d), "
                            "falling back.\n", r);
            state->fallback_mode = true;
            return init_fallback_path(state);
        }
    } else {
        state->fallback_mode = true;
        CIPHER_CUDA_CHECK(init_fallback_path(state));
    }

    state->initialized = true;
    cipher_green_ctx_report(state);
    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public: cipher_green_ctx_destroy
// ---------------------------------------------------------------------------

void cipher_green_ctx_destroy(CipherGreenCtxState* state) {
    if (!state->initialized) return;

    for (int i = 0; i < CIPHER_CTX_COUNT; i++) {
        if (state->stream[i])
            cudaStreamDestroy(state->stream[i]);
        if (!state->fallback_mode && state->green_ctx[i])
            cuGreenCtxDestroy(state->green_ctx[i]);
    }

    memset(state, 0, sizeof(*state));
    fprintf(stderr, "[CIPHER F2] Green Contexts destroyed.\n");
}

// ---------------------------------------------------------------------------
// Public: accessors
// ---------------------------------------------------------------------------

CUcontext cipher_get_ctx(const CipherGreenCtxState* state, CipherCtxId id) {
    if (!state->initialized || id >= CIPHER_CTX_COUNT)
        return NULL;
    return state->ctx[id];
}

cudaStream_t cipher_get_stream(const CipherGreenCtxState* state,
                                CipherCtxId id) {
    if (!state->initialized || id >= CIPHER_CTX_COUNT)
        return 0;
    return state->stream[id];
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

void cipher_green_ctx_report(const CipherGreenCtxState* state) {
    if (!state->initialized) {
        fprintf(stderr, "[CIPHER F2] Not initialized.\n");
        return;
    }
    fprintf(stderr,
        "[CIPHER F2] SM Allocation Report\n"
        "  Total SMs:       %d\n"
        "  CIPHER SMs:      %d  (%.1f%%)\n"
        "  Workload SMs:    %d  (%.1f%%)\n"
        "  Mode:            %s\n"
        "  Contexts:        %d (L3/L2/L1)\n",
        state->total_sms,
        state->cipher_sms,
        state->total_sms > 0
            ? (double)state->cipher_sms * 100.0 / state->total_sms : 0.0,
        state->workload_sms,
        state->total_sms > 0
            ? (double)state->workload_sms * 100.0 / state->total_sms : 0.0,
        state->fallback_mode ? "FALLBACK (shared ctx)" : "GREEN CONTEXT",
        CIPHER_CTX_COUNT);
}
