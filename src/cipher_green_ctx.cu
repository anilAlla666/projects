// =============================================================================
// CIPHER — F2: Green Context Allocation Implementation
// cipher_green_ctx.cu
// =============================================================================

#include "cipher_green_ctx.h"
#include <stdio.h>
#include <string.h>

// ---------------------------------------------------------------------------
// CUDA 12.4+ Green Context API structs / functions
// We define the necessary types here so the file compiles against older CUDA
// SDKs — at runtime we check the driver version and use fallback if needed.
// ---------------------------------------------------------------------------

// Minimum CUDA driver version for Green Context support
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

// ---------------------------------------------------------------------------
// Query driver version — determines Green Ctx availability
// ---------------------------------------------------------------------------

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
    // Step 1: Query SM resource range
    CUdevResourceDesc sm_range;
    memset(&sm_range, 0, sizeof(sm_range));
    sm_range.type = CU_DEV_RESOURCE_TYPE_SM;

    CIPHER_CUDA_CHECK(cuDeviceGetDevResourceRange(
        &sm_range, device, CU_DEV_RESOURCE_TYPE_SM));

    state->total_sms = sm_range.sm.smCount;

    if (state->total_sms < CIPHER_MIN_SM_TOTAL) {
        fprintf(stderr, "[CIPHER F2] Device has only %d SMs — "
                        "minimum %d required. Using fallback.\n",
                state->total_sms, CIPHER_MIN_SM_TOTAL);
        state->fallback_mode = true;
        return CUDA_SUCCESS;
    }

    // Step 2: Split SMs — CIPHER_SM_COUNT for us, rest for workload
    // cuDevSmResourceSplitByCount splits the resource into two parts.
    // First part gets 'minCount' SMs; second gets the remainder.
    CUdevSmResource parts[2];
    uint32_t actual_count = 0;

    CIPHER_CUDA_CHECK(cuDevSmResourceSplitByCount(
        parts,
        &actual_count,
        &sm_range.sm,
        /* flags = */ 0,
        /* minCount = */ CIPHER_SM_COUNT));

    // parts[0] = CIPHER's SMs (actual_count may be >= CIPHER_SM_COUNT
    //            due to hardware SM granularity)
    // parts[1] = remaining SMs for user workload

    state->cipher_sm_resource  = parts[0];   // CIPHER LNNs live here
    state->workload_sm_resource = parts[1];  // User compute lives here
    state->cipher_sms           = (int)actual_count;
    state->workload_sms         = state->total_sms - (int)actual_count;

    // Step 3: Create three Green Contexts — one per LNN
    const char* layer_names[CIPHER_CTX_COUNT] = {
        "Layer3-Substitutor", "Layer2-Orchestrator", "Layer1-Generator"
    };

    for (int i = 0; i < CIPHER_CTX_COUNT; i++) {
        CUdevResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.type    = CU_DEV_RESOURCE_TYPE_SM;
        res_desc.sm      = state->cipher_sm_resource;

        CIPHER_CUDA_CHECK(cuGreenCtxCreate(
            &state->green_ctx[i],
            &res_desc,
            device,
            CU_GREEN_CTX_DEFAULT_STREAM));

        // Get a usable CUcontext from the green context
        CIPHER_CUDA_CHECK(cuCtxFromGreenCtx(
            &state->ctx[i],
            state->green_ctx[i]));

        // Create a dedicated stream on this context
        cudaError_t err = cudaStreamCreateWithPriority(
            &state->stream[i],
            cudaStreamNonBlocking,
            /* priority = */ -1);   // Highest priority for CIPHER work
        if (err != cudaSuccess) {
            fprintf(stderr, "[CIPHER F2] Stream create failed for %s: %s\n",
                    layer_names[i], cudaGetErrorString(err));
            return CUDA_ERROR_UNKNOWN;
        }

        fprintf(stderr, "[CIPHER F2] Green Context [%d] %s: %d SMs allocated\n",
                i, layer_names[i], state->cipher_sms);
    }

    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// Fallback path — no Green Context support
// All three LNNs share the default context + separate streams
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
    state->cipher_sms   = 0;   // No dedicated SMs in fallback
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
