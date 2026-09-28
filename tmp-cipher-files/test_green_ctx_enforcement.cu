/* T4.2.4d Phase 3 microbench — does CUDA green ctx actually enforce SM partition?
 *
 * Three test conditions, each runs the same compute-bound kernel 10 times:
 *
 *   TEST 1 (baseline) :  no partition, default stream.
 *                        Kernel runs on all 132 SMs.
 *
 *   TEST 2 (push API) :  cuCtxPushCurrent(cuCtxFromGreenCtx(8-SM green ctx))
 *                        then cuStreamCreate. Driver-API ref doc says this
 *                        should bind the stream to the green ctx; programming
 *                        guide says it should not.
 *
 *   TEST 3 (explicit):   cuGreenCtxStreamCreate(8-SM green ctx) directly.
 *                        Per docs, this is the canonical binding method.
 *
 * Expected timing on H100 (132 SMs, kernel-bound):
 *   TEST 1: T_baseline
 *   TEST 2 if push works: ~16×T_baseline (132/8)
 *   TEST 2 if push broken: ~T_baseline
 *   TEST 3 if green ctx enforces: ~16×T_baseline
 *
 * Compile:
 *   nvcc -O2 -gencode arch=compute_90,code=sm_90 -lcuda \
 *        /tmp/test_green_ctx_enforcement.cu -o /tmp/test_green_ctx_enforcement
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do {                                                       \
    CUresult _r = (x);                                                      \
    if (_r != CUDA_SUCCESS) {                                               \
        const char *_n = NULL; cuGetErrorName(_r, &_n);                     \
        fprintf(stderr, "DRV FAIL %s:%d %s -> %s\n",                        \
                __FILE__, __LINE__, #x, _n ? _n : "?");                     \
        exit(2);                                                            \
    }                                                                       \
} while (0)

#define CHECK_RT(x) do {                                                    \
    cudaError_t _r = (x);                                                   \
    if (_r != cudaSuccess) {                                                \
        fprintf(stderr, "RT FAIL %s:%d %s -> %s\n",                         \
                __FILE__, __LINE__, #x, cudaGetErrorString(_r));            \
        exit(3);                                                            \
    }                                                                       \
} while (0)

/* Pure compute-bound kernel — no memory traffic.
 * 256-thread blocks, gridDim chosen so total blocks > nSMs to saturate. */
__global__ void burn(unsigned long iters, float *sink) {
    float a = (float)(threadIdx.x + 1);
    float b = (float)(threadIdx.x + 2);
    for (unsigned long i = 0; i < iters; ++i) {
        a = a * b + 1.0f;
        b = b * a + 0.5f;
        a = a / 1.0001f;
    }
    if (a == 12345.6789f && sink) sink[blockIdx.x] = a + b;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static void run_on_stream(const char *label, CUstream stream,
                          int nblocks, unsigned long iters, int repeats) {
    float *dsink = NULL;
    CHECK_RT(cudaMalloc(&dsink, nblocks * sizeof(float)));

    /* warmup */
    burn<<<nblocks, 256, 0, (cudaStream_t)stream>>>(iters, dsink);
    CHECK_RT(cudaStreamSynchronize((cudaStream_t)stream));

    double t0 = now_s();
    for (int r = 0; r < repeats; ++r) {
        burn<<<nblocks, 256, 0, (cudaStream_t)stream>>>(iters, dsink);
    }
    CHECK_RT(cudaStreamSynchronize((cudaStream_t)stream));
    double t1 = now_s();
    double per_launch_ms = (t1 - t0) * 1000.0 / repeats;
    printf("  %-18s repeats=%d total=%.3f s per-launch=%.3f ms\n",
           label, repeats, (t1 - t0), per_launch_ms);

    CHECK_RT(cudaFree(dsink));
}

int main(void) {
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    char name[256] = {0};
    CHECK(cuDeviceGetName(name, sizeof(name) - 1, dev));
    int cc_major = 0, cc_minor = 0;
    CHECK(cuDeviceGetAttribute(&cc_major,
          CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CHECK(cuDeviceGetAttribute(&cc_minor,
          CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    int sm_count = 0;
    CHECK(cuDeviceGetAttribute(&sm_count,
          CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    printf("Device: %s cc=%d.%d SMs=%d\n", name, cc_major, cc_minor, sm_count);

    /* Make a primary context so cudaXxx APIs work. */
    CUcontext primary;
    CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
    CHECK(cuCtxSetCurrent(primary));

    /* Build a small 8-SM green context. */
    CUdevResource full_sm_res;
    CHECK(cuDeviceGetDevResource(dev, &full_sm_res, CU_DEV_RESOURCE_TYPE_SM));
    CUdevResource groups[16];
    CUdevResource remaining;
    unsigned int nb_groups = 16;
    CHECK(cuDevSmResourceSplitByCount(groups, &nb_groups, &full_sm_res,
                                       &remaining, 0, 8));
    printf("Split: nb_groups=%u group0_sms=%u remaining_sms=%u\n",
           nb_groups, groups[0].sm.smCount, remaining.sm.smCount);
    if (nb_groups == 0) { fprintf(stderr, "no groups\n"); return 4; }

    CUdevResourceDesc desc;
    CHECK(cuDevResourceGenerateDesc(&desc, &groups[0], 1));

    CUgreenCtx green = NULL;
    CHECK(cuGreenCtxCreate(&green, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));

    /* Verify green ctx has 8 SMs. */
    CUdevResource verify;
    CHECK(cuGreenCtxGetDevResource(green, &verify, CU_DEV_RESOURCE_TYPE_SM));
    printf("Green ctx: smCount=%u (expected 8)\n", verify.sm.smCount);

    CUcontext green_cuctx = NULL;
    CHECK(cuCtxFromGreenCtx(&green_cuctx, green));

    /* Workload sizing: 1024 blocks of 256 threads = enough to saturate
     * 132 SMs (8 blocks/SM). 200k iters/thread gives ~80 ms per launch
     * on baseline → ~1.2 s for 16 launches. */
    const int nblocks = 1024;
    const unsigned long iters = 200000;
    const int repeats = 10;

    printf("\n=== TEST 1: baseline (primary ctx, default-stream-like) ===\n");
    {
        CUstream s;
        CHECK(cuStreamCreate(&s, CU_STREAM_DEFAULT));
        run_on_stream("primary cuStream", s, nblocks, iters, repeats);
        CHECK(cuStreamDestroy(s));
    }

    printf("\n=== TEST 2: push(cuCtxFromGreenCtx) + cuStreamCreate ===\n");
    {
        CHECK(cuCtxPushCurrent(green_cuctx));
        CUstream s;
        CHECK(cuStreamCreate(&s, CU_STREAM_DEFAULT));
        /* Verify which green ctx the stream landed in. */
        CUgreenCtx s_green = NULL;
        CUresult gr = cuStreamGetGreenCtx(s, &s_green);
        printf("  cuStreamGetGreenCtx rc=%d s_green=%p (expected non-NULL=%p if bound)\n",
               (int)gr, (void *)s_green, (void *)green);
        run_on_stream("push+cuStream", s, nblocks, iters, repeats);
        CHECK(cuStreamDestroy(s));
        CUcontext popped;
        CHECK(cuCtxPopCurrent(&popped));
    }

    printf("\n=== TEST 3: cuGreenCtxStreamCreate (canonical) ===\n");
    {
        CUstream s;
        CHECK(cuGreenCtxStreamCreate(&s, green, CU_STREAM_NON_BLOCKING, 0));
        CUgreenCtx s_green = NULL;
        CUresult gr = cuStreamGetGreenCtx(s, &s_green);
        printf("  cuStreamGetGreenCtx rc=%d s_green=%p (expected %p)\n",
               (int)gr, (void *)s_green, (void *)green);
        run_on_stream("greenctx stream", s, nblocks, iters, repeats);
        CHECK(cuStreamDestroy(s));
    }

    CHECK(cuGreenCtxDestroy(green));
    CHECK(cuDevicePrimaryCtxRelease(dev));
    printf("\ndone.\n");
    return 0;
}
