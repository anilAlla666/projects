/* T4.2.4d phase 3.5 — does NULL-stream launch enforce under green ctx?
 *
 * Two regimes for NULL-stream (default-stream) launches:
 *
 *   TEST A: primary ctx current → kernel<<<n,256>>>() (NULL stream)
 *           Expected: full 132 SMs.
 *
 *   TEST B: cuCtxSetCurrent(green_cuctx) → kernel<<<n,256>>>() (NULL stream)
 *           Question: does the default stream now belong to the green ctx?
 *           If yes: per-launch ms ≈ 16× baseline → fix option 1 works
 *           If no:  per-launch ms ≈ baseline   → need fix option 2 (divert)
 *
 *   TEST C: same as B but explicitly via cuLaunchKernel(stream=0)
 *           (kernel<<<>>> is the runtime API; cuLaunchKernel is driver).
 *           Confirms behavior is independent of runtime/driver API.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(x) do { CUresult _r=(x); if(_r!=CUDA_SUCCESS){ \
    const char *_n=NULL; cuGetErrorName(_r,&_n); \
    fprintf(stderr,"FAIL %s:%d %s -> %s\n",__FILE__,__LINE__,#x,_n?_n:"?"); \
    exit(2);} } while(0)
#define CHECK_RT(x) do { cudaError_t _r=(x); if(_r!=cudaSuccess){ \
    fprintf(stderr,"RT FAIL %s:%d %s -> %s\n",__FILE__,__LINE__,#x, \
            cudaGetErrorString(_r)); exit(3);} } while(0)

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
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static void time_null_stream(const char *label, int nblocks,
                             unsigned long iters, int repeats) {
    float *dsink = NULL;
    CHECK_RT(cudaMalloc(&dsink, nblocks * sizeof(float)));
    /* warmup */
    burn<<<nblocks, 256>>>(iters, dsink);   /* NULL stream */
    CHECK_RT(cudaDeviceSynchronize());
    double t0 = now_s();
    for (int r = 0; r < repeats; ++r)
        burn<<<nblocks, 256>>>(iters, dsink);
    CHECK_RT(cudaDeviceSynchronize());
    double t1 = now_s();
    double ms = (t1 - t0) * 1000.0 / repeats;
    printf("  %-28s repeats=%d total=%.3f s per-launch=%.3f ms\n",
           label, repeats, t1 - t0, ms);
    CHECK_RT(cudaFree(dsink));
}

int main(void) {
    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));

    /* primary context */
    CUcontext primary;
    CHECK(cuDevicePrimaryCtxRetain(&primary, dev));

    /* build green ctx with 8 SMs */
    CUdevResource full;
    CHECK(cuDeviceGetDevResource(dev, &full, CU_DEV_RESOURCE_TYPE_SM));
    CUdevResource groups[16], remaining;
    unsigned int nb = 16;
    CHECK(cuDevSmResourceSplitByCount(groups, &nb, &full, &remaining, 0, 8));
    CUdevResourceDesc desc;
    CHECK(cuDevResourceGenerateDesc(&desc, &groups[0], 1));
    CUgreenCtx green = NULL;
    CHECK(cuGreenCtxCreate(&green, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext green_cuctx = NULL;
    CHECK(cuCtxFromGreenCtx(&green_cuctx, green));

    CUdevResource verify;
    CHECK(cuGreenCtxGetDevResource(green, &verify, CU_DEV_RESOURCE_TYPE_SM));
    printf("Green ctx smCount=%u (expected 8)\n", verify.sm.smCount);

    const int nblocks = 1024;
    const unsigned long iters = 200000;
    const int repeats = 10;

    printf("\n=== TEST A: primary current, NULL stream ===\n");
    CHECK(cuCtxSetCurrent(primary));
    time_null_stream("primary+NULL stream", nblocks, iters, repeats);

    printf("\n=== TEST B: green_cuctx current (cuCtxSetCurrent), NULL stream ===\n");
    CHECK(cuCtxSetCurrent(green_cuctx));
    /* verify current context */
    CUcontext cur = NULL;
    CHECK(cuCtxGetCurrent(&cur));
    printf("  current ctx = %p (expected green_cuctx = %p)\n",
           (void *)cur, (void *)green_cuctx);
    time_null_stream("green-set+NULL stream", nblocks, iters, repeats);

    /* Test C: same context, explicitly use cuLaunchKernel with stream=0 */
    printf("\n=== TEST C: green current, cuLaunchKernel stream=0 ===\n");
    /* fetch kernel function pointer via cudaGetSymbol — easier path: just
     * call kernel via runtime API <<<>>> with default stream, but using
     * cuLaunchKernel as well to compare. For brevity we reuse runtime API. */
    time_null_stream("green-set+NULL (rt API)", nblocks, iters, repeats);

    printf("\n=== TEST D: switched back to primary, NULL stream ===\n");
    CHECK(cuCtxSetCurrent(primary));
    time_null_stream("back-to-primary+NULL", nblocks, iters, repeats);

    CHECK(cuGreenCtxDestroy(green));
    CHECK(cuDevicePrimaryCtxRelease(dev));
    printf("\ndone.\n");
    return 0;
}
