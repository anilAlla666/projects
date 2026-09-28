/* CP 5.3 STEP 2 — item 2 — KU2 smoke check.
 *
 * The load-bearing pre-flight before the actuator edit: can an NVRTC-
 * compiled module that was loaded under the DEVICE PRIMARY CONTEXT be
 * launched, on a GREEN-CONTEXT-bound stream, reading device memory that
 * was ALLOCATED UNDER THE PRIMARY CONTEXT?
 *
 * STEP 2's design (CP_5_3_STEP_2_PARTITION_WIRING_SCOPE.md §3) routes the
 * Marlin GEMM through a green context while quant/repack stays primary-
 * pinned. The Marlin cubin is NVRTC-compiled and cuModuleLoad'd at
 * ensure_marlin_compiled() time under whatever context is then current
 * (primary); the quant-produced weights are cuMemAlloc'd under primary.
 * If a green-context launch cannot see a primary-loaded module or read
 * primary-allocated memory, the STEP 2 design is wrong and we STOP here.
 *
 * This harness reproduces exactly that sequence with a trivial kernel.
 * PASS  -> KU2 resolved, proceed to item 3 (the actuator edit).
 * FAIL  -> stop, write the failure memo, re-adjudicate.
 *
 * It also records each block's %smid (a KU3 pre-check — the binding KU3/B
 * test uses the real Marlin kernel in items 5-6).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>
#include <cuda.h>
#include <nvrtc.h>

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *m = nullptr; cuGetErrorString(r, &m); \
    printf("RESULT status=CUDA_FAIL call=%s rc=%d msg=%s\n", #x, r, m?m:"?"); \
    return 3; } } while (0)

#define NK(x) do { nvrtcResult r = (x); if (r != NVRTC_SUCCESS) { \
    printf("RESULT status=NVRTC_FAIL call=%s rc=%d msg=%s\n", #x, r, \
           nvrtcGetErrorString(r)); return 3; } } while (0)

static const char *KSRC = R"CUDA(
extern "C" __global__
void ku2_probe(const float *in, float *out, int n, int *smids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        smids[blockIdx.x] = (int)smid;
    }
    if (i < n) out[i] = in[i] * 2.0f + (float)blockIdx.x;
}
)CUDA";

int main(void)
{
    const int GRID = 8, BLOCK = 64, N = GRID * BLOCK;

    printf("CP 5.3 STEP 2 item 2 — KU2 smoke check\n");
    printf("question: primary-loaded module + primary-allocated memory, "
           "launched on a green-ctx stream\n");
    printf("=========================================================\n");

    /* ── 1. driver init, device 0, primary context ──────────────────── */
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext primary;
    CK(cuDevicePrimaryCtxRetain(&primary, dev));
    CK(cuCtxSetCurrent(primary));
    printf("[1] primary context retained + current\n");

    /* ── 2. NVRTC compile, load module UNDER PRIMARY ────────────────── */
    nvrtcProgram prog;
    NK(nvrtcCreateProgram(&prog, KSRC, "ku2_probe.cu", 0, nullptr, nullptr));
    const char *opts[] = { "--gpu-architecture=sm_90" };
    nvrtcResult cr = nvrtcCompileProgram(prog, 1, opts);
    if (cr != NVRTC_SUCCESS) {
        size_t lsz = 0; nvrtcGetProgramLogSize(prog, &lsz);
        std::vector<char> log(lsz + 1, 0);
        nvrtcGetProgramLog(prog, log.data());
        printf("RESULT status=NVRTC_COMPILE_FAIL\n%s\n", log.data());
        return 3;
    }
    size_t ptxsz = 0;
    NK(nvrtcGetPTXSize(prog, &ptxsz));
    std::vector<char> ptx(ptxsz);
    NK(nvrtcGetPTX(prog, ptx.data()));
    NK(nvrtcDestroyProgram(&prog));

    CUmodule mod;
    CK(cuModuleLoadData(&mod, ptx.data()));      /* loaded under primary */
    CUfunction fn;
    CK(cuModuleGetFunction(&fn, mod, "ku2_probe"));
    printf("[2] NVRTC kernel compiled + module loaded under PRIMARY ctx\n");

    /* ── 3. create an 8-SM green context (green_ctx.c machinery) ────── */
    CUdevResource full;
    CK(cuDeviceGetDevResource(dev, &full, CU_DEV_RESOURCE_TYPE_SM));
    printf("[3] device has %u SMs\n", full.sm.smCount);

    const unsigned MIN_SM = 8;
    unsigned nb = full.sm.smCount / MIN_SM;     /* 132/8 = 16 */
    std::vector<CUdevResource> groups(nb);
    CUdevResource remaining;
    CK(cuDevSmResourceSplitByCount(groups.data(), &nb, &full,
                                   &remaining, 0, MIN_SM));
    if (nb == 0) { printf("RESULT status=SPLIT_ZERO_GROUPS\n"); return 3; }

    CUdevResourceDesc desc;
    CK(cuDevResourceGenerateDesc(&desc, &groups[0], 1));
    CUgreenCtx gctx;
    CK(cuGreenCtxCreate(&gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext gcuctx;
    CK(cuCtxFromGreenCtx(&gcuctx, gctx));

    CUdevResource vres;
    CK(cuGreenCtxGetDevResource(gctx, &vres, CU_DEV_RESOURCE_TYPE_SM));
    printf("[3] green context created — VERIFIED %u SMs "
           "(device has %u)\n", vres.sm.smCount, full.sm.smCount);
    if (vres.sm.smCount == full.sm.smCount) {
        printf("RESULT status=GREEN_PARTITION_NOT_TAKEN\n");
        return 3;
    }

    /* ── 4. allocate device buffers UNDER PRIMARY ───────────────────── */
    CK(cuCtxSetCurrent(primary));
    CUdeviceptr d_in, d_out, d_smids;
    CK(cuMemAlloc(&d_in,    N * sizeof(float)));
    CK(cuMemAlloc(&d_out,   N * sizeof(float)));
    CK(cuMemAlloc(&d_smids, GRID * sizeof(int)));
    std::vector<float> h_in(N);
    for (int i = 0; i < N; i++) h_in[i] = (float)i * 0.5f;
    CK(cuMemcpyHtoD(d_in, h_in.data(), N * sizeof(float)));
    CK(cuMemsetD32(d_smids, 0xffffffffu, GRID));
    printf("[4] %d-elem input + output + smid buffers allocated under "
           "PRIMARY ctx\n", N);

    /* ── 5. green-bound stream ──────────────────────────────────────── */
    CK(cuCtxSetCurrent(gcuctx));                 /* green ctx current */
    CUstream stream;
    CK(cuStreamCreate(&stream, CU_STREAM_DEFAULT)); /* stream binds to green */
    CUcontext sctx;
    CK(cuStreamGetCtx(stream, &sctx));
    printf("[5] stream created with green ctx current — "
           "stream ctx %s green cuctx\n", sctx == gcuctx ? "==" : "!=");

    /* ── 6. launch the PRIMARY-loaded kernel on the GREEN stream ────── */
    int n = N;
    void *args[] = { &d_in, &d_out, &n, &d_smids };
    printf("[6] launch grid=%d block=%d : primary module, primary memory, "
           "green stream ...\n", GRID, BLOCK);
    CUresult lr = cuLaunchKernel(fn, GRID, 1, 1, BLOCK, 1, 1,
                                 0, stream, args, nullptr);
    if (lr != CUDA_SUCCESS) {
        const char *m = nullptr; cuGetErrorString(lr, &m);
        printf("RESULT status=LAUNCH_FAIL rc=%d msg=%s\n", lr, m?m:"?");
        return 3;
    }
    CUresult sr = cuStreamSynchronize(stream);
    if (sr != CUDA_SUCCESS) {
        const char *m = nullptr; cuGetErrorString(sr, &m);
        printf("RESULT status=SYNC_FAIL rc=%d msg=%s\n", sr, m?m:"?");
        return 3;
    }
    printf("[6] launch + sync OK\n");

    /* ── 7. verify ──────────────────────────────────────────────────── */
    std::vector<float> h_out(N);
    std::vector<int>   h_smids(GRID);
    CK(cuCtxSetCurrent(primary));
    CK(cuMemcpyDtoH(h_out.data(),   d_out,   N * sizeof(float)));
    CK(cuMemcpyDtoH(h_smids.data(), d_smids, GRID * sizeof(int)));

    int bad = 0;
    for (int i = 0; i < N; i++) {
        float want = h_in[i] * 2.0f + (float)(i / BLOCK);
        if (h_out[i] != want) {
            if (bad < 4)
                printf("    mismatch i=%d got=%f want=%f\n",
                       i, h_out[i], want);
            bad++;
        }
    }
    std::set<int> smset(h_smids.begin(), h_smids.end());
    printf("[7] output: %d/%d elements correct ; distinct smids touched=%zu "
           "(<=8 => confined)\n", N - bad, N, smset.size());
    printf("    smids:");
    for (int s : smset) printf(" %d", s);
    printf("\n");

    cuMemFree(d_in); cuMemFree(d_out); cuMemFree(d_smids);
    cuStreamDestroy(stream);
    cuGreenCtxDestroy(gctx);
    cuModuleUnload(mod);

    if (bad != 0) {
        printf("RESULT status=KU2_FAIL detail=output_wrong bad=%d\n", bad);
        return 1;
    }
    printf("RESULT status=KU2_PASS — primary-loaded module + "
           "primary-allocated memory execute correctly on a green-ctx "
           "stream; distinct_smids=%zu\n", smset.size());
    return 0;
}
