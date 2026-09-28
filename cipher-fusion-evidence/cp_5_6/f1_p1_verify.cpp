/* CP 5.6 Priority 1 — full-GPU-path cross-context exposure verify.
 *
 * Variant F proved the GREEN-ctx path of c2c5d313 races. dc804eb3's CP 5.3
 * STEP 2 fix only scopes PrimaryCtxGuard out when the LIBRARY's own green
 * context is active. The no-green / full-GPU path
 * (cipher_rt_green_ctx_sm_count()==0) still wraps the GEMM in
 * PrimaryCtxGuard UNCONDITIONALLY — so a caller running on ANY non-primary
 * context hits the same producer(primary) / consumer(caller) cross-context
 * race that finding F1 describes.
 *
 * This harness reproduces that exposure. It builds the caller's non-primary
 * context with the RAW driver green-ctx API (cuGreenCtxCreate), NOT cipher's
 * cipher_rt_green_ctx_ensure() — so the library's internal g_green_sm_count
 * stays 0 and dispatch is forced down the full-GPU branch. A raw green ctx
 * still shares the primary context's address space (Variant F established
 * this empirically), so the primary-allocated W/A/C buffers stay reachable.
 *
 * Per iter: memset(C,0) on caller ctx ; dispatch ; cuMemcpyDtoH(C) on caller
 * ctx — NO cross-context synchronisation.
 *
 *   argv[1] = path to the engine .so.
 *     baseline (pre-fix dc804eb3 / .pre_cp56) -> expect BREAK
 *     CP 5.6 P1-fixed libcipher_rt.so          -> expect PASS
 *
 * Exit 0 = output stayed correct; exit 1 = degenerate (race reproduced).
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <dlfcn.h>
#include <cuda.h>

static float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) {
        if (m == 0) { o = s << 31; }
        else { int ei = -1; while (!(m & 0x400u)) { m <<= 1; ei--; }
               m &= 0x3ffu; o = (s << 31) | ((uint32_t)(127 - 15 + 1 + ei) << 23) | (m << 13); }
    } else if (e == 0x1f) { o = (s << 31) | (0xffu << 23) | (m << 13); }
    else { o = (s << 31) | ((e - 15 + 127) << 23) | (m << 13); }
    float f; memcpy(&f, &o, 4); return f;
}

/* CUDA driver API resolved by dlsym (no CUDA linkage). */
static decltype(&cuInit)                     cuInit_;
static decltype(&cuDeviceGet)                cuDeviceGet_;
static decltype(&cuDevicePrimaryCtxRetain)   cuPrimaryCtxRetain_;
static decltype(&cuCtxSetCurrent)            cuCtxSetCurrent_;
static decltype(&cuCtxGetCurrent)            cuCtxGetCurrent_;
static decltype(&cuMemAlloc)                 cuMemAlloc_;
static decltype(&cuMemcpyHtoD)               cuMemcpyHtoD_;
static decltype(&cuMemcpyDtoH)               cuMemcpyDtoH_;
static decltype(&cuMemsetD8)                 cuMemsetD8_;
static decltype(&cuCtxSynchronize)           cuCtxSynchronize_;
static decltype(&cuDeviceGetDevResource)     cuDeviceGetDevResource_;
static decltype(&cuDevSmResourceSplitByCount) cuDevSmResourceSplitByCount_;
static decltype(&cuDevResourceGenerateDesc)  cuDevResourceGenerateDesc_;
static decltype(&cuGreenCtxCreate)           cuGreenCtxCreate_;
static decltype(&cuCtxFromGreenCtx)          cuCtxFromGreenCtx_;

#define CK(x) do { CUresult _r = (CUresult)(x); if (_r != CUDA_SUCCESS) { \
    fprintf(stderr, "FAIL %s rc=%d\n", #x, (int)_r); exit(2); } } while (0)

static std::vector<uint8_t> readbin(const char* p, long expect) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (expect && n != expect) { fprintf(stderr, "%s size %ld != %ld\n", p, n, expect); exit(2); }
    std::vector<uint8_t> b(n);
    if ((long)fread(b.data(), 1, n, f) != n) { fprintf(stderr, "short read %s\n", p); exit(2); }
    fclose(f); return b;
}

static const int M = 1, K = 4096, N = 4096;
static const float* g_Yref;

static bool degenerate(const uint16_t* hC, const char* tag, bool verbose) {
    int nan = 0, inf = 0, zeros = 0;
    double l2d = 0, l2r = 0, l2c = 0, dot = 0, maxabs = 0;
    for (int i = 0; i < N; i++) {
        float c = h2f(hC[i]);
        if (std::isnan(c)) nan++;
        if (std::isinf(c)) inf++;
        if (c == 0.0f) zeros++;
        double d = (double)c - g_Yref[i];
        l2d += d * d; l2r += (double)g_Yref[i] * g_Yref[i];
        l2c += (double)c * c; dot += (double)c * g_Yref[i];
        if (std::fabs(d) > maxabs) maxabs = std::fabs(d);
    }
    double rel = std::sqrt(l2d) / (std::sqrt(l2r) + 1e-12);
    double cos = dot / (std::sqrt(l2c) * std::sqrt(l2r) + 1e-12);
    bool bad = nan || inf || cos < 0.85 || rel > 0.35;
    if (verbose || bad)
        printf("    %-26s nan=%d inf=%d zeros=%d max_abs=%.4f rel=%.4f cos=%.4f  %s\n",
               tag, nan, inf, zeros, maxabs, rel, cos, bad ? "<<< DEGENERATE" : "ok");
    return bad;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <engine.so>\n", argv[0]); return 2; }
    const char* SO = argv[1];

    const char* DIR = "cipher-fusion-evidence/cp_5_6/";
    auto path = [&](const char* f) { static char b[256]; snprintf(b, sizeof b, "%s%s", DIR, f); return b; };

    void* cu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!cu) { fprintf(stderr, "no libcuda\n"); return 2; }
    cuInit_=(decltype(cuInit_))dlsym(cu,"cuInit");
    cuDeviceGet_=(decltype(cuDeviceGet_))dlsym(cu,"cuDeviceGet");
    cuPrimaryCtxRetain_=(decltype(cuPrimaryCtxRetain_))dlsym(cu,"cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_=(decltype(cuCtxSetCurrent_))dlsym(cu,"cuCtxSetCurrent");
    cuCtxGetCurrent_=(decltype(cuCtxGetCurrent_))dlsym(cu,"cuCtxGetCurrent");
    cuMemAlloc_=(decltype(cuMemAlloc_))dlsym(cu,"cuMemAlloc_v2");
    cuMemcpyHtoD_=(decltype(cuMemcpyHtoD_))dlsym(cu,"cuMemcpyHtoD_v2");
    cuMemcpyDtoH_=(decltype(cuMemcpyDtoH_))dlsym(cu,"cuMemcpyDtoH_v2");
    cuMemsetD8_=(decltype(cuMemsetD8_))dlsym(cu,"cuMemsetD8_v2");
    cuCtxSynchronize_=(decltype(cuCtxSynchronize_))dlsym(cu,"cuCtxSynchronize");
    cuDeviceGetDevResource_=(decltype(cuDeviceGetDevResource_))dlsym(cu,"cuDeviceGetDevResource");
    cuDevSmResourceSplitByCount_=(decltype(cuDevSmResourceSplitByCount_))dlsym(cu,"cuDevSmResourceSplitByCount");
    cuDevResourceGenerateDesc_=(decltype(cuDevResourceGenerateDesc_))dlsym(cu,"cuDevResourceGenerateDesc");
    cuGreenCtxCreate_=(decltype(cuGreenCtxCreate_))dlsym(cu,"cuGreenCtxCreate");
    cuCtxFromGreenCtx_=(decltype(cuCtxFromGreenCtx_))dlsym(cu,"cuCtxFromGreenCtx");
    if (!cuDeviceGetDevResource_||!cuDevSmResourceSplitByCount_||!cuDevResourceGenerateDesc_||
        !cuGreenCtxCreate_||!cuCtxFromGreenCtx_) {
        fprintf(stderr,"green-ctx driver symbols unavailable\n"); return 2; }

    CUdevice dev; CUcontext primary;
    CK(cuInit_(0));
    CK(cuDeviceGet_(&dev, 0));
    CK(cuPrimaryCtxRetain_(&primary, dev));
    CK(cuCtxSetCurrent_(primary));

    /* inputs + buffers allocated while the PRIMARY context is current */
    auto hW = readbin(path("f1_W.bin"), (long)K*N*2);
    auto hA = readbin(path("f1_A.bin"), (long)M*K*2);
    auto hYref = readbin(path("f1_Yref.bin"), (long)M*N*4);
    g_Yref = (const float*)hYref.data();

    CUdeviceptr dW, dA, dC;
    CK(cuMemAlloc_(&dW,(size_t)K*N*2)); CK(cuMemAlloc_(&dA,(size_t)M*K*2));
    CK(cuMemAlloc_(&dC,(size_t)M*N*2));
    CK(cuMemcpyHtoD_(dW,hW.data(),hW.size()));
    CK(cuMemcpyHtoD_(dA,hA.data(),hA.size()));

    void* eng = dlopen(SO, RTLD_NOW | RTLD_LOCAL);
    if (!eng) { fprintf(stderr, "dlopen engine: %s\n", dlerror()); return 2; }
    auto eng_init=(int(*)(void))dlsym(eng,"cipher_rt_marlin_engine_init");
    auto eng_qr=(int(*)(const void*,int,int))dlsym(eng,"cipher_rt_marlin_engine_quantize_repack");
    auto eng_lookup=(int(*)(const void*,void**,void**,int*,int*,int*))dlsym(eng,"cipher_rt_marlin_engine_lookup");
    auto eng_dispatch=(int(*)(const void*,const void*,const void*,void*,int,int,int,int,void*))dlsym(eng,"cipher_rt_marlin_engine_dispatch");
    auto green_sm_count=(unsigned int(*)(void))dlsym(eng,"cipher_rt_green_ctx_sm_count");
    if (!eng_init||!eng_qr||!eng_lookup||!eng_dispatch) {
        fprintf(stderr,"engine dlsym incomplete\n"); return 2; }

    printf("=== CP 5.6 P1 verify — full-GPU path, caller on non-primary ctx\n    engine = %s\n", SO);

    eng_init();
    if (eng_qr((const void*)dW,K,N)!=0){fprintf(stderr,"quantize_repack failed\n");return 2;}
    void *B=nullptr,*S=nullptr; int lK,lN,lG;
    if (eng_lookup((const void*)dW,&B,&S,&lK,&lN,&lG)!=1){fprintf(stderr,"lookup failed\n");return 2;}

    /* ---- build the caller's non-primary context with the RAW driver API
     * (not cipher_rt_green_ctx_ensure) so the library's g_green_sm_count
     * stays 0 and dispatch is forced down the full-GPU branch ---- */
    CUdevResource fullSM;
    CK(cuDeviceGetDevResource_(dev, &fullSM, CU_DEV_RESOURCE_TYPE_SM));
    CUdevResource groups[16]; CUdevResource remaining; unsigned nb = 16;
    CK(cuDevSmResourceSplitByCount_(groups, &nb, &fullSM, &remaining, 0, 8));
    if (nb == 0) { fprintf(stderr,"SM split returned 0 groups\n"); return 2; }
    CUdevResourceDesc desc;
    CK(cuDevResourceGenerateDesc_(&desc, &groups[0], 1));
    CUgreenCtx gctx;
    CK(cuGreenCtxCreate_(&gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUcontext caller_ctx;
    CK(cuCtxFromGreenCtx_(&caller_ctx, gctx));
    printf("[ctx] raw green-derived caller ctx %p created (primary=%p)\n",
           (void*)caller_ctx, (void*)primary);

    unsigned int lib_sm = green_sm_count ? green_sm_count() : 0u;
    printf("[ctx] library cipher_rt_green_ctx_sm_count() = %u  -> dispatch takes %s path\n",
           lib_sm, lib_sm==0 ? "FULL-GPU (PrimaryCtxGuard)" : "GREEN (guard scoped out)");
    if (lib_sm != 0) {
        fprintf(stderr,"library green ctx is active — this run would test the green path, not full-GPU\n");
        return 2;
    }

    /* switch the caller onto the non-primary context */
    CK(cuCtxSetCurrent_(caller_ctx));
    CUcontext now=nullptr; cuCtxGetCurrent_(&now);
    printf("[ctx] caller now on %p (== primary? %s)\n", (void*)now, now==primary?"YES — bad":"no");

    /* ---- dispatch under the non-primary ctx, read C with NO cross-ctx sync ---- */
    const int iters = 100;
    std::vector<uint16_t> hC(M*N);
    int ndeg = 0;
    for (int i = 0; i < iters; i++) {
        CK(cuMemsetD8_(dC,0,(size_t)M*N*2));
        if (eng_dispatch((const void*)dA,B,S,(void*)dC,M,N,K,lG,nullptr)!=0) {
            fprintf(stderr,"dispatch rc!=0 i=%d\n",i); return 2; }
        /* NO cuCtxSynchronize, NO cross-context sync */
        CK(cuMemcpyDtoH_(hC.data(),dC,(size_t)M*N*2));
        char t[40]; snprintf(t,sizeof t,"iter %d",i);
        if (degenerate(hC.data(),t,i==0||i==iters-1)) ndeg++;
    }

    printf("\n--- last iter first 10 (engine | cuBLAS-FP16 ref) ---\n");
    for (int i=0;i<10;i++) printf("  [%d] %12.5f | %12.5f\n", i, h2f(hC[i]), g_Yref[i]);

    /* control: drain the primary context, then re-read — proves the GEMM
     * math is sound and the defect is purely the missing cross-context order */
    CK(cuCtxSetCurrent_(primary));
    CK(cuCtxSynchronize_());
    CK(cuCtxSetCurrent_(caller_ctx));
    CK(cuMemcpyDtoH_(hC.data(),dC,(size_t)M*N*2));
    printf("\n--- control: after primary-context drain ---\n");
    bool ctrl_bad = degenerate(hC.data(),"post-sync re-read",true);

    printf("\nP1 verify  degenerate iters: %d / %d\n", ndeg, iters);
    bool broke = ndeg > 0;
    printf("RESULT: %s   (control post-sync: %s)\n",
           broke ? "BROKE (full-GPU-path cross-context race reproduced)"
                 : "PASS (output stayed correct — full-GPU path is cross-context safe)",
           ctrl_bad ? "ALSO BAD — not a pure race" : "correct — GEMM math sound");
    return broke ? 1 : 0;
}
