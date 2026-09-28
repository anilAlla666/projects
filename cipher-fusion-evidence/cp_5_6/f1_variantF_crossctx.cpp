/* CP 5.6 / finding F1 — Variant F: cross-context reproduction.
 *
 * Steps 1-2 proved the single-GEMM is numerically sound and that no
 * single-context / stream-level variant (A-E) degrades it. Variant F adds
 * the one structural element A-E could not express: a real SECOND CUDA
 * context. It creates an 8-SM green CUcontext via the library's OWN
 * cipher_rt_green_ctx_ensure() (the exact API path the real decode uses —
 * so the library's internal g_green_sm_count is set, which dc804eb3's
 * dispatch reads), switches the caller onto the green context, runs
 * cipher_rt_marlin_engine_dispatch, then reads output buffer C from the
 * green context WITH NO CROSS-CONTEXT SYNCHRONISATION.
 *
 *   c2c5d313: dispatch unconditionally wraps the GEMM in PrimaryCtxGuard
 *             -> GEMM + its write to C run on the PRIMARY context while the
 *             caller (and the DtoH read) are on the GREEN context. Two
 *             contexts, independent unsynchronised streams -> race.
 *   dc804eb3: CP 5.3 STEP 2 scopes the guard out when a green context is
 *             active -> the GEMM launches on the caller's GREEN context, on
 *             the same NULL stream the read uses -> ordered -> coherent.
 *
 * argv[1] selects the engine build: "c2c5d313" | "dc804eb3".
 * Exit 0 = output stayed correct; exit 1 = degenerate (race reproduced).
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <dlfcn.h>

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

typedef unsigned long long CUdptr;
static int (*cuInit_)(unsigned);
static int (*cuDeviceGet_)(int*, int);
static int (*cuPrimaryCtxRetain_)(void**, int);
static int (*cuCtxSetCurrent_)(void*);
static int (*cuCtxGetCurrent_)(void**);
static int (*cuMemAlloc_)(CUdptr*, size_t);
static int (*cuMemcpyHtoD_)(CUdptr, const void*, size_t);
static int (*cuMemcpyDtoH_)(void*, CUdptr, size_t);
static int (*cuMemsetD8_)(CUdptr, unsigned char, size_t);
static int (*cuCtxSynchronize_)(void);

#define CK(x) do { int _r = (x); if (_r != 0) { \
    fprintf(stderr, "FAIL %s rc=%d\n", #x, _r); exit(2); } } while (0)

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
    if (argc < 2) { fprintf(stderr, "usage: %s c2c5d313|dc804eb3\n", argv[0]); return 2; }
    const char* tag = argv[1];
    const char* SO;
    if      (!strcmp(tag, "c2c5d313")) SO = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.pre_cp5_3_step2";
    else if (!strcmp(tag, "dc804eb3")) SO = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so";
    else { fprintf(stderr, "unknown build %s\n", tag); return 2; }

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

    int dev; void* primary;
    CK(cuInit_(0));
    CK(cuDeviceGet_(&dev, 0));
    CK(cuPrimaryCtxRetain_(&primary, dev));
    CK(cuCtxSetCurrent_(primary));

    /* inputs + buffers allocated while the PRIMARY context is current */
    auto hW = readbin(path("f1_W.bin"), (long)K*N*2);
    auto hA = readbin(path("f1_A.bin"), (long)M*K*2);
    auto hYref = readbin(path("f1_Yref.bin"), (long)M*N*4);
    g_Yref = (const float*)hYref.data();

    CUdptr dW, dA, dC;
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
    auto green_ensure=(int(*)(void))dlsym(eng,"cipher_rt_green_ctx_ensure");
    auto green_make_current=(int(*)(void))dlsym(eng,"cipher_rt_green_ctx_make_current");
    auto green_sm_count=(unsigned int(*)(void))dlsym(eng,"cipher_rt_green_ctx_sm_count"); /* dc804eb3 only */
    if (!eng_init||!eng_qr||!eng_lookup||!eng_dispatch||!green_ensure||!green_make_current) {
        fprintf(stderr,"engine dlsym incomplete\n"); return 2; }

    printf("=== Variant F — cross-context  (engine=%s)\n    %s\n", tag, SO);

    eng_init();
    if (eng_qr((const void*)dW,K,N)!=0){fprintf(stderr,"quantize_repack failed\n");return 2;}
    void *B=nullptr,*S=nullptr; int lK,lN,lG;
    if (eng_lookup((const void*)dW,&B,&S,&lK,&lN,&lG)!=1){fprintf(stderr,"lookup failed\n");return 2;}

    /* ---- create the 8-SM green context and switch the caller onto it ---- */
    int ge = green_ensure();
    printf("[green] cipher_rt_green_ctx_ensure() rc=%d", ge);
    if (green_sm_count) printf("  sm_count=%u", green_sm_count());
    printf("\n");
    if (ge != 0) { fprintf(stderr,"green ctx ensure failed — Variant F cannot run\n"); return 2; }

    void *cur_before=nullptr, *cur_after=nullptr;
    cuCtxGetCurrent_(&cur_before);
    int gm = green_make_current();
    cuCtxGetCurrent_(&cur_after);
    printf("[green] make_current() rc=%d   caller ctx %p -> %p   (primary=%p)\n",
           gm, cur_before, cur_after, primary);
    if (cur_after == primary)
        printf("[green] WARNING: caller still on primary context — green path NOT engaged\n");

    /* ---- Variant F: dispatch under green ctx, read C with NO cross-ctx sync ---- */
    const int iters = 100;
    std::vector<uint16_t> hC(M*N);
    int ndeg = 0;
    for (int i = 0; i < iters; i++) {
        CK(cuMemsetD8_(dC,0,(size_t)M*N*2));          /* green ctx: zero C first */
        if (eng_dispatch((const void*)dA,B,S,(void*)dC,M,N,K,lG,nullptr)!=0) {
            fprintf(stderr,"dispatch rc!=0 i=%d\n",i); return 2; }
        /* NO cuCtxSynchronize, NO cross-context sync — read straight away */
        CK(cuMemcpyDtoH_(hC.data(),dC,(size_t)M*N*2));
        char t[40]; snprintf(t,sizeof t,"iter %d",i);
        if (degenerate(hC.data(),t,i==0||i==iters-1)) ndeg++;
    }

    printf("\n--- iter 0 first 10 (engine | cuBLAS-FP16 ref) ---\n");
    /* re-show the last hC sample values for eyeballing */
    for (int i=0;i<10;i++) printf("  [%d] %12.5f | %12.5f\n", i, h2f(hC[i]), g_Yref[i]);

    /* ---- control: drain the primary context, then re-read ----
     * Proves the GEMM math itself is fine — only the missing cross-context
     * ordering degrades it. This is a cross-context sync, done ONCE after the
     * Variant-F loop, purely as a diagnostic. */
    CK(cuCtxSetCurrent_(primary));
    CK(cuCtxSynchronize_());
    CK(cuCtxSetCurrent_(cur_after));   /* back to green */
    CK(cuMemcpyDtoH_(hC.data(),dC,(size_t)M*N*2));
    printf("\n--- control: after primary-context drain ---\n");
    bool ctrl_bad = degenerate(hC.data(),"post-sync re-read",true);

    printf("\nVariant F  degenerate iters: %d / %d\n", ndeg, iters);
    bool broke = ndeg > 0;
    printf("RESULT engine=%s: %s   (control post-sync: %s)\n", tag,
           broke ? "BROKE (cross-context race reproduced)" : "PASS (output stayed correct)",
           ctrl_bad ? "ALSO BAD — not a pure race" : "correct — GEMM math sound, race is the missing sync");
    return broke ? 1 : 0;
}
