/* CP 5.6 / finding F1 — STEP 2: integration-path bisection.
 *
 * Extends the Step 1 harness. The Step 1 single GEMM is correct; Step 2 adds
 * ONE variable at a time to the working repro until output goes degenerate.
 * Selected by argv[1]:
 *   A  100 sequential dispatches, same NULL stream, synced between each
 *   B  100 sequential dispatches, NULL stream, NO sync between (sync at end)
 *   C  each dispatch wrapped in an explicit cuCtxPushCurrent/Pop(primary)
 *   D  same as A but run under CUDA_INJECTION64_PATH (CUPTI green-ctx active)
 *   E  dispatches interleaved across TWO streams, no sync between
 *
 * Exit 0 = variant PASSED (output stayed correct); exit 1 = BROKE (degenerate).
 * Engine = libcipher_rt.so.pre_cp5_3_step2 (c2c5d313, the F1-broken build).
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
static int (*cuCtxPushCurrent_)(void*);
static int (*cuCtxPopCurrent_)(void**);
static int (*cuMemAlloc_)(CUdptr*, size_t);
static int (*cuMemcpyHtoD_)(CUdptr, const void*, size_t);
static int (*cuMemcpyDtoH_)(void*, CUdptr, size_t);
static int (*cuMemsetD8_)(CUdptr, unsigned char, size_t);
static int (*cuMemsetD8Async_)(CUdptr, unsigned char, size_t, void*);
static int (*cuStreamCreate_)(void**, unsigned);
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

/* returns true if degenerate; prints the line when verbose or when bad */
static bool degenerate(const uint16_t* hC, const char* tag, bool verbose) {
    int nan = 0, inf = 0;
    double l2d = 0, l2r = 0, l2c = 0, dot = 0, maxabs = 0;
    for (int i = 0; i < N; i++) {
        float c = h2f(hC[i]);
        if (std::isnan(c)) nan++;
        if (std::isinf(c)) inf++;
        double d = (double)c - g_Yref[i];
        l2d += d * d; l2r += (double)g_Yref[i] * g_Yref[i];
        l2c += (double)c * c; dot += (double)c * g_Yref[i];
        if (std::fabs(d) > maxabs) maxabs = std::fabs(d);
    }
    double rel = std::sqrt(l2d) / (std::sqrt(l2r) + 1e-12);
    double cos = dot / (std::sqrt(l2c) * std::sqrt(l2r) + 1e-12);
    bool bad = nan || inf || cos < 0.85 || rel > 0.35;
    if (verbose || bad)
        printf("    %-22s nan=%d inf=%d max_abs=%.4f rel=%.4f cos=%.4f  %s\n",
               tag, nan, inf, maxabs, rel, cos, bad ? "<<< DEGENERATE" : "ok");
    return bad;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s A|B|C|D|E\n", argv[0]); return 2; }
    char V = argv[1][0];
    const char* DIR = "cipher-fusion-evidence/cp_5_6/";
    auto path = [&](const char* f) { static char b[256]; snprintf(b, sizeof b, "%s%s", DIR, f); return b; };

    void* cu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!cu) { fprintf(stderr, "no libcuda\n"); return 2; }
    cuInit_=(decltype(cuInit_))dlsym(cu,"cuInit");
    cuDeviceGet_=(decltype(cuDeviceGet_))dlsym(cu,"cuDeviceGet");
    cuPrimaryCtxRetain_=(decltype(cuPrimaryCtxRetain_))dlsym(cu,"cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_=(decltype(cuCtxSetCurrent_))dlsym(cu,"cuCtxSetCurrent");
    cuCtxGetCurrent_=(decltype(cuCtxGetCurrent_))dlsym(cu,"cuCtxGetCurrent");
    cuCtxPushCurrent_=(decltype(cuCtxPushCurrent_))dlsym(cu,"cuCtxPushCurrent_v2");
    cuCtxPopCurrent_=(decltype(cuCtxPopCurrent_))dlsym(cu,"cuCtxPopCurrent_v2");
    cuMemAlloc_=(decltype(cuMemAlloc_))dlsym(cu,"cuMemAlloc_v2");
    cuMemcpyHtoD_=(decltype(cuMemcpyHtoD_))dlsym(cu,"cuMemcpyHtoD_v2");
    cuMemcpyDtoH_=(decltype(cuMemcpyDtoH_))dlsym(cu,"cuMemcpyDtoH_v2");
    cuMemsetD8_=(decltype(cuMemsetD8_))dlsym(cu,"cuMemsetD8_v2");
    cuMemsetD8Async_=(decltype(cuMemsetD8Async_))dlsym(cu,"cuMemsetD8Async");
    cuStreamCreate_=(decltype(cuStreamCreate_))dlsym(cu,"cuStreamCreate");
    cuCtxSynchronize_=(decltype(cuCtxSynchronize_))dlsym(cu,"cuCtxSynchronize");

    int dev; void* ctx;
    CK(cuInit_(0));
    CK(cuDeviceGet_(&dev, 0));
    CK(cuPrimaryCtxRetain_(&ctx, dev));
    CK(cuCtxSetCurrent_(ctx));

    auto hW = readbin(path("f1_W.bin"), (long)K*N*2);
    auto hA = readbin(path("f1_A.bin"), (long)M*K*2);
    auto hYref = readbin(path("f1_Yref.bin"), (long)M*N*4);
    g_Yref = (const float*)hYref.data();

    CUdptr dW, dA, dC, dC2;
    CK(cuMemAlloc_(&dW,(size_t)K*N*2)); CK(cuMemAlloc_(&dA,(size_t)M*K*2));
    CK(cuMemAlloc_(&dC,(size_t)M*N*2)); CK(cuMemAlloc_(&dC2,(size_t)M*N*2));
    CK(cuMemcpyHtoD_(dW,hW.data(),hW.size()));
    CK(cuMemcpyHtoD_(dA,hA.data(),hA.size()));

    const char* SO = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.pre_cp5_3_step2";
    void* eng = dlopen(SO, RTLD_NOW | RTLD_LOCAL);
    if (!eng) { fprintf(stderr, "dlopen engine: %s\n", dlerror()); return 2; }
    auto eng_init=(int(*)(void))dlsym(eng,"cipher_rt_marlin_engine_init");
    auto eng_qr=(int(*)(const void*,int,int))dlsym(eng,"cipher_rt_marlin_engine_quantize_repack");
    auto eng_lookup=(int(*)(const void*,void**,void**,int*,int*,int*))dlsym(eng,"cipher_rt_marlin_engine_lookup");
    auto eng_dispatch=(int(*)(const void*,const void*,const void*,void*,int,int,int,int,void*))dlsym(eng,"cipher_rt_marlin_engine_dispatch");

    eng_init();
    if (eng_qr((const void*)dW,K,N)!=0){fprintf(stderr,"quantize_repack failed\n");return 2;}
    void *B=nullptr,*S=nullptr; int lK,lN,lG;
    if (eng_lookup((const void*)dW,&B,&S,&lK,&lN,&lG)!=1){fprintf(stderr,"lookup failed\n");return 2;}

    const char* inj = getenv("CUDA_INJECTION64_PATH");
    printf("=== Variant %c  (CUDA_INJECTION64_PATH=%s) ===\n", V, inj?inj:"unset");
    std::vector<uint16_t> hC(M*N), hCb(M*N);
    int iters = 100;
    bool broke = false;

    if (V=='A' || V=='D') {
        for (int i=0;i<iters && !broke;i++){
            CK(cuMemsetD8_(dC,0,(size_t)M*N*2));
            if (eng_dispatch((const void*)dA,B,S,(void*)dC,M,N,K,lG,nullptr)!=0){fprintf(stderr,"dispatch rc!=0 i=%d\n",i);return 2;}
            CK(cuCtxSynchronize_());
            CK(cuMemcpyDtoH_(hC.data(),dC,(size_t)M*N*2));
            char t[32]; snprintf(t,sizeof t,"iter %d",i);
            if (degenerate(hC.data(),t,i==0||i==iters-1)) { broke=true; printf("    -> BREAK at iter %d\n",i); }
        }
    } else if (V=='B') {
        for (int i=0;i<iters;i++){
            CK(cuMemsetD8Async_(dC,0,(size_t)M*N*2,nullptr));
            if (eng_dispatch((const void*)dA,B,S,(void*)dC,M,N,K,lG,nullptr)!=0){fprintf(stderr,"dispatch rc!=0 i=%d\n",i);return 2;}
        }
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(hC.data(),dC,(size_t)M*N*2));
        broke = degenerate(hC.data(),"final (100 unsynced)",true);
    } else if (V=='C') {
        for (int i=0;i<iters && !broke;i++){
            CK(cuCtxPushCurrent_(ctx));
            CK(cuMemsetD8_(dC,0,(size_t)M*N*2));
            if (eng_dispatch((const void*)dA,B,S,(void*)dC,M,N,K,lG,nullptr)!=0){fprintf(stderr,"dispatch rc!=0 i=%d\n",i);return 2;}
            CK(cuCtxSynchronize_());
            void* popped; CK(cuCtxPopCurrent_(&popped));
            CK(cuMemcpyDtoH_(hC.data(),dC,(size_t)M*N*2));
            char t[32]; snprintf(t,sizeof t,"iter %d",i);
            if (degenerate(hC.data(),t,i==0||i==iters-1)) { broke=true; printf("    -> BREAK at iter %d\n",i); }
        }
    } else if (V=='E') {
        void *s1,*s2; CK(cuStreamCreate_(&s1,0)); CK(cuStreamCreate_(&s2,0));
        for (int i=0;i<iters/2;i++){
            CK(cuMemsetD8Async_(dC,0,(size_t)M*N*2,s1));
            if (eng_dispatch((const void*)dA,B,S,(void*)dC,M,N,K,lG,s1)!=0){fprintf(stderr,"dispatch s1 rc!=0\n");return 2;}
            CK(cuMemsetD8Async_(dC2,0,(size_t)M*N*2,s2));
            if (eng_dispatch((const void*)dA,B,S,(void*)dC2,M,N,K,lG,s2)!=0){fprintf(stderr,"dispatch s2 rc!=0\n");return 2;}
        }
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(hC.data(),dC,(size_t)M*N*2));
        CK(cuMemcpyDtoH_(hCb.data(),dC2,(size_t)M*N*2));
        bool b1=degenerate(hC.data(),"stream1 final",true), b2=degenerate(hCb.data(),"stream2 final",true);
        broke = b1||b2;
    } else { fprintf(stderr,"unknown variant %c\n",V); return 2; }

    printf("RESULT variant %c: %s\n", V, broke ? "BROKE (degenerate output)" : "PASS (output stayed correct)");
    return broke ? 1 : 0;
}
