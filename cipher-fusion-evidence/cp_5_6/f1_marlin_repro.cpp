/* CP 5.6 / finding F1 — minimal single-GEMM reproduction.
 *
 * Drives the libcipher_rt c2c5d313 public engine C API directly (bypassing the
 * cublas-interception actuator) on ONE real weight: Mistral-7B layer-0 q_proj
 * [4096,4096]. Compares the engine's INT4 Marlin GEMM output against a
 * cuBLAS-FP16 reference (torch F.linear, computed by f1_prep.py).
 *
 *   quantize_repack(W,K,N) -> lookup(W,&B,&S,..) -> dispatch(A,B,S,C,M,N,K,G)
 *
 * Pure CUDA driver API (dlopen libcuda.so.1) for buffer management so this TU
 * links no CUDA runtime — the engine resolves its own libcudart.so.13. The
 * engine .so is dlopen'd RTLD_LOCAL so its cublas hooks never touch us.
 *
 * Run against c2c5d313 = libcipher_rt.so.pre_cp5_3_step2 (the F1-broken build).
 * Build/run: see build_repro.sh.
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
static int (*cuMemAlloc_)(CUdptr*, size_t);
static int (*cuMemcpyHtoD_)(CUdptr, const void*, size_t);
static int (*cuMemcpyDtoH_)(void*, CUdptr, size_t);
static int (*cuMemsetD8_)(CUdptr, unsigned char, size_t);
static int (*cuCtxSynchronize_)(void);

#define CK(x) do { int _r = (x); if (_r != 0) { \
    fprintf(stderr, "FAIL %s rc=%d\n", #x, _r); return 2; } } while (0)

static std::vector<uint8_t> readbin(const char* p, long expect) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (expect && n != expect) {
        fprintf(stderr, "%s: size %ld != expected %ld\n", p, n, expect); exit(2); }
    std::vector<uint8_t> b(n);
    if ((long)fread(b.data(), 1, n, f) != n) { fprintf(stderr, "short read %s\n", p); exit(2); }
    fclose(f); return b;
}

int main() {
    const int M = 1, K = 4096, N = 4096;
    const char* DIR = "cipher-fusion-evidence/cp_5_6/";
    auto path = [&](const char* f) { static char b[256]; snprintf(b, sizeof b, "%s%s", DIR, f); return b; };

    /* ---- CUDA driver API ---- */
    void* cu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!cu) { fprintf(stderr, "no libcuda.so.1: %s\n", dlerror()); return 2; }
    cuInit_             = (decltype(cuInit_))            dlsym(cu, "cuInit");
    cuDeviceGet_        = (decltype(cuDeviceGet_))       dlsym(cu, "cuDeviceGet");
    cuPrimaryCtxRetain_ = (decltype(cuPrimaryCtxRetain_))dlsym(cu, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_    = (decltype(cuCtxSetCurrent_))   dlsym(cu, "cuCtxSetCurrent");
    cuMemAlloc_         = (decltype(cuMemAlloc_))        dlsym(cu, "cuMemAlloc_v2");
    cuMemcpyHtoD_       = (decltype(cuMemcpyHtoD_))      dlsym(cu, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_       = (decltype(cuMemcpyDtoH_))      dlsym(cu, "cuMemcpyDtoH_v2");
    cuMemsetD8_         = (decltype(cuMemsetD8_))        dlsym(cu, "cuMemsetD8_v2");
    cuCtxSynchronize_   = (decltype(cuCtxSynchronize_))  dlsym(cu, "cuCtxSynchronize");

    int dev; void* ctx;
    CK(cuInit_(0));
    CK(cuDeviceGet_(&dev, 0));
    CK(cuPrimaryCtxRetain_(&ctx, dev));
    CK(cuCtxSetCurrent_(ctx));

    /* ---- inputs ---- */
    auto hW = readbin(path("f1_W.bin"),    (long)K * N * 2);
    auto hA = readbin(path("f1_A.bin"),    (long)M * K * 2);
    auto hYref  = readbin(path("f1_Yref.bin"),  (long)M * N * 4);
    auto hYint4 = readbin(path("f1_Yint4.bin"), (long)M * N * 4);
    const float* Yref  = (const float*)hYref.data();
    const float* Yint4 = (const float*)hYint4.data();

    CUdptr dW, dA, dC;
    CK(cuMemAlloc_(&dW, (size_t)K * N * 2));
    CK(cuMemAlloc_(&dA, (size_t)M * K * 2));
    CK(cuMemAlloc_(&dC, (size_t)M * N * 2));
    CK(cuMemcpyHtoD_(dW, hW.data(), hW.size()));
    CK(cuMemcpyHtoD_(dA, hA.data(), hA.size()));
    CK(cuMemsetD8_(dC, 0, (size_t)M * N * 2));

    /* ---- libcipher_rt c2c5d313 engine ---- */
    const char* SO = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.pre_cp5_3_step2";
    void* eng = dlopen(SO, RTLD_NOW | RTLD_LOCAL);
    if (!eng) { fprintf(stderr, "dlopen engine: %s\n", dlerror()); return 2; }
    auto eng_init     = (int(*)(void))                                     dlsym(eng, "cipher_rt_marlin_engine_init");
    auto eng_qr       = (int(*)(const void*, int, int))                    dlsym(eng, "cipher_rt_marlin_engine_quantize_repack");
    auto eng_lookup   = (int(*)(const void*, void**, void**, int*, int*, int*)) dlsym(eng, "cipher_rt_marlin_engine_lookup");
    auto eng_dispatch = (int(*)(const void*, const void*, const void*, void*, int, int, int, int, void*)) dlsym(eng, "cipher_rt_marlin_engine_dispatch");
    if (!eng_init || !eng_qr || !eng_lookup || !eng_dispatch) {
        fprintf(stderr, "engine dlsym incomplete\n"); return 2; }

    printf("[1] engine_init rc=%d\n", eng_init());
    int rc_qr = eng_qr((const void*)dW, K, N);
    printf("[2] quantize_repack(W,K=%d,N=%d) rc=%d\n", K, N, rc_qr);
    if (rc_qr != 0) { fprintf(stderr, "quantize_repack failed\n"); return 2; }

    void *B = nullptr, *S = nullptr; int lK = 0, lN = 0, lG = 0;
    int rc_lk = eng_lookup((const void*)dW, &B, &S, &lK, &lN, &lG);
    printf("[3] lookup rc=%d  B=%p S=%p  K=%d N=%d G=%d\n", rc_lk, B, S, lK, lN, lG);
    if (rc_lk != 1 || !B || !S) { fprintf(stderr, "lookup failed\n"); return 2; }

    int rc_d = eng_dispatch((const void*)dA, B, S, (void*)dC, M, N, K, lG, nullptr);
    printf("[4] dispatch(M=%d,N=%d,K=%d,G=%d) rc=%d\n", M, N, K, lG, rc_d);
    if (rc_d != 0) { fprintf(stderr, "dispatch failed\n"); return 2; }
    CK(cuCtxSynchronize_());

    /* ---- compare ---- */
    std::vector<uint16_t> hC(M * N);
    CK(cuMemcpyDtoH_(hC.data(), dC, (size_t)M * N * 2));

    int n_nan = 0, n_inf = 0;
    double l2d_ref = 0, l2_ref = 0, l2_c = 0, dot = 0, l2d_q = 0, l2_q = 0, max_abs = 0;
    for (int i = 0; i < N; i++) {
        float c = h2f(hC[i]);
        if (std::isnan(c)) n_nan++;
        if (std::isinf(c)) n_inf++;
        float r = Yref[i], q = Yint4[i];
        double d = (double)c - r;
        l2d_ref += d * d; l2_ref += (double)r * r; l2_c += (double)c * c; dot += (double)c * r;
        l2d_q += ((double)c - q) * ((double)c - q); l2_q += (double)q * q;
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
    }
    double rel_ref = std::sqrt(l2d_ref) / (std::sqrt(l2_ref) + 1e-12);
    double rel_q   = std::sqrt(l2d_q)  / (std::sqrt(l2_q)   + 1e-12);
    double cos     = dot / (std::sqrt(l2_c) * std::sqrt(l2_ref) + 1e-12);

    printf("\n--- first 10 outputs (engine | cuBLAS-FP16 | INT4-floor) ---\n");
    for (int i = 0; i < 10; i++)
        printf("  [%d] %12.5f | %12.5f | %12.5f\n", i, h2f(hC[i]), Yref[i], Yint4[i]);

    printf("\n--- F1 single-GEMM verdict ---\n");
    printf("  nan=%d  inf=%d\n", n_nan, n_inf);
    printf("  max_abs_error      = %.6f\n", max_abs);
    printf("  rel_err vs cuBLAS  = %.6f   (INT4 noise floor ~0.14)\n", rel_ref);
    printf("  rel_err vs INT4sim = %.6f   (0 = matches expected quant output)\n", rel_q);
    printf("  cosine vs cuBLAS   = %.6f\n", cos);

    bool degenerate = n_nan || n_inf || cos < 0.85 || rel_ref > 0.35;
    if (degenerate)
        printf("\nRESULT: SINGLE-GEMM BROKEN  -> F1 root cause is in the engine GEMM; go to Step 3 bisection\n");
    else
        printf("\nRESULT: SINGLE-GEMM OK (rel_err within ~2x INT4 floor) -> bug is elsewhere; go to Step 2\n");
    return degenerate ? 1 : 0;
}
