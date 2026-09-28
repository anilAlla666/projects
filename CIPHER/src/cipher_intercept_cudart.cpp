// =============================================================================
// CIPHER — LD_PRELOAD kernel launch hook via PLT export + ELF GOT patching
// cipher_intercept_cudart.cpp
//
// Two-layer interception:
//   Layer 1: Exported symbols (cuLaunchKernel, cudaLaunchKernel, etc.)
//            Picked up by the dynamic linker for any DSO loaded after us.
//   Layer 2: GOT patching via dl_iterate_phdr.  Re-run after CUDA libs load
//            by calling cipher_repatch() or automatically on first shim call.
//
// No dlsym hooking.  No CUDA SDK linkage at compile time.
// =============================================================================

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <sys/mman.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <atomic>

// ── Minimal CUDA types ──────────────────────────────────────────────────────
typedef int      CUresult;
typedef void*    CUfunction;
typedef void*    CUstream;
typedef void*    cudaStream_t;
typedef int      cudaError_t;
typedef int      CUdriverProcAddressQueryResult;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_FOUND 500

struct dim3 { unsigned x, y, z; };

// cudaLaunchConfig_t — ABI-compatible minimal definition
struct cudaLaunchConfig_t {
    dim3     gridDim;
    dim3     blockDim;
    size_t   dynamicSmemBytes;
    void*    stream;
    void*    attrs;
    unsigned numAttrs;
};

// ── Function pointer types ──────────────────────────────────────────────────
typedef CUresult    (*cuLaunchKernel_fn)(CUfunction, unsigned, unsigned, unsigned,
                     unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
typedef CUresult    (*cuLaunchKernelEx_fn)(void*, CUfunction, void**, void**);
typedef cudaError_t (*cudaLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*cudaLaunchKernelExC_fn)(const cudaLaunchConfig_t*, const void*, void**);
typedef CUresult    (*cuGetProcAddress_fn)(const char*, void**, int, uint64_t, CUdriverProcAddressQueryResult*);

// ── Real function pointers ──────────────────────────────────────────────────
static cuLaunchKernel_fn      g_real_cuLaunch           = nullptr;
static cuLaunchKernel_fn      g_real_cuLaunch_ptsz      = nullptr;
static cuLaunchKernelEx_fn    g_real_cuLaunchEx         = nullptr;
static cuLaunchKernelEx_fn    g_real_cuLaunchEx_ptsz    = nullptr;
static cudaLaunchKernel_fn    g_real_cudaLaunch         = nullptr;
static cudaLaunchKernel_fn    g_real_cudaLaunch_ptsz    = nullptr;
static cudaLaunchKernelExC_fn g_real_cudaLaunchExC      = nullptr;
static cudaLaunchKernelExC_fn g_real_cudaLaunchExC_ptsz = nullptr;
static cuGetProcAddress_fn    g_real_cuGetProc          = nullptr;
static cuGetProcAddress_fn    g_real_cuGetProc_v2       = nullptr;

static std::atomic<uint64_t> g_intercept_count{0};
static std::atomic<uint64_t> g_proc_addr_count{0};
static std::atomic<uint64_t> g_got_patches{0};
static std::atomic<bool>     g_resolved{false};
static std::atomic<bool>     g_auto_patched{false};
static long                  g_page_size = 4096;

// Real glibc dlsym — saved from GOT before patching. Used by ensure_cublasLt
// and cipher_substitute_sgemm to avoid recursion through our dlsym wrapper.
static void* (*g_real_dlsym)(void*, const char*) = nullptr;

// ── Thread-local GEMM shape + pointer storage ───────────────────────────────
// Written by cublasGemmEx / cublasLtMatmul shims, read by cipher_dispatch.
static __thread int   tls_M = 0, tls_N = 0, tls_K = 0;
static __thread int   tls_shape_valid = 0;
static __thread const void* tls_A = nullptr;
static __thread const void* tls_B = nullptr;
static __thread void*       tls_C = nullptr;
static __thread const void* tls_alpha = nullptr;
static __thread const void* tls_beta  = nullptr;
static __thread void*       tls_user_stream = nullptr;
static __thread int   tls_Atype = 0, tls_Btype = 0, tls_Ctype = 0;
static __thread int   tls_lda = 0, tls_ldb = 0, tls_ldc = 0;
static __thread int   tls_computeType = 0;

// Export TLS accessors for libcipher_rt.so (called from cipher_dispatch)
extern "C" __attribute__((visibility("default")))
void cipher_tls_get_gemm_shape(int* m, int* n, int* k, int* valid) {
    *m = tls_M; *n = tls_N; *k = tls_K; *valid = tls_shape_valid;
}

extern "C" __attribute__((visibility("default")))
void cipher_tls_get_gemm_ptrs(const void** A, const void** B, void** C,
                               const void** alpha, const void** beta,
                               void** stream, int* valid) {
    *A = tls_A; *B = tls_B; *C = tls_C;
    *alpha = tls_alpha; *beta = tls_beta;
    *stream = tls_user_stream;
    *valid = tls_shape_valid;
}

// cuLaunchKernelEx args — stored so apply_recipe can re-launch the kernel
static __thread void*  tls_launch_config = nullptr;
static __thread void*  tls_launch_fn     = nullptr;
static __thread void** tls_launch_params = nullptr;
static __thread void** tls_launch_extra  = nullptr;
static __thread int    tls_launch_valid  = 0;

// Recursion guard — prevent dispatch loop when apply_recipe calls cuLaunchKernelEx
static __thread int tls_in_recipe = 0;

// Export: Python/C caller sets A,B,C,M,N,K before torch.mm
extern "C" __attribute__((visibility("default")))
void cipher_set_gemm_ptrs(int m, int n, int k,
                          const void* A, const void* B, void* C) {
    tls_M = m; tls_N = n; tls_K = k;
    tls_A = A; tls_B = B; tls_C = C;
    tls_shape_valid = 1;
}

extern "C" __attribute__((visibility("default")))
void cipher_clear_gemm_ptrs(void) {
    tls_shape_valid = 0;
}

// Export: let libcipher_rt.so re-launch the intercepted kernel
extern "C" __attribute__((visibility("default")))
int cipher_tls_relaunch(void) {
    if (!tls_launch_valid || !g_real_cuLaunchEx) return -1;
    tls_in_recipe = 1;
    CUresult r = g_real_cuLaunchEx(tls_launch_config,
                                    (CUfunction)tls_launch_fn,
                                    tls_launch_params,
                                    tls_launch_extra);
    tls_in_recipe = 0;
    return (int)r;
}

// ── cipher_substitute_sgemm — Op 3: direct cublasSgemm with TLS pointers ────
// Called from apply_recipe() in libcipher_rt.so.  Resolves cublasSgemm and
// cublasCreate via g_real_dlsym so we never recurse through our own wrappers.

typedef int (*cublasSgemm_fn)(void*, int, int, int, int, int,
    const float*, const float*, int, const float*, int,
    const float*, float*, int);
typedef int (*cublasCreate_fn)(void**);

static void*          g_cublas_handle  = nullptr;
static cublasSgemm_fn g_real_sgemm     = nullptr;

extern "C" __attribute__((visibility("default")))
int cipher_substitute_sgemm(int M, int N, int K,
                            const void* A, const void* B, void* C)
{
    // One-time resolve
    if (!g_real_sgemm) {
        auto real_sym = g_real_dlsym ? g_real_dlsym
                        : (void*(*)(void*,const char*))dlsym;
        void* h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen("libcublas.so.12", RTLD_NOW);
        if (!h) return -1;
        g_real_sgemm = (cublasSgemm_fn)real_sym(h, "cublasSgemm_v2");
        if (!g_real_sgemm)
            g_real_sgemm = (cublasSgemm_fn)real_sym(h, "cublasSgemm");
        auto cCreate = (cublasCreate_fn)real_sym(h, "cublasCreate_v2");
        if (!cCreate) return -2;
        if (cCreate(&g_cublas_handle) != 0) return -3;
    }
    if (!g_real_sgemm || !g_cublas_handle) return -4;

    // Row-major torch: C = A * B  →  column-major cuBLAS: C^T = B^T * A^T
    // cublasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // where "A" in cuBLAS = B (row-major), "B" in cuBLAS = A (row-major)
    static const float alpha = 1.0f, beta = 0.0f;
    tls_in_recipe = 1;
    int rc = g_real_sgemm(g_cublas_handle,
        0 /*CUBLAS_OP_N*/, 0 /*CUBLAS_OP_N*/,
        N, M, K,
        &alpha,
        (const float*)B, N,   // B^T col-major: N rows
        (const float*)A, K,   // A^T col-major: K rows
        &beta,
        (float*)C, N);        // C^T col-major: N rows
    tls_in_recipe = 0;
    return rc;
}

// ── cublasGemmEx shim — captures M,N,K before cuBLAS launches the kernel ────

typedef int cublasStatus_t;
typedef void* cublasHandle_t;

typedef cublasStatus_t (*cublasGemmEx_fn)(
    cublasHandle_t handle, int transa, int transb,
    int m, int n, int k,
    const void* alpha, const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta, void* C, int Ctype, int ldc,
    int computeType, int algo);

static cublasGemmEx_fn g_real_cublasGemmEx = nullptr;

// Base implementation — versioned alias created below
extern "C" __attribute__((visibility("default")))
cublasStatus_t cipher_cublasGemmEx_impl(
    cublasHandle_t handle, int transa, int transb,
    int m, int n, int k,
    const void* alpha, const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta, void* C, int Ctype, int ldc,
    int computeType, int algo)
{
    if (__builtin_expect(!g_real_cublasGemmEx, 0)) {
        void* h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen("libcublas.so.12", RTLD_NOW);
        if (h) g_real_cublasGemmEx = (cublasGemmEx_fn)dlsym(h, "cublasGemmEx");
        if (!g_real_cublasGemmEx) return 15; // CUBLAS_STATUS_NOT_SUPPORTED
    }
    // Store M,N,K in TLS — the upcoming cuLaunchKernelEx will read them
    tls_M = m; tls_N = n; tls_K = k;
    tls_A = A; tls_B = B; tls_C = C;
    tls_alpha = alpha; tls_beta = beta;
    tls_Atype = Atype; tls_Btype = Btype; tls_Ctype = Ctype;
    tls_lda = lda; tls_ldb = ldb; tls_ldc = ldc;
    tls_computeType = computeType;
    tls_shape_valid = 1;
    cublasStatus_t ret = g_real_cublasGemmEx(handle, transa, transb,
        m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
        beta, C, Ctype, ldc, computeType, algo);
    tls_shape_valid = 0;
    return ret;
}

// ── cublasLtMatmul shim — extracts M,N,K from matrix layout descriptors ─────

typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulAlgo_t;

typedef cublasStatus_t (*cublasLtMatmul_fn)(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    const void*, cublasLtMatrixLayout_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    void*, cublasLtMatrixLayout_t,
    const void*, void*, size_t, cudaStream_t);

typedef cublasStatus_t (*cublasLtLayoutGetAttr_fn)(
    cublasLtMatrixLayout_t, int, void*, size_t, size_t*);

static cublasLtMatmul_fn       g_real_cublasLtMatmul = nullptr;
static cublasLtLayoutGetAttr_fn g_cublasLtLayoutGetAttr = nullptr;

static void ensure_cublasLt() {
    if (g_real_cublasLtMatmul && g_cublasLtLayoutGetAttr) return;
    void* h = dlopen("libcublasLt.so.12", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcublasLt.so.12", RTLD_NOW);
    if (h) {
        // Use g_real_dlsym (the real glibc dlsym, not our patched wrapper)
        // to avoid getting our own shim back.
        auto real_sym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
        if (!g_real_cublasLtMatmul)
            g_real_cublasLtMatmul = (cublasLtMatmul_fn)real_sym(h, "cublasLtMatmul");
        if (!g_cublasLtLayoutGetAttr)
            g_cublasLtLayoutGetAttr = (cublasLtLayoutGetAttr_fn)real_sym(h, "cublasLtMatrixLayoutGetAttribute");
    }
}

extern "C" __attribute__((visibility("default")))
cublasStatus_t cipher_cublasLtMatmul_impl(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc,
    const void* B, cublasLtMatrixLayout_t Bdesc,
    const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc,
    void* D, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t algo, void* workspace,
    size_t workspaceSizeInBytes, cudaStream_t stream)
{
    ensure_cublasLt();
    if (!g_real_cublasLtMatmul) return 15;

    // Extract M,N,K from A and B matrix layouts
    // CUBLASLT_MATRIX_LAYOUT_ROWS = 2, CUBLASLT_MATRIX_LAYOUT_COLS = 3
    if (g_cublasLtLayoutGetAttr) {
        uint64_t rowsA=0, colsA=0, colsB=0;
        size_t written = 0;
        g_cublasLtLayoutGetAttr(Adesc, 2, &rowsA, sizeof(rowsA), &written);
        g_cublasLtLayoutGetAttr(Adesc, 3, &colsA, sizeof(colsA), &written);
        g_cublasLtLayoutGetAttr(Bdesc, 3, &colsB, sizeof(colsB), &written);
        tls_M = (int)rowsA;
        tls_K = (int)colsA;
        tls_N = (int)colsB;
        tls_A = A; tls_B = B; tls_C = D; // D is the output for cublasLtMatmul
        tls_alpha = alpha; tls_beta = beta;
        tls_user_stream = (void*)stream;
        tls_shape_valid = 1;
        fprintf(stderr, "[CIPHER HOOK] cublasLtMatmul intercepted: M=%d N=%d K=%d C=%p\n",
                tls_M, tls_N, tls_K, D);
    }

    cublasStatus_t ret = g_real_cublasLtMatmul(lightHandle, computeDesc,
        alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
        algo, workspace, workspaceSizeInBytes, stream);
    tls_shape_valid = 0;
    return ret;
}

// cublasLtMatmul versioned PLT interception deferred to a later session.
// For now, M/N/K come from geometry inference in gemm_shape_hash().

// ── Runtime bridge to cipher_dispatch() in libcipher_rt.so ──────────────────
// Resolved lazily via dlsym — no link-time dependency.

// Mirror of CipherKernelDesc from cipher_intercept.h
struct HookKernelDesc {
    void*    fn;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t shared_bytes;
    void*    stream;
    void**   params;
    void**   extra;
    uint8_t  op_class;
    uint8_t  confidence;
    uint64_t intercept_ns;
};

// Mirror of CipherDispatchResult
enum { HOOK_PASS_THROUGH = 0, HOOK_SUBSTITUTED = 1, HOOK_DEFERRED = 2 };

typedef int (*cipher_dispatch_fn)(HookKernelDesc*);
static cipher_dispatch_fn g_cipher_dispatch = nullptr;
static std::atomic<int>   g_dispatch_probe{0};  // 0=untried, 1=found, -1=absent

static const char* op_class_name(uint8_t c) {
    switch (c) {
        case 0: return "GEMM";
        case 1: return "ATTENTION";
        case 2: return "CONVOLUTION";
        case 3: return "ELEMENTWISE";
        case 4: return "REDUCTION";
        case 5: return "MEMCPY_TRANSPOSE";
        case 6: return "ITERATIVE_CUSTOM";
        default: return "UNCLASSIFIED";
    }
}

static void try_resolve_dispatch() {
    if (g_dispatch_probe.load(std::memory_order_acquire) != 0) return;
    // Look in the global symbol namespace (libcipher_rt.so must be loaded)
    void* sym = dlsym(RTLD_DEFAULT, "cipher_dispatch");
    if (sym) {
        g_cipher_dispatch = (cipher_dispatch_fn)sym;
        g_dispatch_probe.store(1, std::memory_order_release);
    } else {
        // Try explicit handle
        void* rt = dlopen("libcipher_rt.so", RTLD_NOW | RTLD_NOLOAD);
        if (rt) sym = dlsym(rt, "cipher_dispatch");
        if (sym) {
            g_cipher_dispatch = (cipher_dispatch_fn)sym;
            g_dispatch_probe.store(1, std::memory_order_release);
        } else {
            g_dispatch_probe.store(-1, std::memory_order_release);
        }
    }
}

// Call cipher_dispatch and print classification. Returns the dispatch result.
static int dispatch_and_log(CUfunction f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void** params, void** extra)
{
    if (g_dispatch_probe.load(std::memory_order_acquire) == 0)
        try_resolve_dispatch();
    if (!g_cipher_dispatch) return HOOK_PASS_THROUGH;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    HookKernelDesc desc = {};
    desc.fn          = (void*)f;
    desc.grid_x      = gx; desc.grid_y = gy; desc.grid_z = gz;
    desc.block_x     = bx; desc.block_y = by; desc.block_z = bz;
    desc.shared_bytes = shared;
    desc.stream      = (void*)stream;
    desc.params      = params;
    desc.extra       = extra;
    desc.op_class    = 0xFF;
    desc.confidence  = 0;
    desc.intercept_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;

    int result = g_cipher_dispatch(&desc);

    uint64_t seq = g_intercept_count.load(std::memory_order_relaxed);
    const char* decision = (result == HOOK_SUBSTITUTED) ? "PERMIT" :
                           (result == HOOK_DEFERRED)    ? "DEFER"  : "PASS";
    if (tls_shape_valid && desc.op_class == 0 /* GEMM */) {
        fprintf(stderr,
            "[CIPHER] kernel #%lu fn=%p class=%s conf=%u decision=%s shape=%dx%dx%d\n",
            seq, (void*)f, op_class_name(desc.op_class), desc.confidence, decision,
            tls_M, tls_N, tls_K);
    } else {
        fprintf(stderr,
            "[CIPHER] kernel #%lu fn=%p class=%s conf=%u decision=%s\n",
            seq, (void*)f, op_class_name(desc.op_class), desc.confidence, decision);
    }

    return result;
}

// ── Forward declarations ────────────────────────────────────────────────────
extern "C" {
    CUresult    cuInit(unsigned);
    CUresult    cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
    CUresult    cuLaunchKernel_ptsz(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
    CUresult    cuLaunchKernelEx(void*, CUfunction, void**, void**);
    CUresult    cuLaunchKernelEx_ptsz(void*, CUfunction, void**, void**);
    cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    cudaError_t cudaLaunchKernel_ptsz(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t*, const void*, void**);
    cudaError_t cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t*, const void*, void**);
    CUresult    cuGetProcAddress(const char*, void**, int, uint64_t, CUdriverProcAddressQueryResult*);
    CUresult    cuGetProcAddress_v2(const char*, void**, int, uint64_t, CUdriverProcAddressQueryResult*);
    void        cipher_repatch(void);
    uint64_t    cipher_intercept_count(void);
}

// ── dlsym wrapper (patched into cuBLASLt's GOT only) ────────────────────────
// This is NOT a global dlsym hook. We only replace dlsym's GOT entry inside
// libraries that use dlsym to resolve CUDA functions (e.g. cuBLASLt).

static void* cipher_dlsym_wrapper(void* handle, const char* symbol);

// ── Shim table ──────────────────────────────────────────────────────────────
struct SymPatch { const char* name; void* shim; void** real; };

static SymPatch g_patches[] = {
    { "cuLaunchKernel",            (void*)cuLaunchKernel,            (void**)&g_real_cuLaunch },
    { "cuLaunchKernel_ptsz",       (void*)cuLaunchKernel_ptsz,       (void**)&g_real_cuLaunch_ptsz },
    { "cuLaunchKernelEx",          (void*)cuLaunchKernelEx,          (void**)&g_real_cuLaunchEx },
    { "cuLaunchKernelEx_ptsz",     (void*)cuLaunchKernelEx_ptsz,     (void**)&g_real_cuLaunchEx_ptsz },
    { "cudaLaunchKernel",          (void*)cudaLaunchKernel,          (void**)&g_real_cudaLaunch },
    { "cudaLaunchKernel_ptsz",     (void*)cudaLaunchKernel_ptsz,     (void**)&g_real_cudaLaunch_ptsz },
    { "cudaLaunchKernelExC",       (void*)cudaLaunchKernelExC,       (void**)&g_real_cudaLaunchExC },
    { "cudaLaunchKernelExC_ptsz",  (void*)cudaLaunchKernelExC_ptsz,  (void**)&g_real_cudaLaunchExC_ptsz },
    { "cuGetProcAddress",          (void*)cuGetProcAddress,          (void**)&g_real_cuGetProc },
    { "cuGetProcAddress_v2",       (void*)cuGetProcAddress_v2,       (void**)&g_real_cuGetProc_v2 },
    { "cublasLtMatmul",            (void*)cipher_cublasLtMatmul_impl, (void**)&g_real_cublasLtMatmul },
    { "cublasGemmEx",              (void*)cipher_cublasGemmEx_impl,  (void**)&g_real_cublasGemmEx },
};
static constexpr int N_PATCHES = sizeof(g_patches) / sizeof(g_patches[0]);

static void* get_shim(const char* s) {
    if (!s || s[0] != 'c') return nullptr;
    for (int i = 0; i < N_PATCHES; i++)
        if (strcmp(s, g_patches[i].name) == 0) return g_patches[i].shim;
    return nullptr;
}

// ── Resolve real pointers ───────────────────────────────────────────────────

static void resolve_real() {
    if (g_resolved.load(std::memory_order_acquire)) return;

    void* lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!lc) lc = dlopen("libcuda.so.1", RTLD_NOW);
    void* lr = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
    if (!lr) lr = dlopen("libcudart.so.12", RTLD_NOW);

    if (lc) {
        if (!g_real_cuLaunch)        g_real_cuLaunch        = (cuLaunchKernel_fn)   dlsym(lc, "cuLaunchKernel");
        if (!g_real_cuLaunch_ptsz)   g_real_cuLaunch_ptsz   = (cuLaunchKernel_fn)   dlsym(lc, "cuLaunchKernel_ptsz");
        if (!g_real_cuLaunchEx)      g_real_cuLaunchEx      = (cuLaunchKernelEx_fn)  dlsym(lc, "cuLaunchKernelEx");
        if (!g_real_cuLaunchEx_ptsz) g_real_cuLaunchEx_ptsz = (cuLaunchKernelEx_fn)  dlsym(lc, "cuLaunchKernelEx_ptsz");
        if (!g_real_cuGetProc)       g_real_cuGetProc       = (cuGetProcAddress_fn)  dlsym(lc, "cuGetProcAddress");
        if (!g_real_cuGetProc_v2)    g_real_cuGetProc_v2    = (cuGetProcAddress_fn)  dlsym(lc, "cuGetProcAddress_v2");
    }
    if (lr) {
        if (!g_real_cudaLaunch)         g_real_cudaLaunch         = (cudaLaunchKernel_fn)    dlsym(lr, "cudaLaunchKernel");
        if (!g_real_cudaLaunch_ptsz)    g_real_cudaLaunch_ptsz    = (cudaLaunchKernel_fn)    dlsym(lr, "cudaLaunchKernel_ptsz");
        if (!g_real_cudaLaunchExC)      g_real_cudaLaunchExC      = (cudaLaunchKernelExC_fn) dlsym(lr, "cudaLaunchKernelExC");
        if (!g_real_cudaLaunchExC_ptsz) g_real_cudaLaunchExC_ptsz = (cudaLaunchKernelExC_fn) dlsym(lr, "cudaLaunchKernelExC_ptsz");
    }

    g_resolved.store(true, std::memory_order_release);

}

// ── GOT patching ────────────────────────────────────────────────────────────

static bool patch_slot(void** slot, void* shim, const char* name, const char* dso) {
    void* orig = *slot;
    if (orig == shim) return false;   // already ours
    if (orig == nullptr) return false; // not yet resolved

    // Save original (first writer wins)
    for (int i = 0; i < N_PATCHES; i++) {
        if (strcmp(name, g_patches[i].name) == 0) {
            if (*g_patches[i].real == nullptr)
                *g_patches[i].real = orig;
            break;
        }
    }

    uintptr_t page = (uintptr_t)slot & ~(g_page_size - 1);
    mprotect((void*)page, g_page_size * 2, PROT_READ | PROT_WRITE);
    *slot = shim;
    // Leave writable — dynamic linker needs write access during cleanup
    // (alternative: track pages and restore on teardown)

    g_got_patches.fetch_add(1, std::memory_order_relaxed);
    fprintf(stderr, "[CIPHER HOOK]   patched %-28s in %s (was %p)\n", name, dso, orig);
    return true;
}

static void patch_dso(const char* dso, ElfW(Addr) base, const ElfW(Dyn)* dyn) {
    const ElfW(Rela)* jmprel = nullptr;
    size_t jmprel_sz = 0;
    const ElfW(Sym)* symtab = nullptr;
    const char* strtab = nullptr;
    const ElfW(Rela)* rela = nullptr;
    size_t rela_sz = 0;

    for (const ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; d++) {
        switch (d->d_tag) {
            case DT_JMPREL:   jmprel    = (const ElfW(Rela)*)d->d_un.d_ptr; break;
            case DT_PLTRELSZ: jmprel_sz = d->d_un.d_val; break;
            case DT_SYMTAB:   symtab    = (const ElfW(Sym)*)d->d_un.d_ptr;  break;
            case DT_STRTAB:   strtab    = (const char*)d->d_un.d_ptr;       break;
            case DT_RELA:     rela      = (const ElfW(Rela)*)d->d_un.d_ptr; break;
            case DT_RELASZ:   rela_sz   = d->d_un.d_val; break;
        }
    }
    if (!symtab || !strtab) return;


    auto walk = [&](const ElfW(Rela)* tbl, size_t sz, unsigned req_type) {
        if (!tbl || !sz) return;
        size_t n = sz / sizeof(ElfW(Rela));
        for (size_t i = 0; i < n; i++) {
            unsigned rtype = ELF64_R_TYPE(tbl[i].r_info);
            if (req_type && rtype != req_type && rtype != R_X86_64_JUMP_SLOT) continue;
            unsigned si = ELF64_R_SYM(tbl[i].r_info);
            const char* nm = strtab + symtab[si].st_name;
            void* shim = get_shim(nm);
            if (shim) {
                void** slot = (void**)(base + tbl[i].r_offset);
                patch_slot(slot, shim, nm, dso);
            }
            if (false) {  // Debug: GOT walk trace (disabled)
            }
        }
    };

    walk(jmprel, jmprel_sz, R_X86_64_JUMP_SLOT);
    walk(rela, rela_sz, R_X86_64_GLOB_DAT);
}

// ── Targeted dlsym GOT patch for cuBLASLt ───────────────────────────────────

// g_real_dlsym defined above, near TLS storage

static void* cipher_dlsym_wrapper(void* handle, const char* symbol) {
    // Call the REAL glibc dlsym (saved before patching)
    void* result = g_real_dlsym(handle, symbol);
    if (result && symbol) {
        void* shim = get_shim(symbol);
        if (shim) {
            resolve_real();
            return shim;
        }
    }
    return result;
}

// Patch dlsym GOT entry in a DSO (for libraries that use dlsym to resolve CUDA)
static void patch_dlsym_got(const char* dso, ElfW(Addr) base, const ElfW(Dyn)* dyn) {
    const ElfW(Rela)* jmprel = nullptr;
    size_t jmprel_sz = 0;
    const ElfW(Sym)* symtab = nullptr;
    const char* strtab = nullptr;

    for (const ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; d++) {
        switch (d->d_tag) {
            case DT_JMPREL:   jmprel    = (const ElfW(Rela)*)d->d_un.d_ptr; break;
            case DT_PLTRELSZ: jmprel_sz = d->d_un.d_val; break;
            case DT_SYMTAB:   symtab    = (const ElfW(Sym)*)d->d_un.d_ptr;  break;
            case DT_STRTAB:   strtab    = (const char*)d->d_un.d_ptr;       break;
        }
    }
    if (!jmprel || !jmprel_sz || !symtab || !strtab) return;

    size_t n = jmprel_sz / sizeof(ElfW(Rela));
    for (size_t i = 0; i < n; i++) {
        unsigned si = ELF64_R_SYM(jmprel[i].r_info);
        const char* nm = strtab + symtab[si].st_name;
        if (strcmp(nm, "dlsym") == 0) {
            void** slot = (void**)(base + jmprel[i].r_offset);
            void* orig = *slot;
            if (orig == (void*)cipher_dlsym_wrapper) return; // already patched

            // Save real dlsym (first one wins)
            if (!g_real_dlsym) g_real_dlsym = (void*(*)(void*, const char*))orig;

            uintptr_t page = (uintptr_t)slot & ~(g_page_size - 1);
            mprotect((void*)page, g_page_size * 2, PROT_READ | PROT_WRITE);
            *slot = (void*)cipher_dlsym_wrapper;

            g_got_patches.fetch_add(1, std::memory_order_relaxed);
            fprintf(stderr, "[CIPHER HOOK]   patched dlsym GOT in %s (was %p)\n", dso, orig);
            return;
        }
    }
}

static int phdr_cb(struct dl_phdr_info* info, size_t, void*) {
    for (int i = 0; i < info->dlpi_phnum; i++) {
        if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
            const ElfW(Dyn)* dyn = (const ElfW(Dyn)*)(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
            const char* name = info->dlpi_name && info->dlpi_name[0] ? info->dlpi_name : "[main]";

            // Standard GOT patch for PLT relocations (cuLaunchKernel etc.)
            patch_dso(name, info->dlpi_addr, dyn);

            // Targeted dlsym GOT patch for libraries that use dlsym to resolve
            // CUDA functions (cuBLASLt, cuSPARSELt, cuDNN, etc.)
            if (strstr(name, "cublas") || strstr(name, "cusparse") ||
                strstr(name, "cudnn") || strstr(name, "cufft") ||
                strstr(name, "cusolver") || strstr(name, "curand") ||
                strstr(name, "libtorch_cuda")) {
                patch_dlsym_got(name, info->dlpi_addr, dyn);
            }
            break;
        }
    }
    return 0;
}

extern "C" __attribute__((visibility("default")))
void cipher_repatch(void) {
    resolve_real();
    dl_iterate_phdr(phdr_cb, nullptr);
    fprintf(stderr, "[CIPHER HOOK] repatch done — total GOT patches: %lu\n",
            g_got_patches.load());
}

static void auto_repatch_once();

// ── cudaMalloc shim — triggers auto-repatch on first GPU allocation ─────────
// cudaMalloc is imported via PLT by libc10_cuda and always called before
// any kernel launch (for tensor allocation). We hook it to trigger GOT
// patching at the right moment.

typedef cudaError_t (*cudaMalloc_fn)(void**, size_t);
static cudaMalloc_fn g_real_cudaMalloc = nullptr;

extern "C" __attribute__((visibility("default")))
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!g_real_cudaMalloc) {
        void* lr = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
        if (!lr) lr = dlopen("libcudart.so.12", RTLD_NOW);
        if (lr) g_real_cudaMalloc = (cudaMalloc_fn)dlsym(lr, "cudaMalloc");
        if (!g_real_cudaMalloc) return (cudaError_t)999;
    }
    cudaError_t r = g_real_cudaMalloc(devPtr, size);
    // After first successful malloc, CUDA is fully initialized — patch GOTs
    if (r == 0) {
        resolve_real();
        auto_repatch_once();
    }
    return r;
}

// ── Shim implementations ────────────────────────────────────────────────────

static void auto_repatch_once() {
    if (!g_auto_patched.exchange(true, std::memory_order_acq_rel)) {
        dl_iterate_phdr(phdr_cb, nullptr);
    }
}

#define ENSURE_REAL(ptr) do { if (__builtin_expect(!(ptr), 0)) { resolve_real(); auto_repatch_once(); if (!(ptr)) return CUDA_ERROR_NOT_FOUND; } } while(0)
#define ENSURE_RT(ptr)   do { if (__builtin_expect(!(ptr), 0)) { resolve_real(); auto_repatch_once(); if (!(ptr)) return (cudaError_t)999; } } while(0)

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernel(CUfunction f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void** params, void** extra)
{
    ENSURE_REAL(g_real_cuLaunch);
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    dispatch_and_log(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
    return g_real_cuLaunch(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernel_ptsz(CUfunction f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void** params, void** extra)
{
    ENSURE_REAL(g_real_cuLaunch_ptsz);
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    return g_real_cuLaunch_ptsz(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

// ABI-compatible mirror of CUlaunchConfig_st (CUDA 12+)
struct CUlaunchConfigMirror {
    unsigned gridDimX, gridDimY, gridDimZ;
    unsigned blockDimX, blockDimY, blockDimZ;
    unsigned sharedMemBytes;
    void*    hStream;       // CUstream
    void*    attrs;
    unsigned numAttrs;
};

// ── Extract A,B,C device pointers from cuLaunchKernelEx params ───────────────

typedef CUresult (*cuFuncGetName_fn)(const char**, CUfunction);
typedef CUresult (*cuPointerGetAttribute_fn)(void*, int, uintptr_t);

static cuFuncGetName_fn         g_cuFuncGetName = nullptr;
static cuPointerGetAttribute_fn g_cuPtrGetAttr  = nullptr;

static void resolve_introspection() {
    if (g_cuFuncGetName) return;
    void* lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!lc) return;
    auto rsym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
    g_cuFuncGetName = (cuFuncGetName_fn)rsym(lc, "cuFuncGetName");
    g_cuPtrGetAttr  = (cuPointerGetAttribute_fn)rsym(lc, "cuPointerGetAttribute");
}

// A/B/C extraction from cuLaunchKernelEx params is not feasible for cuBLAS
// (kernel takes one device-struct arg; B/C are in constant memory).
// Use cipher_set_gemm_ptrs() from Python or cublasLtMatmul shim instead.

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernelEx(void* config, CUfunction f, void** params, void** extra)
{
    ENSURE_REAL(g_real_cuLaunchEx);
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    // Skip dispatch if we're inside apply_recipe (recursion guard)
    if (tls_in_recipe)
        return g_real_cuLaunchEx(config, f, params, extra);
    // Store launch args in TLS so apply_recipe can re-launch
    tls_launch_config = config;
    tls_launch_fn     = (void*)f;
    tls_launch_params = params;
    tls_launch_extra  = extra;
    tls_launch_valid  = 1;
    // A/B/C set via cipher_set_gemm_ptrs() or cublasLtMatmul shim (TLS)
    auto* cfg = (CUlaunchConfigMirror*)config;
    int result = dispatch_and_log(f,
        cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
        cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
        cfg->sharedMemBytes, (CUstream)cfg->hStream, params, extra);
    tls_launch_valid = 0;
    if (result == HOOK_SUBSTITUTED)
        return CUDA_SUCCESS; // recipe already computed the result
    return g_real_cuLaunchEx(config, f, params, extra);
}

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernelEx_ptsz(void* config, CUfunction f, void** params, void** extra)
{
    ENSURE_REAL(g_real_cuLaunchEx_ptsz);
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    return g_real_cuLaunchEx_ptsz(config, f, params, extra);
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaLaunchKernel(const void* func, dim3 gd, dim3 bd,
    void** args, size_t smem, cudaStream_t stream)
{
    ENSURE_RT(g_real_cudaLaunch);
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    return g_real_cudaLaunch(func, gd, bd, args, smem, stream);
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaLaunchKernel_ptsz(const void* func, dim3 gd, dim3 bd,
    void** args, size_t smem, cudaStream_t stream)
{
    ENSURE_RT(g_real_cudaLaunch_ptsz);
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    return g_real_cudaLaunch_ptsz(func, gd, bd, args, smem, stream);
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args)
{
    ENSURE_RT(g_real_cudaLaunchExC);
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    return g_real_cudaLaunchExC(config, func, args);
}

extern "C" __attribute__((visibility("default")))
cudaError_t cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t* config, const void* func, void** args)
{
    ENSURE_RT(g_real_cudaLaunchExC_ptsz);
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    return g_real_cudaLaunchExC_ptsz(config, func, args);
}

extern "C" __attribute__((visibility("default")))
CUresult cuGetProcAddress(const char* symbol, void** pfn, int ver,
    uint64_t flags, CUdriverProcAddressQueryResult* status)
{
    ENSURE_REAL(g_real_cuGetProc);
    g_proc_addr_count.fetch_add(1, std::memory_order_relaxed);
    CUresult r = g_real_cuGetProc(symbol, pfn, ver, flags, status);
    if (r == CUDA_SUCCESS && pfn && symbol) {
        void* s = get_shim(symbol);
        if (s) *pfn = s;
    }
    return r;
}

extern "C" __attribute__((visibility("default")))
CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int ver,
    uint64_t flags, CUdriverProcAddressQueryResult* status)
{
    cuGetProcAddress_fn real = g_real_cuGetProc_v2 ? g_real_cuGetProc_v2 : g_real_cuGetProc;
    if (!real) { resolve_real(); real = g_real_cuGetProc_v2 ? g_real_cuGetProc_v2 : g_real_cuGetProc; }
    if (!real) return CUDA_ERROR_NOT_FOUND;
    g_proc_addr_count.fetch_add(1, std::memory_order_relaxed);
    CUresult r = real(symbol, pfn, ver, flags, status);
    if (r == CUDA_SUCCESS && pfn && symbol) {
        void* s = get_shim(symbol);
        if (s) *pfn = s;
    }
    return r;
}

extern "C" __attribute__((visibility("default")))
uint64_t cipher_intercept_count(void) {
    return g_intercept_count.load(std::memory_order_relaxed);
}

// ── Constructor / Destructor ────────────────────────────────────────────────

__attribute__((constructor(101)))
static void cipher_hook_init() {
    g_page_size = sysconf(_SC_PAGESIZE);
    fprintf(stderr, "[CIPHER HOOK] Loaded — PLT export + GOT-patch mode\n");
    // Attempt early patch (will likely find no CUDA libs yet)
    resolve_real();
    if (g_real_cuLaunch)
        dl_iterate_phdr(phdr_cb, nullptr);
}

__attribute__((destructor))
static void cipher_hook_fini() {
    uint64_t ic = g_intercept_count.load();
    uint64_t pc = g_proc_addr_count.load();
    uint64_t gp = g_got_patches.load();
    fprintf(stderr,
        "[CIPHER HOOK] Teardown. Intercepts: %lu | ProcAddr: %lu | GOT patches: %lu\n",
        ic, pc, gp);
    if (ic == 0) fprintf(stderr, "[CIPHER HOOK] WARNING: 0 intercepts fired!\n");
}
