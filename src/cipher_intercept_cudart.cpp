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
#include <math.h>
#include <atomic>
#include <x86intrin.h>

#include "cipher_persist.h"
#include "cipher_attn_koopman.h"

// Weak-link the attention FSM entry points — they live in libcipher_rt.so,
// not the hook DSO. When CIPHER_ATTN_KOOPMAN=0 (default) the FSM returns
// PASS_THROUGH for every call so this adds one branch on the hot path.
extern "C" __attribute__((weak)) CipherAttnAction cipher_attn_fsm_on_gemm(
    void*, int, int, int, int, int,
    const void*, const void*, int, int,
    const void*, int, int,
    const void*, void*, int, int,
    int, int, void*);
extern "C" __attribute__((weak)) CipherAttnAction cipher_attn_fsm_on_kernel(
    const void*, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned,
    void**, void**, void*, void*);
extern "C" __attribute__((weak)) bool cipher_attn_koopman_enabled(void);

// Shim-overhead instrumentation (Change 2 gate measurement).
// Tracks TSC cycles spent inside the shim, excluding the real CUDA launch.
// Exported accessors let the gate test compute per-launch CIPHER overhead
// independently of driver noise.
static std::atomic<uint64_t> g_shim_tsc_total{0};
static std::atomic<uint64_t> g_shim_tsc_calls{0};
extern "C" __attribute__((visibility("default")))
uint64_t cipher_shim_tsc_total(void) {
    return g_shim_tsc_total.load(std::memory_order_relaxed);
}
extern "C" __attribute__((visibility("default")))
uint64_t cipher_shim_tsc_calls(void) {
    return g_shim_tsc_calls.load(std::memory_order_relaxed);
}
extern "C" __attribute__((visibility("default")))
void cipher_shim_tsc_reset(void) {
    g_shim_tsc_total.store(0, std::memory_order_relaxed);
    g_shim_tsc_calls.store(0, std::memory_order_relaxed);
}

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

extern "C" __attribute__((visibility("default")))
void cipher_tls_get_gemm_types(int* Atype, int* Btype, int* Ctype) {
    *Atype = tls_Atype; *Btype = tls_Btype; *Ctype = tls_Ctype;
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
typedef int (*cublasSetStream_fn)(void*, void*);
typedef int (*cudaGetStream_fn)(void**);

static void*             g_cublas_handle   = nullptr;
static cublasSgemm_fn    g_real_sgemm      = nullptr;
static cublasSetStream_fn g_cublasSetStream = nullptr;

// __libc_dlsym bypasses our dlsym wrapper entirely — guaranteed real symbols.
extern "C" void* __libc_dlsym(void*, const char*);

static void resolve_cublas() {
    if (g_real_sgemm) return;
    void* h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcublas.so.12", RTLD_NOW);
    if (!h) return;
    // dlvsym bypasses our GOT-patched dlsym — gets the real versioned symbol
    g_real_sgemm      = (cublasSgemm_fn)    dlvsym(h, "cublasSgemm_v2", "libcublas.so.12");
    g_cublasSetStream = (cublasSetStream_fn) dlvsym(h, "cublasSetStream_v2", "libcublas.so.12");
    auto cCreate      = (cublasCreate_fn)    dlvsym(h, "cublasCreate_v2", "libcublas.so.12");
    if (!g_real_sgemm)
        g_real_sgemm  = (cublasSgemm_fn)    dlsym(h, "cublasSgemm_v2");
    if (!g_cublasSetStream)
        g_cublasSetStream = (cublasSetStream_fn) dlsym(h, "cublasSetStream_v2");
    if (!cCreate)
        cCreate       = (cublasCreate_fn)    dlsym(h, "cublasCreate_v2");
    if (cCreate) cCreate(&g_cublas_handle);
}

extern "C" __attribute__((visibility("default")))
int cipher_substitute_sgemm(int M, int N, int K,
                            const void* A, const void* B, void* C)
{
    resolve_cublas();
    if (!g_real_sgemm || !g_cublas_handle) return -1;

    if (g_cublasSetStream)
        g_cublasSetStream(g_cublas_handle, nullptr);

    // cublasSgemm_v2: Row-major torch C = A*B → col-major C^T = B^T * A^T
    static const float alpha = 1.0f, beta = 0.0f;
    if (!g_real_sgemm) return -5;

    tls_in_recipe = 1;
    int rc = g_real_sgemm(g_cublas_handle,
        0 /*CUBLAS_OP_N*/, 0 /*CUBLAS_OP_N*/,
        N, M, K,
        &alpha,
        (const float*)B, N,
        (const float*)A, K,
        &beta,
        (float*)C, N);
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

    // -- Change 3: Attention Koopman FSM hook (default OFF via env flag) ---
    // The FSM may SUPPRESS this GEMM (state enters SAW_QK) or may have just
    // replayed buffered calls and returned REVERT_AND_PASS (we run the real
    // GEMM normally). On PASS_THROUGH the shim continues unchanged. The FSM
    // never returns a result — it only decides whether we run the real call.
    //
    // We extract the stream bound to this cuBLAS handle via cublasGetStream
    // and pass it to the FSM so its cross-stream-abort rule has something
    // meaningful to compare. Resolved lazily via dlsym; cached across calls.
    if (cipher_attn_fsm_on_gemm) {
        static cublasSetStream_fn g_cublasGetStream = nullptr;
        static int                g_get_stream_resolved = 0;
        if (!g_get_stream_resolved) {
            void* h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
            if (!h) h = dlopen("libcublas.so.12", RTLD_NOW);
            if (h) {
                g_cublasGetStream = (cublasSetStream_fn)dlvsym(
                    h, "cublasGetStream_v2", "libcublas.so.12");
                if (!g_cublasGetStream)
                    g_cublasGetStream = (cublasSetStream_fn)dlsym(
                        h, "cublasGetStream_v2");
            }
            g_get_stream_resolved = 1;
        }
        void* handle_stream = nullptr;
        if (g_cublasGetStream) {
            g_cublasGetStream(handle, &handle_stream);
        }
        CipherAttnAction act = cipher_attn_fsm_on_gemm(
            (void*)handle, transa, transb, m, n, k,
            alpha, A, Atype, lda, B, Btype, ldb,
            beta, C, Ctype, ldc,
            computeType, algo, handle_stream);
        if (act == CIPHER_ATTN_SUPPRESS) {
            tls_shape_valid = 0;
            return (cublasStatus_t)0;   // CUBLAS_STATUS_SUCCESS; C is left
                                         // untouched per scratch-buffer model
        }
        // REVERT_AND_PASS or PASS_THROUGH: fall through and run the real GEMM
        (void)act;
    }

    // Driver-level Koopman substitute — fires in cublasGemmEx shim
    // Geometry-only: attempts substitute for any fp16 shape.
    // Falls through to real cuBLAS if Koopman kernel returns non-zero
    // (no calibrated matrices for this shape yet).
    {
        static int shape_log = 0;
        if (Ctype == 2 && shape_log < 20) {
            fprintf(stderr, "[CIPHER GEMM-EX] #%d m=%d n=%d k=%d\n",
                shape_log, m, n, k);
            shape_log++;
        }
    }
    if (Ctype == 2 /* fp16 */ && m > 0 && k > 0 && n > 0) {
        typedef int (*fp16_launch_shape_fn)(const void*, void*, int, int, int);
        static fp16_launch_shape_fn s_launch = nullptr;
        if (!s_launch)
            s_launch = (fp16_launch_shape_fn)dlsym(RTLD_DEFAULT,
                                             "cipher_koopman_fp16_launch_shape");
        // cublasGemmEx: C(m×n) = A(m×k) @ B(k×n). B=activation, C=output.
        // Koopman sees: M=n rows, input_dim=k, output_dim=m.
        if (s_launch && s_launch(B, C, n, k, m) == 0) {
            tls_shape_valid = 0;
            static uint64_t s_sub_count = 0;
            s_sub_count++;
            if (s_sub_count <= 5 || (s_sub_count % 500) == 0)
                fprintf(stderr, "[O(1)-driver] M=%d K=%d N=%d count=%llu\n",
                    m, k, n, (unsigned long long)s_sub_count);
            return (cublasStatus_t)0;  // CUBLAS_STATUS_SUCCESS
        }
    }

    uint64_t t0_gemm = 0;
    {
        struct timespec _ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &_ts);
        t0_gemm = (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;
    }

    cublasStatus_t ret = g_real_cublasGemmEx(handle, transa, transb,
        m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
        beta, C, Ctype, ldc, computeType, algo);

    // MFU telemetry — geometry only, O(1)
    if (ret == 0 && m > 0 && n > 0 && k > 0) {
        typedef void (*record_gemm_fn)(uint32_t, uint32_t, uint32_t, uint64_t);
        static record_gemm_fn s_record = nullptr;
        static int s_resolved = 0;
        if (!s_resolved) {
            s_record = (record_gemm_fn)dlsym(RTLD_DEFAULT, "cipher_telemetry_record_gemm");
            s_resolved = 1;
        }
        if (s_record) s_record((uint32_t)m, (uint32_t)n, (uint32_t)k, t0_gemm);
    }

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
        // dlvsym gets the real versioned symbol, bypassing our GOT patches
        if (!g_real_cublasLtMatmul)
            g_real_cublasLtMatmul = (cublasLtMatmul_fn)dlvsym(h, "cublasLtMatmul", "libcublasLt.so.12");
        if (!g_cublasLtLayoutGetAttr)
            g_cublasLtLayoutGetAttr = (cublasLtLayoutGetAttr_fn)dlvsym(h, "cublasLtMatrixLayoutGetAttribute", "libcublasLt.so.12");
        // Fallback to unversioned if dlvsym fails
        if (!g_real_cublasLtMatmul) {
            auto rsym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
            g_real_cublasLtMatmul = (cublasLtMatmul_fn)rsym(h, "cublasLtMatmul");
        }
        if (!g_cublasLtLayoutGetAttr) {
            auto rsym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
            g_cublasLtLayoutGetAttr = (cublasLtLayoutGetAttr_fn)rsym(h, "cublasLtMatrixLayoutGetAttribute");
        }
    }
}

// PLT-exported alias so LD_PRELOAD intercepts before libtorch resolves
extern "C" __attribute__((visibility("default"), alias("cipher_cublasLtMatmul_impl")))
cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    const void*, cublasLtMatrixLayout_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    void*, cublasLtMatrixLayout_t,
    const cublasLtMatmulAlgo_t, void*, size_t, cudaStream_t);

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
    // Recursion guard: skip if called from within a recipe/relaunch
    static thread_local int in_lt = 0;
    if (in_lt || tls_in_recipe) return g_real_cublasLtMatmul(lightHandle, computeDesc,
        alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
        algo, workspace, workspaceSizeInBytes, stream);

    in_lt = 1;
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
        // Query output dtype: CUBLASLT_MATRIX_LAYOUT_TYPE = 0
        uint32_t dtype_val = 0;
        g_cublasLtLayoutGetAttr(Ddesc, 0, &dtype_val, sizeof(dtype_val), &written);
        tls_Ctype = (int)dtype_val;  // 0=CUDA_R_32F, 2=CUDA_R_16F
        static int lt_log_count = 0;
        if (lt_log_count < 3) {
            fprintf(stderr, "[CIPHER HOOK] cublasLtMatmul intercepted: M=%d N=%d K=%d C=%p\n",
                    tls_M, tls_N, tls_K, D);
            lt_log_count++;
        }
    }

    // Driver-level Koopman substitute — no Python, no hooks
    {
        static int dtype_log = 0;
        if (dtype_log < 3) {
            fprintf(stderr, "[CIPHER LT-SUB] M=%d K=%d N=%d Ctype=%d A=%p D=%p\n",
                tls_M, tls_K, tls_N, tls_Ctype, A, D);
            dtype_log++;
        }
    }
    if (tls_Ctype == 2 /* fp16 */ && tls_M > 0 && tls_K > 0 && tls_N > 0) {
        typedef int (*fp16_launch_shape_fn)(const void*, void*, int, int, int);
        static fp16_launch_shape_fn s_launch = nullptr;
        if (!s_launch)
            s_launch = (fp16_launch_shape_fn)dlsym(RTLD_DEFAULT,
                                             "cipher_koopman_fp16_launch_shape");
        // cublasLtMatmul: D(M×N) = A(M×K) @ B(K×N).
        // Koopman sees: M=tls_M rows, input_dim=tls_K, output_dim=tls_N.
        if (s_launch && s_launch(tls_A, tls_C, tls_M, tls_K, tls_N) == 0) {
            in_lt = 0;
            tls_shape_valid = 0;
            static uint64_t s_sub_count = 0;
            s_sub_count++;
            if (s_sub_count <= 5 || (s_sub_count % 500) == 0)
                fprintf(stderr, "[O(1)-driver] M=%d K=%d N=%d count=%llu\n",
                    tls_M, tls_K, tls_N, (unsigned long long)s_sub_count);
            return (cublasStatus_t)0;  // CUBLAS_STATUS_SUCCESS
        }
    }

    cublasStatus_t ret = g_real_cublasLtMatmul(lightHandle, computeDesc,
        alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
        algo, workspace, workspaceSizeInBytes, stream);

    in_lt = 0;
    // Block-level calibration: disabled in cublasLtMatmul path.
    // Collection handled by cipher_wrapper.py (dtype-safe).
    // cublasLtMatmul carries fp16/bf16 tensors that cannot be read as fp32.

    tls_shape_valid = 0;
    return ret;
}

// cublasLtMatmul versioned PLT interception deferred to a later session.
// For now, M/N/K come from geometry inference in gemm_shape_hash().

// ── ncclAllReduce shim — intercepts NCCL collectives for timing + overlap ───
// Geometry only: sees count, datatype, message size. No gradient semantics.

typedef int ncclResult_t_hook;
typedef void* ncclComm_t_hook;

typedef ncclResult_t_hook (*ncclAllReduce_fn)(
    const void*, void*, size_t, int, int, ncclComm_t_hook, void*);

static ncclAllReduce_fn g_real_ncclAllReduce = nullptr;

static void ensure_nccl() {
    if (g_real_ncclAllReduce) return;
    void* h = dlopen("libnccl.so.2", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libnccl.so.2", RTLD_NOW);
    if (h) {
        auto rsym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
        g_real_ncclAllReduce = (ncclAllReduce_fn)dlvsym(h, "ncclAllReduce", "libnccl.so.2");
        if (!g_real_ncclAllReduce)
            g_real_ncclAllReduce = (ncclAllReduce_fn)rsym(h, "ncclAllReduce");
    }
}

extern "C" __attribute__((visibility("default")))
ncclResult_t_hook cipher_ncclAllReduce_impl(
    const void* sendbuff, void* recvbuff, size_t count,
    int datatype, int op, ncclComm_t_hook comm, void* stream)
{
    ensure_nccl();
    if (!g_real_ncclAllReduce) return 1;  // ncclInternalError

    // Estimate bytes from count + datatype (geometry only, no gradient knowledge)
    // ncclDataType: 0=int8, 1=uint8, 2=int32, 3=uint32, 4=int64, 5=uint64,
    //               6=fp16, 7=fp32, 8=fp64, 9=bf16
    static const size_t dtype_size[] = {1,1,4,4,8,8,2,4,8,2};
    size_t elem_size = (datatype >= 0 && datatype <= 9) ? dtype_size[datatype] : 2;
    size_t bytes = count * elem_size;

    static int nccl_log = 0;
    if (nccl_log < 10) {
        fprintf(stderr, "[CIPHER HOOK] ncclAllReduce: count=%zu dtype=%d bytes=%zu\n",
                count, datatype, bytes);
        nccl_log++;
    }

    // Change 4 Part A — CfC policy DECIDE before the real collective.
    // Resolve both RT bridges lazily via dlsym; cached in function-static
    // variables so the lookup happens once per process.
    typedef int  (*nccl_decide_fn)(uint64_t, uint32_t);
    typedef void (*nccl_feedback_fn)(uint64_t, uint64_t, int);
    typedef void (*nccl_record_fn)(uint64_t, uint64_t);
    static nccl_decide_fn   s_decide   = nullptr;
    static nccl_feedback_fn s_feedback = nullptr;
    static nccl_record_fn   s_record   = nullptr;
    static int              s_resolved = 0;
    if (!s_resolved) {
        s_decide   = (nccl_decide_fn)  dlsym(RTLD_DEFAULT, "cipher_nccl_record_decide");
        s_feedback = (nccl_feedback_fn)dlsym(RTLD_DEFAULT, "cipher_nccl_record_feedback");
        s_record   = (nccl_record_fn)  dlsym(RTLD_DEFAULT, "cipher_nccl_record");
        s_resolved = 1;
    }

    // num_ranks is not exposed by the ncclAllReduce signature; a proper
    // query would call ncclCommCount(comm, &ranks), but we avoid taking
    // a hard dep on libnccl inside the hook DSO. 0 means "unknown" and
    // the CfC uses its default of 8.
    int chosen_algo = 4 /* NCCL_ALGO_AUTO */;
    if (s_decide) {
        chosen_algo = s_decide((uint64_t)bytes, 0u /* num_ranks unknown */);
    }

    // Time the real AllReduce call
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    ncclResult_t_hook ret = g_real_ncclAllReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream);

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    uint64_t dur_ns = (t1.tv_sec - t0.tv_sec) * 1000000000ULL
                    + (t1.tv_nsec - t0.tv_nsec);

    // Change 4 Part A — FEEDBACK: loop closed.
    if (s_feedback) s_feedback((uint64_t)bytes, dur_ns, chosen_algo);

    // Legacy liquid-state EMA update (kept for telemetry continuity).
    if (s_record) s_record(dur_ns, bytes);

    return ret;
}

// PLT-exported alias so LD_PRELOAD intercepts before PyTorch resolves
extern "C" __attribute__((visibility("default"), alias("cipher_ncclAllReduce_impl")))
ncclResult_t_hook ncclAllReduce(
    const void*, void*, size_t, int, int, ncclComm_t_hook, void*);

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

// Weak extern: implemented in cipher_intercept.cpp (libcipher_rt.so)
// Writes ring entry for Stage 1 consumption. No kernel launch, no passthrough.
// Geometry args (gx..bz, M, K, N) plumbed so Stage 1 SENSE/SHIELD can perform
// real prefill/decode discrimination instead of class-count heuristics.
// (M, K, N) are 0 for non-GEMM kernels; grid_*/block_* are always populated.
extern "C" __attribute__((weak))
void cipher_f1_ring_write(uint8_t op_class, float confidence, uint64_t seq,
                          uint32_t gx, uint32_t gy, uint32_t gz,
                          uint32_t bx, uint32_t by, uint32_t bz,
                          uint32_t M, uint32_t K, uint32_t N,
                          uint64_t func_ptr_hash, uint64_t timestamp_ns);

static void maybe_collect_sample(uint8_t op_class); // forward decl

// Call cipher_dispatch and print classification. Returns the dispatch result.
static int dispatch_and_log(CUfunction f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared, CUstream stream, void** params, void** extra)
{
    // No early exit by grid size — let CLASSIFY decide.
    // Elementwise ops (SiLU, GeLU, RMSNorm) have small grids but are substitutable.

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

    // Verbose per-kernel logging — gated to first 30 + every 5000th to avoid
    // ~2us fprintf overhead on every GEMM dispatch (was the primary overhead source)
    if (seq <= 30 || (seq % 5000) == 0) {
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
    }

    // Collect EDMD sample after dispatch (tensor still alive)
    maybe_collect_sample(desc.op_class);

    // Feed F1 ring for Stage 1 (REMEMBER/VALIDATE/AUDIT/SPECULATE)
    if (cipher_f1_ring_write) {
        uint32_t Mx = tls_shape_valid ? (uint32_t)tls_M : 0u;
        uint32_t Kx = tls_shape_valid ? (uint32_t)tls_K : 0u;
        uint32_t Nx = tls_shape_valid ? (uint32_t)tls_N : 0u;
        cipher_f1_ring_write(
            desc.op_class, (float)desc.confidence / 100.0f, seq,
            gx, gy, gz, bx, by, bz,
            Mx, Kx, Nx,
            (uint64_t)(uintptr_t)f, desc.intercept_ns);
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
    { "ncclAllReduce",             (void*)cipher_ncclAllReduce_impl, (void**)&g_real_ncclAllReduce },
};
static constexpr int N_PATCHES = sizeof(g_patches) / sizeof(g_patches[0]);

static void* get_shim(const char* s) {
    if (!s || (s[0] != 'c' && s[0] != 'n')) return nullptr;
    // Never redirect these — CIPHER calls them for substitution / internal use
    if (strstr(s, "cublasSgemm") || strstr(s, "cublasCreate") ||
        strstr(s, "cublasSetStream"))
        return nullptr;
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

    // Persistent-mode fast path: bypass classify/dispatch/ring on stable
    // repeating sequences. Only valid when we are not inside a GEMM intercept
    // (tls_shape_valid==0), because GEMM has its own handling path.
    uint64_t persist_fp = 0;
    bool persist_fast = false;
    if (!tls_shape_valid) {
        persist_fp = cipher_persist_fingerprint(
            (const void*)f, gx, gy, gz, bx, by, bz, shared);
        persist_fast = cipher_persist_try_fast_path(persist_fp);
        if (persist_fast) {
            return g_real_cuLaunch(f, gx, gy, gz, bx, by, bz, shared,
                                    stream, params, extra);
        }
    }

    // Skip dispatch for GEMM kernels (already handled by cublasGemmEx shim).
    // Only dispatch non-GEMM kernels for classification and Chebyshev opportunity logging.
    if (!tls_shape_valid) {
        int dispatch_result = dispatch_and_log(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
        if (dispatch_result == HOOK_SUBSTITUTED)
            return CUDA_SUCCESS;
    }

    CUresult ret = g_real_cuLaunch(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);

    // Feed persistent-mode detector post-launch (normal path only)
    if (!tls_shape_valid && ret == CUDA_SUCCESS) {
        cipher_persist_observe(persist_fp);
    }

    // Block-level calibration: disabled in cuLaunchKernel path.
    // Collection handled by cipher_wrapper.py (dtype-safe).
    if (0 && tls_shape_valid && tls_M == 4096 && tls_K == 4096 && tls_A && tls_C && ret == 0) {
        typedef bool (*block_collect_fn)(const float*, const float*, uint32_t);
        static block_collect_fn fn = nullptr;
        static int resolved = 0;
        if (!resolved) {
            fn = (block_collect_fn)dlsym(RTLD_DEFAULT, "cipher_block_sub_collect");
            resolved = 1;
        }
        if (fn) {
            // Sync to ensure output is ready (calibration only, not on hot path)
            typedef int (*sync_fn)(void*);
            static sync_fn cu_sync = (sync_fn)dlsym(RTLD_DEFAULT, "cudaStreamSynchronize");
            if (cu_sync) cu_sync(stream);
            fn((const float*)tls_A, (const float*)tls_C, 4096);
        }
    }

    return ret;
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

// ── EDMD sample ring — tensor samples for Koopman derivation ────────────────
// After each GEMM kernel, sample 64 strided floats from C (output tensor)
// into a host ring buffer. ADAPT thread reads this for EDMD snapshots.

#define CIPHER_SAMPLE_DIM    1   // 1D: normalized Frobenius norm
#define CIPHER_SAMPLE_RING  512

struct CipherSampleEntry {
    float x[CIPHER_SAMPLE_DIM];  // input norm (||A||_F / sqrt(M*K))
    float y[CIPHER_SAMPLE_DIM];  // output norm (||C||_F / sqrt(M*N))
    uint8_t op_class;
    uint8_t valid;
};

static CipherSampleEntry g_sample_ring[CIPHER_SAMPLE_RING];
static std::atomic<uint64_t> g_sample_wseq{0};

// Export for ADAPT thread in libcipher_rt.so
extern "C" __attribute__((visibility("default")))
int cipher_read_sample(uint64_t seq, float* x_out, float* y_out, uint8_t* op_class) {
    if (seq >= g_sample_wseq.load(std::memory_order_acquire)) return 0;
    auto& e = g_sample_ring[seq % CIPHER_SAMPLE_RING];
    if (!e.valid) return 0;
    memcpy(x_out, e.x, CIPHER_SAMPLE_DIM * sizeof(float));
    memcpy(y_out, e.y, CIPHER_SAMPLE_DIM * sizeof(float));
    *op_class = e.op_class;
    return 1;
}

extern "C" __attribute__((visibility("default")))
uint64_t cipher_sample_wseq(void) {
    return g_sample_wseq.load(std::memory_order_acquire);
}

// Compute normalized Frobenius norm from a 256-element sample of the tensor.
// Returns ||sample||_F / sqrt(n_elems) — a scalar capturing output magnitude.
static float compute_sample_norm(const void* dev_ptr, int total_elems) {
    if (!dev_ptr || total_elems < 1) return 0.0f;
    typedef int (*cudaMemcpy_fn)(void*, const void*, size_t, int);
    static cudaMemcpy_fn s_memcpy = nullptr;
    if (!s_memcpy) {
        auto rsym = g_real_dlsym ? g_real_dlsym : (void*(*)(void*,const char*))dlsym;
        void* lr = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
        if (lr) s_memcpy = (cudaMemcpy_fn)rsym(lr, "cudaMemcpy");
    }
    if (!s_memcpy) return 0.0f;
    float raw[256];
    int ncopy = (total_elems < 256) ? total_elems : 256;
    if (s_memcpy(raw, dev_ptr, ncopy * sizeof(float), 2) != 0) return 0.0f;
    double sq = 0.0;
    for (int i = 0; i < ncopy; i++) sq += (double)raw[i] * raw[i];
    return (float)(sqrt(sq / ncopy));
}

// Called after kernel launch to collect 1D norm samples for EDMD
static void maybe_collect_sample(uint8_t op_class) {
    static uint64_t s_launch_count = 0;
    s_launch_count++;
    if (s_launch_count > 500) return;
    if (!tls_shape_valid || !tls_A || !tls_C) return;

    float norm_a = compute_sample_norm(tls_A, tls_M * tls_K);
    float norm_c = compute_sample_norm(tls_C, tls_M * tls_N);
    if (norm_a < 1e-10f && norm_c < 1e-10f) return;

    uint64_t idx = g_sample_wseq.load(std::memory_order_relaxed);
    auto& e = g_sample_ring[idx % CIPHER_SAMPLE_RING];
    e.x[0] = norm_a;
    e.y[0] = norm_c;
    e.op_class = op_class;
    e.valid = 1;
    g_sample_wseq.fetch_add(1, std::memory_order_release);
}

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernelEx(void* config, CUfunction f, void** params, void** extra)
{
    ENSURE_REAL(g_real_cuLaunchEx);
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);
    // Skip dispatch if we're inside apply_recipe (recursion guard)
    if (tls_in_recipe)
        return g_real_cuLaunchEx(config, f, params, extra);

    auto* cfg = (CUlaunchConfigMirror*)config;

    // -- Change 3: Attention Koopman FSM observer (default OFF) ------------
    // Only call when the FSM is enabled AND we're not inside a cublas GEMM
    // intercept (the QK/atV GEMMs are handled by cipher_attn_fsm_on_gemm).
    if (cipher_attn_fsm_on_kernel && !tls_shape_valid && !tls_in_recipe
        && cipher_attn_koopman_enabled
        && cipher_attn_koopman_enabled()) {
        CipherAttnAction act = cipher_attn_fsm_on_kernel(
            (const void*)f,
            cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
            cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
            cfg->sharedMemBytes,
            params, extra, config, (void*)cfg->hStream);
        if (act == CIPHER_ATTN_SUPPRESS) {
            return (CUresult)0;
        }
        // REVERT_AND_PASS: FSM already replayed saved calls on this stream;
        // we still need to run the current kernel normally, so fall through.
    }

    // Persistent-mode fast path: skip classify/dispatch/ring on stable
    // repeating sequences. Must not run when the GEMM intercept set the
    // shape TLS, because the cuBLAS-internal launches are the backing
    // kernels for a GEMM we're planning to substitute.
    uint64_t persist_fp = 0;
    bool persist_fast = false;
    if (!tls_shape_valid) {
        persist_fp = cipher_persist_fingerprint(
            (const void*)f,
            cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
            cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
            cfg->sharedMemBytes);
        persist_fast = cipher_persist_try_fast_path(persist_fp);
        if (persist_fast) {
            return g_real_cuLaunchEx(config, f, params, extra);
        }
    }

    // Store launch args in TLS so apply_recipe can re-launch
    tls_launch_config = config;
    tls_launch_fn     = (void*)f;
    tls_launch_params = params;
    tls_launch_extra  = extra;
    tls_launch_valid  = 1;
    // A/B/C set via cipher_set_gemm_ptrs() or cublasLtMatmul shim (TLS)
    int result = dispatch_and_log(f,
        cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
        cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
        cfg->sharedMemBytes, (CUstream)cfg->hStream, params, extra);
    tls_launch_valid = 0;
    if (result == HOOK_SUBSTITUTED)
        return CUDA_SUCCESS; // recipe already computed the result
    CUresult ret = g_real_cuLaunchEx(config, f, params, extra);
    if (!tls_shape_valid && ret == CUDA_SUCCESS) {
        cipher_persist_observe(persist_fp);
    }
    return ret;
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
    // Shim-overhead bracket — excludes the real launch.
    uint64_t tsc_entry = __rdtsc();

    ENSURE_RT(g_real_cudaLaunch);
    auto_repatch_once();
    g_intercept_count.fetch_add(1, std::memory_order_relaxed);

    // Persistent-mode fast path — bypass classify/dispatch/ring on stable
    // repeating sequences. Only valid outside GEMM intercept windows and
    // outside recursive dispatch.
    uint64_t persist_fp = 0;
    bool persist_fast = false;
    if (!tls_in_recipe && !tls_shape_valid) {
        persist_fp = cipher_persist_fingerprint(
            func, gd.x, gd.y, gd.z, bd.x, bd.y, bd.z, (unsigned)smem);
        persist_fast = cipher_persist_try_fast_path(persist_fp);
        if (persist_fast) {
            uint64_t tsc_before_real = __rdtsc();
            cudaError_t ret = g_real_cudaLaunch(func, gd, bd, args, smem, stream);
            uint64_t tsc_after_real = __rdtsc();
            g_shim_tsc_total.fetch_add(
                (tsc_before_real - tsc_entry) + (__rdtsc() - tsc_after_real),
                std::memory_order_relaxed);
            g_shim_tsc_calls.fetch_add(1, std::memory_order_relaxed);
            return ret;
        }
    }

    // Dispatch non-GEMM kernels only. GEMM classification and billing is fully
    // handled by the cublasGemmEx shim — dispatching GEMM kernels again here
    // would be pure overhead (~2us classify + oracle per call).
    // Skip when tls_shape_valid=1 (GEMM shim already processed this call).
    static thread_local int in_dispatch = 0;
    if (!in_dispatch && !tls_in_recipe && !tls_shape_valid) {
        in_dispatch = 1;
        dispatch_and_log(
            (CUfunction)func, gd.x, gd.y, gd.z, bd.x, bd.y, bd.z,
            (unsigned)smem, (CUstream)stream, args, nullptr);
        in_dispatch = 0;
    }

    uint64_t tsc_before_real = __rdtsc();
    cudaError_t ret = g_real_cudaLaunch(func, gd, bd, args, smem, stream);
    uint64_t tsc_after_real = __rdtsc();

    if (!tls_in_recipe && !tls_shape_valid && ret == 0) {
        cipher_persist_observe(persist_fp);
    }
    g_shim_tsc_total.fetch_add(
        (tsc_before_real - tsc_entry) + (__rdtsc() - tsc_after_real),
        std::memory_order_relaxed);
    g_shim_tsc_calls.fetch_add(1, std::memory_order_relaxed);
    return ret;
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

    uint64_t persist_fp = 0;
    bool persist_fast = false;
    if (!tls_in_recipe && !tls_shape_valid && config) {
        persist_fp = cipher_persist_fingerprint(
            func,
            config->gridDim.x,  config->gridDim.y,  config->gridDim.z,
            config->blockDim.x, config->blockDim.y, config->blockDim.z,
            (unsigned)config->dynamicSmemBytes);
        persist_fast = cipher_persist_try_fast_path(persist_fp);
        if (persist_fast) {
            return g_real_cudaLaunchExC(config, func, args);
        }
    }

    cudaError_t ret = g_real_cudaLaunchExC(config, func, args);
    if (!tls_in_recipe && !tls_shape_valid && ret == 0) {
        cipher_persist_observe(persist_fp);
    }
    return ret;
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

    // Print billing report if oracle is available
    typedef void (*billing_fn)(const void*);
    typedef void* (*get_oracle_fn)(void);
    get_oracle_fn go = (get_oracle_fn)dlsym(RTLD_DEFAULT, "cipher_get_oracle");
    billing_fn br = (billing_fn)dlsym(RTLD_DEFAULT, "cipher_oracle_billing_report");
    if (go && br) {
        void* oracle = go();
        if (oracle) br(oracle);
    }
}
