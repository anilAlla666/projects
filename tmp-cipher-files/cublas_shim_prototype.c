/* D6.2 prototype: LD_PRELOAD-able cublasGemmEx + cublasLtMatmul shims.
 *
 * Hooks both APIs, logs the call (with M/N/K extraction for cublasGemmEx
 * and a generic log for cublasLtMatmul), then passes through to the real
 * libcublas. Validates that LD_PRELOAD interception works under modern
 * PyTorch + libcublas on this pod.
 *
 * Build:
 *   gcc -O2 -fPIC -shared -Wl,-soname,libcublas_shim_proto.so \
 *       -o /tmp/libcublas_shim_proto.so /tmp/cublas_shim_prototype.c -ldl
 *
 * Run:
 *   LD_PRELOAD=/tmp/libcublas_shim_proto.so python3 -c "
 *     import torch
 *     a = torch.randn(64, 64, device='cuda', dtype=torch.float16)
 *     b = torch.randn(64, 64, device='cuda', dtype=torch.float16)
 *     c = a @ b
 *     torch.cuda.synchronize()
 *   " 2>&1 | grep -i shim
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdatomic.h>

typedef int cublasStatus_t;
typedef void* cublasHandle_t;

typedef cublasStatus_t (*cublasGemmEx_fn)(
    cublasHandle_t handle, int transa, int transb,
    int m, int n, int k,
    const void* alpha, const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta, void* C, int Ctype, int ldc,
    int computeType, int algo);

typedef cublasStatus_t (*cublasLtMatmul_fn)(
    void*, void*, const void*, const void*, void*,
    const void*, void*, const void*, const void*, void*,
    void*, void*, const void*, void*, unsigned long, void*);

static cublasGemmEx_fn  g_real_gemmEx  = NULL;
static cublasLtMatmul_fn g_real_ltMatmul = NULL;

static atomic_ulong g_gemmEx_count   = 0;
static atomic_ulong g_ltMatmul_count = 0;
static atomic_ulong g_init_logged    = 0;

static void *resolve_in(const char *soname, const char *sym)
{
	void *h = dlopen(soname, RTLD_NOW | RTLD_NOLOAD);
	if (!h) h = dlopen(soname, RTLD_NOW);
	if (!h) return NULL;
	return dlsym(h, sym);
}

cublasStatus_t cipher_cublasGemmEx_impl(
    cublasHandle_t handle, int transa, int transb,
    int m, int n, int k,
    const void* alpha, const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta, void* C, int Ctype, int ldc,
    int computeType, int algo)
{
	unsigned long n_calls;
	if (!g_real_gemmEx) {
		g_real_gemmEx = (cublasGemmEx_fn)resolve_in("libcublas.so.12", "cublasGemmEx");
		if (!g_real_gemmEx) g_real_gemmEx = (cublasGemmEx_fn)resolve_in("libcublas.so.13", "cublasGemmEx");
		if (!g_real_gemmEx) {
			fprintf(stderr, "[shim] cublasGemmEx: real symbol resolve FAILED\n");
			return 15;
		}
	}
	n_calls = atomic_fetch_add(&g_gemmEx_count, 1) + 1;
	if (n_calls <= 3 || (n_calls & 0xff) == 0)
		fprintf(stderr, "[shim] cublasGemmEx #%lu M=%d N=%d K=%d Atype=%d\n",
		        n_calls, m, n, k, Atype);
	return g_real_gemmEx(handle, transa, transb, m, n, k,
	                     alpha, A, Atype, lda, B, Btype, ldb,
	                     beta, C, Ctype, ldc, computeType, algo);
}

cublasStatus_t cipher_cublasLtMatmul_impl(
    void *h, void *d, const void *alpha, const void *A, void *Adesc,
    const void *B, void *Bdesc, const void *beta, const void *C, void *Cdesc,
    void *D, void *Ddesc, const void *algo, void *workspace,
    unsigned long ws_bytes, void *stream)
{
	unsigned long n_calls;
	if (!g_real_ltMatmul) {
		g_real_ltMatmul = (cublasLtMatmul_fn)resolve_in("libcublasLt.so.12", "cublasLtMatmul");
		if (!g_real_ltMatmul) g_real_ltMatmul = (cublasLtMatmul_fn)resolve_in("libcublasLt.so.13", "cublasLtMatmul");
		if (!g_real_ltMatmul) {
			fprintf(stderr, "[shim] cublasLtMatmul: real symbol resolve FAILED\n");
			return 15;
		}
	}
	n_calls = atomic_fetch_add(&g_ltMatmul_count, 1) + 1;
	if (n_calls <= 3 || (n_calls & 0xff) == 0)
		fprintf(stderr, "[shim] cublasLtMatmul #%lu\n", n_calls);
	return g_real_ltMatmul(h, d, alpha, A, Adesc, B, Bdesc,
	                       beta, C, Cdesc, D, Ddesc, algo, workspace,
	                       ws_bytes, stream);
}

/* PLT-exported aliases so LD_PRELOAD intercepts before libtorch resolves. */
extern __attribute__((visibility("default"), alias("cipher_cublasGemmEx_impl")))
cublasStatus_t cublasGemmEx(
    cublasHandle_t, int, int, int, int, int,
    const void*, const void*, int, int,
    const void*, int, int,
    const void*, void*, int, int, int, int);

extern __attribute__((visibility("default"), alias("cipher_cublasLtMatmul_impl")))
cublasStatus_t cublasLtMatmul(
    void *, void *, const void *, const void *, void *,
    const void *, void *, const void *, const void *, void *,
    void *, void *, const void *, void *, unsigned long, void *);

__attribute__((constructor))
static void shim_init(void)
{
	if (atomic_exchange(&g_init_logged, 1)) return;
	fprintf(stderr, "[shim] LD_PRELOAD loaded — cublasGemmEx + cublasLtMatmul hooks armed\n");
}

__attribute__((destructor))
static void shim_fini(void)
{
	fprintf(stderr, "[shim] exit: cublasGemmEx=%lu cublasLtMatmul=%lu\n",
	        atomic_load(&g_gemmEx_count), atomic_load(&g_ltMatmul_count));
}
