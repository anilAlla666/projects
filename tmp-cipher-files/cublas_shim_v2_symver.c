/* D6.2 v2 — same shim but with versioned symbol binding.
 *
 * .symver binds our cipher_cublasGemmEx_impl to the version-tagged name
 * cublasGemmEx@libcublas.so.13 that PyTorch 2.11 looks up. If this
 * resolves cleanly, no GOT patching needed.
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

typedef cublasStatus_t (*cublasGemmStridedBatchedEx_fn)(
    cublasHandle_t, int, int, int, int, int,
    const void*, const void*, int, int, long long,
    const void*, int, int, long long,
    const void*, void*, int, int, long long,
    int, int, int);

static cublasGemmEx_fn  g_real_gemmEx  = NULL;
static cublasGemmStridedBatchedEx_fn g_real_gemmSBEx = NULL;

static atomic_ulong g_gemmEx_count   = 0;
static atomic_ulong g_gemmSBEx_count = 0;

static void *resolve_in(const char *soname, const char *sym)
{
	void *h = dlopen(soname, RTLD_NOW | RTLD_NOLOAD);
	if (!h) h = dlopen(soname, RTLD_NOW);
	if (!h) return NULL;
	/* Use dlvsym to pick the correct version tag */
	void *p = dlvsym(h, sym, soname);
	if (!p) p = dlsym(h, sym);
	return p;
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
		g_real_gemmEx = (cublasGemmEx_fn)resolve_in("libcublas.so.13", "cublasGemmEx");
		if (!g_real_gemmEx) g_real_gemmEx = (cublasGemmEx_fn)resolve_in("libcublas.so.12", "cublasGemmEx");
		if (!g_real_gemmEx) { fprintf(stderr, "[shim] GemmEx resolve FAILED\n"); return 15; }
	}
	n_calls = atomic_fetch_add(&g_gemmEx_count, 1) + 1;
	if (n_calls <= 5 || (n_calls & 0xff) == 0)
		fprintf(stderr, "[shim] cublasGemmEx #%lu M=%d N=%d K=%d Atype=%d\n",
		        n_calls, m, n, k, Atype);
	return g_real_gemmEx(handle, transa, transb, m, n, k,
	                     alpha, A, Atype, lda, B, Btype, ldb,
	                     beta, C, Ctype, ldc, computeType, algo);
}

cublasStatus_t cipher_cublasGemmStridedBatchedEx_impl(
    cublasHandle_t handle, int transa, int transb,
    int m, int n, int k,
    const void* alpha, const void* A, int Atype, int lda, long long sa,
    const void* B, int Btype, int ldb, long long sb,
    const void* beta, void* C, int Ctype, int ldc, long long sc,
    int batchCount, int computeType, int algo)
{
	unsigned long n_calls;
	if (!g_real_gemmSBEx) {
		g_real_gemmSBEx = (cublasGemmStridedBatchedEx_fn)resolve_in("libcublas.so.13", "cublasGemmStridedBatchedEx");
		if (!g_real_gemmSBEx) g_real_gemmSBEx = (cublasGemmStridedBatchedEx_fn)resolve_in("libcublas.so.12", "cublasGemmStridedBatchedEx");
		if (!g_real_gemmSBEx) { fprintf(stderr, "[shim] GemmSBEx resolve FAILED\n"); return 15; }
	}
	n_calls = atomic_fetch_add(&g_gemmSBEx_count, 1) + 1;
	if (n_calls <= 5 || (n_calls & 0xff) == 0)
		fprintf(stderr, "[shim] cublasGemmStridedBatchedEx #%lu M=%d N=%d K=%d B=%d\n",
		        n_calls, m, n, k, batchCount);
	return g_real_gemmSBEx(handle, transa, transb, m, n, k,
	                       alpha, A, Atype, lda, sa, B, Btype, ldb, sb,
	                       beta, C, Ctype, ldc, sc,
	                       batchCount, computeType, algo);
}

/* Version-tag the exported symbol to match PyTorch 2.11 / cu13 lookup */
__asm__(".symver cipher_cublasGemmEx_impl,cublasGemmEx@libcublas.so.13");
__asm__(".symver cipher_cublasGemmStridedBatchedEx_impl,cublasGemmStridedBatchedEx@libcublas.so.13");

/* Also export plain (un-versioned) aliases for fallback / cu12 callers */
extern __attribute__((visibility("default"), alias("cipher_cublasGemmEx_impl")))
cublasStatus_t cublasGemmEx(
    cublasHandle_t, int, int, int, int, int,
    const void*, const void*, int, int,
    const void*, int, int,
    const void*, void*, int, int, int, int);

__attribute__((constructor))
static void shim_init(void) {
	fprintf(stderr, "[shim v2] LD_PRELOAD loaded — versioned cublasGemmEx hook armed\n");
}

__attribute__((destructor))
static void shim_fini(void) {
	fprintf(stderr, "[shim v2] exit: cublasGemmEx=%lu cublasGemmStridedBatchedEx=%lu\n",
	        atomic_load(&g_gemmEx_count), atomic_load(&g_gemmSBEx_count));
}
