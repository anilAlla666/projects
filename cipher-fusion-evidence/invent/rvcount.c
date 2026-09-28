/* rvcount.so — count-only interposition probe for detection COVERAGE on the FP8+EAGLE bundle.
 * Counts calls to the cuBLAS entry points the SDC detector family can host on, plus (m,n,k)
 * histogram for cublasGemmEx. If the FP8 target linears go through cutlass (vllm._C), they are
 * invisible to ALL of these — that is the coverage finding. Dumps to $RVC_OUT.<pid> periodically
 * and at exit. Pure interposition: no CUDA calls of its own. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <pthread.h>

static void *hcb, *hlt;
static void *cbsym(const char *n) {
  if (!hcb) hcb = dlopen("libcublas.so.13", RTLD_NOW | RTLD_GLOBAL);
  void *p = hcb ? dlsym(hcb, n) : 0;
  if (!p) fprintf(stderr, "[rvcount] FATAL no %s\n", n);
  return p;
}
static void *ltsym(const char *n) {
  if (!hlt) hlt = dlopen("libcublasLt.so.13", RTLD_NOW | RTLD_GLOBAL);
  void *p = hlt ? dlsym(hlt, n) : 0;
  if (!p) fprintf(stderr, "[rvcount] FATAL no %s\n", n);
  return p;
}

static unsigned long c_gemmex = 0, c_sbatch = 0, c_lt = 0, c_batched = 0;
typedef struct { int m, n, k; unsigned long cnt; } rec_t;
static rec_t recs[256];
static int nrec = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static void dump(void) {
  const char *base = getenv("RVC_OUT");
  if (!base) return;
  char path[512];
  snprintf(path, sizeof path, "%s.%d", base, (int)getpid());
  FILE *f = fopen(path, "w");
  if (!f) return;
  fprintf(f, "{\"gemmex\":%lu,\"stridedbatched\":%lu,\"ltmatmul\":%lu,\"batched\":%lu,\"shapes\":[",
          c_gemmex, c_sbatch, c_lt, c_batched);
  for (int i = 0; i < nrec; i++)
    fprintf(f, "%s[%d,%d,%d,%lu]", i ? "," : "", recs[i].m, recs[i].n, recs[i].k, recs[i].cnt);
  fprintf(f, "]}\n");
  fclose(f);
}
__attribute__((destructor)) static void fini(void) { dump(); }

static void note_shape(int m, int n, int k) {
  pthread_mutex_lock(&mu);
  for (int i = 0; i < nrec; i++)
    if (recs[i].m == m && recs[i].n == n && recs[i].k == k) { recs[i].cnt++; goto done; }
  if (nrec < 256) { recs[nrec].m = m; recs[nrec].n = n; recs[nrec].k = k; recs[nrec].cnt = 1; nrec++; }
done:
  if ((++c_gemmex & 0x7FF) == 0) dump();
  pthread_mutex_unlock(&mu);
}

typedef int (*gemmex_fn)(void *, int, int, int, int, int, const void *, const void *, int, int,
                         const void *, int, int, const void *, void *, int, int, int, int);
int cublasGemmEx(void *h, int ta, int tb, int m, int n, int k, const void *alpha,
                 const void *A, int At, int lda, const void *B, int Bt, int ldb,
                 const void *beta, void *C, int Ct, int ldc, int ct, int algo) {
  static gemmex_fn real = NULL;
  if (!real) real = (gemmex_fn)cbsym("cublasGemmEx");
  note_shape(m, n, k);
  return real(h, ta, tb, m, n, k, alpha, A, At, lda, B, Bt, ldb, beta, C, Ct, ldc, ct, algo);
}

typedef int (*sbatch_fn)(void *, int, int, int, int, int, const void *, const void *, int, int,
                         long long, const void *, int, int, long long, const void *, void *, int,
                         int, long long, int, int, int);
int cublasGemmStridedBatchedEx(void *h, int ta, int tb, int m, int n, int k, const void *alpha,
                               const void *A, int At, int lda, long long sA, const void *B, int Bt,
                               int ldb, long long sB, const void *beta, void *C, int Ct, int ldc,
                               long long sC, int bc, int ct, int algo) {
  static sbatch_fn real = NULL;
  if (!real) real = (sbatch_fn)cbsym("cublasGemmStridedBatchedEx");
  __sync_fetch_and_add(&c_sbatch, 1);
  return real(h, ta, tb, m, n, k, alpha, A, At, lda, sA, B, Bt, ldb, sB, beta, C, Ct, ldc, sC, bc, ct, algo);
}

typedef int (*lt_fn)(void *, void *, const void *, const void *, void *, const void *, void *,
                     const void *, void *, void *, void *, void *, const void *, void *,
                     size_t, void *);
int cublasLtMatmul(void *lh, void *desc, const void *alpha, const void *A, void *Ad,
                   const void *B, void *Bd, const void *beta, void *C, void *Cd, void *D,
                   void *Dd, const void *algo, void *ws, size_t wss, void *stream) {
  static lt_fn real = NULL;
  if (!real) real = (lt_fn)ltsym("cublasLtMatmul");
  unsigned long v = __sync_add_and_fetch(&c_lt, 1);
  if ((v & 0x7FF) == 0) dump();
  return real(lh, desc, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd, algo, ws, wss, stream);
}

typedef int (*gbatch_fn)(void *, int, int, int, int, int, const void *, const void *const *, int,
                         int, const void *const *, int, int, const void *, void *const *, int, int,
                         int, int, int);
int cublasGemmBatchedEx(void *h, int ta, int tb, int m, int n, int k, const void *alpha,
                        const void *const *A, int At, int lda, const void *const *B, int Bt,
                        int ldb, const void *beta, void *const *C, int Ct, int ldc, int bc,
                        int ct, int algo) {
  static gbatch_fn real = NULL;
  if (!real) real = (gbatch_fn)cbsym("cublasGemmBatchedEx");
  __sync_fetch_and_add(&c_batched, 1);
  return real(h, ta, tb, m, n, k, alpha, A, At, lda, B, Bt, ldb, beta, C, Ct, ldc, bc, ct, algo);
}
