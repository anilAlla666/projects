/* CP 5.4 Step 1.3a — split-order determinism probe (THROWAWAY).
 *
 * Verifies the grp_mask bridge's load-bearing assumption (scope memo §2):
 *   "kmod group g  ↔  the same physical SMs in every process".
 *
 * Two processes (fork before any CUDA init) each:
 *   - cuInit, device 0
 *   - cuDeviceGetDevResource(SM) → full SM set
 *   - cuDevSmResourceSplitByCount(minCount=8) → 16 groups   (== cipher_rt_green_ctx.c)
 *   - for each group g: (1) FNV-1a hash of the raw CUdevResource struct bytes;
 *                       (2) GROUND TRUTH — build a green ctx restricted to
 *                           group g, launch a grid that records %smid, collect
 *                           the set of physical SM ids the partition used.
 * Each child writes "g <smCount> <structHash> <smid,smid,...>" per group.
 * Parent diffs the two children's files: byte-identical ⇒ deterministic.
 *
 * Struct-hash PASS is already sound (the struct is the sole, process-state-free
 * input to cuDevResourceGenerateDesc → identical struct ⇒ identical partition;
 * groups[] is memset to 0 first so padding cannot cause a false mismatch).
 * The %smid set is the unambiguous ground-truth disambiguator.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#define NUM_GROUPS 16
#define MIN_SM      8
#define MAXSMID   256

#define CK(call) do { CUresult _r = (call); if (_r != CUDA_SUCCESS) { \
    const char *_m = 0; cuGetErrorName(_r, &_m); \
    fprintf(stderr, "[probe] %s:%d %s -> %d (%s)\n", __FILE__, __LINE__, \
            #call, _r, _m ? _m : "?"); return 2; } } while (0)

#define CKR(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    fprintf(stderr, "[probe] %s:%d %s -> %d (%s)\n", __FILE__, __LINE__, \
            #call, _e, cudaGetErrorString(_e)); return 2; } } while (0)

__global__ void probe_smid(int *hit)
{
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid < MAXSMID) hit[smid] = 1;
}

/* FNV-1a over raw struct bytes. */
static unsigned long long fnv1a(const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static int run_child(const char *outpath)
{
    CUdevice dev;
    CUdevResource full, groups[NUM_GROUPS], remaining;
    unsigned int nb = NUM_GROUPS;
    int *d_hit = NULL;

    CK(cuInit(0));
    CK(cuDeviceGet(&dev, 0));
    CK(cuDeviceGetDevResource(dev, &full, CU_DEV_RESOURCE_TYPE_SM));

    memset(groups, 0, sizeof(groups));      /* zero padding → noise-free hash */
    memset(&remaining, 0, sizeof(remaining));
    CK(cuDevSmResourceSplitByCount(groups, &nb, &full, &remaining, 0, MIN_SM));

    FILE *f = fopen(outpath, "w");
    if (!f) { perror("fopen"); return 2; }
    fprintf(f, "device_sm_count %u\n", full.sm.smCount);
    fprintf(f, "nb_groups %u remaining_sm %u\n", nb, remaining.sm.smCount);

    /* First runtime call retains the device primary context; green contexts
     * share its address space (the basis of cipher_rt_green_ctx.c's design),
     * so d_hit allocated here is reachable from kernels launched on any
     * group's green context below. */
    CKR(cudaMalloc((void **)&d_hit, MAXSMID * sizeof(int)));

    for (unsigned g = 0; g < nb; g++) {
        unsigned long long hsh = fnv1a(&groups[g], sizeof(CUdevResource));

        CUdevResourceDesc desc;
        CUgreenCtx gctx;
        CUcontext  cctx, popped;
        CK(cuDevResourceGenerateDesc(&desc, &groups[g], 1));
        CK(cuGreenCtxCreate(&gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
        CK(cuCtxFromGreenCtx(&cctx, gctx));
        CK(cuCtxPushCurrent(cctx));

        CKR(cudaMemset(d_hit, 0, MAXSMID * sizeof(int)));
        probe_smid<<<4096, 64>>>(d_hit);
        CKR(cudaGetLastError());
        CKR(cudaDeviceSynchronize());
        int hit[MAXSMID];
        CKR(cudaMemcpy(hit, d_hit, MAXSMID * sizeof(int), cudaMemcpyDeviceToHost));

        CK(cuCtxPopCurrent(&popped));
        CK(cuGreenCtxDestroy(gctx));

        int nsm = 0;
        char ids[1024]; ids[0] = 0;
        for (int s = 0; s < MAXSMID; s++)
            if (hit[s]) {
                char tmp[16];
                snprintf(tmp, sizeof(tmp), "%s%d", nsm ? "," : "", s);
                strncat(ids, tmp, sizeof(ids) - strlen(ids) - 1);
                nsm++;
            }
        fprintf(f, "g %2u smCount %u structHash %016llx nsm %d smids %s\n",
                g, groups[g].sm.smCount, hsh, nsm, ids);
    }
    cudaFree(d_hit);
    fclose(f);
    return 0;
}

int main(void)
{
    /* fork BEFORE any CUDA call — each child inits CUDA independently. */
    pid_t a = fork();
    if (a == 0) { _exit(run_child("/home/ubuntu/cipher-fusion-evidence/cp_5_4/step1_3/probe_proc_0.txt")); }
    pid_t b = fork();
    if (b == 0) { _exit(run_child("/home/ubuntu/cipher-fusion-evidence/cp_5_4/step1_3/probe_proc_1.txt")); }

    int sa = 0, sb = 0;
    waitpid(a, &sa, 0);
    waitpid(b, &sb, 0);
    if (!WIFEXITED(sa) || WEXITSTATUS(sa) != 0 ||
        !WIFEXITED(sb) || WEXITSTATUS(sb) != 0) {
        fprintf(stderr, "[probe] a child failed (sa=%d sb=%d) — AMBIGUOUS\n", sa, sb);
        return 3;
    }
    fprintf(stderr, "[probe] both children exited 0; parent does the diff\n");
    return 0;
}
