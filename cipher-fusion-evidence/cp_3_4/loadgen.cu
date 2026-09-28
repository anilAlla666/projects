// loadgen.cu — CP 3.4 gate workload.
//
// 3-tenant asymmetric cuBLAS fp16 tensor-GEMM load, held >=90 s, so the
// CP 3.4 Grafana + ClickHouse dashboard shows a sustained per-second
// per-tenant silicon-state series.
//
// Derived from cp_3_3/flop_gate.cu's Phase-B worker (run_gemm_child).
// cp_3_3/flop_gate.cu itself is left BYTE-UNCHANGED — this is a
// CP-3.4-owned variant, not an edit of the CP 3.3 artifact.
//
// Build: nvcc loadgen.cu -I/home/ubuntu/cipher_kmod -lcublas \
//        -Wno-deprecated-gpu-targets -o loadgen
// Run:   sudo ./loadgen [duration_s]   (default 100)
//
// The parent never touches CUDA — each tenant is a fresh fork
// (CUDA contexts are not fork-safe).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "cipher_ioctl.h"

static const int D = 8192;

static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
static uint64_t now_ns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
static void register_tenant(int fd, const char *name) {
    struct cipher_register_tenant rt;
    memset(&rt, 0, sizeof(rt));
    rt.pid = getpid(); rt.tgid = getpid();
    strncpy(rt.tenant_id, name, CIPHER_TENANT_ID_LEN - 1);
    if (ioctl(fd, CIPHER_REGISTER_TENANT, &rt) != 0)
        perror("[loadgen] REGISTER_TENANT");
}
static void submit_launches(int fd, uint64_t n) {
    struct cipher_launch_stats ls;
    memset(&ls, 0, sizeof(ls));
    ls.pid = getpid(); ls.tgid = getpid();
    ls.timestamp_ns = now_ns();
    ls.launches_total = n;
    ioctl(fd, CIPHER_SUBMIT_LAUNCH_STATS, &ls);
}

struct GpuCtx { cublasHandle_t h; __half *A, *B, *C; };
static bool gpu_init(GpuCtx *g) {
    if (cudaSetDevice(0) != cudaSuccess) return false;
    if (cublasCreate(&g->h) != CUBLAS_STATUS_SUCCESS) return false;
    size_t n = (size_t)D * D;
    if (cudaMalloc(&g->A, n * sizeof(__half)) != cudaSuccess) return false;
    if (cudaMalloc(&g->B, n * sizeof(__half)) != cudaSuccess) return false;
    if (cudaMalloc(&g->C, n * sizeof(__half)) != cudaSuccess) return false;
    cudaMemset(g->A, 1, n * sizeof(__half));
    cudaMemset(g->B, 1, n * sizeof(__half));
    return true;
}
static void gpu_gemm(GpuCtx *g) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g->h, CUBLAS_OP_N, CUBLAS_OP_N, D, D, D,
                 &alpha, g->A, CUDA_R_16F, D, g->B, CUDA_R_16F, D,
                 &beta, g->C, CUDA_R_16F, D,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
static void gpu_free(GpuCtx *g) {
    cudaFree(g->A); cudaFree(g->B); cudaFree(g->C); cublasDestroy(g->h);
}

/* fixed-duty GEMM worker. sleep_us throttles the tenant: larger sleep =
 * smaller share of the contended device = lower attributed MFU. */
static int run_tenant(const char *name, double secs, int sleep_us) {
    int fd = open("/dev/cipher", O_RDWR);
    if (fd < 0) { perror("[tenant] open"); return 1; }
    GpuCtx g;
    if (!gpu_init(&g)) { fprintf(stderr, "[%s] gpu_init failed\n", name);
                         return 1; }
    register_tenant(fd, name);
    for (int i = 0; i < 5; i++) gpu_gemm(&g);
    cudaDeviceSynchronize();
    uint64_t count = 0;
    double t0 = now_s(), last_s = t0;
    while (now_s() - t0 < secs) {
        gpu_gemm(&g);
        count++;
        if (sleep_us > 0) { cudaDeviceSynchronize(); usleep(sleep_us); }
        if ((count & 0xf) == 0) {
            cudaDeviceSynchronize();
            double t = now_s();
            if (t - last_s > 0.1) { submit_launches(fd, count); last_s = t; }
        }
    }
    cudaDeviceSynchronize();
    submit_launches(fd, count);
    printf("[%s] done: %llu gemms in %.1fs\n",
           name, (unsigned long long)count, now_s() - t0);
    gpu_free(&g); close(fd);
    return 0;
}

int main(int argc, char **argv) {
    double secs = (argc > 1) ? atof(argv[1]) : 100.0;
    /* wide asymmetric duty spread so per-tenant attribution is
     * unambiguous in the dashboard — mirrors flop_gate Phase-B. */
    const char *names[3]   = { "cipher34-T0", "cipher34-T1", "cipher34-T2" };
    int         sleep_us[3] = { 30000,        10000,         0            };

    printf("[loadgen] CP 3.4 — 3-tenant asymmetric load, %.0f s\n", secs);
    pid_t pids[3];
    for (int i = 0; i < 3; i++) {
        pids[i] = fork();
        if (pids[i] == 0) _exit(run_tenant(names[i], secs, sleep_us[i]));
        usleep(200000);
    }
    int rc = 0, st;
    for (int i = 0; i < 3; i++) { waitpid(pids[i], &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 1; }
    printf("[loadgen] all tenants exited rc=%d\n", rc);
    return rc;
}
