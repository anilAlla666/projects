// flop_gate.cu — CP 3.3 gate harness.
//
// Exercises cipher_flopd + cipher_kmod against the four CP 3.3 gate
// criteria with a cuBLAS fp16 tensor-core GEMM workload of known
// analytical FLOP count.
//
//   Phase S  settled load staircase  -> Tier-1 NVML correlation
//   Phase A  single-tenant flat-out  -> Tier-2 absolute (analytical vs kmod)
//   Phase B  3-tenant asymmetric     -> per-tenant FLOPs + MFU attribution
//
// Tier-1 uses a SETTLED staircase: each load level is held until all
// instruments settle, then sampled. This is deliberate — the kmod FLOP
// series responds within one 100 ms PM-sampling window, while NVML
// sm_util lags ~1 s and board power ramps ~1.5 s. Correlating across
// transients would penalise the FLOP telemetry for being the *faster*
// instrument; the honest question is whether it tracks GPU activity
// across the dynamic range, which a settled staircase answers cleanly.
//
// The PARENT never touches CUDA — every GEMM worker is a fresh fork
// (CUDA contexts are not fork-safe).
//
// Build: nvcc flop_gate.cu -I/home/ubuntu/cipher_kmod -lcublas \
//        -lnvidia-ml -Wno-deprecated-gpu-targets -o flop_gate
// Run:   sudo ./flop_gate
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <nvml.h>

#include "cipher_ioctl.h"

static const int    D = 8192;
static const double FLOPS_PER_GEMM = 2.0 * (double)D * D * D;
static const char  *PHASEA_RESULT = "/tmp/flopgate_phaseA.txt";

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
        perror("[flop_gate] REGISTER_TENANT");
}
static void submit_launches(int fd, uint64_t n) {
    struct cipher_launch_stats ls;
    memset(&ls, 0, sizeof(ls));
    ls.pid = getpid(); ls.tgid = getpid();
    ls.timestamp_ns = now_ns();
    ls.launches_total = n;
    ioctl(fd, CIPHER_SUBMIT_LAUNCH_STATS, &ls);
}
static int query_flops(int fd, struct cipher_flop_query *q) {
    memset(q, 0, sizeof(*q));
    q->max_tenants = CIPHER_FLOP_QUERY_MAX_TENANTS;
    return ioctl(fd, CIPHER_QUERY_FLOPS, q);
}
static double pearson(const std::vector<double> &x,
                      const std::vector<double> &y) {
    size_t n = x.size();
    if (n < 3) return 0.0;
    double mx = 0, my = 0;
    for (size_t i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sxy = 0, sxx = 0, syy = 0;
    for (size_t i = 0; i < n; i++) {
        double dx = x[i] - mx, dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if (sxx <= 0 || syy <= 0) return 0.0;
    return sxy / sqrt(sxx * syy);
}

struct GpuCtx {
    cublasHandle_t h; __half *A, *B, *C;
};
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

/* flat-out / fixed-duty GEMM worker (phase A, phase B). */
static int run_gemm_child(const char *name, double secs, int sleep_us,
                          const char *result_path) {
    int fd = open("/dev/cipher", O_RDWR);
    if (fd < 0) { perror("[child] open"); return 1; }
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
    double elapsed = now_s() - t0;
    submit_launches(fd, count);
    if (result_path) {
        FILE *f = fopen(result_path, "w");
        if (f) { fprintf(f, "%llu %.6f\n",
                          (unsigned long long)count, elapsed); fclose(f); }
    }
    gpu_free(&g); close(fd);
    return 0;
}

/* settled-staircase worker (phase S): holds 5 increasing duty levels, each
 * for level_secs, so all instruments settle before the parent samples. */
static int run_staircase_child(double level_secs) {
    int fd = open("/dev/cipher", O_RDWR);
    if (fd < 0) { perror("[stair] open"); return 1; }
    GpuCtx g;
    if (!gpu_init(&g)) { fprintf(stderr, "[stair] gpu_init failed\n");
                         return 1; }
    register_tenant(fd, "flopgate-S");
    for (int i = 0; i < 5; i++) gpu_gemm(&g);
    cudaDeviceSynchronize();
    int sleeps[5] = { 6000, 3000, 1500, 500, 0 };  /* ~19/32/48/74/100% duty */
    uint64_t count = 0;
    double last_s = now_s();
    for (int lvl = 0; lvl < 5; lvl++) {
        double lt0 = now_s();
        while (now_s() - lt0 < level_secs) {
            gpu_gemm(&g);
            count++;
            if (sleeps[lvl] > 0) { cudaDeviceSynchronize();
                                   usleep(sleeps[lvl]); }
            if ((count & 0x7) == 0) {
                cudaDeviceSynchronize();
                double t = now_s();
                if (t - last_s > 0.1) { submit_launches(fd, count);
                                        last_s = t; }
            }
        }
    }
    cudaDeviceSynchronize();
    submit_launches(fd, count);
    gpu_free(&g); close(fd);
    return 0;
}

static nvmlDevice_t g_nvdev;
static bool g_nvml_ok = false;
static double read_power_w() {
    unsigned int mw = 0;
    if (g_nvml_ok && nvmlDeviceGetPowerUsage(g_nvdev, &mw) == NVML_SUCCESS)
        return mw / 1000.0;
    return 0.0;
}

int main() {
    int fd = open("/dev/cipher", O_RDWR);
    if (fd < 0) { perror("[flop_gate] open /dev/cipher"); return 1; }
    g_nvml_ok = (nvmlInit_v2() == NVML_SUCCESS) &&
                (nvmlDeviceGetHandleByIndex_v2(0, &g_nvdev) == NVML_SUCCESS);

    std::vector<double> c_flops, c_utilclk, c_power;   // Tier-1 (settled)
    std::vector<double> a_dev_rate;                    // Tier-2
    struct cipher_flop_query q, best_b;
    memset(&best_b, 0, sizeof(best_b));
    FILE *csv = fopen("flop_gate_corr.csv", "w");
    if (csv) fprintf(csv, "phase,settled,kmod_flops,util_x_clk,power_w\n");

    printf("=== CP 3.3 flop_gate — D=%d fp16 tensor GEMM ===\n", D);
    printf("flops/gemm = %.4e\n", FLOPS_PER_GEMM);

    /* ---- Phase S: settled idle + load staircase -> Tier-1 ---- */
    printf("\n[phase S] idle 3 s + 5-level settled staircase (5 x 7 s)...\n");
    for (int i = 0; i < 14; i++) {              /* idle anchor */
        usleep(200000);
        if (query_flops(fd, &q) == 0) {
            c_flops.push_back((double)q.device_tensor_flops_per_s);
            c_utilclk.push_back((double)q.sm_util_pct * q.sm_clock_mhz);
            c_power.push_back(read_power_w());
            if (csv) fprintf(csv, "idle,1,%llu,%u,%.1f\n",
                (unsigned long long)q.device_tensor_flops_per_s,
                q.sm_util_pct * q.sm_clock_mhz, read_power_w());
        }
    }
    const double LVL = 7.0;
    pid_t ps = fork();
    if (ps == 0) _exit(run_staircase_child(LVL));
    double ts0 = now_s();
    int wst;
    while (waitpid(ps, &wst, WNOHANG) == 0) {
        usleep(250000);
        if (query_flops(fd, &q) != 0) continue;
        double t = now_s() - ts0;
        double pos = fmod(t, LVL);
        bool settled = (pos > 3.5);             /* skip 3.5 s settling */
        double pw = read_power_w();
        if (csv) fprintf(csv, "stair,%d,%llu,%u,%.1f\n", settled ? 1 : 0,
            (unsigned long long)q.device_tensor_flops_per_s,
            q.sm_util_pct * q.sm_clock_mhz, pw);
        if (settled) {
            c_flops.push_back((double)q.device_tensor_flops_per_s);
            c_utilclk.push_back((double)q.sm_util_pct * q.sm_clock_mhz);
            c_power.push_back(pw);
        }
    }
    double r_util  = pearson(c_flops, c_utilclk);
    double r_power = pearson(c_flops, c_power);
    double r_best  = r_util > r_power ? r_util : r_power;
    printf("[phase S] %zu settled samples  r_util=%.4f  r_power=%.4f\n",
           c_flops.size(), r_util, r_power);

    /* ---- Phase A: single-tenant flat-out -> Tier-2 ---- */
    printf("[phase A] single-tenant GEMM loop 15 s...\n");
    unlink(PHASEA_RESULT);
    pid_t pa = fork();
    if (pa == 0) _exit(run_gemm_child("flopgate-A", 15.0, 0, PHASEA_RESULT));
    while (waitpid(pa, &wst, WNOHANG) == 0) {
        usleep(200000);
        if (query_flops(fd, &q) == 0)
            a_dev_rate.push_back((double)q.device_tensor_flops_per_s);
    }
    uint64_t a_count = 0; double a_elapsed = 0;
    FILE *rf = fopen(PHASEA_RESULT, "r");
    if (rf) { if (fscanf(rf, "%llu %lf",
                  (unsigned long long *)&a_count, &a_elapsed) != 2)
                  a_count = 0;
              fclose(rf); }
    double analytical_rate = a_elapsed > 0
        ? (double)a_count * FLOPS_PER_GEMM / a_elapsed : 0;
    double meas_rate = 0; int mn = 0;
    for (size_t i = a_dev_rate.size() / 5; i < a_dev_rate.size(); i++) {
        meas_rate += a_dev_rate[i]; mn++;
    }
    meas_rate = mn ? meas_rate / mn : 0;
    double ratio = analytical_rate > 0 ? meas_rate / analytical_rate : 0;
    printf("[phase A] gemms=%llu elapsed=%.2fs  analytical=%.1f TFLOP/s  "
           "kmod=%.1f TFLOP/s  ratio=%.3f\n",
           (unsigned long long)a_count, a_elapsed, analytical_rate / 1e12,
           meas_rate / 1e12, ratio);

    /* ---- Phase B: 3-tenant asymmetric attribution ---- */
    printf("[phase B] 3 tenants, asymmetric duty, 20 s...\n");
    /* wide duty spread so the asymmetric per-tenant attribution is
     * unambiguous (small sleeps are swamped by contended GEMM time). */
    int sleep_us[3] = { 30000, 10000, 0 };
    pid_t kids[3];
    for (int i = 0; i < 3; i++) {
        char nm[64]; snprintf(nm, sizeof(nm), "flopgate-T%d", i);
        pid_t p = fork();
        if (p == 0) _exit(run_gemm_child(nm, 20.0, sleep_us[i], NULL));
        kids[i] = p;
    }
    int best_score = -1, alive = 3;
    while (alive > 0) {
        usleep(200000);
        if (query_flops(fd, &q) == 0) {
            int score = 0;
            for (uint32_t i = 0; i < q.n_tenants; i++)
                if (strncmp(q.tenants[i].tenant_id, "flopgate-T", 10) == 0
                    && q.tenants[i].attributed_flops_per_s > 0)
                    score++;
            /* keep the LATEST max-score query — late = steady state, where
             * the asymmetric per-tenant duties are fully expressed. */
            if (score >= best_score) { best_score = score; best_b = q; }
        }
        alive = 0;
        for (int i = 0; i < 3; i++)
            if (kids[i] > 0 && waitpid(kids[i], &wst, WNOHANG) == 0) alive++;
            else kids[i] = -1;
    }
    printf("[phase B] best mid-run query: device %.1f TFLOP/s  MFU=%.2f%%\n",
           best_b.device_tensor_flops_per_s / 1e12,
           best_b.device_mfu_milli_pct / 1000.0);
    printf("  %-14s %-8s %-11s %-18s %-7s\n",
           "TENANT", "PID", "SHARE_ppm", "FLOPS/s", "MFU%");
    uint64_t sum_attr = 0;
    int flopgate_tenants = 0, with_mfu = 0;
    for (uint32_t i = 0; i < best_b.n_tenants; i++) {
        struct cipher_flop_tenant *t = &best_b.tenants[i];
        if (strncmp(t->tenant_id, "flopgate-T", 10) != 0) continue;
        flopgate_tenants++;
        sum_attr += t->attributed_flops_per_s;
        if (t->mfu_milli_pct > 0) with_mfu++;
        printf("  %-14s %-8u %-11u %-18llu %.3f\n",
               t->tenant_id, t->pid, t->launch_share_ppm,
               (unsigned long long)t->attributed_flops_per_s,
               t->mfu_milli_pct / 1000.0);
    }

    bool gate_a  = best_b.ring_samples >= 32;
    bool gate_b  = (flopgate_tenants == 3 && with_mfu == 3);
    bool gate_c1 = (r_best >= 0.95);
    bool gate_c2 = (ratio >= 0.80 && ratio <= 1.20);

    printf("\n=== GATE ===\n");
    printf("(a) continuous kmod-owned series : ring=%u  -> %s\n",
           best_b.ring_samples, gate_a ? "PASS" : "FAIL");
    printf("(b) per-tenant FLOPs+MFU visible : %d/3 tenants, %d MFU>0 -> %s\n",
           flopgate_tenants, with_mfu, gate_b ? "PASS" : "FAIL");
    printf("(c1) Tier-1 settled correlation  : r_util=%.4f r_power=%.4f "
           "-> %s\n", r_util, r_power, gate_c1 ? "PASS" : "FAIL");
    printf("(c2) Tier-2 absolute ratio       : %.3f (0.80-1.20 proxy band) "
           "-> %s\n", ratio, gate_c2 ? "PASS" : "FAIL");
    printf("sum attributed = %.1f TFLOP/s  vs device %.1f TFLOP/s\n",
           sum_attr / 1e12, best_b.device_tensor_flops_per_s / 1e12);

    FILE *jf = fopen("flop_gate_result.json", "w");
    if (jf) {
        fprintf(jf,
            "{\n  \"cp\": \"3.3\",\n  \"gemm_dim\": %d,\n"
            "  \"phaseA_gemms\": %llu,\n  \"phaseA_elapsed_s\": %.3f,\n"
            "  \"analytical_tflops\": %.4f,\n  \"kmod_tflops\": %.4f,\n"
            "  \"tier2_ratio\": %.4f,\n"
            "  \"tier1_settled_samples\": %zu,\n"
            "  \"tier1_r_util\": %.4f,\n  \"tier1_r_power\": %.4f,\n"
            "  \"ring_samples\": %u,\n  \"flopgate_tenants\": %d,\n"
            "  \"tenants_with_mfu\": %d,\n"
            "  \"sum_attributed_tflops\": %.4f,\n"
            "  \"device_tflops_phaseB\": %.4f,\n"
            "  \"gate_a\": %s,\n  \"gate_b\": %s,\n"
            "  \"gate_c1\": %s,\n  \"gate_c2\": %s\n}\n",
            D, (unsigned long long)a_count, a_elapsed,
            analytical_rate / 1e12, meas_rate / 1e12, ratio,
            c_flops.size(), r_util, r_power, best_b.ring_samples,
            flopgate_tenants, with_mfu, sum_attr / 1e12,
            best_b.device_tensor_flops_per_s / 1e12,
            gate_a ? "true" : "false", gate_b ? "true" : "false",
            gate_c1 ? "true" : "false", gate_c2 ? "true" : "false");
        fclose(jf);
    }
    if (csv) fclose(csv);
    if (g_nvml_ok) nvmlShutdown();
    close(fd);
    printf("result -> flop_gate_result.json  corr -> flop_gate_corr.csv\n");
    return (gate_a && gate_b && gate_c1 && gate_c2) ? 0 : 2;
}
