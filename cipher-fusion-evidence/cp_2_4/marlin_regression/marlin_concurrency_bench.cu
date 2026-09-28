// marlin_concurrency_bench.cu — CP 2.4 Marlin per-stream registry regression (test B).
//
// Drives the v2 Marlin engine (cipher_rt_phase4/libcipher_rt.so) directly,
// from N threads, each on its own CUDA stream, all sharing one INT4 weight
// — the real multi-tenant shape (one shared model, per-tenant streams).
//
// Two modes:
//   per-stream  : each thread its own stream -> N registry slots -> N-way
//                 concurrency (the CP 2.4 per-stream registry under test).
//   shared      : all threads on ONE stream -> 1 registry slot -> serialized
//                 (mimics the CP 0.4/0.5 single-_MARLIN_STREAM ceiling).
//
// PASS = per-stream aggregate Marlin GEMM/s scales with N while shared stays
// flat, and cipher_rt_marlin_engine_ws_slots() == N in per-stream mode.
//
// Build: nvcc marlin_concurrency_bench.cu -o marlin_concurrency_bench -lcudart -ldl
// Run:   ./marlin_concurrency_bench   (engine NVRTC-compiles ~19s on first call)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <vector>
#include <thread>
#include <atomic>
#include <dlfcn.h>
#include <cuda_runtime.h>

/* ── v2 Marlin engine C-ABI (resolved from libcipher_rt.so via dlsym) ─── */
typedef int (*pf_init)(void);
typedef int (*pf_ensure)(void);
typedef int (*pf_qrepack)(const void *, int, int);
typedef int (*pf_lookup)(const void *, void **, void **, int *, int *, int *);
typedef int (*pf_dispatch)(const void *, const void *, const void *, void *,
                           int, int, int, int, void *);
typedef int (*pf_ws_slots)(void);

static pf_init     eng_init;
static pf_ensure   eng_ensure;
static pf_qrepack  eng_qrepack;
static pf_lookup   eng_lookup;
static pf_dispatch eng_dispatch;
static pf_ws_slots eng_ws_slots;

static const int M = 1;          /* decode: single token */
static const int K = 4096;       /* Mistral-7B hidden */
static const int N = 4096;
static const int G = 128;
static const double DURATION_S = 5.0;

static double now_s() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* shared INT4 weight (quantized once) */
static void *g_B = nullptr, *g_S = nullptr;
static int   g_K = 0, g_N = 0, g_G = 0;

struct ThreadResult { uint64_t gemms; int rc_fail; };

/* One worker: own activation+output buffers, dispatch in a tight loop. */
static void worker(int tid, cudaStream_t stream, double secs, ThreadResult *res) {
    void *a = nullptr, *c = nullptr;
    res->gemms = 0; res->rc_fail = 0;
    if (cudaMalloc(&a, (size_t)M * K * 2) != cudaSuccess ||
        cudaMalloc(&c, (size_t)M * N * 2) != cudaSuccess) {
        fprintf(stderr, "[t%d] activation malloc failed\n", tid);
        return;
    }
    cudaMemset(a, 0x3c, (size_t)M * K * 2);   /* nonzero fp16 activation */

    double t0 = now_s();
    uint64_t n = 0;
    while (now_s() - t0 < secs) {
        for (int i = 0; i < 32; i++) {        /* batch 32 dispatches per time check */
            int rc = eng_dispatch(a, g_B, g_S, c, M, N, K, G, (void *)stream);
            if (rc != 0) { res->rc_fail++; }
            n++;
        }
    }
    cudaStreamSynchronize(stream);
    res->gemms = n;
    cudaFree(a); cudaFree(c);
}

/* Run one (N, mode) point; return aggregate GEMM/s. */
static double run_point(int nthreads, bool per_stream, int *ws_slots_out,
                        uint64_t *rc_fails_out) {
    std::vector<cudaStream_t> streams(nthreads);
    cudaStream_t shared = nullptr;
    if (!per_stream) cudaStreamCreate(&shared);
    for (int i = 0; i < nthreads; i++) {
        if (per_stream) cudaStreamCreate(&streams[i]);
        else            streams[i] = shared;
    }

    std::vector<ThreadResult> results(nthreads);
    std::vector<std::thread>  pool;
    double t0 = now_s();
    for (int i = 0; i < nthreads; i++)
        pool.emplace_back(worker, i, streams[i], DURATION_S, &results[i]);
    for (auto &t : pool) t.join();
    double wall = now_s() - t0;

    uint64_t total = 0, fails = 0;
    for (auto &r : results) { total += r.gemms; fails += r.rc_fail; }
    *ws_slots_out  = eng_ws_slots();
    *rc_fails_out  = fails;

    if (per_stream) for (auto &s : streams) cudaStreamDestroy(s);
    else            cudaStreamDestroy(shared);
    return (double)total / wall;
}

int main() {
    printf("=== CP 2.4 — Marlin per-stream registry regression (test B) ===\n");
    printf("shape: M=%d K=%d N=%d G=%d  duration=%.0fs/point\n\n",
           M, K, N, G, DURATION_S);

    void *lib = dlopen("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so",
                       RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    eng_init     = (pf_init)    dlsym(lib, "cipher_rt_marlin_engine_init");
    eng_ensure   = (pf_ensure)  dlsym(lib, "cipher_rt_marlin_engine_ensure_compiled");
    eng_qrepack  = (pf_qrepack) dlsym(lib, "cipher_rt_marlin_engine_quantize_repack");
    eng_lookup   = (pf_lookup)  dlsym(lib, "cipher_rt_marlin_engine_lookup");
    eng_dispatch = (pf_dispatch)dlsym(lib, "cipher_rt_marlin_engine_dispatch");
    eng_ws_slots = (pf_ws_slots)dlsym(lib, "cipher_rt_marlin_engine_ws_slots");
    if (!eng_init || !eng_ensure || !eng_qrepack || !eng_lookup ||
        !eng_dispatch || !eng_ws_slots) {
        fprintf(stderr, "dlsym: missing engine symbol(s)\n"); return 1;
    }

    cudaSetDevice(0);
    cudaFree(0);                              /* force a CUDA context */

    if (eng_init() != 0) { fprintf(stderr, "engine_init failed\n"); return 1; }
    printf("[setup] engine init OK; NVRTC compile (~19s cold)...\n"); fflush(stdout);
    if (eng_ensure() != 0) { fprintf(stderr, "ensure_compiled failed\n"); return 1; }

    /* one shared INT4 weight — quantize+repack once */
    void *d_w = nullptr;
    if (cudaMalloc(&d_w, (size_t)K * N * 2) != cudaSuccess) {
        fprintf(stderr, "weight malloc failed\n"); return 1;
    }
    cudaMemset(d_w, 0x3c, (size_t)K * N * 2); /* nonzero fp16 weight */
    if (eng_qrepack(d_w, K, N) != 0) {
        fprintf(stderr, "quantize_repack failed\n"); return 1;
    }
    if (eng_lookup(d_w, &g_B, &g_S, &g_K, &g_N, &g_G) != 1) {
        fprintf(stderr, "lookup failed\n"); return 1;
    }
    printf("[setup] weight quantized+repacked (K=%d N=%d G=%d)\n\n", g_K, g_N, g_G);

    const int N_GRID[] = {1, 2, 4, 8, 16, 32, 64};
    const int N_PTS = sizeof(N_GRID) / sizeof(N_GRID[0]);

    printf("%-10s %6s  %14s  %9s  %8s\n",
           "mode", "N", "agg_gemm/s", "ws_slots", "rc_fail");
    printf("---------------------------------------------------------\n");

    FILE *js = fopen("marlin_regression_result.json", "w");
    fprintf(js, "{\n  \"cp\": \"2.4\", \"test\": \"B_marlin_concurrency\",\n");
    fprintf(js, "  \"shape\": {\"M\": %d, \"K\": %d, \"N\": %d, \"G\": %d},\n",
            M, K, N, G);
    fprintf(js, "  \"points\": [\n");

    double base_per = 0, base_shr = 0;
    bool first = true;
    for (int mode = 0; mode < 2; mode++) {
        bool per_stream = (mode == 0);
        for (int i = 0; i < N_PTS; i++) {
            int nt = N_GRID[i];
            int ws = 0; uint64_t rcf = 0;
            double gps = run_point(nt, per_stream, &ws, &rcf);
            if (per_stream && nt == 1) base_per = gps;
            if (!per_stream && nt == 1) base_shr = gps;
            printf("%-10s %6d  %14.0f  %9d  %8lu\n",
                   per_stream ? "per-stream" : "shared", nt, gps, ws, rcf);
            if (!first) fprintf(js, ",\n");
            first = false;
            fprintf(js, "    {\"mode\": \"%s\", \"N\": %d, \"agg_gemm_per_s\": %.1f, "
                        "\"ws_slots\": %d, \"rc_fails\": %lu}",
                    per_stream ? "per_stream" : "shared", nt, gps, ws, rcf);
        }
        printf("---------------------------------------------------------\n");
    }
    fprintf(js, "\n  ]\n}\n");
    fclose(js);

    printf("\nper-stream N=1 -> %.0f gemm/s ; scaling is agg/s vs N above.\n",
           base_per);
    printf("result -> marlin_regression_result.json\n");
    cudaFree(d_w);
    return 0;
}
