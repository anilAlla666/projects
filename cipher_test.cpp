// =============================================================================
// CIPHER Classification Engine — Test + Latency Benchmark
// Compiles without CUDA. Simulates cuLaunchKernel geometry.
// g++ -O3 -std=c++17 cipher_test.cpp -o cipher_test && ./cipher_test
// =============================================================================

#include "cipher_classify.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cipher;
using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Reference kernels: real-world geometry from cuLaunchKernel intercepts
// ---------------------------------------------------------------------------
struct KernelCase {
    const char* name;
    KernelGeom  geom;
    OpClass     expected;
};

// Fake function pointers — unique addresses simulate distinct kernel symbols
static uint8_t _fn_gemm, _fn_attn, _fn_conv, _fn_ew, _fn_red, _fn_cpy, _fn_iter;

static KernelCase kCases[] = {
    // name                  gx    gy   gz    bx    by   bz  shmem   fn         expected
    { "cuBLAS SGEMM 4096²",
      { 256,  256, 1,  16,  16, 1, 8192,  &_fn_gemm }, OpClass::GEMM            },

    { "cuBLAS HGEMM batched",
      { 128,  64,  8,  32,  16, 1, 16384, &_fn_gemm }, OpClass::GEMM            },

    { "Flash-Attn v2 (seq=2048,h=32)",
      { 2048, 32,  1, 128,   1, 1, 32768, &_fn_attn }, OpClass::ATTENTION       },

    { "Standard Attn (seq=512, batch×h=8)",
      { 512,  8,   1,  64,   1, 1, 16384, &_fn_attn }, OpClass::ATTENTION       },

    { "Conv2D 3×3 (ResNet block)",
      { 64,   32,  8,  16,  16, 1, 4096,  &_fn_conv }, OpClass::CONVOLUTION     },

    { "Fused ReLU+Bias (256M elements)",
      { 512*1024, 1, 1, 256, 1, 1, 0,     &_fn_ew   }, OpClass::ELEMENTWISE     },

    { "LayerNorm (batch=512, d=768)",
      { 512,  1,   1, 256,   1, 1, 2048,  &_fn_red  }, OpClass::REDUCTION       },

    { "Softmax 1D reduction",
      { 128,  8,   1, 512,   1, 1, 4096,  &_fn_red  }, OpClass::REDUCTION       },

    { "Matrix Transpose 4096²",
      { 128, 128,  1,  32,  32, 1, 4096,  &_fn_cpy  }, OpClass::MEMCPY_TRANSPOSE},

    { "NCHW→NHWC (32×32 tile)",
      { 256,  64,  1,  16,  16, 1, 2048,  &_fn_cpy  }, OpClass::MEMCPY_TRANSPOSE},

    { "Custom Physics Solver",
      { 37,   13,  5,  48,   7, 2, 11000, &_fn_iter }, OpClass::ITERATIVE_CUSTOM},

    { "Sparse MatMul (irregular)",
      { 1023,  3,  1, 192,   1, 1, 384,   &_fn_iter }, OpClass::ITERATIVE_CUSTOM},
};

static constexpr int kNCases = sizeof(kCases) / sizeof(kCases[0]);

// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------
static int run_correctness() {
    printf("=== Correctness ===\n");
    int pass = 0, fail = 0;
    for (auto& c : kCases) {
        auto res = classify(c.geom);
        bool ok  = (res.op == c.expected);
        printf("  [%s] %-40s  got=%-18s  conf=%3u  cache=%d\n",
               ok ? "PASS" : "FAIL",
               c.name,
               kOpClassName[(int)res.op],
               res.confidence,
               (int)res.cache_hit);
        ok ? ++pass : ++fail;
    }
    printf("\n  %d/%d passed\n\n", pass, kNCases);
    return fail;
}

// ---------------------------------------------------------------------------
// Latency benchmark — cache-miss path (fresh ptr each iteration)
// ---------------------------------------------------------------------------
static void bench_cold(int iterations = 1'000'000) {
    printf("=== Cold-path latency (N=%d, unique fn ptrs) ===\n", iterations);

    // Allocate a dummy byte array to use addresses as fake function pointers
    std::vector<uint8_t> fn_pool(iterations + 1);

    // GEMM geometry as representative "hard" case (2D grid, shared mem checks)
    const KernelGeom base_geom { 256, 256, 1, 16, 16, 1, 8192, nullptr };

    auto t0 = Clock::now();
    volatile OpClass sink = OpClass::GEMM;
    for (int i = 0; i < iterations; ++i) {
        KernelGeom g = base_geom;
        g.fn = &fn_pool[i];          // unique address = guaranteed cache miss
        auto r = classify(g);
        sink = r.op;
    }
    auto t1 = Clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    printf("  Total: %.2f ms\n",  ns / 1e6);
    printf("  Per-call: %.2f ns  (target <100ns)\n\n", ns / iterations);
    (void)sink;
}

// ---------------------------------------------------------------------------
// Latency benchmark — cache-hit path (same ptr repeated)
// ---------------------------------------------------------------------------
static void bench_hot(int iterations = 10'000'000) {
    printf("=== Hot-path latency (N=%d, same fn ptr) ===\n", iterations);

    static uint8_t fn_sym;
    const KernelGeom g { 256, 256, 1, 16, 16, 1, 8192, &fn_sym };

    // Prime cache
    classify(g);

    auto t0 = Clock::now();
    volatile OpClass sink = OpClass::GEMM;
    for (int i = 0; i < iterations; ++i) {
        sink = classify(g).op;
    }
    auto t1 = Clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    printf("  Total: %.2f ms\n",  ns / 1e6);
    printf("  Per-call: %.2f ns  (target <5ns)\n\n", ns / iterations);
    (void)sink;
}

// ---------------------------------------------------------------------------
// Throughput: calls/sec at sustained load (simulates cuLaunchKernel rate)
// ---------------------------------------------------------------------------
static void bench_throughput() {
    printf("=== Throughput (mixed hot+cold, 5-class rotation) ===\n");

    static uint8_t fns[5];
    KernelGeom geoms[5] = {
        { 256, 256, 1,  16,  16, 1, 8192,  &fns[0] },  // GEMM
        { 2048, 32, 1, 128,   1, 1, 32768, &fns[1] },  // ATTN
        { 512*1024, 1,1,256,  1, 1, 0,     &fns[2] },  // EW
        { 512,   1, 1, 256,   1, 1, 2048,  &fns[3] },  // RED
        { 128, 128, 1,  32,  32, 1, 4096,  &fns[4] },  // MEMCPY
    };
    for (auto& g : geoms) classify(g);  // warm cache

    constexpr int N = 10'000'000;
    auto t0 = Clock::now();
    volatile int sink = 0;
    for (int i = 0; i < N; ++i) {
        sink += (int)classify(geoms[i % 5]).op;
    }
    auto t1 = Clock::now();

    double s = std::chrono::duration<double>(t1 - t0).count();
    printf("  %.0f M calls/sec  (%.2f ns avg)\n\n",
           N / s / 1e6, s / N * 1e9);
    (void)sink;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("\nCIPHER Classification Engine v0.1 — Neural Dynamics\n");
    printf("=====================================================\n\n");

    int failures = run_correctness();
    bench_cold();
    bench_hot();
    bench_throughput();

    if (failures == 0) {
        printf("All tests passed. Engine ships.\n\n");
    } else {
        printf("WARNING: %d test(s) failed.\n\n", failures);
    }
    return failures ? 1 : 0;
}
