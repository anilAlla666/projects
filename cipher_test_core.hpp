// cipher_test_core.hpp
// CIPHER C++ Test Core — Mac M4 / Apple Silicon + Linux x86 compatible
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cassert>
#include <string>
#include <pthread.h>

// ── Platform detection ────────────────────────────────────────────────────
#if defined(__APPLE__) && defined(__aarch64__)
  #define CIPHER_M4   1
  #define CIPHER_ARCH "Apple Silicon ARM64 (M-series)"
  #include <mach/mach_time.h>
  #include <sys/sysctl.h>
  #include <sys/mman.h>
#elif defined(__x86_64__)
  #define CIPHER_X86  1
  #define CIPHER_ARCH "x86_64"
  #include <immintrin.h>
#endif

// ── High-resolution timing ─────────────────────────────────────────────────
static inline uint64_t clock_ns() {
#ifdef CIPHER_M4
    static mach_timebase_info_data_t tb;
    static bool init = false;
    if (!init) { mach_timebase_info(&tb); init = true; }
    return mach_absolute_time() * tb.numer / tb.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

// ── Memory barriers ────────────────────────────────────────────────────────
static inline void sfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
static inline void lfence() { std::atomic_thread_fence(std::memory_order_acquire); }
static inline void mfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }

// ── Cache flush ────────────────────────────────────────────────────────────
// ARM macOS: DC CIVAC is privileged — cannot flush from userspace.
// Cold-path tests on M4 use fresh mmap allocations instead.
static inline void cache_flush(const void* ptr, size_t size) {
#ifdef CIPHER_X86
    const char* p = (const char*)ptr;
    for (size_t i = 0; i < size; i += 64)
        __asm__ volatile("clflush (%0)" :: "r"(p+i) : "memory");
    __asm__ volatile("mfence" ::: "memory");
#else
    (void)ptr; (void)size;
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// ── Cache line size ────────────────────────────────────────────────────────
// M4: 128-byte cache lines | x86: 64-byte cache lines
#ifdef CIPHER_M4
  #define CACHE_LINE_SIZE 128
#else
  #define CACHE_LINE_SIZE 64
#endif

#define ALIGN_128 alignas(128)
#define ALIGN_64  alignas(64)

// ── Thread pinning ────────────────────────────────────────────────────────
static bool pin_thread_to_core(int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(),
                                   sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(CIPHER_M4)
    (void)core_id;
    // macOS: set highest QoS class for performance cores
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    return true;
#else
    (void)core_id;
    return false;
#endif
}

// ── System info ────────────────────────────────────────────────────────────
static void print_system_info() {
    printf("  Architecture:  %s\n", CIPHER_ARCH);
#ifdef CIPHER_M4
    int ncpu = 0; size_t len = sizeof(ncpu);
    sysctlbyname("hw.logicalcpu", &ncpu, &len, nullptr, 0);
    int64_t cls = 0; len = sizeof(cls);
    sysctlbyname("hw.cachelinesize", &cls, &len, nullptr, 0);
    printf("  CPU cores:     %d\n", ncpu);
    printf("  Cache line:    %lld bytes (M4 has 128-byte lines)\n", cls);
    printf("  SHA:           ARM SHA2 crypto extensions (hardware)\n");
    printf("  Atomics:       LDADD/STLR/LDAR — ARM64 TSO-like\n");
    printf("  Cache flush:   Not available from userspace (ARM)\n");
    printf("                 Cold-path test uses fresh mmap allocations\n");
#else
    printf("  Cache line:    64 bytes\n");
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char buf[4096]; bool sha = false, avx = false;
        while (fgets(buf, sizeof(buf), f)) {
            if (strstr(buf, "sha_ni")) sha = true;
            if (strstr(buf, "avx2")) avx = true;
        }
        fclose(f);
        printf("  SHA-NI:        %s\n", sha ? "✅ yes" : "⚠️  no");
        printf("  AVX2:          %s\n", avx ? "✅ yes" : "no");
    }
#endif
}

// ── Percentile calculator ──────────────────────────────────────────────────
struct Latencies {
    std::vector<uint64_t> samples;

    void record(uint64_t ns) { samples.push_back(ns); }
    void sort_samples() { std::sort(samples.begin(), samples.end()); }
    size_t count() { return samples.size(); }

    uint64_t percentile(double p) {
        if (samples.empty()) return 0;
        size_t idx = (size_t)(p / 100.0 * (double)(samples.size()-1));
        return samples[idx];
    }

    uint64_t mean() {
        if (samples.empty()) return 0;
        uint64_t s = 0;
        for (auto x : samples) s += x;
        return s / samples.size();
    }

    void print(const char* label) {
        sort_samples();
        printf("  %-32s mean=%4lluns  p50=%4lluns  "
               "p90=%4lluns  p99=%4lluns  p999=%5lluns\n",
               label,
               (unsigned long long)mean(),
               (unsigned long long)percentile(50),
               (unsigned long long)percentile(90),
               (unsigned long long)percentile(99),
               (unsigned long long)percentile(99.9));
    }
};

// ── Helpers ────────────────────────────────────────────────────────────────
static void print_result(const char* name, bool p, const char* note="") {
    printf("  %s  %s  %s\n", p ? "✅ PASS" : "❌ FAIL", name, note);
}

static void print_separator(const char* title) {
    printf("\n%s\n%s\n%s\n",
           "═══════════════════════════════════════════════════════════",
           title,
           "═══════════════════════════════════════════════════════════");
}
