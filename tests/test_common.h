
#ifndef TEST_COMMON_H
#define TEST_COMMON_H
#include <cstdio>
#include <chrono>

extern int g_pass, g_fail;

#define TEST_START(name) printf("\n━━━ %s ━━━\n", name)
#define CHECK(cond, msg, ...) do { \
    if (cond) { g_pass++; printf("  ✅ " msg "\n", ##__VA_ARGS__); } \
    else { g_fail++; printf("  ❌ " msg "\n", ##__VA_ARGS__); } \
} while(0)
#define CHECK_RANGE(val, lo, hi, name) \
    CHECK((val) >= (lo) && (val) <= (hi), "%s = %.4f [%.1f, %.1f]", name, (float)(val), (float)(lo), (float)(hi))

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double elapsed_us() {
        return std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
};
#endif
