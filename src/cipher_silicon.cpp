// CIPHER Silicon Model — implementation.

#include "cipher_silicon.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

CipherSiliconModel g_silicon{};
bool               g_init_attempted = false;

// Sparse FP16 TC peak per architecture, scaled by queried (sm_count / clock).
// CUDA does not expose tensor-core throughput; per-arch reference values are
// the documented NVIDIA peaks for the canonical chip in each family.
//   sm_90 (Hopper)  — 989 TFLOPS sparse @ 132 SMs / 1980 MHz (H100 SXM5)
//   sm_80 (Ampere)  — 312 TFLOPS sparse @ 108 SMs / 1410 MHz (A100)
//   sm_70 (Volta)   — 125 TFLOPS dense  @  80 SMs / 1530 MHz (V100, no sparsity)
double peak_fp16_tflops_for_arch(int major, int /*minor*/, int sm_count, int clock_mhz) {
    const double sm_d = static_cast<double>(sm_count);
    const double clk_d = static_cast<double>(clock_mhz);
    if (major >= 9) return 989.0 * (sm_d / 132.0) * (clk_d / 1980.0);
    if (major == 8) return 312.0 * (sm_d / 108.0) * (clk_d / 1410.0);
    if (major == 7) return 125.0 * (sm_d /  80.0) * (clk_d / 1530.0);
    return 0.0;
}

// Best-effort NVML power-cap query via dlopen. Returns watts or 0.0 on any
// failure (NVML missing, init failure, no device, etc.).
double query_power_cap_via_nvml() {
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) return 0.0;

    auto init_v2     = reinterpret_cast<int (*)(void)>(dlsym(lib, "nvmlInit_v2"));
    auto get_handle  = reinterpret_cast<int (*)(unsigned int, void**)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
    auto get_limit   = reinterpret_cast<int (*)(void*, unsigned int*)>(dlsym(lib, "nvmlDeviceGetPowerManagementLimit"));
    auto shutdown_fn = reinterpret_cast<int (*)(void)>(dlsym(lib, "nvmlShutdown"));

    if (!init_v2 || !get_handle || !get_limit) {
        dlclose(lib);
        return 0.0;
    }
    if (init_v2() != 0) {
        dlclose(lib);
        return 0.0;
    }

    double watts = 0.0;
    void* handle = nullptr;
    if (get_handle(0, &handle) == 0 && handle != nullptr) {
        unsigned int mw = 0;
        if (get_limit(handle, &mw) == 0) {
            watts = static_cast<double>(mw) / 1000.0;
        }
    }
    if (shutdown_fn) shutdown_fn();
    dlclose(lib);
    return watts;
}

} // namespace

extern "C" int cipher_silicon_init(void) {
    if (g_init_attempted) return g_silicon.initialized;
    g_init_attempted = true;

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) return 0;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0;

    g_silicon.sm_count             = prop.multiProcessorCount;
    g_silicon.compute_major        = prop.major;
    g_silicon.compute_minor        = prop.minor;
    g_silicon.clock_boost_mhz      = prop.clockRate / 1000;       // kHz → MHz
    g_silicon.memory_clock_mhz     = prop.memoryClockRate / 1000;
    g_silicon.memory_bus_width_bits = prop.memoryBusWidth;
    g_silicon.l2_total             = static_cast<size_t>(prop.l2CacheSize);
    g_silicon.l2_persist_max       = static_cast<size_t>(prop.persistingL2CacheMaxSize);
    g_silicon.l2_window_max        = static_cast<size_t>(prop.accessPolicyMaxWindowSize);
    g_silicon.hbm_total_bytes      = prop.totalGlobalMem;

    // HBM bandwidth: clock (Hz) * bus_width (bits) / 8 * 2 (DDR transfer rate)
    const double mem_clock_hz = static_cast<double>(prop.memoryClockRate) * 1000.0;
    g_silicon.hbm_bandwidth_bytes_per_sec =
        mem_clock_hz * static_cast<double>(prop.memoryBusWidth) / 8.0 * 2.0;

    g_silicon.peak_fp16_tflops = peak_fp16_tflops_for_arch(
        prop.major, prop.minor, prop.multiProcessorCount, g_silicon.clock_boost_mhz);

    g_silicon.fp16_intensity_threshold =
        (g_silicon.hbm_bandwidth_bytes_per_sec > 0.0)
            ? (g_silicon.peak_fp16_tflops * 1e12) / g_silicon.hbm_bandwidth_bytes_per_sec
            : 0.0;

    g_silicon.power_cap_watts = query_power_cap_via_nvml();

    g_silicon.mig_active = (prop.persistingL2CacheMaxSize == 0) ? 1 : 0;
    g_silicon.mps_active = (getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") != nullptr) ? 1 : 0;

    // Seed dynamic fields with sane defaults — THERMOSTAT will overwrite.
    g_silicon.clock_sustained_mhz.store(g_silicon.clock_boost_mhz, std::memory_order_relaxed);
    g_silicon.thermal_headroom.store(1.0,  std::memory_order_relaxed);
    g_silicon.power_draw_watts.store(0.0,  std::memory_order_relaxed);

    g_silicon.initialized = 1;

    if (getenv("CIPHER_SILICON_VERBOSE")) {
        fprintf(stderr,
            "[CIPHER SILICON] sm=%d cc=%d.%d clk_boost=%dMHz l2=%zuMB l2_persist=%zuMB "
            "l2_window=%zuMB hbm_bw=%.2fTB/s peak_fp16=%.0fTFLOPS thresh=%.0fFLOPS/B "
            "pcap=%.0fW mig=%d mps=%d\n",
            g_silicon.sm_count, g_silicon.compute_major, g_silicon.compute_minor,
            g_silicon.clock_boost_mhz,
            g_silicon.l2_total >> 20,
            g_silicon.l2_persist_max >> 20,
            g_silicon.l2_window_max >> 20,
            g_silicon.hbm_bandwidth_bytes_per_sec / 1e12,
            g_silicon.peak_fp16_tflops, g_silicon.fp16_intensity_threshold,
            g_silicon.power_cap_watts, g_silicon.mig_active, g_silicon.mps_active);
    }
    return 1;
}

extern "C" const CipherSiliconModel* cipher_silicon_get(void) {
    return g_silicon.initialized ? &g_silicon : nullptr;
}

extern "C" void cipher_silicon_update(int clock_mhz, double thermal_headroom, double power_w) {
    if (!g_silicon.initialized) return;
    g_silicon.clock_sustained_mhz.store(clock_mhz, std::memory_order_relaxed);
    g_silicon.thermal_headroom.store(thermal_headroom, std::memory_order_relaxed);
    g_silicon.power_draw_watts.store(power_w, std::memory_order_relaxed);
}

// Self-init when libcipher_rt.so loads. Priority 102 = after the hook DSO's
// constructor (priority 101). Idempotent with cipher_silicon_init() so the
// hook's dlsym call below cannot conflict.
__attribute__((constructor(102)))
static void cipher_silicon_autoinit() {
    cipher_silicon_init();
}
