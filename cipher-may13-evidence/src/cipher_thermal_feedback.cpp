// CIPHER Thermal-Substitution Feedback — implementation.

#include "cipher_thermal_feedback.h"
#include "cipher_silicon.h"

#include <atomic>
#include <thread>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

namespace {

constexpr double STEP_UP   = 0.05;       // increase per tick when needed
constexpr double STEP_DOWN = 0.02;       // decrease per tick when relaxed
constexpr double LOW_HEADROOM  = 0.25;
constexpr double HIGH_HEADROOM = 0.75;
constexpr double MAX_OSCILLATION_DELTA = 0.20;  // clamp single-tick swings

std::atomic<int>       g_enabled{0};
std::atomic<int>       g_initialized{0};
std::atomic<double>    g_aggressiveness{0.0};
std::atomic<double>    g_last_clock_mhz{0.0};
std::atomic<double>    g_last_headroom{1.0};
std::atomic<double>    g_last_power_w{0.0};
std::atomic<uint64_t>  g_tick_calls{0};
std::atomic<uint64_t>  g_inc{0};
std::atomic<uint64_t>  g_dec{0};
std::atomic<uint64_t>  g_clamps{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T'
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

} // namespace

extern "C" int cipher_thermal_feedback_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_THERMAL_FEEDBACK"));
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        fprintf(stderr,
            "[CIPHER THFB] init step_up=%.2f step_down=%.2f low=%.2f high=%.2f\n",
            STEP_UP, STEP_DOWN, LOW_HEADROOM, HIGH_HEADROOM);
    }
    return on;
}

extern "C" int cipher_thermal_feedback_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" void cipher_thermal_feedback_tick(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    g_tick_calls.fetch_add(1, std::memory_order_relaxed);

    const CipherSiliconModel* sil = cipher_silicon_get();
    if (!sil) return;

    int    clk    = sil->clock_sustained_mhz.load(std::memory_order_relaxed);
    double headr  = sil->thermal_headroom.load(std::memory_order_relaxed);
    double pwr    = sil->power_draw_watts.load(std::memory_order_relaxed);

    g_last_clock_mhz.store((double)clk, std::memory_order_relaxed);
    g_last_headroom.store(headr,         std::memory_order_relaxed);
    g_last_power_w.store(pwr,            std::memory_order_relaxed);

    double cur = g_aggressiveness.load(std::memory_order_relaxed);
    double target = cur;
    if (headr < LOW_HEADROOM) {
        target = clamp01(cur + STEP_UP);
        g_inc.fetch_add(1, std::memory_order_relaxed);
    } else if (headr > HIGH_HEADROOM) {
        target = clamp01(cur - STEP_DOWN);
        g_dec.fetch_add(1, std::memory_order_relaxed);
    }
    if (fabs(target - cur) > MAX_OSCILLATION_DELTA) {
        if (target > cur) target = cur + MAX_OSCILLATION_DELTA;
        else              target = cur - MAX_OSCILLATION_DELTA;
        g_clamps.fetch_add(1, std::memory_order_relaxed);
    }
    g_aggressiveness.store(target, std::memory_order_relaxed);
}

extern "C" double cipher_thermal_feedback_aggressiveness(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0.0;
    return g_aggressiveness.load(std::memory_order_relaxed);
}

extern "C" int cipher_thermal_feedback_stats(CipherThermalFeedbackStats* out) {
    if (!out) return 0;
    out->enabled                    = g_enabled.load(std::memory_order_relaxed);
    out->substitute_aggressiveness  = g_aggressiveness.load(std::memory_order_relaxed);
    out->last_clock_mhz             = g_last_clock_mhz.load(std::memory_order_relaxed);
    out->last_headroom              = g_last_headroom.load(std::memory_order_relaxed);
    out->last_power_w               = g_last_power_w.load(std::memory_order_relaxed);
    out->tick_calls                 = g_tick_calls.load(std::memory_order_relaxed);
    out->aggressiveness_increased   = g_inc.load(std::memory_order_relaxed);
    out->aggressiveness_decreased   = g_dec.load(std::memory_order_relaxed);
    out->oscillation_clamps         = g_clamps.load(std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_thermal_feedback_report(void) {
    CipherThermalFeedbackStats s{};
    cipher_thermal_feedback_stats(&s);
    FILE* f = fopen("/tmp/cipher_thermal_feedback_report.json", "w");
    if (!f) return;
    fprintf(f,
        "{\"enabled\":%d,\"substitute_aggressiveness\":%.6f,"
        "\"last_clock_mhz\":%.1f,\"last_headroom\":%.4f,\"last_power_w\":%.1f,"
        "\"tick_calls\":%llu,\"aggressiveness_increased\":%llu,"
        "\"aggressiveness_decreased\":%llu,\"oscillation_clamps\":%llu}\n",
        s.enabled, s.substitute_aggressiveness,
        s.last_clock_mhz, s.last_headroom, s.last_power_w,
        (unsigned long long)s.tick_calls,
        (unsigned long long)s.aggressiveness_increased,
        (unsigned long long)s.aggressiveness_decreased,
        (unsigned long long)s.oscillation_clamps);
    fclose(f);
}

// ── Stage 11 actuation: NVML sampler thread ─────────────────────────────────
// Reads clocks.gr, power.draw via dlopen'd libnvidia-ml.so.1 every 100 ms
// and writes the values (plus a derived thermal_headroom) into the silicon
// model's atomic dynamic fields. cipher_thermal_feedback_tick is called from
// the same thread to keep the substitution-aggressiveness loop alive.
namespace {
typedef int (*nvml_init_fn)(void);
typedef int (*nvml_handle_fn)(unsigned, void**);
typedef int (*nvml_clock_fn)(void*, int /*type*/, unsigned*);
typedef int (*nvml_power_fn)(void*, unsigned*);
typedef int (*nvml_temp_fn)(void*, int, unsigned*);
typedef int (*nvml_shutdown_fn)(void);
typedef int (*nvml_set_locked_clocks_fn)(void*, unsigned /*minMHz*/,
                                          unsigned /*maxMHz*/);

std::thread*          g_sampler_thread = nullptr;
std::atomic<int>      g_sampler_run{0};

// C5 — adaptive DVFS via thermal feedback.
// When CIPHER_DVFS=on, the sampler thread takes over GPU clock control:
// drops the clock when headroom is high (decode is HBM-bandwidth-bound,
// SM cycles are wasted), raises it when headroom is low (compute-bound).
// Default OFF — when off, the sampler is read-only, so the cipher path
// is bit-identical to pre-C5 and cannot regress any T1–X test that
// passed under the user's static `nvidia-smi -lgc` setting.
//
// Hysteresis bands prevent thrash. Updates throttled to once / 5 s.
// Bounds clamped to [DVFS_MIN_MHZ, DVFS_MAX_MHZ] (default 900 / 1800)
// — both env-tunable. Failure modes (NVML returns nonzero, no privilege)
// are silent: the loop keeps sampling, the clock stays where it was.
struct DvfsState {
    bool   enabled        = false;
    int    cur_target_mhz = 0;
    int    min_mhz        = 900;
    int    max_mhz        = 1800;
    double last_set_at_s  = 0.0;
    nvml_set_locked_clocks_fn set_clocks = nullptr;
};

void sampler_main() {
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) return;
    auto init   = (nvml_init_fn)dlsym(lib, "nvmlInit_v2");
    auto getH   = (nvml_handle_fn)dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
    auto getCl  = (nvml_clock_fn)dlsym(lib, "nvmlDeviceGetClockInfo");
    auto getPw  = (nvml_power_fn)dlsym(lib, "nvmlDeviceGetPowerUsage");
    auto shut   = (nvml_shutdown_fn)dlsym(lib, "nvmlShutdown");
    auto setCl  = (nvml_set_locked_clocks_fn)dlsym(lib,
                       "nvmlDeviceSetGpuLockedClocks");
    if (!init || !getH || !getCl || !getPw) { dlclose(lib); return; }
    if (init() != 0) { dlclose(lib); return; }
    void* h = nullptr;
    if (getH(0, &h) != 0 || !h) { if (shut) shut(); dlclose(lib); return; }

    const CipherSiliconModel* sil = cipher_silicon_get();
    double cap = sil ? sil->power_cap_watts : 700.0;
    if (cap <= 0.0) cap = 700.0;
    int boost = sil ? sil->clock_boost_mhz : 1980;
    if (boost <= 0) boost = 1980;

    DvfsState dvfs;
    {
        const char* env = getenv("CIPHER_DVFS");
        dvfs.enabled = (env && *env && env[0] != '0');
        if (const char* m = getenv("CIPHER_DVFS_MIN_MHZ")) {
            int v = atoi(m); if (v > 0) dvfs.min_mhz = v;
        }
        if (const char* m = getenv("CIPHER_DVFS_MAX_MHZ")) {
            int v = atoi(m); if (v > 0) dvfs.max_mhz = v;
        }
        dvfs.set_clocks = setCl;
        if (dvfs.enabled && !setCl) {
            fprintf(stderr,
                "[CIPHER DVFS] requested but nvmlDeviceSetGpuLockedClocks "
                "not resolved — disabled.\n");
            dvfs.enabled = false;
        }
        if (dvfs.enabled) {
            fprintf(stderr,
                "[CIPHER DVFS] enabled, range=[%d,%d] MHz\n",
                dvfs.min_mhz, dvfs.max_mhz);
        }
    }

    auto now_s = []() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    while (g_sampler_run.load(std::memory_order_relaxed)) {
        unsigned clk_mhz = 0, pwr_mw = 0;
        if (getCl(h, /*NVML_CLOCK_GRAPHICS=*/0, &clk_mhz) == 0
            && getPw(h, &pwr_mw) == 0) {
            double pwr_w = pwr_mw / 1000.0;
            double headroom = 1.0 - (pwr_w / cap);
            if (headroom < 0.0) headroom = 0.0;
            if (headroom > 1.0) headroom = 1.0;
            cipher_silicon_update((int)clk_mhz, headroom, pwr_w);
            cipher_thermal_feedback_tick();

            // C5 actuation — only if explicitly enabled.
            if (dvfs.enabled && dvfs.set_clocks) {
                double tnow = now_s();
                if (tnow - dvfs.last_set_at_s >= 5.0) {
                    // Heuristic: if we're using <50% of power budget,
                    // SMs are HBM-stalled. Drop clock 100 MHz. If we're
                    // using >85%, push back up by 100 MHz. Otherwise hold.
                    int target = dvfs.cur_target_mhz > 0
                                 ? dvfs.cur_target_mhz : (int)clk_mhz;
                    double util = pwr_w / cap;
                    if (util < 0.50) target -= 100;
                    else if (util > 0.85) target += 100;
                    if (target < dvfs.min_mhz) target = dvfs.min_mhz;
                    if (target > dvfs.max_mhz) target = dvfs.max_mhz;
                    if (target != dvfs.cur_target_mhz) {
                        unsigned mn = (unsigned)target;
                        unsigned mx = (unsigned)target;
                        if (dvfs.set_clocks(h, mn, mx) == 0) {
                            dvfs.cur_target_mhz = target;
                            fprintf(stderr,
                                "[CIPHER DVFS] -> %d MHz "
                                "(headroom=%.2f, util=%.2f)\n",
                                target, headroom, util);
                        }
                    }
                    dvfs.last_set_at_s = tnow;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (shut) shut();
    dlclose(lib);
}
} // namespace

__attribute__((constructor(111)))
static void cipher_thermal_feedback_autoinit() {
    cipher_thermal_feedback_init();
    if (cipher_thermal_feedback_enabled() && !g_sampler_thread) {
        g_sampler_run.store(1, std::memory_order_release);
        g_sampler_thread = new std::thread(sampler_main);
    }
}

__attribute__((destructor))
static void cipher_thermal_feedback_teardown() {
    g_sampler_run.store(0, std::memory_order_release);
    if (g_sampler_thread && g_sampler_thread->joinable()) {
        g_sampler_thread->join();
        delete g_sampler_thread;
        g_sampler_thread = nullptr;
    }
}
