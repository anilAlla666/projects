// Op 31 HIBERNATE — execution-idle SM power gating (Path B).
//
// Stage 1: per-event last-dispatch-timestamp tracking + pre-engage notify.
// Stage 2: periodic poll evaluates idle gap and (on permissioned hosts)
//          calls nvmlDeviceSetPowerManagementLimit() to drop floor power.
// Pod-degraded: probe returns NOT_SUPPORTED → power-limit writes skipped,
//               idle detection still observable via counters.

#include "cipher_hibernate.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>

namespace {

constexpr int  NVML_SUCCESS             = 0;
constexpr int  NVML_ERROR_NOT_SUPPORTED = 4;

constexpr uint64_t POLL_PERIOD_NS = 10ULL * 1000 * 1000;        // 10 ms cadence
constexpr unsigned DEFAULT_THRESHOLD_MS = 5;

typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;
typedef nvmlReturn_t (*fn_GetHandle_t)(uint32_t, nvmlDevice_t*);
typedef nvmlReturn_t (*fn_GetPL_t)(nvmlDevice_t, unsigned*);
typedef nvmlReturn_t (*fn_GetPLConstraints_t)(nvmlDevice_t, unsigned*, unsigned*);
typedef nvmlReturn_t (*fn_SetPL_t)(nvmlDevice_t, unsigned);

struct HibState {
    std::atomic<int>      enabled;
    std::atomic<int>      actuation_supported;     // 0/1
    std::atomic<int>      idle_gate_flag;          // current gate state
    std::atomic<unsigned> idle_event_count;
    std::atomic<unsigned> pre_engage_count;
    std::atomic<uint64_t> last_dispatch_ns;
    std::atomic<uint64_t> inject_idle_ns;          // test override
    uint64_t              last_poll_ns;
    unsigned              threshold_ns_div_1e6;    // configured threshold (ms)
    unsigned              base_power_mw;
    unsigned              reduced_power_mw;
    unsigned              constraint_lo_mw;
    unsigned              constraint_hi_mw;

    // CACHED NVML pointers (no dlsym in signal handlers)
    void*                 nvml_handle;
    nvmlDevice_t          nvml_device;
    fn_SetPL_t            fn_set_pl;
    fn_GetPL_t            fn_get_pl;
    fn_GetPLConstraints_t fn_get_constraints;
};

HibState g_hib = {
    {0}, {-1}, {0}, {0u}, {0u}, {0}, {0}, 0, DEFAULT_THRESHOLD_MS,
    0, 0, 0, 0,
    nullptr, nullptr, nullptr, nullptr, nullptr
};

std::atomic<int> g_hib_initialized{0};

bool resolve_nvml() {
    if (g_hib.nvml_handle) return g_hib.fn_set_pl != nullptr;
    void* h = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libnvidia-ml.so",   RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libnvidia-ml.so",   RTLD_LAZY);
    if (!h) return false;
    typedef int (*fn_init_t)(void);
    if (auto fn = (fn_init_t)dlsym(h, "nvmlInit_v2")) fn();
    g_hib.nvml_handle = h;
    auto fn_handle = (fn_GetHandle_t)dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
    g_hib.fn_get_pl          = (fn_GetPL_t)           dlsym(h, "nvmlDeviceGetPowerManagementLimit");
    g_hib.fn_get_constraints = (fn_GetPLConstraints_t)dlsym(h, "nvmlDeviceGetPowerManagementLimitConstraints");
    g_hib.fn_set_pl          = (fn_SetPL_t)           dlsym(h, "nvmlDeviceSetPowerManagementLimit");
    if (!fn_handle || !g_hib.fn_set_pl) return false;
    if (fn_handle(0, &g_hib.nvml_device) != NVML_SUCCESS) return false;
    if (g_hib.fn_get_pl) g_hib.fn_get_pl(g_hib.nvml_device, &g_hib.base_power_mw);
    if (g_hib.fn_get_constraints)
        g_hib.fn_get_constraints(g_hib.nvml_device,
                                 &g_hib.constraint_lo_mw,
                                 &g_hib.constraint_hi_mw);
    // Reduced target = 30% of TDP (per spec), clamped to constraint range
    g_hib.reduced_power_mw = g_hib.base_power_mw * 30 / 100;
    if (g_hib.constraint_lo_mw && g_hib.reduced_power_mw < g_hib.constraint_lo_mw)
        g_hib.reduced_power_mw = g_hib.constraint_lo_mw;
    return true;
}

// No-op write probe: write current value back. Returns NVML rc.
int probe_actuation() {
    if (!resolve_nvml() || !g_hib.fn_set_pl || !g_hib.fn_get_pl) return NVML_ERROR_NOT_SUPPORTED;
    unsigned curr = 0;
    if (g_hib.fn_get_pl(g_hib.nvml_device, &curr) != NVML_SUCCESS)
        return NVML_ERROR_NOT_SUPPORTED;
    return g_hib.fn_set_pl(g_hib.nvml_device, curr);
}

// Signal handler restoration — async-signal-safe (cached pointers only).
extern "C" void hib_atexit_restore() {
    if (g_hib.actuation_supported.load(std::memory_order_acquire) != 1) return;
    if (g_hib.fn_set_pl && g_hib.nvml_device && g_hib.base_power_mw)
        g_hib.fn_set_pl(g_hib.nvml_device, g_hib.base_power_mw);
}

extern "C" void hib_signal_handler(int sig, siginfo_t* /*info*/, void* /*ctx*/) {
    if (g_hib.actuation_supported.load(std::memory_order_relaxed) == 1
        && g_hib.fn_set_pl && g_hib.nvml_device && g_hib.base_power_mw) {
        g_hib.fn_set_pl(g_hib.nvml_device, g_hib.base_power_mw);
    }
    struct sigaction sa = {};
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, nullptr);
    raise(sig);
}

void install_handlers_once() {
    static std::atomic<int> installed{0};
    if (installed.exchange(1, std::memory_order_acq_rel)) return;
    struct sigaction sa = {};
    sa.sa_sigaction = hib_signal_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    int sigs[] = {SIGTERM, SIGINT, SIGSEGV, SIGABRT, SIGBUS};
    for (int s : sigs) sigaction(s, &sa, nullptr);
    std::atexit(hib_atexit_restore);
}

inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void poll_impl(bool force) {
    if (!g_hib.enabled.load(std::memory_order_relaxed)) return;

    uint64_t now = now_ns();
    if (!force && g_hib.last_poll_ns != 0
        && (now - g_hib.last_poll_ns) < POLL_PERIOD_NS) return;
    g_hib.last_poll_ns = now;

    uint64_t inject = g_hib.inject_idle_ns.load(std::memory_order_relaxed);
    uint64_t last   = g_hib.last_dispatch_ns.load(std::memory_order_relaxed);
    uint64_t gap    = inject ? inject : (last ? (now - last) : 0);
    uint64_t threshold_ns = (uint64_t)g_hib.threshold_ns_div_1e6 * 1000000ULL;

    bool was_idle = g_hib.idle_gate_flag.load(std::memory_order_relaxed) != 0;
    bool is_idle  = (gap > threshold_ns);

    if (is_idle && !was_idle) {
        g_hib.idle_gate_flag.store(1, std::memory_order_release);
        g_hib.idle_event_count.fetch_add(1, std::memory_order_relaxed);
        if (g_hib.actuation_supported.load(std::memory_order_relaxed)
            && g_hib.fn_set_pl && g_hib.reduced_power_mw) {
            install_handlers_once();
            g_hib.fn_set_pl(g_hib.nvml_device, g_hib.reduced_power_mw);
        }
    } else if (!is_idle && was_idle) {
        g_hib.idle_gate_flag.store(0, std::memory_order_release);
        if (g_hib.actuation_supported.load(std::memory_order_relaxed)
            && g_hib.fn_set_pl && g_hib.base_power_mw) {
            g_hib.fn_set_pl(g_hib.nvml_device, g_hib.base_power_mw);
        }
    }
}

} // namespace

extern "C" int cipher_hibernate_init(void) {
    int already = g_hib_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_hib.enabled.load(std::memory_order_relaxed);

    const char* env = std::getenv("CIPHER_HIBERNATE");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_hib.enabled.store(on, std::memory_order_release);
    if (!on) return 0;

    const char* tenv = std::getenv("CIPHER_HIBERNATE_THRESHOLD_MS");
    if (tenv) {
        unsigned v = (unsigned)std::strtoul(tenv, nullptr, 10);
        if (v >= 1 && v <= 1000) g_hib.threshold_ns_div_1e6 = v;
    }

    int rc = probe_actuation();
    bool supported = (rc == NVML_SUCCESS);
    g_hib.actuation_supported.store(supported ? 1 : 0, std::memory_order_release);
    std::fprintf(stderr,
        "[CIPHER Op31] HIBERNATE %s — threshold=%u ms, base_power=%u mW, "
        "reduced_target=%u mW, constraints[%u..%u] mW (probe rc=%d %s).\n",
        supported ? "ACTIVE" : "DEGRADED",
        g_hib.threshold_ns_div_1e6,
        g_hib.base_power_mw, g_hib.reduced_power_mw,
        g_hib.constraint_lo_mw, g_hib.constraint_hi_mw,
        rc, supported ? "OK" : "NVML_ERROR_NOT_SUPPORTED");
    return 1;
}

extern "C" void cipher_hibernate_observe(const CipherRingEntry* ev) {
    if (!g_hib.enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;

    uint64_t now = ev->timestamp_ns;
    if (now == 0) now = now_ns();
    g_hib.last_dispatch_ns.store(now, std::memory_order_relaxed);

    // Pre-engage: if gate was raised, drop it BEFORE actuation hits next time.
    if (g_hib.idle_gate_flag.exchange(0, std::memory_order_release) != 0) {
        g_hib.pre_engage_count.fetch_add(1, std::memory_order_relaxed);
        if (g_hib.actuation_supported.load(std::memory_order_relaxed)
            && g_hib.fn_set_pl && g_hib.base_power_mw) {
            g_hib.fn_set_pl(g_hib.nvml_device, g_hib.base_power_mw);
        }
    }
}

extern "C" void cipher_hibernate_poll(void)        { poll_impl(false); }
extern "C" void cipher_hibernate_force_poll(void)  { poll_impl(true); }

extern "C" int cipher_hibernate_actuation_supported(void) {
    return g_hib.actuation_supported.load(std::memory_order_acquire) == 1;
}
extern "C" int cipher_hibernate_idle_gate_flag(void) {
    return g_hib.idle_gate_flag.load(std::memory_order_acquire);
}
extern "C" unsigned cipher_hibernate_idle_event_count(void) {
    return g_hib.idle_event_count.load(std::memory_order_relaxed);
}
extern "C" unsigned cipher_hibernate_pre_engage_count(void) {
    return g_hib.pre_engage_count.load(std::memory_order_relaxed);
}
extern "C" const char* cipher_hibernate_status_string(void) {
    if (!g_hib.enabled.load(std::memory_order_relaxed)) return "OFF";
    return g_hib.actuation_supported.load(std::memory_order_relaxed) ? "ACTIVE" : "DEGRADED";
}

extern "C" void cipher_hibernate_inject_idle_gap_ms(unsigned ms) {
    g_hib.inject_idle_ns.store((uint64_t)ms * 1000000ULL, std::memory_order_release);
}
extern "C" void cipher_hibernate_clear_injection(void) {
    g_hib.inject_idle_ns.store(0, std::memory_order_release);
}

extern "C" void cipher_hibernate_report(void) {
    if (!g_hib.enabled.load(std::memory_order_relaxed)) return;
    FILE* fp = std::fopen("/tmp/cipher_hibernate_report.json", "w");
    if (!fp) return;
    std::fprintf(fp,
        "{\n"
        "  \"status\": \"%s\",\n"
        "  \"actuation_supported\": %s,\n"
        "  \"threshold_ms\": %u,\n"
        "  \"base_power_mw\": %u,\n"
        "  \"reduced_power_mw\": %u,\n"
        "  \"constraint_lo_mw\": %u,\n"
        "  \"constraint_hi_mw\": %u,\n"
        "  \"idle_gate_flag\": %d,\n"
        "  \"idle_event_count\": %u,\n"
        "  \"pre_engage_count\": %u\n"
        "}\n",
        cipher_hibernate_status_string(),
        cipher_hibernate_actuation_supported() ? "true" : "false",
        g_hib.threshold_ns_div_1e6,
        g_hib.base_power_mw, g_hib.reduced_power_mw,
        g_hib.constraint_lo_mw, g_hib.constraint_hi_mw,
        g_hib.idle_gate_flag.load(std::memory_order_relaxed),
        g_hib.idle_event_count.load(std::memory_order_relaxed),
        g_hib.pre_engage_count.load(std::memory_order_relaxed));
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op31] HIBERNATE report -> /tmp/cipher_hibernate_report.json\n");
}
