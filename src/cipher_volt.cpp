// Op 30 VOLT — Arithmetic-intensity-aware SM frequency steering.
//
// Path B "production-ready, pod-degraded": full code path including
// atexit + signal handlers and frequency restoration. On a pod where
// nvmlDeviceSetGpuLockedClocks returns NOT_SUPPORTED, we degrade to
// classifier-only mode after a startup probe; the same code activates
// real frequency steering on a permissioned host.

#include "cipher_volt.h"
#include "cipher_thermostat.h"   // for thermostat override stub

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>

namespace {

// NVML constants we use without #include'ing nvml.h.
constexpr int  NVML_SUCCESS               = 0;
constexpr int  NVML_ERROR_NOT_SUPPORTED   = 4;

constexpr unsigned MAX_SHAPES = 64;
constexpr uint64_t ACTUATE_PERIOD_NS = 100ULL * 1000 * 1000;   // 100 ms

constexpr const char* CALIB_PATH = "/tmp/cipher_volt_calibration.json";

typedef int  nvmlReturn_t;
typedef void* nvmlDevice_t;
typedef nvmlReturn_t (*fn_GetHandle_t)(uint32_t, nvmlDevice_t*);
typedef nvmlReturn_t (*fn_GetDriverVer_t)(char*, unsigned);
typedef nvmlReturn_t (*fn_GetClockInfo_t)(nvmlDevice_t, int, uint32_t*);
typedef nvmlReturn_t (*fn_SetLockedClocks_t)(nvmlDevice_t, unsigned, unsigned);
typedef nvmlReturn_t (*fn_ResetLockedClocks_t)(nvmlDevice_t);
typedef nvmlReturn_t (*fn_GetVfOffset_t)(nvmlDevice_t, int*);
typedef nvmlReturn_t (*fn_SetVfOffset_t)(nvmlDevice_t, int);

struct VoltShape {
    std::atomic<uint64_t>  key;        // 0 = empty
    uint32_t               M, K, N;
    CipherShapeClass       cls;        // computed from AI
    double                 ai;
};

alignas(64) VoltShape g_shapes[MAX_SHAPES];

struct VoltState {
    std::atomic<int>      mode;                // CipherVoltMode
    std::atomic<int>      actuation_supported; // 0/1; -1 = unprobed
    std::atomic<int>      offset_applied_khz;  // 0 = baseline (no offset)
    std::atomic<unsigned> reduce_events;
    std::atomic<unsigned> restore_events;
    uint64_t              last_actuate_ns;

    // CACHED NVML pointers — must be loaded at init, never inside signal handlers.
    void*                 nvml_handle;       // RTLD_NOLOAD; do not dlclose
    nvmlDevice_t          nvml_device;
    fn_GetHandle_t        fn_get_handle;
    fn_GetDriverVer_t     fn_get_driver;
    fn_GetClockInfo_t     fn_get_clock;
    fn_SetLockedClocks_t  fn_set_locked;
    fn_ResetLockedClocks_t fn_reset_locked;
    fn_GetVfOffset_t      fn_get_vf;
    fn_SetVfOffset_t      fn_set_vf;
    char                  driver_version[64];
    uint32_t              base_clock_mhz;

    int                   inject_supported; // -1 = no injection; else forces value
};

VoltState g_volt = {
    {CIPHER_VOLT_OFF}, {-1}, {0}, {0u}, {0u}, 0,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    {0}, 0u, -1
};

std::atomic<int> g_volt_initialized{0};

// ── NVML resolution + probe ──────────────────────────────────────────────────

bool resolve_nvml() {
    if (g_volt.nvml_handle) return g_volt.fn_set_locked != nullptr;
    // Prefer RTLD_NOLOAD (re-use the handle telemetry already opened); fall
    // back to RTLD_LAZY if telemetry hasn't initialised yet. NVML's
    // nvmlInit_v2 is idempotent across multiple callers.
    void* h = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libnvidia-ml.so",   RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libnvidia-ml.so",   RTLD_LAZY);
    if (!h) return false;
    // If we did the first real dlopen, ensure NVML is initialised.
    typedef int (*fn_init_t)(void);
    fn_init_t fn_init = (fn_init_t)dlsym(h, "nvmlInit_v2");
    if (fn_init) fn_init();   // idempotent
    g_volt.nvml_handle    = h;
    g_volt.fn_get_handle  = (fn_GetHandle_t)         dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
    g_volt.fn_get_driver  = (fn_GetDriverVer_t)      dlsym(h, "nvmlSystemGetDriverVersion");
    g_volt.fn_get_clock   = (fn_GetClockInfo_t)      dlsym(h, "nvmlDeviceGetClockInfo");
    g_volt.fn_set_locked  = (fn_SetLockedClocks_t)   dlsym(h, "nvmlDeviceSetGpuLockedClocks");
    g_volt.fn_reset_locked= (fn_ResetLockedClocks_t) dlsym(h, "nvmlDeviceResetGpuLockedClocks");
    g_volt.fn_get_vf      = (fn_GetVfOffset_t)       dlsym(h, "nvmlDeviceGetGpcClkVfOffset");
    g_volt.fn_set_vf      = (fn_SetVfOffset_t)       dlsym(h, "nvmlDeviceSetGpcClkVfOffset");
    if (!g_volt.fn_get_handle) return false;
    if (g_volt.fn_get_handle(0, &g_volt.nvml_device) != NVML_SUCCESS) return false;
    if (g_volt.fn_get_driver) g_volt.fn_get_driver(g_volt.driver_version,
                                                   sizeof(g_volt.driver_version));
    if (g_volt.fn_get_clock) {
        uint32_t mhz = 0;
        // 0 == NVML_CLOCK_SM
        if (g_volt.fn_get_clock(g_volt.nvml_device, 0, &mhz) == NVML_SUCCESS)
            g_volt.base_clock_mhz = mhz;
    }
    return true;
}

// Probe write permission with a no-op SetGpuLockedClocks(current, current).
// Returns: rc from NVML; 0 = supported, anything else = not supported.
int probe_actuation() {
    if (g_volt.inject_supported >= 0)
        return g_volt.inject_supported ? NVML_SUCCESS : NVML_ERROR_NOT_SUPPORTED;
    if (!resolve_nvml() || !g_volt.fn_set_locked || !g_volt.fn_reset_locked)
        return NVML_ERROR_NOT_SUPPORTED;
    uint32_t cur = g_volt.base_clock_mhz ? g_volt.base_clock_mhz : 1980;
    int r = g_volt.fn_set_locked(g_volt.nvml_device, cur, cur);
    if (r == NVML_SUCCESS) g_volt.fn_reset_locked(g_volt.nvml_device);
    return r;
}

// ── Signal/atexit restoration ────────────────────────────────────────────────

extern "C" void volt_atexit_restore() {
    if (g_volt.actuation_supported.load(std::memory_order_acquire) != 1) return;
    if (g_volt.offset_applied_khz.load(std::memory_order_acquire) == 0) return;
    if (!g_volt.fn_set_vf || !g_volt.nvml_device) return;
    g_volt.fn_set_vf(g_volt.nvml_device, 0);   // baseline
    g_volt.offset_applied_khz.store(0, std::memory_order_release);
}

extern "C" void volt_signal_handler(int sig, siginfo_t* /*info*/, void* /*ctx*/) {
    // Async-signal-safe path: cached function pointer + atomic state only.
    if (g_volt.actuation_supported.load(std::memory_order_relaxed) == 1
        && g_volt.offset_applied_khz.load(std::memory_order_relaxed) != 0
        && g_volt.fn_set_vf && g_volt.nvml_device) {
        g_volt.fn_set_vf(g_volt.nvml_device, 0);
    }
    // Re-raise with default handler
    struct sigaction sa = {};
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, nullptr);
    raise(sig);
}

void install_signal_handlers_once() {
    static std::atomic<int> installed{0};
    if (installed.exchange(1, std::memory_order_acq_rel)) return;
    struct sigaction sa = {};
    sa.sa_sigaction = volt_signal_handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    int sigs[] = {SIGTERM, SIGINT, SIGSEGV, SIGABRT, SIGBUS};
    for (int s : sigs) sigaction(s, &sa, nullptr);
    std::atexit(volt_atexit_restore);
}

// ── Shape registry ───────────────────────────────────────────────────────────

inline uint64_t pack_key(uint32_t M, uint32_t K, uint32_t N) {
    return ((uint64_t)(M & 0xFFFFFu) << 40)
         | ((uint64_t)(K & 0xFFFFFu) << 20)
         | ((uint64_t)(N & 0xFFFFFu));
}
inline void unpack_mkn(uint64_t ph, uint32_t& M, uint32_t& K, uint32_t& N) {
    if ((ph >> 60) != 0xCULL) { M = K = N = 0; return; }
    M = (uint32_t)((ph >> 40) & 0xFFFFFu);
    K = (uint32_t)((ph >> 20) & 0xFFFFFu);
    N = (uint32_t)( ph        & 0xFFFFFu);
}
inline unsigned slot_idx(uint64_t key) {
    uint64_t mix = key ^ (key >> 32);
    return (unsigned)((mix * 0xC2B2AE3D27D4EB4FULL) >> 58) % MAX_SHAPES;
}

VoltShape* find_or_alloc(uint32_t M, uint32_t K, uint32_t N) {
    if (M == 0 || N == 0) return nullptr;
    uint64_t key = pack_key(M, K, N);
    unsigned start = slot_idx(key);
    for (unsigned probe = 0; probe < 8; ++probe) {
        unsigned i = (start + probe) & (MAX_SHAPES - 1);
        uint64_t cur = g_shapes[i].key.load(std::memory_order_acquire);
        if (cur == key) return &g_shapes[i];
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_shapes[i].key.compare_exchange_strong(
                    expected, key, std::memory_order_acq_rel)) {
                VoltShape& s = g_shapes[i];
                s.M = M; s.K = K; s.N = N;
                s.ai  = cipher_volt_arithmetic_intensity(M, K, N);
                s.cls = cipher_volt_classify(M, K, N);
                return &s;
            }
            return &g_shapes[i];
        }
    }
    return nullptr;
}

// ── Calibration JSON I/O ─────────────────────────────────────────────────────

void write_calibration_json(int probe_rc, bool supported) {
    FILE* fp = std::fopen(CALIB_PATH, "w");
    if (!fp) {
        std::fprintf(stderr, "[CIPHER Op30] VOLT could not write %s\n", CALIB_PATH);
        return;
    }
    time_t t = time(nullptr);
    struct tm tmv;
    gmtime_r(&t, &tmv);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    std::fprintf(fp,
        "{\n"
        "  \"actuation_supported\": %s,\n"
        "  \"probe_rc\": %d,\n"
        "  \"probe_rc_meaning\": \"%s\",\n"
        "  \"pod_driver_version\": \"%s\",\n"
        "  \"current_clock_mhz\": %u,\n"
        "  \"calibrated_at_utc\": \"%s\",\n"
        "  \"shapes\": [\n",
        supported ? "true" : "false",
        probe_rc,
        (probe_rc == NVML_ERROR_NOT_SUPPORTED) ? "NVML_ERROR_NOT_SUPPORTED"
            : (probe_rc == NVML_SUCCESS ? "NVML_SUCCESS" : "OTHER"),
        g_volt.driver_version[0] ? g_volt.driver_version : "unknown",
        (unsigned)g_volt.base_clock_mhz,
        ts);
    bool first = true;
    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        if (g_shapes[i].key.load(std::memory_order_relaxed) == 0) continue;
        const VoltShape& s = g_shapes[i];
        // Conservative fallback: when actuation unsupported, all shapes
        // are reported as FREQ_SENSITIVE regardless of AI — we never
        // reduce frequency for shapes we couldn't measure.
        const char* cls_str = "FREQ_SENSITIVE";
        if (supported) {
            if (s.cls == CIPHER_SHAPE_MEMORY_BOUND_SAFE) cls_str = "MEMORY_BOUND_SAFE";
            else if (s.cls == CIPHER_SHAPE_AMBIGUOUS)    cls_str = "AMBIGUOUS";
        }
        if (!first) std::fprintf(fp, ",\n");
        first = false;
        std::fprintf(fp,
            "    {\"M\": %u, \"K\": %u, \"N\": %u, \"ai\": %.3f, "
            "\"class\": \"%s\"}",
            s.M, s.K, s.N, s.ai, cls_str);
    }
    std::fprintf(fp, "\n  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr,
        "[CIPHER Op30] VOLT calibration -> %s "
        "(supported=%s, probe_rc=%d, driver=%s)\n",
        CALIB_PATH,
        supported ? "true" : "false",
        probe_rc,
        g_volt.driver_version[0] ? g_volt.driver_version : "unknown");
}

// Minimal JSON loader — we only care about "actuation_supported" + driver_version.
bool load_calibration(bool& supported_out, char* drv_buf, size_t drv_buf_sz) {
    FILE* fp = std::fopen(CALIB_PATH, "r");
    if (!fp) return false;
    char buf[4096];
    size_t n = std::fread(buf, 1, sizeof(buf) - 1, fp);
    std::fclose(fp);
    if (n == 0) return false;
    buf[n] = 0;
    const char* p = std::strstr(buf, "\"actuation_supported\"");
    if (!p) return false;
    p = std::strchr(p, ':');
    if (!p) return false;
    while (*++p == ' ');
    supported_out = (std::strncmp(p, "true", 4) == 0);

    drv_buf[0] = 0;
    const char* d = std::strstr(buf, "\"pod_driver_version\"");
    if (d) {
        d = std::strchr(d, ':');
        if (d) {
            while (*++d == ' ');
            if (*d == '"') {
                d++;
                size_t i = 0;
                while (*d && *d != '"' && i < drv_buf_sz - 1) drv_buf[i++] = *d++;
                drv_buf[i] = 0;
            }
        }
    }
    return true;
}

// ── Stage 2 actuation ────────────────────────────────────────────────────────

void actuate_impl(bool force) {
    int mode = g_volt.mode.load(std::memory_order_relaxed);
    if (mode != CIPHER_VOLT_ACTIVE) return;
    if (!g_volt.actuation_supported.load(std::memory_order_relaxed)) return;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    if (!force && g_volt.last_actuate_ns != 0
        && (now - g_volt.last_actuate_ns) < ACTUATE_PERIOD_NS) return;
    g_volt.last_actuate_ns = now;

    // THERMOSTAT override: when thermal aggressive mode is active, force
    // reduced frequency regardless of shape class (and skip restoration).
    bool thermo_force = cipher_thermostat_aggressive_active();

    // Decide: are we serving a memory-bound shape right now?
    bool any_safe = false;
    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        uint64_t k = g_shapes[i].key.load(std::memory_order_relaxed);
        if (k == 0) continue;
        if (g_shapes[i].cls == CIPHER_SHAPE_MEMORY_BOUND_SAFE) { any_safe = true; break; }
    }

    int  cur_offset = g_volt.offset_applied_khz.load(std::memory_order_relaxed);
    bool want_reduced = thermo_force || any_safe;

    if (want_reduced && cur_offset == 0) {
        // Reduce: -200,000 kHz offset (~ -200 MHz) — conservative
        if (g_volt.fn_set_vf && g_volt.fn_set_vf(g_volt.nvml_device, -200000) == NVML_SUCCESS) {
            g_volt.offset_applied_khz.store(-200000, std::memory_order_release);
            g_volt.reduce_events.fetch_add(1, std::memory_order_relaxed);
            install_signal_handlers_once();
        }
    } else if (!want_reduced && cur_offset != 0) {
        if (g_volt.fn_set_vf && g_volt.fn_set_vf(g_volt.nvml_device, 0) == NVML_SUCCESS) {
            g_volt.offset_applied_khz.store(0, std::memory_order_release);
            g_volt.restore_events.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

} // namespace

// ── Public API ───────────────────────────────────────────────────────────────

extern "C" double cipher_volt_arithmetic_intensity(uint32_t M, uint32_t K, uint32_t N) {
    if (M == 0 || K == 0 || N == 0) return 0.0;
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double bytes = 2.0 * ((double)M * K + (double)K * N + (double)M * N);
    return bytes > 0.0 ? flops / bytes : 0.0;
}

extern "C" CipherShapeClass cipher_volt_classify(uint32_t M, uint32_t K, uint32_t N) {
    double ai = cipher_volt_arithmetic_intensity(M, K, N);
    if (ai == 0.0)        return CIPHER_SHAPE_UNKNOWN;
    if (ai < 10.0)        return CIPHER_SHAPE_MEMORY_BOUND_SAFE;
    if (ai > 100.0)       return CIPHER_SHAPE_FREQ_SENSITIVE;
    return CIPHER_SHAPE_AMBIGUOUS;
}

extern "C" int cipher_volt_init(void) {
    int already = g_volt_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_volt.mode.load(std::memory_order_relaxed);

    const char* env = std::getenv("CIPHER_VOLT");
    if (!env) {
        g_volt.mode.store(CIPHER_VOLT_OFF, std::memory_order_release);
        return CIPHER_VOLT_OFF;
    }

    if (std::strcmp(env, "calibrate") == 0) {
        g_volt.mode.store(CIPHER_VOLT_CALIBRATE, std::memory_order_release);
        int rc = probe_actuation();
        bool supported = (rc == NVML_SUCCESS);
        g_volt.actuation_supported.store(supported ? 1 : 0, std::memory_order_release);
        // Seed registry with a couple of canonical shapes so the JSON has
        // something useful even on a degraded pod.
        find_or_alloc(1,    4096, 4096);    // decode
        find_or_alloc(4096, 4096, 4096);    // prefill
        find_or_alloc(64,   4096, 4096);    // ambiguous
        write_calibration_json(rc, supported);
        std::fprintf(stderr,
            "[CIPHER Op30] VOLT calibrate complete — supported=%s, exiting calibrate path.\n",
            supported ? "true" : "false");
        // Calibration mode does NOT proceed to active actuation; stays in
        // CALIBRATE state (observe is a no-op below).
        return CIPHER_VOLT_CALIBRATE;
    }

    if (std::strcmp(env, "on") == 0
        || std::strcmp(env, "1") == 0
        || std::strcmp(env, "ON") == 0) {
        bool supported = false;
        char drv[64] = {};
        if (!load_calibration(supported, drv, sizeof(drv))) {
            std::fprintf(stderr,
                "[CIPHER Op30] VOLT refusing to start: no calibration file at %s. "
                "Run CIPHER_VOLT=calibrate first.\n", CALIB_PATH);
            g_volt.mode.store(CIPHER_VOLT_OFF, std::memory_order_release);
            return CIPHER_VOLT_OFF;
        }
        // Re-probe to detect host changes between calibrate and active.
        int rc = probe_actuation();
        bool now_supported = (rc == NVML_SUCCESS);
        if (supported && !now_supported) {
            std::fprintf(stderr,
                "[CIPHER Op30] VOLT actuation regressed since calibration "
                "(probe rc=%d). Refusing to start active mode.\n", rc);
            g_volt.mode.store(CIPHER_VOLT_OFF, std::memory_order_release);
            return CIPHER_VOLT_OFF;
        }
        // Driver-version check: warn if calibration was on a different driver.
        if (drv[0] && g_volt.driver_version[0]
            && std::strcmp(drv, g_volt.driver_version) != 0) {
            std::fprintf(stderr,
                "[CIPHER Op30] VOLT calibration driver=%s differs from current=%s. "
                "Continuing but recalibration recommended.\n",
                drv, g_volt.driver_version);
        }
        if (supported) {
            g_volt.actuation_supported.store(1, std::memory_order_release);
            g_volt.mode.store(CIPHER_VOLT_ACTIVE, std::memory_order_release);
            std::fprintf(stderr,
                "[CIPHER Op30] VOLT ACTIVE — actuation supported (driver=%s, "
                "base clock=%u MHz). atexit + signal handlers will be installed "
                "on first reduction.\n",
                g_volt.driver_version, g_volt.base_clock_mhz);
        } else {
            g_volt.actuation_supported.store(0, std::memory_order_release);
            g_volt.mode.store(CIPHER_VOLT_DEGRADED, std::memory_order_release);
            std::fprintf(stderr,
                "[CIPHER Op30] VOLT DEGRADED — calibration says actuation unsupported "
                "(probe rc=%d NVML_ERROR_NOT_SUPPORTED). Classifier-only; no "
                "frequency writes will be made.\n", rc);
        }
        return g_volt.mode.load(std::memory_order_relaxed);
    }

    g_volt.mode.store(CIPHER_VOLT_OFF, std::memory_order_release);
    return CIPHER_VOLT_OFF;
}

extern "C" void cipher_volt_observe(const CipherRingEntry* ev) {
    int mode = g_volt.mode.load(std::memory_order_relaxed);
    if (mode == CIPHER_VOLT_OFF || mode == CIPHER_VOLT_CALIBRATE) return;
    if (!ev || ev->kernel_class != 0) return;
    uint32_t M, K, N;
    unpack_mkn(ev->params_hash, M, K, N);
    if (M == 0 || N == 0) return;
    find_or_alloc(M, K, N);   // classifies on first sight
}

extern "C" void cipher_volt_actuate(void)        { actuate_impl(false); }
extern "C" void cipher_volt_force_actuate(void)  { actuate_impl(true); }

extern "C" CipherVoltMode cipher_volt_mode(void) {
    return (CipherVoltMode)g_volt.mode.load(std::memory_order_acquire);
}
extern "C" int cipher_volt_actuation_supported(void) {
    return g_volt.actuation_supported.load(std::memory_order_acquire) == 1;
}
extern "C" unsigned cipher_volt_reduce_event_count(void) {
    return g_volt.reduce_events.load(std::memory_order_relaxed);
}
extern "C" unsigned cipher_volt_restore_event_count(void) {
    return g_volt.restore_events.load(std::memory_order_relaxed);
}
extern "C" const char* cipher_volt_status_string(void) {
    switch (g_volt.mode.load(std::memory_order_relaxed)) {
        case CIPHER_VOLT_OFF:       return "OFF";
        case CIPHER_VOLT_CALIBRATE: return "CALIBRATE";
        case CIPHER_VOLT_ACTIVE:    return "ACTIVE";
        case CIPHER_VOLT_DEGRADED:  return "DEGRADED";
    }
    return "UNKNOWN";
}
extern "C" void cipher_volt_inject_supported(int forced_value) {
    g_volt.inject_supported = forced_value;
}

extern "C" void cipher_volt_report(void) {
    if (g_volt.mode.load(std::memory_order_relaxed) == CIPHER_VOLT_OFF) return;
    FILE* fp = std::fopen("/tmp/cipher_volt_report.json", "w");
    if (!fp) return;
    std::fprintf(fp,
        "{\n"
        "  \"mode\": \"%s\",\n"
        "  \"actuation_supported\": %s,\n"
        "  \"driver_version\": \"%s\",\n"
        "  \"base_clock_mhz\": %u,\n"
        "  \"reduce_events\": %u,\n"
        "  \"restore_events\": %u,\n"
        "  \"current_offset_khz\": %d,\n"
        "  \"shapes\": [\n",
        cipher_volt_status_string(),
        cipher_volt_actuation_supported() ? "true" : "false",
        g_volt.driver_version[0] ? g_volt.driver_version : "unknown",
        g_volt.base_clock_mhz,
        g_volt.reduce_events.load(std::memory_order_relaxed),
        g_volt.restore_events.load(std::memory_order_relaxed),
        g_volt.offset_applied_khz.load(std::memory_order_relaxed));
    bool first = true;
    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        if (g_shapes[i].key.load(std::memory_order_relaxed) == 0) continue;
        const VoltShape& s = g_shapes[i];
        const char* cls_str = "FREQ_SENSITIVE";
        if (s.cls == CIPHER_SHAPE_MEMORY_BOUND_SAFE) cls_str = "MEMORY_BOUND_SAFE";
        else if (s.cls == CIPHER_SHAPE_AMBIGUOUS)    cls_str = "AMBIGUOUS";
        if (!first) std::fprintf(fp, ",\n");
        first = false;
        std::fprintf(fp,
            "    {\"M\": %u, \"K\": %u, \"N\": %u, \"ai\": %.3f, \"class\": \"%s\"}",
            s.M, s.K, s.N, s.ai, cls_str);
    }
    std::fprintf(fp, "\n  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op30] VOLT report -> /tmp/cipher_volt_report.json\n");
}
