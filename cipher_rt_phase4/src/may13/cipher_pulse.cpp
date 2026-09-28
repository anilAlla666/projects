// Op 22 PULSE — hardware fault early warning.
//
// Stage 1: per-(M,K,N) Welford on inter-event ITL (long window, baseline frozen
//          at n == BASELINE_FREEZE_N).
// Stage 2: every N dispatches (or on force_evaluate), evaluate signals 1 + 3
//          and emit /tmp/cipher_pulse.log (INFO) or /tmp/cipher_pulse_alert.json
//          (WARNING). Signal 2 (max_diff degradation) deferred — needs a
//          Stage 0 sentinel hook in Op 3 SUBSTITUTE.

#include "may13/cipher_pulse.h"

#include "cipher_rt_commit.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>

namespace {

constexpr unsigned MAX_SHAPES         = 64;
constexpr uint32_t BASELINE_FREEZE_N  = 100;
constexpr double   DRIFT_GATE_RATIO   = 1.10;
constexpr double   DRIFT_GATE_SIGMAS  = 2.0;
constexpr uint64_t ECC_DELTA_GATE     = 10;
constexpr uint64_t EVAL_DISPATCH_GATE = 1000;
constexpr uint64_t REALERT_PERIOD_NS  = 60ULL * 1000 * 1000 * 1000;  // 60 s

// NVML ECC counter constants (matches NVML headers; we don't include them).
constexpr int NVML_MEMORY_ERROR_TYPE_CORRECTED   = 0;
constexpr int NVML_VOLATILE_ECC                  = 0;
constexpr int NVML_MEMORY_LOCATION_DEVICE_MEMORY = 0;

typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;
typedef nvmlReturn_t (*fn_GetHandle_t)(uint32_t, nvmlDevice_t*);
typedef nvmlReturn_t (*fn_GetEcc_t)(nvmlDevice_t, int, int, int, unsigned long long*);

struct PulseShape {
    std::atomic<uint64_t> key;       // 0 = empty
    uint32_t              M, K, N;
    uint64_t              prev_ts_ns;
    double                mean_ns, M2_ns;
    uint32_t              n;
    double                long_baseline_mean;
    double                long_baseline_sd;
    double                inject_factor;       // 0 = none
};

alignas(64) PulseShape g_shapes[MAX_SHAPES];

struct PulseState {
    std::atomic<uint64_t> dispatch_count;
    uint64_t              last_eval_count;
    std::atomic<uint64_t> ecc_baseline;       // captured at first NVML success
    std::atomic<uint64_t> ecc_inject;         // test-only delta
    std::atomic<int>      score;              // 0..2 in v1
    std::atomic<unsigned> alert_count;
    uint64_t              last_alert_ns;
    std::atomic<int>      last_severity;      // 0..2 ; for re-alert dedup
    bool                  nvml_resolved;
    void*                 nvml_handle;        // RTLD_NOLOAD; do NOT dlclose
    nvmlDevice_t          nvml_device;
    fn_GetEcc_t           fn_get_ecc;
};

PulseState g_pulse = {
    {0}, 0, {0}, {0}, {0}, {0u}, 0, {0}, false, nullptr, nullptr, nullptr
};

std::atomic<int> g_pulse_enabled{0};
std::atomic<int> g_pulse_initialized{0};

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
    return (unsigned)((mix * 0x9FB21C651E98DF25ULL) >> 58) % MAX_SHAPES;
}

PulseShape* find_or_alloc(uint32_t M, uint32_t K, uint32_t N) {
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
                PulseShape& s = g_shapes[i];
                s.M = M; s.K = K; s.N = N;
                s.prev_ts_ns = 0;
                s.mean_ns = 0.0; s.M2_ns = 0.0; s.n = 0;
                s.long_baseline_mean = 0.0;
                s.long_baseline_sd = 0.0;
                s.inject_factor = 0.0;
                return &s;
            }
            return &g_shapes[i];
        }
    }
    return nullptr;
}

inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void try_resolve_nvml() {
    if (g_pulse.nvml_resolved) return;
    g_pulse.nvml_resolved = true;  // record attempt regardless of success

    void* h = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libnvidia-ml.so",   RTLD_LAZY | RTLD_NOLOAD);
    if (!h) {
        std::fprintf(stderr,
            "[CIPHER Op22] PULSE: NVML not loaded (telemetry off?). "
            "Signal 3 (ECC) disabled.\n");
        return;
    }
    g_pulse.nvml_handle = h;

    fn_GetHandle_t fn_handle = (fn_GetHandle_t)dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
    g_pulse.fn_get_ecc       = (fn_GetEcc_t)dlsym(h, "nvmlDeviceGetMemoryErrorCounter");
    if (!fn_handle || !g_pulse.fn_get_ecc) {
        std::fprintf(stderr,
            "[CIPHER Op22] PULSE: NVML symbols missing (handle=%p ecc=%p). "
            "Signal 3 (ECC) disabled.\n", (void*)fn_handle, (void*)g_pulse.fn_get_ecc);
        g_pulse.fn_get_ecc = nullptr;
        return;
    }

    nvmlDevice_t dev = nullptr;
    if (fn_handle(0, &dev) != 0 || !dev) {
        std::fprintf(stderr,
            "[CIPHER Op22] PULSE: nvmlDeviceGetHandleByIndex_v2 failed. "
            "Signal 3 (ECC) disabled.\n");
        g_pulse.fn_get_ecc = nullptr;
        return;
    }
    g_pulse.nvml_device = dev;

    // Capture baseline ECC count
    unsigned long long base = 0;
    if (g_pulse.fn_get_ecc(dev, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                           NVML_VOLATILE_ECC,
                           NVML_MEMORY_LOCATION_DEVICE_MEMORY, &base) == 0) {
        g_pulse.ecc_baseline.store((uint64_t)base, std::memory_order_release);
        std::fprintf(stderr,
            "[CIPHER Op22] PULSE: NVML ECC baseline = %llu\n",
            (unsigned long long)base);
    } else {
        // Some drivers return NOT_SUPPORTED; degrade gracefully.
        std::fprintf(stderr,
            "[CIPHER Op22] PULSE: ECC counter not supported on this device.\n");
        g_pulse.fn_get_ecc = nullptr;
    }
}

bool eval_signal_drift(unsigned* drift_n_out) {
    unsigned drift_n = 0;
    bool any = false;
    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        if (g_shapes[i].key.load(std::memory_order_relaxed) == 0) continue;
        const PulseShape& s = g_shapes[i];
        bool fired = false;
        // Test injection: if inject_factor set, treat as drift if > DRIFT_GATE_RATIO.
        if (s.inject_factor > DRIFT_GATE_RATIO) fired = true;
        // Real path: long baseline frozen.
        if (!fired && s.n >= BASELINE_FREEZE_N && s.long_baseline_mean > 0.0) {
            double effective = s.mean_ns;
            if (s.inject_factor > 0.0) effective *= s.inject_factor;
            double dev = effective - s.long_baseline_mean;
            if (dev > DRIFT_GATE_SIGMAS * s.long_baseline_sd
                && (effective / s.long_baseline_mean) > DRIFT_GATE_RATIO) {
                fired = true;
            }
        }
        if (fired) { drift_n++; any = true; }
    }
    if (drift_n_out) *drift_n_out = drift_n;
    return any;
}

bool eval_signal_ecc(uint64_t* ecc_delta_out) {
    uint64_t delta = g_pulse.ecc_inject.load(std::memory_order_relaxed);
    if (g_pulse.fn_get_ecc && g_pulse.nvml_device) {
        unsigned long long cur = 0;
        if (g_pulse.fn_get_ecc(g_pulse.nvml_device,
                               NVML_MEMORY_ERROR_TYPE_CORRECTED,
                               NVML_VOLATILE_ECC,
                               NVML_MEMORY_LOCATION_DEVICE_MEMORY, &cur) == 0) {
            uint64_t base = g_pulse.ecc_baseline.load(std::memory_order_relaxed);
            if ((uint64_t)cur > base) delta += (uint64_t)cur - base;
        }
    }
    if (ecc_delta_out) *ecc_delta_out = delta;
    return delta > ECC_DELTA_GATE;
}

const char* severity_name(int sev) {
    switch (sev) {
        case 0:  return "NONE";
        case 1:  return "INFO";
        case 2:  return "WARNING";
        default: return "CRITICAL";   // unreachable in v1
    }
}

void emit_alert(int score, unsigned drift_n, uint64_t ecc_delta, bool drift_fired,
                bool ecc_fired) {
    g_pulse.alert_count.fetch_add(1, std::memory_order_relaxed);
    g_pulse.last_alert_ns = now_ns();
    g_pulse.last_severity.store(score, std::memory_order_relaxed);

    if (score == 1) {
        FILE* fp = std::fopen("/tmp/cipher_pulse.log", "a");
        if (fp) {
            std::fprintf(fp,
                "{\"ts\": %llu, \"severity\": \"INFO\", \"score\": %d, "
                "\"drift_shapes\": %u, \"ecc_delta\": %llu, "
                "\"drift_fired\": %s, \"ecc_fired\": %s, "
                "\"signal_2_deferred\": true}\n",
                (unsigned long long)g_pulse.last_alert_ns, score,
                drift_n, (unsigned long long)ecc_delta,
                drift_fired ? "true" : "false",
                ecc_fired   ? "true" : "false");
            std::fclose(fp);
        }
        std::fprintf(stderr, "[CIPHER Op22] PULSE INFO score=1 drift_shapes=%u ecc_delta=%llu\n",
                     drift_n, (unsigned long long)ecc_delta);
    } else if (score >= 2) {
        FILE* fp = std::fopen("/tmp/cipher_pulse_alert.json", "w");
        if (fp) {
            // Build a list of drifting (M,K,N) for the alert
            std::fprintf(fp,
                "{\n"
                "  \"timestamp_ns\": %llu,\n"
                "  \"severity\": \"MEDIUM\",\n"
                "  \"score\": %d,\n"
                "  \"score_ceiling_v1\": 2,\n"
                "  \"signal_1_drift_shapes\": %u,\n"
                "  \"signal_3_ecc_delta_1h\": %llu,\n"
                "  \"signal_2_deferred\": true,\n"
                "  \"deferred_reason\": \"per-substitution max_diff requires Stage 0 sentinel hook in Op 3 SUBSTITUTE\",\n"
                "  \"recommendation\": \"Inspect drifting shapes and ECC counter trend. CRITICAL escalation requires Signal 2 to be wired.\"\n"
                "}\n",
                (unsigned long long)g_pulse.last_alert_ns, score,
                drift_n, (unsigned long long)ecc_delta);
            std::fclose(fp);
        }
        std::fprintf(stderr,
            "[CIPHER Op22] PULSE WARNING score=%d drift_shapes=%u ecc_delta=%llu "
            "→ /tmp/cipher_pulse_alert.json (signal_2_deferred=true, ceiling=2)\n",
            score, drift_n, (unsigned long long)ecc_delta);
    }
}

void evaluate_impl(bool force) {
    if (!g_pulse_enabled.load(std::memory_order_relaxed)) return;

    uint64_t dispatches = g_pulse.dispatch_count.load(std::memory_order_relaxed);
    if (!force && (dispatches - g_pulse.last_eval_count) < EVAL_DISPATCH_GATE) return;
    g_pulse.last_eval_count = dispatches;

    try_resolve_nvml();   // idempotent

    unsigned drift_n = 0;
    uint64_t ecc_delta = 0;
    bool sig1 = eval_signal_drift(&drift_n);
    bool sig3 = eval_signal_ecc(&ecc_delta);

    int score = (sig1 ? 1 : 0) + (sig3 ? 1 : 0);
    g_pulse.score.store(score, std::memory_order_release);

    if (score == 0) return;

    // Re-alert dedup: same severity within 60 s → skip
    int prev_sev = g_pulse.last_severity.load(std::memory_order_relaxed);
    uint64_t t = now_ns();
    if (score <= prev_sev && (t - g_pulse.last_alert_ns) < REALERT_PERIOD_NS) return;

    emit_alert(score, drift_n, ecc_delta, sig1, sig3);
}

} // namespace

extern "C" int cipher_pulse_init(void) {
    int already = g_pulse_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_pulse_enabled.load(std::memory_order_relaxed);

    const char* env = std::getenv("CIPHER_PULSE");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_pulse_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr,
            "[CIPHER Op22] PULSE enabled — drift gate=+%.0f%%/2σ, "
            "ECC gate=%llu/1h, eval every %llu dispatches. "
            "v1 SCORE CEILING = 2 (CRITICAL unreachable; Signal 2 deferred — "
            "needs Stage 0 sentinel hook in Op 3).\n",
            (DRIFT_GATE_RATIO - 1.0) * 100.0,
            (unsigned long long)ECC_DELTA_GATE,
            (unsigned long long)EVAL_DISPATCH_GATE);
    }
    return on;
}

extern "C" void cipher_pulse_observe(const CipherRingEntry* ev) {
    if (!g_pulse_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;
    g_pulse.dispatch_count.fetch_add(1, std::memory_order_relaxed);
    if (ev->kernel_class != 0) return;

    uint32_t M, K, N;
    unpack_mkn(ev->params_hash, M, K, N);
    if (M == 0 || N == 0) return;

    PulseShape* sp = find_or_alloc(M, K, N);
    if (!sp) return;
    PulseShape& s = *sp;

    uint64_t now = ev->timestamp_ns;
    if (now == 0 || s.prev_ts_ns == 0 || now <= s.prev_ts_ns) {
        s.prev_ts_ns = now;
        return;
    }
    double itl = (double)(now - s.prev_ts_ns);
    s.prev_ts_ns = now;

    s.n++;
    double d = itl - s.mean_ns;
    s.mean_ns += d / s.n;
    s.M2_ns   += d * (itl - s.mean_ns);
    if (s.n == BASELINE_FREEZE_N) {
        s.long_baseline_mean = s.mean_ns;
        double var = s.M2_ns / s.n;
        s.long_baseline_sd = std::sqrt(var > 0.0 ? var : 0.0);
    }
}

extern "C" void cipher_pulse_evaluate(void)       { evaluate_impl(false); }
extern "C" void cipher_pulse_force_evaluate(void) { evaluate_impl(true); }

extern "C" int      cipher_pulse_score(void)       { return g_pulse.score.load(std::memory_order_acquire); }
extern "C" unsigned cipher_pulse_alert_count(void) { return g_pulse.alert_count.load(std::memory_order_relaxed); }
extern "C" const char* cipher_pulse_severity(void) {
    return severity_name(g_pulse.last_severity.load(std::memory_order_relaxed));
}

extern "C" void cipher_pulse_inject_drift_for_shape(
    uint32_t M, uint32_t K, uint32_t N, double mean_inflation_factor)
{
    PulseShape* sp = find_or_alloc(M, K, N);
    if (!sp) return;
    sp->inject_factor = mean_inflation_factor;
    if (sp->long_baseline_mean == 0.0) {
        // Synthesize baseline so the test can assert without a real warmup.
        sp->long_baseline_mean = 1000.0;
        sp->long_baseline_sd   = 50.0;
        sp->mean_ns            = 1000.0;
        sp->n                  = BASELINE_FREEZE_N;
    }
}

extern "C" void cipher_pulse_inject_ecc_delta(uint64_t additional_correctable) {
    g_pulse.ecc_inject.store(additional_correctable, std::memory_order_release);
}

extern "C" void cipher_pulse_clear_injection(void) {
    g_pulse.ecc_inject.store(0, std::memory_order_release);
    for (unsigned i = 0; i < MAX_SHAPES; ++i) g_shapes[i].inject_factor = 0.0;
}

extern "C" void cipher_pulse_report(void) {
    /* W7-9 Step 4 — coherent snapshot acquire (slow-path; emits the
     * COMMIT-published state alongside this report's existing aggregate). */
    struct cipher_rt_snapshot _snap;
    cipher_rt_snapshot_acquire(0u, &_snap);
    (void)_snap;

    if (!g_pulse_enabled.load(std::memory_order_relaxed)) return;
    FILE* fp = std::fopen("/tmp/cipher_pulse_report.json", "w");
    if (!fp) return;
    std::fprintf(fp,
        "{\n"
        "  \"score\": %d,\n"
        "  \"severity\": \"%s\",\n"
        "  \"alert_count\": %u,\n"
        "  \"dispatch_count\": %llu,\n"
        "  \"ecc_baseline\": %llu,\n"
        "  \"ecc_inject\": %llu,\n"
        "  \"signal_2_deferred\": true,\n"
        "  \"score_ceiling_v1\": 2\n"
        "}\n",
        g_pulse.score.load(std::memory_order_relaxed),
        severity_name(g_pulse.last_severity.load(std::memory_order_relaxed)),
        g_pulse.alert_count.load(std::memory_order_relaxed),
        (unsigned long long)g_pulse.dispatch_count.load(std::memory_order_relaxed),
        (unsigned long long)g_pulse.ecc_baseline.load(std::memory_order_relaxed),
        (unsigned long long)g_pulse.ecc_inject.load(std::memory_order_relaxed));
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op22] PULSE report -> /tmp/cipher_pulse_report.json\n");
}
