// Op 20 THERMOSTAT — Stage 1 (per-shape Welford) + Stage 2 (poll & decide).
//
// Reads gpu_temp_c from the existing telemetry pipeline (g_cipher.liquid).
// Maintains a 64-entry per-shape Welford on inter-event ITL and combines
// the two signals once per 500 ms.

#include "cipher_thermostat.h"
#include "cipher_liquid_state.h"
#include "cipher.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace {

constexpr unsigned MAX_SHAPES        = 64;
constexpr float    TEMP_HOT_C        = 75.0f;
constexpr float    TEMP_COOL_C       = 72.0f;
constexpr double   DRIFT_GATE_FIRE   = 1.10;   // > +10 % over baseline
constexpr double   DRIFT_GATE_CLEAR  = 1.05;   // < +5 % releases
constexpr uint32_t BASELINE_FREEZE_N = 30;
constexpr uint64_t POLL_PERIOD_NS    = 500ULL * 1000 * 1000;

struct ShapeWelford {
    std::atomic<uint64_t> key;       // (M<<40)|(K<<20)|N — 0 = empty
    uint32_t              M, K, N;
    uint64_t              prev_ts_ns;
    double                mean_ns, M2_ns;
    uint32_t              n;
    double                baseline_mean_ns;
    double                inject_factor;       // 0 = no injection
};

alignas(64) ShapeWelford g_shapes[MAX_SHAPES];

struct ThermoState {
    std::atomic<float>    last_temp_c;
    std::atomic<float>    prev_temp_c;
    std::atomic<float>    inject_temp_c;       // < 0 = no injection
    std::atomic<int>      aggressive;
    std::atomic<unsigned> event_count;
    std::atomic<unsigned> drift_shape_count;
    uint64_t              last_poll_ns;
};

ThermoState g_thermo = {
    {0.0f}, {0.0f}, {-1.0f}, {0}, {0u}, {0u}, 0
};

std::atomic<int>      g_thermostat_enabled{0};
std::atomic<int>      g_thermostat_initialized{0};

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
    return (unsigned)((mix * 0xD1B54A32D192ED03ULL) >> 58) % MAX_SHAPES;
}

ShapeWelford* find_or_alloc(uint32_t M, uint32_t K, uint32_t N) {
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
                ShapeWelford& s = g_shapes[i];
                s.M = M; s.K = K; s.N = N;
                s.prev_ts_ns = 0;
                s.mean_ns = 0.0; s.M2_ns = 0.0; s.n = 0;
                s.baseline_mean_ns = 0.0;
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

float read_temp() {
    float inj = g_thermo.inject_temp_c.load(std::memory_order_relaxed);
    if (inj >= 0.0f) return inj;
    // ::g_cipher is declared extern in cipher.h (already #included). Anonymous-
    // namespace `extern` would give it internal linkage and break the link.
    if (::g_cipher.liquid.initialized && ::g_cipher.liquid.device) {
        return ::g_cipher.liquid.device->hw.gpu_temp_c;
    }
    return 0.0f;
}

bool any_shape_drifting(double gate, unsigned* drift_n_out) {
    unsigned drift_n = 0;
    bool any = false;
    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        if (g_shapes[i].key.load(std::memory_order_relaxed) == 0) continue;
        const ShapeWelford& s = g_shapes[i];
        double effective_mean = s.mean_ns;
        if (s.inject_factor > 0.0) effective_mean *= s.inject_factor;
        if (s.n >= BASELINE_FREEZE_N && s.baseline_mean_ns > 0.0) {
            double ratio = effective_mean / s.baseline_mean_ns;
            if (ratio > gate) {
                drift_n++;
                any = true;
            }
        } else if (s.inject_factor > gate) {
            // Test injection without enough samples — still counts.
            drift_n++;
            any = true;
        }
    }
    if (drift_n_out) *drift_n_out = drift_n;
    return any;
}

} // namespace

extern "C" int cipher_thermostat_init(void) {
    int already = g_thermostat_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_thermostat_enabled.load(std::memory_order_relaxed);

    const char* env = std::getenv("CIPHER_THERMOSTAT");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_thermostat_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr,
            "[CIPHER Op20] THERMOSTAT enabled — temp gate=%.1f→%.1f °C, "
            "drift gate=+%.0f %%/+%.0f %%, poll=%llu ms. "
            "v1: aggressive flag set but not yet consumed by Stage 0.\n",
            TEMP_HOT_C, TEMP_COOL_C,
            (DRIFT_GATE_FIRE - 1.0) * 100.0,
            (DRIFT_GATE_CLEAR - 1.0) * 100.0,
            (unsigned long long)(POLL_PERIOD_NS / 1000000ULL));
    }
    return on;
}

extern "C" void cipher_thermostat_observe(const CipherRingEntry* ev) {
    if (!g_thermostat_enabled.load(std::memory_order_relaxed)) return;
    if (!ev || ev->kernel_class != 0) return;

    uint32_t M, K, N;
    unpack_mkn(ev->params_hash, M, K, N);
    if (M == 0 || N == 0) return;

    ShapeWelford* sp = find_or_alloc(M, K, N);
    if (!sp) return;
    ShapeWelford& s = *sp;

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
    if (s.n == BASELINE_FREEZE_N) s.baseline_mean_ns = s.mean_ns;
}

static void thermostat_poll_impl(bool bypass_cadence) {
    if (!g_thermostat_enabled.load(std::memory_order_relaxed)) return;

    uint64_t t = now_ns();
    if (!bypass_cadence
        && g_thermo.last_poll_ns != 0
        && (t - g_thermo.last_poll_ns) < POLL_PERIOD_NS) return;
    g_thermo.last_poll_ns = t;

    float prev = g_thermo.last_temp_c.load(std::memory_order_relaxed);
    float curr = read_temp();
    g_thermo.prev_temp_c.store(prev, std::memory_order_relaxed);
    g_thermo.last_temp_c.store(curr, std::memory_order_relaxed);

    bool sig_temp = (curr > TEMP_HOT_C) && (curr > prev);

    unsigned drift_n = 0;
    bool sig_drift = any_shape_drifting(DRIFT_GATE_FIRE, &drift_n);
    g_thermo.drift_shape_count.store(drift_n, std::memory_order_relaxed);

    int aggr_now = g_thermo.aggressive.load(std::memory_order_relaxed);

    if (sig_temp && sig_drift) {
        if (aggr_now == 0) {
            g_thermo.aggressive.store(1, std::memory_order_release);
            g_thermo.event_count.fetch_add(1, std::memory_order_relaxed);
            std::fprintf(stderr,
                "[CIPHER Op20] THERMOSTAT FIRE — temp=%.1f°C (prev %.1f), "
                "drift_shapes=%u → aggressive=1 (flag set; Stage 0 hook deferred)\n",
                curr, prev, drift_n);
        }
    } else if (aggr_now == 1) {
        // Clear gate: cool AND no shape > +5 %
        unsigned residual = 0;
        bool still_drifting = any_shape_drifting(DRIFT_GATE_CLEAR, &residual);
        if (curr < TEMP_COOL_C && !still_drifting) {
            g_thermo.aggressive.store(0, std::memory_order_release);
            std::fprintf(stderr,
                "[CIPHER Op20] THERMOSTAT CLEAR — temp=%.1f°C, "
                "no shape > +%.0f %% → aggressive=0\n",
                curr, (DRIFT_GATE_CLEAR - 1.0) * 100.0);
        }
    }
}

extern "C" void cipher_thermostat_poll(void)       { thermostat_poll_impl(false); }
extern "C" void cipher_thermostat_force_poll(void) { thermostat_poll_impl(true); }

extern "C" int cipher_thermostat_aggressive_active(void) {
    return g_thermo.aggressive.load(std::memory_order_acquire);
}

extern "C" unsigned cipher_thermostat_event_count(void) {
    return g_thermo.event_count.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_thermostat_drift_shape_count(void) {
    return g_thermo.drift_shape_count.load(std::memory_order_relaxed);
}

extern "C" void cipher_thermostat_inject_temp(float celsius) {
    g_thermo.inject_temp_c.store(celsius, std::memory_order_release);
}

extern "C" void cipher_thermostat_inject_drift_for_shape(
    uint32_t M, uint32_t K, uint32_t N, double mean_inflation_factor)
{
    ShapeWelford* sp = find_or_alloc(M, K, N);
    if (!sp) return;
    sp->inject_factor = mean_inflation_factor;
    if (sp->baseline_mean_ns == 0.0) {
        // Synthesize a baseline so the test can assert drift without a
        // real warmup cycle.
        sp->baseline_mean_ns = 1000.0;       // arbitrary nonzero baseline
        sp->mean_ns = 1000.0;                // current mean = baseline; inject scales
        sp->n = BASELINE_FREEZE_N;
    }
}

extern "C" void cipher_thermostat_clear_injection(void) {
    g_thermo.inject_temp_c.store(-1.0f, std::memory_order_release);
    for (unsigned i = 0; i < MAX_SHAPES; ++i) g_shapes[i].inject_factor = 0.0;
}

extern "C" void cipher_thermostat_report(void) {
    if (!g_thermostat_enabled.load(std::memory_order_relaxed)) return;
    FILE* fp = std::fopen("/tmp/cipher_thermostat_report.json", "w");
    if (!fp) return;
    std::fprintf(fp,
        "{\n"
        "  \"aggressive\": %d,\n"
        "  \"event_count\": %u,\n"
        "  \"drift_shape_count\": %u,\n"
        "  \"last_temp_c\": %.2f,\n"
        "  \"prev_temp_c\": %.2f,\n"
        "  \"shapes\": [\n",
        g_thermo.aggressive.load(std::memory_order_relaxed),
        g_thermo.event_count.load(std::memory_order_relaxed),
        g_thermo.drift_shape_count.load(std::memory_order_relaxed),
        g_thermo.last_temp_c.load(std::memory_order_relaxed),
        g_thermo.prev_temp_c.load(std::memory_order_relaxed));
    bool first = true;
    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        if (g_shapes[i].key.load(std::memory_order_relaxed) == 0) continue;
        const ShapeWelford& s = g_shapes[i];
        if (!first) std::fprintf(fp, ",\n");
        first = false;
        double ratio = (s.baseline_mean_ns > 0.0)
            ? (s.mean_ns * (s.inject_factor > 0 ? s.inject_factor : 1.0))
              / s.baseline_mean_ns : 0.0;
        std::fprintf(fp,
            "    {\"M\": %u, \"K\": %u, \"N\": %u, \"n\": %u, "
            "\"mean_us\": %.3f, \"baseline_us\": %.3f, "
            "\"drift_ratio\": %.3f, \"inject\": %.3f}",
            s.M, s.K, s.N, s.n,
            s.mean_ns / 1e3, s.baseline_mean_ns / 1e3,
            ratio, s.inject_factor);
    }
    std::fprintf(fp, "\n  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op20] THERMOSTAT report -> /tmp/cipher_thermostat_report.json\n");
}
