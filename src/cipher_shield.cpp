// Op 14 SHIELD — Stage 1 latency-protection monitor.
//
// Reads ring entries during Stage 1 (after SENSE has classified the current
// session). For HUMAN_INTERACTIVE sessions:
//   * Sets CIPHER_BAND_PROTECTED via cipher_sm_set_priority (Protection 1)
//   * Tracks per-session ITL P50/P95 over a 20-event rolling window
//   * Counts >2σ jitter events (Protection 2 detection only — flag not yet
//     consumed by Op 3; wiring is a v2 Stage 0 plan)
//   * When P95/P50 > 3.0 sets oracle_aggressive flag (Protection 3 detection
//     only — flag not yet consumed by oracle; v2 Stage 0 plan)

#include "cipher_shield.h"
#include "cipher_sense.h"
#include "cipher_green_ctx.h"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace {

constexpr unsigned MAX_SHIELD_SESSIONS = 256;
constexpr unsigned ITL_WINDOW          = 20;
constexpr double   JITTER_SIGMA_GATE   = 2.0;
constexpr double   BURST_P95_P50_GATE  = 3.0;

struct ShieldSession {
    std::atomic<uint64_t> session_id;          // 0 = empty
    uint64_t              ts_window[ITL_WINDOW];
    uint8_t               head;
    uint8_t               full;
    uint64_t              last_ts_ns;
    // Welford on inter-event latency
    double                itl_mean_ns;
    double                itl_M2_ns;
    uint32_t              n_itl;
    // Jitter & burst counters (observability)
    uint32_t              jitter_events;
    uint8_t               smoothed;            // raised once P95/P50>3
    std::atomic<uint8_t>  oracle_aggressive;   // for v2 oracle hook
    std::atomic<uint8_t>  cache_aggressive;    // for v2 Op 3 hook
    uint8_t               priority_assigned;   // SHIELD wrote band already?
};

alignas(64) ShieldSession g_shield[MAX_SHIELD_SESSIONS];

std::atomic<int>      g_shield_enabled{0};
std::atomic<int>      g_shield_initialized{0};
std::atomic<unsigned> g_jitter_total{0};
std::atomic<unsigned> g_smoothed_total{0};
std::atomic<unsigned> g_priority_total{0};

inline unsigned slot_idx(uint64_t sid) {
    uint64_t mix = sid ^ (sid >> 32);
    return (unsigned)((mix * 0xBF58476D1CE4E5B9ULL) >> 56) % MAX_SHIELD_SESSIONS;
}

ShieldSession* find_or_alloc(uint64_t sid) {
    if (sid == 0) return nullptr;
    unsigned start = slot_idx(sid);
    for (unsigned probe = 0; probe < 8; ++probe) {
        unsigned i = (start + probe) & (MAX_SHIELD_SESSIONS - 1);
        uint64_t cur = g_shield[i].session_id.load(std::memory_order_acquire);
        if (cur == sid) return &g_shield[i];
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_shield[i].session_id.compare_exchange_strong(
                    expected, sid, std::memory_order_acq_rel)) {
                ShieldSession& s = g_shield[i];
                std::memset(s.ts_window, 0, sizeof(s.ts_window));
                s.head = 0; s.full = 0; s.last_ts_ns = 0;
                s.itl_mean_ns = 0.0; s.itl_M2_ns = 0.0; s.n_itl = 0;
                s.jitter_events = 0;
                s.smoothed = 0;
                s.oracle_aggressive.store(0, std::memory_order_relaxed);
                s.cache_aggressive.store(0, std::memory_order_relaxed);
                s.priority_assigned = 0;
                return &s;
            }
            return &g_shield[i];  // raced; another thread won
        }
    }
    return nullptr;  // probe budget exhausted — drop silently
}

void update_itl(ShieldSession& s, uint64_t now) {
    if (s.last_ts_ns != 0 && now > s.last_ts_ns) {
        double itl = (double)(now - s.last_ts_ns);
        // Welford
        s.n_itl++;
        double d = itl - s.itl_mean_ns;
        s.itl_mean_ns += d / s.n_itl;
        s.itl_M2_ns   += d * (itl - s.itl_mean_ns);

        // Rolling window of timestamps (for P50/P95)
        s.ts_window[s.head] = now;
        s.head = (uint8_t)((s.head + 1u) % ITL_WINDOW);
        if (s.head == 0) s.full = 1;

        // Compute rolling-window gap stats from the timestamp ring (cheap;
        // window size = 20). We reuse the sorted gap list for both the
        // jitter check (current vs median) and the smoothing check (P95/P50).
        if (s.full || s.head >= 5) {
            unsigned count = s.full ? ITL_WINDOW : s.head;
            uint64_t gaps[ITL_WINDOW];
            unsigned gaps_n = 0;
            for (unsigned i = 1; i < count; ++i) {
                unsigned a_idx = (s.head + ITL_WINDOW - count + i - 1) % ITL_WINDOW;
                unsigned b_idx = (s.head + ITL_WINDOW - count + i) % ITL_WINDOW;
                uint64_t a = s.ts_window[a_idx];
                uint64_t b = s.ts_window[b_idx];
                if (b > a) gaps[gaps_n++] = b - a;
            }
            if (gaps_n >= 4) {
                std::sort(gaps, gaps + gaps_n);
                uint64_t p50 = gaps[gaps_n / 2];
                uint64_t p95 = gaps[(gaps_n * 95) / 100];

                // Jitter: current event's ITL is >= 5× the window median.
                // Robust to bimodal "fast burst + slow gap" patterns where a
                // simple Welford 2σ collapses (variance dominated by the
                // outliers, gating itself out).
                if (p50 > 0 && itl >= 5 * (double)p50) {
                    s.jitter_events++;
                    if (s.cache_aggressive.exchange(1, std::memory_order_release) == 0) {
                        g_jitter_total.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                // Smoothing: P95/P50 > BURST gate ⇒ session is bursty.
                if (p50 > 0 && (double)p95 / (double)p50 > BURST_P95_P50_GATE) {
                    if (!s.smoothed) {
                        s.smoothed = 1;
                        g_smoothed_total.fetch_add(1, std::memory_order_relaxed);
                        s.oracle_aggressive.store(1, std::memory_order_release);
                    }
                }
            }
        }
    }
    s.last_ts_ns = now;
}

} // namespace

extern "C" int cipher_shield_init(void) {
    int already = g_shield_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_shield_enabled.load(std::memory_order_relaxed);

    const char* env = std::getenv("CIPHER_SHIELD");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_shield_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr,
            "[CIPHER Op14] SHIELD enabled — ITL_WINDOW=%u P95/P50_gate=%.1f "
            "(v1: log-only enforcement; cache_aggressive/oracle_aggressive "
            "flags set but not yet consumed by Stage 0)\n",
            ITL_WINDOW, BURST_P95_P50_GATE);
    }
    return on;
}

extern "C" void cipher_shield_observe(const CipherRingEntry* ev) {
    if (!g_shield_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;

    uint64_t sid = cipher_sense_current_session();
    if (sid == 0) return;  // SENSE not yet finalized fingerprint

    ShieldSession* sp = find_or_alloc(sid);
    if (!sp) return;

    uint64_t now = ev->timestamp_ns;
    if (now == 0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        now = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }

    // Protection 1 — TTFT Guard: if SENSE says HUMAN_INTERACTIVE and we
    // haven't already assigned a band, write the protected band hint.
    if (!sp->priority_assigned) {
        if (cipher_sense_get_type(sid) == CIPHER_SESSION_HUMAN_INTERACTIVE) {
            cipher_sm_set_priority(sid, CIPHER_BAND_PROTECTED);
            sp->priority_assigned = 1;
            g_priority_total.fetch_add(1, std::memory_order_relaxed);
        }
    }

    update_itl(*sp, now);
}

// cipher_sm_priority_scan_log is defined in cipher_green_ctx.cu, which the
// vLLM-compatible "lite" build of libcipher_rt excludes. Mark the
// reference weak so the lite .so loads without it; check at call time.
extern "C" __attribute__((weak)) void cipher_sm_priority_scan_log(void);

extern "C" void cipher_shield_arbitrate_scan(void) {
    if (!g_shield_enabled.load(std::memory_order_relaxed)) return;
    if (cipher_sm_priority_scan_log) cipher_sm_priority_scan_log();
}

extern "C" unsigned cipher_shield_jitter_event_count(void) {
    return g_jitter_total.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_shield_smoothed_session_count(void) {
    return g_smoothed_total.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_shield_priority_assignments(void) {
    return g_priority_total.load(std::memory_order_relaxed);
}

extern "C" void cipher_shield_report(void) {
    if (!g_shield_enabled.load(std::memory_order_relaxed)) return;
    FILE* fp = std::fopen("/tmp/cipher_shield_report.json", "w");
    if (!fp) return;
    std::fprintf(fp,
        "{\n"
        "  \"jitter_events\": %u,\n"
        "  \"smoothed_sessions\": %u,\n"
        "  \"priority_assignments\": %u,\n"
        "  \"sessions\": [\n",
        g_jitter_total.load(std::memory_order_relaxed),
        g_smoothed_total.load(std::memory_order_relaxed),
        g_priority_total.load(std::memory_order_relaxed));
    bool first = true;
    for (unsigned i = 0; i < MAX_SHIELD_SESSIONS; ++i) {
        uint64_t sid = g_shield[i].session_id.load(std::memory_order_relaxed);
        if (sid == 0) continue;
        const ShieldSession& s = g_shield[i];
        double sd = (s.n_itl > 1) ? std::sqrt(s.itl_M2_ns / s.n_itl) : 0.0;
        if (!first) std::fprintf(fp, ",\n");
        first = false;
        std::fprintf(fp,
            "    {\"session_id\": \"%016llx\", \"itl_mean_us\": %.3f, "
            "\"itl_sd_us\": %.3f, \"n_itl\": %u, \"jitter_events\": %u, "
            "\"smoothed\": %u, \"priority_assigned\": %u, "
            "\"band\": %u}",
            (unsigned long long)sid,
            s.itl_mean_ns / 1e3, sd / 1e3, s.n_itl,
            s.jitter_events, s.smoothed, s.priority_assigned,
            cipher_sm_get_priority(sid));
    }
    std::fprintf(fp, "\n  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op14] SHIELD report -> /tmp/cipher_shield_report.json\n");
}
