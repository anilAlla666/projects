// Op 15 SUSTAIN — KV-cache pressure detector (Stage 1 inline).
//
// Per session, maintains a 20-pt rolling window of (decode_step_index, ITL_ns)
// and recomputes a closed-form OLS slope every 10 decode events. When
// slope > threshold, sets sustain_compress_flag (atomic; not yet consumed
// by Stage 0 — wiring deferred).
//
// All inputs come from the ring entry: `timestamp_ns` is reliable post the
// Op-14 geometry plumb-through, so SUSTAIN does NOT call clock_gettime.

#include "cipher_sustain.h"
#include "cipher_sense.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr unsigned MAX_SUSTAIN_SESSIONS = 256;
constexpr unsigned WIN                  = 20;
constexpr unsigned EVAL_EVERY           = 10;
constexpr double   SLOPE_THRESHOLD_NS   = 50000.0;        // 0.5 ms / 10 steps
constexpr double   AGENT_RELAX_FACTOR   = 3.0;
constexpr uint32_t AGENT_DECODE_GATE    = 500;            // matches spec

struct SustainSession {
    std::atomic<uint64_t> session_id;             // 0 = empty
    uint64_t              prev_ts_ns;
    uint32_t              decode_step_idx;        // monotonic per session
    uint16_t              step_buf[WIN];
    uint64_t              itl_buf[WIN];           // ns
    uint8_t               head, full;
    uint32_t              steps_since_eval;
    double                last_slope;             // ns/step
    std::atomic<uint8_t>  sustain_compress;       // for v2 Stage 0 hook
    uint32_t              pressure_events;
};

alignas(64) SustainSession g_sustain[MAX_SUSTAIN_SESSIONS];

std::atomic<int>      g_sustain_enabled{0};
std::atomic<int>      g_sustain_initialized{0};
std::atomic<unsigned> g_pressure_total{0};
std::atomic<unsigned> g_compress_flag_total{0};

inline unsigned slot_idx(uint64_t sid) {
    uint64_t mix = sid ^ (sid >> 32);
    return (unsigned)((mix * 0x94D049BB133111EBULL) >> 56) % MAX_SUSTAIN_SESSIONS;
}

SustainSession* find_or_alloc(uint64_t sid) {
    if (sid == 0) return nullptr;
    unsigned start = slot_idx(sid);
    for (unsigned probe = 0; probe < 8; ++probe) {
        unsigned i = (start + probe) & (MAX_SUSTAIN_SESSIONS - 1);
        uint64_t cur = g_sustain[i].session_id.load(std::memory_order_acquire);
        if (cur == sid) return &g_sustain[i];
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_sustain[i].session_id.compare_exchange_strong(
                    expected, sid, std::memory_order_acq_rel)) {
                SustainSession& s = g_sustain[i];
                s.prev_ts_ns = 0;
                s.decode_step_idx = 0;
                std::memset(s.step_buf, 0, sizeof(s.step_buf));
                std::memset(s.itl_buf,  0, sizeof(s.itl_buf));
                s.head = 0; s.full = 0;
                s.steps_since_eval = 0;
                s.last_slope = 0.0;
                s.sustain_compress.store(0, std::memory_order_relaxed);
                s.pressure_events = 0;
                return &s;
            }
            return &g_sustain[i];
        }
    }
    return nullptr;
}

inline void unpack_mkn(uint64_t ph, uint32_t& M, uint32_t& K, uint32_t& N) {
    if ((ph >> 60) != 0xCULL) { M = K = N = 0; return; }
    M = (uint32_t)((ph >> 40) & 0xFFFFFu);
    K = (uint32_t)((ph >> 20) & 0xFFFFFu);
    N = (uint32_t)( ph        & 0xFFFFFu);
}

double regress_slope_ns_per_step(const SustainSession& s) {
    unsigned n = s.full ? WIN : s.head;
    if (n < 4) return 0.0;
    long double sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (unsigned i = 0; i < n; ++i) {
        long double x = (long double)s.step_buf[i];
        long double y = (long double)s.itl_buf[i];
        sx  += x;
        sy  += y;
        sxy += x * y;
        sxx += x * x;
    }
    long double denom = (long double)n * sxx - sx * sx;
    if (denom <= 0) return 0.0;
    return (double)(((long double)n * sxy - sx * sy) / denom);
}

} // namespace

extern "C" int cipher_sustain_init(void) {
    int already = g_sustain_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_sustain_enabled.load(std::memory_order_relaxed);

    const char* env = std::getenv("CIPHER_SUSTAIN");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_sustain_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr,
            "[CIPHER Op15] SUSTAIN enabled — window=%u eval_every=%u "
            "threshold=%.0f ns/step (AGENT relax=%.1fx if decodes>%u). "
            "v1: sustain_compress flag set but not yet consumed by Stage 0.\n",
            WIN, EVAL_EVERY, SLOPE_THRESHOLD_NS,
            AGENT_RELAX_FACTOR, AGENT_DECODE_GATE);
    }
    return on;
}

extern "C" void cipher_sustain_observe(const CipherRingEntry* ev) {
    if (!g_sustain_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;
    if (ev->kernel_class != 0) return;     // GEMM only

    uint32_t M, K, N;
    unpack_mkn(ev->params_hash, M, K, N);
    if (M == 0 && N == 0) return;          // no geometry — skip

    uint32_t batch = (M && N) ? (M < N ? M : N) : (M | N);
    if (batch > 8) return;                 // not a decode step

    uint64_t sid = cipher_sense_current_session();
    SustainSession* sp = find_or_alloc(sid);
    if (!sp) return;
    SustainSession& s = *sp;

    uint64_t now = ev->timestamp_ns;        // reliable post Op-14 plumb-through
    if (now == 0 || s.prev_ts_ns == 0 || now <= s.prev_ts_ns) {
        s.prev_ts_ns = now;
        return;
    }

    uint64_t itl = now - s.prev_ts_ns;
    s.prev_ts_ns = now;

    // Append (step_idx, itl) to rolling window
    s.step_buf[s.head] = (uint16_t)(s.decode_step_idx & 0xFFFFu);
    s.itl_buf [s.head] = itl;
    s.head = (uint8_t)((s.head + 1u) % WIN);
    if (s.head == 0) s.full = 1;
    s.decode_step_idx++;
    s.steps_since_eval++;

    if (s.steps_since_eval < EVAL_EVERY) return;
    s.steps_since_eval = 0;

    double slope = regress_slope_ns_per_step(s);
    s.last_slope = slope;

    // AGENT exception: 3× threshold for long reasoning sessions
    double threshold = SLOPE_THRESHOLD_NS;
    if (sid != 0
        && cipher_sense_get_type(sid) == CIPHER_SESSION_AGENT_AUTONOMOUS
        && s.decode_step_idx > AGENT_DECODE_GATE) {
        threshold *= AGENT_RELAX_FACTOR;
    }

    if (slope > threshold) {
        s.pressure_events++;
        g_pressure_total.fetch_add(1, std::memory_order_relaxed);
        if (s.sustain_compress.exchange(1, std::memory_order_release) == 0) {
            g_compress_flag_total.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

extern "C" unsigned cipher_sustain_pressure_event_count(void) {
    return g_pressure_total.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_sustain_compress_flag_count(void) {
    return g_compress_flag_total.load(std::memory_order_relaxed);
}

extern "C" double cipher_sustain_last_slope_ns_per_step(uint64_t session_id) {
    if (session_id == 0) return 0.0;
    unsigned start = slot_idx(session_id);
    for (unsigned probe = 0; probe < 8; ++probe) {
        unsigned i = (start + probe) & (MAX_SUSTAIN_SESSIONS - 1);
        uint64_t cur = g_sustain[i].session_id.load(std::memory_order_acquire);
        if (cur == session_id) return g_sustain[i].last_slope;
        if (cur == 0) return 0.0;
    }
    return 0.0;
}

extern "C" void cipher_sustain_report(void) {
    if (!g_sustain_enabled.load(std::memory_order_relaxed)) return;
    FILE* fp = std::fopen("/tmp/cipher_sustain_report.json", "w");
    if (!fp) return;
    std::fprintf(fp,
        "{\n"
        "  \"pressure_events\": %u,\n"
        "  \"compress_flag_set\": %u,\n"
        "  \"sessions\": [\n",
        g_pressure_total.load(std::memory_order_relaxed),
        g_compress_flag_total.load(std::memory_order_relaxed));
    bool first = true;
    for (unsigned i = 0; i < MAX_SUSTAIN_SESSIONS; ++i) {
        uint64_t sid = g_sustain[i].session_id.load(std::memory_order_relaxed);
        if (sid == 0) continue;
        const SustainSession& s = g_sustain[i];
        if (!first) std::fprintf(fp, ",\n");
        first = false;
        std::fprintf(fp,
            "    {\"session_id\": \"%016llx\", \"decode_steps\": %u, "
            "\"last_slope_ns_per_step\": %.1f, \"pressure_events\": %u, "
            "\"compress_flag\": %u}",
            (unsigned long long)sid,
            s.decode_step_idx, s.last_slope, s.pressure_events,
            s.sustain_compress.load(std::memory_order_relaxed));
    }
    std::fprintf(fp, "\n  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op15] SUSTAIN report -> /tmp/cipher_sustain_report.json\n");
}
