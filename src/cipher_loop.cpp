// Op 26 LOOP — Agentic runaway detection (Stage 1).
//
// Three signals per session:
//   S1 shape-cycle repetition — autocorrelation over a 64-deep shape ring
//   S2 decode burn-rate vs the session's own first-100-event baseline
//   S3 prefill drought (events_since_last_prefill > 500) AND S1 active
//
// Score = S1+S2+S3 (0..3). Score ≥ 2 ⇒ runaway. Scoring is sticky-high:
// once a session reaches the runaway threshold it latches (prevents flapping).

#include "cipher_loop.h"
#include "cipher_sense.h"
#include "cipher_op_counters.h"

extern "C" int cipher_comply_audit(const char* kind, const char* json_kv);

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

namespace {

constexpr unsigned MAX_SESSIONS        = 1024;
constexpr unsigned SHAPE_WIN           = 64;   // shape ring depth
constexpr unsigned MAX_PERIOD          = 8;    // longest cycle we detect
constexpr unsigned MIN_CYCLES          = 3;    // cycles required to flag S1
constexpr uint64_t IDLE_RESET_NS       = 200ULL * 1000 * 1000; // 200 ms — mirror SENSE
constexpr uint32_t PREFILL_DROUGHT_MIN = 500;
constexpr uint32_t BURN_RATIO_MIN      = 500;  // decodes-per-prefill ratio
constexpr uint32_t SCORE_MIN_EVENTS    = 100;  // don't score until N events seen

struct LoopSession {
    std::atomic<uint64_t> fingerprint;    // 0 = empty slot
    std::atomic<uint8_t>  score;          // 0..3, monotonic-high while runaway

    uint64_t shapes[SHAPE_WIN];
    uint32_t shape_head;                  // write cursor
    uint32_t shape_n;                     // events seen (caps at SHAPE_WIN)

    uint32_t event_count;                 // total events in session
    uint32_t decode_count;
    uint32_t prefill_count;
    uint32_t events_since_last_prefill;

    uint64_t first_ts_ns, last_ts_ns;

    // Signal latches (monotonic — once set, stay set for the session).
    uint8_t  s1_cycle, s2_burn, s3_drought;
    uint32_t detected_period;             // 0 or 1..MAX_PERIOD
};

alignas(64) LoopSession g_slots[MAX_SESSIONS];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned> g_session_count{0};
std::atomic<unsigned> g_runaway_count{0};
std::atomic<int>      g_current_slot{-1};
std::atomic<uint64_t> g_current_fp{0};
std::atomic<uint64_t> g_last_ts_ns{0};

uint64_t fnv1a64(const uint8_t* data, size_t n, uint64_t seed = 0xcbf29ce484222325ULL) {
    uint64_t h = seed;
    for (size_t i = 0; i < n; ++i) { h ^= data[i]; h *= 0x100000001b3ULL; }
    return h;
}

uint64_t shape_proxy(const CipherRingEntry* ev) {
    return ev->params_hash ? ev->params_hash : ev->func_ptr_hash;
}

int allocate_slot(uint64_t seed) {
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t expected = 0;
        if (g_slots[i].fingerprint.compare_exchange_strong(
                expected, seed, std::memory_order_acq_rel)) {
            LoopSession& s = g_slots[i];
            s.score.store(0, std::memory_order_relaxed);
            std::memset(s.shapes, 0, sizeof(s.shapes));
            s.shape_head = s.shape_n = 0;
            s.event_count = s.decode_count = s.prefill_count = 0;
            s.events_since_last_prefill = 0;
            s.first_ts_ns = s.last_ts_ns = 0;
            s.s1_cycle = s.s2_burn = s.s3_drought = 0;
            s.detected_period = 0;
            g_session_count.fetch_add(1, std::memory_order_relaxed);
            cipher_op_inc(OP_LOOP);
            return (int)i;
        }
    }
    return -1;
}

// Detect repeating period p (1..MAX_PERIOD) in the last W shapes. Requires
// MIN_CYCLES full repetitions. Returns period p or 0 if none.
uint32_t detect_period(const LoopSession& s) {
    if (s.shape_n < MAX_PERIOD * MIN_CYCLES) return 0;
    // Materialize the last W entries in order (oldest..newest).
    uint64_t buf[SHAPE_WIN];
    uint32_t W = s.shape_n;
    uint32_t start = (s.shape_head + SHAPE_WIN - W) % SHAPE_WIN;
    for (uint32_t i = 0; i < W; ++i) buf[i] = s.shapes[(start + i) % SHAPE_WIN];

    for (uint32_t p = 1; p <= MAX_PERIOD; ++p) {
        if (W < p * MIN_CYCLES) continue;
        uint32_t check = p * MIN_CYCLES;
        bool ok = true;
        // Compare last `check` elements against themselves shifted by p.
        for (uint32_t i = W - check; i + p < W; ++i) {
            if (buf[i] != buf[i + p]) { ok = false; break; }
        }
        if (ok) return p;
    }
    return 0;
}

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

inline void unpack_mkn(uint64_t ph, uint32_t& M, uint32_t& K, uint32_t& N) {
    if ((ph >> 60) != 0xCULL) { M = K = N = 0; return; }
    M = (uint32_t)((ph >> 40) & 0xFFFFFu);
    K = (uint32_t)((ph >> 20) & 0xFFFFFu);
    N = (uint32_t)( ph        & 0xFFFFFu);
}

} // namespace

extern "C" int cipher_loop_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_LOOP");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op26] LOOP enabled — slots=%u win=%u "
                             "max_period=%u min_cycles=%u burn_ratio=%u drought=%u\n",
                     MAX_SESSIONS, SHAPE_WIN, MAX_PERIOD, MIN_CYCLES,
                     BURN_RATIO_MIN, PREFILL_DROUGHT_MIN);
    }
    return on;
}

extern "C" void cipher_loop_observe(const CipherRingEntry* ev) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;

    uint64_t t = ev->timestamp_ns;
    if (t == 0) t = now_ns();
    uint64_t prev = g_last_ts_ns.exchange(t, std::memory_order_relaxed);

    int slot = g_current_slot.load(std::memory_order_relaxed);
    bool start_new = (slot < 0) ||
                     (prev != 0 && t > prev && (t - prev) > IDLE_RESET_NS);
    if (start_new) {
        uint64_t seed = fnv1a64((const uint8_t*)&t, sizeof(t),
                                0xcbf29ce484222325ULL ^ ev->func_ptr_hash);
        slot = allocate_slot(seed);
        if (slot < 0) return;
        g_current_slot.store(slot, std::memory_order_relaxed);
        g_current_fp.store(seed, std::memory_order_relaxed);
        LoopSession& ns = g_slots[slot];
        ns.first_ts_ns = t;
    }

    LoopSession& s = g_slots[slot];
    s.last_ts_ns = t;
    s.event_count++;

    // Shape ring push.
    uint64_t sp = shape_proxy(ev);
    s.shapes[s.shape_head] = sp;
    s.shape_head = (s.shape_head + 1) % SHAPE_WIN;
    if (s.shape_n < SHAPE_WIN) s.shape_n++;

    // Prefill / decode accounting (same M-based rule as SENSE).
    if (ev->kernel_class == 0) {
        uint32_t M, K, N; unpack_mkn(ev->params_hash, M, K, N);
        uint32_t batch = (M && N) ? (M < N ? M : N) : (M | N);
        bool is_decode  = (batch > 0 && batch <= 8);
        bool is_prefill = (batch >= 256);
        if (is_decode) {
            s.decode_count++;
            s.events_since_last_prefill++;
        }
        if (is_prefill) {
            s.prefill_count++;
            s.events_since_last_prefill = 0;
        }
    } else {
        s.events_since_last_prefill++;
    }

    // Score signals (evaluate every 16 events after warmup).
    if (s.event_count >= SCORE_MIN_EVENTS && (s.event_count % 16) == 0) {
        // S1 cycle.
        uint32_t p = detect_period(s);
        if (p > 0) { s.s1_cycle = 1; s.detected_period = p; }

        // S2 decode-to-prefill ratio (session-wide). Treat zero prefills as
        // one for the denominator so the very first decode burst doesn't
        // trivially saturate until SCORE_MIN_EVENTS has passed.
        uint32_t denom = s.prefill_count ? s.prefill_count : 1;
        if (s.decode_count / denom > BURN_RATIO_MIN) s.s2_burn = 1;

        // S3 prefill drought (gated by S1).
        if (s.s1_cycle && s.events_since_last_prefill > PREFILL_DROUGHT_MIN) {
            s.s3_drought = 1;
        }

        uint8_t newscore = (uint8_t)(s.s1_cycle + s.s2_burn + s.s3_drought);
        uint8_t oldscore = s.score.load(std::memory_order_relaxed);
        if (newscore > oldscore) {
            s.score.store(newscore, std::memory_order_release);
            if (oldscore < 2 && newscore >= 2) {
                g_runaway_count.fetch_add(1, std::memory_order_relaxed);
                std::fprintf(stderr,
                    "[CIPHER Op26] RUNAWAY detected — slot=%d period=%u "
                    "decodes=%u since_prefill=%u score=%u\n",
                    slot, s.detected_period, s.decode_count,
                    s.events_since_last_prefill, newscore);
                char kv[160];
                std::snprintf(kv, sizeof(kv),
                    "{\"slot\":%d,\"period\":%u,\"decodes\":%u,\"since_prefill\":%u,\"score\":%u}",
                    slot, s.detected_period, s.decode_count,
                    s.events_since_last_prefill, newscore);
                cipher_comply_audit("loop_runaway", kv);
            }
        }
    }
}

extern "C" int cipher_loop_get_score(uint64_t fingerprint) {
    if (!g_enabled.load(std::memory_order_relaxed)) return -1;
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t fp = g_slots[i].fingerprint.load(std::memory_order_acquire);
        if (fp == fingerprint && fp != 0) {
            return (int)g_slots[i].score.load(std::memory_order_acquire);
        }
    }
    return -1;
}

extern "C" unsigned cipher_loop_session_count(void) {
    return g_session_count.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_loop_runaway_count(void) {
    return g_runaway_count.load(std::memory_order_relaxed);
}

// OP 29 — emit aggregate stats. Walks slot table once.
extern "C" int cipher_loop_stats(CipherLoopStats* out) {
    if (!out) return 0;
    out->enabled              = g_enabled.load(std::memory_order_relaxed);
    out->session_count        = g_session_count.load(std::memory_order_relaxed);
    out->runaway_count        = g_runaway_count.load(std::memory_order_relaxed);
    out->current_fingerprint  = g_current_fp.load(std::memory_order_relaxed);
    unsigned active = 0, demoted = 0, max_score = 0;
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t fp = g_slots[i].fingerprint.load(std::memory_order_acquire);
        if (fp == 0) continue;
        active++;
        uint8_t sc = g_slots[i].score.load(std::memory_order_acquire);
        if (sc > max_score) max_score = sc;
        if (sc >= 2) demoted++;
    }
    out->active_sessions   = active;
    out->max_score_seen    = max_score;
    out->demoted_sessions  = demoted;
    return 1;
}

extern "C" void cipher_loop_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    FILE* fp = std::fopen("/tmp/cipher_loop_report.json", "w");
    if (!fp) return;

    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"session_count\": %u,\n",
                 g_session_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"runaway_count\": %u,\n",
                 g_runaway_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"current_fingerprint\": \"%016llx\",\n",
                 (unsigned long long)g_current_fp.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"sessions\": [\n");
    bool first = true;
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t fp64 = g_slots[i].fingerprint.load(std::memory_order_acquire);
        if (fp64 == 0) continue;
        const LoopSession& s = g_slots[i];
        uint8_t sc = s.score.load(std::memory_order_acquire);
        if (!first) std::fprintf(fp, ",\n");
        first = false;
        std::fprintf(fp,
            "    {\"slot\": %u, \"fingerprint\": \"%016llx\", \"score\": %u,"
            " \"runaway\": %s, \"period\": %u,"
            " \"events\": %u, \"decodes\": %u, \"prefills\": %u,"
            " \"since_prefill\": %u,"
            " \"s1_cycle\": %u, \"s2_burn\": %u, \"s3_drought\": %u}",
            i, (unsigned long long)fp64, sc, (sc >= 2 ? "true" : "false"),
            s.detected_period, s.event_count, s.decode_count, s.prefill_count,
            s.events_since_last_prefill,
            s.s1_cycle, s.s2_burn, s.s3_drought);
    }
    std::fprintf(fp, "\n  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op26] LOOP report -> /tmp/cipher_loop_report.json\n");
}
