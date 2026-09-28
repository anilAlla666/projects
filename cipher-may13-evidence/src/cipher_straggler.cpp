// Op (Phase 3) — Straggler Detection (local v1).
//
// Per-AllReduce telemetry, per-rank labeled, written to
// /tmp/cipher_straggler_rank_<RANK>.jsonl. Local slowdown detector via
// per-bucket EMA. Algorithm hint to bias ORCHESTRATE toward RING when
// local slowdown sustains.
//
// Cross-rank rank attribution: NOT IMPLEMENTED — see header comment for
// honest spec deviation. tools/straggler_aggregate.py performs the
// cross-rank pass offline.

#include "cipher_straggler.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace {

constexpr int      NUM_BUCKETS         = 7;
constexpr double   EMA_ALPHA           = 0.10;     // 0.9 / 0.1 EMA
constexpr double   SLOWDOWN_FACTOR     = 1.50;     // current > 1.5 * EMA
constexpr unsigned SLOWDOWN_RUN_GATE   = 5;        // sustained N consecutive calls
constexpr unsigned SLOWDOWN_CLEAR_RUN  = 5;        // back-to-normal N consecutive
constexpr uint64_t REALERT_PERIOD_NS   = 60ULL * 1000 * 1000 * 1000;

// Bucket upper bounds matching cipher_nccl.cpp's g_buckets layout
constexpr uint64_t BUCKET_UB[NUM_BUCKETS] = {
    1024ULL,
    64ULL * 1024,
    1024ULL * 1024,
    16ULL * 1024 * 1024,
    128ULL * 1024 * 1024,
    1024ULL * 1024 * 1024,
    UINT64_MAX
};

constexpr int  NCCL_ALGO_RING = 1;   // matches cipher_nccl_bpf.h's CipherNcclAlgo

struct BucketStats {
    double   ema_ns_per_byte;
    uint64_t calls;
    uint32_t consec_slow;        // consecutive calls above slowdown gate
    uint32_t consec_normal;      // consecutive calls within band (reset clear gate)
    uint64_t inject_dur_ns;      // test override (per-bucket, last-set wins)
};

struct StragState {
    std::atomic<int>      mode;                    // CipherStragglerMode
    std::atomic<int>      slowdown_active;
    std::atomic<unsigned> event_count;
    int                   my_rank;                 // -1 = unknown
    int                   world_size;              // 0 = unknown
    char                  rank_log_path[128];      // /tmp/cipher_straggler_rank_<R>.jsonl
    BucketStats           buckets[NUM_BUCKETS];
    uint64_t              last_alert_ns;
};

StragState g_strag = {};
std::atomic<int> g_strag_initialized{0};

inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int bucket_index_for(uint64_t bytes) {
    for (int i = 0; i < NUM_BUCKETS; ++i)
        if (bytes <= BUCKET_UB[i]) return i;
    return NUM_BUCKETS - 1;
}

int read_rank_from_env() {
    const char* candidates[] = {"RANK", "OMPI_COMM_WORLD_RANK",
                                "SLURM_PROCID", "PMI_RANK", nullptr};
    for (int i = 0; candidates[i]; ++i) {
        const char* v = std::getenv(candidates[i]);
        if (v && *v) return std::atoi(v);
    }
    return -1;
}
int read_world_size_from_env() {
    const char* candidates[] = {"WORLD_SIZE", "OMPI_COMM_WORLD_SIZE",
                                "SLURM_NTASKS", "PMI_SIZE", nullptr};
    for (int i = 0; candidates[i]; ++i) {
        const char* v = std::getenv(candidates[i]);
        if (v && *v) return std::atoi(v);
    }
    return 0;
}

void emit_jsonl_line(uint64_t bytes, uint64_t dur_ns, int algo, int bi,
                     double ns_per_byte, double ema, bool slowdown_now)
{
    if (!g_strag.rank_log_path[0]) return;
    FILE* fp = std::fopen(g_strag.rank_log_path, "a");
    if (!fp) return;
    std::fprintf(fp,
        "{\"ts_ns\": %llu, \"rank\": %d, \"world_size\": %d, "
        "\"bytes\": %llu, \"dur_ns\": %llu, \"algo\": %d, "
        "\"bucket\": %d, \"ns_per_byte\": %.4f, \"ema_ns_per_byte\": %.4f, "
        "\"slowdown\": %s}\n",
        (unsigned long long)now_ns(), g_strag.my_rank, g_strag.world_size,
        (unsigned long long)bytes, (unsigned long long)dur_ns, algo,
        bi, ns_per_byte, ema, slowdown_now ? "true" : "false");
    std::fclose(fp);
}

void emit_alert_json(int bucket_idx, double ns_per_byte, double ema) {
    FILE* fp = std::fopen("/tmp/cipher_straggler_alert.json", "w");
    if (!fp) return;
    std::fprintf(fp,
        "{\n"
        "  \"timestamp_ns\": %llu,\n"
        "  \"rank\": %d,\n"
        "  \"world_size\": %d,\n"
        "  \"local_slowdown\": true,\n"
        "  \"bucket\": %d,\n"
        "  \"current_ns_per_byte\": %.4f,\n"
        "  \"ema_ns_per_byte\": %.4f,\n"
        "  \"slowdown_factor\": %.2f,\n"
        "  \"rank_attribution\": \"local_observation_only\",\n"
        "  \"requires_aggregator\": true,\n"
        "  \"suggested_aggregator_command\": \"python3 tools/straggler_aggregate.py /tmp/cipher_straggler_rank_*.jsonl\",\n"
        "  \"deferred_reason\": \"NCCL tuner v2 ABI does not expose per-rank completion timing; cross-rank rank attribution requires offline coordination\"\n"
        "}\n",
        (unsigned long long)now_ns(),
        g_strag.my_rank, g_strag.world_size,
        bucket_idx, ns_per_byte, ema, SLOWDOWN_FACTOR);
    std::fclose(fp);
}

} // namespace

extern "C" int cipher_straggler_init(void) {
    int already = g_strag_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_strag.mode.load(std::memory_order_relaxed);

    const char* env = std::getenv("CIPHER_STRAGGLER");
    int mode = CIPHER_STRAGGLER_OFF;
    if (env) {
        if (std::strcmp(env, "observe") == 0) mode = CIPHER_STRAGGLER_OBSERVE;
        else if (std::strcmp(env, "active") == 0
              || std::strcmp(env, "on") == 0
              || std::strcmp(env, "1") == 0)  mode = CIPHER_STRAGGLER_ACTIVE;
    }
    g_strag.mode.store(mode, std::memory_order_release);
    if (mode == CIPHER_STRAGGLER_OFF) return mode;

    g_strag.my_rank    = read_rank_from_env();
    g_strag.world_size = read_world_size_from_env();
    int rank_label = (g_strag.my_rank >= 0) ? g_strag.my_rank : 0;
    std::snprintf(g_strag.rank_log_path, sizeof(g_strag.rank_log_path),
                  "/tmp/cipher_straggler_rank_%d.jsonl", rank_label);

    // Truncate log on (re)start to avoid mixing runs
    FILE* fp = std::fopen(g_strag.rank_log_path, "w");
    if (fp) std::fclose(fp);

    std::fprintf(stderr,
        "[CIPHER PhaseA-Strg] Straggler %s — rank=%d (%s), world_size=%d, "
        "log=%s. Local slowdown detector: > %.1fx EMA over %u calls. "
        "Cross-rank attribution DEFERRED (run tools/straggler_aggregate.py "
        "across rank logs).\n",
        (mode == CIPHER_STRAGGLER_ACTIVE) ? "ACTIVE" : "OBSERVE",
        g_strag.my_rank,
        g_strag.my_rank >= 0 ? "from env" : "local-only (no $RANK env)",
        g_strag.world_size,
        g_strag.rank_log_path,
        SLOWDOWN_FACTOR, SLOWDOWN_RUN_GATE);
    return mode;
}

extern "C" void cipher_straggler_observe(uint64_t bytes, uint64_t dur_ns,
                                          int chosen_algo) {
    int mode = g_strag.mode.load(std::memory_order_relaxed);
    if (mode == CIPHER_STRAGGLER_OFF) return;
    if (bytes == 0) return;

    int bi = bucket_index_for(bytes);
    BucketStats& b = g_strag.buckets[bi];

    // Test injection: if a per-bucket inject is set, override dur_ns once.
    if (b.inject_dur_ns) {
        dur_ns = b.inject_dur_ns;
        b.inject_dur_ns = 0;   // single-shot per inject call
    }

    double ns_per_byte = (double)dur_ns / (double)bytes;
    double ema_before  = b.ema_ns_per_byte;
    if (b.calls == 0) {
        b.ema_ns_per_byte = ns_per_byte;
    } else {
        b.ema_ns_per_byte = (1.0 - EMA_ALPHA) * b.ema_ns_per_byte
                          + EMA_ALPHA * ns_per_byte;
    }
    b.calls++;

    bool slow_now = false;
    if (b.calls > 5 && ema_before > 0.0
        && ns_per_byte > SLOWDOWN_FACTOR * ema_before) {
        b.consec_slow++;
        b.consec_normal = 0;
        if (b.consec_slow >= SLOWDOWN_RUN_GATE) {
            slow_now = true;
            int prev = g_strag.slowdown_active.exchange(1, std::memory_order_release);
            if (prev == 0) {
                g_strag.event_count.fetch_add(1, std::memory_order_relaxed);
                uint64_t t = now_ns();
                if (t - g_strag.last_alert_ns > REALERT_PERIOD_NS) {
                    g_strag.last_alert_ns = t;
                    emit_alert_json(bi, ns_per_byte, ema_before);
                    std::fprintf(stderr,
                        "[CIPHER PhaseA-Strg] LOCAL SLOWDOWN rank=%d bucket=%d "
                        "%.3f vs ema %.3f ns/byte → /tmp/cipher_straggler_alert.json "
                        "(rank attribution DEFERRED to aggregator)\n",
                        g_strag.my_rank, bi, ns_per_byte, ema_before);
                }
            }
        }
    } else {
        b.consec_slow = 0;
        b.consec_normal++;
        if (b.consec_normal >= SLOWDOWN_CLEAR_RUN
            && g_strag.slowdown_active.load(std::memory_order_relaxed)) {
            g_strag.slowdown_active.store(0, std::memory_order_release);
        }
    }

    emit_jsonl_line(bytes, dur_ns, chosen_algo, bi,
                    ns_per_byte, ema_before, slow_now);
}

extern "C" int cipher_straggler_local_slowdown_active(void) {
    return g_strag.slowdown_active.load(std::memory_order_acquire);
}
extern "C" unsigned cipher_straggler_event_count(void) {
    return g_strag.event_count.load(std::memory_order_relaxed);
}
extern "C" int cipher_straggler_my_rank(void)    { return g_strag.my_rank; }
extern "C" int cipher_straggler_world_size(void) { return g_strag.world_size; }
extern "C" const char* cipher_straggler_status_string(void) {
    switch (g_strag.mode.load(std::memory_order_relaxed)) {
        case CIPHER_STRAGGLER_OFF:     return "OFF";
        case CIPHER_STRAGGLER_OBSERVE: return "OBSERVE";
        case CIPHER_STRAGGLER_ACTIVE:  return "ACTIVE";
    }
    return "UNKNOWN";
}

extern "C" int cipher_straggler_algo_hint(uint64_t bytes) {
    if (g_strag.mode.load(std::memory_order_relaxed) != CIPHER_STRAGGLER_ACTIVE)
        return -1;
    if (!g_strag.slowdown_active.load(std::memory_order_relaxed)) return -1;
    (void)bytes;   // bucket-agnostic for v1; RING is the safe tail choice
    return NCCL_ALGO_RING;
}

extern "C" void cipher_straggler_inject_dur(uint64_t bytes, uint64_t dur_ns_inflated) {
    int bi = bucket_index_for(bytes);
    g_strag.buckets[bi].inject_dur_ns = dur_ns_inflated;
}
extern "C" void cipher_straggler_clear_injection(void) {
    for (int i = 0; i < NUM_BUCKETS; ++i) g_strag.buckets[i].inject_dur_ns = 0;
}

extern "C" void cipher_straggler_report(void) {
    if (g_strag.mode.load(std::memory_order_relaxed) == CIPHER_STRAGGLER_OFF) return;
    FILE* fp = std::fopen("/tmp/cipher_straggler_report.json", "w");
    if (!fp) return;
    std::fprintf(fp,
        "{\n"
        "  \"mode\": \"%s\",\n"
        "  \"rank\": %d,\n"
        "  \"world_size\": %d,\n"
        "  \"slowdown_active\": %d,\n"
        "  \"event_count\": %u,\n"
        "  \"rank_log\": \"%s\",\n"
        "  \"buckets\": [\n",
        cipher_straggler_status_string(),
        g_strag.my_rank, g_strag.world_size,
        g_strag.slowdown_active.load(std::memory_order_relaxed),
        g_strag.event_count.load(std::memory_order_relaxed),
        g_strag.rank_log_path);
    bool first = true;
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        const BucketStats& b = g_strag.buckets[i];
        if (b.calls == 0) continue;
        if (!first) std::fprintf(fp, ",\n");
        first = false;
        std::fprintf(fp,
            "    {\"bucket\": %d, \"calls\": %llu, \"ema_ns_per_byte\": %.4f, "
            "\"consec_slow\": %u}",
            i, (unsigned long long)b.calls, b.ema_ns_per_byte, b.consec_slow);
    }
    std::fprintf(fp, "\n  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER PhaseA-Strg] report -> /tmp/cipher_straggler_report.json\n");
}
