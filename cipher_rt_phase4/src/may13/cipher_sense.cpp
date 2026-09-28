// Op 13 SENSE — Session classification (Stage 1).
//
// Reads ring entries observed by Stage 1, infers session boundaries from
// inter-event idle gaps, fingerprints sessions by the first 8 distinct GEMM
// shape proxies, and re-classifies session_type periodically.
//
// As of the Op-14 geometry plumb-through, hook-DSO ring entries populate
// grid_x/y/z, block_x/y/z, timestamp_ns, func_ptr_hash, and a packed
// (M, K, N) inside params_hash with tag 0xC in the top 4 bits. SENSE uses
// min(M, N) as the batch-dimension proxy (cuBLAS's column-major convention
// puts PyTorch batch into N_cublas; row-major path puts it into M).
//
// v1 limitation still standing: no stream_id in ring entries — concurrent
// sessions on one GPU collapse into one logical session. Resolved by adding
// stream_id to CipherRingEntry in a future plan (separate Stage 0 change).

#include "may13/cipher_sense.h"

#include "cipher_rt_commit.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cmath>

namespace {

// Tunables (compile-time defaults; env-overridable in init).
constexpr unsigned MAX_SESSIONS         = 1024;
constexpr uint64_t IDLE_THRESHOLD_NS    = 200ULL * 1000 * 1000;   // 200 ms
constexpr uint64_t HUMAN_GAP_MIN_NS     = 10ULL * 1000 * 1000;    // 10 ms
constexpr uint64_t HUMAN_GAP_MAX_NS     = 500ULL * 1000 * 1000;   // 500 ms
constexpr uint64_t AGENT_PREFILL_GAP_NS = 50ULL * 1000 * 1000;    // 50 ms
constexpr uint32_t AGENT_DECODE_MIN     = 1000;
constexpr uint32_t BATCH_PREFILL_MIN    = 3;
constexpr uint64_t BATCH_WINDOW_NS      = 100ULL * 1000 * 1000;   // 100 ms
constexpr uint32_t RECLASS_DECODE_EVERY = 10;
// Grid-size proxy: GEMM with grid_x*grid_y < SMALL_GRID treated as decode (M small).
// cuBLAS tile sizes are typically 64-256, so a B=1 decode lands at gx*gy < 32.
constexpr uint64_t SMALL_GRID_THRESH    = 32;
constexpr uint64_t LARGE_GRID_THRESH    = 256;

struct SessionState {
    std::atomic<uint64_t>     fingerprint;       // 0 = empty slot
    std::atomic<uint8_t>      session_type;      // CipherSessionType
    uint64_t                  first_seq, last_seq;
    uint64_t                  first_ts_ns, last_ts_ns;
    uint64_t                  first_8_shapes[8]; // params_hash values
    uint8_t                   n_shapes_seen;
    uint32_t                  prefill_count;
    uint32_t                  decode_count;
    uint64_t                  inter_event_min_ns;
    uint64_t                  inter_event_max_ns;
    double                    inter_event_sum_ns;
    uint32_t                  n_inter_event;
    // Prefill arrival timing (for AGENT request-arrival variance).
    uint32_t                  prefill_arrival_n;
    double                    prefill_gap_sum_ns;
    double                    prefill_gap_sum_sq_ns;
    uint64_t                  last_prefill_ts_ns;
    uint32_t                  decodes_since_classify;
};

alignas(64) SessionState g_sessions[MAX_SESSIONS];

std::atomic<int>      g_sense_enabled{0};
std::atomic<int>      g_sense_initialized{0};
std::atomic<unsigned> g_session_count{0};
std::atomic<uint64_t> g_current_fingerprint{0};
std::atomic<int>      g_current_slot{-1};
std::atomic<uint64_t> g_last_ts_ns{0};

uint64_t fnv1a64(const uint8_t* data, size_t n, uint64_t seed = 0xcbf29ce484222325ULL) {
    uint64_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t shape_proxy(const CipherRingEntry* ev) {
    // Use params_hash as the canonical shape key (encodes M*N*K + tile choices).
    return ev->params_hash ? ev->params_hash : ev->func_ptr_hash;
}

// Decode the packed (M, K, N) embedded in params_hash by the geometry
// plumb-through. Tag 0xC in the top nibble is the marker. Returns 0/0/0
// when params_hash is unmarked (non-GEMM kernels or pre-plumb-through entries).
inline void unpack_mkn(uint64_t ph, uint32_t& M, uint32_t& K, uint32_t& N) {
    if ((ph >> 60) != 0xCULL) { M = K = N = 0; return; }
    M = (uint32_t)((ph >> 40) & 0xFFFFFu);
    K = (uint32_t)((ph >> 20) & 0xFFFFFu);
    N = (uint32_t)( ph        & 0xFFFFFu);
}

// Attempt to allocate a fresh session slot. Linear scan with CAS; stable under
// concurrent calls (only one Stage 1 thread today, but written defensively).
int allocate_slot(uint64_t initial_fp_seed) {
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t expected = 0;
        if (g_sessions[i].fingerprint.compare_exchange_strong(
                expected, initial_fp_seed, std::memory_order_acq_rel)) {
            // Initialise the slot.
            SessionState& s = g_sessions[i];
            s.session_type.store(CIPHER_SESSION_UNKNOWN, std::memory_order_relaxed);
            s.first_seq = s.last_seq = 0;
            s.first_ts_ns = s.last_ts_ns = 0;
            std::memset(s.first_8_shapes, 0, sizeof(s.first_8_shapes));
            s.n_shapes_seen = 0;
            s.prefill_count = s.decode_count = 0;
            s.inter_event_min_ns = UINT64_MAX;
            s.inter_event_max_ns = 0;
            s.inter_event_sum_ns = 0;
            s.n_inter_event = 0;
            s.prefill_arrival_n = 0;
            s.prefill_gap_sum_ns = 0;
            s.prefill_gap_sum_sq_ns = 0;
            s.last_prefill_ts_ns = 0;
            s.decodes_since_classify = 0;
            g_session_count.fetch_add(1, std::memory_order_relaxed);
            return (int)i;
        }
    }
    return -1; // table full — drop silently (Rule 6: never wrong, just noop)
}

void classify(SessionState& s) {
    // True M-based rules now that geometry is plumbed.
    //   AGENT_AUTONOMOUS:  long sustained decode chain (>500 decode steps)
    //   BATCH_BACKGROUND:  many prefills, little/no decode (offline batch jobs)
    //   HUMAN_INTERACTIVE: 1–2 prefills + a modest decode response
    bool is_agent = (s.decode_count > 500);

    bool is_batch = (s.prefill_count >= 3)
                 && (s.decode_count  <= 200);

    bool is_human = (s.prefill_count >= 1)
                 && (s.prefill_count <= 2)
                 && (s.decode_count  >= 5)
                 && (s.decode_count  <= 500);

    CipherSessionType t = CIPHER_SESSION_UNKNOWN;
    if      (is_agent) t = CIPHER_SESSION_AGENT_AUTONOMOUS;
    else if (is_batch) t = CIPHER_SESSION_BATCH_BACKGROUND;
    else if (is_human) t = CIPHER_SESSION_HUMAN_INTERACTIVE;

    s.session_type.store((uint8_t)t, std::memory_order_release);
}

} // namespace

extern "C" int cipher_sense_init(void) {
    int already = g_sense_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_sense_enabled.load(std::memory_order_relaxed);

    const char* env = std::getenv("CIPHER_SENSE");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_sense_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op13] SENSE enabled — slots=%u idle_ms=%llu\n",
                     MAX_SESSIONS,
                     (unsigned long long)(IDLE_THRESHOLD_NS / 1000000));
    }
    return on;
}

extern "C" void cipher_sense_observe(const CipherRingEntry* ev) {
    // Single relaxed load. When disabled, this is the only work done.
    if (!g_sense_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;

    // The hook-path ring entry has timestamp_ns == 0; capture our own here
    // (Stage 1, ~30 ns — does not affect Stage 0).
    uint64_t now = ev->timestamp_ns;
    if (now == 0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        now = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    uint64_t prev = g_last_ts_ns.exchange(now, std::memory_order_relaxed);

    int slot = g_current_slot.load(std::memory_order_relaxed);

    // Session boundary: idle gap > threshold OR no slot yet.
    bool start_new = (slot < 0) || (prev != 0 && now > prev
                                    && (now - prev) > IDLE_THRESHOLD_NS);

    if (start_new) {
        uint64_t seed = fnv1a64((const uint8_t*)&now, sizeof(now),
                                0xcbf29ce484222325ULL ^ ev->func_ptr_hash);
        slot = allocate_slot(seed);
        if (slot < 0) return; // table full
        g_current_slot.store(slot, std::memory_order_relaxed);
        g_current_fingerprint.store(seed, std::memory_order_relaxed);
        SessionState& s = g_sessions[slot];
        s.first_seq = ev->sequence;
        s.first_ts_ns = now;
    }

    SessionState& s = g_sessions[slot];
    s.last_seq    = ev->sequence;
    s.last_ts_ns  = now;

    // Inter-event timing (only within a session — skip the first event after boundary).
    if (!start_new && prev != 0 && now > prev) {
        uint64_t gap = now - prev;
        if (gap < s.inter_event_min_ns) s.inter_event_min_ns = gap;
        if (gap > s.inter_event_max_ns) s.inter_event_max_ns = gap;
        s.inter_event_sum_ns += (double)gap;
        s.n_inter_event++;
    }

    // GEMM signals — true M-based prefill/decode discrimination via the
    // geometry plumb-through (Op-14 build).
    if (ev->kernel_class == 0 /* OpClass::GEMM */) {
        uint32_t M, K, N;
        unpack_mkn(ev->params_hash, M, K, N);
        // Batch dim proxy: cuBLAS column-major puts PyTorch batch into N;
        // row-major path puts it into M. min(M,N) is robust to either.
        uint32_t batch = (M && N) ? (M < N ? M : N) : (M | N);

        bool is_decode  = (batch > 0 && batch <= 8);
        bool is_prefill = (batch >= 256);

        if (is_decode) {
            s.decode_count++;
            s.decodes_since_classify++;
        }
        if (is_prefill) {
            s.prefill_count++;
            s.decodes_since_classify++;  // also reclass on prefill bursts
            if (s.last_prefill_ts_ns != 0 && now > s.last_prefill_ts_ns) {
                double pgap = (double)(now - s.last_prefill_ts_ns);
                s.prefill_gap_sum_ns += pgap;
                s.prefill_gap_sum_sq_ns += pgap * pgap;
                s.prefill_arrival_n++;
            }
            s.last_prefill_ts_ns = now;
        }

        // Track distinct shape buckets seen (saturates at 8).
        if (s.n_shapes_seen < 8) {
            uint64_t sp = shape_proxy(ev);
            if (sp != 0) {
                bool dup = false;
                for (uint8_t i = 0; i < s.n_shapes_seen; ++i) {
                    if (s.first_8_shapes[i] == sp) { dup = true; break; }
                }
                if (!dup) {
                    s.first_8_shapes[s.n_shapes_seen++] = sp;
                    if (s.n_shapes_seen == 8) {
                        uint64_t fp = fnv1a64(
                            (const uint8_t*)s.first_8_shapes,
                            sizeof(s.first_8_shapes),
                            0xcbf29ce484222325ULL);
                        s.fingerprint.store(fp, std::memory_order_release);
                        g_current_fingerprint.store(fp, std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    // Re-classify periodically.
    if (s.decodes_since_classify >= RECLASS_DECODE_EVERY) {
        s.decodes_since_classify = 0;
        classify(s);
    }
}

extern "C" CipherSessionType cipher_sense_get_type(uint64_t fingerprint) {
    if (!g_sense_enabled.load(std::memory_order_relaxed))
        return CIPHER_SESSION_UNKNOWN;
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t fp = g_sessions[i].fingerprint.load(std::memory_order_acquire);
        if (fp == fingerprint && fp != 0) {
            return (CipherSessionType)g_sessions[i].session_type.load(
                std::memory_order_acquire);
        }
    }
    return CIPHER_SESSION_UNKNOWN;
}

extern "C" uint64_t cipher_sense_current_session(void) {
    return g_current_fingerprint.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_sense_session_count(void) {
    return g_session_count.load(std::memory_order_relaxed);
}

extern "C" void cipher_sense_report(void) {
    /* W7-9 Step 4 — coherent snapshot acquire (slow-path; emits the
     * COMMIT-published state alongside this report's existing aggregate). */
    struct cipher_rt_snapshot _snap;
    cipher_rt_snapshot_acquire(0u, &_snap);
    (void)_snap;

    if (!g_sense_enabled.load(std::memory_order_relaxed)) return;

    // Force a final classify pass on whatever session is current so the report
    // reflects the most up-to-date state even if RECLASS_DECODE_EVERY hasn't
    // fired yet on the trailing decodes.
    int cur = g_current_slot.load(std::memory_order_relaxed);
    if (cur >= 0) classify(g_sessions[cur]);

    FILE* fp = std::fopen("/tmp/cipher_sense_report.json", "w");
    if (!fp) return;

    static const char* type_str[] = {
        "UNKNOWN", "HUMAN_INTERACTIVE", "AGENT_AUTONOMOUS", "BATCH_BACKGROUND"
    };

    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"session_count\": %u,\n",
                 g_session_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"current_fingerprint\": \"%016llx\",\n",
                 (unsigned long long)g_current_fingerprint.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"sessions\": [\n");
    bool first = true;
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t fp64 = g_sessions[i].fingerprint.load(std::memory_order_acquire);
        if (fp64 == 0) continue;
        const SessionState& s = g_sessions[i];
        uint8_t t = s.session_type.load(std::memory_order_acquire);
        if (t > 3) t = 0;
        double mean_gap_ms = (s.n_inter_event > 0)
            ? (s.inter_event_sum_ns / s.n_inter_event / 1e6) : 0.0;
        if (!first) std::fprintf(fp, ",\n");
        first = false;
        std::fprintf(fp,
            "    {\"slot\": %u, \"fingerprint\": \"%016llx\", \"type\": \"%s\","
            " \"prefill_count\": %u, \"decode_count\": %u,"
            " \"mean_gap_ms\": %.3f, \"first_seq\": %llu, \"last_seq\": %llu}",
            i, (unsigned long long)fp64, type_str[t],
            s.prefill_count, s.decode_count, mean_gap_ms,
            (unsigned long long)s.first_seq, (unsigned long long)s.last_seq);
    }
    std::fprintf(fp, "\n  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op13] SENSE report -> /tmp/cipher_sense_report.json\n");
}
