// Op 27 PIPELINE — v1 observer.
//
// Per-session bounded shape set. At report time, emits pairwise Jaccard
// edges (sim >= 0.5) between sessions, tagged upstream/downstream by
// first_ts_ns. Conforms to OP_CONTRACT.md I1–I6.

#include "cipher_pipeline.h"
#include "cipher_sense.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

namespace {

constexpr unsigned MAX_SESSIONS        = 1024;
constexpr unsigned SHAPES_PER_SESSION  = 32;
constexpr uint64_t IDLE_RESET_NS       = 200ULL * 1000 * 1000;   // 200 ms
constexpr unsigned REPORT_MAX_SESSIONS = 256;                    // pairwise cap
constexpr float    JACCARD_THRESHOLD   = 0.5f;

struct PipelineSession {
    std::atomic<uint64_t> fingerprint;   // 0 = empty slot
    uint64_t first_ts_ns, last_ts_ns;
    uint32_t event_count;
    uint8_t  n_shapes;                   // up to SHAPES_PER_SESSION
    uint64_t shapes[SHAPES_PER_SESSION]; // distinct params_hash values
};

alignas(64) PipelineSession g_slots[MAX_SESSIONS];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned> g_session_count{0};
std::atomic<unsigned> g_edge_count{0};
std::atomic<int>      g_current_slot{-1};
std::atomic<uint64_t> g_last_ts_ns{0};

uint64_t fnv1a64(const uint8_t* data, size_t n, uint64_t seed = 0xcbf29ce484222325ULL) {
    uint64_t h = seed;
    for (size_t i = 0; i < n; ++i) { h ^= data[i]; h *= 0x100000001b3ULL; }
    return h;
}

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

uint64_t shape_key(const CipherRingEntry* ev) {
    return ev->params_hash ? ev->params_hash : ev->func_ptr_hash;
}

int allocate_slot(uint64_t seed) {
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t expected = 0;
        if (g_slots[i].fingerprint.compare_exchange_strong(
                expected, seed, std::memory_order_acq_rel)) {
            PipelineSession& s = g_slots[i];
            s.first_ts_ns = s.last_ts_ns = 0;
            s.event_count = 0;
            s.n_shapes = 0;
            std::memset(s.shapes, 0, sizeof(s.shapes));
            g_session_count.fetch_add(1, std::memory_order_relaxed);
            cipher_op_inc(OP_PIPELINE);
            return (int)i;
        }
    }
    return -1;
}

// Insert shape into bounded set if new. Drop-on-full (I4).
void insert_shape(PipelineSession& s, uint64_t key) {
    if (key == 0) return;
    for (unsigned i = 0; i < s.n_shapes; ++i) {
        if (s.shapes[i] == key) return;  // already present
    }
    if (s.n_shapes < SHAPES_PER_SESSION) {
        s.shapes[s.n_shapes++] = key;
    }
    // else: full, silently drop
}

float jaccard(const PipelineSession& a, const PipelineSession& b) {
    if (a.n_shapes == 0 || b.n_shapes == 0) return 0.0f;
    unsigned inter = 0;
    for (unsigned i = 0; i < a.n_shapes; ++i) {
        for (unsigned j = 0; j < b.n_shapes; ++j) {
            if (a.shapes[i] == b.shapes[j]) { inter++; break; }
        }
    }
    unsigned uni = (unsigned)a.n_shapes + (unsigned)b.n_shapes - inter;
    return uni ? (float)inter / (float)uni : 0.0f;
}

} // namespace

// ── OP 27 — multi-GPU pipeline stage detection ────────────────────────────
namespace {
constexpr int STAGE_MAX_GPUS = 16;

std::atomic<int>      g_stage_enabled{0};
std::atomic<int>      g_stage_initialized{0};
std::atomic<uint64_t> g_stage_window_ns{1000000ULL};   // 1 ms default
std::atomic<uint64_t> g_stage_cross{0};
std::atomic<uint64_t> g_stage_intra{0};

struct GpuLast {
    std::atomic<uint64_t> last_ts_ns;
    std::atomic<int>      seen;
};
GpuLast g_stage_gpu[STAGE_MAX_GPUS];
std::atomic<int> g_stage_edges_table[STAGE_MAX_GPUS][STAGE_MAX_GPUS];
} // namespace

extern "C" void cipher_pipeline_observe_gpu(int gpu_id) {
    if (!g_stage_enabled.load(std::memory_order_relaxed)) return;
    if (gpu_id < 0 || gpu_id >= STAGE_MAX_GPUS) return;
    uint64_t t = now_ns();
    g_stage_gpu[gpu_id].seen.store(1, std::memory_order_release);
    uint64_t window = g_stage_window_ns.load(std::memory_order_relaxed);
    // Look for a cross-GPU predecessor inside the window.
    for (int j = 0; j < STAGE_MAX_GPUS; ++j) {
        if (j == gpu_id) continue;
        if (!g_stage_gpu[j].seen.load(std::memory_order_acquire)) continue;
        uint64_t prev = g_stage_gpu[j].last_ts_ns.load(std::memory_order_acquire);
        if (prev == 0 || t <= prev) continue;
        if ((t - prev) <= window) {
            g_stage_edges_table[j][gpu_id].fetch_add(1, std::memory_order_relaxed);
            g_stage_cross.fetch_add(1, std::memory_order_relaxed);
        }
    }
    uint64_t prev_self = g_stage_gpu[gpu_id].last_ts_ns.exchange(t, std::memory_order_acq_rel);
    if (prev_self != 0) g_stage_intra.fetch_add(1, std::memory_order_relaxed);
}

extern "C" int cipher_pipeline_stage_stats(CipherPipelineStageStats* out) {
    if (!out) return 0;
    out->enabled    = g_stage_enabled.load(std::memory_order_relaxed);
    out->window_ns  = g_stage_window_ns.load(std::memory_order_relaxed);
    out->cross_gpu_events = g_stage_cross.load(std::memory_order_relaxed);
    out->intra_gpu_events = g_stage_intra.load(std::memory_order_relaxed);
    int gpus_seen = 0;
    for (int i = 0; i < STAGE_MAX_GPUS; ++i)
        if (g_stage_gpu[i].seen.load(std::memory_order_relaxed)) gpus_seen++;
    out->gpus_seen = gpus_seen;
    int stages = 0;
    for (int i = 0; i < STAGE_MAX_GPUS; ++i)
        for (int j = 0; j < STAGE_MAX_GPUS; ++j) {
            int c = g_stage_edges_table[i][j].load(std::memory_order_relaxed);
            out->edges_table[i][j] = c;
            if (c > 0 && i != j) stages++;
        }
    out->stages_inferred = stages;
    return 1;
}

__attribute__((constructor(116)))
static void cipher_pipeline_op27_autoinit() {
    if (g_stage_initialized.exchange(1, std::memory_order_acq_rel)) return;
    const char* env = std::getenv("CIPHER_PIPELINE_STAGES");
    int on = env && env[0] && (env[0]=='1' || env[0]=='t' || env[0]=='T'
                            || env[0]=='o' || env[0]=='O');
    g_stage_enabled.store(on, std::memory_order_release);
    if (const char* w = std::getenv("CIPHER_PIPELINE_STAGE_WINDOW_NS")) {
        uint64_t v = std::strtoull(w, nullptr, 10);
        if (v > 0) g_stage_window_ns.store(v, std::memory_order_release);
    }
    if (on) {
        std::fprintf(stderr,
            "[CIPHER OP27] PIPELINE stages enabled — window=%llu ns\n",
            (unsigned long long)g_stage_window_ns.load(std::memory_order_relaxed));
    }
}

extern "C" int cipher_pipeline_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_PIPELINE");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op27] PIPELINE enabled — slots=%u "
                             "shapes/session=%u jaccard_threshold=%.2f\n",
                     MAX_SESSIONS, SHAPES_PER_SESSION, JACCARD_THRESHOLD);
    }
    return on;
}

extern "C" void cipher_pipeline_observe(const CipherRingEntry* ev) {
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
        g_slots[slot].first_ts_ns = t;
    }

    PipelineSession& s = g_slots[slot];
    s.last_ts_ns = t;
    s.event_count++;
    insert_shape(s, shape_key(ev));
}

extern "C" unsigned cipher_pipeline_session_count(void) {
    return g_session_count.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_pipeline_edge_count(void) {
    return g_edge_count.load(std::memory_order_relaxed);
}

extern "C" void cipher_pipeline_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    // Collect allocated indices (bounded).
    unsigned idx[REPORT_MAX_SESSIONS];
    unsigned n = 0;
    for (unsigned i = 0; i < MAX_SESSIONS && n < REPORT_MAX_SESSIONS; ++i) {
        if (g_slots[i].fingerprint.load(std::memory_order_acquire) != 0) {
            idx[n++] = i;
        }
    }

    FILE* fp = std::fopen("/tmp/cipher_pipeline_report.json", "w");
    if (!fp) return;

    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"session_count\": %u,\n",
                 g_session_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"sessions\": [\n");
    bool first_s = true;
    for (unsigned k = 0; k < n; ++k) {
        const PipelineSession& s = g_slots[idx[k]];
        uint64_t fp64 = g_slots[idx[k]].fingerprint.load(std::memory_order_acquire);
        CipherSessionType st = cipher_sense_get_type(fp64);
        const char* st_name = (st == CIPHER_SESSION_AGENT_AUTONOMOUS) ? "AGENT_AUTONOMOUS"
                           : (st == CIPHER_SESSION_HUMAN_INTERACTIVE) ? "HUMAN_INTERACTIVE"
                           : (st == CIPHER_SESSION_BATCH_BACKGROUND)  ? "BATCH_BACKGROUND"
                           : "UNKNOWN";
        if (!first_s) std::fprintf(fp, ",\n");
        first_s = false;
        std::fprintf(fp,
            "    {\"slot\": %u, \"fingerprint\": \"%016llx\","
            " \"sense_type\": \"%s\","
            " \"first_ts_ns\": %llu, \"last_ts_ns\": %llu,"
            " \"events\": %u, \"n_shapes\": %u}",
            idx[k], (unsigned long long)fp64, st_name,
            (unsigned long long)s.first_ts_ns,
            (unsigned long long)s.last_ts_ns,
            s.event_count, (unsigned)s.n_shapes);
    }
    std::fprintf(fp, "\n  ],\n");

    // Pairwise Jaccard edges (i < j). Upstream = earlier first_ts_ns.
    std::fprintf(fp, "  \"edges\": [\n");
    bool first_e = true;
    unsigned edges = 0;
    for (unsigned a = 0; a < n; ++a) {
        for (unsigned b = a + 1; b < n; ++b) {
            const PipelineSession& sa = g_slots[idx[a]];
            const PipelineSession& sb = g_slots[idx[b]];
            float sim = jaccard(sa, sb);
            if (sim < JACCARD_THRESHOLD) continue;
            unsigned up = (sa.first_ts_ns <= sb.first_ts_ns) ? idx[a] : idx[b];
            unsigned dn = (up == idx[a]) ? idx[b] : idx[a];
            uint64_t fp_up = g_slots[up].fingerprint.load(std::memory_order_acquire);
            uint64_t fp_dn = g_slots[dn].fingerprint.load(std::memory_order_acquire);
            CipherSessionType st_up = cipher_sense_get_type(fp_up);
            CipherSessionType st_dn = cipher_sense_get_type(fp_dn);
            const char* u_name = (st_up == CIPHER_SESSION_AGENT_AUTONOMOUS) ? "AGENT_AUTONOMOUS"
                              : (st_up == CIPHER_SESSION_HUMAN_INTERACTIVE) ? "HUMAN_INTERACTIVE"
                              : (st_up == CIPHER_SESSION_BATCH_BACKGROUND)  ? "BATCH_BACKGROUND"
                              : "UNKNOWN";
            const char* d_name = (st_dn == CIPHER_SESSION_AGENT_AUTONOMOUS) ? "AGENT_AUTONOMOUS"
                              : (st_dn == CIPHER_SESSION_HUMAN_INTERACTIVE) ? "HUMAN_INTERACTIVE"
                              : (st_dn == CIPHER_SESSION_BATCH_BACKGROUND)  ? "BATCH_BACKGROUND"
                              : "UNKNOWN";
            if (!first_e) std::fprintf(fp, ",\n");
            first_e = false;
            std::fprintf(fp,
                "    {\"from_slot\": %u, \"to_slot\": %u,"
                " \"from_fingerprint\": \"%016llx\", \"to_fingerprint\": \"%016llx\","
                " \"from_sense_type\": \"%s\", \"to_sense_type\": \"%s\","
                " \"jaccard\": %.4f}",
                up, dn,
                (unsigned long long)fp_up, (unsigned long long)fp_dn,
                u_name, d_name, sim);
            edges++;
        }
    }
    std::fprintf(fp, "\n  ],\n");
    std::fprintf(fp, "  \"edge_count\": %u\n", edges);
    std::fprintf(fp, "}\n");
    std::fclose(fp);

    g_edge_count.store(edges, std::memory_order_relaxed);
    std::fprintf(stderr, "[CIPHER Op27] PIPELINE report -> "
                         "/tmp/cipher_pipeline_report.json (sessions=%u edges=%u)\n",
                 n, edges);
}
