// Op 19 CONTINUITY — v1 observer.
//
// Per-session tracking of attention regions that would be checkpointed by a
// v2 Tier-A worker. No KV capture, no pinned memory, no CUDA calls.
// Conforms to OP_CONTRACT.md I1–I6.

#include "cipher_continuity.h"
#include "cipher_sense.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

namespace {

constexpr unsigned MAX_SESSIONS     = 1024;
constexpr unsigned MAX_REGIONS      = 32;    // per-session region slots
constexpr uint32_t SNAPSHOT_EVERY   = 500;   // attn events between manifests
constexpr uint64_t IDLE_RESET_NS    = 200ULL * 1000 * 1000;

struct Region {
    uint64_t key;              // 0 = empty
    uint64_t first_seq;
    uint64_t last_seq;
    uint32_t count;            // total events observed in this region
    uint32_t count_since_snap; // cleared each manifest
};

struct ContinuitySession {
    std::atomic<uint64_t> fingerprint;   // 0 = empty slot
    uint64_t first_ts_ns, last_ts_ns;
    uint32_t attn_events_total;
    uint32_t event_count;
    uint32_t manifest_count;
    uint64_t last_manifest_seq;
    Region   regions[MAX_REGIONS];
};

alignas(64) ContinuitySession g_slots[MAX_SESSIONS];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned> g_session_count{0};
std::atomic<unsigned> g_manifest_count{0};
std::atomic<int>      g_current_slot{-1};
std::atomic<uint64_t> g_current_fp{0};
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

uint64_t region_key(const CipherRingEntry* ev) {
    uint64_t gxy = (uint64_t)ev->grid_x * (uint64_t)ev->grid_y;
    uint64_t parts[2] = { ev->func_ptr_hash, gxy };
    return fnv1a64((const uint8_t*)parts, sizeof(parts));
}

int allocate_slot(uint64_t seed) {
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t expected = 0;
        if (g_slots[i].fingerprint.compare_exchange_strong(
                expected, seed, std::memory_order_acq_rel)) {
            ContinuitySession& s = g_slots[i];
            s.first_ts_ns = s.last_ts_ns = 0;
            s.attn_events_total = 0;
            s.event_count = 0;
            s.manifest_count = 0;
            s.last_manifest_seq = 0;
            std::memset(s.regions, 0, sizeof(s.regions));
            g_session_count.fetch_add(1, std::memory_order_relaxed);
            cipher_op_inc(OP_CONTINUITY);
            return (int)i;
        }
    }
    return -1;
}

// Linear scan; returns region index or -1 if full and no match.
int upsert_region(ContinuitySession& s, uint64_t key, uint64_t seq) {
    int free_idx = -1;
    for (unsigned i = 0; i < MAX_REGIONS; ++i) {
        if (s.regions[i].key == key) {
            s.regions[i].last_seq = seq;
            s.regions[i].count++;
            s.regions[i].count_since_snap++;
            return (int)i;
        }
        if (s.regions[i].key == 0 && free_idx < 0) free_idx = (int)i;
    }
    if (free_idx >= 0) {
        s.regions[free_idx].key = key;
        s.regions[free_idx].first_seq = seq;
        s.regions[free_idx].last_seq = seq;
        s.regions[free_idx].count = 1;
        s.regions[free_idx].count_since_snap = 1;
        return free_idx;
    }
    return -1; // full — silently drop (I4: never allocate on hot path)
}

} // namespace

extern "C" int cipher_continuity_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_CONTINUITY");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op19] CONTINUITY enabled — slots=%u "
                             "regions/session=%u snapshot_every=%u (AGENT only)\n",
                     MAX_SESSIONS, MAX_REGIONS, SNAPSHOT_EVERY);
    }
    return on;
}

extern "C" void cipher_continuity_observe(const CipherRingEntry* ev) {
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
        g_slots[slot].first_ts_ns = t;
    }

    ContinuitySession& s = g_slots[slot];
    s.last_ts_ns = t;
    s.event_count++;

    // v1 tracks *all* dispatch events (KV state advances on every decode,
    // not only attention kernels). Region key still discriminates per-kernel.
    s.attn_events_total++;
    uint64_t key = region_key(ev);
    upsert_region(s, key, ev->sequence);

    // Manifest gate: every SNAPSHOT_EVERY events, AND session classified
    // as AGENT_AUTONOMOUS. Read-only call into SENSE — contract-permitted.
    if ((s.attn_events_total % SNAPSHOT_EVERY) == 0) {
        uint64_t fp = g_slots[slot].fingerprint.load(std::memory_order_acquire);
        CipherSessionType t_type = cipher_sense_get_type(fp);
        if (t_type == CIPHER_SESSION_AGENT_AUTONOMOUS) {
            s.manifest_count++;
            s.last_manifest_seq = ev->sequence;
            for (unsigned i = 0; i < MAX_REGIONS; ++i) {
                s.regions[i].count_since_snap = 0;
            }
            g_manifest_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

extern "C" unsigned cipher_continuity_session_count(void) {
    return g_session_count.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_continuity_manifest_count(void) {
    return g_manifest_count.load(std::memory_order_relaxed);
}

extern "C" void cipher_continuity_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    FILE* fp = std::fopen("/tmp/cipher_continuity_report.json", "w");
    if (!fp) return;

    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"session_count\": %u,\n",
                 g_session_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"manifest_count_total\": %u,\n",
                 g_manifest_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"sessions\": [\n");
    bool first_sess = true;
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t fp64 = g_slots[i].fingerprint.load(std::memory_order_acquire);
        if (fp64 == 0) continue;
        const ContinuitySession& s = g_slots[i];
        CipherSessionType st = cipher_sense_get_type(fp64);
        const char* st_name = (st == CIPHER_SESSION_AGENT_AUTONOMOUS) ? "AGENT_AUTONOMOUS"
                           : (st == CIPHER_SESSION_HUMAN_INTERACTIVE) ? "HUMAN_INTERACTIVE"
                           : (st == CIPHER_SESSION_BATCH_BACKGROUND)  ? "BATCH_BACKGROUND"
                           : "UNKNOWN";
        if (!first_sess) std::fprintf(fp, ",\n");
        first_sess = false;
        std::fprintf(fp,
            "    {\"slot\": %u, \"fingerprint\": \"%016llx\","
            " \"sense_type\": \"%s\","
            " \"events\": %u, \"attn_events\": %u,"
            " \"manifest_count\": %u, \"last_manifest_seq\": %llu,"
            " \"regions\": [",
            i, (unsigned long long)fp64, st_name,
            s.event_count, s.attn_events_total,
            s.manifest_count, (unsigned long long)s.last_manifest_seq);
        bool first_r = true;
        for (unsigned j = 0; j < MAX_REGIONS; ++j) {
            if (s.regions[j].key == 0) continue;
            if (!first_r) std::fprintf(fp, ", ");
            first_r = false;
            std::fprintf(fp,
                "{\"key\": \"%016llx\", \"first_seq\": %llu,"
                " \"last_seq\": %llu, \"count\": %u}",
                (unsigned long long)s.regions[j].key,
                (unsigned long long)s.regions[j].first_seq,
                (unsigned long long)s.regions[j].last_seq,
                s.regions[j].count);
        }
        std::fprintf(fp, "]}");
    }
    std::fprintf(fp, "\n  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op19] CONTINUITY report -> /tmp/cipher_continuity_report.json\n");
}

// ── OP 28 — checkpoint / restore ─────────────────────────────────────────
namespace {
constexpr uint32_t CONTINUITY_MAGIC   = 0xC0117001;
constexpr uint32_t CONTINUITY_VERSION = 1;

#pragma pack(push, 1)
struct ContHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t session_count;
    uint32_t reserved;
    uint64_t saved_at_ns;
};
struct ContSessionRow {
    uint64_t fingerprint;
    uint64_t first_ts_ns, last_ts_ns;
    uint32_t attn_events_total;
    uint32_t event_count;
    uint32_t manifest_count;
    uint32_t last_manifest_seq_lo;
    uint32_t last_manifest_seq_hi;
    uint32_t reserved;
    Region   regions[MAX_REGIONS];
};
#pragma pack(pop)

std::atomic<unsigned> g_op28_saves_done{0};
std::atomic<unsigned> g_op28_restores_done{0};
std::atomic<uint64_t> g_op28_bytes_written{0};
std::atomic<uint64_t> g_op28_bytes_read{0};
std::atomic<unsigned> g_op28_sessions_saved{0};
std::atomic<unsigned> g_op28_sessions_restored{0};
} // namespace

extern "C" int cipher_continuity_save(const char* path) {
    if (!path || !*path) return 0;
    FILE* fp = std::fopen(path, "wb");
    if (!fp) return 0;
    // Pre-count active sessions.
    unsigned n = 0;
    for (unsigned i = 0; i < MAX_SESSIONS; ++i)
        if (g_slots[i].fingerprint.load(std::memory_order_acquire) != 0) n++;
    ContHeader h = { CONTINUITY_MAGIC, CONTINUITY_VERSION, n, 0, now_ns() };
    if (std::fwrite(&h, sizeof(h), 1, fp) != 1) { std::fclose(fp); return 0; }
    uint64_t bytes = sizeof(h);
    for (unsigned i = 0; i < MAX_SESSIONS; ++i) {
        uint64_t fp64 = g_slots[i].fingerprint.load(std::memory_order_acquire);
        if (fp64 == 0) continue;
        const ContinuitySession& s = g_slots[i];
        ContSessionRow row{};
        row.fingerprint        = fp64;
        row.first_ts_ns        = s.first_ts_ns;
        row.last_ts_ns         = s.last_ts_ns;
        row.attn_events_total  = s.attn_events_total;
        row.event_count        = s.event_count;
        row.manifest_count     = s.manifest_count;
        row.last_manifest_seq_lo = (uint32_t)(s.last_manifest_seq & 0xFFFFFFFFu);
        row.last_manifest_seq_hi = (uint32_t)(s.last_manifest_seq >> 32);
        std::memcpy(row.regions, s.regions, sizeof(row.regions));
        if (std::fwrite(&row, sizeof(row), 1, fp) != 1) { std::fclose(fp); return 0; }
        bytes += sizeof(row);
    }
    std::fclose(fp);
    g_op28_saves_done.fetch_add(1, std::memory_order_relaxed);
    g_op28_bytes_written.fetch_add(bytes, std::memory_order_relaxed);
    g_op28_sessions_saved.fetch_add(n, std::memory_order_relaxed);
    std::fprintf(stderr, "[CIPHER OP28] CONTINUITY saved -> %s (sessions=%u bytes=%llu)\n",
                 path, n, (unsigned long long)bytes);
    return 1;
}

extern "C" int cipher_continuity_restore(const char* path) {
    if (!path || !*path) return 0;
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return 0;
    ContHeader h{};
    if (std::fread(&h, sizeof(h), 1, fp) != 1) { std::fclose(fp); return 0; }
    if (h.magic != CONTINUITY_MAGIC || h.version != CONTINUITY_VERSION) {
        std::fclose(fp); return 0;
    }
    uint64_t bytes = sizeof(h);
    unsigned restored = 0;
    for (unsigned i = 0; i < h.session_count && i < MAX_SESSIONS; ++i) {
        ContSessionRow row{};
        if (std::fread(&row, sizeof(row), 1, fp) != 1) break;
        bytes += sizeof(row);
        // Insert at the slot matching its fingerprint hash; collisions just
        // overwrite.
        unsigned slot = (unsigned)((row.fingerprint * 0x9E3779B97F4A7C15ULL) >> 54)
                        % MAX_SESSIONS;
        ContinuitySession& s = g_slots[slot];
        s.fingerprint.store(row.fingerprint, std::memory_order_release);
        s.first_ts_ns        = row.first_ts_ns;
        s.last_ts_ns         = row.last_ts_ns;
        s.attn_events_total  = row.attn_events_total;
        s.event_count        = row.event_count;
        s.manifest_count     = row.manifest_count;
        s.last_manifest_seq  = ((uint64_t)row.last_manifest_seq_hi << 32)
                              | (uint64_t)row.last_manifest_seq_lo;
        std::memcpy(s.regions, row.regions, sizeof(s.regions));
        g_session_count.fetch_add(1, std::memory_order_relaxed);
        restored++;
    }
    std::fclose(fp);
    g_op28_restores_done.fetch_add(1, std::memory_order_relaxed);
    g_op28_bytes_read.fetch_add(bytes, std::memory_order_relaxed);
    g_op28_sessions_restored.fetch_add(restored, std::memory_order_relaxed);
    std::fprintf(stderr, "[CIPHER OP28] CONTINUITY restored <- %s (sessions=%u bytes=%llu)\n",
                 path, restored, (unsigned long long)bytes);
    return 1;
}

extern "C" int cipher_continuity_stats(CipherContinuityStats* out) {
    if (!out) return 0;
    out->enabled            = g_enabled.load(std::memory_order_relaxed);
    out->saves_done         = g_op28_saves_done.load(std::memory_order_relaxed);
    out->restores_done      = g_op28_restores_done.load(std::memory_order_relaxed);
    out->bytes_written      = g_op28_bytes_written.load(std::memory_order_relaxed);
    out->bytes_read         = g_op28_bytes_read.load(std::memory_order_relaxed);
    out->sessions_saved     = g_op28_sessions_saved.load(std::memory_order_relaxed);
    out->sessions_restored  = g_op28_sessions_restored.load(std::memory_order_relaxed);
    return 1;
}

static void cipher_continuity_op28_atexit() {
    const char* p = std::getenv("CIPHER_CONTINUITY_SAVE_PATH");
    if (p && *p) cipher_continuity_save(p);
}

__attribute__((constructor(117)))
static void cipher_continuity_op28_autoinit() {
    const char* p = std::getenv("CIPHER_CONTINUITY_RESTORE_PATH");
    if (p && *p) cipher_continuity_restore(p);
    if (std::getenv("CIPHER_CONTINUITY_SAVE_PATH"))
        std::atexit(cipher_continuity_op28_atexit);
}
