// Op 19 CONTINUITY — v1 observer.
//
// Per-session tracking of attention regions that would be checkpointed by a
// v2 Tier-A worker. No KV capture, no pinned memory, no CUDA calls.
// Conforms to OP_CONTRACT.md I1–I6.

#include "may13/cipher_continuity.h"
#include "may13/cipher_sense.h"

#include "cipher_rt_commit.h"
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
    /* W7-9 Step 4 — coherent snapshot acquire (slow-path; emits the
     * COMMIT-published state alongside this report's existing aggregate). */
    struct cipher_rt_snapshot _snap;
    cipher_rt_snapshot_acquire(0u, &_snap);
    (void)_snap;

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
