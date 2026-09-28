// Op 28 TRACE — v1 bounded kernel-trace exporter.
//
// Fixed ring of compact records. Observer appends; report flushes JSONL.

#include "cipher_trace.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {

constexpr unsigned CAPACITY = 8192;

struct Rec {
    uint64_t sequence;
    uint64_t timestamp_ns;
    uint64_t func_ptr_hash;
    uint64_t params_hash;
    uint32_t kernel_class;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
};

alignas(64) Rec g_buf[CAPACITY];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<uint64_t> g_written{0};
std::atomic<uint64_t> g_dropped{0};

} // namespace

extern "C" int cipher_trace_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_TRACE");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op28] TRACE enabled — capacity=%u\n", CAPACITY);
    }
    return on;
}

extern "C" void cipher_trace_observe(const CipherRingEntry* ev) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;
    uint64_t idx = g_written.fetch_add(1, std::memory_order_relaxed);
    cipher_op_inc(OP_TRACE);
    if (idx >= CAPACITY) {
        g_dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Rec& r = g_buf[idx];
    r.sequence      = ev->sequence;
    r.timestamp_ns  = ev->timestamp_ns;
    r.func_ptr_hash = ev->func_ptr_hash;
    r.params_hash   = ev->params_hash;
    r.kernel_class  = ev->kernel_class;
    r.grid_x = ev->grid_x; r.grid_y = ev->grid_y; r.grid_z = ev->grid_z;
    r.block_x = ev->block_x; r.block_y = ev->block_y; r.block_z = ev->block_z;
}

extern "C" uint64_t cipher_trace_written(void) {
    uint64_t w = g_written.load(std::memory_order_relaxed);
    return w > CAPACITY ? CAPACITY : w;
}

extern "C" uint64_t cipher_trace_dropped(void) {
    return g_dropped.load(std::memory_order_relaxed);
}

extern "C" void cipher_trace_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    uint64_t w = g_written.load(std::memory_order_relaxed);
    uint64_t n = w > CAPACITY ? CAPACITY : w;

    FILE* fp = std::fopen("/tmp/cipher_trace.jsonl", "w");
    if (!fp) return;
    for (uint64_t i = 0; i < n; ++i) {
        const Rec& r = g_buf[i];
        std::fprintf(fp,
            "{\"seq\":%llu,\"ts_ns\":%llu,\"fp\":\"%016llx\","
            "\"ph\":\"%016llx\",\"k\":%u,"
            "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u]}\n",
            (unsigned long long)r.sequence,
            (unsigned long long)r.timestamp_ns,
            (unsigned long long)r.func_ptr_hash,
            (unsigned long long)r.params_hash,
            r.kernel_class,
            r.grid_x, r.grid_y, r.grid_z,
            r.block_x, r.block_y, r.block_z);
    }
    std::fclose(fp);

    // Also emit a summary JSON for easy test consumption.
    FILE* sp = std::fopen("/tmp/cipher_trace_report.json", "w");
    if (sp) {
        std::fprintf(sp, "{\n");
        std::fprintf(sp, "  \"written\": %llu,\n", (unsigned long long)n);
        std::fprintf(sp, "  \"dropped\": %llu,\n",
                     (unsigned long long)g_dropped.load(std::memory_order_relaxed));
        std::fprintf(sp, "  \"capacity\": %u,\n", CAPACITY);
        std::fprintf(sp, "  \"jsonl_path\": \"/tmp/cipher_trace.jsonl\"\n");
        std::fprintf(sp, "}\n");
        std::fclose(sp);
    }
    std::fprintf(stderr, "[CIPHER Op28] TRACE report -> /tmp/cipher_trace.jsonl "
                         "(written=%llu dropped=%llu)\n",
                 (unsigned long long)n,
                 (unsigned long long)g_dropped.load(std::memory_order_relaxed));
}
