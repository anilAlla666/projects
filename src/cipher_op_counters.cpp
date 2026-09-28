#include "cipher_op_counters.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

static std::atomic<uint64_t> g_counters[OP_COUNT];

static const char* k_names[OP_COUNT] = {
    "CLASSIFY", "PREDICT", "RING_WRITE", "FLOW_RECORD",
    "SPECULATE_CHECK", "FLOW_MATCH", "GUARD", "DETERMINISM",
    "SUBSTITUTE_FP8", "SUBSTITUTE_KOOPMAN", "SUBSTITUTE_MARLIN",
    "FLOW_SUBSTITUTE", "FUSE", "NCCL_TUNER", "PERSIST",
    "THERMOSTAT", "FAIRNESS", "ARBITRATE", "PIPELINE",
    "CONTINUITY", "RECEIPT", "CARBON", "TRACE", "COMPLY",
    "AUDIT", "LOOP", "GRAPH_ENGINE", "REMEMBER", "ADAPT",
    "SPECULATE_WRITE", "VALIDATE", "TOPOLOGY", "WORKLOAD_OBSERVE",
};

extern "C" void cipher_op_inc(int op) {
    if (op < 0 || op >= OP_COUNT) return;
    g_counters[op].fetch_add(1, std::memory_order_relaxed);
}

extern "C" uint64_t cipher_op_counter_get(int op) {
    if (op < 0 || op >= OP_COUNT) return 0;
    return g_counters[op].load(std::memory_order_relaxed);
}

extern "C" const char* cipher_op_name(int op) {
    if (op < 0 || op >= OP_COUNT) return "UNKNOWN";
    return k_names[op];
}

extern "C" void cipher_op_counters_dump_json(const char* path) {
    if (!path || !*path) return;
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\n  \"pid\": %d,\n  \"ops\": {", (int)getpid());
    for (int i = 0; i < OP_COUNT; ++i) {
        fprintf(f, "%s\n    \"%s\": %llu",
                i ? "," : "",
                k_names[i],
                (unsigned long long)g_counters[i].load(std::memory_order_relaxed));
    }
    fprintf(f, "\n  }\n}\n");
    fclose(f);
}

// atexit auto-dump when CIPHER_COUNTERS_DUMP_DIR is set
static void cipher_op_counters_atexit() {
    const char* dir = getenv("CIPHER_COUNTERS_DUMP_DIR");
    if (!dir || !*dir) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/cipher_counters_%d.json", dir, (int)getpid());
    cipher_op_counters_dump_json(path);
}

__attribute__((constructor))
static void cipher_op_counters_init() {
    atexit(cipher_op_counters_atexit);
}
