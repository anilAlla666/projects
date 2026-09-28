// Op 29 COMPLY — v1 compliance pack.

#include "cipher_comply.h"
#include "cipher_receipt.h"
#include "cipher_carbon.h"
#include "cipher_guard.h"
#include "cipher_determinism.h"
#include "cipher_fairness.h"
#include "cipher_topology.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

namespace {
std::atomic<int> g_enabled{0};
std::atomic<int> g_initialized{0};
std::atomic<int> g_last_ok{0};

// OP 31 audit-log state
std::atomic<int>      g_audit_armed{0};
std::atomic<int>      g_audit_fd{-1};
std::atomic<uint64_t> g_audit_events{0};
std::atomic<uint64_t> g_audit_bytes{0};
std::atomic<uint64_t> g_audit_seq{0};
std::atomic<uint64_t> g_audit_open_fails{0};

uint64_t now_ns_audit() {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// ─── Per-tenant policy enforcement (M1.T7) ─────────────────────────
// Loaded from $CIPHER_POLICY_PATH (default /etc/cipher/policy.json).
// SIGHUP triggers reload. Schema:
//   { "tenants": [
//       { "id": "...", "max_power_watts": N,
//         "min_latency_ms_p99": N, "max_tok_per_s": N,
//         "banned_ops": ["OP_NAME", ...] }
//     ] }
// Hot-path queries: cipher_policy_op_banned(op_id), cipher_policy_max_power_watts().
struct TenantPolicy {
    char     id[64] = {0};
    int      max_power_watts = 0;       // 0 = unlimited
    int      min_latency_ms_p99 = 0;    // 0 = no constraint
    int      max_tok_per_s = 0;
    uint64_t banned_op_mask = 0;        // bit i set = OP id i banned
    bool     valid = false;
};
constexpr int MAX_POLICY_TENANTS = 64;
TenantPolicy g_policies[MAX_POLICY_TENANTS];
std::atomic<int> g_n_policies{0};
std::atomic<int> g_policy_self_idx{-1};   // index for our CIPHER_TENANT_ID
std::atomic<uint64_t> g_policy_reloads{0};

// Op-id name → CipherOpId mapping. Mirrors include/cipher_op_counters.h.
struct OpName { const char* name; int id; };
const OpName k_op_names[] = {
    {"CLASSIFY", 0}, {"PREDICT", 1}, {"RING_WRITE", 2}, {"FLOW_RECORD", 3},
    {"SPECULATE_CHECK", 4}, {"FLOW_MATCH", 5}, {"GUARD", 6}, {"DETERMINISM", 7},
    {"SUBSTITUTE_FP8", 8}, {"SUBSTITUTE_KOOPMAN", 9}, {"SUBSTITUTE_MARLIN", 10},
    {"FLOW_SUBSTITUTE", 11}, {"FUSE", 12}, {"NCCL_TUNER", 13}, {"PERSIST", 14},
    {"THERMOSTAT", 15}, {"FAIRNESS", 16}, {"ARBITRATE", 17}, {"PIPELINE", 18},
    {"CONTINUITY", 19}, {"RECEIPT", 20}, {"CARBON", 21}, {"TRACE", 22},
    {"COMPLY", 23}, {"AUDIT", 24}, {"LOOP", 25}, {"GRAPH_ENGINE", 26},
    {"REMEMBER", 27}, {"ADAPT", 28}, {"SPECULATE_WRITE", 29},
    {"VALIDATE", 30}, {"TOPOLOGY", 31}, {"WORKLOAD_OBSERVE", 32},
    {nullptr, -1},
};
int op_id_from_name(const char* n) {
    for (int i = 0; k_op_names[i].name; i++) {
        if (std::strcmp(k_op_names[i].name, n) == 0) return k_op_names[i].id;
    }
    return -1;
}

// Tiny ad-hoc JSON parser — sufficient for the well-known policy.json
// schema. NOT a general parser. Looks for "tenants": [ {...}, {...} ]
// and extracts the four scalar fields + banned_ops array.
void parse_policy_file(const char* path) {
    FILE* fp = std::fopen(path, "r");
    if (!fp) return;
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz > 1<<20) { std::fclose(fp); return; }
    char* buf = (char*)std::malloc(sz + 1);
    if (!buf) { std::fclose(fp); return; }
    sz = (long)std::fread(buf, 1, sz, fp);
    buf[sz] = 0;
    std::fclose(fp);

    int n = 0;
    char* p = std::strstr(buf, "\"tenants\"");
    if (!p) { std::free(buf); return; }
    while ((p = std::strchr(p, '{')) && n < MAX_POLICY_TENANTS) {
        char* end = std::strchr(p, '}');
        if (!end) break;
        TenantPolicy& t = g_policies[n];
        std::memset(&t, 0, sizeof(t));
        // id
        char* idp = std::strstr(p, "\"id\"");
        if (idp && idp < end) {
            char* sq = std::strchr(idp, '"');
            sq = std::strchr(sq + 1, '"');     // closing of "id"
            sq = std::strchr(sq + 1, '"');     // open of value
            char* eq = sq ? std::strchr(sq + 1, '"') : nullptr;
            if (sq && eq && (eq - sq - 1) < (int)sizeof(t.id)) {
                std::memcpy(t.id, sq + 1, eq - sq - 1);
                t.id[eq - sq - 1] = 0;
            }
        }
        auto read_int = [&](const char* key) -> int {
            char* kp = std::strstr(p, key);
            if (!kp || kp >= end) return 0;
            char* col = std::strchr(kp, ':');
            return col ? std::atoi(col + 1) : 0;
        };
        t.max_power_watts    = read_int("\"max_power_watts\"");
        t.min_latency_ms_p99 = read_int("\"min_latency_ms_p99\"");
        t.max_tok_per_s      = read_int("\"max_tok_per_s\"");
        // banned_ops: parse the inner array of strings
        char* ban = std::strstr(p, "\"banned_ops\"");
        t.banned_op_mask = 0;
        if (ban && ban < end) {
            char* lb = std::strchr(ban, '[');
            char* rb = lb ? std::strchr(lb, ']') : nullptr;
            if (lb && rb) {
                char* q = lb;
                while ((q = std::strchr(q, '"')) && q < rb) {
                    char* qend = std::strchr(q + 1, '"');
                    if (!qend || qend > rb) break;
                    char op_name[64] = {0};
                    size_t nlen = qend - q - 1;
                    if (nlen < sizeof(op_name)) std::memcpy(op_name, q + 1, nlen);
                    int oid = op_id_from_name(op_name);
                    if (oid >= 0 && oid < 64)
                        t.banned_op_mask |= (1ULL << oid);
                    q = qend + 1;
                }
            }
        }
        t.valid = (t.id[0] != 0);
        if (t.valid) n++;
        p = end + 1;
    }
    g_n_policies.store(n, std::memory_order_release);

    // Resolve our own tenant idx via CIPHER_TENANT_ID env var.
    const char* my_id = std::getenv("CIPHER_TENANT_ID");
    int self_idx = -1;
    if (my_id) {
        for (int i = 0; i < n; i++) {
            if (std::strcmp(g_policies[i].id, my_id) == 0) { self_idx = i; break; }
        }
    }
    if (self_idx < 0) {
        for (int i = 0; i < n; i++) {
            if (std::strcmp(g_policies[i].id, "default") == 0) { self_idx = i; break; }
        }
    }
    g_policy_self_idx.store(self_idx, std::memory_order_release);
    g_policy_reloads.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr,
        "[CIPHER POLICY] loaded %d tenants from %s; self_idx=%d (tenant_id=%s)\n",
        n, path, self_idx, my_id ? my_id : "(unset)");
    std::free(buf);
}

void policy_sighup_handler(int) {
    const char* p = std::getenv("CIPHER_POLICY_PATH");
    if (!p) p = "/etc/cipher/policy.json";
    parse_policy_file(p);
}
} // namespace

extern "C" int cipher_comply_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_COMPLY");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op29] COMPLY enabled — "
                             "bundles RECEIPT+CARBON+GUARD+DETERMINISM+FAIRNESS+TOPOLOGY\n");
    }
    // M1.T7 — load /etc/cipher/policy.json (or $CIPHER_POLICY_PATH override).
    // Always attempted, regardless of CIPHER_COMPLY: policy.json is the
    // multi-tenant primary control surface and must be live even when
    // the rest of COMPLY bundling is off. Missing file is not an error.
    const char* policy_path = std::getenv("CIPHER_POLICY_PATH");
    if (!policy_path) policy_path = "/etc/cipher/policy.json";
    parse_policy_file(policy_path);
    // SIGHUP triggers reload from the same path. Use sigaction (signal()
    // semantics differ across libc; sigaction is portable + restart-safe).
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = policy_sighup_handler;
    sa.sa_flags   = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGHUP, &sa, nullptr);

    // OP 31 — open audit JSONL log if path is set. Independent of CIPHER_COMPLY,
    // so customers can stream privileged events even without bundling.
    const char* path = std::getenv("CIPHER_COMPLY_AUDIT_PATH");
    if (path && *path) {
        int fd = ::open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (fd >= 0) {
            g_audit_fd.store(fd, std::memory_order_release);
            g_audit_armed.store(1, std::memory_order_release);
            std::fprintf(stderr, "[CIPHER Op31] COMPLY audit JSONL -> %s (fd=%d)\n",
                         path, fd);
        } else {
            g_audit_open_fails.fetch_add(1, std::memory_order_relaxed);
            std::fprintf(stderr, "[CIPHER Op31] COMPLY audit OPEN FAIL %s — disabled\n", path);
        }
    }
    return on;
}

extern "C" int cipher_comply_audit(const char* kind, const char* json_kv) {
    if (!g_audit_armed.load(std::memory_order_relaxed)) return 0;
    int fd = g_audit_fd.load(std::memory_order_acquire);
    if (fd < 0) return 0;
    if (!kind) kind = "?";
    if (!json_kv) json_kv = "{}";
    char buf[1024];
    int n = std::snprintf(buf, sizeof(buf),
        "{\"seq\":%llu,\"ts_ns\":%llu,\"pid\":%d,\"kind\":\"%s\",\"data\":%s}\n",
        (unsigned long long)g_audit_seq.fetch_add(1, std::memory_order_relaxed),
        (unsigned long long)now_ns_audit(),
        (int)getpid(),
        kind, json_kv);
    if (n <= 0) return 0;
    if (n > (int)sizeof(buf)) n = (int)sizeof(buf);
    ssize_t w = ::write(fd, buf, (size_t)n);
    if (w > 0) {
        g_audit_events.fetch_add(1, std::memory_order_relaxed);
        g_audit_bytes.fetch_add((uint64_t)w, std::memory_order_relaxed);
        cipher_op_inc(OP_COMPLY);
        cipher_op_inc(OP_AUDIT);
        return 1;
    }
    return 0;
}

extern "C" int cipher_comply_audit_stats(CipherComplyAuditStats* out) {
    if (!out) return 0;
    out->enabled        = g_enabled.load(std::memory_order_relaxed);
    out->armed          = g_audit_armed.load(std::memory_order_relaxed);
    out->events_written = g_audit_events.load(std::memory_order_relaxed);
    out->bytes_written  = g_audit_bytes.load(std::memory_order_relaxed);
    out->open_failures  = g_audit_open_fails.load(std::memory_order_relaxed);
    return 1;
}

// ─── M1.T7 public API ──────────────────────────────────────────────
// Hot-path queries — called from actuator dispatch sites.

// Returns 1 if the running tenant is permitted to fire `op_id`, else 0.
// Unknown / unconfigured tenant defaults to permitted (open by default
// — operators add a "default" entry to lock that down).
extern "C" int cipher_policy_op_allowed(int op_id) {
    int idx = g_policy_self_idx.load(std::memory_order_acquire);
    if (idx < 0 || op_id < 0 || op_id >= 64) return 1;
    uint64_t mask = g_policies[idx].banned_op_mask;
    return (mask >> op_id) & 1ULL ? 0 : 1;
}

// Returns the configured power cap for the running tenant; 0 means
// "no cap" (THERMOSTAT free-runs).
extern "C" int cipher_policy_max_power_watts(void) {
    int idx = g_policy_self_idx.load(std::memory_order_acquire);
    if (idx < 0) return 0;
    return g_policies[idx].max_power_watts;
}

extern "C" int cipher_policy_min_latency_ms_p99(void) {
    int idx = g_policy_self_idx.load(std::memory_order_acquire);
    if (idx < 0) return 0;
    return g_policies[idx].min_latency_ms_p99;
}

extern "C" int cipher_policy_max_tok_per_s(void) {
    int idx = g_policy_self_idx.load(std::memory_order_acquire);
    if (idx < 0) return 0;
    return g_policies[idx].max_tok_per_s;
}

// Returns the resolved tenant id ("" if unset / unmatched).
extern "C" const char* cipher_policy_tenant_id(void) {
    int idx = g_policy_self_idx.load(std::memory_order_acquire);
    if (idx < 0) return "";
    return g_policies[idx].id;
}

// Reload counter (Prometheus exporter reads this for sighup-tracking).
extern "C" uint64_t cipher_policy_reload_count(void) {
    return g_policy_reloads.load(std::memory_order_relaxed);
}

// Programmatic reload (tests + admin tools — does what SIGHUP does).
extern "C" int cipher_policy_reload(void) {
    const char* p = std::getenv("CIPHER_POLICY_PATH");
    if (!p) p = "/etc/cipher/policy.json";
    parse_policy_file(p);
    return g_n_policies.load(std::memory_order_relaxed);
}

static void cipher_comply_op31_atexit() {
    int fd = g_audit_fd.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0) {
        // Final summary line — useful for log-rotation tooling.
        char buf[256];
        int n = std::snprintf(buf, sizeof(buf),
            "{\"seq\":%llu,\"ts_ns\":%llu,\"pid\":%d,\"kind\":\"audit_close\","
            "\"data\":{\"events\":%llu,\"bytes\":%llu}}\n",
            (unsigned long long)g_audit_seq.fetch_add(1, std::memory_order_relaxed),
            (unsigned long long)now_ns_audit(), (int)getpid(),
            (unsigned long long)g_audit_events.load(std::memory_order_relaxed),
            (unsigned long long)g_audit_bytes.load(std::memory_order_relaxed));
        if (n > 0) ::write(fd, buf, (size_t)n);
        ::close(fd);
    }
}

__attribute__((constructor(115)))
static void cipher_comply_op31_autoinit() {
    cipher_comply_init();
    if (g_audit_armed.load(std::memory_order_relaxed))
        std::atexit(cipher_comply_op31_atexit);
}

extern "C" void cipher_comply_observe(const CipherRingEntry* ev) {
    (void)ev;  // aggregation-only; no hot-path work.
}

extern "C" int cipher_comply_ok(void) {
    return g_last_ok.load(std::memory_order_relaxed);
}

extern "C" void cipher_comply_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    unsigned receipt_sessions = cipher_receipt_session_count();
    unsigned carbon_sessions  = cipher_carbon_session_count();
    unsigned guard_sessions   = cipher_guard_session_count();
    unsigned guard_leaks      = cipher_guard_leak_count();
    uint64_t determ_hash      = cipher_determinism_hash();
    uint64_t determ_count     = cipher_determinism_count();
    unsigned fair_tenants     = cipher_fairness_tenant_count();
    unsigned fair_overruns    = cipher_fairness_overrun_count();
    unsigned topo_devices     = cipher_topology_device_count();

    int ok = (guard_leaks == 0)
          && (receipt_sessions >= 1)
          && (determ_count > 0)
          && (fair_overruns == 0);
    g_last_ok.store(ok, std::memory_order_release);

    FILE* fp = std::fopen("/tmp/cipher_comply_report.json", "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"compliance_ok\": %s,\n", ok ? "true" : "false");
    std::fprintf(fp, "  \"receipt\":     {\"session_count\": %u},\n", receipt_sessions);
    std::fprintf(fp, "  \"carbon\":      {\"session_count\": %u},\n", carbon_sessions);
    std::fprintf(fp, "  \"guard\":       {\"session_count\": %u, \"leak_count\": %u},\n",
                 guard_sessions, guard_leaks);
    std::fprintf(fp, "  \"determinism\": {\"dispatch_hash\": \"%016llx\", \"dispatch_count\": %llu},\n",
                 (unsigned long long)determ_hash, (unsigned long long)determ_count);
    std::fprintf(fp, "  \"fairness\":    {\"tenant_count\": %u, \"overrun_count\": %u},\n",
                 fair_tenants, fair_overruns);
    std::fprintf(fp, "  \"topology\":    {\"device_count\": %u}\n", topo_devices);
    std::fprintf(fp, "}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op29] COMPLY report -> "
                         "/tmp/cipher_comply_report.json (ok=%d)\n", ok);
}
