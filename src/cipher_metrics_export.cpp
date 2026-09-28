// M1.T8 — Prometheus metrics endpoint.
// =================================================================
// Minimal HTTP server (POSIX sockets, no external deps) responding
// to GET /metrics with Prometheus text-format counters and gauges.
// Background thread, listening on $CIPHER_METRICS_PORT (default 9091).
//
// Why not a Prom C++ client lib: drops a 200 KB transitive .so dep.
// Cipher already binds at LD_PRELOAD; an extra dep risks symbol
// collisions. The text format is trivial — one printf per metric.
//
// Wired surfaces (auto-discovered via dlsym from RTLD_DEFAULT — works
// whether you build symbols into rt or hook):
//   - All 33 op counters → cipher_op_total{op="NAME"}
//   - Distributor stats  → cipher_distributor_*
//   - Workload classifier→ cipher_workload_class, cipher_workload_observe_total
//   - Policy            → cipher_policy_tenant, cipher_policy_reload_count
//   - Receipt signing   → cipher_receipt_lines_total, cipher_receipt_bytes_total
//   - Comply audit      → cipher_audit_events_total, cipher_audit_bytes_total

#include "cipher_op_counters.h"

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
std::atomic<int> g_running{0};
std::atomic<int> g_port{0};
std::atomic<uint64_t> g_scrapes{0};
std::thread g_thread;

// Distributor stats struct from include/cipher_distributor.h
struct DistStats {
    int n_tenants;
    int n_gpus;
    uint64_t observations;
    uint64_t migrate_signals;
};

// Receipt signing stats from include/cipher_receipt_signing.h
struct ReceiptStats {
    int      enabled;
    uint64_t lines_written;
    uint64_t bytes_written;
    uint64_t key_load_failures;
    uint64_t write_failures;
};

// Comply audit stats from include/cipher_comply.h
struct AuditStats {
    int enabled;
    int armed;
    uint64_t events_written;
    uint64_t bytes_written;
    uint64_t open_failures;
};

template <typename Fn>
Fn try_resolve(const char* name) {
    return reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, name));
}

// Format-checked appender — printf-attribute lets the compiler verify
// every format string at the call site (catches a stray `%s` mismatch
// or accidental runtime-data-as-format misuse).
__attribute__((format(printf, 4, 5)))
static void buf_appendf(char* out, size_t cap, size_t* off,
                         const char* fmt, ...) {
    if (*off >= cap) return;
    va_list ap;
    va_start(ap, fmt);
    int n = std::vsnprintf(out + *off, cap - *off, fmt, ap);
    va_end(ap);
    if (n > 0) *off += (size_t)n;
}

// Build the metrics body. Returns bytes written (snprintf-style; caller
// truncates if larger than buffer).
size_t build_metrics(char* out, size_t cap) {
    size_t off = 0;
    #define append(...) buf_appendf(out, cap, &off, __VA_ARGS__)

    // ─── op counters ────────────────────────────────────────────
    append("# HELP cipher_op_total CIPHER op invocations (per LD_PRELOAD process).\n");
    append("# TYPE cipher_op_total counter\n");
    auto get = try_resolve<uint64_t(*)(int)>("cipher_op_counter_get");
    auto name = try_resolve<const char*(*)(int)>("cipher_op_name");
    if (get && name) {
        for (int i = 0; i < OP_COUNT; i++) {
            const char* n = name(i);
            uint64_t v = get(i);
            if (n && *n) append("cipher_op_total{op=\"%s\"} %llu\n",
                                 n, (unsigned long long)v);
        }
    }

    // ─── distributor ────────────────────────────────────────────
    auto dist_stats = try_resolve<int(*)(DistStats*)>("cipher_distributor_stats");
    if (dist_stats) {
        DistStats s{};
        if (dist_stats(&s)) {
            append("# TYPE cipher_distributor_tenants gauge\n");
            append("cipher_distributor_tenants %d\n", s.n_tenants);
            append("# TYPE cipher_distributor_gpus gauge\n");
            append("cipher_distributor_gpus %d\n", s.n_gpus);
            append("# TYPE cipher_distributor_observations_total counter\n");
            append("cipher_distributor_observations_total %llu\n",
                   (unsigned long long)s.observations);
            append("# TYPE cipher_distributor_migrate_signals_total counter\n");
            append("cipher_distributor_migrate_signals_total %llu\n",
                   (unsigned long long)s.migrate_signals);
        }
    }

    // ─── workload classifier ────────────────────────────────────
    auto wl_class  = try_resolve<int(*)(void)>("cipher_workload_classify");
    auto wl_name   = try_resolve<const char*(*)(int)>("cipher_workload_name");
    if (wl_class) {
        int c = wl_class();
        append("# HELP cipher_workload_class Active workload classification id.\n");
        append("# TYPE cipher_workload_class gauge\n");
        append("cipher_workload_class %d\n", c);
        if (wl_name) {
            const char* n = wl_name(c);
            if (n) append("cipher_workload_class_label{name=\"%s\"} 1\n", n);
        }
    }

    // ─── policy ─────────────────────────────────────────────────
    auto pol_tenant = try_resolve<const char*(*)(void)>("cipher_policy_tenant_id");
    auto pol_reload = try_resolve<uint64_t(*)(void)>("cipher_policy_reload_count");
    auto pol_max_w  = try_resolve<int(*)(void)>("cipher_policy_max_power_watts");
    auto pol_max_t  = try_resolve<int(*)(void)>("cipher_policy_max_tok_per_s");
    if (pol_tenant) {
        const char* t = pol_tenant();
        append("# TYPE cipher_policy_tenant info\n");
        append("cipher_policy_tenant{id=\"%s\"} 1\n", (t && *t) ? t : "(unset)");
    }
    if (pol_reload) {
        append("# TYPE cipher_policy_reload_count counter\n");
        append("cipher_policy_reload_count %llu\n",
               (unsigned long long)pol_reload());
    }
    if (pol_max_w) {
        append("# TYPE cipher_policy_max_power_watts gauge\n");
        append("cipher_policy_max_power_watts %d\n", pol_max_w());
    }
    if (pol_max_t) {
        append("# TYPE cipher_policy_max_tok_per_s gauge\n");
        append("cipher_policy_max_tok_per_s %d\n", pol_max_t());
    }

    // ─── receipt signing (Tier B DSO; may not be loaded) ────────
    auto rcpt_stats = try_resolve<int(*)(ReceiptStats*)>("cipher_receipt_signing_stats");
    if (rcpt_stats) {
        ReceiptStats r{};
        if (rcpt_stats(&r)) {
            append("# TYPE cipher_receipt_enabled gauge\n");
            append("cipher_receipt_enabled %d\n", r.enabled);
            append("# TYPE cipher_receipt_lines_total counter\n");
            append("cipher_receipt_lines_total %llu\n",
                   (unsigned long long)r.lines_written);
            append("# TYPE cipher_receipt_bytes_total counter\n");
            append("cipher_receipt_bytes_total %llu\n",
                   (unsigned long long)r.bytes_written);
            append("# TYPE cipher_receipt_write_failures_total counter\n");
            append("cipher_receipt_write_failures_total %llu\n",
                   (unsigned long long)r.write_failures);
        }
    }

    // ─── comply audit ───────────────────────────────────────────
    auto audit_stats = try_resolve<int(*)(AuditStats*)>("cipher_comply_audit_stats");
    if (audit_stats) {
        AuditStats a{};
        if (audit_stats(&a)) {
            append("# TYPE cipher_audit_armed gauge\n");
            append("cipher_audit_armed %d\n", a.armed);
            append("# TYPE cipher_audit_events_total counter\n");
            append("cipher_audit_events_total %llu\n",
                   (unsigned long long)a.events_written);
            append("# TYPE cipher_audit_bytes_total counter\n");
            append("cipher_audit_bytes_total %llu\n",
                   (unsigned long long)a.bytes_written);
        }
    }

    // ─── exporter self-stats ────────────────────────────────────
    append("# TYPE cipher_metrics_scrapes_total counter\n");
    append("cipher_metrics_scrapes_total %llu\n",
           (unsigned long long)g_scrapes.fetch_add(1, std::memory_order_relaxed) + 1);

    return off;
    #undef append
}

void serve_loop(int srv_fd) {
    while (g_running.load(std::memory_order_acquire)) {
        struct sockaddr_in cli; socklen_t cl = sizeof(cli);
        int fd = accept(srv_fd, (struct sockaddr*)&cli, &cl);
        if (fd < 0) {
            if (errno == EINTR) continue;
            // Periodic accept timeout (we set SO_RCVTIMEO on srv_fd) lets
            // the loop poll g_running for graceful shutdown.
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            std::fprintf(stderr, "[CIPHER metrics] accept err: %s\n", std::strerror(errno));
            continue;
        }
        // Read HTTP request line; we accept any GET.
        char req[1024];
        ssize_t n = recv(fd, req, sizeof(req)-1, 0);
        (void)n;
        const char* hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain; version=0.0.4\r\n"
            "Connection: close\r\n";
        // Build body with a generous fixed buffer; OP_COUNT=33, ~65 bytes/line
        // including HELP/TYPE → < 8 KB. 64 KB has 8× margin.
        static thread_local char body[65536];
        size_t blen = build_metrics(body, sizeof(body));
        char hbuf[128];
        int hn = std::snprintf(hbuf, sizeof(hbuf),
            "Content-Length: %zu\r\n\r\n", blen);
        send(fd, hdr, std::strlen(hdr), MSG_NOSIGNAL);
        send(fd, hbuf, (size_t)hn, MSG_NOSIGNAL);
        send(fd, body, blen, MSG_NOSIGNAL);
        close(fd);
    }
    close(srv_fd);
}

int start_server(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    // 500 ms accept timeout so the run loop can poll g_running.
    struct timeval tv{0, 500000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { close(fd); return -1; }
    if (listen(fd, 16) < 0) { close(fd); return -1; }
    g_running.store(1, std::memory_order_release);
    g_thread = std::thread([fd]() { serve_loop(fd); });
    return fd;
}
} // namespace

extern "C" int cipher_metrics_export_init(void) {
    if (g_running.load()) return g_port.load();
    const char* env = std::getenv("CIPHER_METRICS_PORT");
    int port = env ? std::atoi(env) : 9091;
    if (port <= 0) {
        std::fprintf(stderr, "[CIPHER metrics] disabled (CIPHER_METRICS_PORT=%s)\n",
                     env ? env : "unset");
        return 0;
    }
    if (start_server(port) < 0) {
        std::fprintf(stderr,
            "[CIPHER metrics] could not bind :%d — exporter disabled\n", port);
        return 0;
    }
    g_port.store(port, std::memory_order_release);
    std::fprintf(stderr, "[CIPHER metrics] exporter listening on :%d/metrics\n", port);
    return port;
}

extern "C" int cipher_metrics_export_port(void) {
    return g_port.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_metrics_export_scrapes(void) {
    return g_scrapes.load(std::memory_order_relaxed);
}

// Auto-init at LD_PRELOAD time. Late priority (130) — runs after silicon,
// op-counters, comply, distributor are all up so the exporter has live data.
__attribute__((constructor(130)))
static void cipher_metrics_export_autoinit(void) {
    cipher_metrics_export_init();
}
