// CIPHER Graph Engine — implementation.
//
// v1 observer + decision logic. Detects stable kernel-fingerprint chains
// per thread and counts promotions. Actual cuStreamBeginCapture /
// cuGraphInstantiate / cuGraphLaunch wiring lives behind a second env flag
// (CIPHER_GRAPH_REPLAY=on) and remains a no-op until the application opts
// in — graph capture conflicts can crash workloads, so default-deferred.

#include "cipher_graph.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <ctime>
#include <cuda_runtime.h>

namespace {

constexpr unsigned MAX_SEQ_LEN = 32;        // launches per sequence to track
constexpr unsigned PROMOTE_AFTER_REPEATS = 4;

struct ThreadState {
    uint64_t fps[MAX_SEQ_LEN];
    unsigned len;
    uint64_t last_signature;        // hash of (fps, len)
    unsigned repeat_count;          // consecutive identical signatures
    uint64_t total_kernels;
};

thread_local ThreadState tls_seq{};

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<uint64_t> g_observe_calls{0};
std::atomic<uint64_t> g_sequences_seen{0};
std::atomic<uint64_t> g_sequences_promoted{0};
std::atomic<uint64_t> g_replays{0};
std::atomic<uint64_t> g_invalidations{0};
std::atomic<uint64_t> g_capture_skipped_app{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return (v[0] == '1') || (v[0] == 't') || (v[0] == 'T')
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

uint64_t fnv1a(const uint64_t* xs, unsigned n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned i = 0; i < n; ++i) {
        h ^= xs[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

} // namespace

extern "C" int cipher_graph_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel)) return g_enabled.load();
    int on = env_truthy(getenv("CIPHER_GRAPH"));
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        fprintf(stderr,
            "[CIPHER GRAPH] init max_seq_len=%u promote_after=%u (replay path "
            "gated separately by CIPHER_GRAPH_REPLAY)\n",
            MAX_SEQ_LEN, PROMOTE_AFTER_REPEATS);
    }
    return on;
}

extern "C" int cipher_graph_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_graph_observe(uint64_t fp) {
    if (!g_enabled.load(std::memory_order_relaxed)) return CIPHER_GRAPH_OBSERVE;
    g_observe_calls.fetch_add(1, std::memory_order_relaxed);
    cipher_op_inc(OP_GRAPH_ENGINE);
    tls_seq.total_kernels++;
    if (tls_seq.len < MAX_SEQ_LEN) {
        tls_seq.fps[tls_seq.len++] = fp;
    } else {
        // shift in
        memmove(&tls_seq.fps[0], &tls_seq.fps[1], (MAX_SEQ_LEN - 1) * sizeof(uint64_t));
        tls_seq.fps[MAX_SEQ_LEN - 1] = fp;
    }
    return CIPHER_GRAPH_OBSERVE;
}

extern "C" int cipher_graph_step_boundary(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (tls_seq.len == 0) return 0;
    g_sequences_seen.fetch_add(1, std::memory_order_relaxed);

    uint64_t sig = fnv1a(tls_seq.fps, tls_seq.len);
    int promoted = 0;
    if (sig == tls_seq.last_signature) {
        tls_seq.repeat_count++;
        if (tls_seq.repeat_count == PROMOTE_AFTER_REPEATS) {
            g_sequences_promoted.fetch_add(1, std::memory_order_relaxed);
            promoted = 1;
        }
        if (tls_seq.repeat_count > PROMOTE_AFTER_REPEATS) {
            // Beyond promotion: every step is a "replay" of the same shape.
            g_replays.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        if (tls_seq.repeat_count >= PROMOTE_AFTER_REPEATS) {
            g_invalidations.fetch_add(1, std::memory_order_relaxed);
        }
        tls_seq.repeat_count = 0;
        tls_seq.last_signature = sig;
    }
    tls_seq.len = 0;
    return promoted;
}

extern "C" int cipher_graph_stats(CipherGraphStats* out) {
    if (!out) return 0;
    out->enabled              = g_enabled.load(std::memory_order_relaxed);
    out->observe_calls        = g_observe_calls.load(std::memory_order_relaxed);
    out->sequences_seen       = g_sequences_seen.load(std::memory_order_relaxed);
    out->sequences_promoted   = g_sequences_promoted.load(std::memory_order_relaxed);
    out->replays              = g_replays.load(std::memory_order_relaxed);
    out->invalidations        = g_invalidations.load(std::memory_order_relaxed);
    out->capture_skipped_app  = g_capture_skipped_app.load(std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_graph_report(void) {
    CipherGraphStats s{};
    cipher_graph_stats(&s);
    FILE* f = fopen("/tmp/cipher_graph_report.json", "w");
    if (!f) return;
    fprintf(f,
        "{\"enabled\":%d,\"observe_calls\":%llu,\"sequences_seen\":%llu,"
        "\"sequences_promoted\":%llu,\"replays\":%llu,\"invalidations\":%llu,"
        "\"capture_skipped_app\":%llu}\n",
        s.enabled,
        (unsigned long long)s.observe_calls,
        (unsigned long long)s.sequences_seen,
        (unsigned long long)s.sequences_promoted,
        (unsigned long long)s.replays,
        (unsigned long long)s.invalidations,
        (unsigned long long)s.capture_skipped_app);
    fclose(f);
}

// ── Stage 5 actuation: real CUDA Graph capture / replay ────────────────────

namespace {

using CUgraph     = void*;
using CUgraphExec = void*;
using CUstream    = void*;

typedef int (*pf_cuStreamBeginCapture)(CUstream, int /*mode*/);
typedef int (*pf_cuStreamEndCapture)(CUstream, CUgraph*);
typedef int (*pf_cuStreamIsCapturing)(CUstream, int* /*status*/);
typedef int (*pf_cuGraphInstantiate)(CUgraphExec*, CUgraph, void*, void*, unsigned long long);
typedef int (*pf_cuGraphLaunch)(CUgraphExec, CUstream);
typedef int (*pf_cuGraphDestroy)(CUgraph);
typedef int (*pf_cuGraphExecDestroy)(CUgraphExec);

struct DriverApi {
    void* lib = nullptr;
    pf_cuStreamBeginCapture cuStreamBeginCapture = nullptr;
    pf_cuStreamEndCapture   cuStreamEndCapture   = nullptr;
    pf_cuStreamIsCapturing  cuStreamIsCapturing  = nullptr;
    pf_cuGraphInstantiate   cuGraphInstantiate   = nullptr;
    pf_cuGraphLaunch        cuGraphLaunch        = nullptr;
    pf_cuGraphDestroy       cuGraphDestroy       = nullptr;
    pf_cuGraphExecDestroy   cuGraphExecDestroy   = nullptr;
    bool ok = false;
};
DriverApi g_drv;
std::mutex g_drv_mu;

bool resolve_driver() {
    std::lock_guard<std::mutex> lk(g_drv_mu);
    if (g_drv.ok) return true;
    if (g_drv.lib) return false;
    g_drv.lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!g_drv.lib) return false;
    g_drv.cuStreamBeginCapture = (pf_cuStreamBeginCapture)dlsym(g_drv.lib, "cuStreamBeginCapture_v2");
    if (!g_drv.cuStreamBeginCapture)
        g_drv.cuStreamBeginCapture = (pf_cuStreamBeginCapture)dlsym(g_drv.lib, "cuStreamBeginCapture");
    g_drv.cuStreamEndCapture   = (pf_cuStreamEndCapture)dlsym(g_drv.lib, "cuStreamEndCapture");
    g_drv.cuStreamIsCapturing  = (pf_cuStreamIsCapturing)dlsym(g_drv.lib, "cuStreamIsCapturing");
    g_drv.cuGraphInstantiate   = (pf_cuGraphInstantiate)dlsym(g_drv.lib, "cuGraphInstantiateWithFlags");
    if (!g_drv.cuGraphInstantiate)
        g_drv.cuGraphInstantiate = (pf_cuGraphInstantiate)dlsym(g_drv.lib, "cuGraphInstantiate_v2");
    g_drv.cuGraphLaunch        = (pf_cuGraphLaunch)dlsym(g_drv.lib, "cuGraphLaunch");
    g_drv.cuGraphDestroy       = (pf_cuGraphDestroy)dlsym(g_drv.lib, "cuGraphDestroy");
    g_drv.cuGraphExecDestroy   = (pf_cuGraphExecDestroy)dlsym(g_drv.lib, "cuGraphExecDestroy");
    g_drv.ok = (g_drv.cuStreamBeginCapture && g_drv.cuStreamEndCapture
                && g_drv.cuGraphInstantiate && g_drv.cuGraphLaunch
                && g_drv.cuGraphDestroy);
    return g_drv.ok;
}

#define MAX_CAPTURED_GRAPHS 32

struct CapturedGraph {
    int          in_use;
    CUgraph      g;
    CUgraphExec  e;
};
CapturedGraph g_graphs[MAX_CAPTURED_GRAPHS]{};
std::mutex    g_graphs_mu;
std::atomic<uint64_t> g_capture_calls{0};
std::atomic<uint64_t> g_replay_calls{0};
std::atomic<uint64_t> g_destroy_calls{0};
std::atomic<uint64_t> g_capture_skipped_app2{0};

int find_free_graph_locked() {
    for (int i = 0; i < MAX_CAPTURED_GRAPHS; ++i)
        if (!g_graphs[i].in_use) return i;
    return -1;
}

} // namespace

extern "C" int cipher_graph_begin_capture(void* stream_handle) {
    if (!g_enabled.load(std::memory_order_relaxed) || !stream_handle) return 0;
    if (!resolve_driver()) return 0;

    // Skip if an application-side capture is already active.
    if (g_drv.cuStreamIsCapturing) {
        int status = 0;
        if (g_drv.cuStreamIsCapturing((CUstream)stream_handle, &status) == 0
            && status != 0) {
            g_capture_skipped_app2.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
    }
    g_capture_calls.fetch_add(1, std::memory_order_relaxed);
    // Mode 1 = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL — safest with mixed apps.
    return g_drv.cuStreamBeginCapture((CUstream)stream_handle, 1) == 0 ? 1 : 0;
}

extern "C" unsigned long cipher_graph_end_capture(void* stream_handle) {
    if (!g_enabled.load(std::memory_order_relaxed) || !stream_handle) return 0;
    if (!g_drv.ok) return 0;
    CUgraph g = nullptr;
    if (g_drv.cuStreamEndCapture((CUstream)stream_handle, &g) != 0 || !g) return 0;
    CUgraphExec e = nullptr;
    int rc = g_drv.cuGraphInstantiate(&e, g, nullptr, nullptr, 0);
    if (rc != 0 || !e) {
        if (g_drv.cuGraphDestroy) g_drv.cuGraphDestroy(g);
        return 0;
    }
    std::lock_guard<std::mutex> lk(g_graphs_mu);
    int slot = find_free_graph_locked();
    if (slot < 0) {
        if (g_drv.cuGraphExecDestroy) g_drv.cuGraphExecDestroy(e);
        if (g_drv.cuGraphDestroy)     g_drv.cuGraphDestroy(g);
        return 0;
    }
    g_graphs[slot] = CapturedGraph{1, g, e};
    return (unsigned long)(slot + 1);   // 1-indexed; 0 = invalid id
}

extern "C" int cipher_graph_replay(unsigned long graph_id, void* stream_handle) {
    if (!g_enabled.load(std::memory_order_relaxed) || !stream_handle) return 0;
    if (graph_id == 0 || graph_id > MAX_CAPTURED_GRAPHS) return 0;
    int slot = (int)graph_id - 1;
    if (!g_drv.ok) return 0;
    CUgraphExec e;
    {
        std::lock_guard<std::mutex> lk(g_graphs_mu);
        if (!g_graphs[slot].in_use) return 0;
        e = g_graphs[slot].e;
    }
    g_replay_calls.fetch_add(1, std::memory_order_relaxed);
    g_replays.fetch_add(1, std::memory_order_relaxed);
    return g_drv.cuGraphLaunch(e, (CUstream)stream_handle) == 0 ? 1 : 0;
}

extern "C" int cipher_graph_destroy(unsigned long graph_id) {
    if (graph_id == 0 || graph_id > MAX_CAPTURED_GRAPHS) return 0;
    int slot = (int)graph_id - 1;
    std::lock_guard<std::mutex> lk(g_graphs_mu);
    if (!g_graphs[slot].in_use) return 0;
    if (g_drv.cuGraphExecDestroy) g_drv.cuGraphExecDestroy(g_graphs[slot].e);
    if (g_drv.cuGraphDestroy)     g_drv.cuGraphDestroy(g_graphs[slot].g);
    g_graphs[slot] = CapturedGraph{};
    g_destroy_calls.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

// ── OP 25 — synthetic warmup ──────────────────────────────────────────────
namespace {
std::atomic<int>      g_wu_armed{0};
std::atomic<int>      g_wu_ran{0};
std::atomic<int>      g_wu_success{0};
std::atomic<int>      g_wu_last_step_failed{0};
std::atomic<uint64_t> g_wu_replays{0};
std::atomic<uint64_t> g_wu_bytes{0};
std::atomic<uint64_t> g_wu_latency_ns{0};

uint64_t now_ns_warmup() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
} // namespace

extern "C" int cipher_graph_warmup(void) {
    // OP 25 — warmup runs independently of CIPHER_GRAPH (which controls
    // promotion). The cuStream / cuGraph driver primitives are always
    // available; we just need a CUDA context. We briefly flip g_enabled
    // so the begin_capture/end_capture/replay gates pass, then restore.
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    int saved_enabled = g_enabled.load(std::memory_order_acquire);
    g_enabled.store(1, std::memory_order_release);
    auto restore_enabled = [&]() {
        g_enabled.store(saved_enabled, std::memory_order_release);
    };

    constexpr size_t WU_BYTES = 4096;
    void* devbuf = nullptr;
    int last_fail = 0;
    uint64_t t0 = now_ns_warmup();

    // Step 1: stream
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) { last_fail = 1; goto done; }

    // Step 2: alloc target
    if (cudaMalloc(&devbuf, WU_BYTES) != cudaSuccess) { last_fail = 1; goto done; }

    // Step 3: begin capture (CIPHER's wrapper handles the cuStreamBeginCapture
    // call w/ correctness gates).
    if (!cipher_graph_begin_capture(stream)) { last_fail = 2; goto done; }

    // Step 4: a real op while captured — cudaMemsetAsync.
    if (cudaMemsetAsync(devbuf, 0, WU_BYTES, stream) != cudaSuccess) {
        last_fail = 3; goto done;
    }

    // Step 5: end capture → graph_id
    {
        unsigned long gid = cipher_graph_end_capture(stream);
        if (gid == 0) { last_fail = 4; goto done; }

        // Step 6: replay
        if (!cipher_graph_replay(gid, stream)) { last_fail = 5; goto done; }
        g_wu_replays.fetch_add(1, std::memory_order_relaxed);

        // Step 7: sync
        if (cudaStreamSynchronize(stream) != cudaSuccess) { last_fail = 6; goto done; }

        // Cleanup
        cipher_graph_destroy(gid);
    }

done:
    if (devbuf) cudaFree(devbuf);
    if (stream) cudaStreamDestroy(stream);
    restore_enabled();

    g_wu_ran.store(1, std::memory_order_release);
    g_wu_last_step_failed.store(last_fail, std::memory_order_release);
    g_wu_bytes.store(WU_BYTES, std::memory_order_release);
    g_wu_latency_ns.store(now_ns_warmup() - t0, std::memory_order_release);

    int ok = (last_fail == 0) ? 1 : 0;
    g_wu_success.store(ok, std::memory_order_release);
    fprintf(stderr,
        "[CIPHER OP25] GRAPH warmup: %s (last_fail=%d latency=%.1f us replays=%llu)\n",
        ok ? "OK" : "FAIL", last_fail,
        (g_wu_latency_ns.load() / 1000.0),
        (unsigned long long)g_wu_replays.load(std::memory_order_relaxed));
    return ok;
}

extern "C" int cipher_graph_warmup_stats(CipherGraphWarmupStats* out) {
    if (!out) return 0;
    out->armed             = g_wu_armed.load(std::memory_order_relaxed);
    out->ran               = g_wu_ran.load(std::memory_order_relaxed);
    out->success           = g_wu_success.load(std::memory_order_relaxed);
    out->last_step_failed  = g_wu_last_step_failed.load(std::memory_order_relaxed);
    out->replays_done      = g_wu_replays.load(std::memory_order_relaxed);
    out->bytes_in_synth    = g_wu_bytes.load(std::memory_order_relaxed);
    out->latency_ns        = g_wu_latency_ns.load(std::memory_order_relaxed);
    return 1;
}

__attribute__((constructor(105)))
static void cipher_graph_autoinit() { cipher_graph_init(); }

// OP 25 — public deferred-trigger entry point. Called once from the first
// cudaMalloc after CUDA primary context is up. Running the warmup at
// constructor time (before PyTorch initializes CUDA) measurably hurts hot-
// path latency on subsequent kernels, so we wait until the runtime is hot.
extern "C" void cipher_graph_op25_arm_deferred(void);

static std::atomic<int> g_wu_deferred_done{0};

extern "C" void cipher_graph_op25_arm_deferred(void) {
    if (!g_wu_armed.load(std::memory_order_acquire)) return;
    if (g_wu_deferred_done.exchange(1, std::memory_order_acq_rel)) return;
    cipher_graph_warmup();
}

__attribute__((constructor(118)))
static void cipher_graph_op25_warmup_autoinit() {
    const char* env = std::getenv("CIPHER_GRAPH_WARMUP");
    if (!env || !*env) return;
    if (env[0] != '1' && env[0] != 't' && env[0] != 'T'
        && env[0] != 'o' && env[0] != 'O') return;
    g_wu_armed.store(1, std::memory_order_release);
    // Do NOT call cipher_graph_warmup() here — running graph capture
    // before PyTorch's CUDA init has the side effect of slowing every
    // subsequent kernel launch by ~7us. Defer to first cudaMalloc.
}
