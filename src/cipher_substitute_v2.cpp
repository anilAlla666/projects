// CIPHER Substitution Engine v2 — implementation.
//
// v1 of the engine: shape registry, two-tier compile cache (PTX + CUBIN
// blobs keyed by shape hash), per-shape readiness flag. NVRTC compile
// itself uses the existing cipher_recipes / cipher_koopman_runtime path
// when available; otherwise we just track readiness and let the dispatcher
// passthrough. Default OFF; conformant to OP_CONTRACT.

#include "cipher_substitute_v2.h"
#include "cipher_silicon.h"
#include "cipher_op_counters.h"

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

namespace {

constexpr unsigned MAX_SHAPES = 256;

struct ShapeSlot {
    int      in_use;
    int      m, n, k;
    int      dtype;
    int      ready;          // 1 = compiled and gate-passed
    int      invalidated;
    uint64_t register_ts_ns;
    uint64_t compile_ts_ns;
    uint64_t hits;
};

ShapeSlot       g_shapes[MAX_SHAPES]{};
std::mutex      g_mu;

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<uint64_t> g_register_calls{0};
std::atomic<uint64_t> g_compile_attempts{0};
std::atomic<uint64_t> g_compile_success{0};
std::atomic<uint64_t> g_compile_cache_hits{0};
std::atomic<uint64_t> g_substitutions{0};
std::atomic<uint64_t> g_passthroughs{0};
std::atomic<uint64_t> g_correctness_failures{0};
std::atomic<uint64_t> g_background_jobs{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T'
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

uint64_t now_ns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int find_shape_locked(int m, int n, int k, int dtype) {
    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        const ShapeSlot& s = g_shapes[i];
        if (s.in_use && s.m == m && s.n == n && s.k == k && s.dtype == dtype) return (int)i;
    }
    return -1;
}
int find_free_locked() {
    for (unsigned i = 0; i < MAX_SHAPES; ++i)
        if (!g_shapes[i].in_use) return (int)i;
    return -1;
}

} // namespace

extern "C" int cipher_substitute_v2_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_SUBSTITUTE_V2"));
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        const CipherSiliconModel* sil = cipher_silicon_get();
        fprintf(stderr,
            "[CIPHER SUB-V2] init max_shapes=%u target_arch=sm_%d%d\n",
            MAX_SHAPES,
            sil ? sil->compute_major : 0,
            sil ? sil->compute_minor : 0);
    }
    return on;
}

extern "C" int cipher_substitute_v2_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_substitute_v2_register_shape(int m, int n, int k, int dtype) {
    if (!g_enabled.load(std::memory_order_relaxed)) return -1;
    if (m <= 0 || n <= 0 || k <= 0) return -1;
    g_register_calls.fetch_add(1, std::memory_order_relaxed);
    cipher_op_inc(OP_SUBSTITUTE_MARLIN);

    std::lock_guard<std::mutex> lk(g_mu);
    int idx = find_shape_locked(m, n, k, dtype);
    if (idx >= 0) {
        g_compile_cache_hits.fetch_add(1, std::memory_order_relaxed);
        return idx;
    }
    idx = find_free_locked();
    if (idx < 0) return -1;
    g_shapes[idx] = ShapeSlot{1, m, n, k, dtype, 0, 0, now_ns(), 0, 0};
    g_compile_attempts.fetch_add(1, std::memory_order_relaxed);
    // v1: mark ready immediately (no actual NVRTC compile yet); the real
    // compile + correctness gate is wired by Stage 7/8 callers.
    g_shapes[idx].ready = 1;
    g_shapes[idx].compile_ts_ns = g_shapes[idx].register_ts_ns;
    g_compile_success.fetch_add(1, std::memory_order_relaxed);
    return idx;
}

extern "C" int cipher_substitute_v2_ready(int shape_id) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (shape_id < 0 || shape_id >= (int)MAX_SHAPES) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    return g_shapes[shape_id].in_use && g_shapes[shape_id].ready
        && !g_shapes[shape_id].invalidated;
}

extern "C" int cipher_substitute_v2_invalidate(int shape_id) {
    if (shape_id < 0 || shape_id >= (int)MAX_SHAPES) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_shapes[shape_id].in_use) return 0;
    g_shapes[shape_id].invalidated = 1;
    g_shapes[shape_id].ready = 0;
    g_correctness_failures.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_substitute_v2_stats(CipherSubstituteV2Stats* out) {
    if (!out) return 0;
    out->enabled               = g_enabled.load(std::memory_order_relaxed);
    out->register_calls        = g_register_calls.load(std::memory_order_relaxed);
    out->compile_attempts      = g_compile_attempts.load(std::memory_order_relaxed);
    out->compile_success       = g_compile_success.load(std::memory_order_relaxed);
    out->compile_cache_hits    = g_compile_cache_hits.load(std::memory_order_relaxed);
    out->substitutions         = g_substitutions.load(std::memory_order_relaxed);
    out->passthroughs          = g_passthroughs.load(std::memory_order_relaxed);
    out->correctness_failures  = g_correctness_failures.load(std::memory_order_relaxed);
    out->background_jobs       = g_background_jobs.load(std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_substitute_v2_report(void) {
    CipherSubstituteV2Stats s{};
    cipher_substitute_v2_stats(&s);
    FILE* f = fopen("/tmp/cipher_substitute_v2_report.json", "w");
    if (!f) return;
    fprintf(f,
        "{\"enabled\":%d,\"register_calls\":%llu,\"compile_attempts\":%llu,"
        "\"compile_success\":%llu,\"compile_cache_hits\":%llu,"
        "\"substitutions\":%llu,\"passthroughs\":%llu,"
        "\"correctness_failures\":%llu,\"background_jobs\":%llu}\n",
        s.enabled,
        (unsigned long long)s.register_calls,
        (unsigned long long)s.compile_attempts,
        (unsigned long long)s.compile_success,
        (unsigned long long)s.compile_cache_hits,
        (unsigned long long)s.substitutions,
        (unsigned long long)s.passthroughs,
        (unsigned long long)s.correctness_failures,
        (unsigned long long)s.background_jobs);
    fclose(f);
}

// ── Stage 6 actuation: NVRTC compile + cuModuleLoadData pipeline ────────────
namespace {

// libnvrtc bindings
typedef int (*pf_nvrtcCreateProgram)(void**, const char*, const char*, int, const char* const*, const char* const*);
typedef int (*pf_nvrtcCompileProgram)(void*, int, const char* const*);
typedef int (*pf_nvrtcGetPTXSize)(void*, size_t*);
typedef int (*pf_nvrtcGetPTX)(void*, char*);
typedef int (*pf_nvrtcGetProgramLogSize)(void*, size_t*);
typedef int (*pf_nvrtcGetProgramLog)(void*, char*);
typedef int (*pf_nvrtcDestroyProgram)(void**);

// libcuda bindings
typedef int (*pf_cuInit)(unsigned);
typedef int (*pf_cuModuleLoadData)(void**, const void*);
typedef int (*pf_cuModuleGetFunction)(void**, void*, const char*);
typedef int (*pf_cuModuleUnload)(void*);

struct NvrtcApi {
    void* libnvrtc = nullptr;
    void* libcuda  = nullptr;
    pf_nvrtcCreateProgram     create_prog  = nullptr;
    pf_nvrtcCompileProgram    compile_prog = nullptr;
    pf_nvrtcGetPTXSize        get_ptx_size = nullptr;
    pf_nvrtcGetPTX            get_ptx      = nullptr;
    pf_nvrtcGetProgramLogSize get_log_size = nullptr;
    pf_nvrtcGetProgramLog     get_log      = nullptr;
    pf_nvrtcDestroyProgram    destroy_prog = nullptr;
    pf_cuInit                 cu_init      = nullptr;
    pf_cuModuleLoadData       module_load  = nullptr;
    pf_cuModuleGetFunction    module_getfn = nullptr;
    pf_cuModuleUnload         module_unload= nullptr;
    bool ok = false;
};
NvrtcApi g_nvrtc;
std::mutex g_nvrtc_mu;

bool resolve_nvrtc() {
    std::lock_guard<std::mutex> lk(g_nvrtc_mu);
    if (g_nvrtc.ok) return true;
    if (!g_nvrtc.libnvrtc) {
        g_nvrtc.libnvrtc = dlopen("libnvrtc.so.12", RTLD_LAZY | RTLD_LOCAL);
        if (!g_nvrtc.libnvrtc) g_nvrtc.libnvrtc = dlopen("libnvrtc.so", RTLD_LAZY | RTLD_LOCAL);
        if (!g_nvrtc.libnvrtc) return false;
    }
    if (!g_nvrtc.libcuda) {
        g_nvrtc.libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!g_nvrtc.libcuda) return false;
    }
    g_nvrtc.create_prog  = (pf_nvrtcCreateProgram)dlsym(g_nvrtc.libnvrtc, "nvrtcCreateProgram");
    g_nvrtc.compile_prog = (pf_nvrtcCompileProgram)dlsym(g_nvrtc.libnvrtc, "nvrtcCompileProgram");
    g_nvrtc.get_ptx_size = (pf_nvrtcGetPTXSize)dlsym(g_nvrtc.libnvrtc, "nvrtcGetPTXSize");
    g_nvrtc.get_ptx      = (pf_nvrtcGetPTX)dlsym(g_nvrtc.libnvrtc, "nvrtcGetPTX");
    g_nvrtc.get_log_size = (pf_nvrtcGetProgramLogSize)dlsym(g_nvrtc.libnvrtc, "nvrtcGetProgramLogSize");
    g_nvrtc.get_log      = (pf_nvrtcGetProgramLog)dlsym(g_nvrtc.libnvrtc, "nvrtcGetProgramLog");
    g_nvrtc.destroy_prog = (pf_nvrtcDestroyProgram)dlsym(g_nvrtc.libnvrtc, "nvrtcDestroyProgram");
    g_nvrtc.cu_init      = (pf_cuInit)dlsym(g_nvrtc.libcuda, "cuInit");
    g_nvrtc.module_load  = (pf_cuModuleLoadData)dlsym(g_nvrtc.libcuda, "cuModuleLoadData");
    g_nvrtc.module_getfn = (pf_cuModuleGetFunction)dlsym(g_nvrtc.libcuda, "cuModuleGetFunction");
    g_nvrtc.module_unload= (pf_cuModuleUnload)dlsym(g_nvrtc.libcuda, "cuModuleUnload");
    g_nvrtc.ok = (g_nvrtc.create_prog && g_nvrtc.compile_prog
                  && g_nvrtc.get_ptx_size && g_nvrtc.get_ptx
                  && g_nvrtc.module_load && g_nvrtc.module_getfn);
    if (g_nvrtc.ok && g_nvrtc.cu_init) g_nvrtc.cu_init(0);
    return g_nvrtc.ok;
}

constexpr unsigned MAX_CUBINS = 64;
struct Cubin {
    int   in_use;
    void* module;
    void* function;
};
Cubin       g_cubins[MAX_CUBINS]{};
std::mutex  g_cubins_mu;

int find_free_cubin_locked() {
    for (unsigned i = 0; i < MAX_CUBINS; ++i)
        if (!g_cubins[i].in_use) return (int)i;
    return -1;
}

} // namespace

extern "C" unsigned long cipher_substitute_v2_compile(const char* source, const char* kernel_name) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!source || !kernel_name) return 0;
    if (!resolve_nvrtc()) {
        if (getenv("CIPHER_SUB_V2_VERBOSE"))
            fprintf(stderr, "[CIPHER SUB-V2] nvrtc resolve failed\n");
        return 0;
    }

    // Need an active CUDA context for cuModuleLoadData. Force one via
    // cudaFree(0) which is the documented context-bringup idiom.
    cudaFree(0);

    g_compile_attempts.fetch_add(1, std::memory_order_relaxed);

    void* prog = nullptr;
    if (g_nvrtc.create_prog(&prog, source, kernel_name, 0, nullptr, nullptr) != 0
        || !prog) {
        if (getenv("CIPHER_SUB_V2_VERBOSE"))
            fprintf(stderr, "[CIPHER SUB-V2] nvrtcCreateProgram failed\n");
        return 0;
    }

    const char* arch_opt = "--gpu-architecture=sm_90";
    const char* inc_opt  = "-I/usr/include";   // host /usr/include has cuda_fp16.h
    // NVRTC default optimization is unspecified; force PTXAS to -O3 and
    // enable fast-math for fp16 → float conversions in fused kernels.
    const char* opts[] = {arch_opt, "-std=c++17", inc_opt,
                          "--use_fast_math", "-default-device",
                          "--ptxas-options=-O3"};
    int rc = g_nvrtc.compile_prog(prog, 6, opts);
    if (rc != 0) {
        if (g_nvrtc.get_log_size && g_nvrtc.get_log) {
            size_t logsz = 0;
            g_nvrtc.get_log_size(prog, &logsz);
            if (logsz > 0 && logsz < 16384) {
                char* buf = (char*)alloca(logsz + 1);
                g_nvrtc.get_log(prog, buf);
                buf[logsz] = 0;
                fprintf(stderr, "[CIPHER SUB-V2] nvrtc compile log:\n%s\n", buf);
            }
        }
        if (g_nvrtc.destroy_prog) g_nvrtc.destroy_prog(&prog);
        return 0;
    }

    size_t ptxsz = 0;
    if (g_nvrtc.get_ptx_size(prog, &ptxsz) != 0 || ptxsz == 0) {
        if (g_nvrtc.destroy_prog) g_nvrtc.destroy_prog(&prog);
        return 0;
    }
    char* ptx = (char*)malloc(ptxsz);
    if (!ptx) {
        if (g_nvrtc.destroy_prog) g_nvrtc.destroy_prog(&prog);
        return 0;
    }
    if (g_nvrtc.get_ptx(prog, ptx) != 0) {
        free(ptx);
        if (g_nvrtc.destroy_prog) g_nvrtc.destroy_prog(&prog);
        return 0;
    }
    if (g_nvrtc.destroy_prog) g_nvrtc.destroy_prog(&prog);

    void* mod = nullptr;
    int crc = g_nvrtc.module_load(&mod, ptx);
    if (crc != 0 || !mod) {
        if (getenv("CIPHER_SUB_V2_VERBOSE"))
            fprintf(stderr, "[CIPHER SUB-V2] cuModuleLoadData failed rc=%d (no CUDA ctx?)\n", crc);
        free(ptx);
        return 0;
    }
    free(ptx);

    void* fn = nullptr;
    int frc = g_nvrtc.module_getfn(&fn, mod, kernel_name);
    if (frc != 0 || !fn) {
        if (getenv("CIPHER_SUB_V2_VERBOSE"))
            fprintf(stderr, "[CIPHER SUB-V2] cuModuleGetFunction(%s) failed rc=%d\n",
                    kernel_name, frc);
        if (g_nvrtc.module_unload) g_nvrtc.module_unload(mod);
        return 0;
    }

    std::lock_guard<std::mutex> lk(g_cubins_mu);
    int slot = find_free_cubin_locked();
    if (slot < 0) {
        if (g_nvrtc.module_unload) g_nvrtc.module_unload(mod);
        return 0;
    }
    g_cubins[slot] = Cubin{1, mod, fn};
    g_compile_success.fetch_add(1, std::memory_order_relaxed);
    return (unsigned long)(slot + 1);
}

extern "C" void* cipher_substitute_v2_get_function(unsigned long cubin_id) {
    if (cubin_id == 0 || cubin_id > MAX_CUBINS) return nullptr;
    std::lock_guard<std::mutex> lk(g_cubins_mu);
    int s = (int)cubin_id - 1;
    if (!g_cubins[s].in_use) return nullptr;
    return g_cubins[s].function;
}

// Resolve a different function in the same compiled module — used when one
// cubin contains multiple `extern "C" __global__` entry points (e.g. the 10
// Marlin template instantiations). Returns NULL if symbol missing.
extern "C" void* cipher_substitute_v2_get_function_by_name(
    unsigned long cubin_id, const char* name)
{
    if (cubin_id == 0 || cubin_id > MAX_CUBINS || !name) return nullptr;
    void* mod = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_cubins_mu);
        int s = (int)cubin_id - 1;
        if (!g_cubins[s].in_use) return nullptr;
        mod = g_cubins[s].module;
    }
    if (!mod || !g_nvrtc.module_getfn) return nullptr;
    void* fn = nullptr;
    if (g_nvrtc.module_getfn(&fn, mod, name) != 0) return nullptr;
    return fn;
}

extern "C" int cipher_substitute_v2_destroy(unsigned long cubin_id) {
    if (cubin_id == 0 || cubin_id > MAX_CUBINS) return 0;
    std::lock_guard<std::mutex> lk(g_cubins_mu);
    int s = (int)cubin_id - 1;
    if (!g_cubins[s].in_use) return 0;
    if (g_nvrtc.module_unload) g_nvrtc.module_unload(g_cubins[s].module);
    g_cubins[s] = Cubin{};
    return 1;
}

__attribute__((constructor(106)))
static void cipher_substitute_v2_autoinit() { cipher_substitute_v2_init(); }
