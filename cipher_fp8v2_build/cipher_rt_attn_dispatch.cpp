// SPDX-License-Identifier: GPL-2.0-or-later
//
// cipher_rt_attn_dispatch.cpp - Phase 4.6.1 attention substrate.
//
// Three trampolines (flash / efficient / cuDNN ::call) interpose on
// libtorch_cpu.so's SDPA dispatcher entries via plain LD_PRELOAD
// (these symbols are NOT version-tagged in libtorch_cpu.so — verified
// against torch 2.11.0+cu130). Each trampoline builds a
// cipher_rt_attn_call descriptor, walks the actuator registry, and on
// PASSTHROUGH calls through to the original via dlsym RTLD_NEXT.
//
// T4.6.1 ships PASSTHROUGH only. HANDLED/REDIRECTED paths are wired
// but exercised by future actuators (T4.6.3 L1 dedup).

#include "cipher_rt_attn_dispatch.h"
#include "cipher_rt_commit.h"
#include "cipher_stream_resolver.h"
#include "cipher_rt_got_patch.h"

#include <ATen/ATen.h>
#include <c10/core/DispatchKeySet.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_ops.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention_ops.h>
#include <ATen/ops/_scaled_dot_product_cudnn_attention_ops.h>

#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <pthread.h>

namespace {

using FlashRet = ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,
                              c10::SymInt,c10::SymInt,
                              at::Tensor,at::Tensor,at::Tensor>;
using FlashFn  = FlashRet(*)(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             double, bool, bool, ::std::optional<double>);

using EffRet   = ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
using EffFn    = EffRet(*)(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                           const ::std::optional<at::Tensor>&, bool, double, bool,
                           ::std::optional<double>);

using CudnnRet = ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,
                              c10::SymInt,c10::SymInt,
                              at::Tensor,at::Tensor,at::Tensor>;
using CudnnFn  = CudnnRet(*)(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const ::std::optional<at::Tensor>&, bool, double, bool, bool,
                             ::std::optional<double>);

constexpr const char* MANGLED_FLASH =
    "_ZN2at4_ops35_scaled_dot_product_flash_attention4callERKNS_6TensorES4_S4_dbbSt8optionalIdE";
constexpr const char* MANGLED_EFF =
    "_ZN2at4_ops39_scaled_dot_product_efficient_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbS5_IdE";
constexpr const char* MANGLED_CUDNN =
    "_ZN2at4_ops35_scaled_dot_product_cudnn_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbbS5_IdE";

struct Registry {
    pthread_mutex_t mu;
    cipher_rt_attn_actuator entries[CIPHER_RT_ATTN_MAX_ACTUATORS];
    int n;
};

static Registry g_reg = { PTHREAD_MUTEX_INITIALIZER, {}, 0 };

static std::atomic<FlashFn> g_flash{nullptr};
static std::atomic<EffFn>   g_eff{nullptr};
static std::atomic<CudnnFn> g_cudnn{nullptr};
static std::atomic<int> g_init_announced{0};

static std::atomic<uint64_t> g_total{0};
static std::atomic<uint64_t> g_handled{0};
static std::atomic<uint64_t> g_passthrough{0};
static std::atomic<uint64_t> g_redirected{0};
static std::atomic<uint64_t> g_per_backend[4] = {};  // index by enum value

// Coverage instrumentation (T4.6.1 hardening / op #2 coverage probe):
// g_tramp_calls counts EVERY trampoline invocation — eager, traced-fake,
// and compiled-real alike — before any guard. g_tramp_fake counts the
// invocations the FakeTensor guard exempts (torch.compile tracing).
// g_total (route()) counts only real, observed calls. Comparing
// g_tramp_calls/g_total against expected dispatch counts answers
// "does the substrate see compiled decode?".
static std::atomic<uint64_t> g_tramp_calls{0};
static std::atomic<uint64_t> g_tramp_fake{0};

// True if q/k/v are real, observable tensors. False for FakeTensor /
// meta / functional tensors seen during torch.compile (dynamo) tracing —
// calling data_ptr() on those throws. Dispatch-key fast path; the
// try/catch in build_attn_descriptor is the certain safety net.
static bool tensors_are_real(const at::Tensor& q, const at::Tensor& k,
                             const at::Tensor& v) {
    const at::Tensor* ts[3] = { &q, &k, &v };
    for (int i = 0; i < 3; ++i) {
        const at::Tensor& t = *ts[i];
        if (!t.defined()) continue;
        if (t.is_meta()) return false;
        c10::DispatchKeySet ks = t.key_set();
        if (ks.has(c10::DispatchKey::Python) ||
            ks.has(c10::DispatchKey::Meta) ||
            ks.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode))
            return false;
    }
    return true;
}

static int env_on(const char* name) {
    const char* v = getenv(name);
    if (!v) return 0;
    return v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T';
}

static void fill_view(cipher_rt_attn_tensor& out, const at::Tensor& t) {
    out.data_ptr = t.defined() ? t.data_ptr() : nullptr;
    int r = t.defined() ? static_cast<int>(t.dim()) : 0;
    if (r > 4) r = 4;
    out.rank = r;
    for (int i = 0; i < 4; ++i) { out.sizes[i] = 0; out.strides[i] = 0; }
    if (t.defined()) {
        auto sz = t.sizes();
        auto st = t.strides();
        for (int i = 0; i < r; ++i) {
            out.sizes[i]   = sz[i];
            out.strides[i] = st[i];
        }
        out.dtype = static_cast<int>(t.scalar_type());
        out.device_type  = static_cast<int>(t.device().type());
        out.device_index = t.device().index();
    } else {
        out.dtype = -1;
        out.device_type = -1;
        out.device_index = -1;
    }
}

static int route(const cipher_rt_attn_call& call) {
    g_total.fetch_add(1, std::memory_order_relaxed);
    int idx = static_cast<int>(call.backend);
    if (idx >= 1 && idx <= 3)
        g_per_backend[idx].fetch_add(1, std::memory_order_relaxed);

    // Snapshot actuator list under lock; release lock before invoking
    // maybe_handle (actuator may call back into substrate APIs).
    cipher_rt_attn_actuator snap[CIPHER_RT_ATTN_MAX_ACTUATORS];
    int n = 0;
    pthread_mutex_lock(&g_reg.mu);
    n = g_reg.n;
    memcpy(snap, g_reg.entries, sizeof(cipher_rt_attn_actuator) * n);
    pthread_mutex_unlock(&g_reg.mu);

    for (int i = 0; i < n; ++i) {
        if (!snap[i].maybe_handle) continue;
        int r = snap[i].maybe_handle(&call);
        switch (r) {
        case CIPHER_RT_ATTN_HANDLED:
            g_handled.fetch_add(1, std::memory_order_relaxed);
            {
                /* W10-12 Step 2 SDPA stream-resolution carry. */
                uint32_t tid = cipher_v2_current_tenant_id_from_stream(
                    (uintptr_t)call.stream);
                if (tid == CIPHER_STREAM_TENANT_NONE_USER) tid = 0u;
                (void)cipher_rt_commit_observe_and_publish(tid);
            }
            return CIPHER_RT_ATTN_HANDLED;
        case CIPHER_RT_ATTN_REDIRECTED:
            // T4.6.3+ — substitution mechanism deferred. For now,
            // treat as PASSTHROUGH but count separately so we know
            // an actuator wanted to redirect.
            g_redirected.fetch_add(1, std::memory_order_relaxed);
            fprintf(stderr, "[cipher-attn] actuator '%s' returned REDIRECTED "
                    "but substitution path not implemented in T4.6.1 — "
                    "passing through\n", snap[i].name ? snap[i].name : "(unnamed)");
            break;
        case CIPHER_RT_ATTN_ERROR:
            fprintf(stderr, "[cipher-attn] actuator '%s' ERROR — falling through\n",
                    snap[i].name ? snap[i].name : "(unnamed)");
            break;
        case CIPHER_RT_ATTN_PASSTHROUGH:
        default:
            break;
        }
    }
    g_passthrough.fetch_add(1, std::memory_order_relaxed);
    {
        /* W10-12 Step 2 SDPA stream-resolution carry (PASSTHROUGH return). */
        uint32_t tid = cipher_v2_current_tenant_id_from_stream(
            (uintptr_t)call.stream);
        if (tid == CIPHER_STREAM_TENANT_NONE_USER) tid = 0u;
        (void)cipher_rt_commit_observe_and_publish(tid);
    }
    return CIPHER_RT_ATTN_PASSTHROUGH;
}

// W12 Step 4 D5 backfill — lazy resolution of c10::cuda::getCurrentCUDAStream.
//
// libcipher_rt.so can't statically link against libtorch_cuda.so because
// libtorch_cuda is not loaded when libcipher_rt is dlopen'd via
// CUDA_INJECTION64_PATH. We resolve the symbol at first SDPA trampoline
// call via dlsym, mirroring the resolve_lazy pattern below.
//
// The mangled symbol _ZN3c104cuda20getCurrentCUDAStreamEa is
// c10::cuda::getCurrentCUDAStream(c10::DeviceIndex device_index = -1).
// It returns c10::cuda::CUDAStream by value. CUDAStream is a thin
// wrapper around c10::Stream which holds a single int64_t stream_id_.
// The actual cudaStream_t can be extracted by passing the stream
// through CUDAStream::stream() — that's another libtorch_cuda symbol.
// To avoid two dlsym chains, we instead call the C ABI helper
// at::cuda::getCurrentCUDAStream(device).stream() pattern indirectly
// via the c10 device-checking + cuda runtime; or we just call the
// well-known cuStreamGetCtx-free workaround: query the cuda runtime
// for the current stream via cudaGetDevice + per-thread stream tracking.
//
// Pragmatic choice: cipher_rt_phase4 already links libcudart for other
// paths. cudaStreamPerThread is a well-known sentinel handle that, when
// passed to cuda kernels, becomes the per-thread default. But that
// doesn't give us the c10 active stream — it gives a sentinel.
//
// The cleanest solution: dlsym to get a function pointer of signature
// (returns 8-byte CUDAStream, takes int8 DeviceIndex). The CUDAStream
// struct layout is { Stream stream_ } where Stream is { StreamData3 }
// which is { DeviceIndex device_index_; uint8_t device_type_;
//            StreamId stream_id_ } — 8 bytes total. We treat the
// returned value as a 16-byte struct (CUDAStream contains 8 bytes of
// Stream wrapper + alignment padding to 16 in some libtorch builds).
// Safer: use the actual CUDAStream class through its .stream() method
// which requires a second dlsym.
//
// We take a different approach: query the cuStream via the
// libtorch-internal `at::cuda::getCurrentCUDAStream(device).stream()`
// chained call. Implemented as: dlsym getCurrentCUDAStream, call it
// (returns CUDAStream-by-value), then dlsym CUDAStream::stream() and
// call on the returned struct. Both symbols resolve from libtorch_cuda.so.

// Function-pointer slot for c10::cuda::getCurrentCUDAStream(int8_t).
// CUDAStream is 8 bytes (single Stream member which is int64_t stream_id_).
// Some libtorch builds may have additional fields; we treat the return
// as 16 bytes to be safe.
struct cipher_cudastream_repr { uint64_t lo; uint64_t hi; };
typedef cipher_cudastream_repr (*get_cur_stream_fn)(int8_t);
static std::atomic<get_cur_stream_fn> g_get_cur_stream{nullptr};

/* CUDAStream::stream() const returns cudaStream_t; treat as void* to
 * avoid pulling cuda_runtime.h into this TU. */
typedef void * (*cudastream_stream_fn)(const cipher_cudastream_repr*);
static std::atomic<cudastream_stream_fn> g_cudastream_stream{nullptr};

static void *resolve_cuda_lazy(const char *symbol_mangled) {
    void *h = dlopen("libtorch_cuda.so", RTLD_NOLOAD | RTLD_NOW);
    if (!h) {
        /* fall back to libtorch.so / libc10_cuda.so which sometimes
         * carries the c10 namespace */
        h = dlopen("libc10_cuda.so", RTLD_NOLOAD | RTLD_NOW);
    }
    if (!h) return nullptr;
    void *sym = dlsym(h, symbol_mangled);
    if (!sym) {
        fprintf(stderr, "[cipher-attn] D5 backfill: dlsym(%s) returned NULL; "
                "SDPA trampoline falls back to NULL stream (tenant_id=0)\n",
                symbol_mangled);
    }
    return sym;
}

/* Returns the cudaStream_t the current c10 device-stream is bound to,
 * or NULL if libtorch_cuda symbols aren't reachable. NULL routes the
 * resolver to NONE then fallback to tenant 0 (Step 4 baseline behavior). */
static void *cipher_attn_current_cuda_stream(void) {
    get_cur_stream_fn fn1 = g_get_cur_stream.load(std::memory_order_acquire);
    if (!fn1) {
        fn1 = (get_cur_stream_fn)resolve_cuda_lazy(
            "_ZN3c104cuda20getCurrentCUDAStreamEa");
        if (!fn1) return nullptr;
        g_get_cur_stream.store(fn1, std::memory_order_release);
    }
    cudastream_stream_fn fn2 = g_cudastream_stream.load(std::memory_order_acquire);
    if (!fn2) {
        /* c10::cuda::CUDAStream::stream() const → cudaStream_t */
        fn2 = (cudastream_stream_fn)resolve_cuda_lazy(
            "_ZNK3c104cuda10CUDAStream6streamEv");
        if (!fn2) return nullptr;
        g_cudastream_stream.store(fn2, std::memory_order_release);
    }
    cipher_cudastream_repr cs = fn1((int8_t)-1);  /* current device */
    return fn2(&cs);
}

// Per-symbol lazy resolution. Called from each trampoline's first call.
//
// CP 2.5: dlsym(RTLD_NEXT) is wrong under CUDA_INJECTION64_PATH-only --
// libcipher_rt is loaded LAST, so RTLD_NEXT from libcipher_rt sees nothing
// further and never finds libtorch_cpu's real SDPA op. Resolve the real
// function straight out of libtorch_cpu.so by handle instead: dlopen with
// RTLD_NOLOAD returns the already-loaded object (it is loaded in any
// process whose GOT the patcher touched), and dlsym(handle, ...) returns
// libtorch_cpu's own definition regardless of link-map scope or our own
// (hidden) trampoline symbols. Verified: soname is "libtorch_cpu.so".
template <typename FnT>
static FnT resolve_lazy(std::atomic<FnT>& slot, const char* mangled) {
    FnT cur = slot.load(std::memory_order_acquire);
    if (cur) return cur;
    void* h = dlopen("libtorch_cpu.so", RTLD_NOLOAD | RTLD_NOW);
    FnT got = h ? reinterpret_cast<FnT>(dlsym(h, mangled)) : nullptr;
    const char* err = h ? "symbol not found in libtorch_cpu.so"
                        : "libtorch_cpu.so not loaded";
    if (got) {
        FnT expected = nullptr;
        slot.compare_exchange_strong(expected, got);
        return slot.load(std::memory_order_acquire);
    }
    fprintf(stderr, "[cipher-attn] resolve %.40s... from libtorch_cpu.so: %s\n",
            mangled, err);
    return nullptr;
}

}  // namespace

extern "C" {

int cipher_rt_attn_register_actuator(const cipher_rt_attn_actuator* a) {
    if (!a) return -1;
    pthread_mutex_lock(&g_reg.mu);
    if (g_reg.n >= CIPHER_RT_ATTN_MAX_ACTUATORS) {
        pthread_mutex_unlock(&g_reg.mu);
        return -1;
    }
    // Insert in priority order (lower first).
    int pos = g_reg.n;
    for (int i = 0; i < g_reg.n; ++i) {
        if (a->priority < g_reg.entries[i].priority) { pos = i; break; }
    }
    for (int i = g_reg.n; i > pos; --i)
        g_reg.entries[i] = g_reg.entries[i - 1];
    g_reg.entries[pos] = *a;
    g_reg.n += 1;
    pthread_mutex_unlock(&g_reg.mu);
    fprintf(stderr, "[cipher-attn] registered actuator '%s' priority=%d (n=%d)\n",
            a->name ? a->name : "(unnamed)", a->priority, g_reg.n);
    return 0;
}

int cipher_rt_attn_dispatch_init(void) {
    // Best-effort eager resolution; failure is fine (we retry per-call).
    // Announce once for the operator log so they know substrate is wired.
    int expected = 0;
    if (g_init_announced.compare_exchange_strong(expected, 1)) {
        fprintf(stderr,
            "[cipher-attn] substrate active (torch verified %s); symbol "
            "resolution is lazy per-call\n", CIPHER_RT_ATTN_TORCH_VERIFIED);
    }
    return 0;
}

__attribute__((destructor))
static void cipher_rt_attn_dispatch_summary(void) {
    if (!g_init_announced.load()) return;  // substrate never armed
    fprintf(stderr,
        "[cipher-attn] exit totals - tramp_calls=%lu tramp_fake=%lu "
        "observed=%lu handled=%lu passthrough=%lu redirected=%lu "
        "(flash=%lu eff=%lu cudnn=%lu)\n",
        g_tramp_calls.load(), g_tramp_fake.load(),
        g_total.load(), g_handled.load(), g_passthrough.load(),
        g_redirected.load(),
        g_per_backend[1].load(), g_per_backend[2].load(), g_per_backend[3].load());
}

unsigned long cipher_rt_attn_tramp_calls(void)        { return g_tramp_calls.load(); }
unsigned long cipher_rt_attn_tramp_fake(void)         { return g_tramp_fake.load(); }
unsigned long cipher_rt_attn_calls_total(void)        { return g_total.load(); }
unsigned long cipher_rt_attn_calls_handled(void)      { return g_handled.load(); }
unsigned long cipher_rt_attn_calls_passthrough(void)  { return g_passthrough.load(); }
unsigned long cipher_rt_attn_calls_redirected(void)   { return g_redirected.load(); }
unsigned long cipher_rt_attn_calls_by_backend(enum cipher_rt_attn_backend b) {
    int idx = static_cast<int>(b);
    if (idx < 1 || idx > 3) return 0;
    return g_per_backend[idx].load();
}

}  // extern "C"

// ============================================================================
// GOT-patch trampolines (CP 2.5).
//
// These three functions are the trampoline bodies. CP 2.5 drops LD_PRELOAD
// link-order interposition: cipher_rt_attn_register_got() registers their
// addresses with the GOT patcher, which writes them into libtorch_cpu.so's
// GOT slots for the SDPA `::call` symbols. They keep the mangled-name
// identifiers (under extern "C", so the symbol name is exactly that) but
// are HIDDEN -- libcipher_rt must not export ATen op symbols. The originals
// are resolved via dlopen(libtorch_cpu.so)+dlsym (resolve_lazy), not the
// patched GOT slot. The identifiers are referenced by MANGLED_* in
// cipher_rt_attn_register_got below.
// ============================================================================

#pragma GCC visibility push(hidden)
extern "C" {

FlashRet
_ZN2at4_ops35_scaled_dot_product_flash_attention4callERKNS_6TensorES4_S4_dbbSt8optionalIdE(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    double dropout_p, bool is_causal, bool return_debug_mask,
    ::std::optional<double> scale)
{
    g_tramp_calls.fetch_add(1, std::memory_order_relaxed);
    FlashFn orig = resolve_lazy(g_flash, MANGLED_FLASH);
    if (!orig) {
        fprintf(stderr, "[cipher-attn] FATAL: flash dlsym never resolved; "
                "cannot fall through. Aborting (host application would "
                "otherwise infinite-loop on self-call).\n");
        abort();
    }
    if (tensors_are_real(q, k, v)) {
        try {
            cipher_rt_attn_call c{};
            c.backend = CIPHER_RT_ATTN_BACKEND_FLASH;
            fill_view(c.q, q); fill_view(c.k, k); fill_view(c.v, v);
            c.dropout_p = dropout_p;
            c.is_causal = is_causal ? 1 : 0;
            c.return_debug_mask = return_debug_mask ? 1 : 0;
            c.compute_log_sumexp = 0;
            if (scale.has_value()) { c.scale = *scale; c.scale_is_set = 1; }
            /* W12 Step 4 D5 backfill: populate c.stream with the c10
             * current CUDA stream so cipher_v2_current_tenant_id_from_stream
             * resolves to the correct tenant. W11 Step 2 added the field
             * but left construction sites at NULL; D5 closes that. */
            c.stream = cipher_attn_current_cuda_stream();
            int r = route(c);
            if (r == CIPHER_RT_ATTN_HANDLED) {
                /* Wave 5 LP-2 (Week 2 Step 1): HANDLED unlocks future attn
                 * substitute actuator (v1.5). Actuator wrote substitute
                 * result into c.out_status_devptr; trampoline would
                 * reconstruct FlashRet from it. NOT IMPLEMENTED in v1
                 * (no actuator returns HANDLED on attn today: AUDIT and
                 * test_actuator both return PASSTHROUGH). Defensive abort
                 * surfaces accidental HANDLED before v1.5 lands the
                 * return-construction path. */
                fprintf(stderr,
                    "[cipher-attn] FATAL: flash trampoline saw HANDLED return "
                    "from actuator chain, but the HANDLED-return-construction "
                    "path is v1.5 work (out_status_devptr=%p). Aborting to "
                    "surface the design gap.\n", c.out_status_devptr);
                abort();
            }
        } catch (...) {
            /* safety net: a non-real tensor slipped the key check */
            g_tramp_fake.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        g_tramp_fake.fetch_add(1, std::memory_order_relaxed);
    }
    return orig(q, k, v, dropout_p, is_causal, return_debug_mask, scale);
}

EffRet
_ZN2at4_ops39_scaled_dot_product_efficient_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbS5_IdE(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const ::std::optional<at::Tensor>& attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, ::std::optional<double> scale)
{
    g_tramp_calls.fetch_add(1, std::memory_order_relaxed);
    EffFn orig = resolve_lazy(g_eff, MANGLED_EFF);
    if (!orig) {
        fprintf(stderr, "[cipher-attn] FATAL: efficient dlsym never resolved\n");
        abort();
    }
    if (tensors_are_real(q, k, v)) {
        try {
            cipher_rt_attn_call c{};
            c.backend = CIPHER_RT_ATTN_BACKEND_EFFICIENT;
            fill_view(c.q, q); fill_view(c.k, k); fill_view(c.v, v);
            c.dropout_p = dropout_p;
            c.is_causal = is_causal ? 1 : 0;
            c.return_debug_mask = 0;
            c.compute_log_sumexp = compute_log_sumexp ? 1 : 0;
            if (scale.has_value()) { c.scale = *scale; c.scale_is_set = 1; }
            if (attn_bias.has_value() && attn_bias->defined()) {
                c.attn_bias_data = attn_bias->data_ptr();
                c.attn_bias_rank = static_cast<int>(std::min<int64_t>(4, attn_bias->dim()));
                for (int i = 0; i < c.attn_bias_rank; ++i)
                    c.attn_bias_sizes[i] = attn_bias->sizes()[i];
            }
            /* W12 Step 4 D5 backfill — see flash trampoline above. */
            c.stream = cipher_attn_current_cuda_stream();
            int r = route(c);
            if (r == CIPHER_RT_ATTN_HANDLED) {
                /* Wave 5 LP-2 (Week 2 Step 1) — see flash trampoline for the
                 * full justification. EffRet return-construction is v1.5 work. */
                fprintf(stderr,
                    "[cipher-attn] FATAL: efficient trampoline saw HANDLED "
                    "return; HANDLED-return-construction path is v1.5 work "
                    "(out_status_devptr=%p). Aborting.\n", c.out_status_devptr);
                abort();
            }
        } catch (...) {
            g_tramp_fake.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        g_tramp_fake.fetch_add(1, std::memory_order_relaxed);
    }
    return orig(q, k, v, attn_bias, compute_log_sumexp, dropout_p, is_causal, scale);
}

CudnnRet
_ZN2at4_ops35_scaled_dot_product_cudnn_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbbS5_IdE(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const ::std::optional<at::Tensor>& attn_bias, bool compute_log_sumexp,
    double dropout_p, bool is_causal, bool return_debug_mask,
    ::std::optional<double> scale)
{
    g_tramp_calls.fetch_add(1, std::memory_order_relaxed);
    CudnnFn orig = resolve_lazy(g_cudnn, MANGLED_CUDNN);
    if (!orig) {
        fprintf(stderr, "[cipher-attn] FATAL: cudnn dlsym never resolved\n");
        abort();
    }
    if (tensors_are_real(q, k, v)) {
        try {
            cipher_rt_attn_call c{};
            c.backend = CIPHER_RT_ATTN_BACKEND_CUDNN;
            fill_view(c.q, q); fill_view(c.k, k); fill_view(c.v, v);
            c.dropout_p = dropout_p;
            c.is_causal = is_causal ? 1 : 0;
            c.return_debug_mask = return_debug_mask ? 1 : 0;
            c.compute_log_sumexp = compute_log_sumexp ? 1 : 0;
            if (scale.has_value()) { c.scale = *scale; c.scale_is_set = 1; }
            if (attn_bias.has_value() && attn_bias->defined()) {
                c.attn_bias_data = attn_bias->data_ptr();
                c.attn_bias_rank = static_cast<int>(std::min<int64_t>(4, attn_bias->dim()));
                for (int i = 0; i < c.attn_bias_rank; ++i)
                    c.attn_bias_sizes[i] = attn_bias->sizes()[i];
            }
            /* W12 Step 4 D5 backfill — see flash trampoline above. */
            c.stream = cipher_attn_current_cuda_stream();
            int r = route(c);
            if (r == CIPHER_RT_ATTN_HANDLED) {
                /* Wave 5 LP-2 (Week 2 Step 1) — see flash trampoline for the
                 * full justification. CudnnRet return-construction is v1.5 work. */
                fprintf(stderr,
                    "[cipher-attn] FATAL: cudnn trampoline saw HANDLED return; "
                    "HANDLED-return-construction path is v1.5 work "
                    "(out_status_devptr=%p). Aborting.\n", c.out_status_devptr);
                abort();
            }
        } catch (...) {
            g_tramp_fake.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        g_tramp_fake.fetch_add(1, std::memory_order_relaxed);
    }
    return orig(q, k, v, attn_bias, compute_log_sumexp, dropout_p, is_causal,
                return_debug_mask, scale);
}

}  // extern "C"
#pragma GCC visibility pop

// ============================================================================
// CP 2.5 — GOT-patch registration. Called from cipher_inject.c before
// cipher_rt_got_patch_init(). Registers the three SDPA `::call` mangled
// names so the patcher writes the trampoline addresses into libtorch_cpu's
// GOT slots. The trampoline identifiers ARE the mangled names (extern "C").
// ============================================================================
extern "C" void cipher_rt_attn_register_got(void) {
    cipher_rt_got_register(MANGLED_FLASH, reinterpret_cast<void*>(
        &_ZN2at4_ops35_scaled_dot_product_flash_attention4callERKNS_6TensorES4_S4_dbbSt8optionalIdE),
        nullptr);
    cipher_rt_got_register(MANGLED_EFF, reinterpret_cast<void*>(
        &_ZN2at4_ops39_scaled_dot_product_efficient_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbS5_IdE),
        nullptr);
    cipher_rt_got_register(MANGLED_CUDNN, reinterpret_cast<void*>(
        &_ZN2at4_ops35_scaled_dot_product_cudnn_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbbS5_IdE),
        nullptr);
}
