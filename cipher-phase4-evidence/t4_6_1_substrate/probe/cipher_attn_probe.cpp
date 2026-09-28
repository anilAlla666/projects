// cipher_attn_probe.cpp - multi-symbol probe.
// Hooks all three SDPA backend ::call entries and counts which fires
// during Mistral-7B inference.
//
// Backends hooked (PyTorch 2.11 / cu130):
//   _scaled_dot_product_flash_attention::call
//   _scaled_dot_product_efficient_attention::call
//   _scaled_dot_product_cudnn_attention::call

#include <ATen/ATen.h>
#include <c10/core/DispatchKeySet.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_ops.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention_ops.h>
#include <ATen/ops/_scaled_dot_product_cudnn_attention_ops.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>

using FlashRet = ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,
                              c10::SymInt,c10::SymInt,
                              at::Tensor,at::Tensor,at::Tensor>;
using FlashFn = FlashRet(*)(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            double, bool, bool, ::std::optional<double>);

using EffRet = ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
using EffFn = EffRet(*)(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const ::std::optional<at::Tensor>&, bool, double, bool,
                        ::std::optional<double>);

using CudnnRet = ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,
                              c10::SymInt,c10::SymInt,
                              at::Tensor,at::Tensor,at::Tensor>;
using CudnnFn = CudnnRet(*)(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                            const ::std::optional<at::Tensor>&, bool, double, bool, bool,
                            ::std::optional<double>);

static FlashFn  g_flash = nullptr;
static EffFn    g_eff   = nullptr;
static CudnnFn  g_cudnn = nullptr;

static std::atomic<uint64_t> g_n_flash{0}, g_n_eff{0}, g_n_cudnn{0};
static std::atomic<uint64_t> g_logged_flash{0}, g_logged_eff{0}, g_logged_cudnn{0};

static void* resolve(const char* sym) {
    dlerror();
    void* p = dlsym(RTLD_NEXT, sym);
    const char* err = dlerror();
    if (err) {
        fprintf(stderr, "[cipher-probe] dlsym(%s) failed: %s\n", sym, err);
        std::abort();
    }
    return p;
}

static void resolve_once() {
    if (g_flash) return;
    g_flash = reinterpret_cast<FlashFn>(resolve(
        "_ZN2at4_ops35_scaled_dot_product_flash_attention4callERKNS_6TensorES4_S4_dbbSt8optionalIdE"));
    g_eff = reinterpret_cast<EffFn>(resolve(
        "_ZN2at4_ops39_scaled_dot_product_efficient_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbS5_IdE"));
    g_cudnn = reinterpret_cast<CudnnFn>(resolve(
        "_ZN2at4_ops35_scaled_dot_product_cudnn_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbbS5_IdE"));
    fprintf(stderr, "[cipher-probe] hooks: flash=%p eff=%p cudnn=%p\n",
            (void*)g_flash, (void*)g_eff, (void*)g_cudnn);
}

static void log_shape(const char* tag, const at::Tensor& t) {
    auto sz = t.sizes();
    fprintf(stderr, "  %s [", tag);
    for (size_t i = 0; i < sz.size(); ++i)
        fprintf(stderr, "%s%lld", i ? "," : "", (long long)sz[i]);
    fprintf(stderr, "] %s %s\n", toString(t.scalar_type()), t.device().str().c_str());
}

extern "C"
FlashRet
_ZN2at4_ops35_scaled_dot_product_flash_attention4callERKNS_6TensorES4_S4_dbbSt8optionalIdE(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    double dp, bool ic, bool rdm, ::std::optional<double> sc)
{
    resolve_once();
    uint64_t n = ++g_n_flash;
    if (g_logged_flash.fetch_add(1) < 2) {
        fprintf(stderr, "[cipher-probe] FLASH#%llu dp=%g ic=%d\n",
                (unsigned long long)n, dp, (int)ic);
        log_shape("Q", q); log_shape("K", k); log_shape("V", v);
    }
    return g_flash(q, k, v, dp, ic, rdm, sc);
}

extern "C"
EffRet
_ZN2at4_ops39_scaled_dot_product_efficient_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbS5_IdE(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const ::std::optional<at::Tensor>& bias, bool cls, double dp, bool ic,
    ::std::optional<double> sc)
{
    resolve_once();
    uint64_t n = ++g_n_eff;
    if (g_logged_eff.fetch_add(1) < 2) {
        fprintf(stderr, "[cipher-probe] EFF#%llu dp=%g ic=%d\n",
                (unsigned long long)n, dp, (int)ic);
        log_shape("Q", q); log_shape("K", k); log_shape("V", v);
    }
    return g_eff(q, k, v, bias, cls, dp, ic, sc);
}

extern "C"
CudnnRet
_ZN2at4_ops35_scaled_dot_product_cudnn_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbbS5_IdE(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const ::std::optional<at::Tensor>& bias, bool cls, double dp, bool ic, bool rdm,
    ::std::optional<double> sc)
{
    resolve_once();
    uint64_t n = ++g_n_cudnn;
    if (g_logged_cudnn.fetch_add(1) < 2) {
        fprintf(stderr, "[cipher-probe] CUDNN#%llu dp=%g ic=%d\n",
                (unsigned long long)n, dp, (int)ic);
        log_shape("Q", q); log_shape("K", k); log_shape("V", v);
    }
    return g_cudnn(q, k, v, bias, cls, dp, ic, rdm, sc);
}

__attribute__((destructor))
static void cipher_probe_summary(void) {
    fprintf(stderr, "[cipher-probe] SUMMARY flash=%llu eff=%llu cudnn=%llu\n",
            (unsigned long long)g_n_flash.load(),
            (unsigned long long)g_n_eff.load(),
            (unsigned long long)g_n_cudnn.load());
}
