// CIPHER cudart compatibility shim.
//
// libcipher_rt.so historically linked against libcudart.so.12, which
// pulled cudart.so.12 into the address space as a NEEDED dependency.
// On vLLM 0.20 + torch 2.11+cu130, that splits CUDA state across two
// cudarts (.12 from CIPHER, .13 from torch) — fatbin registries diverge,
// `cudaLaunchKernel` returns "invalid device function" for kernels
// registered into the wrong cudart.
//
// This file replaces every cudart symbol the rt uses with a local
// wrapper that dlsym's the real implementation at first call from
// whichever cudart soname the host process loaded (RTLD_NOLOAD probe of
// .so.13 first, then .so.12, then unversioned). State is per-call: the
// resolved function pointer is cached, but the lookup is always against
// the live cudart.

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdint.h>

// Forward-declare types we don't include cudart headers for.
typedef int          cudaError_t;
typedef void*        cudaStream_t;

// Resolve cudart at first call; cache the handle.
static void* compat_cudart_handle() {
    static std::atomic<void*> cached{nullptr};
    void* h = cached.load(std::memory_order_acquire);
    if (h) return h;
    static const char* const v[] = {
        "libcudart.so.13", "libcudart.so.12", "libcudart.so.11",
        "libcudart.so", nullptr,
    };
    // NOLOAD only — never eagerly dlopen. An eager dlopen of the
    // generic name pulls in /usr/lib/x86_64-linux-gnu/libcudart.so.12
    // (the system default soname target), which corrupts cu13's
    // kernel registry on vLLM. If no cudart is loaded yet, return null
    // and let the caller's wrapper noop. By the time real cudaXxx
    // calls fire from the application, torch will have loaded the
    // right cudart and these probes will succeed.
    for (const char* const* p = v; *p; ++p) {
        h = dlopen(*p, RTLD_NOW | RTLD_NOLOAD);
        if (h) { cached.store(h, std::memory_order_release); return h; }
    }
    return nullptr;
}

template <typename Fn>
static Fn compat_resolve(const char* name, std::atomic<void*>& slot) {
    void* p = slot.load(std::memory_order_acquire);
    if (p) return reinterpret_cast<Fn>(p);
    void* h = compat_cudart_handle();
    if (!h) return nullptr;
    p = dlsym(h, name);
    if (p) slot.store(p, std::memory_order_release);
    return reinterpret_cast<Fn>(p);
}

#define WRAP0(rettype, fn, defret)                                         \
    extern "C" rettype fn() {                                              \
        static std::atomic<void*> slot{nullptr};                           \
        auto p = compat_resolve<rettype(*)()>(#fn, slot);                  \
        if (!p) return defret;                                             \
        return p();                                                        \
    }

#define WRAP1(rettype, fn, T1, defret)                                     \
    extern "C" rettype fn(T1 a1) {                                         \
        static std::atomic<void*> slot{nullptr};                           \
        auto p = compat_resolve<rettype(*)(T1)>(#fn, slot);                \
        if (!p) return defret;                                             \
        return p(a1);                                                      \
    }

#define WRAP2(rettype, fn, T1, T2, defret)                                 \
    extern "C" rettype fn(T1 a1, T2 a2) {                                  \
        static std::atomic<void*> slot{nullptr};                           \
        auto p = compat_resolve<rettype(*)(T1,T2)>(#fn, slot);             \
        if (!p) return defret;                                             \
        return p(a1, a2);                                                  \
    }

#define WRAP3(rettype, fn, T1, T2, T3, defret)                             \
    extern "C" rettype fn(T1 a1, T2 a2, T3 a3) {                           \
        static std::atomic<void*> slot{nullptr};                           \
        auto p = compat_resolve<rettype(*)(T1,T2,T3)>(#fn, slot);          \
        if (!p) return defret;                                             \
        return p(a1, a2, a3);                                              \
    }

#define WRAP4(rettype, fn, T1, T2, T3, T4, defret)                         \
    extern "C" rettype fn(T1 a1, T2 a2, T3 a3, T4 a4) {                    \
        static std::atomic<void*> slot{nullptr};                           \
        auto p = compat_resolve<rettype(*)(T1,T2,T3,T4)>(#fn, slot);       \
        if (!p) return defret;                                             \
        return p(a1, a2, a3, a4);                                          \
    }

#define WRAP5(rettype, fn, T1, T2, T3, T4, T5, defret)                     \
    extern "C" rettype fn(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5) {             \
        static std::atomic<void*> slot{nullptr};                           \
        auto p = compat_resolve<rettype(*)(T1,T2,T3,T4,T5)>(#fn, slot);    \
        if (!p) return defret;                                             \
        return p(a1, a2, a3, a4, a5);                                      \
    }

// ── Wrappers for every cudart symbol libcipher_rt.so uses ───────────────
//
// The hook (libcipher_hook.so) provides its own LD_PRELOAD'd
// cudaMalloc / cudaFree / cudaMemcpy / cudaMemcpyAsync / cudaLaunchKernel
// so we DO NOT define those here — defining them would conflict.
// The rt-side uses go through the hook's interpose first; the hook's
// shim then forwards to the real cudart via its own resolution path.

WRAP0(cudaError_t,  cudaDeviceSynchronize, 999)
WRAP0(cudaError_t,  cudaGetLastError, 999)
WRAP1(cudaError_t,  cudaCtxResetPersistingL2Cache, void*, 999)
WRAP1(cudaError_t,  cudaFreeHost, void*, 999)
WRAP1(cudaError_t,  cudaStreamDestroy, cudaStream_t, 999)
WRAP1(cudaError_t,  cudaGetDevice, int*, 999)
WRAP1(cudaError_t,  cudaGetDeviceCount, int*, 999)
WRAP1(cudaError_t,  cudaSetDevice, int, 999)
WRAP1(cudaError_t,  cudaStreamCreate, cudaStream_t*, 999)
WRAP1(const char*,  cudaGetErrorString, cudaError_t, "(no-cudart)")
WRAP2(cudaError_t,  cudaMallocManaged, void**, size_t, 999)
WRAP2(cudaError_t,  cudaMallocHost,    void**, size_t, 999)
WRAP2(cudaError_t,  cudaStreamSynchronize, cudaStream_t, unsigned, 999)
WRAP2(cudaError_t,  cudaStreamCreateWithPriority, cudaStream_t*, unsigned, 999)
WRAP3(cudaError_t,  cudaDeviceCanAccessPeer, int*, int, int, 999)
WRAP3(cudaError_t,  cudaDeviceGetAttribute, int*, int, int, 999)
WRAP3(cudaError_t,  cudaHostAlloc,    void**, size_t, unsigned, 999)
WRAP3(cudaError_t,  cudaMemset,       void*, int, size_t, 999)
WRAP3(cudaError_t,  cudaMemPrefetchAsync, const void*, size_t, int, 999)
WRAP4(cudaError_t,  cudaMemsetAsync,  void*, int, size_t, cudaStream_t, 999)
WRAP4(cudaError_t,  cudaGetDeviceProperties_v2, void*, int, void*, void*, 999)

// cudaLaunchCooperativeKernel(func,gridDim,blockDim,args,sharedMem,stream)
// — 6 args. Use a custom one-off wrapper.
extern "C" cudaError_t cudaLaunchCooperativeKernel(
    const void* func, void* gridDim4, void* blockDim4,
    void** args, size_t sharedMem, cudaStream_t stream)
{
    typedef cudaError_t (*F)(const void*, void*, void*, void**, size_t, cudaStream_t);
    static std::atomic<void*> slot{nullptr};
    auto p = compat_resolve<F>("cudaLaunchCooperativeKernel", slot);
    if (!p) return 999;
    return p(func, gridDim4, blockDim4, args, sharedMem, stream);
}

// cudaPointerGetAttributes(attrs, ptr) — 2 args.
extern "C" cudaError_t cudaPointerGetAttributes(void* attrs, const void* ptr) {
    typedef cudaError_t (*F)(void*, const void*);
    static std::atomic<void*> slot{nullptr};
    auto p = compat_resolve<F>("cudaPointerGetAttributes", slot);
    if (!p) return 999;
    return p(attrs, ptr);
}

// __cudaUnregisterFatBinary — emitted by nvcc as part of module dtor.
extern "C" void __cudaUnregisterFatBinary(void** handle) {
    typedef void (*F)(void**);
    static std::atomic<void*> slot{nullptr};
    auto p = compat_resolve<F>("__cudaUnregisterFatBinary", slot);
    if (!p) return;
    // Skip sentinels (CIPHER's own deferred-fatbins).
    uintptr_t v = (uintptr_t)handle;
    if (v >= 0xC1FE'0000'F0F0'0000ULL && v < 0xC1FE'0000'F0F0'0000ULL + 64) return;
    p(handle);
}

// __cudaRegisterVar — emitted by nvcc per __device__ variable.
// Fixed 9-arg signature.
//   handle, hostVar, deviceVar, name, ext, size, constant, global
extern "C" void __cudaRegisterVar(void** handle, char* hostVar, char* deviceVar,
                                   const char* name, int ext, size_t size,
                                   int constant, int global)
{
    typedef void (*F)(void**, char*, char*, const char*, int, size_t, int, int);
    static std::atomic<void*> slot{nullptr};
    auto p = compat_resolve<F>("__cudaRegisterVar", slot);
    if (!p) return;
    // If handle is our sentinel, skip — same logic as RegisterFunction
    // in the hook. Sentinels live in [0xC1FE0000F0F00000, +64).
    uintptr_t v = (uintptr_t)handle;
    if (v >= 0xC1FE'0000'F0F0'0000ULL && v < 0xC1FE'0000'F0F0'0000ULL + 64) return;
    p(handle, hostVar, deviceVar, name, ext, size, constant, global);
}
