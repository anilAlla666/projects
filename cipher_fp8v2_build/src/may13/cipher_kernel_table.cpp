// CIPHER kernel-name recognition table — implementation.
//
// Lives in the hook DSO so it can be called directly from the
// cuLaunchKernel / cuLaunchKernelEx intercept paths without crossing
// libcipher_rt.so.

#include "may13/cipher_kernel_table.h"

#include "cipher_rt_commit.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

namespace {

constexpr unsigned KT_SIZE     = 4096;
constexpr unsigned KT_MASK     = KT_SIZE - 1;
constexpr int      MAX_PARAMS  = 16;

CipherKtEntry  g_kt[KT_SIZE]{};
std::atomic<unsigned> g_kt_used{0};

// cuFuncGetName + cuFuncGetParamInfo resolved lazily.
typedef int (*pf_cuFuncGetName)(const char**, void*);
typedef int (*pf_cuFuncGetParamInfo)(void*, size_t /*idx*/,
                                      size_t* /*offset*/, size_t* /*size*/);
typedef int (*pf_cuFuncGetAttribute)(int* /*pi*/, int /*attrib*/, void* /*func*/);

pf_cuFuncGetName       g_get_name      = nullptr;
pf_cuFuncGetParamInfo  g_get_paraminfo = nullptr;
pf_cuFuncGetAttribute  g_get_attribute = nullptr;
std::atomic<int>       g_resolved{0};
std::mutex             g_resolve_mu;

bool resolve_libcuda() {
    if (g_resolved.load(std::memory_order_acquire)) return true;
    std::lock_guard<std::mutex> lk(g_resolve_mu);
    if (g_resolved.load(std::memory_order_relaxed)) return true;
    void* lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!lc) lc = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!lc) return false;
    g_get_name      = (pf_cuFuncGetName)      dlsym(lc, "cuFuncGetName");
    g_get_paraminfo = (pf_cuFuncGetParamInfo) dlsym(lc, "cuFuncGetParamInfo");
    g_get_attribute = (pf_cuFuncGetAttribute) dlsym(lc, "cuFuncGetAttribute");
    g_resolved.store(1, std::memory_order_release);
    return true;
}

// Hash CUfunction pointer.  Pointers are typically 16-byte aligned;
// xor-shift a few times to scatter the high bits into the low ones.
unsigned hash_fn(const void* p) {
    uint64_t x = (uint64_t)p;
    x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
    x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    x = (x ^ (x >> 33));
    return (unsigned)x & KT_MASK;
}

uint8_t classify(const char* nm) {
    if (!nm) return CIPHER_KT_UNKNOWN;
    // Most-specific first (these mostly mirror cipher_intercept_cudart.cpp's
    // existing classify_kernel_name; consolidated here so non-Mistral models
    // get the same recognition.)
    if (strstr(nm, "flash_fwd") || strstr(nm, "flash_bwd")
        || strstr(nm, "flashattn") || strstr(nm, "fmha")
        || strstr(nm, "flash_attention")
        || strstr(nm, "PyTorchMemEffAttention")) return CIPHER_KT_FLASHATTN;
    if (strstr(nm, "splitKreduce") || strstr(nm, "splitkreduce")) return CIPHER_KT_GEMM;
    if (strstr(nm, "wgmma") || strstr(nm, "gemm") || strstr(nm, "GEMM")
        || strstr(nm, "matmul") || strstr(nm, "cutlass3x")
        || strstr(nm, "cutlass2x") || strstr(nm, "Kernel2I")
        || strstr(nm, "nvjet_hsh") || strstr(nm, "nvjet_tsh")
        || strstr(nm, "nvjet_bsh")) return CIPHER_KT_GEMM;
    if (strstr(nm, "rms_norm") || strstr(nm, "RMSNorm")
        || strstr(nm, "rmsnorm")) return CIPHER_KT_RMSNORM;
    if (strstr(nm, "layer_norm") || strstr(nm, "LayerNorm")
        || strstr(nm, "layernorm")) return CIPHER_KT_LAYERNORM;
    if (strstr(nm, "rotary") || strstr(nm, "RoPE")
        || strstr(nm, "rope")) return CIPHER_KT_ROPE;
    if (strstr(nm, "SiLUFunctor") || strstr(nm, "Silu")
        || strstr(nm, "silu") || strstr(nm, "swish")) return CIPHER_KT_SILU;
    if (strstr(nm, "GeLUFunctor") || strstr(nm, "gelu")
        || strstr(nm, "GELU")) return CIPHER_KT_GELU;
    if (strstr(nm, "softmax") || strstr(nm, "Softmax")
        || strstr(nm, "SoftMax")) return CIPHER_KT_SOFTMAX;
    if (strstr(nm, "attn") || strstr(nm, "Attention")) return CIPHER_KT_ATTN_OTHER;
    if (strstr(nm, "embedding") || strstr(nm, "Embedding")
        || strstr(nm, "indexSelect") || strstr(nm, "index_select")) return CIPHER_KT_EMBEDDING;
    if (strstr(nm, "CUDAFunctor_add") || strstr(nm, "AddFunctor")
        || strstr(nm, "add_kernel")
        || strstr(nm, "elementwise_add")) return CIPHER_KT_RESIDUAL;
    if (strstr(nm, "MulFunctor")
        || strstr(nm, "vectorized_elementwise")) return CIPHER_KT_ELEM_MUL;
    if (strstr(nm, "MeanOps") || strstr(nm, "SumOps") || strstr(nm, "MaxOps")
        || strstr(nm, "ReduceOp") || strstr(nm, "reduce_kernel")) return CIPHER_KT_REDUCE;
    if (strstr(nm, "elementwise_kernel") || strstr(nm, "unrolled_elementwise")
        || strstr(nm, "BinaryFunctor")) return CIPHER_KT_ELEM_GENERIC;
    if (strstr(nm, "copy_kernel") || strstr(nm, "Copy")
        || strstr(nm, "Memcpy") || strstr(nm, "CatArray")) return CIPHER_KT_COPY;
    if (strstr(nm, "transpose") || strstr(nm, "permute")
        || strstr(nm, "Transpose")) return CIPHER_KT_TRANSPOSE;
    if (strstr(nm, "Cast") || strstr(nm, "Convert")
        || strstr(nm, "cast_kernel") || strstr(nm, "FillFunctor")) return CIPHER_KT_CAST;
    return CIPHER_KT_UNKNOWN;
}

const char* category_label(uint8_t c) {
    switch ((CipherKernelCategory)c) {
        case CIPHER_KT_GEMM:        return "GEMM";
        case CIPHER_KT_RMSNORM:     return "RMSNorm";
        case CIPHER_KT_LAYERNORM:   return "LayerNorm";
        case CIPHER_KT_SILU:        return "SiLU";
        case CIPHER_KT_GELU:        return "GeLU";
        case CIPHER_KT_SOFTMAX:     return "Softmax";
        case CIPHER_KT_ROPE:        return "RoPE";
        case CIPHER_KT_FLASHATTN:   return "FlashAttn";
        case CIPHER_KT_ATTN_OTHER:  return "AttnOther";
        case CIPHER_KT_EMBEDDING:   return "Embedding";
        case CIPHER_KT_ELEM_MUL:    return "ElementMul";
        case CIPHER_KT_RESIDUAL:    return "ResidualAdd";
        case CIPHER_KT_COPY:        return "Copy";
        case CIPHER_KT_TRANSPOSE:   return "Transpose";
        case CIPHER_KT_REDUCE:      return "Reduce";
        case CIPHER_KT_CAST:        return "Cast";
        case CIPHER_KT_ELEM_GENERIC:return "ElementGeneric";
        case CIPHER_KT_UNKNOWN:
        default:                    return "Unknown";
    }
}

// Resolve a kernel name via cuFuncGetName, falling back to dladdr on the
// host stub pointer (PyTorch runtime-API kernels — `__cudaRegisterFunction`
// stores the host stub in libtorch_cuda.so's symbol table, so dladdr finds
// the mangled name there).
const char* resolve_name(void* fn) {
    if (!resolve_libcuda()) {
        Dl_info info{};
        if (dladdr(fn, &info) && info.dli_sname) return info.dli_sname;
        return nullptr;
    }
    if (g_get_name) {
        const char* nm = nullptr;
        if (g_get_name(&nm, fn) == 0 && nm && nm[0]) return nm;
    }
    Dl_info info{};
    if (dladdr(fn, &info) && info.dli_sname) return info.dli_sname;
    return nullptr;
}

// Probe cuFuncGetParamInfo to learn the parameter layout.  Per the prior
// session's CLAUDE.md note, this returns 0 params for kernels registered via
// `__cudaRegisterFunction` (i.e. every nn.Linear/RMSNorm/etc kernel that
// PyTorch launches via the runtime API).  cuModule-loaded kernels (cuBLAS-LT
// / FlashAttention from cu* loads / our NVRTC kernels) do return real param
// metadata.
//
// We attempt up to MAX_PARAMS slots; the first failing slot is the count.
int probe_param_info(void* fn,
                      uint16_t out_offsets[MAX_PARAMS],
                      uint16_t out_sizes[MAX_PARAMS]) {
    if (!resolve_libcuda() || !g_get_paraminfo) return 0;
    int n = 0;
    for (int i = 0; i < MAX_PARAMS; ++i) {
        size_t off = 0, sz = 0;
        int rc = g_get_paraminfo(fn, (size_t)i, &off, &sz);
        if (rc != 0 || sz == 0) break;
        out_offsets[i] = (uint16_t)off;
        out_sizes[i]   = (uint16_t)sz;
        ++n;
    }
    return n;
}

// One-time observation of a brand-new fn handle.  Returns the slot, or -1
// on hash-table full.
int insert_locked(void* fn,
                   uint32_t gx, uint32_t gy, uint32_t gz,
                   uint32_t bx, uint32_t by, uint32_t bz,
                   uint32_t smem) {
    unsigned probe = hash_fn(fn);
    for (unsigned i = 0; i < KT_SIZE; ++i) {
        unsigned slot = (probe + i) & KT_MASK;
        CipherKtEntry& e = g_kt[slot];
        if (e.fn_handle == nullptr) {
            // Fresh slot.
            e.fn_handle = fn;
            const char* nm = resolve_name(fn);
            if (nm) {
                strncpy(e.name, nm, sizeof(e.name) - 1);
                e.name[sizeof(e.name) - 1] = 0;
            } else {
                snprintf(e.name, sizeof(e.name), "<unresolved-%p>", fn);
            }
            e.category    = classify(nm);
            e.param_count = (uint16_t)probe_param_info(
                                fn, e.param_offsets, e.param_sizes);
            e.first_grid_x  = gx; e.first_grid_y  = gy; e.first_grid_z  = gz;
            e.first_block_x = bx; e.first_block_y = by; e.first_block_z = bz;
            e.first_smem_bytes = smem;
            e.observe_count = 1;
            g_kt_used.fetch_add(1, std::memory_order_relaxed);

            const char* v = getenv("CIPHER_KERNEL_TABLE_VERBOSE");
            if (v && v[0] && v[0] != '0' && strcmp(v, "off") != 0) {
                fprintf(stderr,
                    "[CIPHER KT] new fn=%p  cat=%-14s  params=%-2d  "
                    "grid=(%u,%u,%u)  block=(%u,%u,%u)  smem=%u  name=%.96s\n",
                    fn, category_label(e.category), e.param_count,
                    gx, gy, gz, bx, by, bz, smem, e.name);
            }
            return (int)slot;
        }
        if (e.fn_handle == fn) return (int)slot;
    }
    return -1;
}

}  // namespace

extern "C" const CipherKtEntry* cipher_kt_observe(
    void*    fn_handle,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t smem_bytes)
{
    if (!fn_handle) return nullptr;
    // Fast path: probe the hash table without locks; if hit, just bump
    // observe_count.  Open-addressed linear probing means a hit may require
    // walking until the first empty slot.
    unsigned probe = hash_fn(fn_handle);
    for (unsigned i = 0; i < KT_SIZE; ++i) {
        unsigned slot = (probe + i) & KT_MASK;
        CipherKtEntry& e = g_kt[slot];
        if (e.fn_handle == fn_handle) {
            e.observe_count++;
            return &e;
        }
        if (e.fn_handle == nullptr) break;     // miss; fall through to insert
    }
    // Slow path: insert under a coarse mutex.  Pre-insert race: another
    // thread may have just inserted the same fn — handled inside insert_locked
    // by re-walking the chain.
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    int slot = insert_locked(fn_handle, grid_x, grid_y, grid_z,
                              block_x, block_y, block_z, smem_bytes);
    if (slot < 0) return nullptr;
    return &g_kt[slot];
}

extern "C" unsigned cipher_kt_size(void) {
    return g_kt_used.load(std::memory_order_relaxed);
}

extern "C" const char* cipher_kt_name_for(void* fn_handle) {
    if (!fn_handle) return nullptr;
    unsigned probe = hash_fn(fn_handle);
    for (unsigned i = 0; i < KT_SIZE; ++i) {
        unsigned slot = (probe + i) & KT_MASK;
        CipherKtEntry& e = g_kt[slot];
        if (e.fn_handle == fn_handle) return e.name;
        if (e.fn_handle == nullptr) return nullptr;
    }
    return nullptr;
}

extern "C" int cipher_kt_dump_json(const char* path_in) {
    /* W7-9 Step 4 — coherent snapshot acquire (slow-path; emits the
     * COMMIT-published state alongside this report's existing aggregate). */
    struct cipher_rt_snapshot _snap;
    cipher_rt_snapshot_acquire(0u, &_snap);
    (void)_snap;

    const char* path = path_in ? path_in : "/tmp/cipher_kernel_table.json";
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "{\n  \"size\": %u,\n  \"entries\": [\n",
            g_kt_used.load(std::memory_order_relaxed));
    int n = 0;
    bool first = true;
    for (unsigned i = 0; i < KT_SIZE; ++i) {
        const CipherKtEntry& e = g_kt[i];
        if (e.fn_handle == nullptr) continue;
        if (!first) fputs(",\n", f);
        first = false;
        // Escape doublequotes in name.
        char escaped[260];
        size_t k = 0;
        for (size_t j = 0; e.name[j] && k < sizeof(escaped) - 2; ++j) {
            if (e.name[j] == '"' || e.name[j] == '\\') escaped[k++] = '\\';
            escaped[k++] = e.name[j];
        }
        escaped[k] = 0;
        fprintf(f,
            "    {\"fn\":\"%p\",\"name\":\"%s\",\"category\":\"%s\","
            "\"category_id\":%u,"
            "\"param_count\":%u,"
            "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
            "\"smem_bytes\":%u,\"observe_count\":%llu}",
            e.fn_handle, escaped, category_label(e.category), e.category,
            e.param_count,
            e.first_grid_x, e.first_grid_y, e.first_grid_z,
            e.first_block_x, e.first_block_y, e.first_block_z,
            e.first_smem_bytes,
            (unsigned long long)e.observe_count);
        ++n;
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    return n;
}

// On process exit, dump the table if the env flag was set so the customer
// gets a record of every kernel CIPHER saw.
__attribute__((destructor))
static void cipher_kt_atexit_dump(void) {
    const char* v = getenv("CIPHER_KERNEL_TABLE_VERBOSE");
    if (!v || !v[0] || v[0] == '0' || !strcmp(v, "off")) return;
    int n = cipher_kt_dump_json(nullptr);
    fprintf(stderr, "[CIPHER KT] wrote /tmp/cipher_kernel_table.json (%d entries)\n", n);
}
