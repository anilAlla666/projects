// Phase 3 graph surgery — Step 1 (introspection).
//
// Intercepts cuGraphInstantiateWithFlags / cuGraphInstantiate so that, after
// PyTorch has captured a stream-graph and asked the driver to compile it, we
// can walk the resulting CUgraph, list every kernel-node, look up each node's
// CUfunction name, and tally what's there. This answers a critical Phase-3
// question before we attempt any node-rewriting:
//
//   "After torch.cuda.graph(...) of an Int4Linear-substituted Mistral forward,
//    is Marlin already in the graph for M≤64, or did capture somehow miss our
//    NVRTC-compiled kernels and fall back to cuBLAS?"
//
// Gated by env var CIPHER_GRAPH_INSPECT=on. With surgery deferred (Phase 4),
// this file is read-only — it never edits the executable graph.
//
// All driver entrypoints resolved lazily via dlsym from libcuda.so.1 to
// preserve the existing libcipher_hook.so layout (no static link to libcuda).

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
typedef int CUresult;
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef void* CUgraphNode;
typedef void* CUfunction;

typedef CUresult (*pf_cuGraphGetNodes)(CUgraph, CUgraphNode*, size_t*);
typedef CUresult (*pf_cuGraphNodeGetType)(CUgraphNode, int*);
typedef CUresult (*pf_cuFuncGetName)(const char**, CUfunction);

// Layout of CUDA_KERNEL_NODE_PARAMS (CUDA 12+, common across versions for
// the leading fields we care about).
struct CipherKernelNodeParams_v1 {
    CUfunction func;
    unsigned   gridDimX, gridDimY, gridDimZ;
    unsigned   blockDimX, blockDimY, blockDimZ;
    unsigned   sharedMemBytes;
    void**     kernelParams;
    void**     extra;
};
typedef CUresult (*pf_cuGraphKernelNodeGetParams)(CUgraphNode,
        CipherKernelNodeParams_v1*);

// Real driver symbols we wrap.
typedef CUresult (*pf_cuGraphInstantiateWithFlags)(CUgraphExec*, CUgraph,
                                                    unsigned long long);
typedef CUresult (*pf_cuGraphInstantiate)(CUgraphExec*, CUgraph,
                                           void*, void*, unsigned long long);

// Runtime API counterpart that PyTorch actually calls. Same handle types as
// the driver, so cuGraphGetNodes etc. work on a cudart-instantiated graph.
typedef int      (*pf_cudaGraphInstantiate)(void*, void*,
                                             void*, void*, size_t);
typedef int      (*pf_cudaGraphInstantiateWithFlags)(void*, void*,
                                                      unsigned long long);
}

namespace {

constexpr int CU_GRAPH_NODE_TYPE_KERNEL = 0;

bool inspect_enabled() {
    static int s = -1;
    if (s == -1) {
        const char* v = std::getenv("CIPHER_GRAPH_INSPECT");
        s = (v && *v && std::strcmp(v, "0") != 0) ? 1 : 0;
    }
    return s == 1;
}

void* libcuda_handle() {
    static void* h = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
        if (!h) h = dlopen("libcuda.so",   RTLD_LAZY | RTLD_GLOBAL);
    });
    return h;
}

template <typename T>
T resolve(const char* name) {
    void* h = libcuda_handle();
    if (!h) return nullptr;
    return reinterpret_cast<T>(dlsym(h, name));
}

pf_cuGraphInstantiateWithFlags  real_instantiate_flags() {
    static pf_cuGraphInstantiateWithFlags p = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        // Try the symbol with version suffix first (CUDA 12 driver).
        p = resolve<pf_cuGraphInstantiateWithFlags>("cuGraphInstantiateWithFlags");
    });
    return p;
}
pf_cuGraphInstantiate           real_instantiate() {
    static pf_cuGraphInstantiate p = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        p = resolve<pf_cuGraphInstantiate>("cuGraphInstantiate_v2");
        if (!p) p = resolve<pf_cuGraphInstantiate>("cuGraphInstantiate");
    });
    return p;
}
pf_cuGraphGetNodes              real_getnodes() {
    static pf_cuGraphGetNodes p = nullptr;
    static std::once_flag once;
    std::call_once(once, [] { p = resolve<pf_cuGraphGetNodes>("cuGraphGetNodes"); });
    return p;
}
pf_cuGraphNodeGetType           real_nodetype() {
    static pf_cuGraphNodeGetType p = nullptr;
    static std::once_flag once;
    std::call_once(once, [] { p = resolve<pf_cuGraphNodeGetType>("cuGraphNodeGetType"); });
    return p;
}
pf_cuGraphKernelNodeGetParams   real_kernelparams() {
    static pf_cuGraphKernelNodeGetParams p = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        p = resolve<pf_cuGraphKernelNodeGetParams>("cuGraphKernelNodeGetParams_v2");
        if (!p) p = resolve<pf_cuGraphKernelNodeGetParams>("cuGraphKernelNodeGetParams");
    });
    return p;
}
pf_cuFuncGetName                real_funcname() {
    static pf_cuFuncGetName p = nullptr;
    static std::once_flag once;
    std::call_once(once, [] { p = resolve<pf_cuFuncGetName>("cuFuncGetName"); });
    return p;
}

void summarize_graph(CUgraph graph) {
    if (!inspect_enabled() || !graph) return;
    auto getnodes = real_getnodes();
    auto nodetype = real_nodetype();
    auto kparams  = real_kernelparams();
    auto fname    = real_funcname();
    if (!getnodes || !nodetype) return;

    size_t n = 0;
    if (getnodes(graph, nullptr, &n) != 0 || n == 0) return;
    std::vector<CUgraphNode> nodes(n);
    if (getnodes(graph, nodes.data(), &n) != 0) return;

    int kernel_count = 0;
    int memcpy_count = 0;
    int memset_count = 0;
    int other_count  = 0;

    // Aggregate by kernel name.
    std::unordered_map<std::string, int> by_name;
    for (auto node : nodes) {
        int t = -1;
        if (nodetype(node, &t) != 0) { other_count++; continue; }
        // CUgraphNodeType: 0=KERNEL, 1=MEMCPY, 2=MEMSET, 3=HOST, 4=GRAPH,
        // 5=EMPTY, 6=WAIT_EVENT, 7=EVENT_RECORD, etc.
        if (t == 0) {
            kernel_count++;
            if (kparams && fname) {
                CipherKernelNodeParams_v1 p{};
                if (kparams(node, &p) == 0 && p.func) {
                    const char* nm = nullptr;
                    fname(&nm, p.func);
                    if (nm && nm[0]) by_name[nm]++;
                    else             by_name["<unknown>"]++;
                }
            }
        } else if (t == 1) memcpy_count++;
        else if (t == 2) memset_count++;
        else other_count++;
    }

    static std::atomic<int> graph_seq{0};
    int gid = graph_seq.fetch_add(1);
    fprintf(stderr,
        "[CIPHER GRAPH-INSPECT #%d] total=%zu  kernels=%d  memcpy=%d  memset=%d  other=%d\n",
        gid, n, kernel_count, memcpy_count, memset_count, other_count);

    // Sort by count desc and print the top 30 unique kernels.
    std::vector<std::pair<std::string,int>> items(by_name.begin(), by_name.end());
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    int show = (int)std::min<size_t>(items.size(), 80);
    for (int i = 0; i < show; ++i) {
        // Truncate verbose mangled names to 120 chars.
        const std::string& nm = items[i].first;
        const char* tail = nm.c_str();
        size_t len = nm.size();
        if (len > 120) { tail = nm.c_str() + (len - 120); }
        fprintf(stderr, "[CIPHER GRAPH-INSPECT #%d]   %4d × %s%s\n",
                gid, items[i].second, len > 120 ? "…" : "", tail);
    }

    // Highlight: any Marlin kernels?
    int marlin_total = 0;
    int gemm_total   = 0;   // cublas / cutlass
    for (auto& kv : by_name) {
        if (kv.first.find("Marlin") != std::string::npos
         || kv.first.find("marlin") != std::string::npos) marlin_total += kv.second;
        if (kv.first.find("gemm")   != std::string::npos
         || kv.first.find("Gemm")   != std::string::npos
         || kv.first.find("cutlass")!= std::string::npos) gemm_total   += kv.second;
    }
    fprintf(stderr,
        "[CIPHER GRAPH-INSPECT #%d] Marlin nodes=%d  cuBLAS/cutlass GEMM nodes=%d\n",
        gid, marlin_total, gemm_total);
}

} // anon namespace

// ─────────────────────────────────────────────────────────────────────────────
// LD_PRELOAD overrides.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" __attribute__((visibility("default")))
CUresult cuGraphInstantiateWithFlags(CUgraphExec* exec, CUgraph graph,
                                     unsigned long long flags)
{
    auto real = real_instantiate_flags();
    if (!real) return (CUresult)3; // NOT_INITIALIZED
    CUresult r = real(exec, graph, flags);
    if (r == 0 && inspect_enabled()) summarize_graph(graph);
    return r;
}

extern "C" __attribute__((visibility("default")))
CUresult cuGraphInstantiate_v2(CUgraphExec* exec, CUgraph graph,
                               void* errorNode, void* logBuffer,
                               unsigned long long bufferSize)
{
    auto real = real_instantiate();
    if (!real) return (CUresult)3;
    CUresult r = real(exec, graph, errorNode, logBuffer, bufferSize);
    if (r == 0 && inspect_enabled()) summarize_graph(graph);
    return r;
}

// PyTorch's CUDAGraph::capture_end calls cudaGraphInstantiate (cudart).
// cudart resolves cuGraphInstantiate via cached fn pointer at first use, so
// the driver-level LD_PRELOAD overrides above don't catch that path. We
// intercept the cudart entry directly here. (Both shims are in place so we
// catch whichever path the user's stack happens to take.)
extern "C" __attribute__((visibility("default")))
int cudaGraphInstantiate(void* pGraphExec, void* graph,
                         void* pErrorNode, void* pLogBuffer,
                         size_t bufferSize)
{
    static pf_cudaGraphInstantiate real = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        // libcudart is in PyTorch's own RPATH; RTLD_NEXT skips us.
        real = reinterpret_cast<pf_cudaGraphInstantiate>(
            dlsym(RTLD_NEXT, "cudaGraphInstantiate"));
    });
    if (!real) return 1; // cudaErrorInvalidValue
    int r = real(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
    if (r == 0 && inspect_enabled()) summarize_graph(graph);
    return r;
}

extern "C" __attribute__((visibility("default")))
int cudaGraphInstantiateWithFlags(void* pGraphExec, void* graph,
                                  unsigned long long flags)
{
    static pf_cudaGraphInstantiateWithFlags real = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        real = reinterpret_cast<pf_cudaGraphInstantiateWithFlags>(
            dlsym(RTLD_NEXT, "cudaGraphInstantiateWithFlags"));
    });
    if (!real) return 1;
    int r = real(pGraphExec, graph, flags);
    if (r == 0 && inspect_enabled()) summarize_graph(graph);
    return r;
}
