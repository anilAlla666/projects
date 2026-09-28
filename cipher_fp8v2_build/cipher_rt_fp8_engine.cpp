/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_fp8_engine.cpp -- D.9 FP8 engine: cuBLASLt per-tensor scalar FP8.
 *
 * Per-tensor scalar FP8 E4M3 (the ONLY fused path on this cu13 cuBLASLt — rowwise
 * NOT_SUPPORTED, measured 2026-05-30). Static per-tensor weight prequant (one-time,
 * cached by w_ptr) + inline per-tensor activation quant + cuBLASLt FP8 scalar GEMM
 * (FAST_ACCUM) -> fp16/bf16 output DIRECTLY (no epilogue).
 *
 * Quant kernels are NVRTC-compiled at runtime + launched via driver-API
 * cuLaunchKernel (NOT statically compiled .cu). Static .cu kernels would register
 * through CIPHER's own __cudaRegisterFunction intercept at LD_PRELOAD time —
 * before the substrate is initialized — and SIGFPE. Driver-API cuModuleLoadData
 * sidesteps the runtime fatbin-registration path entirely (the Marlin pattern).
 * cuBLASLt + driver + nvrtc symbols resolved via dlsym from the in-process libs.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <dlfcn.h>
#include <cuda_runtime.h>

#include "cipher_rt_fp8.h"
#include "cipher_v2_internal.h"   /* cipher_log() macro */

/* cuBLASLt constants (dlsym path; no cublasLt.h dependency). */
enum {
    LT_COMPUTE_32F = 68, LT_R_32F = 0, LT_R_16F = 2, LT_R_16BF = 14, LT_R_8F_E4M3 = 28,
    LT_OP_N = 0, LT_OP_T = 1,
    LT_DESC_TRANSA = 3, LT_DESC_TRANSB = 4,
    LT_DESC_A_SCALE_PTR = 17, LT_DESC_B_SCALE_PTR = 18, LT_DESC_FAST_ACCUM = 25,
    LT_PREF_MAX_WS_BYTES = 1,
};

/* ---- NVRTC quant kernels (per-tensor absmax -> E4M3). compute_90 PTX. ---- */
static const char *kFp8KernelSrc = R"NVRTC(
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#define E4M3_MAX 448.0f
extern "C" __global__ void cipher_fp8_absmax_f16(const __half* in, int* amx, unsigned long long nel){
    __shared__ float sh[256]; float a=0.f;
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<nel;i+=(unsigned long long)gridDim.x*blockDim.x) a=fmaxf(a,fabsf(__half2float(in[i])));
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]=fmaxf(sh[threadIdx.x],sh[threadIdx.x+s]); __syncthreads(); }
    if(threadIdx.x==0) atomicMax(amx,__float_as_int(sh[0]));
}
extern "C" __global__ void cipher_fp8_absmax_bf16(const __nv_bfloat16* in, int* amx, unsigned long long nel){
    __shared__ float sh[256]; float a=0.f;
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<nel;i+=(unsigned long long)gridDim.x*blockDim.x) a=fmaxf(a,fabsf(__bfloat162float(in[i])));
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]=fmaxf(sh[threadIdx.x],sh[threadIdx.x+s]); __syncthreads(); }
    if(threadIdx.x==0) atomicMax(amx,__float_as_int(sh[0]));
}
extern "C" __global__ void cipher_fp8_quant_f16(const __half* in, __nv_fp8_e4m3* out, const int* amx, float* scale, unsigned long long nel){
    float am=__int_as_float(*amx); float sc=(am>0.f)?am/E4M3_MAX:1.f;
    if(blockIdx.x==0&&threadIdx.x==0)*scale=sc; float inv=1.f/sc;
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<nel;i+=(unsigned long long)gridDim.x*blockDim.x){
        float v=__half2float(in[i])*inv; v=fmaxf(-E4M3_MAX,fminf(E4M3_MAX,v)); out[i]=__nv_fp8_e4m3(v);
    }
}
extern "C" __global__ void cipher_fp8_quant_bf16(const __nv_bfloat16* in, __nv_fp8_e4m3* out, const int* amx, float* scale, unsigned long long nel){
    float am=__int_as_float(*amx); float sc=(am>0.f)?am/E4M3_MAX:1.f;
    if(blockIdx.x==0&&threadIdx.x==0)*scale=sc; float inv=1.f/sc;
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<nel;i+=(unsigned long long)gridDim.x*blockDim.x){
        float v=__bfloat162float(in[i])*inv; v=fmaxf(-E4M3_MAX,fminf(E4M3_MAX,v)); out[i]=__nv_fp8_e4m3(v);
    }
}
)NVRTC";

/* cuBLASLt fn types */
typedef int (*pf_create)(void **);
typedef int (*pf_descCreate)(void **, int, int);
typedef int (*pf_descDestroy)(void *);
typedef int (*pf_descSetAttr)(void *, int, const void *, size_t);
typedef int (*pf_layoutCreate)(void **, int, uint64_t, uint64_t, int64_t);
typedef int (*pf_layoutDestroy)(void *);
typedef int (*pf_prefCreate)(void **);
typedef int (*pf_prefDestroy)(void *);
typedef int (*pf_prefSetAttr)(void *, int, const void *, size_t);
typedef int (*pf_algoHeuristic)(void *, void *, void *, void *, void *, void *, void *, int, void *, int *);
typedef int (*pf_matmul)(void *, void *, const void *, const void *, void *, const void *, void *,
                         const void *, const void *, void *, void *, void *, const void *, void *, size_t, void *);
/* CUDA driver fn types */
typedef int (*pf_moduleLoadData)(void **, const void *);
typedef int (*pf_moduleGetFunction)(void **, void *, const char *);
typedef int (*pf_launchKernel)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, void *, void **, void **);
/* NVRTC fn types */
typedef int (*pf_nvrtcCreate)(void **, const char *, const char *, int, const char **, const char **);
typedef int (*pf_nvrtcCompile)(void *, int, const char **);
typedef int (*pf_nvrtcPTXSize)(void *, size_t *);
typedef int (*pf_nvrtcGetPTX)(void *, char *);
typedef int (*pf_nvrtcLogSize)(void *, size_t *);
typedef int (*pf_nvrtcGetLog)(void *, char *);
typedef int (*pf_nvrtcDestroy)(void **);

static struct {
    pf_create create; pf_matmul matmul;
    pf_descCreate descCreate; pf_descDestroy descDestroy; pf_descSetAttr descSetAttr;
    pf_layoutCreate layoutCreate; pf_layoutDestroy layoutDestroy;
    pf_prefCreate prefCreate; pf_prefDestroy prefDestroy; pf_prefSetAttr prefSetAttr;
    pf_algoHeuristic algoHeuristic;
} g_lt;
static struct { pf_moduleLoadData modLoad; pf_moduleGetFunction modGetFn; pf_launchKernel launch; } g_cu;
static struct { pf_nvrtcCreate create; pf_nvrtcCompile compile; pf_nvrtcPTXSize ptxSize;
                pf_nvrtcGetPTX getPTX; pf_nvrtcLogSize logSize; pf_nvrtcGetLog getLog; pf_nvrtcDestroy destroy; } g_nv;

static void *g_handle = nullptr;
static void *g_ws = nullptr;
static const size_t LT_WS = 32u * 1024 * 1024;
static std::atomic<int> g_inited{0};
/* compiled quant kernels */
static void *g_mod = nullptr;
static void *g_k_absmax_f16=nullptr,*g_k_absmax_bf16=nullptr,*g_k_quant_f16=nullptr,*g_k_quant_bf16=nullptr;

struct WeightSlot { int obs=0, ready=0, K=0, N=0, dtype=0; void *e4m3=nullptr; float *scale=nullptr; };
/* v2 FIX 1: cache keyed on (ptr, K, N), not pointer alone. The fwd/bwd
 * alternation presents the same weight pointer with swapped (K,N) every step;
 * pointer-only keying reset the slot each time => 159 weights re-prequantized
 * per step (the measured 06-10 MFU wall). Distinct orientations get distinct
 * slots; a quantized weight is quantized exactly once per (ptr,K,N). */
struct WKey {
    const void *p; int K, N;
    bool operator==(const WKey &o) const { return p == o.p && K == o.K && N == o.N; }
};
struct WKeyHash {
    size_t operator()(const WKey &w) const {
        return std::hash<const void *>()(w.p)
             ^ (std::hash<uint64_t>()(((uint64_t)(uint32_t)w.K << 32) | (uint32_t)w.N) << 1);
    }
};
static std::unordered_map<WKey, WeightSlot, WKeyHash> g_weights;
static std::mutex g_wmu;

/* cuBLASLt heuristic result, real layout (96 bytes) — needed to request >1
 * result; the as-built code aliased a single result through char[256]. */
struct LtHeurResult { uint64_t algo[8]; size_t workspaceSize; int state; float wavesCount; int reserved[4]; };

#define FP8_TUNE_CANDIDATES 8
#define FP8_TUNE_REPS       3

struct ShapeCache {
    bool ready=false; void *desc=nullptr,*la=nullptr,*lb=nullptr,*lc=nullptr;
    void *act_e4m3=nullptr; float *act_scale=nullptr; int *absmax=nullptr; size_t act_bytes=0;
    char algo[256];
    /* v2 FIX 3: autotune — heuristic candidates kept until the first call
     * times them on live buffers and copies the winner into algo. */
    bool tuned=false; int ncand=0;
    LtHeurResult cand[FP8_TUNE_CANDIDATES];
};
static std::unordered_map<uint64_t, ShapeCache> g_shapes;
static std::mutex g_smu;
static int *g_wq_absmax = nullptr;

/* Shared activation quant (opt-in CIPHER_FP8_SHARE_ACT=1). A transformer layer
 * feeds the SAME activation to consecutive GEMMs (q/k/v share the attn input,
 * gate/up share the MLP input). A single-entry (ptr,nel) cache quantizes once
 * and reuses for the immediately-following same-(ptr,nel) calls; any change of
 * ptr OR nel invalidates it (so cross-forward pointer reuse with new data or a
 * different activation re-quantizes). Safe for the forward access pattern;
 * default-OFF (per-call quant) stays the correctness-conservative path. */
static std::atomic<int> g_share_act{0};
struct ActCache { const void *ptr=nullptr; size_t nel=0; void *aq=nullptr; float *scale=nullptr; size_t cap=0; };
static ActCache g_act;
static std::mutex g_act_mu;

static void *resolve_in(const char *soname, const char *sym){
    void *h = dlopen(soname, RTLD_NOW|RTLD_NOLOAD); if(!h) h = dlopen(soname, RTLD_NOW);
    return h ? dlsym(h, sym) : nullptr;
}
static void *resolve_lt(const char *s){ void *p=resolve_in("libcublasLt.so.13",s); return p?p:resolve_in("libcublasLt.so.12",s); }
static void *resolve_cu(const char *s){ void *p=resolve_in("libcuda.so.1",s); return p?p:resolve_in("libcuda.so",s); }
static void *resolve_nv(const char *s){ void *p=resolve_in("libnvrtc.so.13",s); return p?p:resolve_in("libnvrtc.so.12",s); }

extern "C" int cipher_rt_fp8_engine_init(void){
    if(g_inited.load()) return (g_lt.create && g_cu.launch && g_nv.create) ? 0 : -1;
    g_lt.create=(pf_create)resolve_lt("cublasLtCreate"); g_lt.matmul=(pf_matmul)resolve_lt("cublasLtMatmul");
    g_lt.descCreate=(pf_descCreate)resolve_lt("cublasLtMatmulDescCreate"); g_lt.descDestroy=(pf_descDestroy)resolve_lt("cublasLtMatmulDescDestroy");
    g_lt.descSetAttr=(pf_descSetAttr)resolve_lt("cublasLtMatmulDescSetAttribute");
    g_lt.layoutCreate=(pf_layoutCreate)resolve_lt("cublasLtMatrixLayoutCreate"); g_lt.layoutDestroy=(pf_layoutDestroy)resolve_lt("cublasLtMatrixLayoutDestroy");
    g_lt.prefCreate=(pf_prefCreate)resolve_lt("cublasLtMatmulPreferenceCreate"); g_lt.prefDestroy=(pf_prefDestroy)resolve_lt("cublasLtMatmulPreferenceDestroy");
    g_lt.prefSetAttr=(pf_prefSetAttr)resolve_lt("cublasLtMatmulPreferenceSetAttribute");
    g_lt.algoHeuristic=(pf_algoHeuristic)resolve_lt("cublasLtMatmulAlgoGetHeuristic");
    g_cu.modLoad=(pf_moduleLoadData)resolve_cu("cuModuleLoadData"); g_cu.modGetFn=(pf_moduleGetFunction)resolve_cu("cuModuleGetFunction");
    g_cu.launch=(pf_launchKernel)resolve_cu("cuLaunchKernel");
    g_nv.create=(pf_nvrtcCreate)resolve_nv("nvrtcCreateProgram"); g_nv.compile=(pf_nvrtcCompile)resolve_nv("nvrtcCompileProgram");
    g_nv.ptxSize=(pf_nvrtcPTXSize)resolve_nv("nvrtcGetPTXSize"); g_nv.getPTX=(pf_nvrtcGetPTX)resolve_nv("nvrtcGetPTX");
    g_nv.logSize=(pf_nvrtcLogSize)resolve_nv("nvrtcGetProgramLogSize"); g_nv.getLog=(pf_nvrtcGetLog)resolve_nv("nvrtcGetProgramLog");
    g_nv.destroy=(pf_nvrtcDestroy)resolve_nv("nvrtcDestroyProgram");
    g_inited.store(1);
    if(!g_lt.create||!g_lt.matmul||!g_lt.algoHeuristic||!g_cu.launch||!g_cu.modLoad||!g_nv.create||!g_nv.compile){
        cipher_log("FP8: symbol resolution FAILED (lt=%p cu=%p nvrtc=%p) — engine disabled",
                   (void*)g_lt.create,(void*)g_cu.launch,(void*)g_nv.create);
        return -1;
    }
    g_share_act.store(getenv("CIPHER_FP8_SHARE_ACT") ? 1 : 0);
    cipher_log("FP8: engine init OK (cuBLASLt+driver+nvrtc resolved; CUDA lazy; share_act=%d)", g_share_act.load());
    return 0;
}

static int compile_kernels(void){
    void *prog=nullptr;
    if(g_nv.create(&prog, kFp8KernelSrc, "cipher_fp8.cu", 0, nullptr, nullptr) != 0) return -1;
    /* NVRTC needs the CUDA headers (cuda_fp16/bf16/fp8.h) on its include path.
     * Env override CIPHER_FP8_CUDA_INC; else a few common cu13 locations. */
    static char inc0[1024];
    const char *env = getenv("CIPHER_FP8_CUDA_INC");
    snprintf(inc0, sizeof(inc0), "-I%s", env ? env :
             "/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/include");
    const char *opts[] = {
        "--gpu-architecture=compute_90",
        inc0,
        "-I/usr/local/cuda/include",
        "-I/usr/include",
    };
    int crc = g_nv.compile(prog, 4, opts);
    if(crc != 0){
        size_t ls=0; g_nv.logSize(prog,&ls);
        if(ls>1){ char *log=(char*)malloc(ls); g_nv.getLog(prog,log); cipher_log("FP8: NVRTC compile FAILED: %s", log); free(log); }
        g_nv.destroy(&prog); return -1;
    }
    size_t psz=0; g_nv.ptxSize(prog,&psz);
    char *ptx=(char*)malloc(psz); g_nv.getPTX(prog,ptx); g_nv.destroy(&prog);
    int lrc = g_cu.modLoad(&g_mod, ptx); free(ptx);
    if(lrc != 0){ cipher_log("FP8: cuModuleLoadData failed rc=%d", lrc); return -1; }
    if(g_cu.modGetFn(&g_k_absmax_f16, g_mod, "cipher_fp8_absmax_f16") != 0 ||
       g_cu.modGetFn(&g_k_absmax_bf16,g_mod, "cipher_fp8_absmax_bf16")!= 0 ||
       g_cu.modGetFn(&g_k_quant_f16,  g_mod, "cipher_fp8_quant_f16")  != 0 ||
       g_cu.modGetFn(&g_k_quant_bf16, g_mod, "cipher_fp8_quant_bf16") != 0){
        cipher_log("FP8: cuModuleGetFunction failed"); return -1;
    }
    return 0;
}

static std::atomic<int> g_cuda_ready{0};
static std::mutex g_cuda_mu;
static int ensure_cuda(void){
    if(g_cuda_ready.load()) return g_handle ? 0 : -1;
    std::lock_guard<std::mutex> lk(g_cuda_mu);
    if(g_cuda_ready.load()) return g_handle ? 0 : -1;
    if(!g_lt.create){ g_cuda_ready.store(1); return -1; }
    if(g_lt.create(&g_handle)!=0 || !g_handle){ cipher_log("FP8: cublasLtCreate failed"); g_handle=nullptr; g_cuda_ready.store(1); return -1; }
    if(cudaMalloc(&g_ws, LT_WS)!=cudaSuccess){ cipher_log("FP8: workspace malloc failed"); g_handle=nullptr; g_cuda_ready.store(1); return -1; }
    if(cudaMalloc(&g_wq_absmax, sizeof(int))!=cudaSuccess){ g_handle=nullptr; g_cuda_ready.store(1); return -1; }
    if(compile_kernels()!=0){ g_handle=nullptr; g_cuda_ready.store(1); return -1; }
    cipher_log("FP8: CUDA resources ready (handle=%p, 32MB ws, NVRTC quant kernels loaded)", g_handle);
    g_cuda_ready.store(1);
    return 0;
}

/* per-tensor quant: absmax (atomicMax) then quantize to E4M3 + write scalar scale. */
static int fp8_quant_tensor(const void *in, int dtype, void *out_e4m3, float *scale_dev,
                            int *absmax, size_t nel, void *stream){
    if(cudaMemsetAsync(absmax, 0, sizeof(int), (cudaStream_t)stream) != cudaSuccess) return -1;
    unsigned long long n = (unsigned long long)nel;
    unsigned blk = 256;
    unsigned grid = (unsigned)((nel + blk - 1) / blk); if(grid > 4096) grid = 4096; if(grid < 1) grid = 1;
    void *kabs = (dtype == LT_R_16BF) ? g_k_absmax_bf16 : g_k_absmax_f16;
    void *kqnt = (dtype == LT_R_16BF) ? g_k_quant_bf16  : g_k_quant_f16;
    void *p_in=(void*)&in, *p_amx=(void*)&absmax, *p_n=(void*)&n;
    void *abs_args[] = { p_in, p_amx, p_n };
    if(g_cu.launch(kabs, grid,1,1, blk,1,1, 0, stream, abs_args, nullptr) != 0) return -1;
    void *p_out=(void*)&out_e4m3, *p_sc=(void*)&scale_dev;
    void *qnt_args[] = { p_in, p_out, p_amx, p_sc, p_n };
    if(g_cu.launch(kqnt, grid,1,1, blk,1,1, 0, stream, qnt_args, nullptr) != 0) return -1;
    return 0;
}

extern "C" int cipher_rt_fp8_engine_observe_weight(const void *w_ptr, int K, int N){
    std::lock_guard<std::mutex> lk(g_wmu);
    WeightSlot &s = g_weights[WKey{w_ptr, K, N}];   /* v2: orientation swap = different slot, no reset */
    s.K = K; s.N = N; s.obs++;
    return s.obs;
}

extern "C" int cipher_rt_fp8_engine_is_ready(const void *w_ptr, int K, int N){
    std::lock_guard<std::mutex> lk(g_wmu);
    auto it = g_weights.find(WKey{w_ptr, K, N});
    return (it != g_weights.end() && it->second.ready) ? 1 : 0;
}

extern "C" int cipher_rt_fp8_engine_quantize_weight(const void *w_fp, int dtype, int K, int N, void *stream){
    if(ensure_cuda() != 0) return -1;
    size_t nel = (size_t)N * (size_t)K;
    void *e4m3=nullptr; float *scale=nullptr;
    if(cudaMalloc(&e4m3, nel)!=cudaSuccess) return -1;
    if(cudaMalloc(&scale, sizeof(float))!=cudaSuccess){ cudaFree(e4m3); return -1; }
    if(fp8_quant_tensor(w_fp, dtype, e4m3, scale, g_wq_absmax, nel, stream) != 0){ cudaFree(e4m3); cudaFree(scale); return -1; }
    cudaStreamSynchronize((cudaStream_t)stream);
    std::lock_guard<std::mutex> lk(g_wmu);
    WeightSlot &s = g_weights[WKey{w_fp, K, N}];
    if(s.e4m3) cudaFree(s.e4m3); if(s.scale) cudaFree(s.scale);
    s.e4m3=e4m3; s.scale=scale; s.ready=1; s.K=K; s.N=N; s.dtype=dtype;
    return 0;
}

/* v2 FIX 4: non-overlapping bit packing (the as-built 34/17/3 XOR pack let
 * k (up to 32000 -> bit 17) overlap n's low bit — latent collision hazard). */
static inline uint64_t pack_key(int m,int n,int k,int dt){
    return ((uint64_t)(uint32_t)m<<42) | ((uint64_t)(uint32_t)n<<21) | ((uint64_t)(uint32_t)k<<3) | (uint64_t)(dt&7);
}

extern "C" int cipher_rt_fp8_engine_matmul(const void *w_key, const void *activation, void *c_out,
                                           int dtype, int m, int n, int k, void *stream){
    if(ensure_cuda() != 0) return -1;
    void *w_e4m3=nullptr; float *w_scale=nullptr;
    {
        std::lock_guard<std::mutex> lk(g_wmu);
        auto it = g_weights.find(WKey{w_key, k, m});   /* v2: composite key (K=in=k, N=out=m) */
        if(it == g_weights.end() || !it->second.ready) return -1;
        w_e4m3 = it->second.e4m3; w_scale = it->second.scale;
    }
    int out_type = (dtype == LT_R_16BF) ? LT_R_16BF : LT_R_16F;
    uint64_t key = pack_key(m,n,k,out_type);
    ShapeCache *sc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_smu);
        sc = &g_shapes[key];
        if(!sc->ready){
            if(g_lt.descCreate(&sc->desc, LT_COMPUTE_32F, LT_R_32F) != 0) return -1;
            int ta=LT_OP_T, tb=LT_OP_N; int8_t fa=1;
            g_lt.descSetAttr(sc->desc, LT_DESC_TRANSA, &ta, sizeof(ta));
            g_lt.descSetAttr(sc->desc, LT_DESC_TRANSB, &tb, sizeof(tb));
            g_lt.descSetAttr(sc->desc, LT_DESC_FAST_ACCUM, &fa, sizeof(fa));
            int rcA=g_lt.layoutCreate(&sc->la, LT_R_8F_E4M3,(uint64_t)k,(uint64_t)m,(int64_t)k);
            int rcB=g_lt.layoutCreate(&sc->lb, LT_R_8F_E4M3,(uint64_t)k,(uint64_t)n,(int64_t)k);
            int rcC=g_lt.layoutCreate(&sc->lc, out_type,    (uint64_t)m,(uint64_t)n,(int64_t)m);
            if(rcA||rcB||rcC){ g_lt.descDestroy(sc->desc); *sc=ShapeCache{}; return -1; }
            sc->act_bytes = (size_t)n*(size_t)k;
            if(cudaMalloc(&sc->act_e4m3, sc->act_bytes)!=cudaSuccess || cudaMalloc(&sc->act_scale,sizeof(float))!=cudaSuccess || cudaMalloc(&sc->absmax,sizeof(int))!=cudaSuccess) return -1;
            void *pref=nullptr;
            if(g_lt.prefCreate(&pref)!=0||!pref) return -1;
            size_t ws=LT_WS; g_lt.prefSetAttr(pref, LT_PREF_MAX_WS_BYTES, &ws, sizeof(ws));
            /* v2 FIX 3: request up to FP8_TUNE_CANDIDATES heuristic results
             * (as-built took only the first). cand[0] is the provisional algo
             * until first use times them all on the live buffers. */
            int got=0;
            int hrc = g_lt.algoHeuristic(g_handle, sc->desc, sc->la, sc->lb, sc->lc, sc->lc, pref,
                                         FP8_TUNE_CANDIDATES, sc->cand, &got);
            g_lt.prefDestroy(pref);
            if(hrc!=0 || got==0){ cipher_log("FP8: heuristic rc=%d got=%d (m=%d n=%d k=%d) — FP8 unsupported for shape", hrc,got,m,n,k); return -1; }
            sc->ncand = got;
            memcpy(sc->algo, &sc->cand[0], sizeof(LtHeurResult));
            sc->ready = true;
        }
    }
    /* quantize the activation (per-call) or reuse the shared single-entry cache. */
    void *act_buf = sc->act_e4m3; float *act_scale = sc->act_scale;
    if(g_share_act.load()){
        std::lock_guard<std::mutex> lk(g_act_mu);
        size_t nel = (size_t)n*(size_t)k;
        if(!(g_act.ptr==activation && g_act.nel==nel && g_act.aq)){
            if(g_act.cap < nel){ if(g_act.aq) cudaFree(g_act.aq); if(cudaMalloc(&g_act.aq,nel)!=cudaSuccess){ g_act.cap=0; g_act.aq=nullptr; return -1; } g_act.cap=nel; }
            if(!g_act.scale && cudaMalloc(&g_act.scale,sizeof(float))!=cudaSuccess) return -1;
            if(fp8_quant_tensor(activation, dtype, g_act.aq, g_act.scale, sc->absmax, nel, stream) != 0) return -1;
            g_act.ptr=activation; g_act.nel=nel;
        }
        act_buf=g_act.aq; act_scale=g_act.scale;
    } else {
        if(fp8_quant_tensor(activation, dtype, sc->act_e4m3, sc->act_scale, sc->absmax, sc->act_bytes, stream) != 0) return -1;
    }
    g_lt.descSetAttr(sc->desc, LT_DESC_A_SCALE_PTR, &w_scale, sizeof(void*));
    g_lt.descSetAttr(sc->desc, LT_DESC_B_SCALE_PTR, &act_scale, sizeof(void*));
    float alpha=1.f, beta=0.f;

    /* v2 FIX 3: first-use autotune. Time each heuristic candidate on the live
     * quantized buffers (1 warm + FP8_TUNE_REPS timed reps, cudaEvents on the
     * caller's stream), keep the fastest. Every rep computes the same valid
     * product into c_out; the final dispatch below overwrites it with the
     * winner. Happens once per shape, inside the harness warmup window. */
    if(!sc->tuned){
        std::lock_guard<std::mutex> lk(g_smu);
        if(!sc->tuned){
            int best = 0; float best_ms = 0.f; bool have_best = false;
            cudaEvent_t ev0, ev1;
            if(cudaEventCreate(&ev0)==cudaSuccess && cudaEventCreate(&ev1)==cudaSuccess){
                for(int ci=0; ci<sc->ncand; ci++){
                    if(g_lt.matmul(g_handle, sc->desc, &alpha, w_e4m3, sc->la, act_buf, sc->lb, &beta,
                                   c_out, sc->lc, c_out, sc->lc, &sc->cand[ci], g_ws, LT_WS, stream) != 0)
                        continue;   /* candidate fails at runtime: skip */
                    cudaEventRecord(ev0, (cudaStream_t)stream);
                    int ok = 1;
                    for(int r=0; r<FP8_TUNE_REPS; r++)
                        if(g_lt.matmul(g_handle, sc->desc, &alpha, w_e4m3, sc->la, act_buf, sc->lb, &beta,
                                       c_out, sc->lc, c_out, sc->lc, &sc->cand[ci], g_ws, LT_WS, stream) != 0){ ok=0; break; }
                    cudaEventRecord(ev1, (cudaStream_t)stream);
                    if(!ok || cudaEventSynchronize(ev1)!=cudaSuccess) continue;
                    float ms=0.f; cudaEventElapsedTime(&ms, ev0, ev1);
                    if(!have_best || ms < best_ms){ best = ci; best_ms = ms; have_best = true; }
                }
                cudaEventDestroy(ev0); cudaEventDestroy(ev1);
            }
            if(have_best){
                memcpy(sc->algo, &sc->cand[best], sizeof(LtHeurResult));
                cipher_log("FP8: autotune m=%d n=%d k=%d — algo %d/%d wins (%.3f ms / %d reps)",
                           m, n, k, best, sc->ncand, best_ms, FP8_TUNE_REPS);
            } else {
                cipher_log("FP8: autotune m=%d n=%d k=%d — no candidate timed, keeping heuristic-first", m, n, k);
            }
            sc->tuned = true;
        }
    }

    int rc = g_lt.matmul(g_handle, sc->desc, &alpha, w_e4m3, sc->la, act_buf, sc->lb, &beta,
                         c_out, sc->lc, c_out, sc->lc, sc->algo, g_ws, LT_WS, stream);
    return rc==0 ? 0 : -1;
}

extern "C" unsigned long cipher_rt_fp8_engine_weights_count(void){
    std::lock_guard<std::mutex> lk(g_wmu);
    unsigned long c=0; for(auto &kv:g_weights) if(kv.second.ready) c++;
    return c;
}
