// =============================================================================
// CIPHER — Change 4 Part B: NCCL Tuner Plugin
// src/cipher_nccl_tuner.cpp
//
// Standalone DSO loaded by NCCL via NCCL_TUNER_PLUGIN=./libcipher_nccl_tuner.so.
// NCCL looks up the exported symbol ncclTunerPlugin_v2 (or v1 as fallback) and
// calls into it on every collective. This plugin:
//
//   1. At Init: caches nRanks/nNodes, resolves the CIPHER RT bridge symbol
//      cipher_nccl_record_decide via dlsym(RTLD_DEFAULT). The bridge lives in
//      libcipher_rt.so which is LD_PRELOAD'd by the user before launching the
//      NCCL-using process, so the symbol is already in the address space.
//   2. On GetCollInfo: calls cipher_nccl_record_decide(nBytes, nRanks) to get
//      the CfC-recommended CipherNcclAlgo, maps it to NCCL's internal
//      (algorithm, protocol, nChannels) tuple with hardware-capability-aware
//      fallbacks (nvlsSupport / collNetSupport).
//   3. On Destroy: logs a summary and frees the context.
//
// Safety: if the RT bridge is not resolvable the plugin silently returns
// NCCL_ALGO_UNDEF / NCCL_PROTO_UNDEF for every call, which tells NCCL to use
// its built-in tuning — i.e. the plugin is a no-op passthrough.
//
// Thread safety: GetCollInfo may be called from NCCL worker threads; the RT
// bridge takes its own mutex (cipher_nccl.cpp::g_nccl_neural_mtx).
//
// Drift rule ✓: plugin sees only (nBytes, nRanks, nNodes, collType). No
// model knowledge, no kernel names, no tensor content.
// =============================================================================

#include "cipher_nccl_tuner_abi.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------------
// RT bridge — resolved at Init via dlsym(RTLD_DEFAULT).
// Signature matches cipher_nccl.cpp::cipher_nccl_record_decide.
// -----------------------------------------------------------------------------

typedef int (*cipher_nccl_decide_fn)(uint64_t bytes, uint32_t num_ranks);
// Stage 9: v4 decide — fills out an algo override pointer.
//   cipher_nccl_v4_decide(bytes, num_ranks, &out_algo) -> 1 if overrode, 0 else
typedef int (*cipher_nccl_v4_decide_fn)(uint64_t bytes, uint32_t num_ranks, int* out_algo);
typedef int (*cipher_nccl_v4_enabled_fn)(void);

// CipherNcclAlgo enum values (from include/cipher_nccl_bpf.h) — duplicated
// here so the plugin has zero build-time CIPHER header dependencies.
//   0 = AUTO, 1 = RING, 2 = TREE, 3 = NVLS, 4 = LL128
#define CIPHER_ALGO_AUTO   0
#define CIPHER_ALGO_RING   1
#define CIPHER_ALGO_TREE   2
#define CIPHER_ALGO_NVLS   3
#define CIPHER_ALGO_LL128  4

// -----------------------------------------------------------------------------
// Plugin context
// -----------------------------------------------------------------------------

typedef struct {
    size_t                    nRanks;
    size_t                    nNodes;
    ncclDebugLogger_t         logger;
    cipher_nccl_decide_fn     decide_fn;
    cipher_nccl_v4_decide_fn  v4_decide_fn;
    cipher_nccl_v4_enabled_fn v4_enabled_fn;
    uint64_t                  call_count;
    uint64_t                  v4_overrides;
} CipherTunerCtx;

// -----------------------------------------------------------------------------
// CipherNcclAlgo → (NCCL algo, proto) mapping with hardware fallback.
// -----------------------------------------------------------------------------

static void map_cipher_to_nccl(
    int  cipher_algo,
    int  nvlsSupport,
    int  collNetSupport,
    int* out_algo,
    int* out_proto)
{
    (void)collNetSupport;  // currently only guards NVLS → Ring fallback

    switch (cipher_algo) {
        case CIPHER_ALGO_RING:
            *out_algo  = NCCL_ALGO_RING;
            *out_proto = NCCL_PROTO_SIMPLE;
            return;
        case CIPHER_ALGO_TREE:
            *out_algo  = NCCL_ALGO_TREE;
            *out_proto = NCCL_PROTO_SIMPLE;
            return;
        case CIPHER_ALGO_NVLS:
            // NVLS requires NVSwitch (nvlsSupport) AND collnet topology on
            // some configurations. Both guards: fall back to RING when
            // either is unavailable.
            if (nvlsSupport == 0 || collNetSupport == 0) {
                *out_algo  = NCCL_ALGO_RING;
                *out_proto = NCCL_PROTO_SIMPLE;
            } else {
                *out_algo  = NCCL_ALGO_NVLS;
                *out_proto = NCCL_PROTO_SIMPLE;
            }
            return;
        case CIPHER_ALGO_LL128:
            // NCCL's LL128 is a protocol, not an algorithm; it runs over Ring.
            *out_algo  = NCCL_ALGO_RING;
            *out_proto = NCCL_PROTO_LL128;
            return;
        case CIPHER_ALGO_AUTO:
        default:
            // Tell NCCL to use its built-in tuner.
            *out_algo  = NCCL_ALGO_UNDEF;
            *out_proto = NCCL_PROTO_UNDEF;
            return;
    }
}

// nChannels heuristic — matches the nchannels that the CfC's CipherNcclPolicy
// is seeded to return. Kept separate from the mapping function so we can tune
// it without touching the algorithm logic.
static int nchannels_for(int cipher_algo) {
    switch (cipher_algo) {
        case CIPHER_ALGO_LL128: return 2;
        case CIPHER_ALGO_TREE:  return 4;
        case CIPHER_ALGO_NVLS:  return 4;
        case CIPHER_ALGO_RING:  return 8;
        default:                return 0;  // let NCCL pick
    }
}

// -----------------------------------------------------------------------------
// Plugin methods
// -----------------------------------------------------------------------------

static ncclResult_t cipher_tuner_init(size_t nRanks, size_t nNodes,
                                       ncclDebugLogger_t logger,
                                       void** context)
{
    CipherTunerCtx* ctx = (CipherTunerCtx*)calloc(1, sizeof(CipherTunerCtx));
    if (!ctx) return ncclSystemError;
    ctx->nRanks = nRanks;
    ctx->nNodes = nNodes;
    ctx->logger = logger;
    ctx->decide_fn = (cipher_nccl_decide_fn)dlsym(
        RTLD_DEFAULT, "cipher_nccl_record_decide");
    ctx->v4_decide_fn  = (cipher_nccl_v4_decide_fn)dlsym(
        RTLD_DEFAULT, "cipher_nccl_v4_decide");
    ctx->v4_enabled_fn = (cipher_nccl_v4_enabled_fn)dlsym(
        RTLD_DEFAULT, "cipher_nccl_v4_enabled");

    // If RT bridges aren't found, libcipher_rt.so wasn't loaded yet (torch
    // can call NCCL init before our LD_PRELOAD'd hook constructor pulls in
    // the RT). Try explicit dlopen via CIPHER_RT_PATH or default search.
    if (!ctx->v4_decide_fn) {
        const char* rt_path = getenv("CIPHER_RT_PATH");
        const char* candidates[] = {
            rt_path,
            "libcipher_rt.so",
            "/opt/cipher/lib/libcipher_rt.so",
            nullptr,
        };
        for (int i = 0; candidates[i]; i++) {
            if (!candidates[i] || !*candidates[i]) continue;
            void* rt_h = dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
            if (rt_h) {
                fprintf(stderr,
                    "[CIPHER TUNER] explicit dlopen(%s) succeeded; "
                    "re-resolving RT bridges\n", candidates[i]);
                ctx->decide_fn = (cipher_nccl_decide_fn)dlsym(
                    RTLD_DEFAULT, "cipher_nccl_record_decide");
                ctx->v4_decide_fn = (cipher_nccl_v4_decide_fn)dlsym(
                    RTLD_DEFAULT, "cipher_nccl_v4_decide");
                ctx->v4_enabled_fn = (cipher_nccl_v4_enabled_fn)dlsym(
                    RTLD_DEFAULT, "cipher_nccl_v4_enabled");
                break;
            }
        }
    }

    fprintf(stderr,
        "[CIPHER TUNER] init: nRanks=%zu nNodes=%zu decide_fn=%p v4_decide=%p (%s)\n",
        nRanks, nNodes, (void*)ctx->decide_fn, (void*)ctx->v4_decide_fn,
        ctx->decide_fn ? "RT bridge resolved"
                        : "RT bridge NOT available — plugin is passthrough");

    *context = ctx;
    return ncclSuccess;
}

static void decide_and_map(
    CipherTunerCtx* ctx,
    ncclFunc_t      collType,
    size_t          nBytes,
    int             collNetSupport,
    int             nvlsSupport,
    int*            algorithm,
    int*            protocol,
    int*            nChannels)
{
    (void)collType;

    // Topology guard: on small single-node fabrics (≤4 GPUs, no NVSwitch
    // /NVLS), CIPHER's algorithm bias picks RING/LL128/4ch — measured 4×
    // SLOWER than NCCL's auto-tuner on a 2× H100 NV18 box.  CIPHER's bias
    // was tuned for 8+ GPU NVSwitch + NVLS systems where NCCL's defaults
    // are suboptimal.  Bail to passthrough for small fabrics.
    if (ctx && ctx->nRanks > 0 && ctx->nRanks <= 4
            && ctx->nNodes <= 1) {
        if (algorithm) *algorithm = NCCL_ALGO_UNDEF;
        if (protocol)  *protocol  = NCCL_PROTO_UNDEF;
        if (nChannels) *nChannels = 0;
        if (ctx) ctx->call_count++;
        static uint64_t s_pt_log = 0;
        s_pt_log++;
        if (s_pt_log <= 3 || (s_pt_log % 500) == 0) {
            fprintf(stderr,
                "[CIPHER TUNER] passthrough: nRanks=%zu nNodes=%zu — "
                "small-fabric guard #%llu\n",
                ctx->nRanks, ctx->nNodes,
                (unsigned long long)s_pt_log);
        }
        return;
    }

    int cipher_algo = CIPHER_ALGO_AUTO;
    if (ctx && ctx->decide_fn) {
        cipher_algo = ctx->decide_fn((uint64_t)nBytes,
                                      (uint32_t)(ctx->nRanks
                                                 ? ctx->nRanks : 8));
    }
    // Stage 9 v4 override: ask cipher_nccl_v4 for a bias when enabled.
    if (ctx && ctx->v4_enabled_fn && ctx->v4_enabled_fn()
        && ctx->v4_decide_fn) {
        int v4_algo = CIPHER_ALGO_AUTO;
        // v4 uses NCCL_ALGO_* values (1=TREE, 2=RING, 5=NVLS); map back to
        // our CIPHER_ALGO_* enum.
        if (ctx->v4_decide_fn((uint64_t)nBytes,
                              (uint32_t)(ctx->nRanks ? ctx->nRanks : 8),
                              &v4_algo)) {
            switch (v4_algo) {
                case 1: cipher_algo = CIPHER_ALGO_TREE; break;
                case 2: cipher_algo = CIPHER_ALGO_RING; break;
                case 5: cipher_algo = CIPHER_ALGO_NVLS; break;
                default: break;
            }
            ctx->v4_overrides++;
        }
    }

    int nccl_algo  = NCCL_ALGO_UNDEF;
    int nccl_proto = NCCL_PROTO_UNDEF;
    map_cipher_to_nccl(cipher_algo, nvlsSupport, collNetSupport,
                       &nccl_algo, &nccl_proto);

    int n_ch = nchannels_for(cipher_algo);
    if (n_ch <= 0) n_ch = 0;                 // 0 means NCCL picks

    if (algorithm) *algorithm = nccl_algo;
    if (protocol)  *protocol  = nccl_proto;
    if (nChannels) *nChannels = n_ch;

    if (ctx) ctx->call_count++;

    static uint64_t s_log = 0;
    s_log++;
    if (s_log <= 10 || (s_log % 500) == 0) {
        fprintf(stderr,
            "[CIPHER TUNER] GetCollInfo bytes=%zu nvls=%d collnet=%d "
            "→ nccl_algo=%d nccl_proto=%d nChannels=%d cipher=%d (#%llu)\n",
            nBytes, nvlsSupport, collNetSupport,
            nccl_algo, nccl_proto, n_ch, cipher_algo,
            (unsigned long long)s_log);
    }
}

static ncclResult_t cipher_tuner_get_coll_info_v2(
    void*       context,
    ncclFunc_t  collType,
    size_t      nBytes,
    int         collNetSupport,
    int         nvlsSupport,
    int         numPipeOps,
    int*        algorithm,
    int*        protocol,
    int*        nChannels)
{
    (void)numPipeOps;
    decide_and_map((CipherTunerCtx*)context, collType, nBytes,
                   collNetSupport, nvlsSupport,
                   algorithm, protocol, nChannels);
    return ncclSuccess;
}

static ncclResult_t cipher_tuner_get_coll_info_v1(
    void*       context,
    ncclFunc_t  collType,
    size_t      nBytes,
    int         numPipeOps,
    int*        algorithm,
    int*        protocol,
    int*        nChannels)
{
    (void)numPipeOps;
    // v1 has no hardware-capability hints; assume NVLS & CollNet are
    // available and let the mapping fall through to NCCL's own checks.
    decide_and_map((CipherTunerCtx*)context, collType, nBytes,
                   /*collNetSupport=*/1, /*nvlsSupport=*/1,
                   algorithm, protocol, nChannels);
    return ncclSuccess;
}

static ncclResult_t cipher_tuner_destroy(void* context) {
    CipherTunerCtx* ctx = (CipherTunerCtx*)context;
    if (ctx) {
        fprintf(stderr,
            "[CIPHER TUNER] destroy: total GetCollInfo calls = %llu\n",
            (unsigned long long)ctx->call_count);
        free(ctx);
    }
    return ncclSuccess;
}

// -----------------------------------------------------------------------------
// Exported plugin structs
//   ncclTunerPlugin_v2 — NCCL 2.21+
//   ncclTunerPlugin_v1 — NCCL 2.17–2.20 (fallback)
// -----------------------------------------------------------------------------

extern "C" __attribute__((visibility("default")))
ncclTuner_v2_t ncclTunerPlugin_v2 = {
    /* name        */ "cipher-nccl-tuner-v2",
    /* init        */ cipher_tuner_init,
    /* getCollInfo */ cipher_tuner_get_coll_info_v2,
    /* destroy     */ cipher_tuner_destroy,
};

extern "C" __attribute__((visibility("default")))
ncclTuner_v1_t ncclTunerPlugin_v1 = {
    /* name        */ "cipher-nccl-tuner-v1",
    /* init        */ cipher_tuner_init,
    /* getCollInfo */ cipher_tuner_get_coll_info_v1,
    /* destroy     */ cipher_tuner_destroy,
};
