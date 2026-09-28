// =============================================================================
// CIPHER — Change 4 Part B: NCCL Tuner Plugin ABI
// include/cipher_nccl_tuner_abi.h
//
// Minimal subset of NCCL's nccl_tuner.h sufficient to export the tuner-plugin
// ABI without taking a build-time dependency on NCCL headers being installed.
// Values and struct layouts mirror NCCL 2.21+ (v2) and NCCL 2.17–2.20 (v1).
//
// Reference: nccl/src/include/nccl_tuner.h in the NCCL source tree.
// =============================================================================

#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// NCCL return codes / enums — only what the plugin needs
// ---------------------------------------------------------------------------

typedef enum {
    ncclSuccess                 = 0,
    ncclUnhandledCudaError      = 1,
    ncclSystemError             = 2,
    ncclInternalError           = 3,
    ncclInvalidArgument         = 4,
    ncclInvalidUsage            = 5,
    ncclRemoteError             = 6,
    ncclInProgress              = 7,
    ncclNumResults              = 8
} ncclResult_t;

// Collective type enum — matches ncclFunc_t in NCCL headers
typedef enum {
    ncclFuncBroadcast    = 0,
    ncclFuncReduce       = 1,
    ncclFuncAllGather    = 2,
    ncclFuncReduceScatter = 3,
    ncclFuncAllReduce    = 4,
    ncclFuncSendRecv     = 5,
    ncclFuncSend         = 6,
    ncclFuncRecv         = 7,
    ncclNumFuncs         = 8
} ncclFunc_t;

// Algorithm enum (NCCL internal) — matches nccl/src/include/devcomm.h
//   NCCL_ALGO_UNDEF = -1
//   NCCL_ALGO_TREE            = 0
//   NCCL_ALGO_RING            = 1
//   NCCL_ALGO_COLLNET_DIRECT  = 2
//   NCCL_ALGO_COLLNET_CHAIN   = 3
//   NCCL_ALGO_NVLS            = 4
//   NCCL_ALGO_NVLS_TREE       = 5
#define NCCL_ALGO_UNDEF           (-1)
#define NCCL_ALGO_TREE            0
#define NCCL_ALGO_RING            1
#define NCCL_ALGO_COLLNET_DIRECT  2
#define NCCL_ALGO_COLLNET_CHAIN   3
#define NCCL_ALGO_NVLS            4
#define NCCL_ALGO_NVLS_TREE       5

// Protocol enum
//   NCCL_PROTO_UNDEF  = -1
//   NCCL_PROTO_LL     = 0
//   NCCL_PROTO_LL128  = 1
//   NCCL_PROTO_SIMPLE = 2
#define NCCL_PROTO_UNDEF   (-1)
#define NCCL_PROTO_LL      0
#define NCCL_PROTO_LL128   1
#define NCCL_PROTO_SIMPLE  2

// Debug logger callback type — NCCL passes its own logger in Init
typedef void (*ncclDebugLogger_t)(const char* filefunc, int line,
                                   int type, const char* fmt, ...);

// ---------------------------------------------------------------------------
// NCCL tuner plugin ABI — v2 (NCCL 2.21+)
//
// Struct layout MUST match NCCL's nccl_tuner.h exactly. NCCL dlopens this .so
// and looks up the symbol "ncclTunerPlugin_v2". If missing, it falls back to
// "ncclTunerPlugin_v1" (defined below).
// ---------------------------------------------------------------------------

typedef struct {
    const char* name;

    ncclResult_t (*init)(size_t nRanks, size_t nNodes,
                          ncclDebugLogger_t logFunction,
                          void** context);

    ncclResult_t (*getCollInfo)(
        void*            context,
        ncclFunc_t       collType,
        size_t           nBytes,
        int              collNetSupport,
        int              nvlsSupport,
        int              numPipeOps,
        int*             algorithm,    // OUT
        int*             protocol,     // OUT
        int*             nChannels);   // OUT

    ncclResult_t (*destroy)(void* context);
} ncclTuner_v2_t;

// ---------------------------------------------------------------------------
// NCCL tuner plugin ABI — v1 (NCCL 2.17–2.20)
//
// Identical to v2 except getCollInfo takes fewer arguments: no collNetSupport,
// nvlsSupport, or numPipeOps. All collectives get the same fallback path.
// ---------------------------------------------------------------------------

typedef struct {
    const char* name;

    ncclResult_t (*init)(size_t nRanks, size_t nNodes,
                          ncclDebugLogger_t logFunction,
                          void** context);

    ncclResult_t (*getCollInfo)(
        void*            context,
        ncclFunc_t       collType,
        size_t           nBytes,
        int              numPipeOps,
        int*             algorithm,    // OUT
        int*             protocol,     // OUT
        int*             nChannels);   // OUT

    ncclResult_t (*destroy)(void* context);
} ncclTuner_v1_t;

#ifdef __cplusplus
}
#endif
