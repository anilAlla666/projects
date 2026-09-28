// =============================================================================
// CIPHER — CPU Compilation Stubs
// cipher_stubs.h
//
// Minimal CUDA/CUPTI type and function stubs for compiling CIPHER's CPU-only
// logic (classification, oracle, recipes, structural lookup) without a CUDA SDK.
//
// Usage: g++ -I./include -I./include/stubs ... (stubs are found via include path)
// For real GPU builds: use cmake which links against the real CUDA SDK.
// =============================================================================
#pragma once

// If real CUDA headers are already included, skip stubs entirely
#if defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__) || defined(CUDA_VERSION) || defined(__CUDA_RUNTIME_H__)
// Real CUDA SDK is present — do not define stub types
#else

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <atomic>

// ── Core CUDA types ──────────────────────────────────────────────────────────
typedef int            CUresult;
typedef unsigned int   CUdevice;
typedef void*          CUcontext;
typedef void*          CUfunction;
typedef void*          CUstream;
typedef void*          CUgreenCtx;
typedef void*          cudaStream_t;
typedef int            cudaError_t;
typedef struct { int smCount; } CUdevSmResource;
typedef struct { int type; CUdevSmResource sm; } CUdevResourceDesc;
typedef int            CUdriverProcAddressQueryResult;

// ── CUDA error codes ─────────────────────────────────────────────────────────
#define CUDA_SUCCESS                  0
#define CUDA_ERROR_NOT_FOUND        500
#define CUDA_ERROR_UNKNOWN          999
#define cudaSuccess                   0
#define cudaErrorMemoryAllocation     2
#define cudaErrorNotReady             6
#define cudaErrorInvalidValue        11

// ── Device attributes ─────────────────────────────────────────────────────────
#define CU_DEV_RESOURCE_TYPE_SM              1
#define CU_GREEN_CTX_DEFAULT_STREAM          0
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define cudaDevAttrL2CacheSize              38
#define cudaDevAttrMultiProcessorCount      16
#define cudaDevAttrClockRate                13
#define cudaStreamNonBlocking                1
#define cudaStreamAttributeAccessPolicyWindow 1
#define cudaMemAttachGlobal                  1

// ── Access property types ─────────────────────────────────────────────────────
typedef enum {
    cudaAccessPropertyNormal     = 0,
    cudaAccessPropertyStreaming  = 1,
    cudaAccessPropertyPersisting = 2
} cudaAccessProperty;

typedef struct {
    void*  base_ptr;
    size_t num_bytes;
    float  hitRatio;
    int    hitProp;
    int    missProp;
} cudaAccessPolicyWindow;

typedef union {
    cudaAccessPolicyWindow accessPolicyWindow;
} cudaStreamAttrValue;

// ── CUPTI stubs ───────────────────────────────────────────────────────────────
typedef struct { int a; } CUpti_Profiler_Initialize_Params;
typedef struct { int a; } CUpti_Profiler_BeginSession_Params;

// ── Inline stub functions ─────────────────────────────────────────────────────
static inline CUresult    cuInit(int f)                           { (void)f; return 0; }
static inline CUresult    cuDriverGetVersion(int* v)              { *v = 12040; return 0; }
static inline CUresult    cuDeviceGet(CUdevice* d, int i)         { *d = i; return 0; }
static inline CUresult    cuGetErrorString(CUresult r, const char** s) { *s="stub"; (void)r; return 0; }
static inline CUresult    cuCtxGetCurrent(CUcontext* c)           { *c = NULL; return 0; }
static inline CUresult    cuCtxGetDevice(CUdevice* d)             { *d = 0; return 0; }
static inline CUresult    cuDeviceGetAttribute(int* v, int a, CUdevice d) {
    (void)d;
    if (a == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) *v = 132;
    else *v = 0;
    return 0;
}
static inline CUresult    cuDeviceGetDevResourceRange(CUdevResourceDesc* d, CUdevice dev, int t) {
    (void)dev; (void)t; d->sm.smCount = 132; return 0;
}
static inline CUresult    cuDevSmResourceSplitByCount(CUdevSmResource* p, uint32_t* cnt,
    CUdevSmResource* s, uint32_t f, uint32_t m) {
    (void)s; (void)f; *cnt = m; p[0].smCount = m; p[1].smCount = 132 - m; return 0;
}
static inline CUresult    cuGreenCtxCreate(CUgreenCtx* g, void* r, CUdevice d, int f) {
    (void)r; (void)d; (void)f; *g = NULL; return 0;
}
static inline CUresult    cuCtxFromGreenCtx(CUcontext* c, CUgreenCtx g) { (void)g; *c = NULL; return 0; }
static inline void        cuGreenCtxDestroy(CUgreenCtx g)         { (void)g; }

static inline cudaError_t cudaGetDevice(int* d)                   { *d = 0; return 0; }
static inline cudaError_t cudaGetDeviceCount(int* n)              { *n = 1; return 0; }
static inline cudaError_t cudaDeviceGetAttribute(int* v, int a, int d) {
    (void)d;
    if      (a == cudaDevAttrL2CacheSize)          *v = 52428800;
    else if (a == cudaDevAttrMultiProcessorCount)  *v = 132;
    else if (a == cudaDevAttrClockRate)            *v = 1980000;
    else *v = 0;
    return 0;
}
static inline cudaError_t cudaMallocManaged(void** p, size_t n, int f) {
    *p = calloc(1, n); (void)f; return *p ? 0 : cudaErrorMemoryAllocation;
}
static inline cudaError_t cudaMalloc(void** p, size_t n) {
    *p = malloc(n); return *p ? 0 : cudaErrorMemoryAllocation;
}
static inline cudaError_t cudaFree(void* p)                       { free(p); return 0; }
static inline cudaError_t cudaMemset(void* p, int v, size_t n)    { memset(p,v,n); return 0; }
static inline cudaError_t cudaMemPrefetchAsync(void* p, size_t n, int d, void* s) {
    (void)p;(void)n;(void)d;(void)s; return 0;
}
static inline cudaError_t cudaDeviceSynchronize()                 { return 0; }
static inline cudaError_t cudaStreamCreateWithPriority(void** s, int f, int p) {
    *s = NULL; (void)f; (void)p; return 0;
}
static inline cudaError_t cudaStreamDestroy(void* s)              { (void)s; return 0; }
static inline cudaError_t cudaStreamSynchronize(void* s)          { (void)s; return 0; }
static inline cudaError_t cudaStreamSetAttribute(void* s, int a, void* v) {
    (void)s;(void)a;(void)v; return 0;
}
static inline cudaError_t cudaCtxResetPersistingL2Cache()         { return 0; }
static inline const char* cudaGetErrorString(cudaError_t e)       { (void)e; return "stub"; }

#endif // !CUDA_VERSION guard
