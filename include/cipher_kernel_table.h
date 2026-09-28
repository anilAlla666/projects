// CIPHER kernel-name recognition table.
//
// Per the prior session's CLAUDE.md, transparent in-hook substitution of
// PyTorch's runtime-API kernels is structurally limited because
// `cuFuncGetParamInfo` returns 0 parameters for kernels registered via
// `__cudaRegisterFunction` (the host-stub pathway).  This module:
//
//   1. Caches CUfunction → name + classification for every observed kernel.
//   2. Records grid + block dimensions on first observation (a useful signal
//      since param introspection often returns 0 for runtime-API kernels).
//   3. Probes cuFuncGetParamInfo and stores per-param offsets / sizes when
//      the driver reports them (true for cuModule-loaded kernels: cuBLAS
//      LT / FlashAttention / NVRTC).
//   4. Dumps the full table to a JSON file at process exit (or on demand
//      via the public `cipher_kernel_table_dump` API).
//
// Default off.  Enable with `CIPHER_KERNEL_TABLE_VERBOSE=1`; the per-launch
// observation path runs unconditionally (it's a single hash-table probe), but
// only emits the JSON dump and stderr per-kernel logs when the env flag is
// set.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Coarse kernel categories.  Numeric values are stable so callers can match
// against them.
typedef enum {
    CIPHER_KT_UNKNOWN     = 0xFF,
    CIPHER_KT_GEMM        = 0,
    CIPHER_KT_RMSNORM     = 1,
    CIPHER_KT_LAYERNORM   = 2,
    CIPHER_KT_SILU        = 3,
    CIPHER_KT_GELU        = 4,
    CIPHER_KT_SOFTMAX     = 5,
    CIPHER_KT_ROPE        = 6,
    CIPHER_KT_FLASHATTN   = 7,
    CIPHER_KT_ATTN_OTHER  = 8,
    CIPHER_KT_EMBEDDING   = 9,
    CIPHER_KT_ELEM_MUL    = 10,
    CIPHER_KT_RESIDUAL    = 11,
    CIPHER_KT_COPY        = 12,
    CIPHER_KT_TRANSPOSE   = 13,
    CIPHER_KT_REDUCE      = 14,
    CIPHER_KT_CAST        = 15,
    CIPHER_KT_ELEM_GENERIC= 16,
} CipherKernelCategory;

typedef struct CipherKernelEntry {
    void*    fn_handle;          // CUfunction pointer
    char     name[128];          // demangled / mangled name
    uint8_t  category;           // CipherKernelCategory
    uint16_t param_count;        // 0 if cuFuncGetParamInfo failed
    uint16_t param_offsets[16];
    uint16_t param_sizes[16];
    uint32_t first_grid_x, first_grid_y, first_grid_z;
    uint32_t first_block_x, first_block_y, first_block_z;
    uint32_t first_smem_bytes;
    uint64_t observe_count;
} CipherKernelEntry;

// Called from the cuLaunchKernel / cuLaunchKernelEx intercept on every
// launch.  Returns the entry if it has been observed before, NULL on the
// FIRST observation (so the caller can treat first observation as a
// classification event) — or returns the freshly-created entry if the
// caller wants to skip the first-observation distinction.  Both are fine.
//
// Parameters describe what the launch site has at hand.  `kernel_args` may
// be NULL when invoked from cuLaunchKernel (no param array on hand) — the
// table uses this for verbose logging only.
const CipherKernelEntry* cipher_kt_observe(
    void*    fn_handle,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    uint32_t smem_bytes);

// Total number of distinct CUfunctions seen so far.
unsigned cipher_kt_size(void);

// Dump every observed kernel as JSON to `path` (NULL → /tmp/cipher_kernel_table.json).
// Returns number of entries written.
int cipher_kt_dump_json(const char* path);

#ifdef __cplusplus
}
#endif
