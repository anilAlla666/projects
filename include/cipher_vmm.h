// CIPHER VMM Pool — Stage 4
//
// Virtual-memory-managed allocator for CIPHER-internal regions that need
// pointer stability across resize/reallocation (compressed weight tables,
// KV scratch, persistent kernel param blocks).
//
// Backed by libcuda's cuMemAddressReserve / cuMemCreate / cuMemMap (resolved
// via dlopen). Caller treats `cipher_vmm_alloc` like `cudaMalloc` but with
// the guarantee that the virtual address can be re-mapped to a different
// physical handle without changing the pointer (dual-reservation shadow
// swap pattern). PyTorch's expandable_segments uses the same primitive.
//
// Default OFF. Env: CIPHER_VMM=on/1/ON. Init is idempotent.
//
// IMPORTANT: This is a *side allocator* for CIPHER's own use. It does NOT
// intercept cudaMalloc / cuMemAlloc. PyTorch and applications keep their
// allocator. This avoids the entire class of allocator-mismatch bugs.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherVmmStats {
    int      enabled;
    int      driver_available;
    int      gpu_id;
    size_t   granularity;             // minimum alloc/map granularity (bytes)
    size_t   reserved_bytes;          // total VA reserved
    size_t   committed_bytes;         // physical memory mapped
    size_t   live_bytes;              // currently allocated to callers
    size_t   peak_live_bytes;
    uint64_t alloc_calls;
    uint64_t free_calls;
    uint64_t swap_calls;
    uint64_t map_calls;
    uint64_t unmap_calls;
} CipherVmmStats;

// Initialize. Returns 1 on success, 0 if disabled or libcuda unavailable.
int  cipher_vmm_init(void);
int  cipher_vmm_enabled(void);

// Allocate `bytes` of VMM-backed device memory. Returns 0 (NULL ptr) on
// failure. Caller is responsible for `cipher_vmm_free` when done.
unsigned long long cipher_vmm_alloc(size_t bytes);

// Free a VMM allocation. Returns 1 on success.
int  cipher_vmm_free(unsigned long long ptr);

// Dual-reservation shadow swap: replace the physical backing of `dst_ptr`
// with the backing of `src_ptr`. Both must be VMM allocations of the same
// size. Used for atomic weight reload and live-mode replacement.
// Returns 1 on success.
int  cipher_vmm_swap(unsigned long long dst_ptr, unsigned long long src_ptr);

// Snapshot stats. Returns 1 on success.
int  cipher_vmm_stats(CipherVmmStats* out);

// JSON report at /tmp/cipher_vmm_report.json. Off hot path.
void cipher_vmm_report(void);

#ifdef __cplusplus
}
#endif
