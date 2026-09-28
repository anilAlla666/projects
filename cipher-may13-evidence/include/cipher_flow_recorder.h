// CIPHER pointer-flow recorder — Stage 1 of driver-level kernel fusion via
// pattern matching.
//
// Records every cudaLaunchKernel into a fixed-size circular buffer.  Each
// entry captures the func_ptr, grid/block, and the first 8 raw arg values
// (read as uint64_t — most kernel params are pointers or small ints, so
// 8 bytes captures the identifying info).
//
// Pure observation; no substitution or suppression at this stage.  Gated
// by env CIPHER_FLOW_RECORD=on.  Dump on demand via
// cipher_flow_recorder_dump() (called from a destructor when
// CIPHER_FLOW_DUMP=on, or when CIPHER_FLOW_DUMP_AFTER=N total launches
// hits N).

#ifndef CIPHER_FLOW_RECORDER_H
#define CIPHER_FLOW_RECORDER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// One launch entry.  POD; no heap.
struct CipherFlowEntry {
    void*    func_ptr;
    uint64_t args[8];     // raw values from *(uint64_t*)args[i]
    uint8_t  is_ptr[8];   // 1 if args[i] looks like a device pointer
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t smem;
    uint64_t seq;         // monotonic launch index
    uint64_t stream;      // raw stream ptr (for per-stream sequencing)
};

constexpr int CIPHER_FLOW_RING = 64;

int  cipher_flow_recorder_init(void);
int  cipher_flow_recorder_enabled(void);

// Called from cudaLaunchKernel shim.  Cheap: ~150 ns when enabled.
void cipher_flow_recorder_observe(
    void* func, uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem, void** args, void* stream);

// Dump the ring to stderr, oldest-first.  Includes resolved kernel names
// where available (uses cipher_kernel_table classification).
void cipher_flow_recorder_dump(void);

#ifdef __cplusplus
}
#endif

#endif
