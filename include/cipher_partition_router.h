// CIPHER L2 Partition Router — Stage 10
//
// On H100, the 50 MB L2 is split across multiple partitions. Routing kernels
// to the partition that already holds their hot data avoids cross-partition
// hops (508 cycles vs 258 cycles for near-partition).
//
// The partition map is silicon-derived (init reads silicon model and counts
// SMs per partition). At dispatch time, the router maps a hot-region pointer
// to its preferred partition stream.
//
// Default OFF. Env: CIPHER_PARTITION_ROUTER=on.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherPartitionStats {
    int      enabled;
    int      partition_count;
    int      sms_per_partition;
    uint64_t route_calls;
    uint64_t near_partition_hits;
    uint64_t far_partition_hits;
    uint64_t unbound_routes;
    uint64_t bound_routes;
} CipherPartitionStats;

int  cipher_partition_router_init(void);
int  cipher_partition_router_enabled(void);

// Bind a hot pointer to a specific partition (0..N-1). Returns the
// partition id assigned, or -1 on failure.
int  cipher_partition_router_bind(void* ptr, int partition_hint);

// Return the partition id for `ptr`, or -1 if not bound.
int  cipher_partition_router_get(void* ptr);

int  cipher_partition_router_stats(CipherPartitionStats* out);
void cipher_partition_router_report(void);

// Stage 10 actuation: returns the CUstream associated with the partition the
// hot pointer belongs to, or NULL if the router is disabled / no streams
// have been created. Stream pool is created lazily on first call.
void* cipher_partition_router_stream_for(void* ptr);

#ifdef __cplusplus
}
#endif
