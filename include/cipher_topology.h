// Op 25 TOPOLOGY — NVLink/PCIe peer adjacency inference.
//
// At init, queries cudaGetDeviceCount and cudaDeviceCanAccessPeer for every
// device pair, emitting a dense adjacency matrix. Observer is a no-op
// (topology is static post-init). Report dumps JSON.
//
// Default OFF: env var `CIPHER_TOPOLOGY=on`. Contract I1–I6 (CUDA calls are
// init-time only; hot path observe is a single relaxed load + branch).
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_topology_init(void);
void     cipher_topology_observe(const CipherRingEntry* ev);
unsigned cipher_topology_device_count(void);
void     cipher_topology_report(void);

// OP 33 — On CIPHER_NCCL_TUNER=on, infer link topology and setenv() the
// NCCL_ALGO/NCCL_PROTO/NCCL_MIN_NCHANNELS / NCCL_IB_* hints BEFORE any
// NCCL collective fires. Idempotent. Returns 1 if any env was set.
int cipher_topology_setenv_for_nccl(void);

typedef struct CipherNcclTunerStats {
    int      enabled;
    int      ran;                 // setenv pass executed
    int      nvlink_detected;     // any GPU-pair has NVLink
    int      pcie_only;           // GPU-pair P2P but no NVLink
    int      ib_detected;         // /sys/class/infiniband/* present
    int      gpu_count;
    int      nvlink_pairs;
    char     algo[32];
    char     proto[32];
    int      min_nchannels;
    int      ib_disable;          // 0 or 1; 2 = unset
    int      net_gdr_level;       // 0..5; -1 = unset
} CipherNcclTunerStats;

int cipher_topology_nccl_tuner_stats(CipherNcclTunerStats* out);

#ifdef __cplusplus
}
#endif
