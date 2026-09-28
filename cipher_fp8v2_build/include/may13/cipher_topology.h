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
#include "may13/cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_topology_init(void);
void     cipher_topology_observe(const CipherRingEntry* ev);
unsigned cipher_topology_device_count(void);
void     cipher_topology_report(void);

#ifdef __cplusplus
}
#endif
