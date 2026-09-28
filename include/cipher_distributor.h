// CIPHER multi-tenant auto-distributor
// =====================================
// Per-tenant load detection + intra-GPU rebalance via ARBITRATE+FAIRNESS
// + cross-GPU spawn router + advisory shed signal for upstream LBs.
//
// Design (M1.T4):
//   - Per-tenant load metric: rolling tok/s (derived from gemm_calls delta
//     over a 30 s window). Maintains historical p95 for stability.
//   - Trigger: sustained 30 s window where tenant exceeds historical p95
//     by N% (default 50%) AND GPU SM utilization > 95%.
//   - Intra-GPU: yield SMs from low-load tenants via existing FAIRNESS
//     yield_count mechanism.
//   - Cross-GPU: write `cipher_tenant_should_migrate{tenant=X}=1` into
//     the distributor shm. Operator's load balancer (envoy / nginx / k8s
//     hpa) reads via Prometheus and reroutes new requests for X.
//   - Cross-GPU spawn router: at next tenant launch, pick least-loaded
//     GPU on the node by reading the per-GPU load table.
//
// State is in POSIX shm (/dev/shm/cipher_distributor) so all tenants on
// the same node observe the same view. fcntl locks + monotonic seq nums
// guard against torn reads.

#ifndef CIPHER_DISTRIBUTOR_H
#define CIPHER_DISTRIBUTOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Init — opens shm, starts the 50 ms poll thread. Idempotent.
// Returns 1 if enabled, 0 if disabled (CIPHER_DISTRIBUTOR=off or shm open failed).
int  cipher_distributor_init(void);

// Hot-path observe: tenant called a kernel. Bumps the per-tenant
// dispatch counter. Cheap (one atomic add).
void cipher_distributor_observe_kernel(uint32_t tenant_id);

// Query: should tenant X be migrated to a different GPU/node? Returns 1
// if the distributor decided yes. Read by the Prometheus exporter, the
// orchestrator (cipher_run.sh hint), and the cross-GPU spawn router.
int  cipher_distributor_should_migrate(uint32_t tenant_id);

// Pick the least-loaded GPU for a new tenant on this node. Returns
// the GPU index in [0, n_gpus); -1 on error (shm not init).
int  cipher_distributor_pick_gpu_for_new_tenant(void);

// Stats for Prometheus export — fills the user-supplied struct.
struct CipherDistributorStats {
    int      enabled;
    int      n_active_tenants;
    int      n_gpus_seen;
    int      n_overloaded_tenants;
    uint64_t intra_gpu_rebalances;
    uint64_t cross_gpu_migrate_signals;
    uint64_t spawn_router_picks;
};
int  cipher_distributor_stats(struct CipherDistributorStats* out);

#ifdef __cplusplus
}
#endif

#endif // CIPHER_DISTRIBUTOR_H
