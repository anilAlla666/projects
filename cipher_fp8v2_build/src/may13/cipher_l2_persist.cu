// CPU stub redirect
#ifdef CIPHER_CPU_STUB
#  include "may13/cipher_stubs.h"
#else
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#  include <cuda.h>
#endif
// =============================================================================
// CIPHER — F3: L2 Persistent Weight Loading Implementation
// cipher_l2_persist.cu
// =============================================================================

#include "may13/cipher_l2_persist.h"
#include "cipher_rt_commit.h"
#include <stdio.h>
#include <string.h>

#define CIPHER_CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "[CIPHER F3] CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
            return _e; \
        } \
    } while(0)

// ---------------------------------------------------------------------------
// Query hardware L2 size
// ---------------------------------------------------------------------------

static size_t query_l2_size(void) {
    int device;
    cudaGetDevice(&device);
    int l2_size = 0;
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
    return (size_t)l2_size;
}

// ---------------------------------------------------------------------------
// cipher_l2_persist_init
// ---------------------------------------------------------------------------

extern CipherL2PersistState *cipher_l2_persist_state_ptr;

cudaError_t cipher_l2_persist_init(CipherL2PersistState* state,
                                   cudaStream_t          stream)
{
    memset(state, 0, sizeof(*state));
    state->persist_stream    = stream;
    state->l2_capacity_bytes = query_l2_size();
    /* W12 Step 6 D3 backfill: capture state pointer for plugin accessor. */
    cipher_l2_persist_state_ptr = state;

    if (state->l2_capacity_bytes == 0) {
        fprintf(stderr, "[CIPHER F3] WARNING: L2 size query returned 0. "
                        "Persistence may not work on this device.\n");
    }

    state->initialized = true;
    fprintf(stderr, "[CIPHER F3] L2 capacity: %.1f MB  |  CIPHER budget: %.1f MB\n",
            (double)state->l2_capacity_bytes / (1024*1024),
            (double)CIPHER_L2_PERSIST_MAX_BYTES / (1024*1024));
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// cipher_l2_persist_register
// ---------------------------------------------------------------------------

cudaError_t cipher_l2_persist_register(CipherL2PersistState* state,
                                       void*   device_ptr,
                                       size_t  size_bytes,
                                       int     layer_id,
                                       const char* name)
{
    if (!state->initialized)  return cudaErrorNotReady;
    if (state->count >= CIPHER_MAX_WEIGHT_TENSORS) {
        fprintf(stderr, "[CIPHER F3] Tensor table full (%d entries)\n",
                CIPHER_MAX_WEIGHT_TENSORS);
        return cudaErrorMemoryAllocation;
    }

    size_t new_total = state->total_bytes + size_bytes;
    if (new_total > CIPHER_L2_PERSIST_MAX_BYTES) {
        fprintf(stderr,
            "[CIPHER F3] L2 persist budget exceeded: "
            "%.2f MB + %.2f MB > %.2f MB ceiling\n",
            (double)state->total_bytes   / (1024*1024),
            (double)size_bytes           / (1024*1024),
            (double)CIPHER_L2_PERSIST_MAX_BYTES / (1024*1024));
        return cudaErrorMemoryAllocation;
    }

    CipherWeightTensor* t = &state->tensors[state->count++];
    t->device_ptr  = device_ptr;
    t->size_bytes  = size_bytes;
    t->layer_id    = layer_id;
    t->pinned      = false;
    strncpy(t->name, name ? name : "unnamed", sizeof(t->name)-1);

    state->total_bytes = new_total;
    fprintf(stderr, "[CIPHER F3] Registered: %s  %.2f KB  (layer %d)\n",
            t->name, (double)size_bytes/1024.0, layer_id);
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// cipher_l2_persist_apply
//
// cudaAccessPolicyWindow pins a contiguous range of device memory into L2.
// We apply one window per tensor. If the tensor is already pinned we skip.
//
// The 'hitProp' is cudaAccessPropertyPersisting — tells the cache to keep
// this data in L2 as long as possible, evicting normal data first.
// The 'missProp' is cudaAccessPropertyStreaming — new data brought in by
// normal accesses doesn't pollute the persist set.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// cipher_l2_persist_apply_for_tenant — W10-12 Step 3 per-tenant-stream wire.
//
// Re-applies the same global tensors[] policy against the caller's stream
// (typically a per-tenant named CUDA stream from the W7-9 Step 5 view-slot
// table). Each tenant's named stream needs the policy attribute set
// individually for its launches to honor the persisting policy; without
// this, only streams registered at cipher_l2_persist_init time get the
// policy and new tenant streams don't.
//
// Per advisor 2026-05-23 honest framing: this is per-tenant-STREAM L2
// policy application against a shared tensor set. NOT "per-tenant L2
// persistence" — the tensors[] array stays global in v1. Per-tenant
// tensor tracking is W13-14 work.
//
// Emits RING_WRITE slot CIPHER_RT_RING_EVENT_L2_PERSIST=7 on each apply.
// ---------------------------------------------------------------------------

// Weak ref so cipher_l2_persist.cu compiles standalone when linked into
// libcipher_rt without dragging in the ring_write substrate at object
// scope (the runtime resolves it from the same .so).
extern "C" void cipher_rt_ring_write(unsigned int tenant_id,
                                     unsigned int event_type,
                                     unsigned int event_subtype,
                                     unsigned long long commit_seq,
                                     const void *payload,
                                     unsigned long payload_len)
    __attribute__((weak));

extern "C"
cudaError_t cipher_l2_persist_apply_for_tenant(CipherL2PersistState* state,
                                               unsigned int tenant_id,
                                               cudaStream_t stream)
{
    if (!state->initialized) return cudaErrorNotReady;
    if (!stream) return cudaErrorInvalidValue;

    // Verify cuda compute capability supports cudaAccessPolicyWindow (sm_80+).
    // H100 is sm_90; this check guards against future runtime where CC < 8.0.
    int cc_major = 0;
    cudaError_t ce = cudaDeviceGetAttribute(&cc_major,
        cudaDevAttrComputeCapabilityMajor, 0);
    if (ce == cudaSuccess && cc_major < 8) {
        fprintf(stderr,
            "[CIPHER L2] CC %d.x < 8.0; cudaAccessPolicyWindow unavailable; "
            "skipping per-tenant L2 policy for tenant=%u\n",
            cc_major, tenant_id);
        return cudaSuccess;  /* soft no-op */
    }

    int applied = 0;
    unsigned long long total_bytes = 0;
    for (int i = 0; i < state->count; i++) {
        CipherWeightTensor* t = &state->tensors[i];
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));

        cudaAccessPolicyWindow* w = &attr.accessPolicyWindow;
        w->base_ptr        = t->device_ptr;
        w->num_bytes       = t->size_bytes;
        w->hitRatio        = 1.0f;
        w->hitProp         = cudaAccessPropertyPersisting;
        w->missProp        = cudaAccessPropertyStreaming;

        cudaError_t r = cudaStreamSetAttribute(stream,
            cudaStreamAttributeAccessPolicyWindow, &attr);
        if (r != cudaSuccess) {
            fprintf(stderr,
                "[CIPHER L2] cudaStreamSetAttribute failed for tenant=%u "
                "stream=%p tensor=%s: %d\n",
                tenant_id, stream, t->name, (int)r);
            continue;
        }
        applied++;
        total_bytes += t->size_bytes;
    }

    /* RING_WRITE event slot 7. */
    if (cipher_rt_ring_write) {
        struct l2_persist_event {
            unsigned int tenant_id;
            unsigned int hot_bytes_mib;
            unsigned int hit_ratio_x1000;
            unsigned int tensor_count;
        } ev = {
            tenant_id,
            (unsigned int)(total_bytes / (1024 * 1024)),
            1000u,
            (unsigned int)applied,
        };
        cipher_rt_ring_write(tenant_id,
                             7u /* CIPHER_RT_RING_EVENT_L2_PERSIST */,
                             (unsigned int)applied,
                             0ULL, &ev, sizeof(ev));
    }
    return cudaSuccess;
}

cudaError_t cipher_l2_persist_apply(CipherL2PersistState* state) {
    if (!state->initialized) return cudaErrorNotReady;

    for (int i = 0; i < state->count; i++) {
        CipherWeightTensor* t = &state->tensors[i];
        if (t->pinned) continue;   // Already applied

        // Build the access policy window
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));

        cudaAccessPolicyWindow* w = &attr.accessPolicyWindow;
        w->base_ptr        = t->device_ptr;
        w->num_bytes       = t->size_bytes;
        w->hitRatio        = 1.0f;   // Always persist — we're within budget
        w->hitProp         = cudaAccessPropertyPersisting;
        w->missProp        = cudaAccessPropertyStreaming;

        CIPHER_CUDA_CHECK(cudaStreamSetAttribute(
            state->persist_stream,
            cudaStreamAttributeAccessPolicyWindow,
            &attr));

        t->pinned = true;
        fprintf(stderr,
            "[CIPHER F3] Pinned: %s  %.2f KB  → L2 persistent\n",
            t->name, (double)t->size_bytes/1024.0);
    }

    // Sync stream to ensure persistence is applied before any LNN inference
    CIPHER_CUDA_CHECK(cudaStreamSynchronize(state->persist_stream));
    return cudaSuccess;
}

// ---------------------------------------------------------------------------
// cipher_l2_persist_unpin
// Called before weight update (online learning step). Unpins one tensor
// so the update kernel sees a clean L2 state.
// ---------------------------------------------------------------------------

cudaError_t cipher_l2_persist_unpin(CipherL2PersistState* state,
                                    void* device_ptr)
{
    for (int i = 0; i < state->count; i++) {
        CipherWeightTensor* t = &state->tensors[i];
        if (t->device_ptr != device_ptr) continue;
        if (!t->pinned) return cudaSuccess;

        // Set window to streaming (unpins from L2)
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr  = t->device_ptr;
        attr.accessPolicyWindow.num_bytes = t->size_bytes;
        attr.accessPolicyWindow.hitRatio  = 0.0f;
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

        CIPHER_CUDA_CHECK(cudaStreamSetAttribute(
            state->persist_stream,
            cudaStreamAttributeAccessPolicyWindow,
            &attr));

        t->pinned = false;
        state->total_bytes -= t->size_bytes;
        fprintf(stderr, "[CIPHER F3] Unpinned: %s\n", t->name);
        return cudaSuccess;
    }
    return cudaErrorInvalidValue;
}

// ---------------------------------------------------------------------------
// cipher_l2_persist_reset — remove all windows
// ---------------------------------------------------------------------------

void cipher_l2_persist_reset(CipherL2PersistState* state) {
    if (!state->initialized) return;

    // Null window with num_bytes=0 resets to default behavior
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.num_bytes = 0;

    cudaStreamSetAttribute(
        state->persist_stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr);

    // Apply the reset
    cudaCtxResetPersistingL2Cache();

    for (int i = 0; i < state->count; i++)
        state->tensors[i].pinned = false;

    fprintf(stderr, "[CIPHER F3] All L2 persistence windows reset.\n");
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

void cipher_l2_persist_report(const CipherL2PersistState* state) {
    /* W7-9 Step 4 — coherent snapshot acquire (slow-path; emits the
     * COMMIT-published state alongside this report's existing aggregate). */
    struct cipher_rt_snapshot _snap;
    cipher_rt_snapshot_acquire(0u, &_snap);
    (void)_snap;

    fprintf(stderr,
        "[CIPHER F3] L2 Persist Report\n"
        "  Device L2:       %.1f MB\n"
        "  CIPHER pinned:   %.2f MB (%.1f%%)\n"
        "  Tensor count:    %d\n",
        (double)state->l2_capacity_bytes / (1024*1024),
        (double)state->total_bytes       / (1024*1024),
        state->l2_capacity_bytes > 0
            ? (double)state->total_bytes * 100.0 / state->l2_capacity_bytes
            : 0.0,
        state->count);

    for (int i = 0; i < state->count; i++) {
        const CipherWeightTensor* t = &state->tensors[i];
        fprintf(stderr,
            "    [%d] %-24s  %.2f KB  L%d  %s\n",
            i, t->name,
            (double)t->size_bytes / 1024.0,
            t->layer_id + 1,
            t->pinned ? "PINNED" : "unpinned");
    }
}

// ---------------------------------------------------------------------------
// cipher_l2_persist_get_state — W12 Step 6 D3 backfill accessor.
//
// Returns the address of the global CipherL2PersistState in g_cipher.
// Plugin-side caller uses this via ctypes to invoke
// cipher_l2_persist_apply_for_tenant against the runtime's live state
// without needing to know the CipherRuntime struct layout. The state
// pointer is captured at cipher_l2_persist_init time in the file-scope
// variable below. Returns NULL until init runs.
// ---------------------------------------------------------------------------
CipherL2PersistState *cipher_l2_persist_state_ptr = nullptr;

extern "C" CipherL2PersistState *cipher_l2_persist_get_state(void)
{
    return cipher_l2_persist_state_ptr;
}
