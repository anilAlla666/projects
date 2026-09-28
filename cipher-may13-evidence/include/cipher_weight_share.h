// CIPHER weight sharing via CUDA IPC.  Lets multiple processes hosting
// the same model share one GPU copy of the weights.  Driver API only.
#ifndef CIPHER_WEIGHT_SHARE_H
#define CIPHER_WEIGHT_SHARE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the IPC subsystem for the current GPU.  Idempotent.  Reads
// CIPHER_WEIGHT_SHARE=on to gate; returns 0 if disabled.
int  cipher_weight_share_init(void);
int  cipher_weight_share_enabled(void);

// Try to publish (dev_ptr, bytes, content_hash) to /dev/shm so other
// processes can map it.  Returns slot id ≥ 0 on success, -1 on failure
// (non-IPC-mappable pointer, env disabled, etc).  Idempotent on
// (content_hash, bytes) — second caller with the same hash gets the
// existing slot, ref-counts up.
int cipher_weight_share_export(void* dev_ptr, size_t bytes,
                               uint64_t content_hash);

// Look up a SHARED dev pointer for `original_ptr`.  Returns the shared
// pointer if this process has imported the slot for original_ptr's
// content, else NULL.  Lookup is by ORIGINAL pointer — caller must have
// gone through observe→export first.
void* cipher_weight_share_lookup(void* original_ptr);

// Observe a candidate weight pointer.  Internally hashes 128 bytes from
// dev_ptr (1 D2H sync, ~1us), tracks per-pointer hits, and on the second
// matching observation triggers cipher_weight_share_export().  Called
// from the cublasGemmEx shim.  Cheap no-op if CIPHER_WEIGHT_SHARE=off.
void cipher_weight_share_observe(void* dev_ptr, size_t bytes);

// Returns 1 if any sharing is currently active in this process.
int cipher_weight_share_active_count(void);

// Force release of all imports in the current process.  Called at
// teardown.
void cipher_weight_share_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
