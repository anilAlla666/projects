// CIPHER Graph Engine — Stage 5
//
// Detects stable repeating launch sequences and replaces N individual launches
// with one cuGraphLaunch via cuStreamBeginCapture / cuGraphInstantiate. State
// machine: OBSERVE → DECIDE → CAPTURE → REPLAY → INVALIDATE.
//
// Default OFF. Env: CIPHER_GRAPH=on. Coexists with application-side capture
// (checks cuStreamGetCaptureInfo_v2 before any begin_capture).

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CIPHER_GRAPH_OBSERVE   = 0,
    CIPHER_GRAPH_DECIDED   = 1,
    CIPHER_GRAPH_CAPTURED  = 2,
    CIPHER_GRAPH_REPLAYING = 3,
    CIPHER_GRAPH_INVALID   = 4,
} CipherGraphState;

typedef struct CipherGraphStats {
    int      enabled;
    uint64_t observe_calls;
    uint64_t sequences_seen;
    uint64_t sequences_promoted;
    uint64_t replays;
    uint64_t invalidations;
    uint64_t capture_skipped_app;   // skipped because app was already capturing
} CipherGraphStats;

int  cipher_graph_init(void);
int  cipher_graph_enabled(void);

// Feed a launch fingerprint into the sequence detector. Returns the resulting
// state for the kernel-fingerprint chain on the calling thread.
int  cipher_graph_observe(uint64_t kernel_fp);

// Indicate end-of-step boundary. Caller is responsible for calling at the
// natural sequence boundary (e.g., before cudaStreamSynchronize). Returns 1
// if a sequence was promoted to REPLAY at this boundary.
int  cipher_graph_step_boundary(void);

// Snapshot stats / write JSON report.
int  cipher_graph_stats(CipherGraphStats* out);
void cipher_graph_report(void);

// Stage 5 actuation — opt-in API for explicit graph capture/replay.
// `stream_handle` is a CUstream. Returns 1 on success, 0 otherwise.
//
//   begin_capture: starts cuStreamBeginCapture(stream, RELAXED). Must NOT
//                  be called when an application-side capture is already
//                  active (we check via cuStreamIsCapturing).
//   end_capture:   ends the capture, instantiates the graph. Returns a
//                  positive graph_id, or 0 on failure.
//   replay:        calls cuGraphLaunch(graph_id, stream).
//   destroy:       releases the graph + executable.
int           cipher_graph_begin_capture(void* stream_handle);
unsigned long cipher_graph_end_capture(void* stream_handle);
int           cipher_graph_replay(unsigned long graph_id, void* stream_handle);
int           cipher_graph_destroy(unsigned long graph_id);

#ifdef __cplusplus
}
#endif
