// =============================================================================
// CIPHER — Runtime Initialization
// cipher_runtime.cpp
//
// Wires F2 → F3 → F4 → F5 in order.
// cipher_dispatch() is the stub that will grow into full L3 routing in Week 2.
// =============================================================================

#include "cipher.h"
#include <stdio.h>
#include <stdlib.h>

// Global runtime singleton
CipherRuntime g_cipher = {0};

// ---------------------------------------------------------------------------
// cipher_init
// ---------------------------------------------------------------------------

int cipher_init(int device_ordinal) {
    if (g_cipher.initialized) return 0;

    fprintf(stderr,
        "\n╔══════════════════════════════════════════════════════╗\n"
        "║  CIPHER v%s — Neural Dynamics            ║\n"
        "║  Learned Operator Compilation                        ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
        "  O(N)→O(1) at cuLaunchKernel. LD_PRELOAD active.\n\n",
        CIPHER_VERSION_STR);

    g_cipher.device_ordinal = device_ordinal;

    // F2: Green Context Allocation
    CUresult cr = cipher_green_ctx_init(&g_cipher.green_ctx, device_ordinal);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "[CIPHER INIT] F2 Green Ctx failed: %d\n", cr);
        return -1;
    }

    // F3: L2 Persistent Weight Loading
    // Use Layer 3's stream for weight persistence
    cudaStream_t persist_stream = cipher_get_stream(&g_cipher.green_ctx,
                                                     CIPHER_CTX_LAYER3);
    cudaError_t ce = cipher_l2_persist_init(&g_cipher.l2_persist,
                                             persist_stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[CIPHER INIT] F3 L2 Persist failed: %s\n",
                cudaGetErrorString(ce));
        return -1;
    }

    // F4: Shared Liquid State
    ce = cipher_liquid_state_init(&g_cipher.liquid);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[CIPHER INIT] F4 Liquid State failed: %s\n",
                cudaGetErrorString(ce));
        return -1;
    }

    // F5: Hardware Telemetry
    int rc = cipher_telemetry_init(&g_cipher.telemetry, device_ordinal,
                                   &g_cipher.liquid);
    if (rc != 0) {
        fprintf(stderr, "[CIPHER INIT] F5 Telemetry init failed (non-fatal). "
                        "Layer 2 orchestration degraded.\n");
    }

    // Layer 3: Classification + Oracle + Recipes + Registry
    rc = cipher_layer3_init();
    if (rc != 0) {
        fprintf(stderr, "[CIPHER INIT] Layer 3 init failed: %d\n", rc);
        return -1;
    }

    g_cipher.initialized = true;
    fprintf(stderr, "[CIPHER INIT] Phase 0 + Layer 3 complete. All systems GO.\n\n");
    return 0;
}

// ---------------------------------------------------------------------------
// cipher_teardown
// ---------------------------------------------------------------------------

void cipher_teardown(void) {
    if (!g_cipher.initialized) return;
    fprintf(stderr, "\n[CIPHER] Teardown initiated.\n");

    cipher_telemetry_destroy(&g_cipher.telemetry);
    cipher_liquid_state_destroy(&g_cipher.liquid);
    cipher_l2_persist_reset(&g_cipher.l2_persist);
    cipher_green_ctx_destroy(&g_cipher.green_ctx);

    g_cipher.initialized = false;
    fprintf(stderr, "[CIPHER] Shutdown complete.\n");
}

// cipher_dispatch is now in cipher_dispatch.cpp (Layer 3 full implementation)

// ---------------------------------------------------------------------------
// cipher_report
// ---------------------------------------------------------------------------

void cipher_report(void) {
    fprintf(stderr,
        "\n═══════════════════════════════════════════════════════\n"
        "  CIPHER Runtime Report — v%s\n"
        "═══════════════════════════════════════════════════════\n",
        CIPHER_VERSION_STR);

    cipher_green_ctx_report(&g_cipher.green_ctx);
    fprintf(stderr, "\n");
    cipher_l2_persist_report(&g_cipher.l2_persist);
    fprintf(stderr, "\n");
    cipher_liquid_state_report(&g_cipher.liquid);
    fprintf(stderr, "\n");
    cipher_telemetry_report(&g_cipher.telemetry);
    cipher_layer3_report();

    const CipherInterceptStats* is = cipher_intercept_stats();
    uint64_t avg_ns = is->total_intercepts > 0
        ? is->overhead_ns_sum / is->total_intercepts : 0;
    fprintf(stderr,
        "\n[F1] Intercept Stats\n"
        "  Total intercepts:  %lu\n"
        "  Substitutions:     %lu  (%.1f%%)\n"
        "  Passthroughs:      %lu\n"
        "  Deferred:          %lu\n"
        "  Avg overhead:      %lu ns\n"
        "  Max overhead:      %lu ns\n",
        is->total_intercepts,
        is->substitutions,
        is->total_intercepts > 0
            ? (double)is->substitutions * 100.0 / is->total_intercepts : 0.0,
        is->passthroughs,
        is->deferred,
        avg_ns,
        is->overhead_ns_max);

    fprintf(stderr,
        "═══════════════════════════════════════════════════════\n\n");
}
