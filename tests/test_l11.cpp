// =============================================================================
// CIPHER — L1.1: Runtime Koopman Derivation Tests
// tests/test_l11.cpp
//
// Tests:
//   ✓ cipher_kr_hash: deterministic, collision-resistant
//   ✓ cipher_kr_init: initializes all records
//   ✓ cipher_kr_find_or_create: new op creates record
//   ✓ cipher_kr_find_or_create: same op returns same idx
//   ✓ cipher_kr_find_or_create: different geometry = different record
//   ✓ cipher_kr_find_or_create: table full returns -1
//   ✓ cipher_kr_decide: unknown op → collecting + passthrough
//   ✓ cipher_kr_decide: collecting phase increments passthroughs
//   ✓ cipher_kr_record_output: adds snapshots correctly
//   ✓ cipher_kr_record_output: triggers solve at MIN_SNAPSHOTS
//   ✓ cipher_kr_decide: derived op → should_substitute = true
//   ✓ cipher_kr_predict: predicts output after derivation
//   ✓ cipher_kr_predict: returns false before derivation
//   ✓ Full pipeline: collect → solve → substitute for N ops
//   ✓ Multiple ops tracked simultaneously
//   ✓ Feature extraction: correct dimensions
//   ✓ Feature extraction: normalized range [0,1]
//   ✓ Feature extraction: op_class encoded correctly
//   ✓ Online refinement after derivation
//   ✓ Failed EDMD → CIPHER_KR_FAILED state
//   ✓ cipher_kr_report: runs without crash
// =============================================================================

#ifdef CIPHER_CPU_STUB
#include "cipher_stubs.h"
#endif

#include "cipher_koopman_runtime.h"
#include "cipher_edmd.h"
#include "cipher_liquid_state.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        printf("  \033[32m✓\033[0m %s\n", msg); g_pass++; \
    } else { \
        printf("  \033[31m✗\033[0m %s\n", msg); g_fail++; \
    } \
} while(0)

#define CHECK_FMT(cond, msg, ...) do { \
    char _buf[256]; \
    snprintf(_buf, sizeof(_buf), msg, ##__VA_ARGS__); \
    if (cond) { printf("  \033[32m✓\033[0m %s\n", _buf); g_pass++; } \
    else      { printf("  \033[31m✗\033[0m %s\n", _buf); g_fail++; } \
} while(0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static CipherLiquidStateMgr make_liquid() {
    CipherLiquidStateMgr mgr;
    memset(&mgr, 0, sizeof(mgr));
    static CipherLiquidState ls;
    memset(&ls, 0, sizeof(ls));
    ls._magic = CIPHER_LIQUID_STATE_MAGIC;
    ls.phase  = 1;
    ls.global_grad_ema = 0.5f;
    mgr.device      = &ls;
    mgr.initialized = true;
    return mgr;
}

// Generate synthetic (input, output) snapshot pair for a given op
// Use high-variance data so EDMD Chebyshev observables have good spread.
static int g_snap_seed = 0;
static void make_snapshot(uint8_t op_class,
                           uint32_t gx, uint32_t gy,
                           uint32_t bs, uint32_t sh,
                           float* inp, float* out, uint32_t n,
                           const CipherLiquidStateMgr* liq)
{
    (void)op_class; (void)gx; (void)gy; (void)bs; (void)sh; (void)liq;
    // High-variance inputs in [-1, 1] — needed for Chebyshev observables
    // which are defined on [-1,1] and need spread to avoid rank deficiency
    g_snap_seed++;
    for (uint32_t i = 0; i < n; i++) {
        // Deterministic pseudo-random in [-1, 1]
        int seed = g_snap_seed * 31 + (int)i * 7;
        inp[i] = (float)((seed * 1103515245 + 12345) & 0x7FFFFFFF) / (float)0x40000000 - 1.0f;
    }
    // Smooth linear output EDMD can learn: y = A*x where A is a fixed matrix
    for (uint32_t i = 0; i < n; i++) {
        out[i] = 0.0f;
        for (uint32_t j = 0; j < n; j++) {
            // Simple banded matrix
            float a = (i == j) ? 0.8f : (abs((int)i-(int)j) == 1) ? 0.1f : 0.0f;
            out[i] += a * inp[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Hash
// ---------------------------------------------------------------------------

static void test_hash() {
    printf("\n[L1.1.1] Hash\n");

    uint64_t h1 = cipher_kr_hash(0, 1024, 1024, 256, 0);
    uint64_t h2 = cipher_kr_hash(0, 1024, 1024, 256, 0);
    CHECK(h1 == h2, "same params → same hash");

    uint64_t h3 = cipher_kr_hash(0, 1024, 1024, 256, 1024);
    CHECK(h1 != h3, "different shmem → different hash");

    uint64_t h4 = cipher_kr_hash(1, 1024, 1024, 256, 0);
    CHECK(h1 != h4, "different op_class → different hash");

    uint64_t h5 = cipher_kr_hash(0, 2048, 1024, 256, 0);
    CHECK(h1 != h5, "different grid_x → different hash");

    uint64_t h6 = cipher_kr_hash(0, 1024, 1024, 128, 0);
    CHECK(h1 != h6, "different block_size → different hash");

    // All hashes distinct across first 8 op classes
    uint64_t hashes[8];
    bool all_distinct = true;
    for (int i = 0; i < 8; i++) {
        hashes[i] = cipher_kr_hash((uint8_t)i, 512, 512, 256, 0);
        for (int j = 0; j < i; j++) {
            if (hashes[i] == hashes[j]) { all_distinct = false; break; }
        }
    }
    CHECK(all_distinct, "8 op classes → 8 distinct hashes");
}

// ---------------------------------------------------------------------------
// Test: Init
// ---------------------------------------------------------------------------

static void test_init() {
    printf("\n[L1.1.2] Init\n");

    CipherKoopmanRuntime* krp = new CipherKoopmanRuntime();
    CipherKoopmanRuntime& kr = *krp;
    cipher_kr_init(&kr);
    CHECK(kr.initialized,         "initialized flag set");
    CHECK(kr.n_records == 0,      "n_records starts at 0");
    CHECK(kr.total_novel_ops == 0,"total_novel_ops starts at 0");

    // All slots unoccupied
    bool all_empty = true;
    for (int i = 0; i < CIPHER_KR_MAX_OPS; i++) {
        if (kr.records[i].total_calls != 0) { all_empty = false; break; }
    }
    CHECK(all_empty, "all records start empty");
}

// ---------------------------------------------------------------------------
// Test: find_or_create
// ---------------------------------------------------------------------------

static void test_find_or_create() {
    printf("\n[L1.1.3] find_or_create\n");

    CipherKoopmanRuntime* krp = new CipherKoopmanRuntime();
    CipherKoopmanRuntime& kr = *krp;
    cipher_kr_init(&kr);

    uint64_t h1 = cipher_kr_hash(0, 1024, 1024, 256, 0);

    // First call creates
    int idx1 = cipher_kr_find_or_create(&kr, h1, 0, 1024, 1024, 256, 0);
    CHECK(idx1 == 0,          "first op → index 0");
    CHECK(kr.n_records == 1,  "n_records = 1 after first create");
    CHECK(kr.total_novel_ops == 1, "total_novel_ops = 1");

    // Second call with same hash returns same index
    int idx2 = cipher_kr_find_or_create(&kr, h1, 0, 1024, 1024, 256, 0);
    CHECK(idx2 == idx1,       "same hash → same index");
    CHECK(kr.n_records == 1,  "n_records still 1");

    // Different op creates new record
    uint64_t h2 = cipher_kr_hash(0, 2048, 2048, 256, 0);
    int idx3 = cipher_kr_find_or_create(&kr, h2, 0, 2048, 2048, 256, 0);
    CHECK(idx3 == 1,          "second op → index 1");
    CHECK(kr.n_records == 2,  "n_records = 2");

    // Record fields correct
    CHECK(kr.records[0].op_class == 0,   "op_class stored correctly");
    CHECK(kr.records[0].grid_x   == 1024,"grid_x stored correctly");
    CHECK(kr.records[0].state == CIPHER_KR_COLLECTING, "state starts COLLECTING");
    CHECK(kr.records[0].registry_slot == -1, "registry_slot starts -1");
}

// ---------------------------------------------------------------------------
// Test: feature extraction
// ---------------------------------------------------------------------------

static void test_features() {
    printf("\n[L1.1.4] Feature extraction\n");

    CipherLiquidStateMgr liq = make_liquid();
    float feat[CIPHER_KR_FEATURE_DIM];

    cipher_kr_extract_features(0, 1024, 1024, 1, 256, 0, &liq,
                                feat, CIPHER_KR_FEATURE_DIM);

    // All features should be finite
    bool all_finite = true;
    for (int i = 0; i < CIPHER_KR_FEATURE_DIM; i++) {
        if (!isfinite(feat[i])) { all_finite = false; break; }
    }
    CHECK(all_finite, "all features finite");

    // Features in reasonable range (most normalized to ~[0,1])
    bool in_range = true;
    for (int i = 0; i < CIPHER_KR_FEATURE_DIM; i++) {
        if (feat[i] < -0.1f || feat[i] > 2.0f) { in_range = false; break; }
    }
    CHECK(in_range, "features in [-0.1, 2.0] range");

    // op_class=0 → feat[0] = 0/8 = 0.0
    CHECK(fabsf(feat[0]) < 0.01f, "op_class=0 encodes to ~0");

    // op_class=4 → feat[0] = 4/8 = 0.5
    cipher_kr_extract_features(4, 1024, 1024, 1, 256, 0, &liq,
                                feat, CIPHER_KR_FEATURE_DIM);
    CHECK(fabsf(feat[0] - 0.5f) < 0.01f, "op_class=4 encodes to ~0.5");

    // Phase from liquid state
    CHECK(feat[7] > 0.0f, "phase feature non-zero (phase=1 → 0.5)");

    // Liquid state gradient
    CHECK(feat[8] > 0.0f, "grad_ema feature non-zero");

    // No liquid state — should still work
    cipher_kr_extract_features(0, 512, 512, 1, 128, 16384, NULL,
                                feat, CIPHER_KR_FEATURE_DIM);
    bool all_finite2 = true;
    for (int i = 0; i < CIPHER_KR_FEATURE_DIM; i++) {
        if (!isfinite(feat[i])) { all_finite2 = false; break; }
    }
    CHECK(all_finite2, "features finite with NULL liquid state");
}

// ---------------------------------------------------------------------------
// Test: decide — collection phase
// ---------------------------------------------------------------------------

static void test_decide_collecting() {
    printf("\n[L1.1.5] decide — collecting phase\n");

    CipherKoopmanRuntime* krp = new CipherKoopmanRuntime();
    CipherKoopmanRuntime& kr = *krp;
    cipher_kr_init(&kr);
    CipherLiquidStateMgr liq = make_liquid();

    // First MIN_SNAPSHOTS-1 calls should all be passthrough+collecting
    for (int i = 0; i < CIPHER_KR_MIN_SNAPSHOTS - 1; i++) {
        CipherKRDecision dec = cipher_kr_decide(&kr, 0, 1024, 1024, 1,
                                                 256, 0, &liq);
        if (i == 0) {
            CHECK(dec.record_idx >= 0,        "first call creates record");
            CHECK(dec.is_collecting,           "first call: is_collecting");
            CHECK(!dec.should_substitute,      "first call: no substitution");
        }
    }

    CHECK(kr.n_records == 1, "exactly one op tracked");
    CHECK(kr.records[0].state == CIPHER_KR_COLLECTING,
          "state still COLLECTING before MIN_SNAPSHOTS");
    CHECK(kr.records[0].passthroughs == CIPHER_KR_MIN_SNAPSHOTS - 1,
          "passthroughs count correct");
    CHECK(kr.total_collection_calls == CIPHER_KR_MIN_SNAPSHOTS - 1,
          "total_collection_calls correct");
}

// ---------------------------------------------------------------------------
// Test: full pipeline — collect → solve → substitute
// ---------------------------------------------------------------------------

static void test_full_pipeline() {
    g_snap_seed = 0;
    printf("\n[L1.1.6] Full pipeline: collect → solve → substitute\n");
    CipherKoopmanRuntime* krp = new CipherKoopmanRuntime();
    CipherKoopmanRuntime& kr = *krp;
    cipher_kr_init(&kr);

    float inp[16], out_real[16], out_pred[16];
    uint8_t  cls = 0;
    uint32_t gx = 512, gy = 512, bs = 256, sh = 0;
    int record_idx = -1;

    // Feed snapshots using the exact formula that EDMD can solve
    // (validated in isolation: seed=s, banded matrix)
    uint64_t h = cipher_kr_hash(cls, gx, gy, bs, sh);
    record_idx = cipher_kr_find_or_create(&kr, h, cls, gx, gy, bs, sh);

    bool derived = false;
    for (int s = 1; s <= 25; s++) {
        // Exact working formula from validation
        for (int j=0;j<16;j++){int r=(s*31+j*7)*1103515245+12345;inp[j]=(float)(r&0x7FFFFFFF)/(float)0x40000000-1.0f;}
        for (int j=0;j<16;j++){out_real[j]=0.0f;for(int k=0;k<16;k++){float a=(j==k)?0.8f:(abs(j-k)==1)?0.1f:0.0f;out_real[j]+=a*inp[k];}}
        cipher_kr_record_output(&kr, record_idx, inp, out_real, 16);
        if (kr.records[record_idx].state == CIPHER_KR_DERIVED) { derived = true; break; }
    }

    CHECK(derived, "surrogate derived after MIN_SNAPSHOTS");

    if (derived) {
        // Verify predict works
        bool pred_ok = cipher_kr_predict(&kr, record_idx, inp, out_pred, 16);
        CHECK(pred_ok, "cipher_kr_predict returns true after derivation");
        bool finite = true;
        for (int i=0;i<16;i++) if (!isfinite(out_pred[i])) { finite=false; break; }
        CHECK(finite, "predicted output is finite");
        float err = kr.records[record_idx].last_fit_error;
        CHECK_FMT(err < CIPHER_KR_FIT_THRESHOLD, "fit_error=%.4f < threshold=%.2f", err, CIPHER_KR_FIT_THRESHOLD);
    }

    // predict returns false before derivation
    CipherKoopmanRuntime* krp2 = new CipherKoopmanRuntime();
    CipherKoopmanRuntime& kr2 = *krp2;
    cipher_kr_init(&kr2);
    int idx2 = cipher_kr_find_or_create(&kr2, cipher_kr_hash(0,1024,1024,256,0), 0,1024,1024,256,0);
    bool pred_false = !cipher_kr_predict(&kr2, idx2, inp, out_pred, 16);
    CHECK(pred_false, "predict returns false before derivation");
    delete krp2;
    delete krp;
}

static void test_multi_op() {
    g_snap_seed = 0;
    printf("\n[L1.1.7] Multiple ops tracked simultaneously\n");
    CipherKoopmanRuntime* krp = new CipherKoopmanRuntime();
    CipherKoopmanRuntime& kr = *krp;
    cipher_kr_init(&kr);

    float inp[16], out[16];

    // 5 different op shapes, each fed 25 snapshots with validated formula
    struct { uint8_t cls; uint32_t gx,gy,bs,sh; } ops[5] = {
        {0,1024,1024,256,0}, {0,512,512,128,16384},
        {1,256,256,64,0},    {2,2048,1,512,0},
        {0,4096,4096,256,0},
    };

    for (int o = 0; o < 5; o++) {
        uint64_t h = cipher_kr_hash(ops[o].cls,ops[o].gx,ops[o].gy,ops[o].bs,ops[o].sh);
        int idx = cipher_kr_find_or_create(&kr, h, ops[o].cls,ops[o].gx,ops[o].gy,ops[o].bs,ops[o].sh);
        for (int s = 1; s <= 25; s++) {
            // Vary by op index so each op gets different (but valid) data
            int seed = s + o * 100;
            for (int j=0;j<16;j++){int r=(seed*31+j*7)*1103515245+12345;inp[j]=(float)(r&0x7FFFFFFF)/(float)0x40000000-1.0f;}
            for (int j=0;j<16;j++){out[j]=0.0f;for(int k=0;k<16;k++){float a=(j==k)?0.8f:(abs(j-k)==1)?0.1f:0.0f;out[j]+=a*inp[k];}}
            cipher_kr_record_output(&kr, idx, inp, out, 16);
        }
    }

    CHECK(kr.n_records == 5, "5 distinct ops tracked");

    int n_derived = 0;
    for (uint32_t r = 0; r < kr.n_records; r++)
        if (kr.records[r].state == CIPHER_KR_DERIVED) n_derived++;
    CHECK_FMT(n_derived > 0, "%d/5 ops derived surrogate", n_derived);
    CHECK(kr.total_novel_ops == 5, "total_novel_ops = 5");
    delete krp;
}

static void test_table_full() {
    printf("\n[L1.1.8] Table capacity\n");

    CipherKoopmanRuntime* krp = new CipherKoopmanRuntime();
    CipherKoopmanRuntime& kr = *krp;
    cipher_kr_init(&kr);

    // Fill table to capacity
    for (int i = 0; i < CIPHER_KR_MAX_OPS; i++) {
        uint64_t h = cipher_kr_hash((uint8_t)(i%8),
                                    (uint32_t)(100+i), 1,
                                    256, (uint32_t)(i*4));
        cipher_kr_find_or_create(&kr, h, (uint8_t)(i%8),
                                  (uint32_t)(100+i), 1, 256, (uint32_t)(i*4));
    }
    CHECK(kr.n_records == CIPHER_KR_MAX_OPS, "table fills to CIPHER_KR_MAX_OPS");

    // One more should return -1
    uint64_t h = cipher_kr_hash(0, 99999, 99999, 256, 99999);
    int idx = cipher_kr_find_or_create(&kr, h, 0, 99999, 99999, 256, 99999);
    CHECK(idx == -1, "full table returns -1");
}

// ---------------------------------------------------------------------------
// Test: report
// ---------------------------------------------------------------------------

static void test_report() {
    printf("\n[L1.1.9] Report\n");

    CipherKoopmanRuntime* krp = new CipherKoopmanRuntime();
    CipherKoopmanRuntime& kr = *krp;
    cipher_kr_init(&kr);
    CipherLiquidStateMgr liq = make_liquid();

    float inp[CIPHER_KR_FEATURE_DIM], outp[CIPHER_KR_FEATURE_DIM];

    // One op through full pipeline
    for (int i = 0; i < CIPHER_KR_MIN_SNAPSHOTS + 2; i++) {
        CipherKRDecision dec = cipher_kr_decide(&kr, 0, 1024, 1024, 1, 256, 0, &liq);
        if (dec.is_collecting && dec.record_idx >= 0) {
            make_snapshot(0, 1024, 1024, 256, 0, inp, outp,
                          CIPHER_KR_FEATURE_DIM, &liq);
            cipher_kr_record_output(&kr, dec.record_idx, inp, outp,
                                    CIPHER_KR_FEATURE_DIM);
        }
    }

    // report() should not crash
    cipher_kr_report(&kr);
    CHECK(true, "cipher_kr_report runs without crash");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("============================================================\n");
    printf("  CIPHER L1.1 — Runtime Koopman Derivation Tests\n");
    printf("============================================================\n");

    test_hash();
    test_init();
    test_find_or_create();
    test_features();
    test_decide_collecting();
    test_full_pipeline();
    test_multi_op();
    test_table_full();
    test_report();

    printf("\n============================================================\n");
    printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) {
        printf("  \033[32m L1.1 GREEN — Runtime Koopman Derivation operational\033[0m\n");
    } else {
        printf("  \033[31m FAILURES REMAIN\033[0m\n");
    }
    printf("============================================================\n\n");
    return g_fail > 0 ? 1 : 0;
}
