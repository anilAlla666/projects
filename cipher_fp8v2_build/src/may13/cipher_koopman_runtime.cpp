// =============================================================================
// CIPHER — L1.1: Runtime Koopman Derivation
// cipher_koopman_runtime.cpp
// =============================================================================

#include "may13/cipher_koopman_runtime.h"
#include "may13/cipher_edmd.h"
#include "may13/cipher_stubs.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// FNV-1a hash over op identity tuple
// ---------------------------------------------------------------------------

uint64_t cipher_kr_hash(uint8_t  op_class,
                         uint32_t grid_x,
                         uint32_t grid_y,
                         uint32_t block_size,
                         uint32_t shmem_bytes)
{
    const uint64_t FNV_OFFSET = 14695981039346656037ULL;
    const uint64_t FNV_PRIME  = 1099511628211ULL;
    uint64_t h = FNV_OFFSET;

#define FNV_BYTE(b) h ^= (uint64_t)(b); h *= FNV_PRIME

    // Hash each field byte-by-byte
    FNV_BYTE(op_class);

    for (int i = 0; i < 4; i++) FNV_BYTE((grid_x   >> (i*8)) & 0xFF);
    for (int i = 0; i < 4; i++) FNV_BYTE((grid_y   >> (i*8)) & 0xFF);
    for (int i = 0; i < 4; i++) FNV_BYTE((block_size >> (i*8)) & 0xFF);
    for (int i = 0; i < 4; i++) FNV_BYTE((shmem_bytes >> (i*8)) & 0xFF);

#undef FNV_BYTE
    return h;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

void cipher_kr_init(CipherKoopmanRuntime* kr)
{
    if (!kr) return;
    memset(kr, 0, sizeof(*kr));
    for (int i = 0; i < CIPHER_KR_MAX_OPS; i++) {
        kr->records[i].registry_slot = -1;
        kr->records[i].state = CIPHER_KR_COLLECTING;
        // EDMD pipeline initialized lazily in find_or_create
    }
    kr->initialized = true;
}

// ---------------------------------------------------------------------------
// Find or create record for op_hash
// ---------------------------------------------------------------------------

int cipher_kr_find_or_create(CipherKoopmanRuntime* kr,
                              uint64_t op_hash,
                              uint8_t  op_class,
                              uint32_t grid_x,
                              uint32_t grid_y,
                              uint32_t block_size,
                              uint32_t shmem_bytes)
{
    if (!kr) return -1;

    // Linear scan (max 256 entries — acceptable)
    for (uint32_t i = 0; i < kr->n_records; i++) {
        if (kr->records[i].op_hash == op_hash) return (int)i;
    }

    // New op
    if (kr->n_records >= CIPHER_KR_MAX_OPS) return -1;

    int idx = (int)kr->n_records++;
    CipherKRRecord* rec = &kr->records[idx];

    rec->op_hash    = op_hash;
    rec->op_class   = op_class;
    rec->grid_x     = grid_x;
    rec->grid_y     = grid_y;
    rec->block_size = block_size;
    rec->shmem_bytes = shmem_bytes;
    rec->state      = CIPHER_KR_COLLECTING;
    rec->registry_slot = -1;

    // Human-readable name
    const char* cls_names[] = {
        "GEMM","ATTN","EW","REDUCE","CONV","NORM","EMBED","COPY","UNKNOWN"
    };
    const char* cls = (op_class < 8) ? cls_names[op_class] : cls_names[8];
    snprintf(rec->op_name, sizeof(rec->op_name),
             "%s_g%ux%u_b%u_s%u",
             cls, grid_x, grid_y, block_size, shmem_bytes);

    cipher_edmd_init(&rec->edmd, rec->op_name, CIPHER_KR_FEATURE_DIM, CIPHER_KR_DICT_SIZE);
    kr->total_novel_ops++;

    return idx;
}

// ---------------------------------------------------------------------------
// Feature extraction
//
// Maps kernel launch geometry + liquid state → normalized feature vector.
// In production: samples actual tensor memory. In stub: uses geometry proxy.
//
// Feature layout (16 dims):
//   [0]  op_class / 8.0          (normalized op class)
//   [1]  log2(grid_x) / 20.0     (grid size X)
//   [2]  log2(grid_y) / 20.0     (grid size Y)
//   [3]  log2(block_size) / 10.0 (block size)
//   [4]  log2(shmem+1) / 16.0    (shared memory)
//   [5]  grid_x * grid_y / 1M    (total blocks normalized)
//   [6]  shmem / 48KB            (shared mem fraction)
//   [7]  phase / 2.0             (training phase from liquid state)
//   [8]  global_grad_ema         (gradient signal)
//   [9]  sub_counter[0] / 100    (substitution pressure layer 0)
//   [10] hbm_bw_utilized         (memory bandwidth from liquid)
//   [11] l2_hit_rate             (cache from liquid)
//   [12] sm_idle_fraction        (idle from liquid)
//   [13] nvlink_utilization      (interconnect from liquid)
//   [14] block_size / 1024.0     (raw block size)
//   [15] (grid_x * grid_y * block_size) / 1e7  (total threads)
// ---------------------------------------------------------------------------

void cipher_kr_extract_features(
    uint8_t  op_class,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_size, uint32_t shmem_bytes,
    const CipherLiquidStateMgr* liquid,
    float*   out,
    uint32_t n_features)
{
    (void)grid_z;
    if (!out || n_features == 0) return;

    memset(out, 0, n_features * sizeof(float));
    uint32_t n = n_features < CIPHER_KR_FEATURE_DIM
               ? n_features : CIPHER_KR_FEATURE_DIM;

    // Geometry features
    float log_gx  = (grid_x   > 0) ? log2f((float)grid_x)   : 0.0f;
    float log_gy  = (grid_y   > 0) ? log2f((float)grid_y)   : 0.0f;
    float log_bs  = (block_size > 0) ? log2f((float)block_size) : 0.0f;
    float log_sh  = log2f((float)(shmem_bytes + 1));
    float total_blocks = (float)grid_x * (float)grid_y;
    float total_threads = total_blocks * (float)block_size;

    float geom[16] = {
        (float)op_class   / 8.0f,
        log_gx            / 20.0f,
        log_gy            / 20.0f,
        log_bs            / 10.0f,
        log_sh            / 16.0f,
        total_blocks      / 1000000.0f,
        (float)shmem_bytes / 49152.0f,
        0.0f,   // phase
        0.0f,   // global_grad_ema
        0.0f,   // sub_counter
        0.0f,   // hbm_bw
        0.0f,   // l2_hit
        0.0f,   // sm_idle
        0.0f,   // nvlink
        (float)block_size  / 1024.0f,
        fminf(total_threads / 1e7f, 1.0f),  // clamped to [0,1]
    };

    // Liquid state features
    if (liquid && liquid->initialized && liquid->device
        && liquid->device->_magic == CIPHER_LIQUID_STATE_MAGIC)
    {
        const CipherLiquidState* ls = liquid->device;
        geom[7]  = (float)ls->phase / 2.0f;
        geom[8]  = ls->global_grad_ema;
        geom[9]  = (float)ls->layer[0].sub_counter / 100.0f;
        geom[10] = ls->hw.hbm_bw_utilized;
        geom[11] = ls->hw.l2_hit_rate;
        geom[12] = ls->hw.sm_idle_fraction;
        geom[13] = ls->hw.nvlink_utilization;
    }

    for (uint32_t i = 0; i < n; i++) out[i] = geom[i];
}

// ---------------------------------------------------------------------------
// Attempt EDMD solve for a record.
// Called when snapshot count reaches MIN_SNAPSHOTS (and every REFINE_EVERY
// snapshots thereafter).
// Returns true if surrogate was successfully derived (fit_error < threshold).
// ---------------------------------------------------------------------------

static bool try_solve(CipherKRRecord* rec)
{
    float fit_error = cipher_edmd_solve(&rec->edmd);
    bool ok = (fit_error >= 0.0f);

    rec->last_fit_error = fit_error;

    if (ok && fit_error < CIPHER_KR_FIT_THRESHOLD) {
        if (rec->state != CIPHER_KR_DERIVED) {
            fprintf(stderr,
                "[CIPHER L1.1] Surrogate derived for %s: "
                "fit_error=%.4f snapshots=%u\n",
                rec->op_name,
                fit_error,
                rec->edmd.buffer.count);
        }
        rec->state = CIPHER_KR_DERIVED;
        return true;
    }

    if (!ok || fit_error >= 1.0f) {
        // EDMD failed completely — mark as failed
        rec->state = CIPHER_KR_FAILED;
        fprintf(stderr,
            "[CIPHER L1.1] EDMD failed for %s: "
            "fit_error=%.4f — keeping passthrough\n",
            rec->op_name, fit_error);
        return false;
    }

    // Partial fit — keep collecting, try again later
    return false;
}

// ---------------------------------------------------------------------------
// cipher_kr_decide — main dispatch entry point
// ---------------------------------------------------------------------------

CipherKRDecision cipher_kr_decide(
    CipherKoopmanRuntime*       kr,
    uint8_t                     op_class,
    uint32_t                    grid_x,
    uint32_t                    grid_y,
    uint32_t                    grid_z,
    uint32_t                    block_size,
    uint32_t                    shmem_bytes,
    const CipherLiquidStateMgr* liquid)
{
    (void)grid_z;

    CipherKRDecision dec = {false, false, -1, 0.0f, 0};

    if (!kr || !kr->initialized) return dec;

    uint64_t h   = cipher_kr_hash(op_class, grid_x, grid_y, block_size, shmem_bytes);
    int      idx = cipher_kr_find_or_create(kr, h, op_class,
                                             grid_x, grid_y, block_size, shmem_bytes);
    if (idx < 0) return dec;  // Table full — passthrough

    CipherKRRecord* rec = &kr->records[idx];
    rec->total_calls++;
    dec.record_idx    = idx;
    dec.snapshot_count = rec->edmd.buffer.count;
    dec.fit_error      = rec->last_fit_error;

    switch (rec->state) {

    case CIPHER_KR_COLLECTING: {
        // Still building snapshot buffer
        dec.is_collecting      = true;
        dec.should_substitute  = false;
        rec->passthroughs++;
        kr->total_collection_calls++;

        // Extract features for this call so caller can record output
        // (features stored in dec for caller to pass to cipher_kr_record_output)
        return dec;
    }

    case CIPHER_KR_DERIVED: {
        // Surrogate ready — substitute
        dec.should_substitute = true;
        dec.is_collecting     = false;
        rec->substitutions++;
        kr->total_surrogate_calls++;

        // Periodic online refinement
        if (rec->edmd.buffer.count % CIPHER_KR_REFINE_EVERY == 0) {
            try_solve(rec);  // Non-blocking refinement
        }
        return dec;
    }

    case CIPHER_KR_FAILED: {
        // EDMD failed — passthrough forever
        dec.should_substitute = false;
        dec.is_collecting     = false;
        rec->passthroughs++;
        return dec;
    }
    }

    return dec;
}

// ---------------------------------------------------------------------------
// cipher_kr_record_output
// Called after real kernel runs during collection phase.
// Adds a (input, output) snapshot pair to the EDMD buffer.
// Triggers solve when MIN_SNAPSHOTS is reached.
// ---------------------------------------------------------------------------

void cipher_kr_record_output(
    CipherKoopmanRuntime* kr,
    int                   record_idx,
    const float*          input_features,
    const float*          output_features,
    uint32_t              n_features)
{
    if (!kr || record_idx < 0 || record_idx >= (int)kr->n_records) return;
    if (!input_features || !output_features) return;

    CipherKRRecord* rec = &kr->records[record_idx];
    if (rec->state != CIPHER_KR_COLLECTING) return;

    // Add snapshot to EDMD buffer
    // cipher_edmd_collect auto-solves at MIN_SNAPSHOTS and every 10 after
    bool edmd_solved = cipher_edmd_collect(&rec->edmd, input_features, output_features);

    // Propagate EDMD solve result to KR state
    if (edmd_solved && rec->state == CIPHER_KR_COLLECTING) {
        rec->last_fit_error = rec->edmd.koopman.fit_error;
        if (rec->edmd.koopman.valid &&
            rec->edmd.koopman.fit_error < CIPHER_KR_FIT_THRESHOLD) {
            rec->state = CIPHER_KR_DERIVED;
            kr->total_derived++;
            fprintf(stderr,
                "[CIPHER L1.1] Surrogate derived for %s: "
                "fit_error=%.4f snapshots=%u\n",
                rec->op_name, rec->last_fit_error,
                rec->edmd.buffer.count);
        }
    } else if (!edmd_solved &&
               rec->edmd.status == 3 /* CIPHER_EDMD_FAILED */ &&
               rec->edmd.buffer.count >= CIPHER_KR_MIN_SNAPSHOTS &&
               rec->edmd.koopman.fit_error >= 1.0f) {
        // Complete EDMD failure — mark as failed
        rec->state = CIPHER_KR_FAILED;
        rec->last_fit_error = rec->edmd.koopman.fit_error;
        kr->total_failed++;
    }
}

// ---------------------------------------------------------------------------
// cipher_kr_predict
// Apply the derived Koopman surrogate to predict output from input features.
// ---------------------------------------------------------------------------

bool cipher_kr_predict(
    CipherKoopmanRuntime* kr,
    int                   record_idx,
    const float*          input_features,
    float*                output_features,
    uint32_t              n_features)
{
    if (!kr || record_idx < 0 || record_idx >= (int)kr->n_records) return false;

    CipherKRRecord* rec = &kr->records[record_idx];
    if (rec->state != CIPHER_KR_DERIVED) return false;
    if (!input_features || !output_features) return false;

    return cipher_edmd_predict(&rec->edmd, input_features, output_features);
}

// ---------------------------------------------------------------------------
// cipher_kr_report
// ---------------------------------------------------------------------------

void cipher_kr_report(const CipherKoopmanRuntime* kr)
{
    if (!kr) return;
    fprintf(stderr, "\n[CIPHER L1.1] Runtime Koopman Report\n");
    fprintf(stderr, "  Novel ops tracked:    %u / %u\n",
            kr->n_records, CIPHER_KR_MAX_OPS);
    fprintf(stderr, "  Unique shapes seen:   %lu\n", kr->total_novel_ops);
    fprintf(stderr, "  Derived surrogates:   %lu\n", kr->total_derived);
    fprintf(stderr, "  Failed derivations:   %lu\n", kr->total_failed);
    fprintf(stderr, "  Collection calls:     %lu\n", kr->total_collection_calls);
    fprintf(stderr, "  Surrogate calls:      %lu\n", kr->total_surrogate_calls);

    if (kr->total_surrogate_calls + kr->total_collection_calls > 0) {
        float sub_rate = (float)kr->total_surrogate_calls /
                         (float)(kr->total_surrogate_calls +
                                 kr->total_collection_calls);
        fprintf(stderr, "  Substitution rate:    %.1f%%\n", sub_rate * 100.0f);
    }

    fprintf(stderr, "\n  Per-op breakdown:\n");
    for (uint32_t i = 0; i < kr->n_records; i++) {
        const CipherKRRecord* rec = &kr->records[i];
        const char* state_str =
            (rec->state == CIPHER_KR_DERIVED)    ? "DERIVED"   :
            (rec->state == CIPHER_KR_FAILED)     ? "FAILED"    :
                                                    "COLLECTING";
        fprintf(stderr,
            "    [%2u] %-36s  %s  snaps=%-3u  err=%.4f  "
            "calls=%-6lu  subs=%-6lu\n",
            i, rec->op_name, state_str,
            rec->edmd.buffer.count, rec->last_fit_error,
            rec->total_calls, rec->substitutions);
    }
}
