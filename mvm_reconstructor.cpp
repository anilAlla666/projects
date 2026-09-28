
// ============================================================
// Neural Dynamics — Classical MVM Wavefront Reconstructor
// Baseline: O(N²) zonal least-squares
// Target: Xilinx Alveo U50 @ 250MHz
// ============================================================
#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_int.h>

// Fixed-point types
typedef ap_fixed<16, 6> sensor_t;    // WFS slope input
typedef ap_fixed<8,  2> weight_t;    // INT8 reconstructor matrix
typedef ap_fixed<24, 8> accum_t;     // Accumulator (wider)
typedef ap_fixed<16, 6> command_t;   // DM actuator command

// Dimensions
#define IN_DIM  3072
#define OUT_DIM 1024

// Reconstructor matrix (pre-calibrated, stored in BRAM)
// In production: loaded from AO calibration routine
weight_t R[OUT_DIM][IN_DIM];

void mvm_reconstructor(
    sensor_t  s[IN_DIM],      // WFS sensor input
    command_t a[OUT_DIM]      // DM actuator commands
) {
#pragma HLS INTERFACE ap_fifo port=s
#pragma HLS INTERFACE ap_fifo port=a
#pragma HLS ARRAY_PARTITION variable=R complete dim=2
#pragma HLS PIPELINE II=1

    // Matrix-vector multiply: a = R @ s
    // This is the O(N²) operation we are replacing
    OUTER: for (int i = 0; i < OUT_DIM; i++) {
#pragma HLS UNROLL
        accum_t sum = 0;
        INNER: for (int j = 0; j < IN_DIM; j++) {
#pragma HLS UNROLL
            sum += (accum_t)(R[i][j]) * (accum_t)(s[j]);
        }
        a[i] = (command_t)sum;
    }
}

// Testbench
int main() {
    sensor_t  s_test[IN_DIM];
    command_t a_test[OUT_DIM];

    // Fill with test pattern
    for (int i = 0; i < IN_DIM; i++) s_test[i] = 0.1;

    mvm_reconstructor(s_test, a_test);

    // Check output is non-zero
    if (a_test[0] == 0 && a_test[OUT_DIM-1] == 0) {
        return 1;  // FAIL
    }
    return 0;  // PASS
}
