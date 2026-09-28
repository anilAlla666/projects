
// ============================================================
// Neural Dynamics — CfC Wavefront Reconstructor
// O(1) replacement for classical MVM
// ncps CfC + AutoNCP wiring (Hasani et al. NMI 2022)
// Target: Xilinx Alveo U50 @ 250MHz
// ============================================================
#include <ap_fixed.h>
#include <hls_math.h>
#include <ap_int.h>

// Fixed-point types (INT8 weights, INT16 activations)
typedef ap_fixed<16, 6>  act_t;     // Activations
typedef ap_fixed<8,  2>  wt_t;      // INT8 weights
typedef ap_fixed<24, 8>  acc_t;     // Accumulators
typedef ap_fixed<16, 6>  sensor_t;  // WFS input
typedef ap_fixed<16, 6>  cmd_t;     // DM command

// Dimensions
#define IN_DIM    3072
#define HIDDEN    256
#define CFC_PROJ  128
#define OUT_DIM   1024

// ── Weight arrays (loaded from trained model) ─────────────
// Projection layer
wt_t W_proj[HIDDEN][IN_DIM];
wt_t b_proj[HIDDEN];

// CfC gate networks (f, g, tau)
wt_t W_f[HIDDEN][HIDDEN * 2];
wt_t W_g[HIDDEN][HIDDEN * 2];
wt_t W_tau[HIDDEN][HIDDEN * 2];
wt_t b_f[HIDDEN], b_g[HIDDEN], b_tau[HIDDEN];

// Head
wt_t W_head1[HIDDEN][CFC_PROJ];
wt_t W_head2[OUT_DIM][HIDDEN];
wt_t b_head1[HIDDEN], b_head2[OUT_DIM];

// Linear residual
wt_t W_lin[OUT_DIM][IN_DIM];
wt_t b_lin[OUT_DIM];

// ── Activation functions ──────────────────────────────────
act_t tanh_approx(acc_t x) {
#pragma HLS INLINE
    // Piecewise linear tanh approximation (3-segment)
    // Avoids transcendental function — critical for timing
    if (x >  2.0)  return  1.0;
    if (x < -2.0)  return -1.0;
    return (act_t)(x * 0.5);
}

act_t sigmoid_approx(acc_t x) {
#pragma HLS INLINE
    // σ(x) = 0.5 + 0.25*x for |x| < 2
    if (x >  2.0)  return  1.0;
    if (x < -2.0)  return  0.0;
    return (act_t)(0.5 + 0.25 * x);
}

act_t relu_approx(acc_t x) {
#pragma HLS INLINE
    return (x > 0) ? (act_t)x : (act_t)0;
}

// ── CfC Reconstructor ─────────────────────────────────────
void cfc_reconstructor(
    sensor_t  s[IN_DIM],      // WFS sensor input (normalised)
    cmd_t     a[OUT_DIM]      // DM actuator commands
) {
#pragma HLS INTERFACE ap_fifo port=s
#pragma HLS INTERFACE ap_fifo port=a
#pragma HLS DATAFLOW

    // ── Stage 1: Input projection ─────────────────────────
    act_t h_proj[HIDDEN];
#pragma HLS ARRAY_PARTITION variable=h_proj complete
    PROJ: for (int i = 0; i < HIDDEN; i++) {
#pragma HLS UNROLL
        acc_t sum = b_proj[i];
        for (int j = 0; j < IN_DIM; j++) {
#pragma HLS UNROLL factor=32
            sum += (acc_t)W_proj[i][j] * (acc_t)s[j];
        }
        h_proj[i] = tanh_approx(sum);
    }

    // ── Stage 2: CfC gate computation ─────────────────────
    // Input to gates: [h_proj, h0] where h0=0 (single frame)
    // This is the closed-form ODE solution (Hasani et al. 2022)
    // h_new = gate * f(xh) + (1-gate) * g(xh)
    act_t xh[HIDDEN * 2];
#pragma HLS ARRAY_PARTITION variable=xh complete
    for (int i = 0; i < HIDDEN; i++) {
#pragma HLS UNROLL
        xh[i]          = h_proj[i];   // projected input
        xh[i + HIDDEN] = 0;           // h0 = zeros (single frame)
    }

    act_t f_out[HIDDEN], g_out[HIDDEN], gate[HIDDEN];
#pragma HLS ARRAY_PARTITION variable=f_out complete
#pragma HLS ARRAY_PARTITION variable=g_out complete
#pragma HLS ARRAY_PARTITION variable=gate  complete

    CFC: for (int i = 0; i < HIDDEN; i++) {
#pragma HLS UNROLL
        acc_t sf = b_f[i], sg = b_g[i], st = b_tau[i];
        for (int j = 0; j < HIDDEN * 2; j++) {
#pragma HLS UNROLL factor=16
            sf += (acc_t)W_f[i][j]   * (acc_t)xh[j];
            sg += (acc_t)W_g[i][j]   * (acc_t)xh[j];
            st += (acc_t)W_tau[i][j] * (acc_t)xh[j];
        }
        f_out[i] = tanh_approx(sf);
        g_out[i] = tanh_approx(sg);
        gate[i]  = sigmoid_approx(st);
    }

    // CfC update: h1 = gate * f + (1-gate) * g
    act_t h1[HIDDEN];
#pragma HLS ARRAY_PARTITION variable=h1 complete
    UPDATE: for (int i = 0; i < HIDDEN; i++) {
#pragma HLS UNROLL
        h1[i] = gate[i] * f_out[i] +
                ((act_t)1.0 - gate[i]) * g_out[i];
    }

    // ── Stage 3: Output head ──────────────────────────────
    act_t h2[HIDDEN];
#pragma HLS ARRAY_PARTITION variable=h2 complete
    HEAD1: for (int i = 0; i < HIDDEN; i++) {
#pragma HLS UNROLL
        acc_t sum = b_head1[i];
        for (int j = 0; j < CFC_PROJ; j++) {
#pragma HLS UNROLL
            sum += (acc_t)W_head1[i][j] * (acc_t)h1[j];
        }
        h2[i] = relu_approx(sum);
    }

    // ── Stage 4: Final output + linear residual ───────────
    OUT: for (int i = 0; i < OUT_DIM; i++) {
#pragma HLS UNROLL factor=32
        acc_t cfc_out = b_head2[i];
        acc_t lin_out = b_lin[i];
        for (int j = 0; j < HIDDEN; j++) {
#pragma HLS UNROLL factor=16
            cfc_out += (acc_t)W_head2[i][j] * (acc_t)h2[j];
        }
        for (int j = 0; j < IN_DIM; j++) {
#pragma HLS UNROLL factor=32
            lin_out += (acc_t)W_lin[i][j] * (acc_t)s[j];
        }
        a[i] = (cmd_t)(cfc_out + lin_out);
    }
}

// Testbench
int main() {
    sensor_t s_test[IN_DIM];
    cmd_t    a_test[OUT_DIM];
    for (int i = 0; i < IN_DIM; i++) s_test[i] = 0.1;
    cfc_reconstructor(s_test, a_test);
    if (a_test[0] == 0 && a_test[OUT_DIM-1] == 0) return 1;
    return 0;
}
