/*
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║  S.O.M.A. v7 — C SDK                                           ║
 * ║  Self-Optimizing Motor Architecture                              ║
 * ║  Neural Dynamics Inc.                                            ║
 * ║                                                                  ║
 * ║  Zero dependencies. Pure C99. Runs on anything.                  ║
 * ║  Compiles with: gcc -O2 -lm soma.c -o soma                      ║
 * ║                                                                  ║
 * ║  API:                                                            ║
 * ║    SomaHandle* soma_init(const char* weights_dir);               ║
 * ║    void soma_step(SomaHandle* h, const float* state,             ║
 * ║                   float dt, float* torque_out);                  ║
 * ║    void soma_reset(SomaHandle* h);                               ║
 * ║    void soma_free(SomaHandle* h);                                ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */

#ifndef SOMA_H
#define SOMA_H

#include <stddef.h>

/* --- Configuration (matches training) --- */
#define SOMA_INPUT_DIM    28
#define SOMA_OUTPUT_DIM   7
#define SOMA_INTER_SIZE   128
#define SOMA_CMD_SIZE     64
#define SOMA_MOTOR_SIZE   16
#define SOMA_HIDDEN_SIZE  (SOMA_INTER_SIZE + SOMA_CMD_SIZE + SOMA_MOTOR_SIZE)  /* 208 */

/* --- Opaque handle --- */
typedef struct SomaHandle SomaHandle;

/* --- API --- */
SomaHandle* soma_init(const char* weights_dir);
void        soma_step(SomaHandle* h, const float* state, float dt, float* torque_out);
void        soma_reset(SomaHandle* h);
void        soma_free(SomaHandle* h);

/* --- Optional: get hidden state for inspection --- */
const float* soma_get_hidden(const SomaHandle* h);
int          soma_get_hidden_size(void);

#endif /* SOMA_H */


#ifdef SOMA_IMPLEMENTATION
/* ================================================================
 *  IMPLEMENTATION — include this in exactly ONE .c file:
 *    #define SOMA_IMPLEMENTATION
 *    #include "soma.h"
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- Weight dimensions (derived from architecture) ---
 *
 *  CfC cell input sizes:
 *    inter_cell:   input_proj(128) + h_inter(128) + dt(1) = 257
 *    command_cell: mask_output(64) + h_cmd(64) + dt(1)    = 129
 *    motor_cell:   mask_output(16) + h_motor(16) + dt(1)  = 33
 *
 *  Each cell has ff and gg linear layers: W(out, in) + b(out)
 */

/* Inter cell */
#define INTER_CELL_IN   (SOMA_INTER_SIZE + SOMA_INTER_SIZE + 1)   /* 257 */
/* Command cell */
#define CMD_CELL_IN     (SOMA_CMD_SIZE + SOMA_CMD_SIZE + 1)       /* 129 */
/* Motor cell */
#define MOTOR_CELL_IN   (SOMA_MOTOR_SIZE + SOMA_MOTOR_SIZE + 1)   /* 33 */

struct SomaHandle {
    /* Hidden state (persistent across steps) */
    float hidden[SOMA_HIDDEN_SIZE];

    /* State normalizer */
    float norm_mean[SOMA_INPUT_DIM];
    float norm_std[SOMA_INPUT_DIM];

    /* Input projection: (INTER, INPUT) */
    float input_w[SOMA_INTER_SIZE * SOMA_INPUT_DIM];
    float input_b[SOMA_INTER_SIZE];

    /* Inter cell */
    float inter_ff_w[SOMA_INTER_SIZE * INTER_CELL_IN];
    float inter_ff_b[SOMA_INTER_SIZE];
    float inter_gg_w[SOMA_INTER_SIZE * INTER_CELL_IN];
    float inter_gg_b[SOMA_INTER_SIZE];

    /* Command cell */
    float cmd_ff_w[SOMA_CMD_SIZE * CMD_CELL_IN];
    float cmd_ff_b[SOMA_CMD_SIZE];
    float cmd_gg_w[SOMA_CMD_SIZE * CMD_CELL_IN];
    float cmd_gg_b[SOMA_CMD_SIZE];

    /* Motor cell */
    float motor_ff_w[SOMA_MOTOR_SIZE * MOTOR_CELL_IN];
    float motor_ff_b[SOMA_MOTOR_SIZE];
    float motor_gg_w[SOMA_MOTOR_SIZE * MOTOR_CELL_IN];
    float motor_gg_b[SOMA_MOTOR_SIZE];

    /* Layer norms: weight and bias for each layer */
    float ln_inter_w[SOMA_INTER_SIZE];
    float ln_inter_b[SOMA_INTER_SIZE];
    float ln_cmd_w[SOMA_CMD_SIZE];
    float ln_cmd_b[SOMA_CMD_SIZE];
    float ln_motor_w[SOMA_MOTOR_SIZE];
    float ln_motor_b[SOMA_MOTOR_SIZE];

    /* Output projection: (OUTPUT, MOTOR) */
    float output_w[SOMA_OUTPUT_DIM * SOMA_MOTOR_SIZE];
    float output_b[SOMA_OUTPUT_DIM];

    /* Sparse masks */
    float mask_ic[SOMA_INTER_SIZE * SOMA_CMD_SIZE];    /* (128, 64) — inter_to_command transposed */
    float mask_cm[SOMA_CMD_SIZE * SOMA_MOTOR_SIZE];    /* (64, 16)  — command_to_motor transposed */
};


/* --- Internal helpers --- */

static float soma_sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* dst[M] = W[M x N] @ src[N] + bias[M] */
static void soma_matvec(float* dst, const float* W, const float* src, 
                        const float* bias, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = bias[i];
        const float* row = W + i * N;
        for (int j = 0; j < N; j++) {
            sum += row[j] * src[j];
        }
        dst[i] = sum;
    }
}

/* dst[M] = src[N] @ W[N x M]  (no bias, used for sparse mask multiply) */
static void soma_vecmat(float* dst, const float* src, const float* W, int N, int M) {
    for (int j = 0; j < M; j++) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += src[i] * W[i * M + j];
        }
        dst[j] = sum;
    }
}

/* In-place layer norm: x[N] = (x - mean) / sqrt(var + eps) * w + b */
static void soma_layer_norm(float* x, const float* w, const float* b, int N) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < N; i++) mean += x[i];
    mean /= (float)N;
    for (int i = 0; i < N; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)N;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < N; i++) {
        x[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }
}

/* CfC cell: h_new = (1-f)*h + f*g where f=sigmoid(ff(xhd)), g=tanh(gg(xhd))
 * xhd = concat(x, h, dt)
 * Operates in-place on h_out (can alias h_in if desired via temp buffer)
 */
static void soma_cfc_cell(float* h_out,
                          const float* x, int x_size,
                          const float* h_in, int h_size,
                          float dt,
                          const float* ff_w, const float* ff_b,
                          const float* gg_w, const float* gg_b) {
    /* Build xhd = [x; h; dt] */
    int xhd_size = x_size + h_size + 1;
    float xhd[512]; /* max size is 257, well within bounds */
    memcpy(xhd, x, x_size * sizeof(float));
    memcpy(xhd + x_size, h_in, h_size * sizeof(float));
    xhd[x_size + h_size] = dt;

    /* f = sigmoid(ff_w @ xhd + ff_b) */
    float f_vec[256]; /* max h_size is 128 */
    soma_matvec(f_vec, ff_w, xhd, ff_b, h_size, xhd_size);
    for (int i = 0; i < h_size; i++) f_vec[i] = soma_sigmoidf(f_vec[i]);

    /* g = tanh(gg_w @ xhd + gg_b) */
    float g_vec[256];
    soma_matvec(g_vec, gg_w, xhd, gg_b, h_size, xhd_size);
    for (int i = 0; i < h_size; i++) g_vec[i] = tanhf(g_vec[i]);

    /* h_new = (1-f)*h + f*g */
    for (int i = 0; i < h_size; i++) {
        h_out[i] = (1.0f - f_vec[i]) * h_in[i] + f_vec[i] * g_vec[i];
    }
}

/* --- File loading helpers --- */

static int soma_load_bin(const char* path, float* dst, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "SOMA: cannot open %s\n", path);
        return -1;
    }
    size_t read = fread(dst, sizeof(float), count, f);
    fclose(f);
    if ((int)read != count) {
        fprintf(stderr, "SOMA: %s expected %d floats, got %d\n", path, count, (int)read);
        return -1;
    }
    return 0;
}

/* Load a weight file from weights_dir/filename into dst */
static int soma_load_weight(const char* dir, const char* filename, float* dst, int count) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, filename);
    return soma_load_bin(path, dst, count);
}


/* --- Public API implementation --- */

SomaHandle* soma_init(const char* weights_dir) {
    SomaHandle* h = (SomaHandle*)calloc(1, sizeof(SomaHandle));
    if (!h) return NULL;

    int err = 0;

    /* Input projection */
    err |= soma_load_weight(weights_dir, "input_proj_weight.bin", h->input_w, 
                            SOMA_INTER_SIZE * SOMA_INPUT_DIM);
    err |= soma_load_weight(weights_dir, "input_proj_bias.bin", h->input_b, 
                            SOMA_INTER_SIZE);

    /* Inter cell */
    err |= soma_load_weight(weights_dir, "inter_cell_ff_weight.bin", h->inter_ff_w,
                            SOMA_INTER_SIZE * INTER_CELL_IN);
    err |= soma_load_weight(weights_dir, "inter_cell_ff_bias.bin", h->inter_ff_b,
                            SOMA_INTER_SIZE);
    err |= soma_load_weight(weights_dir, "inter_cell_gg_weight.bin", h->inter_gg_w,
                            SOMA_INTER_SIZE * INTER_CELL_IN);
    err |= soma_load_weight(weights_dir, "inter_cell_gg_bias.bin", h->inter_gg_b,
                            SOMA_INTER_SIZE);

    /* Command cell */
    err |= soma_load_weight(weights_dir, "command_cell_ff_weight.bin", h->cmd_ff_w,
                            SOMA_CMD_SIZE * CMD_CELL_IN);
    err |= soma_load_weight(weights_dir, "command_cell_ff_bias.bin", h->cmd_ff_b,
                            SOMA_CMD_SIZE);
    err |= soma_load_weight(weights_dir, "command_cell_gg_weight.bin", h->cmd_gg_w,
                            SOMA_CMD_SIZE * CMD_CELL_IN);
    err |= soma_load_weight(weights_dir, "command_cell_gg_bias.bin", h->cmd_gg_b,
                            SOMA_CMD_SIZE);

    /* Motor cell */
    err |= soma_load_weight(weights_dir, "motor_cell_ff_weight.bin", h->motor_ff_w,
                            SOMA_MOTOR_SIZE * MOTOR_CELL_IN);
    err |= soma_load_weight(weights_dir, "motor_cell_ff_bias.bin", h->motor_ff_b,
                            SOMA_MOTOR_SIZE);
    err |= soma_load_weight(weights_dir, "motor_cell_gg_weight.bin", h->motor_gg_w,
                            SOMA_MOTOR_SIZE * MOTOR_CELL_IN);
    err |= soma_load_weight(weights_dir, "motor_cell_gg_bias.bin", h->motor_gg_b,
                            SOMA_MOTOR_SIZE);

    /* Layer norms */
    err |= soma_load_weight(weights_dir, "ln_inter_weight.bin", h->ln_inter_w, SOMA_INTER_SIZE);
    err |= soma_load_weight(weights_dir, "ln_inter_bias.bin", h->ln_inter_b, SOMA_INTER_SIZE);
    err |= soma_load_weight(weights_dir, "ln_command_weight.bin", h->ln_cmd_w, SOMA_CMD_SIZE);
    err |= soma_load_weight(weights_dir, "ln_command_bias.bin", h->ln_cmd_b, SOMA_CMD_SIZE);
    err |= soma_load_weight(weights_dir, "ln_motor_weight.bin", h->ln_motor_w, SOMA_MOTOR_SIZE);
    err |= soma_load_weight(weights_dir, "ln_motor_bias.bin", h->ln_motor_b, SOMA_MOTOR_SIZE);

    /* Output projection */
    err |= soma_load_weight(weights_dir, "output_proj_weight.bin", h->output_w,
                            SOMA_OUTPUT_DIM * SOMA_MOTOR_SIZE);
    err |= soma_load_weight(weights_dir, "output_proj_bias.bin", h->output_b,
                            SOMA_OUTPUT_DIM);

    /* Sparse masks */
    err |= soma_load_weight(weights_dir, "mask_ic.bin", h->mask_ic,
                            SOMA_INTER_SIZE * SOMA_CMD_SIZE);
    err |= soma_load_weight(weights_dir, "mask_cm.bin", h->mask_cm,
                            SOMA_CMD_SIZE * SOMA_MOTOR_SIZE);

    /* State normalizer (optional — defaults to mean=0, std=1 if missing) */
    for (int i = 0; i < SOMA_INPUT_DIM; i++) {
        h->norm_mean[i] = 0.0f;
        h->norm_std[i] = 1.0f;
    }
    /* Try loading, ignore failure */
    soma_load_weight(weights_dir, "norm_mean.bin", h->norm_mean, SOMA_INPUT_DIM);
    soma_load_weight(weights_dir, "norm_std.bin", h->norm_std, SOMA_INPUT_DIM);

    if (err) {
        fprintf(stderr, "SOMA: failed to load some weights from %s\n", weights_dir);
        free(h);
        return NULL;
    }

    /* Zero hidden state */
    memset(h->hidden, 0, sizeof(h->hidden));

    return h;
}


void soma_step(SomaHandle* h, const float* state, float dt, float* torque_out) {
    /* --- Normalize state --- */
    float state_norm[SOMA_INPUT_DIM];
    for (int i = 0; i < SOMA_INPUT_DIM; i++) {
        state_norm[i] = (state[i] - h->norm_mean[i]) / (h->norm_std[i] + 1e-8f);
    }

    /* --- Split hidden state --- */
    float* h_inter = h->hidden;                                    /* [0..127]   */
    float* h_cmd   = h->hidden + SOMA_INTER_SIZE;                  /* [128..191] */
    float* h_motor = h->hidden + SOMA_INTER_SIZE + SOMA_CMD_SIZE;  /* [192..207] */

    /* --- Input projection + ReLU --- */
    float x_proj[SOMA_INTER_SIZE];
    soma_matvec(x_proj, h->input_w, state_norm, h->input_b, SOMA_INTER_SIZE, SOMA_INPUT_DIM);
    for (int i = 0; i < SOMA_INTER_SIZE; i++) {
        if (x_proj[i] < 0.0f) x_proj[i] = 0.0f;
    }

    /* --- Inter cell --- */
    float h_inter_new[SOMA_INTER_SIZE];
    soma_cfc_cell(h_inter_new, x_proj, SOMA_INTER_SIZE, h_inter, SOMA_INTER_SIZE, dt,
                  h->inter_ff_w, h->inter_ff_b, h->inter_gg_w, h->inter_gg_b);
    soma_layer_norm(h_inter_new, h->ln_inter_w, h->ln_inter_b, SOMA_INTER_SIZE);
    memcpy(h_inter, h_inter_new, SOMA_INTER_SIZE * sizeof(float));

    /* --- Sparse: inter → command --- */
    float inter_to_cmd[SOMA_CMD_SIZE];
    soma_vecmat(inter_to_cmd, h_inter, h->mask_ic, SOMA_INTER_SIZE, SOMA_CMD_SIZE);

    /* --- Command cell --- */
    float h_cmd_new[SOMA_CMD_SIZE];
    soma_cfc_cell(h_cmd_new, inter_to_cmd, SOMA_CMD_SIZE, h_cmd, SOMA_CMD_SIZE, dt,
                  h->cmd_ff_w, h->cmd_ff_b, h->cmd_gg_w, h->cmd_gg_b);
    soma_layer_norm(h_cmd_new, h->ln_cmd_w, h->ln_cmd_b, SOMA_CMD_SIZE);
    memcpy(h_cmd, h_cmd_new, SOMA_CMD_SIZE * sizeof(float));

    /* --- Sparse: command → motor --- */
    float cmd_to_motor[SOMA_MOTOR_SIZE];
    soma_vecmat(cmd_to_motor, h_cmd, h->mask_cm, SOMA_CMD_SIZE, SOMA_MOTOR_SIZE);

    /* --- Motor cell --- */
    float h_motor_new[SOMA_MOTOR_SIZE];
    soma_cfc_cell(h_motor_new, cmd_to_motor, SOMA_MOTOR_SIZE, h_motor, SOMA_MOTOR_SIZE, dt,
                  h->motor_ff_w, h->motor_ff_b, h->motor_gg_w, h->motor_gg_b);
    soma_layer_norm(h_motor_new, h->ln_motor_w, h->ln_motor_b, SOMA_MOTOR_SIZE);
    memcpy(h_motor, h_motor_new, SOMA_MOTOR_SIZE * sizeof(float));

    /* --- Output projection --- */
    soma_matvec(torque_out, h->output_w, h_motor, h->output_b, SOMA_OUTPUT_DIM, SOMA_MOTOR_SIZE);
}


void soma_reset(SomaHandle* h) {
    memset(h->hidden, 0, sizeof(h->hidden));
}


void soma_free(SomaHandle* h) {
    if (h) free(h);
}


const float* soma_get_hidden(const SomaHandle* h) {
    return h->hidden;
}


int soma_get_hidden_size(void) {
    return SOMA_HIDDEN_SIZE;
}

#endif /* SOMA_IMPLEMENTATION */
