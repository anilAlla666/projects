/*
 * SOMA v7 — Standalone Example
 * Loads weights, runs inference, benchmarks latency.
 *
 * Build:
 *   gcc -O2 -o soma_example example_standalone.c -lm
 *
 * Run:
 *   ./soma_example ./weights
 */

#define SOMA_IMPLEMENTATION
#include "soma.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* High-resolution timer */
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1000.0;
}

int main(int argc, char** argv) {
    const char* weights_dir = (argc > 1) ? argv[1] : "./weights";

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  S.O.M.A. v7 — C SDK Example                ║\n");
    printf("║  Neural Dynamics Inc.                        ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    /* --- Init --- */
    printf("Loading weights from: %s\n", weights_dir);
    SomaHandle* soma = soma_init(weights_dir);
    if (!soma) {
        fprintf(stderr, "Failed to initialize SOMA\n");
        return 1;
    }
    printf("✅ SOMA initialized.\n");
    printf("   Input:  %d dimensions\n", SOMA_INPUT_DIM);
    printf("   Output: %d dimensions\n", SOMA_OUTPUT_DIM);
    printf("   Hidden: %d dimensions\n", soma_get_hidden_size());
    printf("   Struct: %lu bytes\n\n", (unsigned long)sizeof(SomaHandle));

    /* --- Create dummy state --- */
    /* In real use: state = [q_error(7), q_vel(7), gravity_comp(7), q_ref(7)] */
    float state[SOMA_INPUT_DIM];
    for (int i = 0; i < SOMA_INPUT_DIM; i++) {
        state[i] = 0.1f * sinf((float)i);  /* arbitrary test input */
    }
    float torque[SOMA_OUTPUT_DIM];
    float dt = 0.002f;  /* 2ms = 500Hz */

    /* --- Warmup --- */
    printf("Warming up (1000 steps)...\n");
    for (int i = 0; i < 1000; i++) {
        soma_step(soma, state, dt, torque);
    }

    /* --- Print one output --- */
    soma_reset(soma);
    soma_step(soma, state, dt, torque);
    printf("Sample output (first step from zero hidden):\n  torque = [");
    for (int i = 0; i < SOMA_OUTPUT_DIM; i++) {
        printf("%.4f%s", torque[i], i < SOMA_OUTPUT_DIM - 1 ? ", " : "");
    }
    printf("]\n\n");

    /* --- Benchmark --- */
    int N = 10000;
    soma_reset(soma);

    double* times = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double t0 = get_time_us();
        soma_step(soma, state, dt, torque);
        double t1 = get_time_us();
        times[i] = t1 - t0;
    }

    /* Sort for percentiles */
    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            if (times[j] < times[i]) {
                double tmp = times[i];
                times[i] = times[j];
                times[j] = tmp;
            }
        }
    }

    double sum = 0;
    for (int i = 0; i < N; i++) sum += times[i];

    printf("Benchmark (%d iterations):\n", N);
    printf("  Mean:   %7.1f µs\n", sum / N);
    printf("  Median: %7.1f µs\n", times[N / 2]);
    printf("  P95:    %7.1f µs\n", times[(int)(N * 0.95)]);
    printf("  P99:    %7.1f µs\n", times[(int)(N * 0.99)]);
    printf("  Min:    %7.1f µs\n", times[0]);
    printf("  Max:    %7.1f µs\n\n", times[N - 1]);

    double median = times[N / 2];
    double max_freq = 1e6 / median;
    printf("  Max control frequency: %.0f Hz\n", max_freq);
    printf("  Target 1kHz budget:    1000.0 µs\n");
    printf("  Headroom:              %.1f µs (%.0f%%)\n\n",
           1000.0 - median, (1000.0 - median) / 1000.0 * 100);

    if (median < 200.0) {
        printf("  ✅ PASSES <200µs target\n");
    } else if (median < 1000.0) {
        printf("  ⚠️  Under 1kHz budget but above 200µs target\n");
    } else {
        printf("  ❌ Exceeds 1kHz budget\n");
    }

    free(times);
    soma_free(soma);

    printf("\n═══════════════════════════════════════════════\n");
    return 0;
}
