/*
 * SOMA v11 — Minimal Test Harness
 * Compile: gcc -O2 -o test_soma test_soma.c -lm
 * Run: ./test_soma
 */
#include <stdio.h>
#include <time.h>
#include "soma_controller.h"

int main(void) {
    soma_state_t state;
    soma_init(&state);

    soma_obs_t obs;
    memset(&obs, 0, sizeof(obs));
    obs.base_quat[0] = 1.0f; /* Identity quaternion */
    obs.base_height = 0.3f;

    /* Set descriptors (example: Go2-like limits) */
    for (int i = 0; i < SOMA_NU; i++) {
        obs.descriptors[i][2] = 45.4f;  /* torque_limit */
        obs.descriptors[i][8] = -1.0f;  /* joint_range_min */
        obs.descriptors[i][9] = 1.0f;   /* joint_range_max */
        obs.joint_mask[i] = 1.0f;
    }

    soma_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.locomotion[0] = 0.5f;  /* Walk forward at 0.5 m/s */
    cmd.posture[0] = 0.3f;     /* Target height 30cm */

    float torques[SOMA_NU];

    /* Latency benchmark */
    int N = 100000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) {
        soma_step(&state, &obs, &cmd, torques, 0.001f);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_us = ((t1.tv_sec - t0.tv_sec) * 1e6 +
                         (t1.tv_nsec - t0.tv_nsec) / 1e3) / N;

    printf("SOMA v11 INT8 Test\n");
    printf("==================\n");
    printf("Joints: %d\n", SOMA_NU);
    printf("Backbone H: %d\n", SOMA_BACKBONE_H);
    printf("Expert H: %d\n", SOMA_EXPERT_H);
    printf("Latency: %.1f μs/step\n", elapsed_us);
    printf("Frequency: %.0f Hz\n", 1e6 / elapsed_us);
    printf("\nTorques (first step):\n");
    for (int i = 0; i < SOMA_NU; i++) {
        printf("  joint[%d]: %.4f Nm\n", i, torques[i]);
    }

    /* Safety check: all torques within limits */
    int safe = 1;
    for (int i = 0; i < SOMA_NU; i++) {
        if (torques[i] > obs.descriptors[i][2] ||
            torques[i] < -obs.descriptors[i][2]) {
            printf("  VIOLATION: joint[%d] = %.2f > limit %.2f\n",
                   i, torques[i], obs.descriptors[i][2]);
            safe = 0;
        }
    }
    printf("\nSafety: %s\n", safe ? "PASS (0 violations)" : "FAIL");

    return safe ? 0 : 1;
}
