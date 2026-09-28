
#include "test_common.h"
#include "HyperFluxMount.h"
void test_mount() {
    TEST_START("MOUNT DYNAMICS (horse uphill)");
    CHECK(HyperFlux_Initialize(), "Init");

    HyperFluxMountInput in = {};
    in.type = HYPERFLUX_MOUNT_HORSE;
    in.velocityX = 10; in.velocityY = 0; in.velocityZ = 2;
    in.currentSpeed = 10.2f; in.throttle = 1.0f; in.steering = 0.1f;
    in.terrainSlope = 0.15f; in.terrainFacing = 0.8f;
    in.isGrounded = 1; in.stamina = 0.7f;

    HyperFluxMountOutput out = {};
    CHECK(HyperFlux_Predict(&in, &out) == 1, "Predict ok");
    printf("  ℹ️  Delta: (%.4f, %.4f, %.4f)\n", out.deltaVelocityX, out.deltaVelocityY, out.deltaVelocityZ);
    CHECK(out.deltaVelocityX != 0 || out.deltaVelocityY != 0 || out.deltaVelocityZ != 0, "Non-zero delta");

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) HyperFlux_Predict(&in, &out);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);
    HyperFlux_Shutdown();
}
