
#include "test_common.h"
#include "HyperFlux.h"
void test_ballistics() {
    TEST_START("BALLISTICS (M4A1 at 200m)");
    HyperFlux::Engine engine("HYPERFLUX-TEST-KEY-12345678");
    CHECK(engine.isValid(), "Engine initialized");

    HyperFlux::Vec3 velocity(900.0f, 5.0f, 0.0f);
    HyperFlux::Vec3 pos = engine.predict(velocity, 0.2f, HyperFlux::Weapon::M4A1);
    CHECK(pos.x != 0 || pos.y != 0 || pos.z != 0, "Non-zero output: (%.3f, %.3f, %.3f)", pos.x, pos.y, pos.z);
    CHECK(pos.x > 0, "Forward travel: x=%.3f > 0", pos.x);

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) engine.predict(velocity, 0.2f, HyperFlux::Weapon::M4A1);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);

    // Batch
    HyperFlux::Vec3 vels[4] = {velocity, velocity, velocity, velocity};
    float times[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    HyperFlux::Vec3 positions[4];
    HyperFlux::BatchInput bi; bi.velocities = vels; bi.times = times; bi.count = 4;
    bi.singleWeapon = HyperFlux::Weapon::M4A1;
    HyperFlux::BatchOutput bo; bo.positions = positions; bo.count = 4;
    bool ok = engine.predictBatch(bi, bo);
    CHECK(ok, "Batch predict ok");
    CHECK(positions[0].x < positions[1].x, "Increasing distance over time");
}
