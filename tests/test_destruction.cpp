
#include "test_common.h"
#include "HyperFluxDestruction.h"
void test_destruction() {
    TEST_START("DESTRUCTION (rocket vs wood)");
    CHECK(HyperFlux::DestructionKernel::Initialize(), "Init");

    HyperFlux::WallState wall;
    wall.position = HyperFlux::Vec3(10, 2, 0);
    wall.dimensions = HyperFlux::Vec3(4, 3, 0.3f);
    wall.currentHP = 500; wall.maxHP = 1000;
    wall.material = HyperFlux::Material::WOOD;

    HyperFlux::DamageEvent dmg;
    dmg.impactPosition = HyperFlux::Vec3(10, 1.5f, 0);
    dmg.impactDirection = HyperFlux::Vec3(0, 0, 1);
    dmg.damageType = HyperFlux::DamageType::ROCKET;
    dmg.damage = 800; dmg.radius = 3.0f;

    auto r = HyperFlux::DestructionKernel::Predict(wall, dmg);
    printf("  ℹ️  Destroyed=%d HP=%.1f Frags=%.1f Conf=%.3f\n", r.destroyed, r.newHP, r.fragmentCount, r.confidence);
    CHECK_RANGE(r.confidence, 0, 1, "Confidence");

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) HyperFlux::DestructionKernel::Predict(wall, dmg);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);
    HyperFlux::DestructionKernel::Shutdown();
}
