
#include "test_common.h"
#include "HitboxKernel.h"
void test_hitbox() {
    TEST_START("HITBOX (headshot at 10m)");
    HyperFlux::Hitbox::Engine engine;
    CHECK(engine.Initialize(), "Engine initialized");

    HyperFlux::Hitbox::PlayerState player;
    player.position = HyperFlux::Hitbox::Vec3(0, 0, 0);
    player.yaw = 0; player.pose = HyperFlux::Hitbox::Pose::Standing;

    HyperFlux::Hitbox::Ray ray;
    ray.origin = HyperFlux::Hitbox::Vec3(10, 1.7f, 0);
    ray.direction = HyperFlux::Hitbox::Vec3(-1, 0, 0);

    auto result = engine.CheckHit(ray, player);
    CHECK(result.hit, "Hit detected");
    CHECK_RANGE(result.confidence, 0, 1, "Confidence");
    if (result.hit) printf("  ℹ️  Part: %s\n", HyperFlux::Hitbox::BodyPartToString(result.part));

    // Miss
    HyperFlux::Hitbox::Ray miss_ray;
    miss_ray.origin = HyperFlux::Hitbox::Vec3(100, 0, 0);
    miss_ray.direction = HyperFlux::Hitbox::Vec3(0, 1, 0);
    CHECK(!engine.CheckHit(miss_ray, player).hit, "Miss on perpendicular ray");

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) engine.CheckHit(ray, player);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);
}
