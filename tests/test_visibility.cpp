
#include "test_common.h"
#include "VisibilityKernel.h"
void test_visibility() {
    TEST_START("VISIBILITY (line-of-sight)");
    HyperFlux::Visibility::Engine engine;
    CHECK(engine.Initialize("HYPERFLUX-VIS-KEY-12345"), "Engine initialized");

    HyperFlux::Visibility::Box wall;
    wall.center = HyperFlux::Visibility::Vec3(5, 2, 0);
    wall.halfExtents = HyperFlux::Visibility::Vec3(0.5f, 2, 5);
    HyperFlux::Visibility::MapBounds bounds(
        HyperFlux::Visibility::Vec3(-50, -10, -50),
        HyperFlux::Visibility::Vec3(50, 20, 50));
    auto err = engine.LoadMap(&wall, 1, nullptr, 0, &bounds, nullptr);
    CHECK(err == HyperFlux::Visibility::ErrorCode::Success, "Map loaded");

    HyperFlux::Visibility::Vec3 obs(0, 1.5f, 0), tgt(10, 1.5f, 0);
    auto r = engine.CheckVisibility(obs, tgt);
    printf("  ℹ️  Through wall: visible=%d conf=%.3f\n", r.visible, r.confidence);
    CHECK_RANGE(r.confidence, 0, 1, "Confidence");

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) engine.CheckVisibility(obs, tgt);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);
    engine.UnloadMap();
}
