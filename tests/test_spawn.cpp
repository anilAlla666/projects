
#include "test_common.h"
#include "HyperFluxSpawn.h"
void test_spawn() {
    TEST_START("SPAWN SELECTION (6v3)");
    HyperFlux::Spawn::Engine engine;
    CHECK(engine.Initialize("HYPERFLUX-SPAWN-KEY"), "Engine initialized");

    HyperFlux::Spawn::GameState gs;
    gs.numEnemies = 6; gs.numTeammates = 3;
    for (int i = 0; i < 6; i++) { gs.enemies[i].position = HyperFlux::Spawn::Vec3(5+i, 0, 5); gs.enemies[i].active = true; }
    for (int i = 0; i < 3; i++) { gs.teammates[i].position = HyperFlux::Spawn::Vec3(50, 0, 50-i*5); gs.teammates[i].active = true; }
    gs.gameTime = 120; gs.scoreDiff = -5;

    HyperFlux::Spawn::SpawnResult result;
    auto err = engine.SelectSpawn(gs, result);
    CHECK(err == HyperFlux::Spawn::ErrorCode::Success, "SelectSpawn ok");
    CHECK(result.bestSpawnIdx < 30, "Spawn idx valid: %d", result.bestSpawnIdx);
    CHECK_RANGE(result.confidence, 0, 1, "Confidence");

    bool scores_ok = true;
    for (int i = 0; i < 30; i++) if (result.scores[i] < 0 || result.scores[i] > 1) scores_ok = false;
    CHECK(scores_ok, "All 30 scores in [0,1]");

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) engine.SelectSpawn(gs, result);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);
}
