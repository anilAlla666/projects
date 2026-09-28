
#include "test_common.h"
#include "HGLODSelection.h"
void test_lod() {
    TEST_START("LOD SELECTION (hero@50m vs detail@200m)");
    HG_LOD_Init();

    HG_CameraState cam = {};
    cam.cameraPosZ = 0; cam.cameraPosY = 1.5f;
    cam.cameraForwardZ = 1; cam.cameraFOV = 90;

    HG_ObjectInfo hero = {}; hero.posZ = 50; hero.boundingRadius = 2; hero.importance = IMPORTANCE_HERO;
    HG_LODResult r1 = {}; HG_LOD_Select(&cam, &hero, &r1);
    printf("  ℹ️  Hero@50m: LOD=%d Blend=%.3f Cull=%d\n", r1.lodLevel, r1.blendFactor, r1.shouldCull);
    CHECK(r1.lodLevel >= 0 && r1.lodLevel <= 4, "LOD in [0,4]: %d", r1.lodLevel);
    CHECK_RANGE(r1.blendFactor, 0, 1, "Blend");

    HG_ObjectInfo det = {}; det.posZ = 200; det.boundingRadius = 0.5f; det.importance = IMPORTANCE_DETAIL;
    HG_LODResult r2 = {}; HG_LOD_Select(&cam, &det, &r2);
    printf("  ℹ️  Detail@200m: LOD=%d Blend=%.3f Cull=%d\n", r2.lodLevel, r2.blendFactor, r2.shouldCull);
    CHECK(r2.lodLevel >= r1.lodLevel || r2.shouldCull, "Far → higher LOD or culled");

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) HG_LOD_Select(&cam, &hero, &r1);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);
    HG_LOD_Shutdown();
}
