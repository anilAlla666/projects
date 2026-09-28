
#include "test_common.h"
#include "HGLumenGI.h"
void test_lumen() {
    TEST_START("LUMEN GI (outdoor stone, noon)");
    HG_Lumen_Init();

    HG_LumenInput in = {};
    in.posX = 10; in.posZ = 10;
    in.normalY = 1;
    in.sunDirX = 0.3f; in.sunDirY = -0.9f; in.sunDirZ = 0.1f;
    in.sunIntensity = 0.9f; in.timeOfDay = 0.5f;
    in.surfaceType = SURFACE_STONE;

    HG_LumenOutput out = {};
    CHECK(HG_Lumen_Compute(&in, &out) == 0, "Compute ok");
    printf("  ℹ️  RGB=(%.3f,%.3f,%.3f) I=%.3f AO=%.3f Sky=%.3f\n",
        out.indirectR, out.indirectG, out.indirectB,
        out.indirectIntensity, out.ambientOcclusion, out.skyVisibility);

    CHECK_RANGE(out.indirectR, 0, 1, "R");
    CHECK_RANGE(out.indirectG, 0, 1, "G");
    CHECK_RANGE(out.indirectB, 0, 1, "B");
    CHECK_RANGE(out.indirectIntensity, 0, 2, "Intensity");
    CHECK_RANGE(out.ambientOcclusion, 0, 1, "AO");
    CHECK_RANGE(out.skyVisibility, 0, 1, "Sky");

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) HG_Lumen_Compute(&in, &out);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);
    HG_Lumen_Shutdown();
}
