
#include "test_common.h"
#include "HGOcclusionCulling.h"
void test_occlusion() {
    TEST_START("OCCLUSION CULLING (object behind wall)");
    HG_Occlusion_Init();

    HG_CameraInfo cam = {};
    cam.cameraPosY = 1.5f; cam.cameraForwardZ = 1; cam.cameraFOV = 90;

    HG_ObjectBounds obj = {}; obj.posY = 1; obj.posZ = 30; obj.boundingRadius = 2;
    HG_OccluderInfo occ = {}; occ.posY = 1.5f; occ.posZ = 15; occ.size = 10; occ.type = OCCLUDER_WALL_THICK;

    HG_VisibilityResult r = {};
    HG_Occlusion_Test(&cam, &obj, &occ, &r);
    printf("  ℹ️  Visible=%d Conf=%.3f\n", r.isVisible, r.confidence);
    CHECK_RANGE(r.confidence, 0, 1, "Confidence");

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) HG_Occlusion_Test(&cam, &obj, &occ, &r);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);
    HG_Occlusion_Shutdown();
}
