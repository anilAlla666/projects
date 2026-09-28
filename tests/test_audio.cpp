
#include "test_common.h"
#include "HGAudioOcclusion.h"
void test_audio() {
    TEST_START("AUDIO OCCLUSION (gunfire through stone)");
    HG_Audio_Init();

    HG_AudioInput in = {};
    in.listenerX = 0; in.listenerY = 1.5f; in.listenerZ = 0;
    in.sourceX = 20; in.sourceY = 1.5f; in.sourceZ = 0;
    in.soundType = SOUND_GUNFIRE; in.envType = ENV_BASE_INTERIOR;
    in.wallConfig = WALL_THICK_STONE;
    in.wallDistance = 10; in.wallThickness = 0.5f;

    HG_AudioOutput out = {};
    CHECK(HG_Audio_Compute(&in, &out) == 0, "Compute ok");
    printf("  ℹ️  Atten=%.3f LPF=%.0f Rev=%.3f Delay=%.1f\n",
        out.attenuation, out.lowpassCutoff, out.reverbWet, out.delayMs);

    CHECK_RANGE(out.attenuation, 0, 1, "Attenuation");
    CHECK_RANGE(out.lowpassCutoff, 200, 20000, "LPF Hz");
    CHECK_RANGE(out.reverbWet, 0, 1, "Reverb");
    CHECK_RANGE(out.delayMs, 0, 100, "Delay ms");

    Timer t; t.start();
    for (int i = 0; i < 10000; i++) HG_Audio_Compute(&in, &out);
    double us = t.elapsed_us() / 10000;
    printf("  ⏱  Latency: %.2f µs/call\n", us);
    CHECK(us < 50.0, "Latency < 50µs (got %.2f)", us);
    HG_Audio_Shutdown();
}
