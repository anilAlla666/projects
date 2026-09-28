
#include "test_common.h"
int g_pass = 0, g_fail = 0;

void test_ballistics();
void test_hitbox();
void test_visibility();
void test_spawn();
void test_mount();
void test_destruction();
void test_audio();
void test_lod();
void test_occlusion();
void test_lumen();

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     HYPERFLUX SDK — MAC INTEGRATION TEST (10/10)       ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    test_mount();
    test_destruction();
    test_audio();
    test_lod();
    test_lumen();
    test_occlusion();
    test_hitbox();
    test_ballistics();
    test_spawn();
    test_visibility();

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    if (g_fail == 0)
        printf("║  ✅ ALL %d CHECKS PASSED                                ║\n", g_pass);
    else
        printf("║  %d PASS, %d FAIL                                      ║\n", g_pass, g_fail);
    printf("╚══════════════════════════════════════════════════════════╝\n");
    return g_fail > 0 ? 1 : 0;
}
