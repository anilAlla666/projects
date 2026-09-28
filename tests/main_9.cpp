#include "test_common.h"
int g_pass = 0, g_fail = 0;
void test_mount(); void test_destruction(); void test_audio();
void test_lod(); void test_lumen(); void test_occlusion();
void test_hitbox(); void test_spawn(); void test_visibility();
int main() {
    printf("═══ 9 KERNELS (excl. Ballistics) ═══\n");
    test_mount(); test_destruction(); test_audio();
    test_lod(); test_lumen(); test_occlusion();
    test_hitbox(); test_spawn(); test_visibility();
    printf("\nResult: %d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
