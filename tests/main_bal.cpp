#include "test_common.h"
int g_pass = 0, g_fail = 0;
void test_ballistics();
int main() {
    printf("═══ BALLISTICS ═══\n");
    test_ballistics();
    printf("\nResult: %d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
