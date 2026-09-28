#!/bin/bash
set -e
cd "$(dirname "$0")"

CXX="clang++"
CFLAGS="-std=c++17 -O2 -mcpu=apple-m1"

echo "Compiling tests..."
for entry in \
    "main.cpp:" \
    "test_mount.cpp:MountDynamics" \
    "test_destruction.cpp:Destruction" \
    "test_audio.cpp:AudioOcclusion" \
    "test_lod.cpp:LODSelection" \
    "test_lumen.cpp:LumenGI" \
    "test_occlusion.cpp:OcclusionCulling" \
    "test_hitbox.cpp:Hitbox" \
    "test_ballistics.cpp:Ballistics" \
    "test_spawn.cpp:SpawnSelection" \
    "test_visibility.cpp:Visibility"; do
    IFS=':' read -r SRC KNAME <<< "$entry"
    INCLUDES="-Itests"
    [ -n "$KNAME" ] && INCLUDES="$INCLUDES -Ikernels/$KNAME/include"
    $CXX $CFLAGS $INCLUDES -c "tests/$SRC" -o "tests/${SRC%.cpp}.o"
    echo "  ✅ $SRC"
done

echo ""

# Binary 1: 9 kernels (no Ballistics — conflicts with MountDynamics C symbols)
echo "Linking test_9kernels..."
cat > tests/main_9.cpp << 'M9'
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
M9
$CXX $CFLAGS -Itests -c tests/main_9.cpp -o tests/main_9.o

$CXX $CFLAGS \
    tests/main_9.o \
    tests/test_mount.o tests/test_destruction.o tests/test_audio.o \
    tests/test_lod.o tests/test_lumen.o tests/test_occlusion.o \
    tests/test_hitbox.o tests/test_spawn.o tests/test_visibility.o \
    -Lkernels/MountDynamics/lib -lHyperFluxMount \
    -Lkernels/Hitbox/lib -lHitboxKernel \
    -Lkernels/Visibility/lib -lVisibilityKernel \
    -Lkernels/SpawnSelection/lib -lHyperFluxSpawn \
    -Lkernels/Destruction/lib -lHyperFluxDestruction \
    -Lkernels/AudioOcclusion/lib -lHyperFluxAudioOcclusion \
    -Lkernels/LODSelection/lib -lHyperFluxLODSelection \
    -Lkernels/OcclusionCulling/lib -lHyperFluxOcclusionCulling \
    -Lkernels/LumenGI/lib -lHyperFluxLumenGI \
    -o test_9kernels
echo "  ✅ test_9kernels"

# Binary 2: Ballistics only
echo "Linking test_ballistics..."
cat > tests/main_bal.cpp << 'MB'
#include "test_common.h"
int g_pass = 0, g_fail = 0;
void test_ballistics();
int main() {
    printf("═══ BALLISTICS ═══\n");
    test_ballistics();
    printf("\nResult: %d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
MB
$CXX $CFLAGS -Itests -c tests/main_bal.cpp -o tests/main_bal.o

$CXX $CFLAGS \
    tests/main_bal.o tests/test_ballistics.o \
    -Lkernels/Ballistics/lib -lHyperFlux \
    -o test_ballistics
echo "  ✅ test_ballistics"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  RUNNING 9-KERNEL TEST"
echo "═══════════════════════════════════════════════════"
echo ""
./test_9kernels

echo ""
echo "═══════════════════════════════════════════════════"
echo "  RUNNING BALLISTICS TEST"
echo "═══════════════════════════════════════════════════"
echo ""
./test_ballistics
