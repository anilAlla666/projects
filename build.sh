#!/bin/bash
#
# HyperFlux SDK — Build for macOS Apple Silicon
# Usage: ./build.sh
#

set -e
cd "$(dirname "$0")"

CXX="clang++"
CFLAGS="-std=c++17 -O3 -mcpu=apple-m1 -ffast-math -fPIC -DNDEBUG"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  HyperFlux SDK — macOS Apple Silicon Build              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Compiler: $($CXX --version | head -1)"
echo "Flags: $CFLAGS"
echo ""

KERNELS=(
    "Ballistics:HyperFlux.cpp:libHyperFlux.a"
    "Hitbox:HitboxKernel.cpp:libHitboxKernel.a"
    "Visibility:VisibilityKernel.cpp:libVisibilityKernel.a"
    "SpawnSelection:HyperFluxSpawn.cpp:libHyperFluxSpawn.a"
    "MountDynamics:HyperFluxMount.cpp:libHyperFluxMount.a"
    "Destruction:HyperFluxDestruction.cpp:libHyperFluxDestruction.a"
    "AudioOcclusion:HyperFluxAudioOcclusion.cpp:libHyperFluxAudioOcclusion.a"
    "LODSelection:HyperFluxLODSelection.cpp:libHyperFluxLODSelection.a"
    "OcclusionCulling:HyperFluxOcclusionCulling.cpp:libHyperFluxOcclusionCulling.a"
    "LumenGI:HyperFluxLumenGI.cpp:libHyperFluxLumenGI.a"
)

PASS=0
FAIL=0

for entry in "${KERNELS[@]}"; do
    IFS=':' read -r KNAME SRC LIB <<< "$entry"
    KDIR="kernels/$KNAME"
    
    echo "── $KNAME ──"
    
    if $CXX $CFLAGS \
        -I"$KDIR/include" \
        -I"$KDIR/PRIVATE" \
        -I"$KDIR/src" \
        -c "$KDIR/src/$SRC" \
        -o "$KDIR/lib/${SRC%.cpp}.o" 2>&1; then
        
        ar rcs "$KDIR/lib/$LIB" "$KDIR/lib/${SRC%.cpp}.o"
        rm -f "$KDIR/lib/${SRC%.cpp}.o"
        SIZE=$(stat -f%z "$KDIR/lib/$LIB" 2>/dev/null || stat -c%s "$KDIR/lib/$LIB" 2>/dev/null)
        echo "  ✅ $LIB ($SIZE bytes)"
        PASS=$((PASS + 1))
    else
        echo "  ❌ COMPILE FAILED"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

echo "════════════════════════════════════════════════════"
echo "  BUILD: $PASS pass, $FAIL fail"
echo "════════════════════════════════════════════════════"

if [ $FAIL -gt 0 ]; then
    echo "  ❌ Fix errors above before running tests"
    exit 1
fi

echo ""
echo "  Now run: ./test.sh"
