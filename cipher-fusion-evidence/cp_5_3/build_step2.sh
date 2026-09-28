#!/bin/bash
# CP 5.3 STEP 2 (Axis A) side-build. NOT part of libcipher_rt.so.
#   - cp53_marlin_engine_step2.cpp   side-build engine (shipped fix + 9th arg)
#   - cp53_marlin_kernel_src_step2.cpp  %smid-instrumented kernel
#   - cp53_step2_harness.cpp         gate harness
# Links the REAL shipped green-ctx object so green-ctx creation, group
# selection and the STEP 2 accessors are the production code paths.
set -e
CXX=g++
CXXFLAGS="-O2 -std=c++17 -D_GNU_SOURCE -D_GLIBCXX_USE_CXX11_ABI=1 -I. -I/usr/include"
GREEN_O=/home/ubuntu/cipher_rt_phase4/cipher_rt_green_ctx.o

echo "[build] side-engine ..."
$CXX $CXXFLAGS -c cp53_marlin_engine_step2.cpp     -o cp53_marlin_engine_step2.o
echo "[build] instrumented kernel src ..."
$CXX $CXXFLAGS -c cp53_marlin_kernel_src_step2.cpp -o cp53_marlin_kernel_src_step2.o
echo "[build] harness ..."
$CXX $CXXFLAGS -c cp53_step2_harness.cpp           -o cp53_step2_harness.o
echo "[link] (real green-ctx object: $GREEN_O) ..."
$CXX cp53_step2_harness.o cp53_marlin_engine_step2.o \
     cp53_marlin_kernel_src_step2.o "$GREEN_O" \
     -o cp53_step2_harness \
     -L/usr/lib/x86_64-linux-gnu -lcudart -lcublas -lcuda -ldl -lpthread
echo "[build] OK -> cp53_step2_harness"
