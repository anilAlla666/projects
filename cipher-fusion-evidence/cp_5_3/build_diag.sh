#!/bin/bash
set -e
CXX=g++
CXXFLAGS="-O2 -std=c++17 -D_GNU_SOURCE -D_GLIBCXX_USE_CXX11_ABI=1 -I. -I/usr/include"
echo "[build] engine diag..."
$CXX $CXXFLAGS -c cp53_marlin_engine_diag.cpp -o cp53_marlin_engine_diag.o
echo "[build] kernel src..."
$CXX $CXXFLAGS -c cipher_rt_marlin_kernel_src.cpp -o cipher_rt_marlin_kernel_src.o
echo "[build] harness..."
$CXX $CXXFLAGS -c cp53_splitk_diag.cpp -o cp53_splitk_diag.o
echo "[link]..."
$CXX cp53_splitk_diag.o cp53_marlin_engine_diag.o cipher_rt_marlin_kernel_src.o \
    -o cp53_splitk_diag \
    -L/usr/lib/x86_64-linux-gnu -lcudart -lcublas -ldl -lpthread
echo "[build] OK -> cp53_splitk_diag"
