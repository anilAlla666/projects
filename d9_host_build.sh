#!/bin/bash
# D.9 FP8 actuator — HOST build (container down). Host cu13 paths (python3.10 .local).
set -e
cd /home/ubuntu/cipher_rt_phase4
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
CU13I=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/include
# -l link symlinks (no sudo; we own .local)
ln -sf libcusolver.so.12 "$CU13L/libcusolver.so"
ln -sf libcudart.so.13   "$CU13L/libcudart.so"
ln -sf libcupti.so.13    "$CU13L/libcupti.so"
ln -sf libcublas.so.13   "$CU13L/libcublas.so" 2>/dev/null || true
ln -sf libcublasLt.so.13 "$CU13L/libcublasLt.so" 2>/dev/null || true
# preserve the current d7-rh1-close build as fallback before rebuilding
[ -f build_cuda13/libcipher_rt.so ] && cp -f build_cuda13/libcipher_rt.so build_cuda13/libcipher_rt.so.pre_d9fp8 2>/dev/null || true
make clean >/dev/null 2>&1 || true
make -j"$(nproc)" \
  CUDA_INCLUDE="$CU13I" \
  CUPTI_INCLUDE="$CU13I" \
  LDFLAGS="-shared -fPIC -L$CU13L -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,$CU13L" \
  > /home/ubuntu/d9_fp8_make.log 2>&1
echo "make rc=$?"
if [ -f libcipher_rt.so ]; then
  cp -f libcipher_rt.so build_cuda13/libcipher_rt.so.d9fp8_staging
  echo "BUILD_OK md5=$(md5sum libcipher_rt.so | cut -d' ' -f1) size=$(stat -c%s libcipher_rt.so)"
  echo "FP8 symbols: $(nm -D libcipher_rt.so | grep -ciE 'cipher_rt_fp8')"
  nm -D libcipher_rt.so | grep -iE 'cipher_rt_fp8_init|fp8_engine_matmul' | head
else
  echo "BUILD_FAILED — tail of log:"; tail -30 /home/ubuntu/d9_fp8_make.log
fi
