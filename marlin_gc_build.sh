#!/bin/bash
# Marlin GC-on-free — HOST build (container down). Host cu13 paths.
set -e
cd /home/ubuntu/cipher_rt_phase4
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
CU13I=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/include
ln -sf libcusolver.so.12 "$CU13L/libcusolver.so"
ln -sf libcudart.so.13   "$CU13L/libcudart.so"
ln -sf libcupti.so.13    "$CU13L/libcupti.so"
ln -sf libcublas.so.13   "$CU13L/libcublas.so" 2>/dev/null || true
ln -sf libcublasLt.so.13 "$CU13L/libcublasLt.so" 2>/dev/null || true
# preserve current staging .so as fallback before rebuilding
[ -f libcipher_rt.so ] && cp -f libcipher_rt.so /home/ubuntu/libcipher_rt.so.pre_marlin_gc 2>/dev/null || true
make clean >/dev/null 2>&1 || true
make -j"$(nproc)" \
  CUDA_INCLUDE="$CU13I" \
  CUPTI_INCLUDE="$CU13I" \
  LDFLAGS="-shared -fPIC -L$CU13L -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,$CU13L" \
  > /home/ubuntu/marlin_gc_make.log 2>&1
RC=$?
echo "make rc=$RC"
if [ -f libcipher_rt.so ]; then
  cp -f libcipher_rt.so build_cuda13/libcipher_rt.so.marlin_gc_staging
  echo "BUILD_OK md5=$(md5sum libcipher_rt.so | cut -d' ' -f1) size=$(stat -c%s libcipher_rt.so)"
  echo "GC symbols exported:"
  nm -D libcipher_rt.so | grep -iE 'evict_weight|reclaim_retired|bf16_surrogate|engine_is_ready' || echo "  (none found!)"
else
  echo "BUILD_FAILED — tail of log:"; tail -40 /home/ubuntu/marlin_gc_make.log
fi
