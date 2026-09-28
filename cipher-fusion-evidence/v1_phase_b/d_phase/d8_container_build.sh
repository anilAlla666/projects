set -e
cd /home/ubuntu/cipher_rt_phase4
CU13L=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
CU13I=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libssl-dev >/dev/null 2>&1 && echo "libssl-dev OK"
ln -sf "$CU13L/libcusolver.so.12" "$CU13L/libcusolver.so" 2>/dev/null
for h in "$CU13I"/*.h; do b=$(basename "$h"); [ -e "/usr/local/cuda/include/$b" ] || ln -sf "$h" /usr/local/cuda/include/; done
make clean >/dev/null 2>&1
make -j$(nproc) \
  CUDA_INCLUDE=/usr/local/cuda/include \
  CUPTI_INCLUDE=/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/include \
  LDFLAGS="-shared -fPIC -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -L$CU13L -L/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/lib/cupti -Wl,-rpath,/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/lib/cupti" \
  > /home/ubuntu/d8_make.log 2>&1
echo "make rc=$?"
if [ -f libcipher_rt.so ]; then
  cp libcipher_rt.so build_cuda13/libcipher_rt.so.d8_staging
  echo "D8_BUILD_OK md5=$(md5sum build_cuda13/libcipher_rt.so.d8_staging|cut -d' ' -f1)"
  nm -D build_cuda13/libcipher_rt.so.d8_staging | grep -ciE "cipher_rt_fairness" | xargs echo "d8 fairness symbols:"
  make clean >/dev/null 2>&1
else echo "D8_BUILD_FAILED"; grep -iE "error:|undefined" /home/ubuntu/d8_make.log | head -20; fi
