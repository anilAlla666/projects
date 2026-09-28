set -e
cd /home/ubuntu/cipher_rt_phase4
CU13L=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
CU13I=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
CUPTI=/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/lib/cupti
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libssl-dev >/dev/null 2>&1 && echo "libssl-dev OK"
ln -sf "$CU13L/libcusolver.so.12" "$CU13L/libcusolver.so" 2>/dev/null
# add MISSING cu13 headers (cusolver*, etc.) into CUDA_INCLUDE without clobbering existing
for h in "$CU13I"/*.h; do b=$(basename "$h"); [ -e "/usr/local/cuda/include/$b" ] || ln -sf "$h" /usr/local/cuda/include/; done
echo "cusolverDn.h resolvable: $(ls /usr/local/cuda/include/cusolverDn.h 2>/dev/null || echo NO)"
make clean >/dev/null 2>&1
make -j$(nproc) \
  CUDA_INCLUDE=/usr/local/cuda/include \
  CUPTI_INCLUDE=/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/include \
  LDFLAGS="-shared -fPIC -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -L$CU13L -L$CUPTI -Wl,-rpath,$CUPTI" \
  > /home/ubuntu/d7_make_full.log 2>&1
echo "make rc=$?"
if [ -f libcipher_rt.so ]; then
  cp libcipher_rt.so build_cuda13/libcipher_rt.so.d7_staging
  echo "BUILD_OK staging md5: $(md5sum build_cuda13/libcipher_rt.so.d7_staging|cut -d' ' -f1)"
  nm -D build_cuda13/libcipher_rt.so.d7_staging | grep -qi bind_model && echo "bind_model EXPORTED"
  make clean >/dev/null 2>&1
else echo "BUILD_FAILED"; fi
