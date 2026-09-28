#!/usr/bin/env python3
# CP 2.5 load test — NON-PyTorch CUDA process.
# Drives cuInit via ctypes (no torch import). With CUDA_INJECTION64_PATH set,
# the driver dlopen's libcipher_rt.so; if it carries an unresolved symbol the
# dlopen fails. We assert cuInit returns 0 and report whether the injection
# library actually loaded (it must, for CP 2.5 — not the graceful-skip path).
import ctypes, sys, os

cuda = ctypes.CDLL("libcuda.so.1")
rc = cuda.cuInit(0)
n = ctypes.c_int(0)
cuda.cuDeviceGetCount(ctypes.byref(n))
print(f"NONTORCH: cuInit rc={rc} device_count={n.value} "
      f"CUDA_INJECTION64_PATH={os.environ.get('CUDA_INJECTION64_PATH','<unset>')}",
      file=sys.stderr)
sys.exit(0 if rc == 0 else 1)
