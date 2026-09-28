# CIPHER ABI probes

Utilities for extracting byte offsets from PyTorch templated structs that
CIPHER's driver-level fusion path needs to navigate. Run after each PyTorch
version bump to refresh the offset tables in `src/cipher_flow_substitute.cpp`.

## probe_reduceop_offsets.cu

Prints byte offsets of the `at::native::ReduceOp<...>` fields used by the
RMSNorm K1 (MeanOps reduce) kernel. The `src` field is the input X pointer.

```
TORCH_INC=/path/to/torch/include
nvcc -std=c++17 -O0 -arch=sm_90 -x cu --extended-lambda \
    -I$TORCH_INC -I$TORCH_INC/torch/csrc/api/include \
    tools/probe_reduceop_offsets.cu -o /tmp/probe && /tmp/probe
```

Verified outputs per PyTorch version:
| PyTorch | sizeof(R) | offsetof(src) | offsetof(dst) |
|---------|-----------|---------------|---------------|
| 2.6     | 1048      | 984           | 992           |
