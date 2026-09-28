# HyperFlux v16.0 - INT8 Inference SDK

## Overview

HyperFlux v16.0 uses **weight-only INT8 quantization** for efficient inference:

- **INT8 weights** with per-channel FP16 scales (~2x smaller)
- **FP32 activations** (preserves accuracy)
- **FP32 LayerNorm & SiLU** (same as v15.5)
- **Deterministic** (bit-exact across platforms)

## Performance

| Metric | v15.5 (FP16) | v16.0 (INT8) |
|--------|--------------|--------------|
| **MAE** | 3.06mm | **3.05mm** ✓ |
| **Model Size** | 1.5 MB | **0.75 MB** |
| **Throughput** | Baseline | **~7500/sec** (scalar) |

## Files

```
hyperflux_int8_single.hpp  - Single-header implementation
test_quick.cpp             - Quick validation test
```

## Usage

### 1. Include the header

```cpp
#define HYPERFLUX_INT8_IMPLEMENTATION  // In ONE .cpp file only
#include "hyperflux_int8_single.hpp"
```

### 2. Load and run inference

```cpp
hyperflux::HyperFluxInt8 model;

// Load from NPZ (exported from Python training)
model.load_from_npz("int8_weights.npz");

// Run inference
float input[14] = {
    vx/1000, vy/1000, vz/1000,  // Velocity (m/s) / 1000
    time/10,                     // Time (s) / 10
    1, 0, 0, 0,                  // Drag encoding (G7)
    effective_range/2000,        // Range / 2000
    has_thrust,                  // 0 or 1
    t_max/5,                     // Max time / 5
    hidden_dim/512               // Hidden dim / 512
};

float output[3];  // Position delta (x, y, z) in meters
model.forward(input, output);
```

### 3. Batch inference

```cpp
float inputs[1000 * 14];
float outputs[1000 * 3];
model.forward_batch(inputs, outputs, 1000);
```

## Building

### Scalar (portable)
```bash
g++ -std=c++17 -O3 -ffast-math your_code.cpp -o your_program
```

### With AVX2 (x86-64)
```bash
g++ -std=c++17 -O3 -mavx2 -mfma -ffast-math your_code.cpp -o your_program
```

### With NEON (ARM)
```bash
g++ -std=c++17 -O3 -ffast-math your_code.cpp -o your_program
# NEON is auto-enabled on ARM
```

## Weight Format (NPZ)

The `int8_weights.npz` file contains:

```
layer_0_weight_int8   [512, 14]   INT8
layer_0_weight_scale  [512]       FP16 (per-channel)
layer_0_bias          [512]       FP16
layer_0_ln_weight     [512]       FP16
layer_0_ln_bias       [512]       FP16
...
layer_4_weight_int8   [256, 256]  INT8
layer_4_weight_scale  [256]       FP16
layer_4_bias          [256]       FP16
layer_4_ln_weight     [256]       FP16
layer_4_ln_bias       [256]       FP16

output_weight_int8    [3, 256]    INT8
output_weight_scale   [3]         FP16
output_bias           [3]         FP16
```

## Architecture

```
Input [14]
  │
  ▼
Linear_INT8 [14 → 512] + LayerNorm + SiLU
  │
  ▼
Linear_INT8 [512 → 512] + LayerNorm + SiLU
  │
  ▼
Linear_INT8 [512 → 512] + LayerNorm + SiLU
  │
  ▼
Linear_INT8 [512 → 256] + LayerNorm + SiLU
  │
  ▼
Linear_INT8 [256 → 256] + LayerNorm + SiLU
  │
  ▼
Linear_INT8 [256 → 3]
  │
  ▼
Output [3] (position delta in meters)
```

## License

HyperFlux v16.0 - Neural Physics Engine
Copyright (c) 2026 HyperFlux Team
