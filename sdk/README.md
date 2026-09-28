# S.O.M.A. v7 — C SDK

**Self-Optimizing Motor Architecture** | Neural Dynamics Inc.

A 344KB neural controller that replaces hand-tuned PD controllers on any robot.
Zero dependencies. Pure C99. Runs on anything with a C compiler.

## What is SOMA?

SOMA sits between the robot's brain (VLA/policy) and its motors — the "spinal cord" layer:

```
┌───────────────────────────────────┐
│  BRAIN (VLA / Policy / Planner)   │  ← "pick up the cup"
│  Runs at 10-200 Hz, GPU           │
└───────────────────────────────────┘
              ↓ joint targets
┌───────────────────────────────────┐
│  SOMA (Spinal Cord)              │  ← converts targets to torques
│  Runs at 1000 Hz, CPU, 344KB     │
│  • Dynamics compensation          │
│  • Safety certificates            │
│  • Perturbation adaptation        │
└───────────────────────────────────┘
              ↓ torque commands
┌───────────────────────────────────┐
│  MOTORS                           │
└───────────────────────────────────┘
```

## Performance

| Metric                    | Value                        |
|---------------------------|------------------------------|
| Model size                | 88,023 params / 344 KB       |
| Inference (NumPy CPU)     | 131 µs                       |
| Inference (C SDK, est.)   | 20-50 µs                     |
| Tracking accuracy         | 2.37° across all trajectories|
| Safety violations         | 0 (under 15N perturbation)   |
| Frequency invariant       | 200 Hz – 2000 Hz, same weights|
| Training time             | ~20 minutes on 1 GPU         |

## Quick Start

### 1. Export weights from checkpoint

```bash
python export_weights.py /path/to/soma_v7_checkpoint.pt ./weights
```

### 2. Build

```bash
# Option A: Direct compilation (simplest)
gcc -O2 -o soma_example examples/example_standalone.c -I include -lm
./soma_example ./weights

# Option B: CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./soma_example ../weights
```

### 3. Integrate

```c
#define SOMA_IMPLEMENTATION
#include "soma.h"

// Init once (loads ~420KB of weight files)
SomaHandle* soma = soma_init("./weights");

// Call every control cycle (1kHz)
float state[28] = { /* q_err, q_vel, grav_comp, q_ref */ };
float torque[7];
soma_step(soma, state, 0.002f, torque);
// → torque[] now contains motor commands

// Reset hidden state (e.g., new trajectory)
soma_reset(soma);

// Cleanup
soma_free(soma);
```

## API

```c
// Initialize: loads all weights from binary files in directory
SomaHandle* soma_init(const char* weights_dir);

// Inference: one control step, ~30-50µs on ARM CPU
//   state:      [q_error(nu), q_vel(nu), grav_comp(nu), q_ref(nu)]
//   dt:         timestep in seconds (e.g., 0.001 for 1kHz)
//   torque_out: output torque commands (nu dimensions)
void soma_step(SomaHandle* h, const float* state, float dt, float* torque_out);

// Reset hidden state to zeros
void soma_reset(SomaHandle* h);

// Free all resources
void soma_free(SomaHandle* h);

// Optional: inspect hidden state (208 floats)
const float* soma_get_hidden(const SomaHandle* h);
int soma_get_hidden_size(void);
```

## File Structure

```
soma_sdk/
├── include/
│   └── soma.h                    ← Single-header library (header + implementation)
├── examples/
│   └── example_standalone.c      ← Benchmark & test program
├── ros2_control/
│   └── soma_ros2_controller.hpp  ← ros2_control plugin reference
├── unitree/
│   └── soma_unitree.hpp          ← Unitree G1/H1 integration reference
├── export_weights.py             ← Checkpoint → binary weight converter
├── CMakeLists.txt                ← Build system
└── README.md                     ← This file
```

## Weight Files

After running `export_weights.py`, the weights directory contains:

```
weights/
├── manifest.json              ← Architecture config + metadata
├── input_proj_weight.bin      ← (128, 28) = 3,584 floats
├── input_proj_bias.bin        ← (128,) 
├── inter_cell_ff_weight.bin   ← (128, 257) = 32,896 floats  [largest]
├── inter_cell_ff_bias.bin     ← (128,)
├── inter_cell_gg_weight.bin   ← (128, 257) = 32,896 floats
├── inter_cell_gg_bias.bin     ← (128,)
├── command_cell_ff_weight.bin ← (64, 129) = 8,256 floats
├── ...                        ← (remaining cells + layer norms)
├── output_proj_weight.bin     ← (7, 16) = 112 floats
├── output_proj_bias.bin       ← (7,)
├── mask_ic.bin                ← (128, 64) sparse connectivity
├── mask_cm.bin                ← (64, 16) sparse connectivity
├── norm_mean.bin              ← (28,) state normalizer
└── norm_std.bin               ← (28,) state normalizer
```

Total: ~420 KB

## Dependencies

**None.** The entire SDK uses only:

- `<math.h>` — `tanhf()`, `expf()`, `sqrtf()`
- `<string.h>` — `memcpy()`, `memset()`
- `<stdio.h>` — `fopen()`, `fread()` (weight loading only)
- `<stdlib.h>` — `malloc()`, `free()` (init/free only)

Compiles with any C99 compiler: GCC, Clang, MSVC, ARM GCC, etc.
No BLAS, no LAPACK, no Python, no PyTorch, no framework.

## Platform Support

| Platform               | Status | Notes                          |
|------------------------|--------|--------------------------------|
| x86_64 Linux           | ✅     | Primary development            |
| ARM64 (Jetson Orin)    | ✅     | Unitree G1 dev computer        |
| ARM Cortex-A (RPi)     | ✅     | General robotics               |
| Any POSIX system       | ✅     | Needs `clock_gettime` for bench|
| Windows                | ✅     | Replace timer in example       |
| Bare metal / RTOS      | ✅     | No file I/O needed at runtime* |

*For bare metal: compile weights into a C array with `xxd -i` and point
`soma_init` at the memory, or modify to load from flash.

## Integration Guides

### ros2_control
See `ros2_control/soma_ros2_controller.hpp` for the integration pattern.
SOMA becomes a `ControllerInterface` plugin that reads `position` and
`velocity` state interfaces and writes `effort` command interfaces.

### Unitree G1/H1
See `unitree/soma_unitree.hpp` for the integration pattern.
Uses `unitree_sdk2` C++ API with CycloneDDS to read `/rt/lowstate`
and publish torques to `/rt/lowcmd`.

### Custom integration
The API is 3 functions. Call `soma_init()` once, then `soma_step()`
every control cycle. That's it.

## Architecture

SOMA uses a Closed-form Continuous-time (CfC) neural network with
Neural Circuit Policy (NCP) wiring — a biologically-inspired architecture
with three neuron populations:

- **Inter neurons** (128): Process sensory input
- **Command neurons** (64): Integrate and plan
- **Motor neurons** (16): Generate torque commands

Key properties:
- **Time-continuous**: Built-in time constants adapt to any control frequency
- **Sparse wiring**: NCP connectivity reduces computation
- **Recurrent**: Hidden state carries dynamics information across timesteps

## License

Proprietary — Neural Dynamics Inc. All rights reserved.
Contact: anil@neuraldynamics.ai
