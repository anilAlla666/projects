# HyperFlux SDK — Integration Guide
### Version 1.0.0 | Neural Dynamics | CONFIDENTIAL

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Build Instructions](#3-build-instructions)
4. [Quick Start](#4-quick-start)
5. [API Reference](#5-api-reference)
6. [Kernel Specifications](#6-kernel-specifications)
7. [Threading & Concurrency](#7-threading--concurrency)
8. [Performance Guide](#8-performance-guide)
9. [Unreal Engine Integration](#9-unreal-engine-integration)
10. [Debug & Profiling](#10-debug--profiling)
11. [FAQ & Troubleshooting](#11-faq--troubleshooting)

---

## 1. Overview

HyperFlux SDK replaces traditional iterative game algorithms with constant-time (O(1)) inference.
Where conventional implementations scale linearly or quadratically with scene complexity—more
players, more objects, more physics steps—HyperFlux evaluates the same function in fixed time
regardless of input complexity.

**What this means for your game:**

- **Ballistics:** Full exterior ballistics (drop, drift, drag, spin) in ~2μs instead of iterative
  Runge-Kutta integration at ~200μs+.
- **Spawn Selection:** Score all 20 spawn points against 64 players in ~3μs instead of O(N²) threat
  assessment at ~300μs.
- **Visibility:** Line-of-sight determination in ~1μs instead of multi-ray occlusion queries.
- **All 10 kernels combined:** Under 30μs total per frame. The entire SDK is 2.7MB.

HyperFlux ships as a static library (`libhyperflux.a` / `hyperflux.lib`) with a single header
include. No external dependencies. No GPU required. No runtime model loading. Everything is
compiled in.

### Supported Platforms

| Platform | Architecture | SIMD Backend | Status |
|----------|-------------|-------------|--------|
| Windows | x86-64 | AVX2 / AVX-512 / VNNI | ✅ |
| Linux | x86-64 | AVX2 / AVX-512 / VNNI | ✅ |
| macOS | ARM64 (Apple Silicon) | NEON | ✅ |
| macOS | x86-64 (Intel) | AVX2 | ✅ |
| Linux | ARM64 | NEON | ✅ |
| Consoles | Contact us | Optimized per-platform | 🔜 |

### Minimum Requirements

- C++17 compiler (MSVC 2019+, GCC 9+, Clang 11+, Apple Clang 13+)
- CMake 3.16+
- ~3MB disk for static library + weights
- ~512KB runtime memory per kernel instance (stack + scratch buffers)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Your Game Code                     │
│          #include <hyperflux/hyperflux.h>           │
├─────────────────────────────────────────────────────┤
│            HyperFlux Public API Layer               │
│   Ballistics · Hitbox · Visibility · SpawnSelect    │
│   MountDynamics · Destruction · LumenGI · Audio     │
│   OcclusionCulling · LODSelection                   │
├─────────────────────────────────────────────────────┤
│          Inference Engine (internal)                 │
│   INT8 quantized computation · memory pool          │
├─────────────────────────────────────────────────────┤
│          SIMD Backend (auto-detected)               │
│   AVX2 │ AVX-512 │ VNNI │ NEON │ Scalar            │
├─────────────────────────────────────────────────────┤
│          Compiled Weight Data                       │
│   Obfuscated · integrity-checked · read-only        │
└─────────────────────────────────────────────────────┘
```

Each kernel is a self-contained class. No kernel shares state with another. The inference engine
and SIMD backend are shared but stateless—the only mutable state is the per-kernel scratch buffer
used during computation.

---

## 3. Build Instructions

### CMake (Recommended)

```bash
# Clone or copy HyperFlux_SDK into your project
cd HyperFlux_SDK

# Basic build (static library, AVX2)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# With AVX-512 support (server/PC builds)
cmake .. -DCMAKE_BUILD_TYPE=Release -DHF_ENABLE_AVX512=ON
make -j$(nproc)

# With debug logging
cmake .. -DCMAKE_BUILD_TYPE=Debug -DHF_DEBUG=ON
make -j$(nproc)

# With profiling (timing per inference)
cmake .. -DCMAKE_BUILD_TYPE=Release -DHF_PROFILE=ON
make -j$(nproc)
```

### Integrate into Existing CMake Project

```cmake
add_subdirectory(vendor/HyperFlux_SDK)
target_link_libraries(YourGame PRIVATE HyperFlux::hyperflux)
```

### Manual Compilation (if not using CMake)

```bash
# Collect all .cpp files
SOURCES=$(find src -name '*.cpp')

# x86-64 with AVX2
g++ -std=c++17 -O3 -mavx2 -mfma -ffast-math -c $SOURCES -Iinclude -Isrc
ar rcs libhyperflux.a *.o

# ARM64 with NEON
clang++ -std=c++17 -O3 -ffast-math -c $SOURCES -Iinclude -Isrc
ar rcs libhyperflux.a *.o
```

---

## 4. Quick Start

```cpp
#include <hyperflux/hyperflux.h>
#include <cstdio>

int main() {
    // 1. Initialize library (detects SIMD, sets up dispatch)
    hyperflux::Status status = hyperflux::Init();
    printf("HyperFlux %s | Backend: %s\n",
           hyperflux::GetVersion(), hyperflux::GetBackendName());

    // 2. Create and initialize a kernel
    hyperflux::Ballistics ballistics;
    ballistics.Init();

    // 3. Set up input
    hyperflux::BallisticsInput input = {};
    input.origin[0] = 0.0f;  input.origin[1] = 1.5f;   input.origin[2] = 0.0f;
    input.direction[0] = 1.0f; input.direction[1] = 0.01f; input.direction[2] = 0.0f;
    input.muzzle_velocity = 800.0f;
    input.projectile_mass = 0.0042f;
    input.drag_coefficient = 0.295f;
    input.gravity = 9.81f;
    input.air_density = 1.225f;
    input.temperature = 20.0f;

    // 4. Run inference
    hyperflux::BallisticsOutput output;
    status = ballistics.Infer(input, &output);

    if (status == hyperflux::Status::OK) {
        printf("Impact: (%.2f, %.2f, %.2f)\n",
               output.impact_point[0], output.impact_point[1], output.impact_point[2]);
        printf("Drop: %.4f m | Drift: %.4f m | ToF: %.4f s\n",
               output.bullet_drop, output.wind_drift, output.time_of_flight);
    }

    // 5. Cleanup
    ballistics.Shutdown();
    hyperflux::Shutdown();
    return 0;
}
```

**Compile and run:**
```bash
g++ -std=c++17 -O3 -mavx2 -o quick_test quick_test.cpp -Lbuild -lhyperflux -Iinclude
./quick_test
```

---

## 5. API Reference

### Library Lifecycle

| Function | Description |
|----------|-------------|
| `hyperflux::Init(config)` | Initialize runtime. Call once at startup. Optional `InitConfig` for backend preference. |
| `hyperflux::Shutdown()` | Release runtime resources. Call once at exit. |
| `hyperflux::GetVersion()` | Returns `"1.0.0"` |
| `hyperflux::GetBackendName()` | Returns active SIMD backend: `"AVX2"`, `"NEON"`, etc. |
| `hyperflux::GetSimdCapabilities()` | Returns bitmask of detected SIMD features. |

### Kernel Lifecycle (same for all 10 kernels)

```cpp
hyperflux::Ballistics kernel;   // Construct
kernel.Init();                  // Initialize (loads weights, allocates scratch)
kernel.Infer(input, &output);   // Single inference
kernel.InferBatch(in, out, n);  // Batch inference (up to 64)
kernel.IsReady();               // Check initialization state
kernel.GetStats();              // Get timing statistics (profile mode)
kernel.Shutdown();              // Release resources
// Destructor also calls Shutdown() automatically
```

### Status Codes

| Code | Value | Meaning |
|------|-------|---------|
| `OK` | 0 | Success |
| `ERROR_NOT_INITIALIZED` | -1 | Kernel not initialized. Call `Init()` first. |
| `ERROR_INVALID_INPUT` | -2 | Null pointer or malformed input. |
| `ERROR_INVALID_BATCH_SIZE` | -3 | Batch size 0 or > 64. |
| `ERROR_OUT_OF_MEMORY` | -4 | Scratch buffer allocation failed. |
| `ERROR_WEIGHT_CORRUPT` | -6 | Weight integrity check failed. |
| `ERROR_INTERNAL` | -99 | Unexpected internal error. |

### InitConfig Options

```cpp
hyperflux::InitConfig config;
config.preferred_backend = hyperflux::SimdBackend::AUTO;  // Let SDK choose best
config.enable_profiling = true;   // Track per-inference timing
config.enable_validation = true;  // Range-check inputs (slower, for debugging)
hyperflux::Init(config);
```

---

## 6. Kernel Specifications

### Ballistics

Replaces iterative Runge-Kutta trajectory integration with constant-time evaluation.

| Property | Value |
|----------|-------|
| Typical latency | ~2 μs |
| Relative error | < 4% vs reference solver |
| Model size | 482 KB (INT8) |

**Input:** Projectile initial conditions (position, velocity, mass, drag, wind, environment).
**Output:** Impact point, velocity at impact, bullet drop, wind drift, time of flight, energy.

**Usage notes:**
- All positions in world-space meters.
- Direction must be normalized.
- Temperature affects air density internally—provide ambient temperature for best accuracy.
- For tracer rendering, call multiple times with decreasing `max_range` to sample the arc.

---

### Hitbox

Replaces per-bone ray-capsule intersection tests.

| Property | Value |
|----------|-------|
| Typical latency | ~1.5 μs |
| Relative error | < 2.1% |
| Model size | 102 KB (INT8) |

**Input:** Ray origin/direction, bone transforms (up to 16 bones).
**Output:** Per-bone hit probability (0-1). Threshold at 0.5 for binary hit/miss.

---

### Visibility

Replaces multi-ray line-of-sight occlusion queries.

| Property | Value |
|----------|-------|
| Typical latency | ~1.5 μs |
| Accuracy | 99.93% |
| Model size | 502 KB (INT8) |

**Input:** Observer/target positions, occluder spatial hash, atmospheric conditions.
**Output:** Visibility probability (0 = fully occluded, 1 = fully visible).

**Usage notes:**
- For AI perception, threshold at 0.5.
- For UI indicators (spotted icon), use raw probability for fade.
- Occluder data should be pre-encoded via your spatial hash system. See encoding spec below.

---

### Spawn Selection

Replaces O(N²) threat/safety scoring across all spawn points.

| Property | Value |
|----------|-------|
| Typical latency | ~3 μs |
| Top-1 agreement | 92% |
| Teacher-top-1-in-top-3 | 99.7% |
| Model size | 520 KB (INT8) |

**Input:** Full match state (player positions, teams, objectives, combat zones, spawn heat).
**Output:** Scores for all 20 spawn points, sorted ranking, top recommendation.

**Usage notes:**
- `ranked[0]` through `ranked[4]` give the top 5 spawns.
- Randomize selection from top 3-5 to prevent predictable spawning.
- Update `spawn_heat[]` with exponential decay to prevent clustering.
- Zero-pad player arrays beyond active `player_count`.

---

### Mount Dynamics

Replaces iterative vehicle physics solver.

| Property | Value |
|----------|-------|
| Typical latency | ~1 μs |
| Accuracy | 99.9% |
| Model size | 219 KB (INT8) |

**Input:** Current mount state (position, velocity, orientation), control inputs, terrain.
**Output:** Next-frame predicted state.

---

### Destruction

Replaces iterative structural damage propagation.

| Property | Value |
|----------|-------|
| Typical latency | ~1 μs |
| Accuracy | 99.99% |
| Model size | 452 KB (INT8) |

**Input:** Impact parameters, material properties, structural element state.
**Output:** Damage propagation, fracture probability, collapse risk.

---

### Lumen GI

Replaces real-time global illumination probe evaluation.

| Property | Value |
|----------|-------|
| Typical latency | ~1 μs |
| Relative error | < 1.8% |
| Model size | 94 KB (INT8) |

**Input:** Probe position, nearby light sources, bounce colors, ambient occlusion.
**Output:** Spherical harmonics L2 coefficients (27 values: RGB × 9 SH), dominant direction, luminance.

---

### Audio Occlusion

Replaces multi-path sound propagation raycast.

| Property | Value |
|----------|-------|
| Typical latency | ~1 μs |
| Relative error | < 2.2% |
| Model size | 95 KB (INT8) |

**Input:** Listener/source positions, nearby walls, openings, materials.
**Output:** Attenuation, occlusion, reverb parameters, EQ adjustments, perceived direction.

---

### Occlusion Culling

Replaces GPU-based or software occlusion queries.

| Property | Value |
|----------|-------|
| Typical latency | ~1.5 μs |
| Relative error | < 2.2% |
| Model size | 94 KB (INT8) |

**Input:** Camera parameters, object positions and bounding spheres (up to 16 per call).
**Output:** Per-object visibility (0 = culled, 1 = visible). Threshold at 0.5.

---

### LOD Selection

Replaces heuristic LOD selection logic.

| Property | Value |
|----------|-------|
| Typical latency | ~1 μs |
| Relative error | < 2.9% |
| Model size | 92 KB (INT8) |

**Input:** Camera state, object state, triangle counts per LOD, GPU budget.
**Output:** Score per LOD level. Select `argmax(scores)`.

---

## 7. Threading & Concurrency

### Thread Safety Guarantees

- `hyperflux::Init()` / `Shutdown()` — **NOT** thread-safe. Call from main thread only.
- Kernel instances — **Thread-safe between instances.** Each kernel object owns its own scratch
  buffer. Two different `Ballistics` instances can run `Infer()` concurrently from different threads.
- **NOT safe:** Calling `Infer()` on the **same** kernel instance from two threads simultaneously.

### Recommended Pattern: Per-Thread Kernel Instances

```cpp
// Game initialization
hyperflux::Init();

// Per worker thread
void PhysicsWorker(int thread_id) {
    hyperflux::Ballistics ballistics;  // Each thread gets its own
    ballistics.Init();

    while (running) {
        BallisticsInput in = GetNextProjectile();
        BallisticsOutput out;
        ballistics.Infer(in, &out);
        // ... use result
    }

    ballistics.Shutdown();
}
```

### Alternative: Pool of Kernel Instances

```cpp
// Pre-create a pool of N instances
std::vector<hyperflux::Ballistics> pool(num_threads);
for (auto& k : pool) k.Init();

// Worker grabs instance by thread index
void Worker(int tid) {
    pool[tid].Infer(input, &output);
}
```

### Memory Footprint

Each kernel instance uses ~512KB of scratch memory. For 10 kernels × 8 threads = 80 instances ≈
40MB. This is well within typical game memory budgets.

---

## 8. Performance Guide

### Latency Characteristics

All latencies measured on Intel i9-13900K (AVX2) and Apple M2 Pro (NEON):

| Kernel | x86-64 AVX2 | ARM64 NEON | Batch-4 AVX2 |
|--------|------------|-----------|-------------|
| Ballistics | 2.1 μs | 3.4 μs | 6.8 μs |
| Hitbox | 1.4 μs | 2.1 μs | 4.2 μs |
| Visibility | 1.6 μs | 2.5 μs | 5.0 μs |
| SpawnSelection | 3.2 μs | 5.1 μs | 10.5 μs |
| MountDynamics | 1.0 μs | 1.6 μs | 3.2 μs |
| Destruction | 1.0 μs | 1.5 μs | 3.1 μs |
| LumenGI | 1.1 μs | 1.7 μs | 3.4 μs |
| AudioOcclusion | 1.0 μs | 1.6 μs | 3.2 μs |
| OcclusionCulling | 1.4 μs | 2.1 μs | 4.3 μs |
| LODSelection | 0.9 μs | 1.4 μs | 2.8 μs |

### Optimization Tips

1. **Warm up on first frame.** The first `Infer()` call after `Init()` may be ~2× slower due to
   cache cold start. Call once with dummy data during loading screen.

2. **Use batch inference** when processing multiple entities. `InferBatch()` amortizes function call
   overhead and improves cache utilization.

3. **Avoid Init/Shutdown per frame.** Initialize kernels once during level load, keep them alive
   until level unload.

4. **Memory layout matters.** If processing many entities, consider SoA (struct-of-arrays) layout
   for your game data and marshal into HyperFlux input structs.

5. **AVX-512 is not always faster** on consumer CPUs due to frequency throttling. Test both
   `AVX2` and `AVX512` backends on your target hardware. Force via `InitConfig::preferred_backend`.

---

## 9. Unreal Engine Integration

### Step 1: Add as a Third-Party Library

Place the SDK in your project:
```
YourProject/
  Source/
    ThirdParty/
      HyperFlux/
        include/
        lib/
          Win64/
            hyperflux.lib
          Mac/
            libhyperflux.a
          Linux/
            libhyperflux.a
```

### Step 2: Update Build.cs

```csharp
// YourProject.Build.cs
using UnrealBuildTool;

public class YourProject : ModuleRules
{
    public YourProject(ReadOnlyTargetRules Target) : base(Target)
    {
        // ... existing config ...

        string HyperFluxPath = Path.Combine(ModuleDirectory, "../ThirdParty/HyperFlux");
        PublicIncludePaths.Add(Path.Combine(HyperFluxPath, "include"));

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PublicAdditionalLibraries.Add(
                Path.Combine(HyperFluxPath, "lib/Win64/hyperflux.lib"));
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            PublicAdditionalLibraries.Add(
                Path.Combine(HyperFluxPath, "lib/Mac/libhyperflux.a"));
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            PublicAdditionalLibraries.Add(
                Path.Combine(HyperFluxPath, "lib/Linux/libhyperflux.a"));
        }
    }
}
```

### Step 3: Create a Subsystem Wrapper

```cpp
// HyperFluxSubsystem.h
#pragma once
#include "Subsystems/GameInstanceSubsystem.h"
#include <hyperflux/hyperflux.h>

UCLASS()
class UHyperFluxSubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override
    {
        hyperflux::Init();
        Ballistics.Init();
        Visibility.Init();
        SpawnSelection.Init();
        // ... init other kernels as needed
    }

    virtual void Deinitialize() override
    {
        Ballistics.Shutdown();
        Visibility.Shutdown();
        SpawnSelection.Shutdown();
        hyperflux::Shutdown();
    }

    hyperflux::Ballistics Ballistics;
    hyperflux::Visibility Visibility;
    hyperflux::SpawnSelection SpawnSelection;
};
```

### Step 4: Use in Game Code

```cpp
// In your projectile actor or wherever you compute trajectories
auto* HF = GetGameInstance()->GetSubsystem<UHyperFluxSubsystem>();

hyperflux::BallisticsInput Input = {};
// ... fill from UE world state ...

hyperflux::BallisticsOutput Output;
HF->Ballistics.Infer(Input, &Output);

// Use Output.impact_point to set projectile destination
FVector ImpactPoint(Output.impact_point[0], Output.impact_point[1], Output.impact_point[2]);
```

---

## 10. Debug & Profiling

### Debug Build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DHF_DEBUG=ON
```

Enables `[HyperFlux]` log messages to stderr:
```
[HyperFlux] SIMD backend: AVX2 (flags=0x01)
[HyperFlux] HyperFlux SDK v1.0.0 initialized
[HyperFlux] Ballistics initialized: 5 layers, 24->12
```

### Profiling Build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DHF_PROFILE=ON
```

Logs per-inference timing:
```
[HyperFlux] PROFILE infer: 2.14 us
```

Access programmatically:
```cpp
hyperflux::KernelStats stats = ballistics.GetStats();
printf("Avg: %.2f us, Min: %.2f us, Max: %.2f us, Count: %llu\n",
       stats.avg_inference_us, stats.min_inference_us,
       stats.max_inference_us, stats.total_inferences);
```

### Input Validation

Enable in `InitConfig` to catch malformed inputs early:
```cpp
hyperflux::InitConfig config;
config.enable_validation = true;
hyperflux::Init(config);
```

When enabled, `Infer()` will return `ERROR_INVALID_INPUT` if values contain NaN, Inf, or
are outside expected ranges.

---

## 11. FAQ & Troubleshooting

**Q: Do I need a GPU?**
No. HyperFlux runs entirely on CPU using SIMD instructions. It's designed for the game thread,
not the render thread.

**Q: What happens if AVX2 isn't available (old CPU)?**
The SDK falls back to a portable scalar implementation. It's ~4× slower but still significantly
faster than the algorithms it replaces.

**Q: Can I use this in a dedicated server (headless Linux)?**
Yes. HyperFlux has no graphics dependencies. Servers with AVX2-capable CPUs (virtually all
modern server hardware) get the fastest path.

**Q: How do I update to new kernel weights?**
Contact Neural Dynamics for updated weight files. Use `tools/export_weights.py` to convert
and recompile.

**Q: What's the memory overhead?**
~512KB per kernel instance for scratch buffers. The compiled weights add ~2.7MB to your binary
(all 10 kernels combined). Total runtime: under 10MB even with 8 threads × 10 kernels.

**Q: Is the output deterministic?**
Yes. Given identical inputs on the same hardware with the same SIMD backend, output is bit-exact.
Different SIMD backends may produce slightly different results (< 0.01% difference) due to
floating-point ordering.

**Q: Can I use only some kernels?**
Yes. Only `Init()` and the specific kernel constructors you use will pull in weights. However,
since weights are compiled in, the full 2.7MB is included in the binary regardless. Contact us
for a custom build with only selected kernels if binary size is critical.

**Q: Compile error: 'immintrin.h' not found?**
You're likely compiling for ARM without the right flags. Ensure your build system detects the
architecture correctly. On Apple Silicon, do NOT pass `-mavx2`.

**Q: I'm getting `ERROR_WEIGHT_CORRUPT` at Init()?**
The compiled weight data failed its integrity check. This usually means the `.inc` files were
modified or a build system issue corrupted the data. Regenerate weights with `export_weights.py`.

---

*For support, contact: sdk-support@neuraldynamics.ai*
*This document is confidential. Do not distribute outside your development team.*
