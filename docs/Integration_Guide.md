# HyperFlux Mount Dynamics SDK v1.0

## Quick Start

### 1. Add to Your Project

**Include the header:**
```c
#include "HyperFluxMount.h"
```

**Link the library:**
- Linux: `libHyperFluxMount.a`
- Windows: `HyperFluxMount.lib`

### 2. Initialize

```c
// Call once at game startup
if (!HyperFlux_Initialize()) {
    printf("Failed to initialize HyperFlux\n");
}
```

### 3. Use in Game Loop

```c
// For each mount, every frame:
HyperFluxMountInput input = {0};
input.type = HYPERFLUX_MOUNT_PANTHER;
input.velocityX = mount->vel.x;
input.velocityY = mount->vel.y;
input.velocityZ = mount->vel.z;
input.currentSpeed = sqrtf(input.velocityX*input.velocityX + input.velocityZ*input.velocityZ);
input.throttle = player->gas;
input.steering = player->steer;
input.jumpInput = player->jump ? 1.0f : 0.0f;
input.terrainSlope = GetTerrainSlope(mount->pos);
input.terrainFacing = GetTerrainFacing(mount->pos, mount->forward);
input.isGrounded = mount->onGround;
input.stamina = mount->stamina;

HyperFluxMountOutput output;
if (HyperFlux_Predict(&input, &output)) {
    mount->vel.x += output.deltaVelocityX;
    mount->vel.y += output.deltaVelocityY;
    mount->vel.z += output.deltaVelocityZ;
}
```

### 4. Shutdown

```c
// Call once at game exit
HyperFlux_Shutdown();
```

---

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.90% (198 errors in 200K tests) |
| Latency | <5 µs per mount |
| Speedup | 100×+ vs traditional physics |
| Memory | ~250 KB |

---

## Mount Types

| Type | Max Speed | Acceleration | Turn Rate | Jump Force |
|------|-----------|--------------|-----------|------------|
| HORSE | 15 m/s | 8 m/s² | 2.5 rad/s | 6 m/s |
| PANTHER | 20 m/s | 12 m/s² | 4.0 rad/s | 8 m/s |
| BEAR | 12 m/s | 5 m/s² | 1.8 rad/s | 4 m/s |
| GRYPHON | 25 m/s | 10 m/s² | 3.0 rad/s | 12 m/s |

---

## API Reference

```c
int HyperFlux_Initialize(void);
void HyperFlux_Shutdown(void);
int HyperFlux_IsInitialized(void);
int HyperFlux_Predict(const HyperFluxMountInput* input, HyperFluxMountOutput* output);
int HyperFlux_PredictBatch(const HyperFluxMountInput* inputs, HyperFluxMountOutput* outputs, int count);
const char* HyperFlux_GetVersion(void);
void HyperFlux_GetMountParams(HyperFluxMountType type, float* maxSpeed, float* accel, float* turnRate, float* jumpForce);
```

---

## Support

- Email: support@hyperflux.ai
- Website: https://hyperflux.ai

© 2026 HyperFlux Neural Dynamics. All Rights Reserved.
