# HyperFlux SDK — Mac Build & Test

## Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)

## Quick Start

```bash
# 1. Build all 10 kernels
./build.sh

# 2. Run integration tests
./test.sh
```

## What's Inside
- `kernels/` — 10 neural physics kernels (source + weights + headers)
- `tests/` — Integration test for each kernel
- `build.sh` — Compiles all kernels for Apple Silicon
- `test.sh` — Builds and runs the test harness

## Expected Results
All kernels should:
- Initialize without errors
- Produce outputs in valid ranges
- Run at <50µs per inference on Apple Silicon

## Notes
- This is an INTERNAL build for testing — not customer delivery
- Customer delivery contains only headers + pre-compiled .a (no source/weights)
- The `-Wl,-force_load` flag resolves the HyperFlux_Predict symbol conflict
  between Ballistics and MountDynamics (only relevant when linking all 10 together)
