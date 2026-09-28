# SOMA v8 — Pick Up Here

## What is SOMA?
Full System 0 competitor. Whole-body robot controller. 10kHz, O(1), certified safe, any URDF in 20 minutes.

**Key differentiator from System 0:** We don't use RL. We train CfC directly through differentiable physics (MuJoCo MJX). Exact gradients, not noisy policy gradients. Minutes, not months.

## Architecture (2 kernels, unchanged)
```
Kernel 1: CfC Controller     <- 51K params, ~65us, 15kHz, freq-invariant
    | raw torques
    v
Kernel 2: CBF Safety Filter   <- analytical, ~5us, joint/vel/rate/tilt limits
    | safe torques
    v
Motors
```

## Training method (THE change)
```
OLD (broken):   MPPI teacher -> expert data -> CfC learns from data (teacher too weak)
RL (rejected):  PPO teacher -> expert data -> CfC learns from data (just System 0 clone)
NEW (build this):
    CfC outputs torques -> MJX simulates physics (differentiable!) -> loss
         ^                                                            |
         +-------------- exact gradients via jax.grad ----------------+
    
    No teacher. No expert data. No distillation. No reward engineering.
    CfC learns by backpropagating through the physics engine directly.
```

## Why this beats RL/System 0
| | System 0 (RL) | SOMA (Diff Sim) |
|---|---|---|
| Training time | 1000s GPU-hours | 20 minutes |
| Training signal | Noisy policy gradient | Exact analytic gradient |
| Sample efficiency | Billions of steps | Thousands of steps |
| New robot | Retrain from scratch (days) | Retrain from URDF (20 min) |
| Reward hacking | Yes (common RL failure) | No (loss = direct physics) |

## Three Moats
1. **Safety** - CBF hard constraints, ISO 13849 path
2. **Frequency invariance** - sigma=sigmoid(ta*dt+tb), same weights at any Hz
3. **O(1) constant time** - CfC forward pass + analytical CBF, ~70us total

## Why 10kHz matters
At 1kHz: joint at 10 rad/s travels 0.01 rad between safety checks.
At 10kHz: 0.001 rad. 10x tighter safety margins. That's the pitch to factories.

## Current state of code
- soma_v8_runtime_b.py exists (1340 lines, PyTorch, MPPI teacher - old approach)
- Sections 1-3 (robot setup, standing, helpers) - KEEP
- Section 4 (MPPI solver) - DELETE, replaced by diff sim
- Sections 5-6 need FULL REWRITE: PyTorch to JAX, MPPI to differentiable sim
- Section 7 CBF safety filter - KEEP (numpy, framework-agnostic)
- Sections 8-10 eval/benchmark/report - update to use JAX CfC

## What to build in this session
Rewrite soma_v8_runtime_b.py:
1. New section 4: JAX/MJX setup, differentiable rollout function
2. New section 5: CfC in JAX (Equinox), trajectory loss, train via jax.grad through physics
3. Keep section 6 CBF (renumber)
4. Section 7 eval: test CfC+CBF on push recovery
5. Section 8 latency benchmark
6. Test on Go2: push recovery via direct gradient optimization

## Competitors
- Figure Helix S0: 1kHz, RL-trained, black-box, uncertified
- Boston Dynamics: hand-tuned MPC, years per robot
- Unitree: RL sim-to-real
- None have: certified safety, 10kHz, O(1), 20-min-from-URDF

## Technical notes
- MJX = MuJoCo in JAX (jit-compiled, GPU-accelerated, autodiff-compatible)
- CfC math is identical in JAX vs PyTorch - just framework swap
- Key challenge: gradient stability through long rollouts (may need short horizon + curriculum)
- CBF can also be made differentiable (train CfC to satisfy safety during training)
