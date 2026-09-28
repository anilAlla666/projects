// ==========================================================================
// HyperFlux SDK v1.0.0
// O(1) Inference Engine for Real-Time Game Systems
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// CONFIDENTIAL AND PROPRIETARY - Do not distribute.
// ==========================================================================
//
// Usage:
//   #include <hyperflux/hyperflux.h>
//
//   hyperflux::Init();
//
//   hyperflux::Ballistics ballistics;
//   ballistics.Init();
//
//   hyperflux::BallisticsInput in = { ... };
//   hyperflux::BallisticsOutput out;
//   ballistics.Infer(in, &out);
//
//   ballistics.Shutdown();
//   hyperflux::Shutdown();
//
#pragma once

#include "hyperflux_config.h"
#include "hyperflux_types.h"

// Kernel headers
#include "kernels/hf_ballistics.h"
#include "kernels/hf_hitbox.h"
#include "kernels/hf_visibility.h"
#include "kernels/hf_spawn_selection.h"
#include "kernels/hf_mount_dynamics.h"
#include "kernels/hf_destruction.h"
#include "kernels/hf_lumen_gi.h"
#include "kernels/hf_audio_occlusion.h"
#include "kernels/hf_occlusion_culling.h"
#include "kernels/hf_lod_selection.h"

namespace hyperflux {

/// Initialize the HyperFlux runtime. Call once at application startup.
Status Init(const InitConfig& config = InitConfig{});

/// Shutdown the HyperFlux runtime. Call once at application exit.
void Shutdown();

/// Get the active SIMD backend name (e.g., "AVX2", "NEON").
const char* GetBackendName();

/// Get detected SIMD capability flags.
uint32_t GetSimdCapabilities();

/// Get SDK version string.
const char* GetVersion();

} // namespace hyperflux
