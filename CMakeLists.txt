# ==========================================================================
# HyperFlux SDK v1.0.0
# Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
# ==========================================================================
cmake_minimum_required(VERSION 3.16)
project(HyperFlux VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(HF_BUILD_SHARED  "Build shared library"       OFF)
option(HF_DEBUG          "Enable debug logging"        OFF)
option(HF_PROFILE        "Enable profiling"            OFF)
option(HF_ENABLE_AVX512  "Enable AVX-512 codegen"     OFF)
option(HF_BUILD_TESTS    "Build test executable"       OFF)

# Sources
set(HF_SOURCES
    src/engine/hf_hyperflux.cpp
    src/engine/hf_inference_engine.cpp
    src/engine/hf_memory_pool.cpp
    src/engine/hf_simd_dispatch.cpp
    src/engine/hf_weight_loader.cpp
    src/kernels/hf_ballistics.cpp
    src/kernels/hf_hitbox.cpp
    src/kernels/hf_visibility.cpp
    src/kernels/hf_spawn_selection.cpp
    src/kernels/hf_mount_dynamics.cpp
    src/kernels/hf_destruction.cpp
    src/kernels/hf_lumen_gi.cpp
    src/kernels/hf_audio_occlusion.cpp
    src/kernels/hf_occlusion_culling.cpp
    src/kernels/hf_lod_selection.cpp
)

# Library target
if(HF_BUILD_SHARED)
    add_library(hyperflux SHARED ${HF_SOURCES})
    target_compile_definitions(hyperflux PRIVATE HF_EXPORT)
else()
    add_library(hyperflux STATIC ${HF_SOURCES})
endif()

target_include_directories(hyperflux
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Debug/profile flags
if(HF_DEBUG)
    target_compile_definitions(hyperflux PUBLIC HF_DEBUG)
endif()
if(HF_PROFILE)
    target_compile_definitions(hyperflux PUBLIC HF_PROFILE)
endif()

# Platform-specific flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    target_compile_options(hyperflux PRIVATE -mavx2 -mfma)
    if(HF_ENABLE_AVX512)
        target_compile_options(hyperflux PRIVATE -mavx512f -mavx512bw -mavx512vnni)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64)")
    # NEON is default on AArch64
endif()

# Optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
target_compile_options(hyperflux PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG -ffast-math>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Install
install(TARGETS hyperflux
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/hyperflux DESTINATION include)

# Alias for FetchContent users
add_library(HyperFlux::hyperflux ALIAS hyperflux)
