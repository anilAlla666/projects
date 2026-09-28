cmake_minimum_required(VERSION 3.18)
project(cipher LANGUAGES CXX CUDA C)

# ============================================================
# CIPHER Build System
# Neural Dynamics | Phase 0: Foundation
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture — H100 (sm_90), A100 (sm_80), A10/RTX (sm_86)
# Build for all supported architectures
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# Find packages
# ============================================================

find_package(CUDAToolkit REQUIRED)

# CUPTI — part of CUDA toolkit
find_path(CUPTI_INCLUDE_DIR cupti.h
    PATHS ${CUDAToolkit_INCLUDE_DIRS}
          ${CUDAToolkit_ROOT}/extras/CUPTI/include
    REQUIRED)
find_library(CUPTI_LIBRARY cupti
    PATHS ${CUDAToolkit_LIBRARY_DIR}
          ${CUDAToolkit_ROOT}/extras/CUPTI/lib64
    REQUIRED)

# Threads
find_package(Threads REQUIRED)

# ============================================================
# Compiler flags
# ============================================================

# Release: aggressive optimization, no debug overhead
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native -flto -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# Debug: keep symbols, sanitizers
set(CMAKE_CXX_FLAGS_DEBUG    "-O0 -g -fsanitize=address,undefined")
set(CMAKE_CUDA_FLAGS_DEBUG   "-O0 -G -g")

# Common flags
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fvisibility=default -fPIC -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --relocatable-device-code=true")

# ============================================================
# CIPHER shared library — LD_PRELOAD target
# ============================================================

set(CIPHER_SOURCES
    src/cipher_intercept.cpp
    src/cipher_green_ctx.cu
    src/cipher_l2_persist.cu
    src/cipher_liquid_state.cu
    src/cipher_telemetry.cpp
    src/cipher_runtime.cpp
    # Layer 3 — Week 2-4
    src/cipher_structural_lookup.cpp
    src/cipher_oracle.cpp
    src/cipher_recipes.cpp
    src/cipher_dispatch.cpp
)

add_library(cipher SHARED ${CIPHER_SOURCES})

target_include_directories(cipher PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUPTI_INCLUDE_DIR})

target_link_libraries(cipher
    CUDA::cudart
    CUDA::cuda_driver
    ${CUPTI_LIBRARY}
    Threads::Threads
    dl
    m)

# Export all symbols — required for LD_PRELOAD override of CUDA symbols
set_target_properties(cipher PROPERTIES
    OUTPUT_NAME "cipher"
    SOVERSION   1
    VERSION     0.1.0
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    # Don't hide symbols — we need cuGetProcAddress to be found by the linker
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF)

# Link options: --no-as-needed ensures cuGetProcAddress override is seen
target_link_options(cipher PRIVATE -Wl,--no-as-needed)

# ============================================================
# Tests
# ============================================================

add_subdirectory(tests)

# ============================================================
# Install
# ============================================================

install(TARGETS cipher
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY include/
    DESTINATION include/cipher
    FILES_MATCHING PATTERN "*.h")

# ============================================================
# Status report
# ============================================================

message(STATUS "")
message(STATUS "═══════════════════════════════════════════")
message(STATUS "  CIPHER Build Configuration")
message(STATUS "═══════════════════════════════════════════")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA toolkit:  ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA archs:    ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUPTI:         ${CUPTI_INCLUDE_DIR}")
message(STATUS "  Install to:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "═══════════════════════════════════════════")
message(STATUS "")
message(STATUS "  Build:   cmake --build . --parallel")
message(STATUS "  Test:    CIPHER=1 LD_PRELOAD=./libcipher.so python train.py")
message(STATUS "")
