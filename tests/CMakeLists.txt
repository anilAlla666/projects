cmake_minimum_required(VERSION 3.18)

# ============================================================
# CIPHER Test Suite — Phase 0: Foundation
# ============================================================

# F1: Intercept overhead test (CPU-only, no GPU needed)
add_executable(test_f1_intercept test_f1_intercept.cpp)
target_include_directories(test_f1_intercept PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_f1_intercept cipher Threads::Threads dl)
target_compile_options(test_f1_intercept PRIVATE -O3)

# F2: Green Context allocation test
add_executable(test_f2_green_ctx test_f2_green_ctx.cu)
target_include_directories(test_f2_green_ctx PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_f2_green_ctx cipher CUDA::cudart CUDA::cuda_driver)
set_target_properties(test_f2_green_ctx PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# F4: Liquid state unit test
add_executable(test_f4_liquid test_f4_liquid.cu)
target_include_directories(test_f4_liquid PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_f4_liquid cipher CUDA::cudart)
set_target_properties(test_f4_liquid PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# Phase 0 integration test
add_executable(test_phase0 test_phase0.cu)
target_include_directories(test_phase0 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_phase0 cipher CUDA::cudart CUDA::cuda_driver dl m)
set_target_properties(test_phase0 PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# Layer 3 tests (CPU-only, no GPU needed)
add_executable(test_layer3 test_layer3.cpp)
target_include_directories(test_layer3 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_layer3 cipher Threads::Threads m)
target_compile_options(test_layer3 PRIVATE -O3)

# CTest registration
enable_testing()
add_test(NAME F1_Intercept   COMMAND test_f1_intercept)
add_test(NAME F2_GreenCtx    COMMAND test_f2_green_ctx)
add_test(NAME F4_LiquidState COMMAND test_f4_liquid)
add_test(NAME Phase0_Full    COMMAND test_phase0)
add_test(NAME Layer3_Full    COMMAND test_layer3)
