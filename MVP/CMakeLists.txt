cmake_minimum_required(VERSION 3.10)
project(HyperFlux)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Point to the downloaded ONNX Runtime folder
# CHANGE THIS if your folder name is different
set(ONNX_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")

# Include Headers
include_directories(${ONNX_ROOT}/include)

# Find the library file (onnxruntime.dll or libonnxruntime.so)
# This assumes you extracted it correctly
link_directories(${ONNX_ROOT}/lib)

# Define the executable
add_executable(HyperFlux main.cpp)

# Link the library
if(WIN32)
    target_link_libraries(HyperFlux onnxruntime)
    # Copy the .dll to the output folder so it runs
    add_custom_command(TARGET HyperFlux POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:HyperFlux>)
else()
    target_link_libraries(HyperFlux onnxruntime)
endif()