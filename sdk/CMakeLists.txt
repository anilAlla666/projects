cmake_minimum_required(VERSION 3.10)
project(soma_sdk VERSION 7.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# ============================================================
# Core library — zero dependencies, pure C
# ============================================================
add_library(soma INTERFACE)
target_include_directories(soma INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================
# Standalone example
# ============================================================
add_executable(soma_example examples/example_standalone.c)
target_include_directories(soma_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(soma_example m)

# Optimization flags
target_compile_options(soma_example PRIVATE -O2 -march=native)

# ============================================================
# Shared library build (for integration as .so)
# ============================================================
add_library(soma_shared SHARED examples/example_standalone.c)
set_target_properties(soma_shared PROPERTIES
    OUTPUT_NAME "soma"
    VERSION 7.0
    SOVERSION 7
)
target_include_directories(soma_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(soma_shared m)
target_compile_definitions(soma_shared PRIVATE SOMA_BUILD_SHARED)
target_compile_options(soma_shared PRIVATE -O2)

# NOTE: For .so build, create a soma.c that just does:
#   #define SOMA_IMPLEMENTATION
#   #include "soma.h"
# and compile that instead of example_standalone.c

# ============================================================
# Installation
# ============================================================
install(FILES include/soma.h DESTINATION include)
install(TARGETS soma_example DESTINATION bin)

# ============================================================
# Optional: ros2_control integration
# ============================================================
# Uncomment if building within a ROS2 workspace:
#
# find_package(ament_cmake REQUIRED)
# find_package(controller_interface REQUIRED)
# find_package(hardware_interface REQUIRED)
# find_package(pluginlib REQUIRED)
#
# add_library(soma_controller SHARED ros2_control/soma_ros2_controller.cpp)
# target_include_directories(soma_controller PRIVATE include)
# target_link_libraries(soma_controller m)
# ament_target_dependencies(soma_controller controller_interface hardware_interface pluginlib)
# pluginlib_export_plugin_description_file(controller_interface ros2_control/plugin.xml)
# install(TARGETS soma_controller DESTINATION lib)

# ============================================================
# Optional: Unitree SDK2 integration
# ============================================================
# Uncomment if unitree_sdk2 is installed:
#
# find_library(UNITREE_SDK unitree_sdk2)
# if(UNITREE_SDK)
#   add_executable(soma_unitree unitree/soma_unitree.cpp)
#   target_include_directories(soma_unitree PRIVATE include)
#   target_link_libraries(soma_unitree m ${UNITREE_SDK} cyclonedds)
# endif()

message(STATUS "SOMA SDK v${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Core library: header-only (soma.h)")
message(STATUS "  Dependencies: libm only")
