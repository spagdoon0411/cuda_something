# =============================================================================
# CMake Configuration
# =============================================================================
cmake_minimum_required(VERSION 3.18)
project(my_project 
    VERSION 1.0.0
    DESCRIPTION "CUDA Tensor Operations"
    LANGUAGES CXX CUDA
)

# =============================================================================
# Global Settings
# =============================================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Fix for CUDA20 dialect issue
if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
    set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
endif()

# =============================================================================
# Dependencies
# =============================================================================
# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")

# cuTENSOR Library - use find_library to locate automatically
find_library(CUTENSOR_LIB cutensor)
if(NOT CUTENSOR_LIB)
    message(FATAL_ERROR "Could not find libcutensor.so. Please install cuTENSOR.")
endif()
message(STATUS "Found cuTENSOR library: ${CUTENSOR_LIB}")

# Google Test (for testing)
find_package(GTest REQUIRED)

# =============================================================================
# Project Configuration
# =============================================================================
# Directories
set(PROJECT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(PROJECT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(PROJECT_TEST_DIR "${CMAKE_SOURCE_DIR}/test")

# CUDA Architecture
set(CUDA_ARCHITECTURE "75")

# Enable testing
enable_testing()

# =============================================================================
# Common Target Properties
# =============================================================================
# Function to set common properties for CUDA targets
function(set_cuda_target_properties TARGET_NAME)
    target_include_directories(${TARGET_NAME} 
        PUBLIC 
            ${PROJECT_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    set_target_properties(${TARGET_NAME} 
        PROPERTIES 
            CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE}
    )
endfunction()

# =============================================================================
# Source Files
# =============================================================================
set(COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/tensor.cpp
    ${PROJECT_SOURCE_DIR}/device.cpp
)

# =============================================================================
# Executables
# =============================================================================

# main - Main application
add_executable(main 
    ${PROJECT_SOURCE_DIR}/kernel.cu
    ${COMMON_SOURCES}
)
target_link_libraries(main 
    ${CUDAToolkit_LIBRARIES} 
    ${CUTENSOR_LIB}
)
set_cuda_target_properties(main)

# list_devs - Device listing utility
add_executable(list_devs 
    ${PROJECT_SOURCE_DIR}/list_devs.cu
)
target_link_libraries(list_devs 
    ${CUDAToolkit_LIBRARIES}
)
set_cuda_target_properties(list_devs)

# =============================================================================
# Tests
# =============================================================================

# test_tensors
add_executable(test_tensors 
    ${PROJECT_TEST_DIR}/test_tensors.cu
    ${COMMON_SOURCES}
)
target_link_libraries(test_tensors 
    GTest::gtest 
    GTest::gtest_main 
    ${CUDAToolkit_LIBRARIES}
    ${CUTENSOR_LIB}
)
set_cuda_target_properties(test_tensors)
add_test(NAME TestTensors COMMAND test_tensors)

# test_devices
add_executable(test_devices 
    ${PROJECT_TEST_DIR}/test_devices.cu
    ${COMMON_SOURCES}
)
target_link_libraries(test_devices 
    GTest::gtest 
    GTest::gtest_main 
    ${CUDAToolkit_LIBRARIES}
    ${CUTENSOR_LIB}
)
set_cuda_target_properties(test_devices)
add_test(NAME TestDevices COMMAND test_devices)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architecture: ${CUDA_ARCHITECTURE}")

