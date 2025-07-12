cmake_minimum_required(VERSION 3.8)
project(my_project LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")

set(CUTENSOR_LIB "/usr/lib/x86_64-linux-gnu/libcutensor.so")

set(TEST_DIR "${CMAKE_SOURCE_DIR}/test")
enable_testing()

# my_app
add_executable(my_app src/kernel.cu src/tensor.cpp src/device.cpp)
target_link_libraries(my_app ${CUDAToolkit_LIBRARIES} ${CUTENSOR_LIB})
target_include_directories(my_app PUBLIC "${CMAKE_SOURCE_DIR}/include"
                                          ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(my_app PROPERTIES CUDA_ARCHITECTURES "75")

# list_devs
add_executable(list_devs src/list_devs.cu)
target_link_libraries(list_devs ${CUDAToolkit_LIBRARIES})
target_include_directories(list_devs PUBLIC "${CMAKE_SOURCE_DIR}/include"
                                             ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(list_devs PROPERTIES CUDA_ARCHITECTURES "75")

# test_tensors
add_executable(test_tensors ${TEST_DIR}/test_tensors.cu src/tensor.cpp src/device.cpp)
target_link_libraries(test_tensors gtest gtest_main ${CUDAToolkit_LIBRARIES}
                       ${CUTENSOR_LIB})
target_include_directories(test_tensors PUBLIC "${CMAKE_SOURCE_DIR}/include"
                                                ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(test_tensors PROPERTIES CUDA_ARCHITECTURES "75")
add_test(NAME TestTensors COMMAND test_tensors)

# test_devices
add_executable(test_devices ${TEST_DIR}/test_devices.cu src/tensor.cpp src/device.cpp)
target_link_libraries(test_devices gtest gtest_main ${CUDAToolkit_LIBRARIES}
                       ${CUTENSOR_LIB})
target_include_directories(test_devices PUBLIC "${CMAKE_SOURCE_DIR}/include"
                                                ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(test_devices PROPERTIES CUDA_ARCHITECTURES "75")
add_test(NAME TestDevices COMMAND test_devices)

