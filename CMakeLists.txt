cmake_minimum_required(VERSION 3.8)
project(my_project LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(sources src/kernel.cu)

add_executable(my_app ${sources})

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")

set_target_properties(my_app PROPERTIES CUDA_ARCHITECTURES "75")

target_include_directories(
  my_app PUBLIC "${CMAKE_SOURCE_DIR}/include"
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(
  my_app PUBLIC 
)

cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0074 NEW)

target_link_libraries(my_app ${CUDAToolkit_LIBRARIES})


