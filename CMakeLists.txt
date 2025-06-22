cmake_minimum_required(VERSION 3.8)
project(my_project LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")

add_executable(my_app src/kernel.cu)
add_executable(list_devs src/list_devs.cu)

cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0074 NEW)

target_link_libraries(list_devs ${CUDAToolkit_LIBRARIES})
target_link_libraries(my_app ${CUDAToolkit_LIBRARIES})

target_include_directories(
  my_app PUBLIC "${CMAKE_SOURCE_DIR}/include"
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_include_directories(
  list_devs PUBLIC "${CMAKE_SOURCE_DIR}/include"
  ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(my_app PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(list_devs PROPERTIES CUDA_ARCHITECTURES "75")


