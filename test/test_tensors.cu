#include <assert.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <tensor.hpp>
#include <unordered_map>
#include <vector>

TEST(SimpleCopyTest, Copy) {
  float *hostPtr;
  float *devicePtr;

  size_t size = 1024 * sizeof(float);

  hostPtr = (float *)malloc(size);
  ASSERT_NE(hostPtr, nullptr) << "Failed to allocate host memory";

  cudaError_t err = cudaMalloc((void **)&devicePtr, size);
  ASSERT_EQ(err, cudaSuccess)
      << "Failed to allocate device memory: " << cudaGetErrorString(err);

  cudaError_t copyErr =
      cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
  ASSERT_EQ(copyErr, cudaSuccess)
      << "Failed to copy memory from host to device: "
      << cudaGetErrorString(copyErr);
}

int getGpuId() {
  int deviceCount;
  cudaError_t res;

  if ((res = cudaGetDeviceCount(&deviceCount)) != cudaSuccess) {
    std::cerr << "Error getting device count: " << cudaGetErrorString(res)
              << std::endl;
    return -1;
  }

  if (deviceCount == 0) {
    std::cerr << "No CUDA devices found." << std::endl;
    return -1;
  }

  int gpuId = 0; // Default to the first GPU
  cudaSetDevice(gpuId);
  return gpuId;
}

// Tests that we can allocate a tensor and move it between the CPU and the GPU.
TEST(TestTensorAlloc, Size) {
  std::vector<size_t> shape = {3, 4};

  // Get some CUDA devices
  int gpuId = getGpuId();
  ASSERT_GE(gpuId, 0) << "Failed to get a valid GPU ID";

  float *data = (float *)malloc(sizeof(float) * 12);
  ASSERT_NE(data, nullptr) << "Failed to allocate host memory for data";

  Device cpu = {0, DeviceType::CPU};
  Tensor tens(shape, cpu);

  Device gpu = {gpuId, DeviceType::CUDA};
  tens.moveToDevice(gpu);
  printf("Tensor moved to GPU %d\n", gpuId);

  tens.moveToDevice(cpu);
  printf("Tensor moved back to CPU\n");

  free(data);
}
