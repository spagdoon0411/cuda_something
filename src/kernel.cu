#include "tensor.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int original_i = i;
  int count = 0;
  for (int stride = blockDim.x * gridDim.x; i < n; i += stride)
    y[i] = x[i] + y[i];

  printf("Thread %d processed elements from %d to %d\n", threadIdx.x,
         original_i, i - 1);
}

// The ID of the first CUDA device
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

int main(int argc, char *argv[]) {
  std::vector<size_t> shape = {3, 4};

  // Get some CUDA devices
  int gpuId = getGpuId();
  printf("Using GPU ID: %d\n", gpuId);

  float *data = (float *)malloc(sizeof(float) * 12);
  Device cpu = {0, DeviceType::CPU};
  Tensor tens(shape, data, cpu);

  Device gpu = {gpuId, DeviceType::CUDA};
  tens.moveToDevice(gpu);
  printf("Tensor moved to GPU %d\n", gpuId);
  tens.moveToDevice(cpu);
  printf("Tensor moved back to CPU\n");

  return 0;
}
