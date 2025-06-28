#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaError_t res = cudaGetDeviceProperties(&prop, i);
    if (res != cudaSuccess) {
      printf("Error getting properties for device %d\n", i);
      continue;
    }
    printf("Device %d: %s\n", i, prop.name);
  }
  return 0;
}
