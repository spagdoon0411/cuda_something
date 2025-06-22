#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device %d: %s\n", i, prop.name);
  }
  return 0;
}
