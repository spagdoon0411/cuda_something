#include "tensor.hpp"
#include "device.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <errno.h>
#include <iostream>
#include <string.h>
#include <vector>

struct Device Tensor::getDevice() const {
  return device;
}

float *Tensor::getData() const {
  if (data == nullptr) {
    throw std::runtime_error("Tensor data pointer is not set.");
  }

  return data;
}

size_t getDataSize(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t dim : shape) {
    size *= dim;
  }

  return size * sizeof(float); // Assuming float data type
}

/// Allocates a new tensor with the specified shape, device
Tensor::Tensor(const std::vector<size_t> &shape, struct Device device)
    : shape(shape), device(device) {

  if (shape.empty()) {
    throw std::invalid_argument("Shape cannot be empty.");
  }

  switch (device.type) {
  case DeviceType::CPU: {
    float *cpuData = (float *)std::malloc(getDataSize(shape));

    if (cpuData == nullptr) {
      std::string errorMsg = strerror(errno);
      throw std::runtime_error("Failed to allocate memory on CPU: " + errorMsg);
    }

    this->data = cpuData;
    break;
  }

  case DeviceType::CUDA: {
    float *gpuData = nullptr;
    size_t dataSize = getDataSize(shape);

    cudaError_t setDeviceRes = cudaSetDevice(device.id);
    if (setDeviceRes != cudaSuccess) {
      throw std::runtime_error("Failed to set CUDA device: " +
                               std::string(cudaGetErrorString(setDeviceRes)));
    }

    cudaError_t cudaMallocRes = cudaMalloc(&gpuData, dataSize);
    if (cudaMallocRes != cudaSuccess) {
      throw std::runtime_error("Failed to allocate memory on GPU: " +
                               std::string(cudaGetErrorString(cudaMallocRes)));
    }

    this->data = gpuData;
    break;
  }
  }
}

/// Moves the tensor to a different device, assuming one GPU device
/// and one CPU device.
void Tensor::moveToDevice(struct Device device) {
  if (this->device.type == device.type && this->device.id == device.id) {
    return;
  }

  if (this->data == nullptr) {
    throw std::runtime_error("Tensor data pointer is not set.");
  }

  // device1 | device2 : device1 -> device2
  int pair = (this->device.type << 8) | device.type;
  size_t dataSize = getDataSize(shape);

  switch (pair) {
  case (DeviceType::CPU << 8) | DeviceType::CUDA: {
    size_t dataSize = getDataSize(shape);
    float *gpuData = nullptr;

    cudaError_t setDeviceRes = cudaSetDevice(device.id);
    if (setDeviceRes != cudaSuccess) {
      throw std::runtime_error("Failed to set CUDA device: " +
                               std::string(cudaGetErrorString(setDeviceRes)));
    }

    cudaError_t mallocRes = cudaMalloc(&gpuData, dataSize);
    if (mallocRes != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(mallocRes)
                << std::endl;
      break;
    }

    cudaError_t memcpyStatus =
        cudaMemcpy(gpuData, this->data, dataSize, cudaMemcpyHostToDevice);
    if (memcpyStatus != cudaSuccess) {
      std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(memcpyStatus)
                << std::endl;
      cudaFree(gpuData); // Free allocated memory on failure
      break;
    }

    std::free(this->data);
    this->data = gpuData;

    break;
  }

  case (DeviceType::CUDA << 8) | DeviceType::CPU: {
    std::cout << "Moving tensor from GPU to CPU." << std::endl;
    float *cpuData = (float *)std::malloc(dataSize);
    if (cpuData == nullptr) {
      std::string errorMsg = strerror(errno);
      throw std::runtime_error("Failed to allocate memory on CPU: " + errorMsg);
    }

    cudaError_t res =
        cudaMemcpy(cpuData, this->data, dataSize, cudaMemcpyDeviceToHost);
    if (res != cudaSuccess) {
      throw std::runtime_error("Failed to copy data from GPU to CPU: " +
                               std::string(cudaGetErrorString(res)));
    }

    cudaFree(this->data);
    this->data = cpuData;

    break;
  }

  case (DeviceType::CUDA << 8) | DeviceType::CUDA: {
    std::cout << "Moving tensor from GPU " << this->device.id << " to GPU "
              << device.id << " via CPU is not implemented." << std::endl;
    throw std::runtime_error("GPU-to-GPU transfers are not implemented.");
    break;
  }

  default:
    throw std::invalid_argument("Unsupported device transfer.");
  }

  this->device = device;
}

const std::vector<size_t> &Tensor::getShape() const { return shape; }
