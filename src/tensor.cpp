#include "tensor.hpp"
#include "device.hpp"
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
    // Unreachable if initializer sets data pointer
    throw std::runtime_error("Data pointer is null.");
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

Tensor::Tensor(const std::vector<size_t> &shape, struct Device device)
    : shape(shape), device(device) {

  if (shape.empty()) {
    throw std::invalid_argument("Shape cannot be empty.");
  }

  switch (device.type) {
  case DeviceType::CPU: {
    float *cpuData = (float *)malloc(getDataSize(shape));

    this->data = cpuData;
    break;
  }

  case DeviceType::CUDA: {
    float *gpuData = nullptr;
    size_t dataSize = getDataSize(shape);

    cudaError_t err = cudaMalloc(&gpuData, dataSize);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate memory on GPU: " +
                               std::string(cudaGetErrorString(err)));
    }

    this->data = gpuData;
    break;
  }
  }
}

void Tensor::moveToDevice(struct Device device) {
  if (this->device.type == device.type && this->device.id == device.id) {
    std::cout << "Tensor is already on the target device." << std::endl;
    return;
  }

  // (d1, d2): d1 -> d2
  int pair = (this->device.type << 8) | device.type;
  size_t dataSize = getDataSize(shape);

  switch (pair) {
  case (DeviceType::CPU << 8) | DeviceType::CUDA: {
    size_t dataSize = getDataSize(shape);
    float *gpuData = nullptr;

    cudaError_t mallocStatus = cudaMalloc(&gpuData, dataSize);
    if (mallocStatus != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(mallocStatus)
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

    break;
  }

  case (DeviceType::CUDA << 8) | DeviceType::CPU: {
    std::cout << "Moving tensor from GPU to CPU." << std::endl;
    float *cpuData = (float *)malloc(dataSize);
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

  // Update the internal state
  this->device = device;
}

const std::vector<size_t> &Tensor::getShape() const { return shape; }
