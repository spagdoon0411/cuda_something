#include "tensor.hpp"
#include "device.hpp"
#include <cuda_runtime.h>
#include <iostream>
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

Tensor::Tensor(const std::vector<size_t> &shape, float *data,
               struct Device device)
    : shape(shape), data(data), device(device) {
  if (data == nullptr) {
    throw std::invalid_argument("Data pointer cannot be null.");
  }
  if (shape.empty()) {
    throw std::invalid_argument("Shape cannot be empty.");
  }
}

size_t getDataSize(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t dim : shape) {
    size *= dim;
  }
  return size * sizeof(float); // Assuming float data type
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
    cudaMalloc(&gpuData, dataSize);
    cudaMemcpy(gpuData, this->data, dataSize, cudaMemcpyHostToDevice);
    break;
  }

  case (DeviceType::CUDA << 8) | DeviceType::CPU: {
    std::cout << "Moving tensor from GPU to CPU." << std::endl;
    float *cpuData = (float *)malloc(dataSize);
    cudaMemcpy(cpuData, this->data, dataSize, cudaMemcpyDeviceToHost);
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
