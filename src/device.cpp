#include "device.hpp"
#include <cutensor.h>
#include <cutensor/types.h>
#include <iostream>
#include <sstream>

// Constructor implementation
Device::Device(int device_id, DeviceType device_type)
    : id(device_id), type(device_type) {
  if (type == DeviceType::CUDA) {
    create_handle();
  }
}

// Private helper function to create cuTENSOR handle
void Device::create_handle() {
  // Save the current CUDA device
  int previous_device;
  cudaError_t cuda_status = cudaGetDevice(&previous_device);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error("Failed to get current CUDA device: " +
                             std::string(cudaGetErrorString(cuda_status)));
  }

  // Set the CUDA device to the specified device ID
  cuda_status = cudaSetDevice(id);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device: " +
                             std::string(cudaGetErrorString(cuda_status)));
  }

  // Initialize cuTENSOR handle for the specified CUDA device
  cutensorHandle_t handle;
  cutensorStatus_t status = cutensorCreate(&handle);
  if (status != CUTENSOR_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuTENSOR handle: " +
                             std::to_string(status));
  }

  // Restore the previous CUDA device
  cuda_status = cudaSetDevice(previous_device);
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error("Failed to restore previous CUDA device: " +
                             std::string(cudaGetErrorString(cuda_status)));
  }

  this->handle = handle;
};

cutensorHandle_t Device::get_cutensor_handle() {
  if (type != DeviceType::CUDA) {
    throw std::runtime_error(
        "cuTENSOR handle is only available for CUDA devices.");
  }

  if (handle == nullptr) {
    create_handle();
  }

  return handle;
}

// Implementation of the equality operator
bool Device::operator==(const Device &other) const {
  return id == other.id && type == other.type;
}

// Implementation of the inequality operator
bool Device::operator!=(const Device &other) const { return !(*this == other); }

// Implementation of the to_string method
std::string Device::to_string() const {
  std::ostringstream oss;
  oss << "Device(id: " << id
      << ", type: " << (type == DeviceType::CUDA ? "CUDA" : "CPU") << ")";
  return oss.str();
}

Device::~Device() {
  if (type == DeviceType::CUDA && handle != nullptr) {
    cutensorStatus_t status = cutensorDestroy(handle);
    if (status != CUTENSOR_STATUS_SUCCESS) {
      std::cerr << "Failed to destroy cuTENSOR handle: "
                << std::to_string(status) << std::endl;
    }
    handle = nullptr; // Ensure the handle is reset
  }
}

// Hold cuTENSOR handles in devices
