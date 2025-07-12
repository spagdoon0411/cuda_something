#include "device.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

// Helper function to check if CUDA devices are available
bool hasCUDADevices() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  return (err == cudaSuccess && deviceCount > 0);
}

// Test case for equality operator
TEST(DeviceTest, EqualityOperator) {
  Device device1(0, DeviceType::CUDA);
  Device device2(0, DeviceType::CUDA);
  Device device3(1, DeviceType::CPU);
  Device device4(0, DeviceType::CPU);

  // Devices with the same ID and type should be equal
  EXPECT_TRUE(device1 == device2)
      << "Devices with the same ID and type should be equal.";

  // Devices with different IDs should not be equal
  EXPECT_FALSE(device4 == device3)
      << "Devices with different IDs should not be equal.";

  // Devices with different types should not be equal
  EXPECT_FALSE(device1 == device4)
      << "Devices with different types should not be equal.";
}

// Test case for inequality operator
TEST(DeviceTest, InequalityOperator) {
  Device device1(0, DeviceType::CUDA);
  Device device2(0, DeviceType::CUDA);
  Device device3(1, DeviceType::CPU);
  Device device4(0, DeviceType::CPU);

  // Devices with the same ID and type should not be unequal
  EXPECT_FALSE(device1 != device2)
      << "Devices with the same ID and type should not be unequal.";

  // Devices with different IDs should be unequal
  EXPECT_TRUE(device4 != device3)
      << "Devices with different IDs should be unequal.";

  // Devices with different types should be unequal
  EXPECT_TRUE(device1 != device4)
      << "Devices with different types should be unequal.";
}

// Test case for Device constructor with CUDA device
TEST(DeviceTest, ConstructorWithCUDADevice) {
  if (!hasCUDADevices()) {
    GTEST_SKIP() << "No CUDA devices available for testing.";
  }

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  ASSERT_EQ(err, cudaSuccess) << "Failed to retrieve CUDA device count.";
  ASSERT_GT(deviceCount, 0) << "No CUDA devices available.";

  // Use the first available CUDA device for testing
  Device cudaDevice(0, DeviceType::CUDA);
  EXPECT_EQ(cudaDevice.id, 0) << "Device ID should be 0.";
  EXPECT_EQ(cudaDevice.type, DeviceType::CUDA) << "Device type should be CUDA.";
}

// Test case for cuTENSOR handle creation
TEST(DeviceTest, CUDADeviceHandleCreation) {
  if (!hasCUDADevices()) {
    GTEST_SKIP() << "No CUDA devices available for testing.";
  }

  Device cudaDevice(0, DeviceType::CUDA);

  // Ensure cuTENSOR handle is created successfully
  EXPECT_NO_THROW(cudaDevice.get_cutensor_handle())
      << "Failed to create cuTENSOR handle.";
}

// Test case for multiple CUDA devices
TEST(DeviceTest, MultipleCUDADevices) {
  if (!hasCUDADevices()) {
    GTEST_SKIP() << "No CUDA devices available for testing.";
  }

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  ASSERT_EQ(err, cudaSuccess) << "Failed to retrieve CUDA device count.";
  ASSERT_GT(deviceCount, 0) << "No CUDA devices available.";

  for (int i = 0; i < deviceCount; ++i) {
    Device cudaDevice(i, DeviceType::CUDA);
    EXPECT_EQ(cudaDevice.id, i) << "Device ID mismatch for device " << i << ".";
    EXPECT_EQ(cudaDevice.type, DeviceType::CUDA)
        << "Device type mismatch for device " << i << ".";

    // Ensure cuTENSOR handle is created successfully for each device
    EXPECT_NO_THROW(cudaDevice.get_cutensor_handle())
        << "Failed to create cuTENSOR handle for device " << i << ".";
  }
}
