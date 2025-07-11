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

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
      printf("Error: %s\n", cutensorGetErrorString(err));                      \
      exit(-1);                                                                \
    }                                                                          \
  };

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      printf("Error: %s\n", cudaGetErrorString(err));                          \
      exit(-1);                                                                \
    }                                                                          \
  };

// A sample from the cuTENSOR documentation.
// Exiting nonzero causes the test to fail.
TEST(TestCuTensor, Sample) {
  // Host element type definition
  typedef float floatTypeA;
  typedef float floatTypeB;
  typedef float floatTypeC;
  typedef float floatTypeCompute;

  // CUDA types
  cutensorDataType_t typeA = CUTENSOR_R_32F;
  cutensorDataType_t typeB = CUTENSOR_R_32F;
  cutensorDataType_t typeC = CUTENSOR_R_32F;
  cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

  printf("Include headers and define data types\n");

  /* ***************************** */

  // Create vector of modes
  std::vector<int> modeC{'m', 'u', 'n', 'v'};
  std::vector<int> modeA{'m', 'h', 'k', 'n'};
  std::vector<int> modeB{'u', 'k', 'v', 'h'};
  int nmodeA = modeA.size();
  int nmodeB = modeB.size();
  int nmodeC = modeC.size();

  // Extents
  std::unordered_map<int, int64_t> extent;
  extent['m'] = 96;
  extent['n'] = 96;
  extent['u'] = 96;
  extent['v'] = 64;
  extent['h'] = 64;
  extent['k'] = 64;

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentC;
  for (auto mode : modeC)
    extentC.push_back(extent[mode]);
  std::vector<int64_t> extentA;
  for (auto mode : modeA)
    extentA.push_back(extent[mode]);
  std::vector<int64_t> extentB;
  for (auto mode : modeB)
    extentB.push_back(extent[mode]);

  printf("Define modes and extents\n");

  /* ***************************** */

  // Number of elements of each tensor
  size_t elementsA = 1;
  for (auto mode : modeA)
    elementsA *= extent[mode];
  size_t elementsB = 1;
  for (auto mode : modeB)
    elementsB *= extent[mode];
  size_t elementsC = 1;
  for (auto mode : modeC)
    elementsC *= extent[mode];

  // Size in bytes
  size_t sizeA = sizeof(floatTypeA) * elementsA;
  size_t sizeB = sizeof(floatTypeB) * elementsB;
  size_t sizeC = sizeof(floatTypeC) * elementsC;

  // Allocate on device
  void *A_d, *B_d, *C_d;
  cudaMalloc((void **)&A_d, sizeA);
  cudaMalloc((void **)&B_d, sizeB);
  cudaMalloc((void **)&C_d, sizeC);

  // Allocate on host
  floatTypeA *A = (floatTypeA *)malloc(sizeof(floatTypeA) * elementsA);
  floatTypeB *B = (floatTypeB *)malloc(sizeof(floatTypeB) * elementsB);
  floatTypeC *C = (floatTypeC *)malloc(sizeof(floatTypeC) * elementsC);

  // Initialize data on host
  for (int64_t i = 0; i < elementsA; i++)
    A[i] = (((float)rand()) / RAND_MAX - 0.5) * 100;
  for (int64_t i = 0; i < elementsB; i++)
    B[i] = (((float)rand()) / RAND_MAX - 0.5) * 100;
  for (int64_t i = 0; i < elementsC; i++)
    C[i] = (((float)rand()) / RAND_MAX - 0.5) * 100;

  // Copy to device
  HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

  const uint32_t kAlignment =
      128; // Alignment of the global-memory device pointers (bytes)
  assert(uintptr_t(A_d) % kAlignment == 0);
  assert(uintptr_t(B_d) % kAlignment == 0);
  assert(uintptr_t(C_d) % kAlignment == 0);

  printf("Allocate, initialize and transfer tensors\n");

  /*************************
   * cuTENSOR
   *************************/

  cutensorHandle_t handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  /**********************
   * Create Tensor Descriptors
   **********************/

  cutensorTensorDescriptor_t descA;
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descA, nmodeA,
                                              extentA.data(), NULL, /*stride*/
                                              typeA, kAlignment));

  cutensorTensorDescriptor_t descB;
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descB, nmodeB,
                                              extentB.data(), NULL, /*stride*/
                                              typeB, kAlignment));

  cutensorTensorDescriptor_t descC;
  HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descC, nmodeC,
                                              extentC.data(), NULL, /*stride*/
                                              typeC, kAlignment));

  printf("Initialize cuTENSOR and tensor descriptors\n");

  /*******************************
   * Create Contraction Descriptor
   *******************************/

  cutensorOperationDescriptor_t desc;
  HANDLE_ERROR(cutensorCreateContraction(
      handle, &desc, descA, modeA.data(),
      /* unary operator A*/ CUTENSOR_OP_IDENTITY, descB, modeB.data(),
      /* unary operator B*/ CUTENSOR_OP_IDENTITY, descC, modeC.data(),
      /* unary operator C*/ CUTENSOR_OP_IDENTITY, descC, modeC.data(),
      descCompute));

  /*****************************
   * Optional (but recommended): ensure that the scalar type is correct.
   *****************************/

  cutensorDataType_t scalarType;
  HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(
      handle, desc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
      (void *)&scalarType, sizeof(scalarType)));

  assert(scalarType == CUTENSOR_R_32F);
  typedef float floatTypeCompute;
  floatTypeCompute alpha = (floatTypeCompute)1.1f;
  floatTypeCompute beta = (floatTypeCompute)0.f;

  /**************************
   * Set the algorithm to use
   ***************************/

  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPref, algo,
                                            CUTENSOR_JIT_MODE_NONE));

  /**********************
   * Query workspace estimate
   **********************/

  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  HANDLE_ERROR(cutensorEstimateWorkspaceSize(
      handle, desc, planPref, workspacePref, &workspaceSizeEstimate));

  /**************************
   * Create Contraction Plan
   **************************/

  cutensorPlan_t plan;
  HANDLE_ERROR(
      cutensorCreatePlan(handle, &plan, desc, planPref, workspaceSizeEstimate));

  /**************************
   * Optional: Query information about the created plan
   **************************/

  // query actually used workspace
  uint64_t actualWorkspaceSize = 0;
  HANDLE_ERROR(cutensorPlanGetAttribute(
      handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));

  // At this point the user knows exactly how much memory is need by the
  // operation and only the smaller actual workspace needs to be allocated
  assert(actualWorkspaceSize <= workspaceSizeEstimate);

  void *work = nullptr;
  if (actualWorkspaceSize > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
    assert(uintptr_t(work) % 128 ==
           0); // workspace must be aligned to 128 byte-boundary
  }

  /**********************
   * Execute
   **********************/

  cudaStream_t stream;
  HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

  HANDLE_ERROR(cutensorContract(handle, plan, (void *)&alpha, A_d, B_d,
                                (void *)&beta, C_d, C_d, work,
                                actualWorkspaceSize, stream));

  /**********************
   * Free allocated data
   **********************/
  HANDLE_ERROR(cutensorDestroy(handle));
  HANDLE_ERROR(cutensorDestroyPlan(plan));
  HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
  HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
  HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

  if (A)
    free(A);
  if (B)
    free(B);
  if (C)
    free(C);
  if (A_d)
    cudaFree(A_d);
  if (B_d)
    cudaFree(B_d);
  if (C_d)
    cudaFree(C_d);
  if (work)
    cudaFree(work);
}
