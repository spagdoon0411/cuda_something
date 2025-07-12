#ifndef CUDA_ERROR_UTILS_HPP
#define CUDA_ERROR_UTILS_HPP

#define HANDLE_CUTENSOR_ERROR(x)                                               \
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
      printf("CUDA Error: %s\n", cudaGetErrorString(err));                     \
      exit(-1);                                                                \
    }                                                                          \
  };

#endif // CUDA_ERROR_UTILS_HPP
