#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "device.hpp"
#include <cstddef>
#include <vector>

class Tensor {
private:
  const std::vector<size_t> shape;
  float *data = nullptr;
  struct Device device;

public:
  struct Device getDevice() const;
  float *getData() const;
  Tensor(const std::vector<size_t> &shape, struct Device device);
  void moveToDevice(struct Device device);
  const std::vector<size_t> &getShape() const;
  void toDevice(struct Device device);
  size_t getSize() const;
  ~Tensor();
};

#endif // TENSOR_HPP
