#include "tensor.hpp"
#include <functional>
#include <vector>

class Operation {
protected:
  Tensor compute_output(const std::vector<Tensor> &gradInputs);
  void grad(const std::vector<Tensor> &gradInputs);

public:
  std::vector<Tensor> inputs;
  std::vector<Tensor> localGrads;

  // A vector containing functions of all input tensors.
  // Input tensors are passed by reference.
  std::vector<std::function<Tensor(const std::vector<Tensor> &)>>
      partialGradsFunctions;

  // Computes the output of the operation and computes local grads.
  virtual Tensor forward(const std::vector<Tensor> &inputs);
};
