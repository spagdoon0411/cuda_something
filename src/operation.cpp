#include "operation.hpp"
#include "tensor.hpp"
#include <functional>
#include <stdexcept>
#include <vector>

Tensor Operation::forward(const std::vector<Tensor> &inputs) {
  if (inputs.empty()) {
    throw std::invalid_argument("Inputs cannot be empty.");
  }

  this->inputs = inputs;
  Tensor output = this->compute_output(inputs);
  this->grad(inputs);

  return output;
}

void Operation::grad(const std::vector<Tensor> &gradInputs) {
  this->localGrads.clear();

  for (std::function<Tensor(const std::vector<Tensor> &)> &func :
       partialGradsFunctions) {
    this->localGrads.push_back(func(inputs));
  }
}
