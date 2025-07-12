
#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <cutensor.h>
#include <stdexcept>
#include <string>

enum DeviceType {
  CUDA,
  CPU,
};

// TODO: translate to a class

// Declaration of the Device struct
struct Device {

private:
  void create_handle();
  cutensorHandle_t handle;

public:
  // TODO: mark as const
  int id;
  DeviceType type;

  Device(int device_id, DeviceType device_type);
  bool operator==(const Device &other) const;
  bool operator!=(const Device &other) const;
  std::string to_string() const;
  cutensorHandle_t get_cutensor_handle();
  ~Device();
};

#endif // DEVICE_HPP
