
#ifndef DEVICE_HPP
#define DEVICE_HPP

enum DeviceType {
  CUDA,
  CPU,
};

struct Device {
  int id;
  DeviceType type;
};

#endif // DEVICE_HPP
