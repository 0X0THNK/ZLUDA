#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace cuda_emulator {

struct VirtualDeviceProperties {
    char name[256] = "ZLUDA Virtual CUDA GPU";
    size_t totalGlobalMem = 0;
    int multiProcessorCount = 0;
    int warpSize = 32;
    int regsPerBlock = 65536;
    int sharedMemPerBlock = 99 * 1024;
    int major = 8;
    int minor = 0;
};

class DevicePropertiesEmulation {
public:
    static DevicePropertiesEmulation& instance();
    VirtualDeviceProperties get_properties() const;
    int get_attribute(int attr, int* value) const;

private:
    DevicePropertiesEmulation();
    VirtualDeviceProperties props_;
};

} // namespace cuda_emulator
