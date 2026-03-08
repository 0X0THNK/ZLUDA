#include "device_props.h"
#include "virtual_vram.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <cstdint>

namespace cuda_emulator {

DevicePropertiesEmulation& DevicePropertiesEmulation::instance() {
    static DevicePropertiesEmulation props;
    return props;
}

DevicePropertiesEmulation::DevicePropertiesEmulation() {
    std::strncpy(props_.name, "ZLUDA Virtual CUDA GPU", sizeof(props_.name));
    props_.totalGlobalMem = VirtualVramManager::instance().available_virtual_memory();
    props_.multiProcessorCount = std::max(4u, std::thread::hardware_concurrency());
}

VirtualDeviceProperties DevicePropertiesEmulation::get_properties() const {
    auto out = props_;
    out.totalGlobalMem = VirtualVramManager::instance().available_virtual_memory();
    return out;
}

int DevicePropertiesEmulation::get_attribute(int attr, int* value) const {
    if (!value) {
        return 1;
    }
    const auto p = get_properties();
    switch (attr) {
    case 0:
        *value = static_cast<int>(p.totalGlobalMem > static_cast<size_t>(INT32_MAX) ? INT32_MAX : p.totalGlobalMem);
        return 0;
    case 1:
        *value = p.multiProcessorCount;
        return 0;
    case 2:
        *value = p.warpSize;
        return 0;
    case 3:
        *value = p.regsPerBlock;
        return 0;
    case 4:
        *value = p.sharedMemPerBlock;
        return 0;
    case 5:
        *value = p.major;
        return 0;
    case 6:
        *value = p.minor;
        return 0;
    default:
        return 1;
    }
}

} // namespace cuda_emulator
