#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cuda_emulator {

struct VirtualAllocation {
    std::vector<uint8_t> bytes;
};

struct AllocationView {
    std::shared_ptr<VirtualAllocation> allocation;
    size_t offset = 0;
};

class VirtualVramManager {
public:
    static VirtualVramManager& instance();

    int allocate(void** virtual_ptr, size_t size);
    int release(void* virtual_ptr);
    int memcpy_to_virtual(void* dst_virtual, const void* src_host, size_t size);
    int memcpy_from_virtual(void* dst_host, const void* src_virtual, size_t size);
    int memcpy_virtual_to_virtual(void* dst_virtual, const void* src_virtual, size_t size);

    AllocationView resolve(const void* ptr) const;

    size_t total_allocated() const;
    size_t available_virtual_memory() const;

private:
    VirtualVramManager();

    mutable std::mutex mutex_;
    std::unordered_map<void*, std::shared_ptr<VirtualAllocation>> allocations_;
    size_t total_allocated_ = 0;
    size_t ram_budget_ = 0;
};

} // namespace cuda_emulator
