#include "virtual_vram.h"

#include <algorithm>
#include <cstring>

namespace cuda_emulator {

VirtualVramManager& VirtualVramManager::instance() {
    static VirtualVramManager manager;
    return manager;
}

VirtualVramManager::VirtualVramManager() : ram_budget_(size_t(64) * 1024 * 1024 * 1024) {}

int VirtualVramManager::allocate(void** virtual_ptr, size_t size) {
    if (!virtual_ptr || size == 0) {
        return 1;
    }
    auto* raw = new (std::nothrow) uint8_t[size];
    if (!raw) {
        return 2;
    }
    auto allocation = std::make_shared<VirtualAllocation>();
    allocation->bytes.resize(size);

    std::scoped_lock lock(mutex_);
    if (total_allocated_ + size > ram_budget_) {
        delete[] raw;
        return 2;
    }
    allocations_[raw] = std::move(allocation);
    total_allocated_ += size;
    *virtual_ptr = raw;
    return 0;
}

int VirtualVramManager::release(void* virtual_ptr) {
    if (!virtual_ptr) {
        return 1;
    }
    std::scoped_lock lock(mutex_);
    auto it = allocations_.find(virtual_ptr);
    if (it == allocations_.end()) {
        return 1;
    }
    total_allocated_ -= it->second->bytes.size();
    allocations_.erase(it);
    delete[] static_cast<uint8_t*>(virtual_ptr);
    return 0;
}

AllocationView VirtualVramManager::resolve(const void* ptr) const {
    std::scoped_lock lock(mutex_);
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    for (const auto& [base, allocation] : allocations_) {
        const auto begin = reinterpret_cast<uintptr_t>(base);
        const auto end = begin + allocation->bytes.size();
        if (addr >= begin && addr < end) {
            return AllocationView{allocation, static_cast<size_t>(addr - begin)};
        }
    }
    return AllocationView{};
}

int VirtualVramManager::memcpy_to_virtual(void* dst_virtual, const void* src_host, size_t size) {
    if (!src_host) {
        return 1;
    }
    auto dst = resolve(dst_virtual);
    if (!dst.allocation || dst.offset + size > dst.allocation->bytes.size()) {
        return 1;
    }
    std::memcpy(dst.allocation->bytes.data() + dst.offset, src_host, size);
    return 0;
}

int VirtualVramManager::memcpy_from_virtual(void* dst_host, const void* src_virtual, size_t size) {
    if (!dst_host) {
        return 1;
    }
    auto src = resolve(src_virtual);
    if (!src.allocation || src.offset + size > src.allocation->bytes.size()) {
        return 1;
    }
    std::memcpy(dst_host, src.allocation->bytes.data() + src.offset, size);
    return 0;
}

int VirtualVramManager::memcpy_virtual_to_virtual(void* dst_virtual, const void* src_virtual, size_t size) {
    auto dst = resolve(dst_virtual);
    auto src = resolve(src_virtual);
    if (!dst.allocation || !src.allocation || dst.offset + size > dst.allocation->bytes.size() ||
        src.offset + size > src.allocation->bytes.size()) {
        return 1;
    }
    std::memmove(dst.allocation->bytes.data() + dst.offset, src.allocation->bytes.data() + src.offset, size);
    return 0;
}

size_t VirtualVramManager::total_allocated() const {
    std::scoped_lock lock(mutex_);
    return total_allocated_;
}

size_t VirtualVramManager::available_virtual_memory() const {
    std::scoped_lock lock(mutex_);
    return ram_budget_ > total_allocated_ ? ram_budget_ - total_allocated_ : 0;
}

} // namespace cuda_emulator
