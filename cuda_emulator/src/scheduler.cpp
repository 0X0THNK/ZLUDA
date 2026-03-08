#include "scheduler.h"

#include <algorithm>
#include <cstdlib>
#include <future>
#include <vector>

namespace cuda_emulator {

Scheduler& Scheduler::instance() {
    static Scheduler scheduler;
    return scheduler;
}

Scheduler::Scheduler() : gpu_count_(1) {
    if (const char* env = std::getenv("CUDA_EMULATOR_GPU_COUNT")) {
        gpu_count_ = std::max(1, std::atoi(env));
    }
}

int Scheduler::run_chunked(size_t bytes, size_t chunk_size, const std::function<int(size_t, size_t, int)>& fn) {
    if (bytes == 0 || chunk_size == 0) {
        return 1;
    }
    const size_t chunks = (bytes + chunk_size - 1) / chunk_size;
    std::vector<std::future<int>> futures;
    futures.reserve(chunks);
    for (size_t i = 0; i < chunks; ++i) {
        const size_t offset = i * chunk_size;
        const size_t size = std::min(chunk_size, bytes - offset);
        const int target_gpu = static_cast<int>(i % static_cast<size_t>(gpu_count_));
        futures.push_back(std::async(std::launch::async, [=]() { return fn(offset, size, target_gpu); }));
    }
    for (auto& future : futures) {
        if (future.get() != 0) {
            return 1;
        }
    }
    return 0;
}

int Scheduler::simulated_gpu_count() const {
    return gpu_count_;
}

} // namespace cuda_emulator
