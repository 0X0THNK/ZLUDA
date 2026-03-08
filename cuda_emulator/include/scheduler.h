#pragma once

#include <cstddef>
#include <functional>

namespace cuda_emulator {

class Scheduler {
public:
    static Scheduler& instance();

    int run_chunked(size_t bytes, size_t chunk_size, const std::function<int(size_t, size_t, int)>& fn);
    int simulated_gpu_count() const;

private:
    Scheduler();
    int gpu_count_;
};

} // namespace cuda_emulator
