#include "gpu_emulator.h"
#include "scheduler.h"
#include "utils.h"
#include "virtual_vram.h"

#include <algorithm>
#include <cstdint>

namespace cuda_emulator {

GpuComputeEmulator& GpuComputeEmulator::instance() {
    static GpuComputeEmulator emulator;
    return emulator;
}

int GpuComputeEmulator::launch(const KernelLaunch& launch) {
    if (launch_cpu_known_kernel(launch) == 0) {
        return 0;
    }
    return launch_gpu_chunked(launch);
}

int GpuComputeEmulator::launch_cpu_known_kernel(const KernelLaunch& launch) {
    if (launch.name == "vector_add") {
        auto* a_ptr = *reinterpret_cast<void**>(launch.args[0]);
        auto* b_ptr = *reinterpret_cast<void**>(launch.args[1]);
        auto* c_ptr = *reinterpret_cast<void**>(launch.args[2]);
        const auto n = *reinterpret_cast<int*>(launch.args[3]);

        auto a = VirtualVramManager::instance().resolve(a_ptr);
        auto b = VirtualVramManager::instance().resolve(b_ptr);
        auto c = VirtualVramManager::instance().resolve(c_ptr);
        if (!a.allocation || !b.allocation || !c.allocation) {
            return 1;
        }

        if (a.offset != 0 || b.offset != 0 || c.offset != 0) {
            return 1;
        }

        if (a.allocation->bytes.size() < sizeof(float) * static_cast<size_t>(n) ||
            b.allocation->bytes.size() < sizeof(float) * static_cast<size_t>(n) ||
            c.allocation->bytes.size() < sizeof(float) * static_cast<size_t>(n)) {
            return 1;
        }

        auto* a_data = reinterpret_cast<const float*>(a.allocation->bytes.data());
        auto* b_data = reinterpret_cast<const float*>(b.allocation->bytes.data());
        auto* c_data = reinterpret_cast<float*>(c.allocation->bytes.data());

        for (int i = 0; i < n; ++i) {
            c_data[i] = a_data[i] + b_data[i];
        }
        return 0;
    }
    return 1;
}

int GpuComputeEmulator::launch_gpu_chunked(const KernelLaunch& launch) {
    if (launch.name != "matrix_mul") {
        log(LogLevel::Warn, "Unknown kernel fallback: " + launch.name);
        return 1;
    }

    auto* a_ptr = *reinterpret_cast<void**>(launch.args[0]);
    auto* b_ptr = *reinterpret_cast<void**>(launch.args[1]);
    auto* c_ptr = *reinterpret_cast<void**>(launch.args[2]);
    const auto n = *reinterpret_cast<int*>(launch.args[3]);

    auto a = VirtualVramManager::instance().resolve(a_ptr);
    auto b = VirtualVramManager::instance().resolve(b_ptr);
    auto c = VirtualVramManager::instance().resolve(c_ptr);
    if (!a.allocation || !b.allocation || !c.allocation || a.offset != 0 || b.offset != 0 || c.offset != 0) {
        return 1;
    }

    auto* a_data = reinterpret_cast<const float*>(a.allocation->bytes.data());
    auto* b_data = reinterpret_cast<const float*>(b.allocation->bytes.data());
    auto* c_data = reinterpret_cast<float*>(c.allocation->bytes.data());

    return Scheduler::instance().run_chunked(static_cast<size_t>(n), 128, [&](size_t row, size_t row_count, int) {
        for (size_t i = row; i < row + row_count; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    sum += a_data[i * n + k] * b_data[k * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }
        return 0;
    });
}

} // namespace cuda_emulator
