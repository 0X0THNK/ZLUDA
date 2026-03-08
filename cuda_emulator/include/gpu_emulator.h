#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cuda_emulator {

struct Dim3 {
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;
};

struct KernelLaunch {
    std::string name;
    void** args;
    Dim3 grid;
    Dim3 block;
};

class GpuComputeEmulator {
public:
    static GpuComputeEmulator& instance();

    int launch(const KernelLaunch& launch);

private:
    GpuComputeEmulator() = default;
    int launch_cpu_known_kernel(const KernelLaunch& launch);
    int launch_gpu_chunked(const KernelLaunch& launch);
};

} // namespace cuda_emulator
