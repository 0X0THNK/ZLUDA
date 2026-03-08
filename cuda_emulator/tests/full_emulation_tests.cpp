#include "cuda_emulator.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <vector>

int main() {
    {
        std::ofstream ptx("dummy.ptx");
        ptx << ".version 7.0\n"
               ".target sm_80\n"
               ".address_size 64\n"
               ".visible .entry vector_add(\n"
               ") {\n"
               "}\n";
    }

    constexpr int n = 1024;
    std::vector<float> a(n, 1.5f), b(n, 2.0f), c(n, 0.0f);

    void *da = nullptr, *db = nullptr, *dc = nullptr;
    assert(cudaMalloc(&da, sizeof(float) * n) == cudaSuccess);
    assert(cudaMalloc(&db, sizeof(float) * n) == cudaSuccess);
    assert(cudaMalloc(&dc, sizeof(float) * n) == cudaSuccess);

    cudaStream_t stream = nullptr;
    assert(cudaStreamCreate(&stream) == cudaSuccess);
    assert(cudaMemcpyAsync(da, a.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream) == cudaSuccess);
    assert(cudaMemcpyAsync(db, b.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream) == cudaSuccess);

    cudaEvent_t ready_event = nullptr;
    assert(cudaEventCreate(&ready_event) == cudaSuccess);
    assert(cudaEventRecord(ready_event, stream) == cudaSuccess);
    assert(cudaEventSynchronize(ready_event) == cudaSuccess);

    CUmodule mod;
    CUfunction fn;
    assert(cuModuleLoad(&mod, "dummy.ptx") == cudaSuccess);
    assert(cuModuleGetFunction(&fn, mod, "vector_add") == cudaSuccess);

    void* args[] = {&da, &db, &dc, (void*)&n};
    assert(cuLaunchKernel(fn, (n + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr) == cudaSuccess);
    assert(cuStreamSynchronize(stream) == cudaSuccess);

    assert(cudaMemcpy(c.data(), dc, sizeof(float) * n, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : c) {
        assert(v == 3.5f);
    }

    std::vector<uint8_t> byte_src(64, 0xAB), byte_dst(64, 0);
    void* dbytes = nullptr;
    assert(cudaMalloc(&dbytes, 128) == cudaSuccess);
    auto* mid = static_cast<void*>(static_cast<uint8_t*>(dbytes) + 32);
    assert(cudaMemcpy(mid, byte_src.data(), byte_src.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(byte_dst.data(), mid, byte_dst.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(byte_dst == byte_src);

    size_t free_mem = 0;
    size_t total_mem = 0;
    assert(cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess);
    assert(total_mem >= free_mem);

    cudaDeviceProp prop{};
    assert(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
    assert(prop.warpSize == 32);

    assert(cuModuleUnload(mod) == cudaSuccess);
    assert(cudaEventDestroy(ready_event) == cudaSuccess);
    assert(cudaStreamDestroy(stream) == cudaSuccess);
    assert(cudaFree(da) == cudaSuccess);
    assert(cudaFree(db) == cudaSuccess);
    assert(cudaFree(dc) == cudaSuccess);
    assert(cudaFree(dbytes) == cudaSuccess);
    return 0;
}
