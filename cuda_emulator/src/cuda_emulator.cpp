#include "cuda_emulator.h"

#include "device_props.h"
#include "gpu_emulator.h"
#include "parser.h"
#include "utils.h"
#include "virtual_vram.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct StreamState {
    std::mutex mutex;
    std::vector<std::future<int>> pending;
};

struct EventState {
    std::mutex mutex;
    std::vector<std::future<int>> captured;
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
};

struct ModuleState {
    std::string path;
    std::unordered_set<std::string> kernels;
};

struct FunctionState {
    CUmodule module = nullptr;
    std::string name;
};

std::mutex g_state_mutex;
std::unordered_map<CUmodule, ModuleState> g_modules;
std::unordered_map<const void*, FunctionState> g_functions;
std::unordered_map<cudaStream_t, std::shared_ptr<StreamState>> g_streams;
std::unordered_map<cudaEvent_t, std::shared_ptr<EventState>> g_events;

cuda_emulator::KernelLaunch to_launch(const std::string& name,
                                      unsigned int gridDimX,
                                      unsigned int gridDimY,
                                      unsigned int gridDimZ,
                                      unsigned int blockDimX,
                                      unsigned int blockDimY,
                                      unsigned int blockDimZ,
                                      void** kernelParams) {
    return cuda_emulator::KernelLaunch{name,
                                       kernelParams,
                                       cuda_emulator::Dim3{gridDimX, gridDimY, gridDimZ},
                                       cuda_emulator::Dim3{blockDimX, blockDimY, blockDimZ}};
}

std::shared_ptr<StreamState> get_stream(cudaStream_t stream) {
    std::scoped_lock lock(g_state_mutex);
    if (!stream) {
        static auto default_stream = std::make_shared<StreamState>();
        return default_stream;
    }
    auto it = g_streams.find(stream);
    if (it == g_streams.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<EventState> get_event(cudaEvent_t event) {
    std::scoped_lock lock(g_state_mutex);
    auto it = g_events.find(event);
    if (it == g_events.end()) {
        return nullptr;
    }
    return it->second;
}

int stream_push(cudaStream_t stream, std::future<int>&& future) {
    auto s = get_stream(stream);
    if (!s) {
        return cudaErrorInvalidValue;
    }
    std::scoped_lock lock(s->mutex);
    s->pending.push_back(std::move(future));
    return cudaSuccess;
}

int stream_sync(cudaStream_t stream) {
    auto s = get_stream(stream);
    if (!s) {
        return cudaErrorInvalidValue;
    }
    std::vector<std::future<int>> pending;
    {
        std::scoped_lock lock(s->mutex);
        pending.swap(s->pending);
    }
    for (auto& task : pending) {
        if (task.get() != 0) {
            return cudaErrorUnknown;
        }
    }
    return cudaSuccess;
}

cudaError_t do_memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    switch (kind) {
    case cudaMemcpyHostToDevice:
        return cuda_emulator::VirtualVramManager::instance().memcpy_to_virtual(dst, src, count);
    case cudaMemcpyDeviceToHost:
        return cuda_emulator::VirtualVramManager::instance().memcpy_from_virtual(dst, src, count);
    case cudaMemcpyDeviceToDevice:
        return cuda_emulator::VirtualVramManager::instance().memcpy_virtual_to_virtual(dst, src, count);
    case cudaMemcpyHostToHost:
        std::memcpy(dst, src, count);
        return cudaSuccess;
    default:
        return cudaErrorInvalidValue;
    }
}

std::unordered_set<std::string> parse_module_kernels(const std::string& path) {
    std::ifstream file(path);
    if (!file.good()) {
        return {};
    }
    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::unordered_set<std::string> names;
    for (const auto& kernel : cuda_emulator::parse_ptx(source)) {
        names.insert(kernel.name);
    }
    return names;
}

} // namespace

extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    return cuda_emulator::VirtualVramManager::instance().allocate(devPtr, size);
}

cudaError_t cudaFree(void* devPtr) {
    return cuda_emulator::VirtualVramManager::instance().release(devPtr);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    return do_memcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    return stream_push(stream, std::async(std::launch::async, [=]() { return do_memcpy(dst, src, count, kind); }));
}

cudaError_t cudaMemGetInfo(size_t* free_mem, size_t* total_mem) {
    if (!free_mem || !total_mem) {
        return cudaErrorInvalidValue;
    }
    auto& vram = cuda_emulator::VirtualVramManager::instance();
    *free_mem = vram.available_virtual_memory();
    *total_mem = vram.total_allocated() + vram.available_virtual_memory();
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
    if (!prop) {
        return cudaErrorInvalidValue;
    }
    const auto p = cuda_emulator::DevicePropertiesEmulation::instance().get_properties();
    std::memcpy(prop->name, p.name, sizeof(prop->name));
    prop->totalGlobalMem = p.totalGlobalMem;
    prop->multiProcessorCount = p.multiProcessorCount;
    prop->warpSize = p.warpSize;
    prop->regsPerBlock = p.regsPerBlock;
    prop->sharedMemPerBlock = p.sharedMemPerBlock;
    prop->major = p.major;
    prop->minor = p.minor;
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t, cudaStream_t stream) {
    std::string kernel_name;
    {
        std::scoped_lock lock(g_state_mutex);
        auto it = g_functions.find(func);
        if (it == g_functions.end()) {
            return cudaErrorInvalidValue;
        }
        kernel_name = it->second.name;
    }

    auto launch = to_launch(kernel_name, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, args);
    if (!stream) {
        return cuda_emulator::GpuComputeEmulator::instance().launch(launch);
    }
    return stream_push(stream,
                       std::async(std::launch::async,
                                  [launch]() mutable { return cuda_emulator::GpuComputeEmulator::instance().launch(launch); }));
}

cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    if (!stream) {
        return cudaErrorInvalidValue;
    }
    auto state = std::make_shared<StreamState>();
    auto* token = new int(0x1234);
    {
        std::scoped_lock lock(g_state_mutex);
        g_streams[token] = state;
    }
    *stream = token;
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (!stream) {
        return cudaErrorInvalidValue;
    }
    auto rc = stream_sync(stream);
    std::scoped_lock lock(g_state_mutex);
    g_streams.erase(stream);
    delete static_cast<int*>(stream);
    return rc;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return stream_sync(stream);
}

cudaError_t cudaDeviceSynchronize() {
    std::vector<cudaStream_t> handles;
    {
        std::scoped_lock lock(g_state_mutex);
        handles.reserve(g_streams.size());
        for (const auto& [h, _] : g_streams) {
            handles.push_back(h);
        }
    }
    for (auto h : handles) {
        if (stream_sync(h) != cudaSuccess) {
            return cudaErrorUnknown;
        }
    }
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    if (!event) {
        return cudaErrorInvalidValue;
    }
    auto* token = new int(0xEEEE);
    auto state = std::make_shared<EventState>();
    {
        std::scoped_lock lock(g_state_mutex);
        g_events[token] = state;
    }
    *event = token;
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (!event) {
        return cudaErrorInvalidValue;
    }
    {
        auto state = get_event(event);
        if (!state) {
            return cudaErrorInvalidValue;
        }
        std::vector<std::future<int>> pending;
        {
            std::scoped_lock lock(state->mutex);
            pending.swap(state->captured);
        }
        for (auto& task : pending) {
            (void)task.get();
        }
    }
    std::scoped_lock lock(g_state_mutex);
    g_events.erase(event);
    delete static_cast<int*>(event);
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    auto e = get_event(event);
    auto s = get_stream(stream);
    if (!e || !s) {
        return cudaErrorInvalidValue;
    }
    std::vector<std::future<int>> captured;
    {
        std::scoped_lock lock(s->mutex);
        for (auto& f : s->pending) {
            captured.push_back(std::move(f));
        }
        s->pending.clear();
    }
    {
        std::scoped_lock lock(e->mutex);
        for (auto& f : captured) {
            e->captured.push_back(std::move(f));
        }
        e->timestamp = std::chrono::steady_clock::now();
    }
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    auto e = get_event(event);
    if (!e) {
        return cudaErrorInvalidValue;
    }
    std::vector<std::future<int>> pending;
    {
        std::scoped_lock lock(e->mutex);
        pending.swap(e->captured);
    }
    for (auto& task : pending) {
        if (task.get() != 0) {
            return cudaErrorUnknown;
        }
    }
    return cudaSuccess;
}

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    return cuda_emulator::VirtualVramManager::instance().allocate(dptr, bytesize);
}

CUresult cuMemFree(CUdeviceptr dptr) {
    return cuda_emulator::VirtualVramManager::instance().release(dptr);
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    return cuda_emulator::VirtualVramManager::instance().memcpy_to_virtual(dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    return cuda_emulator::VirtualVramManager::instance().memcpy_from_virtual(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    return cuda_emulator::VirtualVramManager::instance().memcpy_virtual_to_virtual(dstDevice, srcDevice, ByteCount);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
    return cudaMemcpyAsync(dstDevice, srcHost, ByteCount, cudaMemcpyHostToDevice, hStream);
}

CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return cudaMemcpyAsync(dstHost, srcDevice, ByteCount, cudaMemcpyDeviceToHost, hStream);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return cudaMemcpyAsync(dstDevice, srcDevice, ByteCount, cudaMemcpyDeviceToDevice, hStream);
}

CUresult cuMemGetInfo(size_t* free_mem, size_t* total_mem) {
    return cudaMemGetInfo(free_mem, total_mem);
}

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    if (!module || !fname) {
        return cudaErrorInvalidValue;
    }

    auto kernels = parse_module_kernels(fname);
    if (kernels.empty()) {
        return cudaErrorInvalidValue;
    }

    auto* token = new int(0xCAFE);
    {
        std::scoped_lock lock(g_state_mutex);
        g_modules[token] = ModuleState{fname, std::move(kernels)};
    }
    *module = token;
    return cudaSuccess;
}

CUresult cuModuleUnload(CUmodule module) {
    if (!module) {
        return cudaErrorInvalidValue;
    }
    std::scoped_lock lock(g_state_mutex);
    for (auto it = g_functions.begin(); it != g_functions.end();) {
        if (it->second.module == module) {
            delete static_cast<int*>(const_cast<void*>(it->first));
            it = g_functions.erase(it);
        } else {
            ++it;
        }
    }
    g_modules.erase(module);
    delete static_cast<int*>(module);
    return cudaSuccess;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (!hfunc || !name || !hmod) {
        return cudaErrorInvalidValue;
    }
    std::scoped_lock lock(g_state_mutex);
    auto module_it = g_modules.find(hmod);
    if (module_it == g_modules.end() || module_it->second.kernels.find(name) == module_it->second.kernels.end()) {
        return cudaErrorInvalidValue;
    }
    auto* token = new int(42);
    g_functions[token] = FunctionState{hmod, name};
    *hfunc = reinterpret_cast<CUfunction>(token);
    return cudaSuccess;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int,
                        CUstream hStream,
                        void** kernelParams,
                        void**) {
    std::string kernel_name;
    {
        std::scoped_lock lock(g_state_mutex);
        auto it = g_functions.find(f);
        if (it == g_functions.end()) {
            return cudaErrorInvalidValue;
        }
        kernel_name = it->second.name;
    }
    auto launch = to_launch(kernel_name, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, kernelParams);
    if (!hStream) {
        return cuda_emulator::GpuComputeEmulator::instance().launch(launch);
    }
    return stream_push(hStream,
                       std::async(std::launch::async,
                                  [launch]() mutable { return cuda_emulator::GpuComputeEmulator::instance().launch(launch); }));
}

CUresult cuDeviceGetAttribute(int* pi, int attrib, CUdevice) {
    return cuda_emulator::DevicePropertiesEmulation::instance().get_attribute(attrib, pi);
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int) {
    return cudaStreamCreate(phStream);
}

CUresult cuStreamDestroy(CUstream hStream) {
    return cudaStreamDestroy(hStream);
}

CUresult cuStreamSynchronize(CUstream hStream) {
    return cudaStreamSynchronize(hStream);
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int) {
    return cudaEventCreate(phEvent);
}

CUresult cuEventDestroy(CUevent hEvent) {
    return cudaEventDestroy(hEvent);
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    return cudaEventRecord(hEvent, hStream);
}

CUresult cuEventSynchronize(CUevent hEvent) {
    return cudaEventSynchronize(hEvent);
}

CUresult cuCtxSynchronize() {
    return cudaDeviceSynchronize();
}

} // extern "C"
