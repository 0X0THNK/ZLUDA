cmake_minimum_required(VERSION 3.16)
project(cuda_emulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cuda_emulator SHARED
    src/cuda_emulator.cpp
    src/virtual_vram.cpp
    src/gpu_emulator.cpp
    src/scheduler.cpp
    src/device_props.cpp
    src/utils.cpp
    ptx_parser/parser.cpp
    ptx_parser/jit_compiler.cpp
)

target_include_directories(cuda_emulator
    PUBLIC include
    PRIVATE ptx_parser
)

add_executable(full_emulation_tests tests/full_emulation_tests.cpp)
target_link_libraries(full_emulation_tests PRIVATE cuda_emulator)
target_include_directories(full_emulation_tests PRIVATE include)

enable_testing()
add_test(NAME full_emulation_tests COMMAND full_emulation_tests)
