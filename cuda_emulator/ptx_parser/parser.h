#pragma once

#include <string>
#include <vector>

namespace cuda_emulator {

struct ParsedKernel {
    std::string name;
    std::string body;
};

std::vector<ParsedKernel> parse_ptx(const std::string& ptx_source);

} // namespace cuda_emulator
