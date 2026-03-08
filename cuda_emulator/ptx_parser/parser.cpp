#include "parser.h"

#include <cctype>
#include <sstream>

namespace cuda_emulator {
namespace {
std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}
} // namespace

std::vector<ParsedKernel> parse_ptx(const std::string& ptx_source) {
    std::vector<ParsedKernel> kernels;
    std::stringstream ss(ptx_source);
    std::string line;
    while (std::getline(ss, line)) {
        auto entry_pos = line.find(".entry");
        if (entry_pos == std::string::npos) {
            continue;
        }
        auto name = trim(line.substr(entry_pos + 6));
        auto paren = name.find('(');
        if (paren != std::string::npos) {
            name = name.substr(0, paren);
        }
        auto space = name.find(' ');
        if (space != std::string::npos) {
            name = name.substr(0, space);
        }
        name = trim(name);
        if (!name.empty()) {
            kernels.push_back({name, {}});
        }
    }
    return kernels;
}

} // namespace cuda_emulator
