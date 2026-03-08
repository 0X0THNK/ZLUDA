#include "utils.h"

#include <iostream>

namespace cuda_emulator {
namespace {
std::mutex g_log_mutex;
LogLevel g_level = LogLevel::Info;

const char* level_to_str(LogLevel level) {
    switch (level) {
    case LogLevel::Error:
        return "ERROR";
    case LogLevel::Warn:
        return "WARN";
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Debug:
        return "DEBUG";
    }
    return "UNKNOWN";
}
} // namespace

void set_log_level(LogLevel level) {
    std::scoped_lock lock(g_log_mutex);
    g_level = level;
}

void log(LogLevel level, const std::string& message) {
    std::scoped_lock lock(g_log_mutex);
    if (static_cast<int>(level) > static_cast<int>(g_level)) {
        return;
    }
    std::cerr << "[cuda_emulator][" << level_to_str(level) << "] " << message << '\n';
}

ScopedTimer::ScopedTimer(std::string name) : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_);
    log(LogLevel::Debug, name_ + " took " + std::to_string(elapsed.count()) + " us");
}

} // namespace cuda_emulator
