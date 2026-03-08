#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

namespace cuda_emulator {

enum class LogLevel : uint8_t { Error = 0, Warn, Info, Debug };

void set_log_level(LogLevel level);
void log(LogLevel level, const std::string& message);

class ScopedTimer {
public:
    explicit ScopedTimer(std::string name);
    ~ScopedTimer();

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace cuda_emulator
