#include "common/logger.h"

#include <iostream>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace cressim::neo::common
{

namespace
{

struct LoggerState
{
    std::mutex mutex;
    LogSeverity minSeverity = LogSeverity::Info;
    LogCallback callback    = nullptr;
    void *callbackUserData  = nullptr;
};

LoggerState &loggerState()
{
    static LoggerState state;
    return state;
}

const char *severityLabel(LogSeverity severity) noexcept
{
    switch (severity)
    {
    case LogSeverity::Trace:
        return "Trace";
    case LogSeverity::Debug:
        return "Debug";
    case LogSeverity::Info:
        return "Info";
    case LogSeverity::Warning:
        return "Warning";
    case LogSeverity::Error:
        return "Error";
    case LogSeverity::Fatal:
        return "Fatal";
    }
    return "Unknown";
}

const char *fileBasename(const char *path) noexcept
{
    if (path == nullptr)
    {
        return "";
    }

    const char *fileName = path;
    for (const char *cursor = path; *cursor != '\0'; ++cursor)
    {
        if (*cursor == '/' || *cursor == '\\')
        {
            fileName = cursor + 1;
        }
    }
    return fileName;
}

std::string formatMessage(LogSeverity severity, const SourceLocation &location,
                          const std::string &message)
{
    std::string normalizedMessage = message;
    while (!normalizedMessage.empty() &&
           (normalizedMessage.back() == '\n' || normalizedMessage.back() == '\r'))
    {
        normalizedMessage.pop_back();
    }

    std::ostringstream stream;
    stream << '[' << severityLabel(severity) << "] " << fileBasename(location.file) << ':'
           << location.line;
    if (location.function != nullptr && location.function[0] != '\0')
    {
        stream << ' ' << location.function;
    }
    stream << ": " << normalizedMessage;
    return stream.str();
}

void writeToDefaultSink(LogSeverity severity, const std::string &formattedMessage)
{
    std::ostream &stream =
        severity >= LogSeverity::Warning ? static_cast<std::ostream &>(std::cerr) : std::cout;
    stream << formattedMessage << '\n';
    stream.flush();

#if defined(_WIN32)
    std::string debugLine = formattedMessage;
    debugLine.push_back('\n');
    OutputDebugStringA(debugLine.c_str());
#endif
}

} // namespace

void setMinLogSeverity(LogSeverity severity) noexcept
{
    LoggerState &state = loggerState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.minSeverity = severity;
}

LogSeverity minLogSeverity() noexcept
{
    LoggerState &state = loggerState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.minSeverity;
}

bool shouldLog(LogSeverity severity) noexcept
{
    return static_cast<int>(severity) >= static_cast<int>(minLogSeverity());
}

void setLogCallback(LogCallback callback, void *userData) noexcept
{
    LoggerState &state = loggerState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.callback         = callback;
    state.callbackUserData = userData;
}

void clearLogCallback() noexcept
{
    setLogCallback(nullptr, nullptr);
}

void writeLogMessage(LogSeverity severity, const SourceLocation &location,
                     const std::string &message)
{
    LoggerState &state = loggerState();
    std::lock_guard<std::mutex> lock(state.mutex);

    const std::string formatted = formatMessage(severity, location, message);
    writeToDefaultSink(severity, formatted);

    if (state.callback != nullptr)
    {
        const LogMessageView view{severity, location, message.c_str()};
        state.callback(view, state.callbackUserData);
    }
}

} // namespace cressim::neo::common
