#ifndef CRESSIM_NEO_COMMON_LOGGER_H
#define CRESSIM_NEO_COMMON_LOGGER_H

#include "common/export.h"

#include <sstream>
#include <string>
#include <utility>

namespace cressim::neo::common
{

enum class LogSeverity
{
    Trace = 0,
    Debug,
    Info,
    Warning,
    Error,
    Fatal
};

struct SourceLocation
{
    const char *file     = "";
    const char *function = "";
    int line             = 0;
};

struct LogMessageView
{
    LogSeverity severity{};
    SourceLocation location{};
    const char *message = "";
};

using LogCallback = void (*)(const LogMessageView &message, void *userData);

CRESSIM_NEO_COMMON_API void setMinLogSeverity(LogSeverity severity) noexcept;
CRESSIM_NEO_COMMON_API LogSeverity minLogSeverity() noexcept;
CRESSIM_NEO_COMMON_API bool shouldLog(LogSeverity severity) noexcept;
CRESSIM_NEO_COMMON_API void setLogCallback(LogCallback callback, void *userData) noexcept;
CRESSIM_NEO_COMMON_API void clearLogCallback() noexcept;
CRESSIM_NEO_COMMON_API void writeLogMessage(LogSeverity severity, const SourceLocation &location,
                                            const std::string &message);

namespace detail
{

template <typename... Args>
std::string buildLogMessage(Args &&...args)
{
    std::ostringstream stream;
    (stream << ... << std::forward<Args>(args));
    return stream.str();
}

template <typename... Args>
void logMessage(LogSeverity severity, const SourceLocation &location, Args &&...args)
{
    if (!shouldLog(severity))
    {
        return;
    }

    writeLogMessage(severity, location, buildLogMessage(std::forward<Args>(args)...));
}

} // namespace detail

} // namespace cressim::neo::common

#define CRESSIM_LOG_TRACE(...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Trace,                                            \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

#define CRESSIM_LOG_DEBUG(...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Debug,                                            \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

#define CRESSIM_LOG_INFO(...)                                                                      \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Info,                                             \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

#define CRESSIM_LOG_WARNING(...)                                                                   \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Warning,                                          \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

#define CRESSIM_LOG_ERROR(...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Error,                                            \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

#define CRESSIM_LOG_FATAL(...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Fatal,                                            \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

#define CRESSIM_LOG_WARNING_ONCE(...)                                                              \
    do                                                                                             \
    {                                                                                              \
        static bool isFirstTime = true;                                                            \
        if (isFirstTime)                                                                           \
        {                                                                                          \
            CRESSIM_LOG_WARNING(__VA_ARGS__);                                                      \
            isFirstTime = false;                                                                   \
        }                                                                                          \
    } while (false)

#define CRESSIM_LOG_ERROR_ONCE(...)                                                                \
    do                                                                                             \
    {                                                                                              \
        static bool isFirstTime = true;                                                            \
        if (isFirstTime)                                                                           \
        {                                                                                          \
            CRESSIM_LOG_ERROR(__VA_ARGS__);                                                        \
            isFirstTime = false;                                                                   \
        }                                                                                          \
    } while (false)

#define CRESSIM_LOG_INFO_ONCE(...)                                                                 \
    do                                                                                             \
    {                                                                                              \
        static bool isFirstTime = true;                                                            \
        if (isFirstTime)                                                                           \
        {                                                                                          \
            CRESSIM_LOG_INFO(__VA_ARGS__);                                                         \
            isFirstTime = false;                                                                   \
        }                                                                                          \
    } while (false)

#endif // CRESSIM_NEO_COMMON_LOGGER_H
