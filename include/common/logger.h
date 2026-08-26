#ifndef CRESSIM_NEO_COMMON_LOGGER_H
#define CRESSIM_NEO_COMMON_LOGGER_H

#include "common/export.h"

#include <sstream>
#include <string>
#include <utility>

/// @file logger.h
/// @brief Core logging subsystem, severity levels, message formatting, and log handler callbacks.

namespace cressim::neo::common
{

/// @brief Logging severity levels in ascending priority.
enum class LogSeverity
{
    Trace = 0, ///< Verbose low-level diagnostic tracing.
    Debug,     ///< Diagnostic information useful for debugging.
    Info,      ///< General informational operational events.
    Warning,   ///< Non-fatal warnings indicating unexpected conditions.
    Error,     ///< Recoverable errors or operation failures.
    Fatal      ///< Unrecoverable fatal errors.
};

/// @brief Source code origin information for a log entry.
struct SourceLocation
{
    const char *file     = ""; ///< Source file path.
    const char *function = ""; ///< Function or method name.
    int line             = 0;  ///< Source line number.
};

/// @brief Non-owning view of a log message event passed to log sink callbacks.
struct LogMessageView
{
    LogSeverity severity{};    ///< Severity level of the log message.
    SourceLocation location{}; ///< Source location where the message was logged.
    const char *message = "";  ///< Null-terminated log message text.
};

/// @brief Function pointer signature for custom log sink callbacks.
/// @param message Non-owning view of the formatted log message and metadata.
/// @param userData User-supplied context pointer passed during callback registration.
using LogCallback = void (*)(const LogMessageView &message, void *userData);

/// @brief Sets the minimum severity threshold for log messages to be emitted.
/// @param severity Minimum severity level to log.
CRESSIM_NEO_COMMON_API void setMinLogSeverity(LogSeverity severity) noexcept;

/// @brief Gets the current minimum logging severity threshold.
/// @return Currently active minimum LogSeverity.
CRESSIM_NEO_COMMON_API LogSeverity minLogSeverity() noexcept;

/// @brief Checks whether the logging helpers will emit a message of the given severity.
/// @param severity Severity level to test.
/// @return True if @p severity meets the current minimum threshold, false otherwise.
CRESSIM_NEO_COMMON_API bool shouldLog(LogSeverity severity) noexcept;

/// @brief Registers a callback that observes each message written through writeLogMessage().
///
/// Messages continue to be written to the default console sink while a callback is registered.
/// @param callback Function pointer to invoke after default-sink output.
/// @param userData User context pointer passed to the callback.
CRESSIM_NEO_COMMON_API void setLogCallback(LogCallback callback, void *userData) noexcept;

/// @brief Clears any previously registered log callback, resetting to default console output.
CRESSIM_NEO_COMMON_API void clearLogCallback() noexcept;

/// @brief Writes a message to the default sink and, if registered, the callback.
///
/// This low-level function does not apply the minimum severity threshold; use logMessage() or a
/// CRESSIM_LOG_* macro when threshold filtering is required. The callback receives the original
/// @p message text, while console output includes formatted severity and source information.
/// @param severity Log severity level.
/// @param location Source code origin location.
/// @param message Log message string content.
CRESSIM_NEO_COMMON_API void writeLogMessage(LogSeverity severity, const SourceLocation &location,
                                            const std::string &message);

namespace detail
{

/// @brief Concatenates variadic arguments into a single string using `std::ostringstream`.
/// @tparam Args Argument types supported by stream insertion (`operator<<`).
/// @param args Arguments to format into the message.
/// @return Resulting concatenated message string.
template <typename... Args>
std::string buildLogMessage(Args &&...args)
{
    std::ostringstream stream;
    (stream << ... << std::forward<Args>(args));
    return stream.str();
}

/// @brief Checks severity and formats/writes a log message if active.
/// @tparam Args Argument types supported by stream insertion.
/// @param severity Log severity level.
/// @param location Source code location.
/// @param args Arguments to format into the message.
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

/// @def CRESSIM_LOG_TRACE(...)
/// @brief Logs a formatted trace message with current file, function, and line.
#define CRESSIM_LOG_TRACE(...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Trace,                                            \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

/// @def CRESSIM_LOG_DEBUG(...)
/// @brief Logs a formatted debug message with current file, function, and line.
#define CRESSIM_LOG_DEBUG(...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Debug,                                            \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

/// @def CRESSIM_LOG_INFO(...)
/// @brief Logs a formatted informational message with current file, function, and line.
#define CRESSIM_LOG_INFO(...)                                                                      \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Info,                                             \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

/// @def CRESSIM_LOG_WARNING(...)
/// @brief Logs a formatted warning message with current file, function, and line.
#define CRESSIM_LOG_WARNING(...)                                                                   \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Warning,                                          \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

/// @def CRESSIM_LOG_ERROR(...)
/// @brief Logs a formatted error message with current file, function, and line.
#define CRESSIM_LOG_ERROR(...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Error,                                            \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

/// @def CRESSIM_LOG_FATAL(...)
/// @brief Logs a formatted fatal message with current file, function, and line.
#define CRESSIM_LOG_FATAL(...)                                                                     \
    do                                                                                             \
    {                                                                                              \
        ::cressim::neo::common::detail::logMessage(                                                \
            ::cressim::neo::common::LogSeverity::Fatal,                                            \
            ::cressim::neo::common::SourceLocation{__FILE__, __FUNCTION__, __LINE__},              \
            __VA_ARGS__);                                                                          \
    } while (false)

/// @def CRESSIM_LOG_WARNING_ONCE(...)
/// @brief Logs a formatted warning message only the first time execution passes through the call
/// site.
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

/// @def CRESSIM_LOG_ERROR_ONCE(...)
/// @brief Logs a formatted error message only the first time execution passes through the call
/// site.
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

/// @def CRESSIM_LOG_INFO_ONCE(...)
/// @brief Logs a formatted info message only the first time execution passes through the call site.
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
