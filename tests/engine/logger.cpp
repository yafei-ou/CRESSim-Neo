#include "common/logger.h"

#include <string>
#include <vector>

namespace
{

struct CapturedLogMessage
{
    cressim::neo::common::LogSeverity severity{};
    cressim::neo::common::SourceLocation location{};
    std::string message;
};

void captureLog(const cressim::neo::common::LogMessageView& message, void* userData)
{
    auto* captured = static_cast<std::vector<CapturedLogMessage>*>(userData);
    captured->push_back(
        CapturedLogMessage{message.severity, message.location, message.message != nullptr
                                                                ? message.message
                                                                : ""});
}

int testSeverityFiltering()
{
    using namespace cressim::neo::common;

    std::vector<CapturedLogMessage> captured;
    setLogCallback(&captureLog, &captured);
    setMinLogSeverity(LogSeverity::Warning);

    CRESSIM_LOG_INFO("filtered info");
    CRESSIM_LOG_WARNING("visible warning");

    clearLogCallback();
    setMinLogSeverity(LogSeverity::Info);

    if (captured.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Expected exactly one captured log after severity filtering.");
        return 1;
    }
    if (captured.front().severity != LogSeverity::Warning ||
        captured.front().message != "visible warning")
    {
        CRESSIM_LOG_ERROR("Unexpected captured warning payload.");
        return 1;
    }
    return 0;
}

int testCallbackMetadata()
{
    using namespace cressim::neo::common;

    std::vector<CapturedLogMessage> captured;
    setLogCallback(&captureLog, &captured);
    setMinLogSeverity(LogSeverity::Trace);

    CRESSIM_LOG_ERROR("metadata check");

    clearLogCallback();
    setMinLogSeverity(LogSeverity::Info);

    if (captured.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Expected callback metadata test to capture one message.");
        return 1;
    }
    if (captured.front().severity != LogSeverity::Error || captured.front().location.line <= 0 ||
        captured.front().location.file == nullptr || captured.front().location.function == nullptr)
    {
        CRESSIM_LOG_ERROR("Captured log metadata was incomplete.");
        return 1;
    }
    if (captured.front().message != "metadata check")
    {
        CRESSIM_LOG_ERROR("Captured log message text did not match.");
        return 1;
    }
    return 0;
}

int testOnceLogging()
{
    using namespace cressim::neo::common;

    std::vector<CapturedLogMessage> captured;
    setLogCallback(&captureLog, &captured);
    setMinLogSeverity(LogSeverity::Info);

    for (int i = 0; i < 3; ++i)
    {
        CRESSIM_LOG_WARNING_ONCE("once warning");
    }

    clearLogCallback();

    if (captured.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Expected once logger to emit exactly one message.");
        return 1;
    }
    if (captured.front().severity != LogSeverity::Warning ||
        captured.front().message != "once warning")
    {
        CRESSIM_LOG_ERROR("Once logger emitted an unexpected payload.");
        return 1;
    }
    return 0;
}

} // namespace

int main()
{
    if (const int result = testSeverityFiltering(); result != 0)
    {
        return result;
    }
    if (const int result = testCallbackMetadata(); result != 0)
    {
        return result;
    }
    if (const int result = testOnceLogging(); result != 0)
    {
        return result;
    }

    CRESSIM_LOG_INFO("Logger checks passed.");
    return 0;
}
