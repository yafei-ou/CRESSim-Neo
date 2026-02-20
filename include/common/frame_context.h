#ifndef CRESSIM_NEO_COMMON_FRAME_CONTEXT_H
#define CRESSIM_NEO_COMMON_FRAME_CONTEXT_H

#include <cstdint>

namespace cressim::neo::common
{

struct FrameContext
{
    std::uint64_t frameIndex = 0;
    double timeSeconds       = 0.0;
    float deltaSeconds       = 0.0f;
};

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_FRAME_CONTEXT_H
