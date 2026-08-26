#ifndef CRESSIM_NEO_COMMON_FRAME_CONTEXT_H
#define CRESSIM_NEO_COMMON_FRAME_CONTEXT_H

#include <cstdint>

namespace cressim::neo::common
{

/// @brief Temporal execution state and timing parameters for a single simulation or rendering
/// frame.
struct FrameContext
{
    std::uint64_t frameIndex = 0;    ///< Index of this frame, typically monotonically increasing.
    double timeSeconds       = 0.0;  ///< Accumulated simulation time at this frame, in seconds.
    float deltaSeconds       = 0.0f; ///< Elapsed time step for this frame, in seconds.
};

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_FRAME_CONTEXT_H
