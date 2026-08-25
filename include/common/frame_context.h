#ifndef CRESSIM_NEO_COMMON_FRAME_CONTEXT_H
#define CRESSIM_NEO_COMMON_FRAME_CONTEXT_H

#include <cstdint>

namespace cressim::neo::common
{

/// @brief Temporal execution state and timing parameters for a single simulation or rendering frame.
///
/// In Python, this structure is exposed as `cressim_neo.FrameContext` and is passed to tick callbacks
/// such as `before_tick` and `after_tick` during viewer execution.
///
/// @see cressim_neo.FrameContext
struct FrameContext
{
    std::uint64_t frameIndex = 0;   ///< Monotonically increasing frame index counter.
    double timeSeconds       = 0.0; ///< Total accumulated simulation time in seconds.
    float deltaSeconds       = 0.0f;///< Elapsed time step for the current frame in seconds.
};

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_FRAME_CONTEXT_H
