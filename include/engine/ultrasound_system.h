#ifndef CRESSIM_NEO_ENGINE_ULTRASOUND_SYSTEM_H
#define CRESSIM_NEO_ENGINE_ULTRASOUND_SYSTEM_H

#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/export.h"

#include <memory>

namespace cressim::neo::gpu
{
class GpuDevice;
}

namespace cressim::neo::physics
{
class PhysicsSolver;
}

namespace cressim::neo::engine
{

class World;

/// @brief Computes the output image layout implied by ultrasound probe and renderer settings.
/// @param probeComponent Probe configuration. It must be enabled.
/// @param rendererComponent Renderer configuration that determines requested output dimensions.
/// @param outLayout Receives an empty layout on failure.
/// @return False when ultrasound support is unavailable, the probe is disabled, or the underlying
/// ultrasound engine cannot be configured.
CRESSIM_NEO_ENGINE_API bool computeUltrasoundProbeLayout(
    const UltrasoundProbeComponent &probeComponent,
    const UltrasoundRendererComponent &rendererComponent, UltrasoundProbeLayout &outLayout);

class CRESSIM_NEO_ENGINE_API UltrasoundSystem
{
public:
    /// @brief Creates an ultrasound system using the supplied GPU device and physics solver.
    UltrasoundSystem(gpu::GpuDevice &device, physics::PhysicsSolver &physicsSolver);

    /// @brief Shuts down the system and releases per-probe resources.
    ~UltrasoundSystem();

    /// @brief Resets and initializes system state.
    bool initialize();

    /// @brief Releases probe runtimes, output targets, and CUDA bridge resources.
    void shutdown();

    /// @brief Prepares enabled probes and publishes their current output metadata to @p world.
    /// @return False only when the system is uninitialized or has been disabled after a fatal
    /// ultrasound setup failure; unavailable optional backends cause preparation to be skipped.
    bool prepare(World &world);

    /// @brief Executes enabled ultrasound probes for @p frameContext and updates @p world results.
    /// @return False only when the system is uninitialized or disabled after a fatal setup error;
    /// unavailable optional backends cause execution to be skipped.
    bool execute(const common::FrameContext &frameContext, World &world);

    /// @brief Computes a probe output layout using the same rules as the free function.
    bool computeProbeLayout(const UltrasoundProbeComponent &probeComponent,
                            const UltrasoundRendererComponent &rendererComponent,
                            UltrasoundProbeLayout &outLayout) const;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::engine

#endif
