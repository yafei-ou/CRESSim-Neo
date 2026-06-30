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

CRESSIM_NEO_ENGINE_API bool computeUltrasoundProbeLayout(
    const UltrasoundProbeComponent &probeComponent,
    const UltrasoundRendererComponent &rendererComponent, UltrasoundProbeLayout &outLayout);

class CRESSIM_NEO_ENGINE_API UltrasoundSystem
{
public:
    UltrasoundSystem(gpu::GpuDevice &device, physics::PhysicsSolver &physicsSolver);
    ~UltrasoundSystem();

    bool initialize();
    void shutdown();
    bool prepare(World &world);
    bool execute(const common::FrameContext &frameContext, World &world);
    bool computeProbeLayout(const UltrasoundProbeComponent &probeComponent,
                            const UltrasoundRendererComponent &rendererComponent,
                            UltrasoundProbeLayout &outLayout) const;

private:
    struct Impl;

    gpu::GpuDevice &mDevice;
    physics::PhysicsSolver &mPhysicsSolver;
    std::unique_ptr<Impl> mImpl;
    bool mInitialized = false;
};

} // namespace cressim::neo::engine

#endif
