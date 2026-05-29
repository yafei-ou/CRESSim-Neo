#ifndef CRESSIM_NEO_ENGINE_ULTRASOUND_SYSTEM_H
#define CRESSIM_NEO_ENGINE_ULTRASOUND_SYSTEM_H

#include "common/frame_context.h"
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

class CRESSIM_NEO_ENGINE_API UltrasoundSystem
{
public:
    UltrasoundSystem(gpu::GpuDevice &device, physics::PhysicsSolver &physicsSolver);
    ~UltrasoundSystem();

    bool initialize();
    void shutdown();
    bool tick(const common::FrameContext &frameContext, World &world);

private:
    struct Impl;

    gpu::GpuDevice &mDevice;
    physics::PhysicsSolver &mPhysicsSolver;
    std::unique_ptr<Impl> mImpl;
    bool mInitialized = false;
};

} // namespace cressim::neo::engine

#endif
