#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H

#include "common/frame_context.h"
#include "gpu/gpu_device.h"
#include "physics/export.h"
#include "physics/physics_types.h"
#include "physics/physics_world.h"

#include <memory>

namespace cressim::neo::physics
{

struct PhysicsSolverDesc
{
    bool enableGpuCompute = true;
};

class CRESSIM_NEO_PHYSICS_API PhysicsSolver
{
public:
    explicit PhysicsSolver(gpu::GpuDevice& device, const PhysicsSolverDesc& desc = {});
    ~PhysicsSolver();

    bool initialize();
    void shutdown();
    bool step(const common::FrameContext& frameContext, PhysicsWorld& world);

private:
    struct Impl;

    gpu::GpuDevice& mDevice;
    PhysicsSolverDesc mDesc{};
    PhysicsGpuBuffers mBuffers{};
    std::unique_ptr<Impl> mImpl;
    bool mInitialized = false;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
