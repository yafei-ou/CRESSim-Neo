#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H

#include "common/frame_context.h"
#include "gpu/gpu_device.h"
#include "physics/export.h"
#include "physics/physics_types.h"
#include "physics/physics_world.h"

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

    bool initialize();
    void shutdown();
    bool step(const common::FrameContext& frameContext, PhysicsWorld& world);

private:
    gpu::GpuDevice& mDevice;
    PhysicsSolverDesc mDesc{};
    PhysicsGpuBuffers mBuffers{};
    bool mInitialized = false;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
