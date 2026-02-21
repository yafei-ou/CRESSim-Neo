#include "physics/physics_solver.h"

namespace cressim::neo::physics
{

PhysicsSolver::PhysicsSolver(gpu::GpuDevice& device, const PhysicsSolverDesc& desc)
    : mDevice(device), mDesc(desc)
{
}

bool PhysicsSolver::initialize()
{
    shutdown();
    // TODO: create compute PSOs and persistent GPU SoA buffers for full PBD stages.
    mInitialized = true;
    return true;
}

void PhysicsSolver::shutdown()
{
    mBuffers     = {};
    mInitialized = false;
}

bool PhysicsSolver::step(const common::FrameContext& frameContext, PhysicsWorld& world)
{
    if (!mInitialized)
    {
        return false;
    }

    (void)mDevice;
    // TODO: replace CPU placeholder integration with GPU compute dispatch using
    // assets/shaders/physics/physics_placeholder_integrate.cs.hlsl and readback.
    for (RigidBodyState& rb : world.rigidBodies())
    {
        rb.position += rb.linearVelocity * frameContext.deltaSeconds;
    }

    return true;
}

} // namespace cressim::neo::physics
