#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H

#include "common/frame_context.h"
#include "gpu/gpu_device.h"
#include "physics/export.h"
#include "physics/physics_gpu_scene_view.h"
#include "physics/physics_world.h"

#include <cstdint>
#include <memory>

namespace cressim::neo::physics
{

struct PhysicsSolverDesc
{
    std::uint32_t substeps                    = 1;
    std::uint32_t defaultIterations           = 20;
    std::uint32_t fluidIterations             = 0;
    std::uint32_t softInternalIterations      = 0;
    std::uint32_t softContactIterations       = 0;
    std::uint32_t rigidJointIterations        = 0;
    std::uint32_t rigidRigidContactIterations = 0;
    float fluidBoundaryDensityScale           = 1.0f;
    bool enableBlockingReadback               = true;
};

class CRESSIM_NEO_PHYSICS_API PhysicsSolver
{
public:
    explicit PhysicsSolver(gpu::GpuDevice &device, const PhysicsSolverDesc &desc = {});
    ~PhysicsSolver();

    bool initialize();
    void shutdown();
    bool step(const common::FrameContext &frameContext, PhysicsWorld &world);
    bool validateGpuMetaBlocking();
    PhysicsGpuSceneView gpuSceneView() const noexcept;

private:
    struct Impl;

    gpu::GpuDevice &mDevice;
    PhysicsSolverDesc mDesc{};
    std::unique_ptr<Impl> mImpl;
    bool mInitialized = false;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
