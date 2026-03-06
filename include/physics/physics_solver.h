#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H

#include "common/frame_context.h"
#include "gpu/gpu_device.h"
#include "physics/export.h"
#include "physics/physics_types.h"
#include "physics/physics_world.h"

#include <array>
#include <cstdint>
#include <memory>

namespace cressim::neo::physics
{

struct PhysicsSolverDesc
{
    bool enableGpuCompute          = true;
    std::uint32_t substeps         = 1;
    std::uint32_t solverIterations = 20;
    bool enableBlockingReadback    = true;
};

enum class PhysicsSolverStage : std::uint32_t
{
    PredictState = 0u,
    UpdateWorldAabbs,
    BuildBroadPhase,
    GenerateContacts,
    GenerateBroadPhasePairs,
    SolveConstraints,
    UpdateVelocities,
    CommitResults,
    Count,
};

struct PhysicsSolverStageStats
{
    std::array<bool, static_cast<std::size_t>(PhysicsSolverStage::Count)> executed{};
    std::uint32_t dispatchedStages = 0;
    std::uint32_t skippedStages    = 0;
};

class CRESSIM_NEO_PHYSICS_API PhysicsSolver
{
public:
    explicit PhysicsSolver(gpu::GpuDevice& device, const PhysicsSolverDesc& desc = {});
    ~PhysicsSolver();

    bool initialize();
    void shutdown();
    bool step(const common::FrameContext& frameContext, PhysicsWorld& world);
    const PhysicsSolverStageStats& lastStageStats() const noexcept;

private:
    struct Impl;

    gpu::GpuDevice& mDevice;
    PhysicsSolverDesc mDesc{};
    std::unique_ptr<Impl> mImpl;
    bool mInitialized = false;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
