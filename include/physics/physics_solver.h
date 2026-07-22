#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H

#include "common/frame_context.h"
#include "gpu/gpu_device.h"
#include "gpu/shared_export_buffer.h"
#include "physics/export.h"
#include "physics/physics_gpu_scene_view.h"
#include "physics/physics_world.h"

#include <cstdint>
#include <memory>

namespace cressim::neo::physics
{

// Solver-wide numerical policy. Material properties remain scene data.
// Each group occupies 32 bytes so its GPU counterpart has an unambiguous layout.
struct ContactSolverSettings
{
    float slop                                 = 1.0e-3f;
    float manifoldMergeSlopMultiplier          = 4.0f;
    float softRelaxation                       = 0.95f;
    float softMaxCorrectionPerIteration        = 0.05f;
    float restitutionVelocityThreshold         = 0.5f;
    float restitutionPenetrationSlopMultiplier = 2.0f;
    float rigidRestitutionVelocityThreshold    = 0.0f;
    float positionFrictionMinDistance          = 1.0e-5f;
};

struct RigidSolverSettings
{
    float depenetrationBaumgarte                   = 0.25f;
    float depenetrationRelaxation                  = 0.90f;
    float depenetrationMaxCorrectionPerIteration   = 0.10f;
    float velocityPenetrationStiffness             = 1.0f;
    float maxTranslationCorrectionPerIteration     = 0.01f;
    float maxRotationCorrectionPerIteration        = 0.10f;
    float maxLinearVelocityCorrectionPerIteration  = 0.8f;
    float maxAngularVelocityCorrectionPerIteration = 0.5f;
};

struct JointSolverSettings
{
    float ballRelaxation           = 0.95f;
    float articulatedRelaxation    = 0.7f;
    float maxError                 = 0.05f;
    float maxTranslationCorrection = 0.02f;
    float maxAngularCorrection     = 0.12f;
    float regularization           = 1.0e-5f;
    float angularRegularization    = 5.0e-5f;
    float minXpbdDt                = 1.0e-5f;
};

struct SoftSolverSettings
{
    float internalRelaxation = 0.2f;
    float reserved0          = 0.0f;
    float reserved1          = 0.0f;
    float reserved2          = 0.0f;
    float reserved3          = 0.0f;
    float reserved4          = 0.0f;
    float reserved5          = 0.0f;
    float reserved6          = 0.0f;
};

struct FluidSolverSettings
{
    float constraintRelaxation = 0.7f;
    float positionRelaxation   = 1.0f;
    float boundaryDensityScale = 0.12f;
    float boundaryDeltaScale   = 0.08f;
    float maxUnderDensityRatio = 0.0f;
    float reserved1            = 0.0f;
    float reserved2            = 0.0f;
    float reserved3            = 0.0f;
};

static_assert(sizeof(ContactSolverSettings) == 32u);
static_assert(sizeof(RigidSolverSettings) == 32u);
static_assert(sizeof(JointSolverSettings) == 32u);
static_assert(sizeof(SoftSolverSettings) == 32u);
static_assert(sizeof(FluidSolverSettings) == 32u);

struct PhysicsSolverDesc
{
    Diligent::float3 gravity{0.0f, -9.81f, 0.0f};
    std::uint32_t substeps                    = 1;
    std::uint32_t defaultIterations           = 20;
    std::uint32_t fluidIterations             = 0;
    std::uint32_t softInternalIterations      = 0;
    std::uint32_t softContactIterations       = 0;
    std::uint32_t rigidJointIterations        = 0;
    std::uint32_t rigidRigidContactIterations = 0;
    ContactSolverSettings contact{};
    RigidSolverSettings rigid{};
    JointSolverSettings joints{};
    SoftSolverSettings soft{};
    FluidSolverSettings fluid{};
    bool enableBlockingReadback = true;
};

class CRESSIM_NEO_PHYSICS_API PhysicsSolver
{
public:
    explicit PhysicsSolver(gpu::GpuDevice &device, const PhysicsSolverDesc &desc = {});
    ~PhysicsSolver();

    bool initialize();
    void shutdown();
    bool step(const common::FrameContext &frameContext, PhysicsWorld &world);
    bool syncWorldState(PhysicsWorld &world);
    bool validateGpuMetaBlocking();
    void setGravity(const Diligent::float3 &gravity) noexcept;
    PhysicsGpuSceneView gpuSceneView() const noexcept;
    const gpu::SharedExportBuffer *softPositionsInvMassSharedBuffer() const noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
