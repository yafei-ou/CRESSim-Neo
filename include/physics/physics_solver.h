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

/// @file physics_solver.h
/// @brief GPU-accelerated XPBD physics solver, numerical configuration descriptors, and stepping
/// interfaces.

namespace cressim::neo::physics
{

/// @brief Numerical stabilization parameters for contact and collision resolution.
///
/// Occupies 32 bytes to match its GPU uniform counterpart layout.
struct ContactSolverSettings
{
    float slop = 1.0e-3f; ///< Contact penetration allowance slop.
    float manifoldMergeSlopMultiplier =
        4.0f;                     ///< Multiplier for clustering contact manifold points.
    float softRelaxation = 0.95f; ///< Relaxation coefficient for soft particle contact.
    float softMaxCorrectionPerIteration =
        0.05f; ///< Maximum position correction clamp per iteration for soft contact.
    float restitutionVelocityThreshold =
        0.5f; ///< Minimum relative impact velocity required to trigger restitution.
    float restitutionPenetrationSlopMultiplier =
        2.0f; ///< Slop multiplier above which restitution is suppressed.
    float rigidRestitutionVelocityThreshold =
        0.0f; ///< Minimum velocity threshold for rigid-body restitution.
    float positionFrictionMinDistance =
        1.0e-5f; ///< Minimum separation distance threshold for applying positional friction.
};

/// @brief Numerical stabilization and clamp parameters for rigid-body position and velocity
/// solvers.
///
/// Occupies 32 bytes to match its GPU uniform counterpart layout.
struct RigidSolverSettings
{
    float depenetrationBaumgarte  = 0.25f; ///< Baumgarte stabilization factor for depenetration.
    float depenetrationRelaxation = 0.90f; ///< Relaxation coefficient for rigid depenetration.
    float depenetrationMaxCorrectionPerIteration =
        0.10f; ///< Maximum depenetration position correction per iteration.
    float velocityPenetrationStiffness =
        1.0f; ///< Penetration stiffness coefficient for velocity projection.
    float maxTranslationCorrectionPerIteration =
        0.01f; ///< Maximum translation clamp per iteration.
    float maxRotationCorrectionPerIteration =
        0.10f; ///< Maximum angular rotation correction clamp per iteration.
    float maxLinearVelocityCorrectionPerIteration =
        0.8f; ///< Maximum linear velocity update clamp per iteration.
    float maxAngularVelocityCorrectionPerIteration =
        0.5f; ///< Maximum angular velocity update clamp per iteration.
};

/// @brief Numerical stabilization and relaxation parameters for articulated rigid joints.
///
/// Occupies 32 bytes to match its GPU uniform counterpart layout.
struct JointSolverSettings
{
    float ballRelaxation = 0.95f; ///< Relaxation factor for ball joints.
    float articulatedRelaxation =
        0.7f;               ///< Relaxation factor for hinge, slider, and spherical joints.
    float maxError = 0.05f; ///< Maximum allowable joint position error before clamping.
    float maxTranslationCorrection =
        0.02f; ///< Maximum linear translation correction applied per iteration.
    float maxAngularCorrection = 0.12f;   ///< Maximum angular correction applied per iteration.
    float regularization       = 1.0e-5f; ///< Regularization epsilon for joint constraint matrices.
    float angularRegularization =
        5.0e-5f;               ///< Regularization epsilon for angular constraint matrices.
    float minXpbdDt = 1.0e-5f; ///< Minimum delta-time floor used in XPBD compliance computation.
};

/// @brief Numerical relaxation parameters for soft-body internal constraint solving.
///
/// Occupies 32 bytes to match its GPU uniform counterpart layout.
struct SoftSolverSettings
{
    float internalRelaxation =
        0.2f; ///< Relaxation factor for distance, bending, and tetrahedral volume constraints.
    float reserved0 = 0.0f; ///< Reserved padding.
    float reserved1 = 0.0f; ///< Reserved padding.
    float reserved2 = 0.0f; ///< Reserved padding.
    float reserved3 = 0.0f; ///< Reserved padding.
    float reserved4 = 0.0f; ///< Reserved padding.
    float reserved5 = 0.0f; ///< Reserved padding.
    float reserved6 = 0.0f; ///< Reserved padding.
};

/// @brief Numerical relaxation and boundary scaling parameters for Position-Based Fluids (PBF).
///
/// Occupies 32 bytes to match its GPU uniform counterpart layout.
struct FluidSolverSettings
{
    float constraintRelaxation = 0.7f; ///< Relaxation factor for density constraint solving.
    float positionRelaxation = 1.0f; ///< Position relaxation factor applied after fluid advection.
    float boundaryDensityScale = 0.12f; ///< Boundary density coupling scale against solid bodies.
    float boundaryDeltaScale   = 0.08f; ///< Boundary position delta scale.
    float maxUnderDensityRatio =
        0.0f;               ///< Threshold below rest density for allowing negative pressure.
    float reserved1 = 0.0f; ///< Reserved padding.
    float reserved2 = 0.0f; ///< Reserved padding.
    float reserved3 = 0.0f; ///< Reserved padding.
};

static_assert(sizeof(ContactSolverSettings) == 32u);
static_assert(sizeof(RigidSolverSettings) == 32u);
static_assert(sizeof(JointSolverSettings) == 32u);
static_assert(sizeof(SoftSolverSettings) == 32u);
static_assert(sizeof(FluidSolverSettings) == 32u);

/// @brief Configuration descriptor for initializing the GPU physics solver.
struct PhysicsSolverDesc
{
    Diligent::float3 gravity{0.0f, -9.81f, 0.0f}; ///< Global gravitational acceleration vector.
    std::uint32_t substeps          = 1;  ///< Number of simulation substeps per frame step (>= 1).
    std::uint32_t defaultIterations = 20; ///< Default position-solver iteration count per substep.
    std::uint32_t fluidIterations =
        0; ///< Fluid density constraint iterations (0 uses defaultIterations).
    std::uint32_t softInternalIterations =
        0; ///< Soft-body internal constraint iterations (0 uses defaultIterations).
    std::uint32_t softContactIterations =
        0; ///< Soft-body contact iterations (0 uses defaultIterations).
    std::uint32_t rigidJointIterations =
        0; ///< Rigid joint constraint iterations (0 uses defaultIterations).
    std::uint32_t rigidRigidContactIterations =
        0;                           ///< Rigid-rigid contact iterations (0 uses defaultIterations).
    ContactSolverSettings contact{}; ///< Contact solver numerical tuning parameters.
    RigidSolverSettings rigid{};     ///< Rigid-body solver numerical tuning parameters.
    JointSolverSettings joints{};    ///< Joint constraint solver numerical tuning parameters.
    SoftSolverSettings soft{};       ///< Soft-body solver numerical tuning parameters.
    FluidSolverSettings fluid{};     ///< Fluid solver numerical tuning parameters.
    bool enableBlockingReadback = true; ///< Whether each physics step blocks to copy GPU simulated
                                        ///< state back to CPU host memory.
};

/// @brief Primary GPU-accelerated XPBD physics simulation engine.
///
/// Manages GPU compute passes, scene buffers, broadphase spatial partitioning,
/// narrowphase collision detection, constraint projection, and state synchronization for
/// rigid bodies, articulated joints, soft bodies, elastic strands, and fluids.
class CRESSIM_NEO_PHYSICS_API PhysicsSolver
{
public:
    /// @brief Constructs a physics solver bound to a GPU device.
    /// @param device GPU device reference providing render/compute resources.
    /// @param desc Initial physics solver configuration descriptor.
    explicit PhysicsSolver(gpu::GpuDevice &device, const PhysicsSolverDesc &desc = {});

    /// @brief Destructor.
    ~PhysicsSolver();

    /// @brief Initializes GPU compute pipelines, shader passes, and internal state buffers.
    /// @return True if initialization succeeded. With no physics-capable GPU backend, it succeeds
    /// in headless mode without creating compute pipelines.
    bool initialize();

    /// @brief Releases all allocated GPU pipelines, buffers, and resources.
    void shutdown();

    /// @brief Advances the physics simulation by the time step provided in the frame context.
    /// @param frameContext Current frame index and delta time information.
    /// @param world Physics world containing authored entities, colliders, and constraints.
    /// @return True if simulation step completed successfully, false on execution failure.
    bool step(const common::FrameContext &frameContext, PhysicsWorld &world);

    /// @brief Synchronizes modified authored state from the host PhysicsWorld to GPU buffers.
    /// @param world Physics world containing authored modifications.
    /// @return True if synchronization and upload succeeded; in headless mode this is a no-op
    /// success after initialization.
    bool syncWorldState(PhysicsWorld &world);

    /// @brief Blocks until GPU compute work finishes and checks applicable metadata for overflow.
    ///
    /// Metadata is read back only for broad-phase and particle-neighbor work performed by the
    /// most recent step.
    /// @return False before initialization or when no physics GPU backend is available.
    bool validateGpuMetaBlocking();

    /// @brief Sets the global gravitational acceleration vector.
    /// @param gravity 3D gravity vector (m/s^2).
    void setGravity(const Diligent::float3 &gravity) noexcept;

    /// @brief Retrieves the current non-owning GPU buffer views for rigid, joint, soft, and curve
    /// scenes.
    /// @return PhysicsGpuSceneView containing pointers and descriptor counts.
    PhysicsGpuSceneView gpuSceneView() const noexcept;

    /// @brief Returns the shared export buffer holding particle positions and inverse mass for CUDA
    /// / renderer interop.
    /// @return Pointer to SharedExportBuffer or nullptr if not created.
    const gpu::SharedExportBuffer *softPositionsInvMassSharedBuffer() const noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_SOLVER_H
