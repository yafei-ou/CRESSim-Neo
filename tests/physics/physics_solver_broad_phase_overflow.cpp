#include "common/frame_context.h"
#include "engine/runtime.h"
#include "physics/physics_solver.h"
#include "physics/physics_world.h"
#include "common/logger.h"


namespace
{

cressim::neo::physics::RigidBodyState makeOverlappingRigidBody(cressim::neo::common::EntityId id,
                                                               float xOffset)
{
    using namespace cressim::neo;

    physics::RigidBodyState state{};
    state.entityId = id;
    state.position = {xOffset, 0.0f, 0.0f};
    state.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    state.scale = {1.0f, 1.0f, 1.0f};
    state.linearVelocity = {0.0f, 0.0f, 0.0f};
    state.angularVelocity = {0.0f, 0.0f, 0.0f};
    state.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    state.inverseMass = 1.0f;
    return state;
}

cressim::neo::physics::ColliderState makeSphereCollider(cressim::neo::common::EntityId id)
{
    cressim::neo::physics::ColliderState state{};
    state.entityId = id;
    state.shapeType = cressim::neo::physics::ColliderShapeType::Sphere;
    state.shapeParams = {0.5f, 0.0f, 0.0f, 0.0f};
    return state;
}

} // namespace

int main()
{
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
        return 1;
    }

    gpu::GpuDevice* device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        CRESSIM_LOG_ERROR( "Runtime returned null GPU device.\n");
        runtime.shutdown();
        return 1;
    }

    physics::PhysicsSolver solver(*device);
    if (!solver.initialize())
    {
        CRESSIM_LOG_ERROR( "Physics solver initialization failed.\n");
        runtime.shutdown();
        return 1;
    }

    physics::PhysicsWorld world;
    constexpr std::uint32_t kBodyCount = 128u;
    for (std::uint32_t i = 0u; i < kBodyCount; ++i)
    {
        const common::EntityId entityId = 1000u + i;
        world.upsertRigidBody(
            makeOverlappingRigidBody(entityId, static_cast<float>(i % 4u) * 0.01f));
        world.replaceColliders(entityId, {makeSphereCollider(entityId)});
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    const bool stepSucceeded = solver.step(frame, world);
    if (stepSucceeded)
    {
        CRESSIM_LOG_ERROR( "Broad-phase overflow test unexpectedly succeeded.\n");
        solver.shutdown();
        runtime.shutdown();
        return 1;
    }

    const physics::PhysicsSolverStageStats& stats = solver.lastStageStats();
    if (!stats.executed[static_cast<std::size_t>(physics::PhysicsSolverStage::PredictState)] ||
        !stats.executed[static_cast<std::size_t>(physics::PhysicsSolverStage::UpdateWorldAabbs)])
    {
        CRESSIM_LOG_ERROR( "Expected broad-phase stages did not execute before overflow.\n");
        solver.shutdown();
        runtime.shutdown();
        return 1;
    }

    solver.shutdown();
    runtime.shutdown();
    CRESSIM_LOG_INFO( "Physics broad-phase overflow checks passed.\n");
    return 0;
}
