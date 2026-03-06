#include "common/frame_context.h"
#include "engine/runtime.h"
#include "physics/physics_solver.h"
#include "physics/physics_world.h"

#include <iostream>

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
    state.colliderShape = physics::ColliderShapeType::Sphere;
    state.colliderParams = {0.5f, 0.0f, 0.0f, 0.0f};
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
        std::cerr << "Runtime initialization failed.\n";
        return 1;
    }

    gpu::GpuDevice* device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        std::cerr << "Runtime returned null GPU device.\n";
        runtime.shutdown();
        return 1;
    }

    physics::PhysicsSolver solver(*device);
    if (!solver.initialize())
    {
        std::cerr << "Physics solver initialization failed.\n";
        runtime.shutdown();
        return 1;
    }

    physics::PhysicsWorld world;
    constexpr std::uint32_t kBodyCount = 128u;
    for (std::uint32_t i = 0u; i < kBodyCount; ++i)
    {
        world.upsertRigidBody(makeOverlappingRigidBody(1000u + i, static_cast<float>(i % 4u) * 0.01f));
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    const bool stepSucceeded = solver.step(frame, world);
    if (stepSucceeded)
    {
        std::cerr << "Broad-phase overflow test unexpectedly succeeded.\n";
        solver.shutdown();
        runtime.shutdown();
        return 1;
    }

    const physics::PhysicsSolverStageStats& stats = solver.lastStageStats();
    if (!stats.executed[static_cast<std::size_t>(physics::PhysicsSolverStage::PredictState)] ||
        !stats.executed[static_cast<std::size_t>(physics::PhysicsSolverStage::UpdateWorldAabbs)])
    {
        std::cerr << "Expected broad-phase stages did not execute before overflow.\n";
        solver.shutdown();
        runtime.shutdown();
        return 1;
    }

    solver.shutdown();
    runtime.shutdown();
    std::cout << "Physics broad-phase overflow checks passed.\n";
    return 0;
}
