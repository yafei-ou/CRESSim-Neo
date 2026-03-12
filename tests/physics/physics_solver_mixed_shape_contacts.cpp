#include "common/frame_context.h"
#include "engine/runtime.h"
#include "physics/physics_solver.h"
#include "physics/physics_world.h"

#include <cmath>
#include <iostream>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyState;

RigidBodyState makeBody(EntityId id, ColliderShapeType shape, const Diligent::float3& position)
{
    RigidBodyState state{};
    state.entityId = id;
    state.position = position;
    state.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    state.scale = {1.0f, 1.0f, 1.0f};
    state.linearVelocity = {0.0f, 0.0f, 0.0f};
    state.angularVelocity = {0.0f, 0.0f, 0.0f};
    state.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    state.inverseMass = 1.0f;
    state.colliderShape = shape;

    switch (shape)
    {
        case ColliderShapeType::Sphere:
            state.colliderParams = {0.6f, 0.0f, 0.0f, 0.0f};
            break;
        case ColliderShapeType::Box:
            state.colliderParams = {0.55f, 0.45f, 0.5f, 0.0f};
            break;
        case ColliderShapeType::Capsule:
            state.colliderParams = {0.35f, 0.55f, 0.0f, 0.0f};
            break;
    }

    return state;
}

bool isFinite(const RigidBodyState& state)
{
    const auto finite3 = [](const Diligent::float3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    };

    return finite3(state.position) && finite3(state.scale) && finite3(state.linearVelocity) &&
           finite3(state.angularVelocity) && finite3(state.inverseInertiaLocal) &&
           std::isfinite(state.inverseMass) && std::isfinite(state.rotation.q.x) &&
           std::isfinite(state.rotation.q.y) && std::isfinite(state.rotation.q.z) &&
           std::isfinite(state.rotation.q.w);
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
    world.upsertRigidBody(makeBody(2001u, ColliderShapeType::Sphere, {-0.20f, 0.0f, 0.0f}));
    world.upsertRigidBody(makeBody(2002u, ColliderShapeType::Sphere, {0.15f, 0.0f, 0.0f}));
    world.upsertRigidBody(makeBody(2003u, ColliderShapeType::Box, {0.0f, 0.10f, 0.0f}));
    world.upsertRigidBody(makeBody(2004u, ColliderShapeType::Box, {0.10f, -0.10f, 0.05f}));
    world.upsertRigidBody(makeBody(2005u, ColliderShapeType::Capsule, {-0.05f, 0.0f, 0.10f}));
    world.upsertRigidBody(makeBody(2006u, ColliderShapeType::Capsule, {0.05f, 0.0f, -0.10f}));

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    for (std::uint32_t stepIndex = 0u; stepIndex < 4u; ++stepIndex)
    {
        if (!solver.step(frame, world))
        {
            std::cerr << "Solver step failed at iteration " << stepIndex << ".\n";
            solver.shutdown();
            runtime.shutdown();
            return 1;
        }

        const physics::PhysicsSolverStageStats& stats = solver.lastStageStats();
        if (!stats.executed[static_cast<std::size_t>(physics::PhysicsSolverStage::GenerateBroadPhasePairs)] ||
            !stats.executed[static_cast<std::size_t>(physics::PhysicsSolverStage::GenerateContacts)])
        {
            std::cerr << "Expected broad-phase pair/contact stages were skipped at iteration "
                      << stepIndex << ".\n";
            solver.shutdown();
            runtime.shutdown();
            return 1;
        }

        for (const physics::RigidBodyState& body : world.rigidBodySnapshot())
        {
            if (!isFinite(body))
            {
                std::cerr << "Non-finite rigid body state detected after iteration "
                          << stepIndex << ".\n";
                solver.shutdown();
                runtime.shutdown();
                return 1;
            }
        }
    }

    solver.shutdown();
    runtime.shutdown();
    std::cout << "Physics mixed-shape contact checks passed.\n";
    return 0;
}
