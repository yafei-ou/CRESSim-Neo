#include "common/frame_context.h"
#include "engine/runtime.h"
#include "physics/physics_solver.h"
#include "physics/physics_world.h"
#include "common/logger.h"

#include <cmath>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::ColliderState;
using cressim::neo::physics::RigidBodyState;

RigidBodyState makeBody(EntityId id, const Diligent::float3& position)
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
    return state;
}

ColliderState makeCollider(EntityId id, ColliderShapeType shape)
{
    ColliderState state{};
    state.entityId = id;
    state.shapeType = shape;

    switch (shape)
    {
    case ColliderShapeType::Sphere:
        state.shapeParams = {0.6f, 0.0f, 0.0f, 0.0f};
        break;
    case ColliderShapeType::Box:
        state.shapeParams = {0.55f, 0.45f, 0.5f, 0.0f};
        break;
    case ColliderShapeType::Capsule:
        state.shapeParams = {0.35f, 0.55f, 0.0f, 0.0f};
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
    world.upsertRigidBody(makeBody(2001u, {-0.20f, 0.0f, 0.0f}));
    world.replaceColliders(2001u, {makeCollider(2001u, ColliderShapeType::Sphere)});
    world.upsertRigidBody(makeBody(2002u, {0.15f, 0.0f, 0.0f}));
    world.replaceColliders(2002u, {makeCollider(2002u, ColliderShapeType::Sphere)});
    world.upsertRigidBody(makeBody(2003u, {0.0f, 0.10f, 0.0f}));
    world.replaceColliders(2003u, {makeCollider(2003u, ColliderShapeType::Box)});
    world.upsertRigidBody(makeBody(2004u, {0.10f, -0.10f, 0.05f}));
    world.replaceColliders(2004u, {makeCollider(2004u, ColliderShapeType::Box)});
    world.upsertRigidBody(makeBody(2005u, {-0.05f, 0.0f, 0.10f}));
    world.replaceColliders(2005u, {makeCollider(2005u, ColliderShapeType::Capsule)});
    world.upsertRigidBody(makeBody(2006u, {0.05f, 0.0f, -0.10f}));
    world.replaceColliders(2006u, {makeCollider(2006u, ColliderShapeType::Capsule)});

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    for (std::uint32_t stepIndex = 0u; stepIndex < 4u; ++stepIndex)
    {
        if (!solver.step(frame, world))
        {
            CRESSIM_LOG_ERROR( "Solver step failed at iteration " , stepIndex , ".\n");
            solver.shutdown();
            runtime.shutdown();
            return 1;
        }

        for (const physics::RigidBodyState& body : world.rigidBodySnapshot())
        {
            if (!isFinite(body))
            {
                CRESSIM_LOG_ERROR( "Non-finite rigid body state detected after iteration "
                          , stepIndex , ".\n");
                solver.shutdown();
                runtime.shutdown();
                return 1;
            }
        }
    }

    solver.shutdown();
    runtime.shutdown();
    CRESSIM_LOG_INFO( "Physics mixed-shape contact checks passed.\n");
    return 0;
}
