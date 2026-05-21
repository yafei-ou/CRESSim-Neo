#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/runtime.h"
#include "physics/physics_solver.h"
#include "physics/physics_world.h"

#include <cmath>
#include <vector>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::ColliderState;
using cressim::neo::physics::PhysicsSolver;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::RigidBodyState;
using cressim::neo::physics::RigidBodyType;

RigidBodyState makeBody(EntityId id, const Diligent::float3 &position, const Diligent::float3 &velocity)
{
    RigidBodyState state{};
    state.entityId = id;
    state.position = position;
    state.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    state.scale = {1.0f, 1.0f, 1.0f};
    state.linearVelocity = velocity;
    state.angularVelocity = {0.0f, 0.0f, 0.0f};
    state.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    state.inverseMass = 1.0f;
    state.bodyType = RigidBodyType::Dynamic;
    return state;
}

RigidBodyState makeStaticBody(EntityId id, const Diligent::float3 &position)
{
    RigidBodyState state = makeBody(id, position, {0.0f, 0.0f, 0.0f});
    state.bodyType = RigidBodyType::Static;
    state.inverseMass = 0.0f;
    state.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    return state;
}

ColliderState makeSphereCollider(EntityId id, float radius, const Diligent::float3 &localPosition,
                                 float restitution)
{
    ColliderState state{};
    state.entityId = id;
    state.shapeType = ColliderShapeType::Sphere;
    state.shapeParams = {radius, 0.0f, 0.0f, 0.0f};
    state.localPosition = localPosition;
    state.restitution = restitution;
    return state;
}

ColliderState makeWallCollider(EntityId id, const Diligent::float3 &halfExtents, float restitution)
{
    ColliderState state{};
    state.entityId = id;
    state.shapeType = ColliderShapeType::Box;
    state.shapeParams = {halfExtents.x, halfExtents.y, halfExtents.z, 0.0f};
    state.restitution = restitution;
    return state;
}

bool isFinite(const RigidBodyState &state)
{
    const auto finite3 = [](const Diligent::float3 &value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    };

    return finite3(state.position) && finite3(state.linearVelocity) &&
           finite3(state.angularVelocity) && std::isfinite(state.rotation.q.x) &&
           std::isfinite(state.rotation.q.y) && std::isfinite(state.rotation.q.z) &&
           std::isfinite(state.rotation.q.w);
}

bool runScenario(PhysicsSolver &solver, PhysicsWorld &world, EntityId bodyEntity,
                 const char *scenarioName)
{
    cressim::neo::common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    for (std::uint32_t stepIndex = 0u; stepIndex < 40u; ++stepIndex)
    {
        if (!solver.step(frame, world))
        {
            CRESSIM_LOG_ERROR(scenarioName, ": solver step failed at iteration ", stepIndex,
                              ".\n");
            return false;
        }

        const RigidBodyState *body = world.tryGetRigidBody(bodyEntity);
        if (body == nullptr || !isFinite(*body))
        {
            CRESSIM_LOG_ERROR(scenarioName, ": non-finite body state detected.\n");
            return false;
        }

        frame.frameIndex++;
        frame.timeSeconds += frame.deltaSeconds;
    }

    const RigidBodyState *body = world.tryGetRigidBody(bodyEntity);
    if (body == nullptr)
    {
        CRESSIM_LOG_ERROR(scenarioName, ": moving body missing after simulation.\n");
        return false;
    }

    if (body->linearVelocity.x >= -0.1f)
    {
        CRESSIM_LOG_ERROR(scenarioName, ": expected bounce with negative x velocity, got ",
                          body->linearVelocity.x, ".\n");
        return false;
    }

    return true;
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
        CRESSIM_LOG_ERROR("Runtime initialization failed.\n");
        return 1;
    }

    gpu::GpuDevice *device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        CRESSIM_LOG_ERROR("Runtime returned null GPU device.\n");
        runtime.shutdown();
        return 1;
    }

    PhysicsSolver solver(*device);
    if (!solver.initialize())
    {
        CRESSIM_LOG_ERROR("Physics solver initialization failed.\n");
        runtime.shutdown();
        return 1;
    }

    {
        PhysicsWorld world;
        world.upsertRigidBody(makeBody(3001u, {-1.7f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f}));
        world.replaceColliders(3001u, {makeSphereCollider(3001u, 0.35f, {0.0f, 0.0f, 0.0f}, 0.8f)});
        world.upsertRigidBody(makeStaticBody(3002u, {0.0f, 0.0f, 0.0f}));
        world.replaceColliders(3002u, {makeWallCollider(3002u, {0.5f, 1.5f, 1.5f}, 0.8f)});

        if (!runScenario(solver, world, 3001u, "single collider restitution"))
        {
            solver.shutdown();
            runtime.shutdown();
            return 1;
        }
    }

    {
        PhysicsWorld world;
        world.upsertRigidBody(makeBody(3011u, {-1.8f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f}));
        std::vector<ColliderState> colliders;
        colliders.push_back(makeSphereCollider(3011u, 0.35f, {-0.45f, 0.0f, 0.0f}, 0.8f));
        colliders.push_back(makeSphereCollider(3012u, 0.35f, {0.45f, 0.0f, 0.0f}, 0.8f));
        world.replaceColliders(3011u, colliders);
        world.upsertRigidBody(makeStaticBody(3013u, {0.0f, 0.0f, 0.0f}));
        world.replaceColliders(3013u, {makeWallCollider(3013u, {0.5f, 1.5f, 1.5f}, 0.8f)});

        if (!runScenario(solver, world, 3011u, "compound restitution"))
        {
            solver.shutdown();
            runtime.shutdown();
            return 1;
        }
    }

    solver.shutdown();
    runtime.shutdown();
    CRESSIM_LOG_INFO("Physics rigid restitution checks passed.\n");
    return 0;
}
