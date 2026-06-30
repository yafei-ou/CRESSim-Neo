#include "common/logger.h"
#include "engine/runtime.h"

#include <algorithm>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::common::SceneLayoutDesc;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::engine::World;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::physics::BallJointState;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SphericalJointState;

EntityId authorRigidEntity(World &world, const std::uint32_t envIndex,
                           const Diligent::float3 &position)
{
    const EntityId entity = world.createEntity(envIndex);

    TransformComponent transform{};
    transform.worldTransform.position = position;
    world.setTransform(entity, transform);

    RigidBodyComponent body{};
    body.bodyType            = RigidBodyType::Dynamic;
    body.inverseMass         = 1.0f;
    body.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    world.setRigidBody(entity, body);
    return entity;
}

ColliderComponent makeSphereCollider(const float radius)
{
    ColliderComponent collider{};
    collider.shapeType   = ColliderShapeType::Sphere;
    collider.shapeParams = {radius, 0.0f, 0.0f, 0.0f};
    return collider;
}

ColliderComponent makeBoxCollider(const Diligent::float4 &halfExtents)
{
    ColliderComponent collider{};
    collider.shapeType   = ColliderShapeType::Box;
    collider.shapeParams = halfExtents;
    return collider;
}

bool verifyJointFacade(World &world)
{
    const EntityId env0A = authorRigidEntity(world, 0u, {0.0f, 0.0f, 0.0f});
    const EntityId env0B = authorRigidEntity(world, 0u, {0.5f, 0.0f, 0.0f});
    const EntityId env1A = authorRigidEntity(world, 1u, {0.0f, 0.0f, 1.0f});

    BallJointState ball{};
    ball.jointId      = 101u;
    ball.bodyA        = env0A;
    ball.bodyB        = env0B;
    ball.localAnchorA = {0.1f, 0.0f, 0.0f};
    ball.localAnchorB = {-0.1f, 0.0f, 0.0f};
    if (!world.upsertBallJoint(ball))
    {
        CRESSIM_LOG_ERROR("World failed to author a valid ball joint.");
        return false;
    }
    if (world.tryGetBallJoint(ball.jointId) == nullptr)
    {
        CRESSIM_LOG_ERROR("World failed to query the authored ball joint.");
        return false;
    }

    SphericalJointState spherical{};
    spherical.jointId                = 202u;
    spherical.bodyA                  = env0A;
    spherical.bodyB                  = env0B;
    spherical.limitEnabled           = true;
    spherical.swingLimitY            = 0.3f;
    spherical.swingLimitZ            = 0.4f;
    spherical.twistLimitMin          = -0.2f;
    spherical.twistLimitMax          = 0.2f;
    spherical.localAnchorA           = {0.0f, 0.1f, 0.0f};
    spherical.localAnchorB           = {0.0f, -0.1f, 0.0f};
    spherical.driveTargetOrientation = {0.0f, 0.0f, 0.0f, 1.0f};
    if (!world.upsertSphericalJoint(spherical))
    {
        CRESSIM_LOG_ERROR("World failed to author a valid spherical joint.");
        return false;
    }
    if (world.tryGetSphericalJoint(spherical.jointId) == nullptr)
    {
        CRESSIM_LOG_ERROR("World failed to query the authored spherical joint.");
        return false;
    }

    BallJointState crossEnvBall = ball;
    crossEnvBall.jointId        = 102u;
    crossEnvBall.bodyB          = env1A;
    if (world.upsertBallJoint(crossEnvBall))
    {
        CRESSIM_LOG_ERROR("World accepted a cross-environment ball joint.");
        return false;
    }

    SphericalJointState missingBodySpherical = spherical;
    missingBodySpherical.jointId             = 203u;
    missingBodySpherical.bodyB               = world.createEntity(0u);
    if (world.upsertSphericalJoint(missingBodySpherical))
    {
        CRESSIM_LOG_ERROR("World accepted a spherical joint without rigid bodies on both ends.");
        return false;
    }

    if (!world.removeBallJoint(ball.jointId) || world.tryGetBallJoint(ball.jointId) != nullptr)
    {
        CRESSIM_LOG_ERROR("World failed to remove the authored ball joint.");
        return false;
    }
    if (!world.removeSphericalJoint(spherical.jointId) ||
        world.tryGetSphericalJoint(spherical.jointId) != nullptr)
    {
        CRESSIM_LOG_ERROR("World failed to remove the authored spherical joint.");
        return false;
    }

    return true;
}

bool verifyColliderReplacementAuthoring(World &world, EntityId &outBodyEntity)
{
    const EntityId bodyEntity = authorRigidEntity(world, 0u, {0.0f, 1.0f, 0.0f});
    const auto initialHandle = world.addCollider(bodyEntity, makeSphereCollider(0.2f));
    if (!initialHandle.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to author the initial collider.");
        return false;
    }

    std::vector<ColliderComponent> replacement{};
    replacement.push_back(makeBoxCollider({0.3f, 0.1f, 0.1f, 0.0f}));
    replacement.push_back(makeSphereCollider(0.15f));
    if (!world.replaceColliders(bodyEntity, replacement))
    {
        CRESSIM_LOG_ERROR("World failed to replace collider authoring.");
        return false;
    }

    if (world.tryGetCollider(initialHandle).has_value())
    {
        CRESSIM_LOG_ERROR("Old collider handle remained valid after collider replacement.");
        return false;
    }

    const auto &handles = world.colliderHandles(bodyEntity);
    if (handles.size() != 2u)
    {
        CRESSIM_LOG_ERROR("Collider replacement did not rebuild the expected handle list.");
        return false;
    }
    if (!world.tryGetCollider(handles[0]).has_value() || !world.tryGetCollider(handles[1]).has_value())
    {
        CRESSIM_LOG_ERROR("Replaced collider handles could not be queried.");
        return false;
    }

    outBodyEntity = bodyEntity;
    return true;
}

bool verifyPreparedRigidMapping(Runtime &runtime, const EntityId bodyEntity)
{
    World &world = runtime.getWorld();

    runtime.prepare();

    cressim::neo::engine::RigidLayoutMapping mapping{};
    if (!runtime.tryGetPreparedRigidLayoutMapping(mapping))
    {
        CRESSIM_LOG_ERROR("Prepared rigid layout mapping query failed after collider replacement.");
        return false;
    }
    const auto bodyIt = std::find(mapping.rigidBodyEntityIds.begin(), mapping.rigidBodyEntityIds.end(),
                                  bodyEntity);
    if (mapping.colliderCount < 2u || bodyIt == mapping.rigidBodyEntityIds.end())
    {
        CRESSIM_LOG_ERROR("Prepared rigid layout mapping did not include the replaced rigid body.");
        return false;
    }
    const std::size_t bodyIndex = static_cast<std::size_t>(bodyIt - mapping.rigidBodyEntityIds.begin());
    if (bodyIndex >= mapping.bodyColliderCounts.size() || mapping.bodyColliderCounts[bodyIndex] != 2u)
    {
        CRESSIM_LOG_ERROR("Prepared rigid layout mapping did not reflect the collider count.");
        return false;
    }
    if (mapping.bodyColliderIndices.size() < 2u)
    {
        CRESSIM_LOG_ERROR("Prepared rigid layout mapping did not reflect collider replacement.");
        return false;
    }

    return true;
}

} // namespace

int main()
{
    World world;
    SceneLayoutDesc layout{};
    layout.envCount = 2u;
    world.setSceneLayout(layout);

    EntityId ignoredBodyEntity = cressim::neo::common::kInvalidEntityId;
    if (!verifyJointFacade(world) ||
        !verifyColliderReplacementAuthoring(world, ignoredBodyEntity))
    {
        return 1;
    }

    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.sceneLayout = layout;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING("Skipping world rigid completion test because runtime "
                            "initialization failed. World-only checks already passed.");
        return 0;
    }

    World &runtimeWorld = runtime.getWorld();
    EntityId runtimeColliderBody = cressim::neo::common::kInvalidEntityId;
    if (!verifyColliderReplacementAuthoring(runtimeWorld, runtimeColliderBody) ||
        !verifyPreparedRigidMapping(runtime, runtimeColliderBody))
    {
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    return 0;
}
