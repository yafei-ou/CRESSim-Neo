#include "physics/physics_world.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo::physics;

    PhysicsWorld world;

    RigidBodyState sphereBody{};
    sphereBody.entityId = 2001u;
    sphereBody.position = {0.0f, 0.0f, 0.0f};
    sphereBody.scale = {1.0f, 1.0f, 1.0f};
    world.upsertRigidBody(sphereBody);
    world.replaceColliders(2001u, {ColliderState{kInvalidColliderId, 2001u, kInvalidRigidBodyId, 0u,
                                                  ColliderShapeType::Sphere,
                                                  {0.5f, 0.0f, 0.0f, 0.0f}}});

    RigidBodyState boxBody{};
    boxBody.entityId = 2002u;
    boxBody.position = {3.0f, 0.0f, 0.0f};
    boxBody.scale = {1.0f, 1.0f, 1.0f};
    world.upsertRigidBody(boxBody);
    world.replaceColliders(2002u, {ColliderState{kInvalidColliderId, 2002u, kInvalidRigidBodyId, 0u,
                                                  ColliderShapeType::Box,
                                                  {0.5f, 0.25f, 0.75f, 0.0f}}});

    RigidBodyState capsuleBody{};
    capsuleBody.entityId = 2003u;
    capsuleBody.position = {6.0f, 0.0f, 0.0f};
    capsuleBody.scale = {1.0f, 1.0f, 1.0f};
    world.upsertRigidBody(capsuleBody);
    world.replaceColliders(2003u, {ColliderState{kInvalidColliderId, 2003u, kInvalidRigidBodyId, 0u,
                                                  ColliderShapeType::Capsule,
                                                  {0.35f, 0.75f, 0.0f, 0.0f}}});

    const auto &samples = world.rigidSurfaceParticles();
    if (samples.size() != 22u)
    {
        CRESSIM_LOG_ERROR("Unexpected rigid surface particle count: ", samples.size());
        return 1;
    }

    std::uint32_t sphereSamples = 0u;
    std::uint32_t boxSamples = 0u;
    std::uint32_t capsuleSamples = 0u;
    const RigidBodyId sphereId = world.tryGetRigidBody(2001u)->rigidBodyId;
    const RigidBodyId boxId = world.tryGetRigidBody(2002u)->rigidBodyId;
    const RigidBodyId capsuleId = world.tryGetRigidBody(2003u)->rigidBodyId;
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        if (samples.owningRigidBodyIds[i] == sphereId)
        {
            ++sphereSamples;
        }
        else if (samples.owningRigidBodyIds[i] == boxId)
        {
            ++boxSamples;
        }
        else if (samples.owningRigidBodyIds[i] == capsuleId)
        {
            ++capsuleSamples;
        }
    }

    if (sphereSamples != 6u || boxSamples != 8u || capsuleSamples != 8u)
    {
        CRESSIM_LOG_ERROR("Unexpected per-shape surface sample distribution: sphere=", sphereSamples,
                          " box=", boxSamples, " capsule=", capsuleSamples);
        return 1;
    }

    CRESSIM_LOG_INFO("Rigid surface particle generation checks passed.");
    return 0;
}
