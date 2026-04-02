#include "physics/physics_world.h"
#include "physics/soft_body_utilities.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo::physics;

    PhysicsWorld world;

    RigidBodyState rigidA{};
    rigidA.entityId = 3001u;
    rigidA.position = {0.0f, 0.0f, 0.0f};
    world.upsertRigidBody(rigidA);
    ColliderState colliderA{};
    colliderA.entityId = 3001u;
    colliderA.environmentIndex = 0u;
    colliderA.shapeType = ColliderShapeType::Sphere;
    colliderA.shapeParams = {0.5f, 0.0f, 0.0f, 0.0f};
    world.replaceColliders(3001u, {colliderA});

    RigidBodyState rigidB{};
    rigidB.entityId = 3002u;
    rigidB.position = {0.0f, 0.0f, 0.0f};
    world.upsertRigidBody(rigidB);
    ColliderState colliderB{};
    colliderB.entityId = 3002u;
    colliderB.environmentIndex = 1u;
    colliderB.shapeType = ColliderShapeType::Sphere;
    colliderB.shapeParams = {0.5f, 0.0f, 0.0f, 0.0f};
    world.replaceColliders(3002u, {colliderB});

    SoftParticleSoAHost softParticles;
    softParticles.positionsInvMass = {{0.5f, 0.0f, 0.0f, 1.0f}, {0.5f, 0.0f, 0.0f, 1.0f}};
    softParticles.previousPositions = {{0.5f, 0.0f, 0.0f, 0.0f}, {0.5f, 0.0f, 0.0f, 0.0f}};
    softParticles.velocities = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    softParticles.radii = {0.2f, 0.2f};
    softParticles.environmentIndices = {0u, 1u};
    softParticles.owningSoftBodyIndices = {0u, 1u};
    softParticles.collisionLayers = {1u, 1u};
    softParticles.collisionMasks = {0xffffffffu, 0xffffffffu};

    const auto candidates =
        buildSoftRigidBroadPhaseCandidatesCpu(softParticles, world.rigidSurfaceParticles(), 0.4f);
    if (candidates.empty())
    {
        CRESSIM_LOG_ERROR("Expected same-environment broadphase candidates.");
        return 1;
    }

    for (const SoftRigidBroadPhaseCandidate &candidate : candidates)
    {
        const std::uint32_t softEnv = softParticles.environmentIndices[candidate.softParticleIndex];
        const RigidBodyState &rigid = world.rigidBodySnapshot()[candidate.rigidBodyIndex];
        const ColliderState &collider =
            world.colliderSnapshot()[world.bodyColliderMapping().colliderOffsets[candidate.rigidBodyIndex]];
        if (softEnv != collider.environmentIndex)
        {
            CRESSIM_LOG_ERROR("Cross-environment soft-rigid candidate was emitted.");
            return 1;
        }
        (void)rigid;
    }

    CRESSIM_LOG_INFO("Soft-body multi-environment isolation checks passed.");
    return 0;
}
