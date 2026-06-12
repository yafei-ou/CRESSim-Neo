#include "physics/physics_world.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo;

    physics::PhysicsWorld world;

    physics::SoftBodyState softBody{};
    softBody.entityId = 8101u;
    softBody.environmentIndex = 1u;
    softBody.supportsSuturing = true;
    softBody.source.kind = physics::SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = {1.0f, 1.0f, 1.0f};
    softBody.source.regularGrid.targetParticleSpacing = 0.5f;
    if (!world.upsertSoftBody(softBody))
    {
        CRESSIM_LOG_ERROR("Failed to author suturing soft body.\n");
        return 1;
    }

    physics::RigidBodyState rigidBody{};
    rigidBody.entityId = 8102u;
    rigidBody.environmentIndex = 1u;
    rigidBody.bodyType = physics::RigidBodyType::Dynamic;
    rigidBody.inverseMass = 0.8f;
    rigidBody.proxyParticleRadius = 0.1f;
    rigidBody.proxyParticleLocalPositions = {
        {-0.4f, 0.0f, 0.0f},
        {0.0f, 0.2f, 0.0f},
        {0.4f, 0.0f, 0.0f},
    };
    rigidBody.suturingEnabled = true;
    rigidBody.needleTipProxyIndex = 2u;
    world.upsertRigidBody(rigidBody);

    const auto &particles = world.particles();
    if (particles.size() < rigidBody.proxyParticleLocalPositions.size())
    {
        CRESSIM_LOG_ERROR("Rigid proxy particles were not emitted.\n");
        return 1;
    }

    const std::uint32_t groupId = static_cast<std::uint32_t>(world.strandCount());
    const auto *authored = world.tryGetRigidBody(rigidBody.entityId);
    if (authored == nullptr || authored->proxyParticleCount != 3u)
    {
        CRESSIM_LOG_ERROR("Rigid body authored proxy count is incorrect.\n");
        return 1;
    }

    const std::uint32_t particleStart = authored->proxyParticleOffset;
    if (particles.ownerTypes[particleStart] !=
            static_cast<std::uint32_t>(physics::ParticleOwnerType::RigidBody) ||
        particles.strandIds[particleStart] != groupId ||
        particles.strandOrders[particleStart + 2u] != 2u ||
        particles.strandRoles[particleStart + 2u] !=
            static_cast<std::uint32_t>(physics::ParticleStrandRole::NeedleTip) ||
        particles.strandRoles[particleStart + 1u] !=
            static_cast<std::uint32_t>(physics::ParticleStrandRole::NeedleBody))
    {
        CRESSIM_LOG_ERROR("Rigid proxy suturing particle metadata is incorrect.\n");
        return 1;
    }

    if (world.reservedSuturingPathHeaderCount() == 0u ||
        world.reservedSuturingPathNodeCount() == 0u)
    {
        CRESSIM_LOG_ERROR("Rigid proxy suturing pair reservation was not created.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Rigid proxy suturing authoring checks passed.\n");
    return 0;
}
