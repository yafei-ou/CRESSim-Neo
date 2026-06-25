#include "common/logger.h"
#include "engine/runtime.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace
{

struct AuthoredConstraintEnv
{
    cressim::neo::common::EntityId base   = cressim::neo::common::kInvalidEntityId;
    cressim::neo::common::EntityId target = cressim::neo::common::kInvalidEntityId;
    cressim::neo::common::EntityId strand = cressim::neo::common::kInvalidEntityId;
    cressim::neo::physics::RigidParticleAttachmentConstraintId attachment =
        cressim::neo::physics::kInvalidRigidParticleAttachmentConstraintId;
    cressim::neo::physics::RigidDistanceConstraintId distance =
        cressim::neo::physics::kInvalidRigidDistanceConstraintId;
    cressim::neo::physics::RoutedCableConstraintId cable =
        cressim::neo::physics::kInvalidRoutedCableConstraintId;
};

AuthoredConstraintEnv authorEnv(cressim::neo::engine::World &world, const std::uint32_t envIndex,
                                const float zOffset)
{
    using namespace cressim::neo;

    AuthoredConstraintEnv authored{};

    authored.base = world.createEntity(envIndex);
    engine::TransformComponent baseTransform{};
    baseTransform.worldTransform.position = {0.0f, 0.5f, zOffset};
    world.setTransform(authored.base, baseTransform);

    engine::RigidBodyComponent baseBody{};
    baseBody.bodyType    = physics::RigidBodyType::Static;
    baseBody.inverseMass = 0.0f;
    baseBody.simulated   = true;
    world.setRigidBody(authored.base, baseBody);

    engine::ColliderComponent baseCollider{};
    baseCollider.shapeType   = physics::ColliderShapeType::Box;
    baseCollider.shapeParams = {0.15f, 0.15f, 0.15f, 0.0f};
    world.addCollider(authored.base, baseCollider);

    authored.target = world.createEntity(envIndex);
    engine::TransformComponent targetTransform{};
    targetTransform.worldTransform.position = {0.5f, 0.5f, zOffset};
    world.setTransform(authored.target, targetTransform);

    engine::RigidBodyComponent targetBody{};
    targetBody.bodyType            = physics::RigidBodyType::Dynamic;
    targetBody.inverseMass         = 1.0f;
    targetBody.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    targetBody.simulated           = true;
    world.setRigidBody(authored.target, targetBody);

    engine::ColliderComponent targetCollider{};
    targetCollider.shapeType   = physics::ColliderShapeType::Box;
    targetCollider.shapeParams = {0.12f, 0.12f, 0.12f, 0.0f};
    world.addCollider(authored.target, targetCollider);

    authored.strand = world.createEntity(envIndex);
    engine::StrandComponent strand{};
    strand.restPositions = {{0.0f, 0.9f, zOffset}, {0.0f, 1.1f, zOffset}};
    strand.staticParticleIndices = {0u};
    strand.particleRadius        = 0.02f;
    strand.particleMass          = 0.1f;
    strand.simulated             = true;
    if (!world.setStrand(authored.strand, strand))
    {
        return authored;
    }

    physics::AuthoredRigidParticleAttachmentConstraintState attachment{};
    attachment.constraintId      = 1000u + envIndex;
    attachment.rigidBodyEntityId = authored.target;
    attachment.particle.entityId = authored.strand;
    attachment.particle.type     = physics::AuthoredParticleReferenceType::StrandParticle;
    attachment.particle.localParticleIndex = 1u;
    attachment.enabled                    = true;
    physics::AuthoredRigidParticleAttachmentConstraintState authoredAttachment{};
    if (!world.upsertRigidParticleAttachmentConstraint(attachment, &authoredAttachment))
    {
        return authored;
    }
    authored.attachment = authoredAttachment.constraintId;

    physics::AuthoredRigidDistanceConstraintState distance{};
    distance.constraintId = 2000u + envIndex;
    distance.entityA      = authored.base;
    distance.entityB      = authored.target;
    distance.enabled      = true;
    physics::AuthoredRigidDistanceConstraintState authoredDistance{};
    if (!world.upsertRigidDistanceConstraint(distance, &authoredDistance))
    {
        return authored;
    }
    authored.distance = authoredDistance.constraintId;

    physics::AuthoredRoutedCableConstraintState cable{};
    cable.constraintId = 3000u + envIndex;
    cable.routePoints.push_back({authored.base, {0.0f, 0.0f, 0.0f}});
    cable.routePoints.push_back({authored.target, {0.0f, 0.05f, 0.0f}});
    cable.enabled = true;
    physics::AuthoredRoutedCableConstraintState authoredCable{};
    if (!world.upsertRoutedCableConstraint(cable, &authoredCable))
    {
        return authored;
    }
    authored.cable = authoredCable.constraintId;

    return authored;
}

bool hasResource(const std::vector<cressim::neo::engine::CustomComputeResourceDesc> &resources,
                 const char *key)
{
    return std::any_of(resources.begin(), resources.end(),
                       [key](const cressim::neo::engine::CustomComputeResourceDesc &resource)
                       { return resource.key == key; });
}

bool verifyConstraintMapping(const cressim::neo::engine::RigidLayoutMapping &rigidMapping,
                             const cressim::neo::engine::ConstraintLayoutMapping &mapping,
                             const std::vector<AuthoredConstraintEnv> &authored)
{
    using namespace cressim::neo;

    if (mapping.bindingGeneration == 0u)
    {
        CRESSIM_LOG_ERROR("Constraint layout binding generation should be non-zero after prepare.");
        return false;
    }
    if (mapping.rigidParticleAttachments.count != authored.size() ||
        mapping.rigidDistanceConstraints.count != authored.size() ||
        mapping.routedCables.count != authored.size())
    {
        CRESSIM_LOG_ERROR("Unexpected prepared constraint counts.");
        return false;
    }

    std::unordered_map<common::EntityId, std::uint32_t> rigidSlotByEntity;
    for (std::uint32_t i = 0; i < rigidMapping.rigidBodyCount; ++i)
    {
        rigidSlotByEntity.emplace(rigidMapping.rigidBodyEntityIds[i], i);
    }

    for (std::size_t envIndex = 0; envIndex < authored.size(); ++envIndex)
    {
        const std::uint32_t expectedBaseSlot = rigidSlotByEntity.at(authored[envIndex].base);
        const std::uint32_t expectedTargetSlot = rigidSlotByEntity.at(authored[envIndex].target);

        if (mapping.rigidParticleAttachments.constraintIds[envIndex] != authored[envIndex].attachment ||
            mapping.rigidDistanceConstraints.constraintIds[envIndex] != authored[envIndex].distance ||
            mapping.routedCables.constraintIds[envIndex] != authored[envIndex].cable)
        {
            CRESSIM_LOG_ERROR("Prepared constraint ids did not preserve authored identity.");
            return false;
        }

        if (mapping.rigidParticleAttachments.rigidBodyIndices[envIndex] != expectedTargetSlot ||
            mapping.rigidParticleAttachments.rigidBodyIds[envIndex] !=
                rigidMapping.rigidBodyIds[expectedTargetSlot] ||
            mapping.rigidParticleAttachments.environmentIndices[envIndex] != envIndex ||
            mapping.rigidParticleAttachments.particleEntityIds[envIndex] != authored[envIndex].strand ||
            mapping.rigidParticleAttachments.particleLocalIndices[envIndex] != 1u ||
            mapping.rigidParticleAttachments.enabledFlags[envIndex] != 1u)
        {
            CRESSIM_LOG_ERROR("Prepared rigid-particle attachment mapping is incorrect.");
            return false;
        }

        if (mapping.rigidDistanceConstraints.rigidBodyIndicesA[envIndex] != expectedBaseSlot ||
            mapping.rigidDistanceConstraints.rigidBodyIndicesB[envIndex] != expectedTargetSlot ||
            mapping.rigidDistanceConstraints.rigidBodyIdsA[envIndex] !=
                rigidMapping.rigidBodyIds[expectedBaseSlot] ||
            mapping.rigidDistanceConstraints.rigidBodyIdsB[envIndex] !=
                rigidMapping.rigidBodyIds[expectedTargetSlot] ||
            mapping.rigidDistanceConstraints.environmentIndices[envIndex] != envIndex ||
            mapping.rigidDistanceConstraints.enabledFlags[envIndex] != 1u)
        {
            CRESSIM_LOG_ERROR("Prepared rigid-distance mapping is incorrect.");
            return false;
        }

        if (mapping.routedCables.environmentIndices[envIndex] != envIndex ||
            mapping.routedCables.routePointCounts[envIndex] != 2u ||
            mapping.routedCables.enabledFlags[envIndex] != 1u)
        {
            CRESSIM_LOG_ERROR("Prepared routed cable mapping header is incorrect.");
            return false;
        }

        const std::uint32_t routePointOffset = mapping.routedCables.routePointOffsets[envIndex];
        if (mapping.routedCables.routePointRigidBodyIndices[routePointOffset] != expectedBaseSlot ||
            mapping.routedCables.routePointRigidBodyIndices[routePointOffset + 1u] != expectedTargetSlot ||
            mapping.routedCables.routePointRigidBodyIds[routePointOffset] !=
                rigidMapping.rigidBodyIds[expectedBaseSlot] ||
            mapping.routedCables.routePointRigidBodyIds[routePointOffset + 1u] !=
                rigidMapping.rigidBodyIds[expectedTargetSlot])
        {
            CRESSIM_LOG_ERROR("Prepared routed cable route-point body mapping is incorrect.");
            return false;
        }
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
    config.sceneLayout.envCount           = 2u;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING("Skipping runtime constraint layout mapping test because runtime "
                            "initialization failed.");
        return 0;
    }

    auto &world = runtime.getWorld();
    std::vector<AuthoredConstraintEnv> authored;
    authored.push_back(authorEnv(world, 0u, 0.0f));
    authored.push_back(authorEnv(world, 1u, 2.0f));
    for (const AuthoredConstraintEnv &env : authored)
    {
        if (env.attachment == physics::kInvalidRigidParticleAttachmentConstraintId ||
            env.distance == physics::kInvalidRigidDistanceConstraintId ||
            env.cable == physics::kInvalidRoutedCableConstraintId)
        {
            CRESSIM_LOG_ERROR("Failed to author constraint test scene.");
            runtime.shutdown();
            return 1;
        }
    }

    runtime.prepare();

    engine::RigidLayoutMapping rigidMapping{};
    if (!runtime.tryGetPreparedRigidLayoutMapping(rigidMapping))
    {
        CRESSIM_LOG_ERROR("Prepared rigid layout mapping query failed.");
        runtime.shutdown();
        return 1;
    }

    engine::ConstraintLayoutMapping mapping{};
    if (!runtime.tryGetPreparedConstraintLayoutMapping(mapping))
    {
        CRESSIM_LOG_ERROR("Prepared constraint layout mapping query failed.");
        runtime.shutdown();
        return 1;
    }
    if (!verifyConstraintMapping(rigidMapping, mapping, authored))
    {
        runtime.shutdown();
        return 1;
    }

    const std::uint64_t previousGeneration = mapping.bindingGeneration;

    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload world before querying constraint resources.");
        runtime.shutdown();
        return 1;
    }
    const std::vector<engine::CustomComputeResourceDesc> resources =
        runtime.listCustomComputeResources();
    if (!hasResource(resources, "constraint.rigid_particle_attachments") ||
        !hasResource(resources, "constraint.rigid_distance_constraints") ||
        !hasResource(resources, "constraint.routed_cable_descriptors") ||
        !hasResource(resources, "constraint.routed_cable_route_points") ||
        !hasResource(resources, "constraint.routed_cable_debug_segments"))
    {
        CRESSIM_LOG_ERROR("Expected constraint custom compute resources are missing.");
        runtime.shutdown();
        return 1;
    }

    physics::AuthoredRigidDistanceConstraintState extraDistance{};
    extraDistance.constraintId = 9999u;
    extraDistance.entityA      = authored[0].base;
    extraDistance.entityB      = authored[0].target;
    extraDistance.enabled      = true;
    if (!world.upsertRigidDistanceConstraint(extraDistance))
    {
        CRESSIM_LOG_ERROR("Failed to author additional rigid distance constraint.");
        runtime.shutdown();
        return 1;
    }

    runtime.prepare();
    engine::ConstraintLayoutMapping updatedMapping{};
    if (!runtime.tryGetPreparedConstraintLayoutMapping(updatedMapping))
    {
        CRESSIM_LOG_ERROR("Updated prepared constraint layout mapping query failed.");
        runtime.shutdown();
        return 1;
    }
    if (updatedMapping.rigidDistanceConstraints.count !=
            mapping.rigidDistanceConstraints.count + 1u ||
        updatedMapping.bindingGeneration == previousGeneration)
    {
        CRESSIM_LOG_ERROR("Prepared constraint layout mapping did not update after structural authoring.");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    return 0;
}
