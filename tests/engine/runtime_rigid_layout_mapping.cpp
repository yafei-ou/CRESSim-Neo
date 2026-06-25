#include "common/logger.h"
#include "engine/runtime.h"

#include <algorithm>
#include <cmath>

namespace
{

struct AuthoredRigidEntities
{
    cressim::neo::common::EntityId ground = cressim::neo::common::kInvalidEntityId;
    cressim::neo::common::EntityId box    = cressim::neo::common::kInvalidEntityId;
};

AuthoredRigidEntities authorEnv(cressim::neo::engine::World &world, const std::uint32_t envIndex,
                                const float zOffset)
{
    using namespace cressim::neo;

    AuthoredRigidEntities authored{};

    authored.ground = world.createEntity(envIndex);
    engine::TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, 0.0f, zOffset};
    world.setTransform(authored.ground, groundTransform);

    engine::RigidBodyComponent groundBody{};
    groundBody.bodyType  = physics::RigidBodyType::Static;
    groundBody.simulated = true;
    world.setRigidBody(authored.ground, groundBody);

    engine::ColliderComponent groundCollider{};
    groundCollider.shapeType   = physics::ColliderShapeType::Box;
    groundCollider.shapeParams = {0.75f, 0.1f, 0.75f, 0.0f};
    world.addCollider(authored.ground, groundCollider);

    authored.box = world.createEntity(envIndex);
    engine::TransformComponent boxTransform{};
    boxTransform.worldTransform.position = {0.0f, 0.5f, zOffset};
    world.setTransform(authored.box, boxTransform);

    engine::RigidBodyComponent boxBody{};
    boxBody.bodyType    = physics::RigidBodyType::Kinematic;
    boxBody.inverseMass = 1.0f;
    boxBody.simulated   = true;
    world.setRigidBody(authored.box, boxBody);

    engine::ColliderComponent boxCollider{};
    boxCollider.shapeType   = physics::ColliderShapeType::Box;
    boxCollider.shapeParams = {0.25f, 0.25f, 0.25f, 0.0f};
    world.addCollider(authored.box, boxCollider);

    return authored;
}

bool verifyBaseMapping(const cressim::neo::engine::RigidLayoutMapping &mapping,
                       const std::vector<AuthoredRigidEntities> &authored)
{
    using namespace cressim::neo;

    if (mapping.rigidBodyCount != authored.size() * 2u || mapping.colliderCount != authored.size() * 2u)
    {
        CRESSIM_LOG_ERROR("Unexpected rigid mapping counts.");
        return false;
    }
    if (mapping.bindingGeneration == 0u)
    {
        CRESSIM_LOG_ERROR("Prepared rigid mapping generation should be non-zero after prepare.");
        return false;
    }
    if (mapping.rigidBodyEntityIds.size() != mapping.rigidBodyCount ||
        mapping.rigidBodyIds.size() != mapping.rigidBodyCount ||
        mapping.rigidBodyEnvironmentIndices.size() != mapping.rigidBodyCount ||
        mapping.colliderIds.size() != mapping.colliderCount ||
        mapping.colliderEntityIds.size() != mapping.colliderCount ||
        mapping.colliderOwnerBodyIds.size() != mapping.colliderCount ||
        mapping.colliderOwnerBodyIndices.size() != mapping.colliderCount ||
        mapping.colliderEnvironmentIndices.size() != mapping.colliderCount ||
        mapping.colliderShapeTypes.size() != mapping.colliderCount ||
        mapping.colliderEnabledFlags.size() != mapping.colliderCount ||
        mapping.colliderCollisionLayers.size() != mapping.colliderCount ||
        mapping.colliderCollisionMasks.size() != mapping.colliderCount ||
        mapping.colliderLocalPositions.size() != mapping.colliderCount ||
        mapping.colliderLocalRotations.size() != mapping.colliderCount ||
        mapping.colliderShapeParams.size() != mapping.colliderCount ||
        mapping.bodyColliderOffsets.size() != mapping.rigidBodyCount ||
        mapping.bodyColliderCounts.size() != mapping.rigidBodyCount)
    {
        CRESSIM_LOG_ERROR("Rigid mapping array sizes do not match reported counts.");
        return false;
    }

    for (std::size_t envIndex = 0; envIndex < authored.size(); ++envIndex)
    {
        const std::size_t groundSlot = envIndex * 2u;
        const std::size_t boxSlot    = groundSlot + 1u;
        if (mapping.rigidBodyEntityIds[groundSlot] != authored[envIndex].ground ||
            mapping.rigidBodyEntityIds[boxSlot] != authored[envIndex].box)
        {
            CRESSIM_LOG_ERROR("Rigid body entity slot mapping did not preserve authored order.");
            return false;
        }
        if (mapping.rigidBodyIds[groundSlot] == physics::kInvalidRigidBodyId ||
            mapping.rigidBodyIds[boxSlot] == physics::kInvalidRigidBodyId)
        {
            CRESSIM_LOG_ERROR("Rigid body ids should be populated in prepared mapping.");
            return false;
        }
        if (mapping.rigidBodyEnvironmentIndices[groundSlot] != envIndex ||
            mapping.rigidBodyEnvironmentIndices[boxSlot] != envIndex)
        {
            CRESSIM_LOG_ERROR("Rigid body environment mapping is incorrect.");
            return false;
        }
        if (mapping.bodyColliderCounts[groundSlot] != 1u || mapping.bodyColliderCounts[boxSlot] != 1u)
        {
            CRESSIM_LOG_ERROR("Expected one collider per rigid body in the authored test scene.");
            return false;
        }

        const std::uint32_t groundColliderSlot = mapping.bodyColliderIndices[mapping.bodyColliderOffsets[groundSlot]];
        const std::uint32_t boxColliderSlot = mapping.bodyColliderIndices[mapping.bodyColliderOffsets[boxSlot]];
        if (groundColliderSlot != groundSlot || boxColliderSlot != boxSlot)
        {
            CRESSIM_LOG_ERROR("Body-collider mapping is not aligned with expected authored layout.");
            return false;
        }
        if (mapping.colliderOwnerBodyIndices[groundColliderSlot] != groundSlot ||
            mapping.colliderOwnerBodyIndices[boxColliderSlot] != boxSlot)
        {
            CRESSIM_LOG_ERROR("Collider owner-body indices are incorrect.");
            return false;
        }
        if (mapping.colliderOwnerBodyIds[groundColliderSlot] != mapping.rigidBodyIds[groundSlot] ||
            mapping.colliderOwnerBodyIds[boxColliderSlot] != mapping.rigidBodyIds[boxSlot])
        {
            CRESSIM_LOG_ERROR("Collider owner-body ids are incorrect.");
            return false;
        }
        if (mapping.colliderEntityIds[groundColliderSlot] != authored[envIndex].ground ||
            mapping.colliderEntityIds[boxColliderSlot] != authored[envIndex].box)
        {
            CRESSIM_LOG_ERROR("Collider entity mapping did not preserve authored identity.");
            return false;
        }
        if (mapping.colliderEnvironmentIndices[groundColliderSlot] != envIndex ||
            mapping.colliderEnvironmentIndices[boxColliderSlot] != envIndex)
        {
            CRESSIM_LOG_ERROR("Collider environment mapping is incorrect.");
            return false;
        }
        if (mapping.colliderShapeTypes[groundColliderSlot] !=
                static_cast<std::uint32_t>(physics::ColliderShapeType::Box) ||
            mapping.colliderShapeTypes[boxColliderSlot] !=
                static_cast<std::uint32_t>(physics::ColliderShapeType::Box))
        {
            CRESSIM_LOG_ERROR("Collider shape types are incorrect.");
            return false;
        }
        if (mapping.colliderEnabledFlags[groundColliderSlot] != 1u ||
            mapping.colliderEnabledFlags[boxColliderSlot] != 1u)
        {
            CRESSIM_LOG_ERROR("Collider enabled flags are incorrect.");
            return false;
        }
        if (mapping.colliderCollisionLayers[groundColliderSlot] != 1u ||
            mapping.colliderCollisionLayers[boxColliderSlot] != 1u ||
            mapping.colliderCollisionMasks[groundColliderSlot] != 0xffffffffu ||
            mapping.colliderCollisionMasks[boxColliderSlot] != 0xffffffffu)
        {
            CRESSIM_LOG_ERROR("Collider filter metadata is incorrect.");
            return false;
        }
        if (std::abs(mapping.colliderLocalPositions[groundColliderSlot].x) > 1.0e-6f ||
            std::abs(mapping.colliderLocalPositions[groundColliderSlot].y) > 1.0e-6f ||
            std::abs(mapping.colliderLocalPositions[groundColliderSlot].z) > 1.0e-6f ||
            std::abs(mapping.colliderLocalPositions[boxColliderSlot].x) > 1.0e-6f ||
            std::abs(mapping.colliderLocalPositions[boxColliderSlot].y) > 1.0e-6f ||
            std::abs(mapping.colliderLocalPositions[boxColliderSlot].z) > 1.0e-6f)
        {
            CRESSIM_LOG_ERROR("Collider local positions are incorrect.");
            return false;
        }
        const Diligent::QuaternionF &groundRotation =
            mapping.colliderLocalRotations[groundColliderSlot];
        const Diligent::QuaternionF &boxRotation = mapping.colliderLocalRotations[boxColliderSlot];
        if (std::abs(groundRotation.q.w - 1.0f) > 1.0e-6f ||
            std::abs(boxRotation.q.w - 1.0f) > 1.0e-6f)
        {
            CRESSIM_LOG_ERROR("Collider local rotations are incorrect.");
            return false;
        }
    }

    return true;
}

bool hasResource(const std::vector<cressim::neo::engine::CustomComputeResourceDesc> &resources,
                 const char *key)
{
    return std::any_of(resources.begin(), resources.end(),
                       [key](const cressim::neo::engine::CustomComputeResourceDesc &resource)
                       { return resource.key == key; });
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
        CRESSIM_LOG_WARNING("Skipping runtime rigid layout mapping test because runtime "
                            "initialization failed.");
        return 0;
    }

    auto &world = runtime.getWorld();
    std::vector<AuthoredRigidEntities> authored;
    authored.push_back(authorEnv(world, 0u, 0.0f));
    authored.push_back(authorEnv(world, 1u, 2.0f));

    runtime.prepare();
    engine::RigidLayoutMapping mapping{};
    if (!runtime.tryGetPreparedRigidLayoutMapping(mapping))
    {
        CRESSIM_LOG_ERROR("Runtime prepared rigid layout mapping query failed.");
        runtime.shutdown();
        return 1;
    }
    if (!verifyBaseMapping(mapping, authored))
    {
        runtime.shutdown();
        return 1;
    }

    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload world before querying rigid/collider resources.");
        runtime.shutdown();
        return 1;
    }
    const std::vector<engine::CustomComputeResourceDesc> resources =
        runtime.listCustomComputeResources();
    if (!hasResource(resources, "rigid.body_types") ||
        !hasResource(resources, "rigid.inverse_inertia_local") ||
        !hasResource(resources, "rigid.proxy_particle_contact_materials") ||
        !hasResource(resources, "collider.owner_body_indices") ||
        !hasResource(resources, "collider.broad_phase") ||
        !hasResource(resources, "collider.geometry") ||
        !hasResource(resources, "collider.materials") ||
        !hasResource(resources, "collider.shape_types") ||
        !hasResource(resources, "collider.enabled_flags") ||
        !hasResource(resources, "rigid.body_collider_offsets") ||
        !hasResource(resources, "rigid.body_collider_counts") ||
        !hasResource(resources, "rigid.body_collider_ranges") ||
        !hasResource(resources, "rigid.body_collider_indices"))
    {
        CRESSIM_LOG_ERROR("Expected rigid/collider custom compute resources are missing.");
        runtime.shutdown();
        return 1;
    }

    const std::uint64_t previousGeneration = mapping.bindingGeneration;
    const std::uint32_t previousBodyCount  = mapping.rigidBodyCount;

    const AuthoredRigidEntities addedEnvBody = authorEnv(world, 1u, 4.0f);
    (void)addedEnvBody;
    runtime.prepare();
    engine::RigidLayoutMapping updatedMapping{};
    if (!runtime.tryGetPreparedRigidLayoutMapping(updatedMapping))
    {
        CRESSIM_LOG_ERROR("Updated runtime prepared rigid layout mapping query failed.");
        runtime.shutdown();
        return 1;
    }
    if (updatedMapping.rigidBodyCount != previousBodyCount + 2u ||
        updatedMapping.colliderCount != mapping.colliderCount + 2u)
    {
        CRESSIM_LOG_ERROR("Rigid layout mapping counts did not update after structural authoring.");
        runtime.shutdown();
        return 1;
    }
    if (updatedMapping.bindingGeneration == previousGeneration)
    {
        CRESSIM_LOG_ERROR("Rigid layout binding generation did not change after structural "
                          "authoring.");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    return 0;
}
