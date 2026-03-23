#include "engine/components.h"
#include "engine/world.h"
#include "graphics/render_resource_manager.h"
#include "common/logger.h"

#include <cstdint>
#include <cmath>

int main()
{
    using namespace cressim::neo;

    engine::World world;
    gpu::GpuSceneLayoutDesc layout{};
    layout.envCount = 2u;
    world.setSceneLayout(layout);

    const common::EntityId entity = world.createEntity();
    const common::EntityId cameraEntity = world.createEntity();
    engine::TransformComponent transform{};
    transform.worldTransform.position = {1.0f, 2.0f, 3.0f};
    transform.worldTransform.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    world.setTransform(entity, transform);

    engine::TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 1.0f, -4.0f};
    world.setTransform(cameraEntity, cameraTransform);
    engine::CameraComponent camera{};
    camera.outputWidth = 1600u;
    camera.outputHeight = 900u;
    camera.viewport = {0.5f, 0.0f, 0.5f, 1.0f};
    world.setCamera(cameraEntity, camera);

    engine::RigidBodyComponent rigidBody{};
    rigidBody.simulated = true;
    rigidBody.bodyType = physics::RigidBodyType::Kinematic;
    rigidBody.inverseMass = 0.5f;
    rigidBody.linearVelocity = {2.0f, 0.0f, -1.0f};
    rigidBody.angularVelocity = {0.0f, 3.0f, 0.0f};
    rigidBody.kinematicTargetPosition = {7.0f, 8.0f, 9.0f};
    rigidBody.kinematicTargetRotation = {0.0f, 0.0f, 0.2588190f, 0.9659258f};
    rigidBody.kinematicTargetEnabled = true;
    world.setRigidBody(entity, rigidBody);
    engine::ColliderComponent collider{};
    collider.shapeType = physics::ColliderShapeType::Capsule;
    collider.shapeParams = {0.7f, 1.4f, 0.0f, 0.0f};
    collider.collisionLayer = 1u << 2u;
    collider.collisionMask = (1u << 0u) | (1u << 2u);
    world.addCollider(entity, collider);
    if (!world.setEntityEnvironment(entity, 1u))
    {
        CRESSIM_LOG_ERROR( "Failed to update entity environment.\n");
        return 1;
    }

    const physics::RigidBodyState* state = world.physicsWorld().tryGetRigidBody(entity);
    if (state == nullptr)
    {
        CRESSIM_LOG_ERROR( "Rigid body missing in physics world.\n");
        return 1;
    }
    if (world.physicsWorld().colliderCount() != 1u)
    {
        CRESSIM_LOG_ERROR( "Collider missing in physics world.\n");
        return 1;
    }
    const physics::ColliderState& colliderState = world.physicsWorld().colliderSnapshot().front();
    if (state->bodyType != rigidBody.bodyType ||
        state->angularVelocity.y != rigidBody.angularVelocity.y ||
        static_cast<std::uint32_t>(colliderState.shapeType) !=
            static_cast<std::uint32_t>(physics::ColliderShapeType::Capsule) ||
        colliderState.shapeParams.x != collider.shapeParams.x ||
        colliderState.environmentIndex != 1u ||
        colliderState.collisionLayer != collider.collisionLayer ||
        colliderState.collisionMask != collider.collisionMask ||
        state->kinematicTargetPosition.x != rigidBody.kinematicTargetPosition.x ||
        state->kinematicTargetRotation.q.z != rigidBody.kinematicTargetRotation.q.z ||
        !state->kinematicTargetEnabled)
    {
        CRESSIM_LOG_ERROR( "World->physics sync did not preserve new rigid fields.\n");
        return 1;
    }

    if (!world.setEntityEnvironment(entity, 0u))
    {
        CRESSIM_LOG_ERROR( "Failed to restore entity environment.\n");
        return 1;
    }
    const physics::ColliderState& updatedColliderState = world.physicsWorld().colliderSnapshot().front();
    if (updatedColliderState.environmentIndex != 0u)
    {
        CRESSIM_LOG_ERROR( "Entity environment change did not propagate to collider state.\n");
        return 1;
    }

    graphics::RenderResourceManager resources;
    world.ensureRenderStateUpToDate(resources);

    const auto& cameraInputs = world.cameraInputs();
    bool foundCamera = false;
    for (const gpu::GpuCameraInput& input : cameraInputs)
    {
        if (input.active == 0u)
        {
            continue;
        }
        foundCamera = true;
        const float expectedViewportWidth = 0.5f;
        const float expectedViewportHeight = 1.0f;
        const float expectedOutputWidth = 1600.0f;
        const float expectedOutputHeight = 900.0f;
        if (std::fabs(input.projectionParams.x - camera.verticalFovDegrees) > 1e-5f ||
            std::fabs(input.projectionParams.y - camera.nearClip) > 1e-5f ||
            std::fabs(input.projectionParams.z - camera.farClip) > 1e-5f ||
            std::fabs(input.viewportAndOutputSize.x - expectedViewportWidth) > 1e-5f ||
            std::fabs(input.viewportAndOutputSize.y - expectedViewportHeight) > 1e-5f ||
            std::fabs(input.viewportAndOutputSize.z - expectedOutputWidth) > 1e-5f ||
            std::fabs(input.viewportAndOutputSize.w - expectedOutputHeight) > 1e-5f)
        {
            CRESSIM_LOG_ERROR( "World->GPU camera sync did not preserve viewport-aware projection data.\n");
            return 1;
        }
        break;
    }
    if (!foundCamera)
    {
        CRESSIM_LOG_ERROR( "Camera input missing in world GPU sync data.\n");
        return 1;
    }

    const Diligent::float4 writebackPos{4.0f, 5.0f, 6.0f, state->inverseMass};
    const Diligent::float4 writebackRot{0.0f, 0.0f, 0.3826834f, 0.9238795f};
    const Diligent::float4 writebackLin{state->linearVelocity.x, state->linearVelocity.y,
                                        state->linearVelocity.z, 0.0f};
    const Diligent::float4 writebackAng{state->angularVelocity.x, state->angularVelocity.y,
                                        state->angularVelocity.z, 0.0f};
    if (!world.physicsWorld().writeBackRigidBodyState(0u, writebackPos, writebackRot, writebackLin,
                                                      writebackAng))
    {
        CRESSIM_LOG_ERROR( "Failed to write back rigid state into physics world.\n");
        return 1;
    }
    world.physicsWorld().finalizeRigidBodyWriteback();

    world.refreshFromPhysics();
    const std::optional<engine::TransformComponent> syncedTransform = world.tryGetTransform(entity);
    if (!syncedTransform)
    {
        CRESSIM_LOG_ERROR( "Physics->world sync removed transform unexpectedly.\n");
        return 1;
    }

    const float dx = std::fabs(syncedTransform->worldTransform.position.x - writebackPos.x);
    const float dy = std::fabs(syncedTransform->worldTransform.position.y - writebackPos.y);
    const float dz = std::fabs(syncedTransform->worldTransform.position.z - writebackPos.z);
    if (dx > 1e-5f || dy > 1e-5f || dz > 1e-5f)
    {
        CRESSIM_LOG_ERROR( "Physics->world transform position sync mismatch.\n");
        return 1;
    }
    const float dr = std::fabs(syncedTransform->worldTransform.rotation.q.z - writebackRot.z);
    if (dr > 1e-5f)
    {
        CRESSIM_LOG_ERROR( "Physics->world transform rotation sync mismatch.\n");
        return 1;
    }

    const std::optional<engine::RigidBodyComponent> syncedRigidBody = world.tryGetRigidBody(entity);
    if (!syncedRigidBody || syncedRigidBody->bodyType != physics::RigidBodyType::Kinematic ||
        !syncedRigidBody->kinematicTargetEnabled ||
        std::fabs(syncedRigidBody->kinematicTargetPosition.x - rigidBody.kinematicTargetPosition.x) > 1e-5f)
    {
        CRESSIM_LOG_ERROR( "Physics->world rigid body sync mismatch.\n");
        return 1;
    }

    CRESSIM_LOG_INFO( "Runtime physics sync field checks passed.\n");
    return 0;
}
