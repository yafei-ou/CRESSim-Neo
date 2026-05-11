#include "common/logger.h"
#include "engine/components.h"
#include "engine/world.h"
#include "physics/soft_phase.h"

#include <cmath>

namespace
{

bool nearlyEqual(float lhs, float rhs)
{
    return std::abs(lhs - rhs) <= 1.0e-5f;
}

bool nearlyEqual(const Diligent::float4 &lhs, const Diligent::float4 &rhs)
{
    return nearlyEqual(lhs.x, rhs.x) && nearlyEqual(lhs.y, rhs.y) && nearlyEqual(lhs.z, rhs.z) &&
           nearlyEqual(lhs.w, rhs.w);
}

} // namespace

int main()
{
    using namespace cressim::neo;

    engine::World world;
    common::SceneLayoutDesc layout{};
    layout.envCount = 2u;
    world.setSceneLayout(layout);

    const common::EntityId entity = world.createEntity();

    engine::SoftBodyComponent softBody{};
    softBody.source.kind                          = physics::SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size              = {1.0f, 1.0f, 1.0f};
    softBody.source.regularGrid.targetParticleSpacing = 1.0f;
    softBody.particleMass                         = 1.0f;
    softBody.particleRadius                       = 0.10f;
    softBody.edgeCompliance                       = 0.01f;
    softBody.volumeCompliance                     = 0.02f;
    softBody.material.contact.friction            = 0.30f;
    softBody.material.contact.restitution         = 0.10f;
    softBody.material.contact.damping             = 0.05f;
    softBody.selfCollisionEnabled                 = false;
    softBody.collisionLayer                       = 0x2u;
    softBody.collisionMask                        = 0x5u;
    if (!world.setSoftBody(entity, softBody))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body for runtime-update preservation test.");
        return 1;
    }

    physics::PhysicsWorld &physicsWorld  = world.physicsWorld();
    const auto &initialParticles         = physicsWorld.particles();
    const physics::SoftBodyState *softState = physicsWorld.tryGetSoftBody(entity);
    if (softState == nullptr || softState->particleCount == 0u || initialParticles.empty())
    {
        CRESSIM_LOG_ERROR("Authored soft body did not produce any particles.");
        return 1;
    }

    const std::uint32_t particleIndex = softState->particleOffset;
    const Diligent::float4 displacedPositionInvMass{-2.0f, 3.5f, 1.25f, 1.0f};
    const Diligent::float4 displacedPrevious{-2.25f, 3.0f, 1.0f, 0.0f};
    const Diligent::float4 displacedVelocity{4.0f, -1.5f, 0.75f, 0.0f};
    if (!physicsWorld.syncParticleStateFromSimulation(particleIndex, displacedPositionInvMass,
                                                          displacedPrevious, displacedVelocity))
    {
        CRESSIM_LOG_ERROR("Failed to seed deformed particle state.");
        return 1;
    }
    physicsWorld.finalizeParticleWriteback();

    if (!world.setEntityEnvironment(entity, 1u))
    {
        CRESSIM_LOG_ERROR("setEntityEnvironment failed for soft body.");
        return 1;
    }

    softState = physicsWorld.tryGetSoftBody(entity);
    const auto &particlesAfterEnv = physicsWorld.particles();
    if (softState == nullptr || softState->environmentIndex != 1u)
    {
        CRESSIM_LOG_ERROR("Soft body environment change did not stick.");
        return 1;
    }
    if (!nearlyEqual(particlesAfterEnv.positionsInvMass[particleIndex], displacedPositionInvMass) ||
        !nearlyEqual(particlesAfterEnv.previousPositions[particleIndex], displacedPrevious) ||
        !nearlyEqual(particlesAfterEnv.velocities[particleIndex], displacedVelocity) ||
        particlesAfterEnv.environmentIndices[particleIndex] != 1u)
    {
        CRESSIM_LOG_ERROR("Environment update reset live soft-particle state.");
        return 1;
    }

    const std::optional<engine::SoftBodyComponent> storedComponent = world.tryGetSoftBody(entity);
    if (!storedComponent.has_value())
    {
        CRESSIM_LOG_ERROR("World failed to return the stored soft-body component.");
        return 1;
    }

    engine::SoftBodyComponent updated = *storedComponent;
    updated.material.contact.friction    = 0.65f;
    updated.material.contact.restitution = 0.20f;
    updated.material.contact.damping     = 0.15f;
    updated.particleMass              = 2.0f;
    updated.particleRadius            = 0.20f;
    updated.edgeCompliance            = 0.05f;
    updated.volumeCompliance          = 0.08f;
    updated.selfCollisionEnabled      = true;
    updated.collisionLayer            = 0x8u;
    updated.collisionMask             = 0x14u;
    if (!world.setSoftBody(entity, updated))
    {
        CRESSIM_LOG_ERROR("Runtime soft-body update failed.");
        return 1;
    }

    softState = physicsWorld.tryGetSoftBody(entity);
    const auto &particlesAfterUpdate = physicsWorld.particles();
    const auto &edgesAfterUpdate     = physicsWorld.softEdges();
    const auto &tetsAfterUpdate      = physicsWorld.softTets();
    if (softState == nullptr)
    {
        CRESSIM_LOG_ERROR("Soft body disappeared after runtime update.");
        return 1;
    }

    if (!nearlyEqual(particlesAfterUpdate.positionsInvMass[particleIndex], Diligent::float4{
                                                                        displacedPositionInvMass.x,
                                                                        displacedPositionInvMass.y,
                                                                        displacedPositionInvMass.z,
                                                                        0.5f}) ||
        !nearlyEqual(particlesAfterUpdate.previousPositions[particleIndex], displacedPrevious) ||
        !nearlyEqual(particlesAfterUpdate.velocities[particleIndex], displacedVelocity))
    {
        CRESSIM_LOG_ERROR("Runtime soft-body update reset deformed particle state.");
        return 1;
    }

    if (particlesAfterUpdate.environmentIndices[particleIndex] != 1u ||
        !physics::softParticlePhaseSelfCollideEnabled(particlesAfterUpdate.phases[particleIndex]))
    {
        CRESSIM_LOG_ERROR("Runtime soft-body update did not refresh particle metadata.");
        return 1;
    }

    const std::uint32_t materialIndex = particlesAfterUpdate.particleMaterialIndices[particleIndex];
    const auto &contactMaterials = physicsWorld.particleContactMaterials();
    if (materialIndex >= contactMaterials.size() ||
        !nearlyEqual(contactMaterials[materialIndex].x, 0.65f) ||
        !nearlyEqual(contactMaterials[materialIndex].y, 0.20f) ||
        !nearlyEqual(contactMaterials[materialIndex].z, 0.15f) ||
        !nearlyEqual(particlesAfterUpdate.radii[particleIndex], 0.20f) ||
        particlesAfterUpdate.collisionLayers[particleIndex] != 0x8u ||
        particlesAfterUpdate.collisionMasks[particleIndex] != 0x14u)
    {
        CRESSIM_LOG_ERROR("Runtime soft-body update did not patch particle properties in place.");
        return 1;
    }

    if (softState->edgeCount == 0u || softState->tetCount == 0u ||
        !nearlyEqual(edgesAfterUpdate[softState->edgeOffset].compliance, 0.05f) ||
        !nearlyEqual(tetsAfterUpdate[softState->tetOffset].compliance, 0.08f))
    {
        CRESSIM_LOG_ERROR("Runtime soft-body update did not patch constraint compliance.");
        return 1;
    }

    CRESSIM_LOG_INFO("Soft body runtime updates preserve live state.");
    return 0;
}
