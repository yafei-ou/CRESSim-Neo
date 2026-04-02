#include "physics/soft_body_utilities.h"
#include "common/logger.h"

namespace
{

bool checkContact(const cressim::neo::physics::SoftParticleSoAHost &softParticles,
                  const cressim::neo::physics::RigidBodyState &rigidBody,
                  const cressim::neo::physics::ColliderState &collider)
{
    cressim::neo::physics::SoftRigidContact contact{};
    return cressim::neo::physics::computeSoftRigidContactCpu(softParticles, 0u, rigidBody, 0u, collider,
                                                             0u, contact) &&
           contact.penetration > 0.0f;
}

} // namespace

int main()
{
    using namespace cressim::neo::physics;

    RigidBodyState rigidBody{};
    rigidBody.position = {0.0f, 0.0f, 0.0f};
    rigidBody.scale = {1.0f, 1.0f, 1.0f};

    SoftParticleSoAHost soft;
    soft.positionsInvMass.push_back({0.0f, 0.0f, 0.0f, 1.0f});
    soft.previousPositions.push_back({0.0f, 0.0f, 0.0f, 0.0f});
    soft.velocities.push_back({0.0f, 0.0f, 0.0f, 0.0f});
    soft.environmentIndices.push_back(0u);
    soft.owningSoftBodyIndices.push_back(0u);
    soft.collisionLayers.push_back(1u);
    soft.collisionMasks.push_back(0xffffffffu);
    soft.radii.push_back(0.2f);

    ColliderState sphere{};
    sphere.environmentIndex = 0u;
    sphere.shapeType = ColliderShapeType::Sphere;
    sphere.shapeParams = {0.5f, 0.0f, 0.0f, 0.0f};
    sphere.collisionLayer = 1u;
    sphere.collisionMask = 0xffffffffu;
    soft.positionsInvMass[0] = {0.55f, 0.0f, 0.0f, 1.0f};
    if (!checkContact(soft, rigidBody, sphere))
    {
        CRESSIM_LOG_ERROR("Expected sphere soft-rigid contact.");
        return 1;
    }

    ColliderState box{};
    box.environmentIndex = 0u;
    box.shapeType = ColliderShapeType::Box;
    box.shapeParams = {0.5f, 0.5f, 0.5f, 0.0f};
    box.collisionLayer = 1u;
    box.collisionMask = 0xffffffffu;
    soft.positionsInvMass[0] = {0.6f, 0.0f, 0.0f, 1.0f};
    if (!checkContact(soft, rigidBody, box))
    {
        CRESSIM_LOG_ERROR("Expected box soft-rigid contact.");
        return 1;
    }

    ColliderState capsule{};
    capsule.environmentIndex = 0u;
    capsule.shapeType = ColliderShapeType::Capsule;
    capsule.shapeParams = {0.35f, 0.6f, 0.0f, 0.0f};
    capsule.collisionLayer = 1u;
    capsule.collisionMask = 0xffffffffu;
    soft.positionsInvMass[0] = {0.45f, 0.0f, 0.0f, 1.0f};
    if (!checkContact(soft, rigidBody, capsule))
    {
        CRESSIM_LOG_ERROR("Expected capsule soft-rigid contact.");
        return 1;
    }

    capsule.environmentIndex = 2u;
    if (checkContact(soft, rigidBody, capsule))
    {
        CRESSIM_LOG_ERROR("Unexpected cross-environment soft-rigid contact.");
        return 1;
    }

    CRESSIM_LOG_INFO("Soft-body analytic narrow-phase checks passed.");
    return 0;
}
