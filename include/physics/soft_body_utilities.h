#ifndef CRESSIM_NEO_PHYSICS_SOFT_BODY_UTILITIES_H
#define CRESSIM_NEO_PHYSICS_SOFT_BODY_UTILITIES_H

#include "physics/export.h"
#include "physics/physics_types.h"

#include <cstdint>
#include <vector>

namespace cressim::neo::physics
{

struct SoftRigidBroadPhaseCandidate
{
    std::uint32_t softParticleIndex = 0u;
    std::uint32_t rigidBodyIndex    = 0u;
};

struct SoftRigidContact
{
    std::uint32_t softParticleIndex = 0u;
    std::uint32_t rigidBodyIndex    = 0u;
    std::uint32_t colliderIndex     = 0u;
    Diligent::float3 normal{0.0f, 1.0f, 0.0f};
    float penetration = 0.0f;
};

CRESSIM_NEO_PHYSICS_API std::vector<SoftRigidBroadPhaseCandidate>
buildSoftRigidBroadPhaseCandidatesCpu(const SoftParticleSoAHost &softParticles,
                                      const RigidSurfaceParticleSoAHost &surfaceParticles,
                                      float cellSize);

CRESSIM_NEO_PHYSICS_API bool computeSoftRigidContactCpu(
    const SoftParticleSoAHost &softParticles, std::uint32_t softParticleIndex,
    const RigidBodyState &rigidBody, std::uint32_t rigidBodyIndex, const ColliderState &collider,
    std::uint32_t colliderIndex, SoftRigidContact &outContact);

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_SOFT_BODY_UTILITIES_H
