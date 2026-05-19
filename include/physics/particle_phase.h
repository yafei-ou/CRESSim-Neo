#ifndef CRESSIM_NEO_PHYSICS_PARTICLE_PHASE_H
#define CRESSIM_NEO_PHYSICS_PARTICLE_PHASE_H

#include <cstdint>

namespace cressim::neo::physics
{

constexpr std::uint32_t kParticlePhaseGroupMask       = 0x7fffffffu;
constexpr std::uint32_t kParticlePhaseSelfCollideFlag = 0x80000000u;

constexpr std::uint32_t packParticlePhase(std::uint32_t groupId, bool selfCollisionEnabled) noexcept
{
    return (groupId & kParticlePhaseGroupMask) |
           (selfCollisionEnabled ? kParticlePhaseSelfCollideFlag : 0u);
}

constexpr std::uint32_t particlePhaseGroup(std::uint32_t phase) noexcept
{
    return phase & kParticlePhaseGroupMask;
}

constexpr bool particlePhaseSelfCollideEnabled(std::uint32_t phase) noexcept
{
    return (phase & kParticlePhaseSelfCollideFlag) != 0u;
}

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_PHASE_H
