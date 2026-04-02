#ifndef CRESSIM_NEO_PHYSICS_SOFT_PHASE_H
#define CRESSIM_NEO_PHYSICS_SOFT_PHASE_H

#include <cstdint>

namespace cressim::neo::physics
{

constexpr std::uint32_t kSoftPhaseGroupMask       = 0x7fffffffu;
constexpr std::uint32_t kSoftPhaseSelfCollideFlag = 0x80000000u;

constexpr std::uint32_t packSoftParticlePhase(std::uint32_t groupId,
                                              bool selfCollisionEnabled) noexcept
{
    return (groupId & kSoftPhaseGroupMask) |
           (selfCollisionEnabled ? kSoftPhaseSelfCollideFlag : 0u);
}

constexpr std::uint32_t softParticlePhaseGroup(std::uint32_t phase) noexcept
{
    return phase & kSoftPhaseGroupMask;
}

constexpr bool softParticlePhaseSelfCollideEnabled(std::uint32_t phase) noexcept
{
    return (phase & kSoftPhaseSelfCollideFlag) != 0u;
}

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_SOFT_PHASE_H
