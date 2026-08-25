#ifndef CRESSIM_NEO_PHYSICS_PARTICLE_PHASE_H
#define CRESSIM_NEO_PHYSICS_PARTICLE_PHASE_H

#include <cstdint>

/// @file particle_phase.h
/// @brief Bitmask definitions and packing helpers for particle phase and self-collision flags.

namespace cressim::neo::physics
{

/// @brief Bitmask isolating the 31-bit particle collision phase group ID.
constexpr std::uint32_t kParticlePhaseGroupMask = 0x7fffffffu;

/// @brief Bit flag indicating whether intra-group particle self-collision is enabled.
constexpr std::uint32_t kParticlePhaseSelfCollideFlag = 0x80000000u;

/// @brief Packs a particle phase group ID and a self-collision boolean into a single 32-bit
/// integer.
/// @param groupId Unique phase group identifier (lower 31 bits).
/// @param selfCollisionEnabled Whether particles in this group can collide with each other.
/// @return Packed 32-bit particle phase bitfield.
constexpr std::uint32_t packParticlePhase(std::uint32_t groupId, bool selfCollisionEnabled) noexcept
{
    return (groupId & kParticlePhaseGroupMask) |
           (selfCollisionEnabled ? kParticlePhaseSelfCollideFlag : 0u);
}

/// @brief Extracts the phase group ID from a packed particle phase bitfield.
/// @param phase Packed 32-bit particle phase value.
/// @return Unmasked phase group identifier.
constexpr std::uint32_t particlePhaseGroup(std::uint32_t phase) noexcept
{
    return phase & kParticlePhaseGroupMask;
}

/// @brief Checks whether self-collision is enabled in a packed particle phase bitfield.
/// @param phase Packed 32-bit particle phase value.
/// @return True if the self-collision flag bit is set.
constexpr bool particlePhaseSelfCollideEnabled(std::uint32_t phase) noexcept
{
    return (phase & kParticlePhaseSelfCollideFlag) != 0u;
}

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_PHASE_H
