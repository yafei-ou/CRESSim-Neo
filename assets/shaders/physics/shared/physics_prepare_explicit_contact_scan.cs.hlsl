#include "physics/particle/physics_particle_types.hlsli"
#include "physics/rigid/physics_rigid_broad_phase_types.hlsli"

#define CRESSIM_PARTICLE_CONTACT_SCAN_COUNT_EXPR \
    CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u).particleParticleCandidateCount
#include "physics/physics_prepare_particle_contact_scan.hlsli"
