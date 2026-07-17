#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/particle/physics_particle_types.hlsli"
#include "physics/particle/physics_particle_grid.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);

CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= particleCount)
    {
        return;
    }

    GpuParticleBroadPhaseEntry entry;
    const float3 position = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, idx).xyz;
    const float safeCellSize = max(particleGridCellSize, 1.0e-4);
    const float invCellSize = 1.0 / safeCellSize;
    const int gx = int(floor(position.x * invCellSize));
    const int gy = int(floor(position.y * invCellSize));
    const int gz = int(floor(position.z * invCellSize));
    entry.cellKey = ComputeParticleGridCellKey(gx, gy, gz);
    entry.cellX = gx;
    entry.cellY = gy;
    entry.cellZ = gz;
    entry.particleIndex = idx;
    entry.particleType = kParticleBroadPhaseEntryTypeParticle;
    entry.reserved0 = 0u;
    entry.reserved1 = 0u;

    CRESSIM_SB_STORE(g_ParticleBroadPhaseEntries, idx, entry);
}
