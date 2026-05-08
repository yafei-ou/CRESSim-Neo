#include "../../../include/physics/physics_soft_dispatch_constants.hlsli"
#include "../../../include/physics/soft/physics_soft_types.hlsli"
#include "../../../include/physics/soft/physics_soft_grid.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);

CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= softParticleCount)
    {
        return;
    }

    GpuParticleBroadPhaseEntry entry;
    const float3 position = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, idx).xyz;
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
    entry.particleType = kParticleBroadPhaseEntryTypeSoft;
    entry.reserved0 = 0u;
    entry.reserved1 = 0u;

    CRESSIM_SB_STORE(g_ParticleBroadPhaseEntries, idx, entry);
}
