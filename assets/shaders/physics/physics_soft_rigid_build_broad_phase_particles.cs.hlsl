#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleOwningSoftBodyIndices);

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidSurfaceParticleWorldPositions);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleOwningRigidBodyIndices);

CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);

uint ComputeParticleGridCellKey(int gx, int gy, int gz)
{
    uint seed = uint(gx) * 73856093u;
    seed ^= uint(gy) * 19349663u;
    seed ^= uint(gz) * 83492791u;
    return seed;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    const uint totalParticleCount = softParticleCount + rigidSurfaceParticleCount;
    if (idx >= totalParticleCount)
    {
        return;
    }

    GpuParticleBroadPhaseEntry entry;

    if (idx < softParticleCount)
    {
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
        entry.ownerIndex = CRESSIM_SB_LOAD(g_SoftParticleOwningSoftBodyIndices, idx);
        entry.reserved0 = 0u;
    }
    else
    {
        const uint surfaceIndex = idx - softParticleCount;
        const float3 position =
            CRESSIM_SB_LOAD(g_RigidSurfaceParticleWorldPositions, surfaceIndex).xyz;
        const float safeCellSize = max(particleGridCellSize, 1.0e-4);
        const float invCellSize = 1.0 / safeCellSize;
        const int gx = int(floor(position.x * invCellSize));
        const int gy = int(floor(position.y * invCellSize));
        const int gz = int(floor(position.z * invCellSize));
        entry.cellKey = ComputeParticleGridCellKey(gx, gy, gz);
        entry.cellX = gx;
        entry.cellY = gy;
        entry.cellZ = gz;
        entry.particleIndex = surfaceIndex;
        entry.particleType = kParticleBroadPhaseEntryTypeRigidSurface;
        entry.ownerIndex = CRESSIM_SB_LOAD(g_RigidSurfaceParticleOwningRigidBodyIndices, surfaceIndex);
        entry.reserved0 = 0u;
    }

    CRESSIM_SB_STORE(g_ParticleBroadPhaseEntries, idx, entry);
}
