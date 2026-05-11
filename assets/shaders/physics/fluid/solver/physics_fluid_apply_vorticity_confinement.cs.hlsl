#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);
CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint4, g_ParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(GpuFluidMaterial, g_FluidMaterials);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidVorticities);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocitiesRW);

#include "../../../include/physics/fluid/physics_fluid_common.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    if (CRESSIM_SB_LOAD(g_ParticleKinds, particleIndex) != kParticleKindFluid)
    {
        return;
    }

    const uint fluidMaterialIndex = CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex);
    const GpuFluidMaterial fluidMaterial = CRESSIM_SB_LOAD(g_FluidMaterials, fluidMaterialIndex);
    const float confinement = max(fluidMaterial.vorticityConfinement, 0.0);
    if (confinement <= kEpsilon)
    {
        return;
    }

    const float4 selfPositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float selfInvMass = selfPositionInvMass.w;
    if (selfInvMass <= kEpsilon)
    {
        return;
    }

    const float4 selfCurl4 = CRESSIM_SB_LOAD(g_FluidVorticities, particleIndex);
    const float3 selfCurl = selfCurl4.xyz;
    const float selfCurlMagnitude = selfCurl4.w;
    if (selfCurlMagnitude <= kEpsilon)
    {
        return;
    }

    const float3 selfPosition = selfPositionInvMass.xyz;
    const float smoothingRadius = max(fluidMaterial.smoothingRadius, 1.0e-4);
    const uint4 selfMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, particleIndex);
    const uint selfEnvironment = selfMetadata.x;
    const uint selfLayer = selfMetadata.z;
    const uint selfMask = selfMetadata.w;
    const GpuParticleBroadPhaseEntry selfEntry =
        CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, particleIndex);

    float3 vorticityGrad = float3(0.0, 0.0, 0.0);

    [loop]
    for (int dz = -1; dz <= 1; ++dz)
    {
        [loop]
        for (int dy = -1; dy <= 1; ++dy)
        {
            [loop]
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int targetCellX = selfEntry.cellX + dx;
                const int targetCellY = selfEntry.cellY + dy;
                const int targetCellZ = selfEntry.cellZ + dz;
                const uint targetKey =
                    ComputeParticleGridCellKey(targetCellX, targetCellY, targetCellZ);
                const GpuParticleCellRange range = FindParticleCellRange(targetKey);
                if (range.cellKey == kInvalidIndex)
                {
                    continue;
                }

                [loop]
                for (uint sortedIndex = range.startIndex; sortedIndex < range.endIndex; ++sortedIndex)
                {
                    const GpuMortonCodeElement keyEntry =
                        CRESSIM_SB_LOAD(g_SortedParticleBroadPhaseKeys, sortedIndex);
                    const GpuParticleBroadPhaseEntry candidateEntry =
                        CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, keyEntry.elementIdx);
                    if (candidateEntry.cellX != targetCellX || candidateEntry.cellY != targetCellY ||
                        candidateEntry.cellZ != targetCellZ)
                    {
                        continue;
                    }

                    const uint neighborIndex = candidateEntry.particleIndex;
                    if (!ShouldProcessFluidNeighbor(particleIndex, selfEnvironment, selfLayer,
                                                    selfMask, neighborIndex) ||
                        !AreSameFluidMaterial(particleIndex, neighborIndex))
                    {
                        continue;
                    }

                    const float4 neighborPositionInvMass =
                        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, neighborIndex);
                    const float neighborInvMass = neighborPositionInvMass.w;
                    if (neighborInvMass <= kEpsilon)
                    {
                        continue;
                    }

                    const float3 delta = selfPosition - neighborPositionInvMass.xyz;
                    const float distance = length(delta);
                    if (distance <= kEpsilon || distance >= smoothingRadius)
                    {
                        continue;
                    }

                    const float neighborCurlMagnitude =
                        CRESSIM_SB_LOAD(g_FluidVorticities, neighborIndex).w;
                    vorticityGrad += (neighborCurlMagnitude - selfCurlMagnitude) *
                                     FluidCubicKernelGradient(delta, distance, smoothingRadius);
                }
            }
        }
    }

    const float gradLength = length(vorticityGrad);
    if (gradLength <= kEpsilon)
    {
        return;
    }

    const float3 vorticityForce = confinement * cross(vorticityGrad / gradLength, selfCurl);
    const float3 velocity = CRESSIM_SB_LOAD(g_ParticleVelocitiesRW, particleIndex).xyz;
    CRESSIM_SB_STORE(g_ParticleVelocitiesRW, particleIndex,
                     float4(velocity + dt * vorticityForce, 0.0));
}
