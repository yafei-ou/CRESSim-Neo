#ifndef CRESSIM_NEO_PHYSICS_SOFT_RIGID_NEIGHBOR_HLSLI
#define CRESSIM_NEO_PHYSICS_SOFT_RIGID_NEIGHBOR_HLSLI

bool IsValidSoftRigidCandidate(float3 softPosition, float softRadius, uint softEnvironment,
                               uint softLayer, uint softMask,
                               GpuParticleBroadPhaseEntry candidateEntry,
                               out uint rigidBodyIndex, out uint surfaceIndex)
{
    if (candidateEntry.particleType != kParticleBroadPhaseEntryTypeRigidSurface)
    {
        rigidBodyIndex = 0u;
        surfaceIndex = 0u;
        return false;
    }

    surfaceIndex = candidateEntry.particleIndex;
    rigidBodyIndex = candidateEntry.ownerIndex;
    const uint surfaceEnvironment =
        CRESSIM_SB_LOAD(g_RigidSurfaceParticleEnvironmentIndices, surfaceIndex);
    if (surfaceEnvironment != softEnvironment)
    {
        return false;
    }

    const uint surfaceLayer = CRESSIM_SB_LOAD(g_RigidSurfaceParticleCollisionLayers, surfaceIndex);
    const uint surfaceMask = CRESSIM_SB_LOAD(g_RigidSurfaceParticleCollisionMasks, surfaceIndex);
    if ((softMask & surfaceLayer) == 0u || (surfaceMask & softLayer) == 0u)
    {
        return false;
    }

    const float3 surfacePosition =
        CRESSIM_SB_LOAD(g_RigidSurfaceParticleWorldPositions, surfaceIndex).xyz;
    const float combinedRadius =
        softRadius + CRESSIM_SB_LOAD(g_RigidSurfaceParticleSampleRadii, surfaceIndex);
    const float3 deltaPos = surfacePosition - softPosition;
    return dot(deltaPos, deltaPos) <= combinedRadius * combinedRadius;
}

#endif
