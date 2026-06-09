#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleStrandRoles);
CRESSIM_STRUCTURED_BUFFER(uint4, g_SuturingNeighborLinks);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwningSoftBodyIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleContactMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_ParticleCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);
CRESSIM_STRUCTURED_BUFFER(GpuStrandInsertionStateStorage, g_SuturingInsertionStates);

CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleContact, g_ParticleContacts);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_ContactActiveFlags);

bool ShouldSuppressSequenceFollowerContact(uint particleIndex, uint otherOwningSoftBody)
{
    if (otherOwningSoftBody == kInvalidSuturingIndex)
    {
        return false;
    }

    const uint strandRole = CRESSIM_SB_LOAD(g_ParticleStrandRoles, particleIndex);
    if (strandRole != kParticleStrandRoleNeedleBody && strandRole != kParticleStrandRoleThread)
    {
        return false;
    }

    const uint predecessorIndex = CRESSIM_SB_LOAD(g_SuturingNeighborLinks, particleIndex).x;
    if (predecessorIndex == kInvalidSuturingIndex)
    {
        return false;
    }

    const GpuStrandInsertionStateStorage predecessorState =
        CRESSIM_SB_LOAD(g_SuturingInsertionStates, predecessorIndex);
    if (predecessorState.state == kStrandInsertionStateInside &&
        predecessorState.softBodyIndex == otherOwningSoftBody)
    {
        return true;
    }

    return false;
}

bool ShouldSuppressSequenceLeaderContact(uint particleIndex, uint otherOwningSoftBody)
{
    if (otherOwningSoftBody == kInvalidSuturingIndex)
    {
        return false;
    }

    const uint strandRole = CRESSIM_SB_LOAD(g_ParticleStrandRoles, particleIndex);
    if (strandRole != kParticleStrandRoleNeedleBody && strandRole != kParticleStrandRoleThread)
    {
        return false;
    }

    const uint nextIndex = CRESSIM_SB_LOAD(g_SuturingNeighborLinks, particleIndex).y;
    if (nextIndex == kInvalidSuturingIndex)
    {
        return false;
    }

    const GpuStrandInsertionStateStorage nextState =
        CRESSIM_SB_LOAD(g_SuturingInsertionStates, nextIndex);
    if (nextState.state == kStrandInsertionStateInside &&
        nextState.softBodyIndex == otherOwningSoftBody)
    {
        return true;
    }

    return false;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint pairIndex = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (pairIndex >= meta.particleParticleCandidateCount)
    {
        return;
    }

    GpuParticleContact outContact = (GpuParticleContact)0;

    const GpuParticleCandidatePair pair = CRESSIM_SB_LOAD(g_ParticleCandidatePairs, pairIndex);

    const uint particleA = pair.indexA;
    const uint particleB = pair.indexB;
    const GpuStrandInsertionStateStorage insertionA =
        CRESSIM_SB_LOAD(g_SuturingInsertionStates, particleA);
    const GpuStrandInsertionStateStorage insertionB =
        CRESSIM_SB_LOAD(g_SuturingInsertionStates, particleB);
    const uint ownerTypeA = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleA);
    const uint ownerTypeB = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleB);
    const uint owningSoftBodyA = CRESSIM_SB_LOAD(g_ParticleOwningSoftBodyIndices, particleA);
    const uint owningSoftBodyB = CRESSIM_SB_LOAD(g_ParticleOwningSoftBodyIndices, particleB);
    const bool suppressA = insertionA.state == kStrandInsertionStateInside &&
                           insertionA.softBodyIndex != kInvalidSuturingIndex &&
                           owningSoftBodyB == insertionA.softBodyIndex;
    const bool suppressB = insertionB.state == kStrandInsertionStateInside &&
                           insertionB.softBodyIndex != kInvalidSuturingIndex &&
                           owningSoftBodyA == insertionB.softBodyIndex;
    const bool suppressNeedleFollowerA =
        ShouldSuppressSequenceFollowerContact(particleA, owningSoftBodyB);
    const bool suppressNeedleFollowerB =
        ShouldSuppressSequenceFollowerContact(particleB, owningSoftBodyA);
    const bool suppressNeedleLeaderA =
        ShouldSuppressSequenceLeaderContact(particleA, owningSoftBodyB);
    const bool suppressNeedleLeaderB =
        ShouldSuppressSequenceLeaderContact(particleB, owningSoftBodyA);
    const bool suppressNeedleTipA =
        ownerTypeA == kParticleOwnerTypeRigidBody &&
        CRESSIM_SB_LOAD(g_ParticleStrandRoles, particleA) == kParticleStrandRoleNeedleTip &&
        owningSoftBodyB != kInvalidSuturingIndex;
    const bool suppressNeedleTipB =
        ownerTypeB == kParticleOwnerTypeRigidBody &&
        CRESSIM_SB_LOAD(g_ParticleStrandRoles, particleB) == kParticleStrandRoleNeedleTip &&
        owningSoftBodyA != kInvalidSuturingIndex;
    if (suppressA || suppressB || suppressNeedleFollowerA || suppressNeedleFollowerB ||
        suppressNeedleLeaderA || suppressNeedleLeaderB || suppressNeedleTipA ||
        suppressNeedleTipB)
    {
        CRESSIM_SB_STORE(g_ParticleContacts, pairIndex, outContact);
        CRESSIM_SB_STORE(g_ContactActiveFlags, pairIndex, 0u);
        return;
    }

    const uint kindA = CRESSIM_SB_LOAD(g_ParticleKinds, particleA);
    const uint kindB = CRESSIM_SB_LOAD(g_ParticleKinds, particleB);
    const uint materialIndexA = CRESSIM_SB_LOAD(g_ParticleMaterialIndices, particleA);
    const uint materialIndexB = CRESSIM_SB_LOAD(g_ParticleMaterialIndices, particleB);
    outContact.material.xyz = CombineContactMaterial(
        CRESSIM_SB_LOAD(g_ParticleContactMaterials, materialIndexA),
        CRESSIM_SB_LOAD(g_ParticleContactMaterials, materialIndexB));
    if (kindA == kParticleKindFluid && kindB == kParticleKindFluid)
    {
        CRESSIM_SB_STORE(g_ParticleContacts, pairIndex, outContact);
        CRESSIM_SB_STORE(g_ContactActiveFlags, pairIndex, 0u);
        return;
    }
    const float3 positionA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleA).xyz;
    const float3 positionB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleB).xyz;
    const float radiusA = CRESSIM_SB_LOAD(g_ParticleRadii, particleA);
    const float radiusB = CRESSIM_SB_LOAD(g_ParticleRadii, particleB);
    const float3 delta = positionB - positionA;
    const float distanceSq = dot(delta, delta);
    const float combinedRadius = radiusA + radiusB;
    if (distanceSq <= kEpsilon)
    {
        outContact.particleA = particleA;
        outContact.particleB = particleB;
        outContact.active = 1u;
        outContact.normalPenetration = float4(0.0, 1.0, 0.0, combinedRadius);
        CRESSIM_SB_STORE(g_ParticleContacts, pairIndex, outContact);
        CRESSIM_SB_STORE(g_ContactActiveFlags, pairIndex, 1u);
        return;
    }

    const float distance = sqrt(distanceSq);
    const float penetration = combinedRadius - distance;
    if (penetration > 0.0)
    {
        outContact.particleA = particleA;
        outContact.particleB = particleB;
        outContact.active = 1u;
        outContact.normalPenetration = float4(delta / distance, penetration);
    }

    CRESSIM_SB_STORE(g_ParticleContacts, pairIndex, outContact);
    CRESSIM_SB_STORE(g_ContactActiveFlags, pairIndex, outContact.active);
}
