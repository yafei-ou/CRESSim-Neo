#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyLinearVelocityCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyAngularVelocityCorrections);

static const float kVelocityCorrectionAtomicScale = 100000.0;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    const uint bodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyIndex);
    float4 linearVelocity = CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, bodyIndex);
    float4 angularVelocity = CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, bodyIndex);
    const float3 linearCorrection =
        float3(CRESSIM_SB_REF(g_RigidBodyLinearVelocityCorrections, bodyIndex).xyz) / kVelocityCorrectionAtomicScale;
    const float3 angularCorrection =
        float3(CRESSIM_SB_REF(g_RigidBodyAngularVelocityCorrections, bodyIndex).xyz) / kVelocityCorrectionAtomicScale;

    if (bodyType == 2u)
    {
        linearVelocity.xyz += linearCorrection;
        angularVelocity.xyz += angularCorrection;
    }

    CRESSIM_SB_STORE(g_PredictedRigidBodyLinearVelocities, bodyIndex, linearVelocity);
    CRESSIM_SB_STORE(g_PredictedRigidBodyAngularVelocities, bodyIndex, angularVelocity);
    CRESSIM_SB_STORE(g_RigidBodyLinearVelocityCorrections, bodyIndex, int4(0, 0, 0, 0));
    CRESSIM_SB_STORE(g_RigidBodyAngularVelocityCorrections, bodyIndex, int4(0, 0, 0, 0));
}
