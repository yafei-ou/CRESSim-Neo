#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_RW_BYTE_ADDRESS_BUFFER(g_RigidBodyLinearVelocityCorrections);
CRESSIM_RW_BYTE_ADDRESS_BUFFER(g_RigidBodyAngularVelocityCorrections);

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
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_RigidBodyLinearVelocityCorrections, bodyIndex);
    const float3 angularCorrection =
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_RigidBodyAngularVelocityCorrections, bodyIndex);

    if (bodyType == kRigidBodyTypeDynamic)
    {
        linearVelocity.xyz += linearCorrection;
        angularVelocity.xyz += angularCorrection;
    }

    CRESSIM_SB_STORE(g_PredictedRigidBodyLinearVelocities, bodyIndex, linearVelocity);
    CRESSIM_SB_STORE(g_PredictedRigidBodyAngularVelocities, bodyIndex, angularVelocity);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_RigidBodyLinearVelocityCorrections, bodyIndex);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_RigidBodyAngularVelocityCorrections, bodyIndex);
}
