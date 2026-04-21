#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_atomic_float.hlsli"
CRESSIM_RW_BYTE_ADDRESS_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_BYTE_ADDRESS_BUFFER(g_RigidBodyRotationCorrections);
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

    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_RigidBodyTranslationCorrections, bodyIndex);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_RigidBodyRotationCorrections, bodyIndex);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_RigidBodyLinearVelocityCorrections, bodyIndex);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_RigidBodyAngularVelocityCorrections, bodyIndex);
}
