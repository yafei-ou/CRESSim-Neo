#include "physics/include/physics_rigid_dispatch_constants.hlsli"
RWStructuredBuffer<int4> g_RigidBodyTranslationCorrections;
RWStructuredBuffer<int4> g_RigidBodyRotationCorrections;
RWStructuredBuffer<int4> g_RigidBodyLinearVelocityCorrections;
RWStructuredBuffer<int4> g_RigidBodyAngularVelocityCorrections;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    g_RigidBodyTranslationCorrections[bodyIndex] = int4(0, 0, 0, 0);
    g_RigidBodyRotationCorrections[bodyIndex] = int4(0, 0, 0, 0);
    g_RigidBodyLinearVelocityCorrections[bodyIndex] = int4(0, 0, 0, 0);
    g_RigidBodyAngularVelocityCorrections[bodyIndex] = int4(0, 0, 0, 0);
}
