#include "physics/include/physics_rigid_dispatch_constants.hlsli"
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyTranslationCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyRotationCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyLinearVelocityCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyAngularVelocityCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    CRESSIM_SB_STORE(g_RigidBodyTranslationCorrections, bodyIndex, int4(0, 0, 0, 0));
    CRESSIM_SB_STORE(g_RigidBodyRotationCorrections, bodyIndex, int4(0, 0, 0, 0));
    CRESSIM_SB_STORE(g_RigidBodyLinearVelocityCorrections, bodyIndex, int4(0, 0, 0, 0));
    CRESSIM_SB_STORE(g_RigidBodyAngularVelocityCorrections, bodyIndex, int4(0, 0, 0, 0));
}
