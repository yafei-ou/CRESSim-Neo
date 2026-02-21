struct RigidBodyState
{
    float4 positionInvMass;
    float4 rotation;
    float4 linearVelocity;
};

cbuffer PhysicsStepConstantsBuffer
{
    float dt;
    uint bodyCount;
    float2 _padding;
};

RWStructuredBuffer<RigidBodyState> g_RigidBodies;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= bodyCount)
    {
        return;
    }

    RigidBodyState rb = g_RigidBodies[idx];
    rb.positionInvMass.xyz += rb.linearVelocity.xyz * dt;
    g_RigidBodies[idx] = rb;
}
