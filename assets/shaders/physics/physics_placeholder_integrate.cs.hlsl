cbuffer PhysicsStepConstantsBuffer
{
    float dt;
    uint bodyCount;
    uint substepIndex;
    uint iterationIndex;
};

RWStructuredBuffer<float4> g_RigidBodyPositionsInvMass;
RWStructuredBuffer<float4> g_RigidBodyOrientations;
RWStructuredBuffer<float4> g_RigidBodyLinearVelocities;
RWStructuredBuffer<float4> g_RigidBodyAngularVelocities;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= bodyCount)
    {
        return;
    }

    float4 positionInvMass = g_RigidBodyPositionsInvMass[idx];
    const float4 linearVelocity = g_RigidBodyLinearVelocities[idx];
    positionInvMass.xyz += linearVelocity.xyz * dt;
    g_RigidBodyPositionsInvMass[idx] = positionInvMass;

    // Keep these UAVs explicitly read-write until dedicated solve/update passes land.
    g_RigidBodyLinearVelocities[idx] = linearVelocity;
    g_RigidBodyAngularVelocities[idx] = g_RigidBodyAngularVelocities[idx];
}
