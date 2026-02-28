cbuffer PhysicsDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint particleCount;
    uint substepIndex;
    uint iterationIndex;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

RWStructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
RWStructuredBuffer<float4> g_PredictedRigidBodyOrientations;
RWStructuredBuffer<float4> g_PredictedRigidBodyLinearVelocities;
RWStructuredBuffer<float4> g_PredictedRigidBodyAngularVelocities;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= rigidBodyCount)
    {
        return;
    }

    float4 positionInvMass = g_PredictedRigidBodyPositionsInvMass[idx];
    const float4 linearVelocity = g_PredictedRigidBodyLinearVelocities[idx];
    positionInvMass.xyz += linearVelocity.xyz * dt;
    g_PredictedRigidBodyPositionsInvMass[idx] = positionInvMass;

    // Keep these UAVs explicitly read-write until dedicated solve/update passes land.
    g_PredictedRigidBodyLinearVelocities[idx] = linearVelocity;
    g_PredictedRigidBodyAngularVelocities[idx] = g_PredictedRigidBodyAngularVelocities[idx];
}
