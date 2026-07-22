#ifndef CRESSIM_NEO_PHYSICS_RIGID_DISPATCH_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_DISPATCH_CONSTANTS_HLSLI

cbuffer PhysicsRigidDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint colliderCount;
    uint activeMovingCount;
    uint staticBodyCount;
    uint candidatePairCapacity;
    uint contactCapacity;
    uint reservedSubstepIndex;
    uint reservedIterationIndex;
    uint reservedSolverIterations;
    uint reserved0;
    uint reserved1;
    float4 gravity;
};

#endif // CRESSIM_NEO_PHYSICS_RIGID_DISPATCH_CONSTANTS_HLSLI
