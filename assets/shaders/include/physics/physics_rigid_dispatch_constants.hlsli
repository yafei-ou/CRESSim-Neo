#ifndef CRESSIM_NEO_PHYSICS_RIGID_DISPATCH_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_DISPATCH_CONSTANTS_HLSLI

cbuffer PhysicsRigidDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint colliderCount;
    uint activeMovingCount;
    uint staticBodyCount;
    uint candidatePairCount;
    uint candidatePairCapacity;

    // These 3 are not used for now
    uint substepIndex;
    uint iterationIndex;
    uint solverIterations;

    uint reserved0;
    uint reserved1;
};

#endif // CRESSIM_NEO_PHYSICS_RIGID_DISPATCH_CONSTANTS_HLSLI
