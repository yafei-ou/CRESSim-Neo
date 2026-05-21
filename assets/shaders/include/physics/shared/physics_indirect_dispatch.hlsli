#ifndef CRESSIM_NEO_PHYSICS_SHARED_INDIRECT_DISPATCH_HLSLI
#define CRESSIM_NEO_PHYSICS_SHARED_INDIRECT_DISPATCH_HLSLI

#include "../core/physics_base.hlsli"

static const uint kRigidContactsPerPair = 4u;
static const uint kPhysicsIndirectSoftGenerateContacts = 0u;
static const uint kPhysicsIndirectSoftGenerateRigidContacts = 1u;
static const uint kPhysicsIndirectSoftCompactContacts = 2u;
static const uint kPhysicsIndirectSoftCompactRigidContacts = 3u;
static const uint kPhysicsIndirectSoftSolveContacts = 4u;
static const uint kPhysicsIndirectSoftSolveRigidContacts = 5u;
static const uint kPhysicsIndirectSoftSolveContactVelocities = 6u;
static const uint kPhysicsIndirectRigidGenerateContacts = 7u;
static const uint kPhysicsIndirectRigidSolveContacts = 8u;
static const uint kPhysicsIndirectRigidSolveContactVelocities = 9u;
static const uint kPhysicsIndirectDispatchSlotCount = 10u;

struct GpuDispatchIndirectArgs
{
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

#endif // CRESSIM_NEO_PHYSICS_SHARED_INDIRECT_DISPATCH_HLSLI
