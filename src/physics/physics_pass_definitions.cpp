#include "physics/physics_pass_definitions.h"

namespace cressim::neo::physics::passdefs
{
namespace
{
constexpr Diligent::ShaderResourceVariableDesc kPredictVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyKinematicTargetPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyKinematicTargetOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyKinematicTargetFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSoftPredictVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePreviousPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSyncRigidProxyParticlesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidProxyLocalPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildParticleBroadPhaseEntriesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseEntries",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildParticleBroadPhaseKeysVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseEntries",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kMarkParticleCellRangeStartsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SortedParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRangeStartFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearParticleCellRangesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildParticleCellRangesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SortedParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRangeStartFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRangeStartOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kCountParticleParticleCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseEntries",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SortedParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleAdjacencyOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleAdjacencyCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleAdjacencyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kFinalizeSoftCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kEmitParticleParticleCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseEntries",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SortedParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleAdjacencyOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleAdjacencyCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleAdjacencyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kCountParticleRigidCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyColliderRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kEmitParticleRigidCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyColliderRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kCountFluidBoundaryCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kEmitFluidBoundaryCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildFluidNeighborPairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseEntries",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SortedParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidIterationDelta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborPairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kGenerateParticleRigidContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyColliderRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyColliderIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderGeometryData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kGenerateParticleExplicitContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleStrandRoles",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingNeighborLinks",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwningSoftBodyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleContactMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingInsertionStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kPrepareSoftIndirectArgsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PhysicsIndirectDispatchArgs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kFinalizeActiveContactVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kCompactActiveParticleExplicitContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveSoftContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kCompactActiveParticleRigidContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveSoftRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearSoftConstraintStateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdgeLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBendLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTetLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandJointLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandDistanceLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearRigidParticleAttachmentConstraintStateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidParticleAttachmentLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearStrandRigidAttachmentConstraintStateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandRigidAttachmentLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearRoutedCableConstraintStateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RoutedCableLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearRigidDistanceConstraintStateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidDistanceConstraintLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearSuturingCandidatesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingCandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingCandidateParticles",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kGatherSuturingCandidatesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingNeighborLinks",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingCandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingCandidateParticles",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClassifySuturingParticlesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingParticleRefs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwningSoftBodyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingCandidateParticles",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleTetRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleIncidentTets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTets", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingPairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingInsertionStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateSuturingTipPathsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTets", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingPairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingInsertionStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingPathHeaders",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingPathNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kAssignSuturingInsideParticlesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingParticleRefs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingPairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingInsertionStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingPathHeaders",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingPathNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTets", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSuturingNodePathConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingParticleRefs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePreviousPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidProxyLocalPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingInsertionStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SuturingPathNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTets", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearHingeJointConstraintStateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidJointDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_HingeJointLambdas0123",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_HingeJointLambdas45",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearSphericalJointConstraintStateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidJointDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SphericalJointTranslationLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SphericalJointRotationLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearSliderJointConstraintStateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidJointDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SliderJointLambdas0123",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SliderJointLambdas45",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSoftEdgeConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdges", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdgeLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdgeCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSoftBendConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBends", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBendLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBendCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSoftTetConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTets", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTetLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTetCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplySoftEdgeCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleEdgeRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleIncidentEdges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdgeCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplySoftBendCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBendRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleIncidentBends",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBendCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplySoftTetCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleTetRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleIncidentTets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTetCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveStrandSegmentConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegments",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyStrandSegmentCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleStrandSegmentRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleIncidentStrandSegments",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveStrandJointConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandJoints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegments",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandJointLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandJointCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyStrandJointCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandJointCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SegmentStrandJointRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SegmentIncidentStrandJoints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyStrandRigidAttachmentCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandRigidAttachmentCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SegmentStrandRigidAttachmentRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SegmentIncidentStrandRigidAttachments",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveStrandDistanceConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandDistanceConstraints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandDistanceLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandDistanceCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyStrandDistanceCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleStrandSegmentRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleIncidentStrandSegments",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandDistanceCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveParticleRigidContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePreviousPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleContactMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveParticleExplicitContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePreviousPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidProxyLocalPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyParticlePositionCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateParticleVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePreviousPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleContactMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidIterationDelta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kComputeFluidDensityConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidIterationDelta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborPairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidBoundaryCandidateRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidBoundaryCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderGeometryData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidSurfaceNormalConstraints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kComputeFluidDeltaPositionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidIterationDelta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidSurfaceNormalConstraints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborPairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidBoundaryCandidateRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidBoundaryCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderGeometryData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidDeltaPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyFluidDeltaPositionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidDeltaPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidIterationDelta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClampFluidBoundaryVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidIterationDelta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidBoundaryCandidateRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidBoundaryCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderGeometryData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidIterationDeltaRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kProjectFluidBoundaryVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleContactMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidBoundaryCandidateRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidBoundaryCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderGeometryData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kComputeFluidVorticityVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocitiesRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborPairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidVorticities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyFluidVorticityConfinementVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidVorticities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborPairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocitiesRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildFluidRenderAnisotropyVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleKinds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidNeighborPairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidAnisotropy1RW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidAnisotropy2RW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_FluidAnisotropy3RW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveParticleContactVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidProxyLocalPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveParticleRigidContactVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleMaterialIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleContactMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyParticleContactVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsParticleDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSkinSoftRenderVerticesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftRenderDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRenderVertexBindings",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBodyRenderPositionsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateSoftRenderNormalsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftRenderDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRenderTriangleNormals",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRenderVertexTriangleRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRenderVertexTriangleIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRenderFallbackNormals",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBodyRenderNormalsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateSoftTriangleNormalsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftRenderDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRenderTriangleParticleIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRenderTriangleNormalsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateCurveRenderDataVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsCurveRenderDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CurveRenderDescriptors",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CurveRenderParticleIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CurveRenderPositionsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CurveRenderNormalsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CurveWorldAabbsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateSoftBodyBoundsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftRenderDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBodyBoundsChunks",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBodyChunkAabbsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kFinalizeSoftBodyBoundsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftRenderDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBodyChunkRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBodyChunkAabbs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBodyWorldAabbsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateWorldAabbsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderOwnerRigidBodyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderShapeTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderGeometryData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderEnabledFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyMeta", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBodyFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kScanBlockVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsScanDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanInput", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanOutput",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BlockSums", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kScanAddOffsetsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsScanDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScannedBlockOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanOutput",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kPrepareParticleContactScanVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanConstantsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanIndirectArgsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};
constexpr Diligent::ShaderResourceVariableDesc kCompactBodySetVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodySetFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodySetOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseBodyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyMeta", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kFinalizeActiveBodiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBodyFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBodyOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildBroadPhaseElementsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidBroadPhaseBuildConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseBodyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kReduceExtentElementsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidBroadPhaseReductionConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_GroupExtents",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kReduceExtentExtentsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidBroadPhaseReductionConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_InputExtents",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_OutputExtents",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kMortonCodesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidBroadPhaseBuildConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_GlobalExtent",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kRadixClassifyVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRadixConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodesIn",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kRadixFinalizeVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRadixConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RadixMeta", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kRadixScatterVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRadixConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodesIn",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RadixMeta", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodesOut",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kBvhHierarchyVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidBroadPhaseBuildConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SortedMortonCodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhConstructionInfos",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBvhBoundingBoxesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidBroadPhaseBuildConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhConstructionInfos",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kCountPairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseBodyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_JointCollisionSuppressionOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_JointCollisionSuppressionNeighbors",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereSphere",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereBox",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsBoxBox",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsBoxCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsCapsuleCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kFinalizePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereSphere",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereBox",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsSphereCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsBoxBox",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsBoxCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairCountsCapsuleCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereSphere",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereBox",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsBoxBox",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsBoxCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsCapsuleCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidPairRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kEmitPairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseBodyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_JointCollisionSuppressionOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_JointCollisionSuppressionNeighbors",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereSphere",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereBox",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsSphereCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsBoxBox",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsBoxCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsetsCapsuleCapsule",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidPairRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildNarrowPhaseChunksVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidPairRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseChunks",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseChunkCounter",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kPrepareRigidIndirectArgsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ProxyRigidContactMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PhysicsIndirectDispatchArgs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kGenerateContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderContactData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseChunks",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_NarrowPhaseChunkCounter",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kGenerateProxyRigidContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ProxyRigidContactMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleOwnerIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyProxyParticleContactMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyColliderRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyColliderIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderContactData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kFinalRigidContactDepenetrationVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ProxyRigidContactMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kInitRigidContactVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ProxyRigidContactMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateMap",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateActiveCount",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateHeaders",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateSlots",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearRigidBodyPairContactAggregatesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateMap",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateActiveCount",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateHeaders",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kPrepareRigidContactVelocityIndirectArgsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateActiveCount",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PhysicsIndirectDispatchArgs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveRigidContactVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateActiveCount",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateHeaders",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPairAggregateSlots",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveBallJointConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BallJoints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSphericalJointConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidJointDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SphericalJoints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SphericalJointTranslationLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SphericalJointRotationLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveHingeJointConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_HingeJoints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_HingeJointIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_HingeJointLambdas0123",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_HingeJointLambdas45",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSliderJointConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SliderJoints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SliderJointIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SliderJointLambdas0123",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SliderJointLambdas45",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveRoutedCableConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RoutedCableConstraints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RoutedCableRoutePoints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RoutedCableLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveRigidDistanceConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidDistanceConstraints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidDistanceConstraintLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveRigidParticleAttachmentConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidParticleAttachments",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidParticleAttachmentLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveStrandRigidAttachmentConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegments",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandSegmentStates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandRigidAttachments",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandRigidAttachmentLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StrandRigidAttachmentCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticlePositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveHingeJointTargetVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidJointDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_HingeJoints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_HingeJointIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSliderJointTargetVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidJointDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SliderJoints",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SliderJointIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderMacro kJointDriveModePassiveMacros[] = {
    {"CRESSIM_JOINT_DRIVE_MODE_TARGET_POSITION", "0"},
};

constexpr Diligent::ShaderMacro kJointDriveModeTargetPositionMacros[] = {
    {"CRESSIM_JOINT_DRIVE_MODE_TARGET_POSITION", "1"},
};

constexpr Diligent::ShaderResourceVariableDesc kClearCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplyRigidContactVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsRigidDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

} // namespace

const gpu::GpuComputePassDefinition kPredictRigid{
    "physics/rigid/solver/physics_rigid_predict.cs.hlsl",
    "CRESSimNeo.Physics.RigidPredict.CS",
    "CRESSimNeo.Physics.RigidPredict.PSO",
    kPredictVars,
    std::size(kPredictVars),
};

const gpu::GpuComputePassDefinition kSoftPredict{
    "physics/soft/solver/physics_soft_predict.cs.hlsl",
    "CRESSimNeo.Physics.SoftPredict.CS",
    "CRESSimNeo.Physics.SoftPredict.PSO",
    kSoftPredictVars,
    std::size(kSoftPredictVars),
};

const gpu::GpuComputePassDefinition kSyncRigidProxyParticles{
    "physics/particle/solver/physics_particle_sync_rigid_proxy_particles.cs.hlsl",
    "CRESSimNeo.Physics.SyncRigidProxyParticles.CS",
    "CRESSimNeo.Physics.SyncRigidProxyParticles.PSO",
    kSyncRigidProxyParticlesVars,
    std::size(kSyncRigidProxyParticlesVars),
};

const gpu::GpuComputePassDefinition kBuildParticleBroadPhaseEntries{
    "physics/particle/broad_phase/physics_particle_build_broad_phase_entries.cs.hlsl",
    "CRESSimNeo.Physics.ParticleBuildBroadPhaseEntries.CS",
    "CRESSimNeo.Physics.ParticleBuildBroadPhaseEntries.PSO",
    kBuildParticleBroadPhaseEntriesVars,
    std::size(kBuildParticleBroadPhaseEntriesVars),
};

const gpu::GpuComputePassDefinition kBuildParticleBroadPhaseKeys{
    "physics/particle/broad_phase/physics_particle_build_broad_phase_keys.cs.hlsl",
    "CRESSimNeo.Physics.ParticleBuildBroadPhaseKeys.CS",
    "CRESSimNeo.Physics.ParticleBuildBroadPhaseKeys.PSO",
    kBuildParticleBroadPhaseKeysVars,
    std::size(kBuildParticleBroadPhaseKeysVars),
};

const gpu::GpuComputePassDefinition kMarkParticleCellRangeStarts{
    "physics/particle/broad_phase/physics_particle_mark_cell_range_starts.cs.hlsl",
    "CRESSimNeo.Physics.ParticleMarkCellRangeStarts.CS",
    "CRESSimNeo.Physics.ParticleMarkCellRangeStarts.PSO",
    kMarkParticleCellRangeStartsVars,
    std::size(kMarkParticleCellRangeStartsVars),
};

const gpu::GpuComputePassDefinition kClearParticleCellRanges{
    "physics/particle/broad_phase/physics_particle_clear_cell_ranges.cs.hlsl",
    "CRESSimNeo.Physics.ParticleClearCellRanges.CS",
    "CRESSimNeo.Physics.ParticleClearCellRanges.PSO",
    kClearParticleCellRangesVars,
    std::size(kClearParticleCellRangesVars),
};

const gpu::GpuComputePassDefinition kBuildParticleCellRanges{
    "physics/particle/broad_phase/physics_particle_build_cell_ranges.cs.hlsl",
    "CRESSimNeo.Physics.ParticleBuildCellRanges.CS",
    "CRESSimNeo.Physics.ParticleBuildCellRanges.PSO",
    kBuildParticleCellRangesVars,
    std::size(kBuildParticleCellRangesVars),
};

const gpu::GpuComputePassDefinition kCountParticleParticleCandidatePairs{
    "physics/particle/broad_phase/physics_particle_count_particle_particle_candidates.cs.hlsl",
    "CRESSimNeo.Physics.CountParticleParticleCandidatePairs.CS",
    "CRESSimNeo.Physics.CountParticleParticleCandidatePairs.PSO",
    kCountParticleParticleCandidatePairsVars,
    std::size(kCountParticleParticleCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kFinalizeParticleParticleCandidatePairs{
    "physics/particle/broad_phase/physics_particle_finalize_particle_particle_candidates.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeParticleParticleCandidatePairs.CS",
    "CRESSimNeo.Physics.FinalizeParticleParticleCandidatePairs.PSO",
    kFinalizeSoftCandidatePairsVars,
    std::size(kFinalizeSoftCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kEmitParticleParticleCandidatePairs{
    "physics/particle/broad_phase/physics_particle_emit_particle_particle_candidates.cs.hlsl",
    "CRESSimNeo.Physics.EmitParticleParticleCandidatePairs.CS",
    "CRESSimNeo.Physics.EmitParticleParticleCandidatePairs.PSO",
    kEmitParticleParticleCandidatePairsVars,
    std::size(kEmitParticleParticleCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kCountParticleRigidCandidatePairs{
    "physics/particle/broad_phase/physics_particle_count_particle_rigid_candidates.cs.hlsl",
    "CRESSimNeo.Physics.CountParticleRigidCandidatePairs.CS",
    "CRESSimNeo.Physics.CountParticleRigidCandidatePairs.PSO",
    kCountParticleRigidCandidatePairsVars,
    std::size(kCountParticleRigidCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kFinalizeParticleRigidCandidatePairs{
    "physics/particle/broad_phase/physics_particle_finalize_particle_rigid_candidates.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeParticleRigidCandidatePairs.CS",
    "CRESSimNeo.Physics.FinalizeParticleRigidCandidatePairs.PSO",
    kFinalizeSoftCandidatePairsVars,
    std::size(kFinalizeSoftCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kEmitParticleRigidCandidatePairs{
    "physics/particle/broad_phase/physics_particle_emit_particle_rigid_candidates.cs.hlsl",
    "CRESSimNeo.Physics.EmitParticleRigidCandidatePairs.CS",
    "CRESSimNeo.Physics.EmitParticleRigidCandidatePairs.PSO",
    kEmitParticleRigidCandidatePairsVars,
    std::size(kEmitParticleRigidCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kCountFluidBoundaryCandidatePairs{
    "physics/particle/broad_phase/physics_particle_count_fluid_boundary_candidates.cs.hlsl",
    "CRESSimNeo.Physics.CountFluidBoundaryCandidatePairs.CS",
    "CRESSimNeo.Physics.CountFluidBoundaryCandidatePairs.PSO",
    kCountFluidBoundaryCandidatePairsVars,
    std::size(kCountFluidBoundaryCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kFinalizeFluidBoundaryCandidatePairs{
    "physics/particle/broad_phase/physics_particle_finalize_fluid_boundary_candidates.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeFluidBoundaryCandidatePairs.CS",
    "CRESSimNeo.Physics.FinalizeFluidBoundaryCandidatePairs.PSO",
    kFinalizeSoftCandidatePairsVars,
    std::size(kFinalizeSoftCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kEmitFluidBoundaryCandidatePairs{
    "physics/particle/broad_phase/physics_particle_emit_fluid_boundary_candidates.cs.hlsl",
    "CRESSimNeo.Physics.EmitFluidBoundaryCandidatePairs.CS",
    "CRESSimNeo.Physics.EmitFluidBoundaryCandidatePairs.PSO",
    kEmitFluidBoundaryCandidatePairsVars,
    std::size(kEmitFluidBoundaryCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kGenerateParticleRigidContacts{
    "physics/particle/narrow_phase/physics_particle_rigid_generate_contacts.cs.hlsl",
    "CRESSimNeo.Physics.GenerateParticleRigidContacts.CS",
    "CRESSimNeo.Physics.GenerateParticleRigidContacts.PSO",
    kGenerateParticleRigidContactsVars,
    std::size(kGenerateParticleRigidContactsVars),
};

const gpu::GpuComputePassDefinition kGenerateParticleExplicitContacts{
    "physics/particle/narrow_phase/physics_particle_generate_explicit_contacts.cs.hlsl",
    "CRESSimNeo.Physics.GenerateParticleExplicitContacts.CS",
    "CRESSimNeo.Physics.GenerateParticleExplicitContacts.PSO",
    kGenerateParticleExplicitContactsVars,
    std::size(kGenerateParticleExplicitContactsVars),
};

const gpu::GpuComputePassDefinition kPrepareExplicitContactScan{
    "physics/shared/physics_prepare_explicit_contact_scan.cs.hlsl",
    "CRESSimNeo.Physics.PrepareExplicitContactScan.CS",
    "CRESSimNeo.Physics.PrepareExplicitContactScan.PSO",
    kPrepareParticleContactScanVars,
    std::size(kPrepareParticleContactScanVars),
};

const gpu::GpuComputePassDefinition kPrepareRigidContactScan{
    "physics/shared/physics_prepare_rigid_contact_scan.cs.hlsl",
    "CRESSimNeo.Physics.PrepareRigidContactScan.CS",
    "CRESSimNeo.Physics.PrepareRigidContactScan.PSO",
    kPrepareParticleContactScanVars,
    std::size(kPrepareParticleContactScanVars),
};

const gpu::GpuComputePassDefinition kPrepareParticleCandidateIndirectArgs{
    "physics/shared/physics_prepare_soft_candidate_indirect_args.cs.hlsl",
    "CRESSimNeo.Physics.PrepareParticleCandidateIndirectArgs.CS",
    "CRESSimNeo.Physics.PrepareParticleCandidateIndirectArgs.PSO",
    kPrepareSoftIndirectArgsVars,
    std::size(kPrepareSoftIndirectArgsVars),
};

const gpu::GpuComputePassDefinition kPrepareParticleActiveIndirectArgs{
    "physics/shared/physics_prepare_soft_active_indirect_args.cs.hlsl",
    "CRESSimNeo.Physics.PrepareParticleActiveIndirectArgs.CS",
    "CRESSimNeo.Physics.PrepareParticleActiveIndirectArgs.PSO",
    kPrepareSoftIndirectArgsVars,
    std::size(kPrepareSoftIndirectArgsVars),
};

const gpu::GpuComputePassDefinition kFinalizeActiveParticleExplicitContacts{
    "physics/particle/narrow_phase/physics_particle_finalize_active_explicit_contacts.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeActiveParticleExplicitContacts.CS",
    "CRESSimNeo.Physics.FinalizeActiveParticleExplicitContacts.PSO",
    kFinalizeActiveContactVars,
    std::size(kFinalizeActiveContactVars),
};

const gpu::GpuComputePassDefinition kCompactActiveParticleExplicitContacts{
    "physics/particle/narrow_phase/physics_particle_compact_active_explicit_contacts.cs.hlsl",
    "CRESSimNeo.Physics.CompactActiveParticleExplicitContacts.CS",
    "CRESSimNeo.Physics.CompactActiveParticleExplicitContacts.PSO",
    kCompactActiveParticleExplicitContactsVars,
    std::size(kCompactActiveParticleExplicitContactsVars),
};

const gpu::GpuComputePassDefinition kFinalizeActiveParticleRigidContacts{
    "physics/particle/narrow_phase/physics_particle_finalize_active_rigid_contacts.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeActiveParticleRigidContacts.CS",
    "CRESSimNeo.Physics.FinalizeActiveParticleRigidContacts.PSO",
    kFinalizeActiveContactVars,
    std::size(kFinalizeActiveContactVars),
};

const gpu::GpuComputePassDefinition kCompactActiveParticleRigidContacts{
    "physics/particle/narrow_phase/physics_particle_compact_active_rigid_contacts.cs.hlsl",
    "CRESSimNeo.Physics.CompactActiveParticleRigidContacts.CS",
    "CRESSimNeo.Physics.CompactActiveParticleRigidContacts.PSO",
    kCompactActiveParticleRigidContactsVars,
    std::size(kCompactActiveParticleRigidContactsVars),
};

const gpu::GpuComputePassDefinition kClearSoftConstraintState{
    "physics/shared/physics_clear_soft_constraint_state.cs.hlsl",
    "CRESSimNeo.Physics.ClearSoftConstraintState.CS",
    "CRESSimNeo.Physics.ClearSoftConstraintState.PSO",
    kClearSoftConstraintStateVars,
    std::size(kClearSoftConstraintStateVars),
};

const gpu::GpuComputePassDefinition kClearRigidParticleAttachmentConstraintState{
    "physics/rigid/solver/physics_rigid_clear_rigid_particle_attachment_constraint_state.cs.hlsl",
    "CRESSimNeo.Physics.ClearRigidParticleAttachmentConstraintState.CS",
    "CRESSimNeo.Physics.ClearRigidParticleAttachmentConstraintState.PSO",
    kClearRigidParticleAttachmentConstraintStateVars,
    std::size(kClearRigidParticleAttachmentConstraintStateVars),
};

const gpu::GpuComputePassDefinition kClearStrandRigidAttachmentConstraintState{
    "physics/rigid/solver/physics_rigid_clear_strand_rigid_attachment_constraint_state.cs.hlsl",
    "CRESSimNeo.Physics.ClearStrandRigidAttachmentConstraintState.CS",
    "CRESSimNeo.Physics.ClearStrandRigidAttachmentConstraintState.PSO",
    kClearStrandRigidAttachmentConstraintStateVars,
    std::size(kClearStrandRigidAttachmentConstraintStateVars),
};

const gpu::GpuComputePassDefinition kClearRoutedCableConstraintState{
    "physics/rigid/solver/physics_rigid_clear_routed_cable_constraint_state.cs.hlsl",
    "CRESSimNeo.Physics.ClearRoutedCableConstraintState.CS",
    "CRESSimNeo.Physics.ClearRoutedCableConstraintState.PSO",
    kClearRoutedCableConstraintStateVars,
    std::size(kClearRoutedCableConstraintStateVars),
};

const gpu::GpuComputePassDefinition kClearRigidDistanceConstraintState{
    "physics/rigid/solver/physics_rigid_clear_rigid_distance_constraint_state.cs.hlsl",
    "CRESSimNeo.Physics.ClearRigidDistanceConstraintState.CS",
    "CRESSimNeo.Physics.ClearRigidDistanceConstraintState.PSO",
    kClearRigidDistanceConstraintStateVars,
    std::size(kClearRigidDistanceConstraintStateVars),
};

const gpu::GpuComputePassDefinition kClearSuturingCandidates{
    "physics/soft/solver/physics_suturing_clear_candidates.cs.hlsl",
    "CRESSimNeo.Physics.ClearSuturingCandidates.CS",
    "CRESSimNeo.Physics.ClearSuturingCandidates.PSO",
    kClearSuturingCandidatesVars,
    std::size(kClearSuturingCandidatesVars),
};

const gpu::GpuComputePassDefinition kGatherSuturingCandidates{
    "physics/soft/solver/physics_suturing_gather_candidates.cs.hlsl",
    "CRESSimNeo.Physics.GatherSuturingCandidates.CS",
    "CRESSimNeo.Physics.GatherSuturingCandidates.PSO",
    kGatherSuturingCandidatesVars,
    std::size(kGatherSuturingCandidatesVars),
};

const gpu::GpuComputePassDefinition kClassifySuturingParticles{
    "physics/soft/solver/physics_suturing_classify_strand_particles.cs.hlsl",
    "CRESSimNeo.Physics.ClassifySuturingParticles.CS",
    "CRESSimNeo.Physics.ClassifySuturingParticles.PSO",
    kClassifySuturingParticlesVars,
    std::size(kClassifySuturingParticlesVars),
};

const gpu::GpuComputePassDefinition kUpdateSuturingTipPaths{
    "physics/soft/solver/physics_suturing_update_tip_paths.cs.hlsl",
    "CRESSimNeo.Physics.UpdateSuturingTipPaths.CS",
    "CRESSimNeo.Physics.UpdateSuturingTipPaths.PSO",
    kUpdateSuturingTipPathsVars,
    std::size(kUpdateSuturingTipPathsVars),
};

const gpu::GpuComputePassDefinition kAssignSuturingInsideParticles{
    "physics/soft/solver/physics_suturing_assign_inside_particles.cs.hlsl",
    "CRESSimNeo.Physics.AssignSuturingInsideParticles.CS",
    "CRESSimNeo.Physics.AssignSuturingInsideParticles.PSO",
    kAssignSuturingInsideParticlesVars,
    std::size(kAssignSuturingInsideParticlesVars),
};

const gpu::GpuComputePassDefinition kSolveSuturingNodePathConstraints{
    "physics/soft/solver/physics_suturing_solve_node_path_constraints.cs.hlsl",
    "CRESSimNeo.Physics.SolveSuturingNodePathConstraints.CS",
    "CRESSimNeo.Physics.SolveSuturingNodePathConstraints.PSO",
    kSolveSuturingNodePathConstraintsVars,
    std::size(kSolveSuturingNodePathConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveSoftEdgeConstraints{
    "physics/soft/solver/physics_soft_solve_edge_constraints.cs.hlsl",
    "CRESSimNeo.Physics.SolveSoftEdgeConstraints.CS",
    "CRESSimNeo.Physics.SolveSoftEdgeConstraints.PSO",
    kSolveSoftEdgeConstraintsVars,
    std::size(kSolveSoftEdgeConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveSoftBendConstraints{
    "physics/soft/solver/physics_soft_solve_bend_constraints.cs.hlsl",
    "CRESSimNeo.Physics.SolveSoftBendConstraints.CS",
    "CRESSimNeo.Physics.SolveSoftBendConstraints.PSO",
    kSolveSoftBendConstraintsVars,
    std::size(kSolveSoftBendConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveSoftTetConstraints{
    "physics/soft/solver/physics_soft_solve_tet_constraints.cs.hlsl",
    "CRESSimNeo.Physics.SolveSoftTetConstraints.CS",
    "CRESSimNeo.Physics.SolveSoftTetConstraints.PSO",
    kSolveSoftTetConstraintsVars,
    std::size(kSolveSoftTetConstraintsVars),
};

const gpu::GpuComputePassDefinition kApplySoftEdgeCorrections{
    "physics/soft/solver/physics_soft_apply_edge_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplySoftEdgeCorrections.CS",
    "CRESSimNeo.Physics.ApplySoftEdgeCorrections.PSO",
    kApplySoftEdgeCorrectionsVars,
    std::size(kApplySoftEdgeCorrectionsVars),
};

const gpu::GpuComputePassDefinition kApplySoftBendCorrections{
    "physics/soft/solver/physics_soft_apply_bend_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplySoftBendCorrections.CS",
    "CRESSimNeo.Physics.ApplySoftBendCorrections.PSO",
    kApplySoftBendCorrectionsVars,
    std::size(kApplySoftBendCorrectionsVars),
};

const gpu::GpuComputePassDefinition kApplySoftTetCorrections{
    "physics/soft/solver/physics_soft_apply_tet_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplySoftTetCorrections.CS",
    "CRESSimNeo.Physics.ApplySoftTetCorrections.PSO",
    kApplySoftTetCorrectionsVars,
    std::size(kApplySoftTetCorrectionsVars),
};

const gpu::GpuComputePassDefinition kSolveStrandSegmentConstraints{
    "physics/soft/solver/physics_strand_solve_segment_constraints.cs.hlsl",
    "CRESSimNeo.Physics.SolveStrandSegmentConstraints.CS",
    "CRESSimNeo.Physics.SolveStrandSegmentConstraints.PSO",
    kSolveStrandSegmentConstraintsVars,
    std::size(kSolveStrandSegmentConstraintsVars),
};

const gpu::GpuComputePassDefinition kApplyStrandSegmentCorrections{
    "physics/soft/solver/physics_strand_apply_segment_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplyStrandSegmentCorrections.CS",
    "CRESSimNeo.Physics.ApplyStrandSegmentCorrections.PSO",
    kApplyStrandSegmentCorrectionsVars,
    std::size(kApplyStrandSegmentCorrectionsVars),
};

const gpu::GpuComputePassDefinition kSolveStrandJointConstraints{
    "physics/soft/solver/physics_strand_solve_joint_constraints.cs.hlsl",
    "CRESSimNeo.Physics.SolveStrandJointConstraints.CS",
    "CRESSimNeo.Physics.SolveStrandJointConstraints.PSO",
    kSolveStrandJointConstraintsVars,
    std::size(kSolveStrandJointConstraintsVars),
};

const gpu::GpuComputePassDefinition kApplyStrandJointCorrections{
    "physics/soft/solver/physics_strand_apply_joint_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplyStrandJointCorrections.CS",
    "CRESSimNeo.Physics.ApplyStrandJointCorrections.PSO",
    kApplyStrandJointCorrectionsVars,
    std::size(kApplyStrandJointCorrectionsVars),
};

const gpu::GpuComputePassDefinition kApplyStrandRigidAttachmentCorrections{
    "physics/soft/solver/physics_strand_apply_rigid_attachment_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplyStrandRigidAttachmentCorrections.CS",
    "CRESSimNeo.Physics.ApplyStrandRigidAttachmentCorrections.PSO",
    kApplyStrandRigidAttachmentCorrectionsVars,
    std::size(kApplyStrandRigidAttachmentCorrectionsVars),
};

const gpu::GpuComputePassDefinition kSolveStrandDistanceConstraints{
    "physics/soft/solver/physics_strand_solve_distance_constraints.cs.hlsl",
    "CRESSimNeo.Physics.SolveStrandDistanceConstraints.CS",
    "CRESSimNeo.Physics.SolveStrandDistanceConstraints.PSO",
    kSolveStrandDistanceConstraintsVars,
    std::size(kSolveStrandDistanceConstraintsVars),
};

const gpu::GpuComputePassDefinition kApplyStrandDistanceCorrections{
    "physics/soft/solver/physics_strand_apply_distance_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplyStrandDistanceCorrections.CS",
    "CRESSimNeo.Physics.ApplyStrandDistanceCorrections.PSO",
    kApplyStrandDistanceCorrectionsVars,
    std::size(kApplyStrandDistanceCorrectionsVars),
};

const gpu::GpuComputePassDefinition kSolveParticleRigidContacts{
    "physics/particle/solver/physics_particle_rigid_solve_contacts.cs.hlsl",
    "CRESSimNeo.Physics.SolveParticleRigidContacts.CS",
    "CRESSimNeo.Physics.SolveParticleRigidContacts.PSO",
    kSolveParticleRigidContactsVars,
    std::size(kSolveParticleRigidContactsVars),
};

const gpu::GpuComputePassDefinition kSolveParticleExplicitContacts{
    "physics/particle/solver/physics_particle_solve_explicit_contacts.cs.hlsl",
    "CRESSimNeo.Physics.SolveParticleExplicitContacts.CS",
    "CRESSimNeo.Physics.SolveParticleExplicitContacts.PSO",
    kSolveParticleExplicitContactsVars,
    std::size(kSolveParticleExplicitContactsVars),
};

const gpu::GpuComputePassDefinition kApplyParticlePositionCorrections{
    "physics/particle/solver/physics_particle_apply_position_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplyParticlePositionCorrections.CS",
    "CRESSimNeo.Physics.ApplyParticlePositionCorrections.PSO",
    kApplyParticlePositionCorrectionsVars,
    std::size(kApplyParticlePositionCorrectionsVars),
};

const gpu::GpuComputePassDefinition kUpdateParticleVelocities{
    "physics/particle/solver/physics_particle_update_velocities.cs.hlsl",
    "CRESSimNeo.Physics.UpdateParticleVelocities.CS",
    "CRESSimNeo.Physics.UpdateParticleVelocities.PSO",
    kUpdateParticleVelocitiesVars,
    std::size(kUpdateParticleVelocitiesVars),
};

const gpu::GpuComputePassDefinition kClearHingeJointConstraintState{
    "physics/rigid/solver/physics_rigid_clear_hinge_joint_constraint_state.cs.hlsl",
    "CRESSimNeo.Physics.ClearHingeJointConstraintState.CS",
    "CRESSimNeo.Physics.ClearHingeJointConstraintState.PSO",
    kClearHingeJointConstraintStateVars,
    std::size(kClearHingeJointConstraintStateVars),
};

const gpu::GpuComputePassDefinition kClearSphericalJointConstraintState{
    "physics/rigid/solver/physics_rigid_clear_spherical_joint_constraint_state.cs.hlsl",
    "CRESSimNeo.Physics.ClearSphericalJointConstraintState.CS",
    "CRESSimNeo.Physics.ClearSphericalJointConstraintState.PSO",
    kClearSphericalJointConstraintStateVars,
    std::size(kClearSphericalJointConstraintStateVars),
};

const gpu::GpuComputePassDefinition kClearSliderJointConstraintState{
    "physics/rigid/solver/physics_rigid_clear_slider_joint_constraint_state.cs.hlsl",
    "CRESSimNeo.Physics.ClearSliderJointConstraintState.CS",
    "CRESSimNeo.Physics.ClearSliderJointConstraintState.PSO",
    kClearSliderJointConstraintStateVars,
    std::size(kClearSliderJointConstraintStateVars),
};

const gpu::GpuComputePassDefinition kBuildFluidNeighborPairs{
    "physics/fluid/solver/physics_fluid_build_neighbors.cs.hlsl",
    "CRESSimNeo.Physics.BuildFluidNeighborPairs.CS",
    "CRESSimNeo.Physics.BuildFluidNeighborPairs.PSO",
    kBuildFluidNeighborPairsVars,
    std::size(kBuildFluidNeighborPairsVars),
};

const gpu::GpuComputePassDefinition kComputeFluidDensityConstraints{
    "physics/fluid/solver/physics_fluid_compute_density_lambda.cs.hlsl",
    "CRESSimNeo.Physics.ComputeFluidDensityConstraints.CS",
    "CRESSimNeo.Physics.ComputeFluidDensityConstraints.PSO",
    kComputeFluidDensityConstraintsVars,
    std::size(kComputeFluidDensityConstraintsVars),
};

const gpu::GpuComputePassDefinition kComputeFluidDeltaPositions{
    "physics/fluid/solver/physics_fluid_compute_delta_positions.cs.hlsl",
    "CRESSimNeo.Physics.ComputeFluidDeltaPositions.CS",
    "CRESSimNeo.Physics.ComputeFluidDeltaPositions.PSO",
    kComputeFluidDeltaPositionsVars,
    std::size(kComputeFluidDeltaPositionsVars),
};

const gpu::GpuComputePassDefinition kApplyFluidDeltaPositions{
    "physics/fluid/solver/physics_fluid_apply_delta_positions.cs.hlsl",
    "CRESSimNeo.Physics.ApplyFluidDeltaPositions.CS",
    "CRESSimNeo.Physics.ApplyFluidDeltaPositions.PSO",
    kApplyFluidDeltaPositionsVars,
    std::size(kApplyFluidDeltaPositionsVars),
};

const gpu::GpuComputePassDefinition kClampFluidBoundary{
    "physics/fluid/solver/physics_fluid_clamp_boundary.cs.hlsl",
    "CRESSimNeo.Physics.ClampFluidBoundary.CS",
    "CRESSimNeo.Physics.ClampFluidBoundary.PSO",
    kClampFluidBoundaryVars,
    std::size(kClampFluidBoundaryVars),
};

const gpu::GpuComputePassDefinition kProjectFluidBoundaryVelocities{
    "physics/fluid/solver/physics_fluid_project_boundary_velocities.cs.hlsl",
    "CRESSimNeo.Physics.ProjectFluidBoundaryVelocities.CS",
    "CRESSimNeo.Physics.ProjectFluidBoundaryVelocities.PSO",
    kProjectFluidBoundaryVelocitiesVars,
    std::size(kProjectFluidBoundaryVelocitiesVars),
};

const gpu::GpuComputePassDefinition kComputeFluidVorticity{
    "physics/fluid/solver/physics_fluid_compute_vorticity.cs.hlsl",
    "CRESSimNeo.Physics.ComputeFluidVorticity.CS",
    "CRESSimNeo.Physics.ComputeFluidVorticity.PSO",
    kComputeFluidVorticityVars,
    std::size(kComputeFluidVorticityVars),
};

const gpu::GpuComputePassDefinition kApplyFluidVorticityConfinement{
    "physics/fluid/solver/physics_fluid_apply_vorticity_confinement.cs.hlsl",
    "CRESSimNeo.Physics.ApplyFluidVorticityConfinement.CS",
    "CRESSimNeo.Physics.ApplyFluidVorticityConfinement.PSO",
    kApplyFluidVorticityConfinementVars,
    std::size(kApplyFluidVorticityConfinementVars),
};

const gpu::GpuComputePassDefinition kBuildFluidRenderAnisotropy{
    "physics/fluid/render/physics_fluid_build_render_anisotropy.cs.hlsl",
    "CRESSimNeo.Physics.BuildFluidRenderAnisotropy.CS",
    "CRESSimNeo.Physics.BuildFluidRenderAnisotropy.PSO",
    kBuildFluidRenderAnisotropyVars,
    std::size(kBuildFluidRenderAnisotropyVars),
};

const gpu::GpuComputePassDefinition kSolveParticleContactVelocities{
    "physics/particle/solver/physics_particle_solve_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.SolveParticleContactVelocities.CS",
    "CRESSimNeo.Physics.SolveParticleContactVelocities.PSO",
    kSolveParticleContactVelocitiesVars,
    std::size(kSolveParticleContactVelocitiesVars),
};

const gpu::GpuComputePassDefinition kSolveParticleRigidContactVelocities{
    "physics/particle/solver/physics_particle_rigid_solve_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.SolveParticleRigidContactVelocities.CS",
    "CRESSimNeo.Physics.SolveParticleRigidContactVelocities.PSO",
    kSolveParticleRigidContactVelocitiesVars,
    std::size(kSolveParticleRigidContactVelocitiesVars),
};

const gpu::GpuComputePassDefinition kApplyParticleContactVelocities{
    "physics/particle/solver/physics_particle_apply_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.ApplyParticleContactVelocities.CS",
    "CRESSimNeo.Physics.ApplyParticleContactVelocities.PSO",
    kApplyParticleContactVelocitiesVars,
    std::size(kApplyParticleContactVelocitiesVars),
};

const gpu::GpuComputePassDefinition kUpdateSoftTriangleNormals{
    "physics/soft/render/physics_soft_update_triangle_normals.cs.hlsl",
    "CRESSimNeo.Physics.UpdateSoftTriangleNormals.CS",
    "CRESSimNeo.Physics.UpdateSoftTriangleNormals.PSO",
    kUpdateSoftTriangleNormalsVars,
    std::size(kUpdateSoftTriangleNormalsVars),
};

const gpu::GpuComputePassDefinition kSkinSoftRenderVertices{
    "physics/soft/render/physics_soft_skin_render_vertices.cs.hlsl",
    "CRESSimNeo.Physics.SkinSoftRenderVertices.CS",
    "CRESSimNeo.Physics.SkinSoftRenderVertices.PSO",
    kSkinSoftRenderVerticesVars,
    std::size(kSkinSoftRenderVerticesVars),
};

const gpu::GpuComputePassDefinition kUpdateSoftRenderNormals{
    "physics/soft/render/physics_soft_update_render_normals.cs.hlsl",
    "CRESSimNeo.Physics.UpdateSoftRenderNormals.CS",
    "CRESSimNeo.Physics.UpdateSoftRenderNormals.PSO",
    kUpdateSoftRenderNormalsVars,
    std::size(kUpdateSoftRenderNormalsVars),
};

const gpu::GpuComputePassDefinition kUpdateCurveRenderData{
    "physics/curve/render/physics_curve_update_render_data.cs.hlsl",
    "CRESSimNeo.Physics.UpdateCurveRenderData.CS",
    "CRESSimNeo.Physics.UpdateCurveRenderData.PSO",
    kUpdateCurveRenderDataVars,
    std::size(kUpdateCurveRenderDataVars),
};

const gpu::GpuComputePassDefinition kUpdateSoftBodyBounds{
    "physics/soft/render/physics_soft_update_body_bounds.cs.hlsl",
    "CRESSimNeo.Physics.UpdateSoftBodyBounds.CS",
    "CRESSimNeo.Physics.UpdateSoftBodyBounds.PSO",
    kUpdateSoftBodyBoundsVars,
    std::size(kUpdateSoftBodyBoundsVars),
};

const gpu::GpuComputePassDefinition kFinalizeSoftBodyBounds{
    "physics/soft/render/physics_soft_finalize_body_bounds.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeSoftBodyBounds.CS",
    "CRESSimNeo.Physics.FinalizeSoftBodyBounds.PSO",
    kFinalizeSoftBodyBoundsVars,
    std::size(kFinalizeSoftBodyBoundsVars),
};

const gpu::GpuComputePassDefinition kUpdateRigidWorldAabbs{
    "physics/rigid/broad_phase/physics_rigid_update_world_aabbs.cs.hlsl",
    "CRESSimNeo.Physics.RigidUpdateWorldAabbs.CS",
    "CRESSimNeo.Physics.RigidUpdateWorldAabbs.PSO",
    kUpdateWorldAabbsVars,
    std::size(kUpdateWorldAabbsVars),
};

const gpu::GpuComputePassDefinition kScanBlock{
    "physics/shared/physics_scan_block.cs.hlsl",
    "CRESSimNeo.Physics.ScanBlock.CS",
    "CRESSimNeo.Physics.ScanBlock.PSO",
    kScanBlockVars,
    std::size(kScanBlockVars),
};

const gpu::GpuComputePassDefinition kScanAddOffsets{
    "physics/shared/physics_scan_add_offsets.cs.hlsl",
    "CRESSimNeo.Physics.ScanAddOffsets.CS",
    "CRESSimNeo.Physics.ScanAddOffsets.PSO",
    kScanAddOffsetsVars,
    std::size(kScanAddOffsetsVars),
};

const gpu::GpuComputePassDefinition kCompactBodySet{
    "physics/rigid/broad_phase/physics_rigid_compact_body_set.cs.hlsl",
    "CRESSimNeo.Physics.RigidCompactBodySet.CS",
    "CRESSimNeo.Physics.RigidCompactBodySet.PSO",
    kCompactBodySetVars,
    std::size(kCompactBodySetVars),
};

const gpu::GpuComputePassDefinition kFinalizeActiveBodies{
    "physics/rigid/broad_phase/physics_rigid_finalize_active_bodies.cs.hlsl",
    "CRESSimNeo.Physics.RigidFinalizeActiveBodies.CS",
    "CRESSimNeo.Physics.RigidFinalizeActiveBodies.PSO",
    kFinalizeActiveBodiesVars,
    std::size(kFinalizeActiveBodiesVars),
};

const gpu::GpuComputePassDefinition kBuildBroadPhaseElements{
    "physics/rigid/broad_phase/physics_rigid_build_broad_phase_elements.cs.hlsl",
    "CRESSimNeo.Physics.RigidBuildBroadPhaseElements.CS",
    "CRESSimNeo.Physics.RigidBuildBroadPhaseElements.PSO",
    kBuildBroadPhaseElementsVars,
    std::size(kBuildBroadPhaseElementsVars),
};

const gpu::GpuComputePassDefinition kReduceExtentElements{
    "physics/rigid/broad_phase/physics_rigid_reduce_extent_elements.cs.hlsl",
    "CRESSimNeo.Physics.RigidReduceExtentElements.CS",
    "CRESSimNeo.Physics.RigidReduceExtentElements.PSO",
    kReduceExtentElementsVars,
    std::size(kReduceExtentElementsVars),
};

const gpu::GpuComputePassDefinition kReduceExtentExtents{
    "physics/rigid/broad_phase/physics_rigid_reduce_extent_extents.cs.hlsl",
    "CRESSimNeo.Physics.RigidReduceExtentExtents.CS",
    "CRESSimNeo.Physics.RigidReduceExtentExtents.PSO",
    kReduceExtentExtentsVars,
    std::size(kReduceExtentExtentsVars),
};

const gpu::GpuComputePassDefinition kMortonCodes{
    "physics/rigid/broad_phase/physics_rigid_morton_codes.cs.hlsl",
    "CRESSimNeo.Physics.RigidMortonCodes.CS",
    "CRESSimNeo.Physics.RigidMortonCodes.PSO",
    kMortonCodesVars,
    std::size(kMortonCodesVars),
};

const gpu::GpuComputePassDefinition kRadixClassify{
    "physics/shared/physics_radix_classify.cs.hlsl",
    "CRESSimNeo.Physics.RigidRadixClassify.CS",
    "CRESSimNeo.Physics.RigidRadixClassify.PSO",
    kRadixClassifyVars,
    std::size(kRadixClassifyVars),
};

const gpu::GpuComputePassDefinition kRadixFinalize{
    "physics/shared/physics_radix_finalize.cs.hlsl",
    "CRESSimNeo.Physics.RigidRadixFinalize.CS",
    "CRESSimNeo.Physics.RigidRadixFinalize.PSO",
    kRadixFinalizeVars,
    std::size(kRadixFinalizeVars),
};

const gpu::GpuComputePassDefinition kRadixScatter{
    "physics/shared/physics_radix_scatter.cs.hlsl",
    "CRESSimNeo.Physics.RigidRadixScatter.CS",
    "CRESSimNeo.Physics.RigidRadixScatter.PSO",
    kRadixScatterVars,
    std::size(kRadixScatterVars),
};

const gpu::GpuComputePassDefinition kBvhHierarchy{
    "physics/rigid/broad_phase/physics_rigid_bvh_hierarchy.cs.hlsl",
    "CRESSimNeo.Physics.RigidBvhHierarchy.CS",
    "CRESSimNeo.Physics.RigidBvhHierarchy.PSO",
    kBvhHierarchyVars,
    std::size(kBvhHierarchyVars),
};
const gpu::GpuComputePassDefinition kBvhBoundingBoxes{
    "physics/rigid/broad_phase/physics_rigid_bvh_bounding_boxes.cs.hlsl",
    "CRESSimNeo.Physics.RigidBvhBoundingBoxes.CS",
    "CRESSimNeo.Physics.RigidBvhBoundingBoxes.PSO",
    kBvhBoundingBoxesVars,
    std::size(kBvhBoundingBoxesVars),
};

const gpu::GpuComputePassDefinition kCountPairs{
    "physics/rigid/broad_phase/physics_rigid_count_pairs.cs.hlsl",
    "CRESSimNeo.Physics.RigidCountPairs.CS",
    "CRESSimNeo.Physics.RigidCountPairs.PSO",
    kCountPairsVars,
    std::size(kCountPairsVars),
};

const gpu::GpuComputePassDefinition kFinalizePairs{
    "physics/rigid/broad_phase/physics_rigid_finalize_pairs.cs.hlsl",
    "CRESSimNeo.Physics.RigidFinalizePairs.CS",
    "CRESSimNeo.Physics.RigidFinalizePairs.PSO",
    kFinalizePairsVars,
    std::size(kFinalizePairsVars),
};

const gpu::GpuComputePassDefinition kEmitPairs{
    "physics/rigid/broad_phase/physics_rigid_emit_pairs.cs.hlsl",
    "CRESSimNeo.Physics.RigidEmitPairs.CS",
    "CRESSimNeo.Physics.RigidEmitPairs.PSO",
    kEmitPairsVars,
    std::size(kEmitPairsVars),
};

const gpu::GpuComputePassDefinition kBuildNarrowPhaseChunks{
    "physics/rigid/narrow_phase/physics_rigid_build_narrow_phase_chunks.cs.hlsl",
    "CRESSimNeo.Physics.RigidBuildNarrowPhaseChunks.CS",
    "CRESSimNeo.Physics.RigidBuildNarrowPhaseChunks.PSO",
    kBuildNarrowPhaseChunksVars,
    std::size(kBuildNarrowPhaseChunksVars),
};

const gpu::GpuComputePassDefinition kPrepareRigidIndirectArgs{
    "physics/shared/physics_prepare_rigid_indirect_args.cs.hlsl",
    "CRESSimNeo.Physics.PrepareRigidIndirectArgs.CS",
    "CRESSimNeo.Physics.PrepareRigidIndirectArgs.PSO",
    kPrepareRigidIndirectArgsVars,
    std::size(kPrepareRigidIndirectArgsVars),
};

const gpu::GpuComputePassDefinition kGenerateRigidContacts{
    "physics/rigid/narrow_phase/physics_rigid_generate_contacts.cs.hlsl",
    "CRESSimNeo.Physics.RigidGenerateContacts.CS",
    "CRESSimNeo.Physics.RigidGenerateContacts.PSO",
    kGenerateContactsVars,
    std::size(kGenerateContactsVars),
};

const gpu::GpuComputePassDefinition kGenerateProxyRigidContacts{
    "physics/rigid/narrow_phase/physics_rigid_generate_proxy_contacts.cs.hlsl",
    "CRESSimNeo.Physics.RigidGenerateProxyContacts.CS",
    "CRESSimNeo.Physics.RigidGenerateProxyContacts.PSO",
    kGenerateProxyRigidContactsVars,
    std::size(kGenerateProxyRigidContactsVars),
};

const gpu::GpuComputePassDefinition kFinalRigidContactDepenetration{
    "physics/rigid/solver/physics_rigid_depenetrate_contacts.cs.hlsl",
    "CRESSimNeo.Physics.RigidFinalContactDepenetration.CS",
    "CRESSimNeo.Physics.RigidFinalContactDepenetration.PSO",
    kFinalRigidContactDepenetrationVars,
    std::size(kFinalRigidContactDepenetrationVars),
};

const gpu::GpuComputePassDefinition kClearRigidBodyPairContactAggregates{
    "physics/rigid/solver/physics_rigid_clear_body_pair_contact_aggregates.cs.hlsl",
    "CRESSimNeo.Physics.RigidClearBodyPairContactAggregates.CS",
    "CRESSimNeo.Physics.RigidClearBodyPairContactAggregates.PSO",
    kClearRigidBodyPairContactAggregatesVars,
    std::size(kClearRigidBodyPairContactAggregatesVars),
};

const gpu::GpuComputePassDefinition kInitRigidContactVelocities{
    "physics/rigid/solver/physics_rigid_init_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidInitContactVelocities.CS",
    "CRESSimNeo.Physics.RigidInitContactVelocities.PSO",
    kInitRigidContactVelocitiesVars,
    std::size(kInitRigidContactVelocitiesVars),
};

const gpu::GpuComputePassDefinition kPrepareRigidContactVelocityIndirectArgs{
    "physics/shared/physics_prepare_rigid_contact_velocity_indirect_args.cs.hlsl",
    "CRESSimNeo.Physics.PrepareRigidContactVelocityIndirectArgs.CS",
    "CRESSimNeo.Physics.PrepareRigidContactVelocityIndirectArgs.PSO",
    kPrepareRigidContactVelocityIndirectArgsVars,
    std::size(kPrepareRigidContactVelocityIndirectArgsVars),
};

const gpu::GpuComputePassDefinition kSolveRigidContactVelocities{
    "physics/rigid/solver/physics_rigid_solve_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveContactVelocities.CS",
    "CRESSimNeo.Physics.RigidSolveContactVelocities.PSO",
    kSolveRigidContactVelocitiesVars,
    std::size(kSolveRigidContactVelocitiesVars),
};

const gpu::GpuComputePassDefinition kSolveBallJointConstraints{
    "physics/rigid/solver/physics_rigid_solve_ball_joints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveBallJointConstraints.CS",
    "CRESSimNeo.Physics.RigidSolveBallJointConstraints.PSO",
    kSolveBallJointConstraintsVars,
    std::size(kSolveBallJointConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveSphericalJointConstraints{
    "physics/rigid/solver/physics_rigid_solve_spherical_joints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveSphericalJointConstraints.CS",
    "CRESSimNeo.Physics.RigidSolveSphericalJointConstraints.PSO",
    kSolveSphericalJointConstraintsVars,
    std::size(kSolveSphericalJointConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveHingeJointConstraintsPassive{
    "physics/rigid/solver/physics_rigid_solve_hinge_joints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveHingeJointConstraintsPassive.CS",
    "CRESSimNeo.Physics.RigidSolveHingeJointConstraintsPassive.PSO",
    kSolveHingeJointConstraintsVars,
    std::size(kSolveHingeJointConstraintsVars),
    kJointDriveModePassiveMacros,
    std::size(kJointDriveModePassiveMacros),
};

const gpu::GpuComputePassDefinition kSolveHingeJointConstraintsTargetPosition{
    "physics/rigid/solver/physics_rigid_solve_hinge_joints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveHingeJointConstraintsTargetPosition.CS",
    "CRESSimNeo.Physics.RigidSolveHingeJointConstraintsTargetPosition.PSO",
    kSolveHingeJointConstraintsVars,
    std::size(kSolveHingeJointConstraintsVars),
    kJointDriveModeTargetPositionMacros,
    std::size(kJointDriveModeTargetPositionMacros),
};

const gpu::GpuComputePassDefinition kSolveSliderJointConstraintsPassive{
    "physics/rigid/solver/physics_rigid_solve_slider_joints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveSliderJointConstraintsPassive.CS",
    "CRESSimNeo.Physics.RigidSolveSliderJointConstraintsPassive.PSO",
    kSolveSliderJointConstraintsVars,
    std::size(kSolveSliderJointConstraintsVars),
    kJointDriveModePassiveMacros,
    std::size(kJointDriveModePassiveMacros),
};

const gpu::GpuComputePassDefinition kSolveSliderJointConstraintsTargetPosition{
    "physics/rigid/solver/physics_rigid_solve_slider_joints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveSliderJointConstraintsTargetPosition.CS",
    "CRESSimNeo.Physics.RigidSolveSliderJointConstraintsTargetPosition.PSO",
    kSolveSliderJointConstraintsVars,
    std::size(kSolveSliderJointConstraintsVars),
    kJointDriveModeTargetPositionMacros,
    std::size(kJointDriveModeTargetPositionMacros),
};

const gpu::GpuComputePassDefinition kSolveRoutedCableConstraints{
    "physics/rigid/solver/physics_rigid_solve_routed_cable_constraints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveRoutedCableConstraints.CS",
    "CRESSimNeo.Physics.RigidSolveRoutedCableConstraints.PSO",
    kSolveRoutedCableConstraintsVars,
    std::size(kSolveRoutedCableConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveRigidDistanceConstraints{
    "physics/rigid/solver/physics_rigid_solve_rigid_distance_constraints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveRigidDistanceConstraints.CS",
    "CRESSimNeo.Physics.RigidSolveRigidDistanceConstraints.PSO",
    kSolveRigidDistanceConstraintsVars,
    std::size(kSolveRigidDistanceConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveRigidParticleAttachmentConstraints{
    "physics/rigid/solver/physics_rigid_solve_rigid_particle_attachment_constraints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveRigidParticleAttachmentConstraints.CS",
    "CRESSimNeo.Physics.RigidSolveRigidParticleAttachmentConstraints.PSO",
    kSolveRigidParticleAttachmentConstraintsVars,
    std::size(kSolveRigidParticleAttachmentConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveStrandRigidAttachmentConstraints{
    "physics/rigid/solver/physics_rigid_solve_strand_rigid_attachment_constraints.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveStrandRigidAttachmentConstraints.CS",
    "CRESSimNeo.Physics.RigidSolveStrandRigidAttachmentConstraints.PSO",
    kSolveStrandRigidAttachmentConstraintsVars,
    std::size(kSolveStrandRigidAttachmentConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveHingeJointTargetVelocities{
    "physics/rigid/solver/physics_rigid_solve_hinge_joint_target_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveHingeJointTargetVelocities.CS",
    "CRESSimNeo.Physics.RigidSolveHingeJointTargetVelocities.PSO",
    kSolveHingeJointTargetVelocitiesVars,
    std::size(kSolveHingeJointTargetVelocitiesVars),
};

const gpu::GpuComputePassDefinition kSolveSliderJointTargetVelocities{
    "physics/rigid/solver/physics_rigid_solve_slider_joint_target_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveSliderJointTargetVelocities.CS",
    "CRESSimNeo.Physics.RigidSolveSliderJointTargetVelocities.PSO",
    kSolveSliderJointTargetVelocitiesVars,
    std::size(kSolveSliderJointTargetVelocitiesVars),
};

const gpu::GpuComputePassDefinition kClearRigidCorrections{
    "physics/rigid/solver/physics_rigid_clear_corrections.cs.hlsl",
    "CRESSimNeo.Physics.RigidClearCorrections.CS",
    "CRESSimNeo.Physics.RigidClearCorrections.PSO",
    kClearCorrectionsVars,
    std::size(kClearCorrectionsVars),
};

const gpu::GpuComputePassDefinition kApplyRigidCorrections{
    "physics/rigid/solver/physics_rigid_apply_corrections.cs.hlsl",
    "CRESSimNeo.Physics.RigidApplyCorrections.CS",
    "CRESSimNeo.Physics.RigidApplyCorrections.PSO",
    kApplyCorrectionsVars,
    std::size(kApplyCorrectionsVars),
};

const gpu::GpuComputePassDefinition kUpdateRigidVelocities{
    "physics/rigid/solver/physics_rigid_update_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidUpdateVelocities.CS",
    "CRESSimNeo.Physics.RigidUpdateVelocities.PSO",
    kUpdateVelocitiesVars,
    std::size(kUpdateVelocitiesVars),
};

const gpu::GpuComputePassDefinition kApplyRigidContactVelocities{
    "physics/rigid/solver/physics_rigid_apply_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidApplyContactVelocities.CS",
    "CRESSimNeo.Physics.RigidApplyContactVelocities.PSO",
    kApplyRigidContactVelocitiesVars,
    std::size(kApplyRigidContactVelocitiesVars),
};

} // namespace cressim::neo::physics::passdefs
