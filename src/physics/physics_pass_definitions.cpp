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
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePreviousPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildParticleBroadPhaseEntriesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleOwningSoftBodyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseEntries",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildParticleBroadPhaseKeysVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseEntries",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kMarkParticleCellRangeStartsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SortedParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRangeStartFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearParticleCellRangesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kBuildParticleCellRangesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
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

constexpr Diligent::ShaderResourceVariableDesc kCountSoftSoftCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseEntries",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SortedParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleAdjacencyOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleAdjacencyCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleAdjacencyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kFinalizeSoftCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kEmitSoftSoftCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleBroadPhaseEntries",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleCellRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SortedParticleBroadPhaseKeys",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleAdjacencyOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleAdjacencyCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleAdjacencyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kCountSoftRigidCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyColliderRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kEmitSoftRigidCandidatePairsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleBroadPhaseMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_StaticBvhNodes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BodyColliderRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CandidateOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kGenerateSoftRigidContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleBroadPhaseMetadata",
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
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderShapeParams",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderLocalPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderLocalOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderBroadPhaseData",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kGenerateSoftContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleRadii",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftCandidatePairs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kPrepareSoftIndirectArgsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PhysicsIndirectDispatchArgs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kFinalizeActiveContactVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kCompactActiveSoftContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveSoftContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kCompactActiveSoftRigidContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ContactActiveOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveSoftRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kClearSoftConstraintStateVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftPositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdgeLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTetLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSoftEdgeConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdges", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdgeLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdgeCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSoftTetConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTets", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTetLambdas",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTetCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplySoftEdgeCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleEdgeRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleIncidentEdges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftEdgeCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplySoftTetCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleTetRanges",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ParticleIncidentTets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftTetCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSoftRigidContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftPositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSoftContactsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePreviousPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftPositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplySoftPositionCorrectionsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftPositionCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateSoftVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePreviousPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kSolveSoftRigidContactVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleMaterials",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleVelocities",
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
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftNeighborMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kApplySoftContactVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleVelocities",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticleVelocityCorrections",
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
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRenderTriangleParticleIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftRenderTriangleNormalsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr Diligent::ShaderResourceVariableDesc kUpdateSoftBodyBoundsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsSoftRenderDispatchConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftParticlePositionsInvMass",
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
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderShapeParams",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderLocalPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderLocalOrientations",
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
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsScanConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanInput", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanOutput",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BlockSums", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr Diligent::ShaderResourceVariableDesc kScanAddOffsetsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "PhysicsScanConstantsBuffer",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScannedBlockOffsets",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ScanOutput",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
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
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderOwnerRigidBodyIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderShapeTypes",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderShapeParams",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderLocalPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderLocalOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_ColliderMaterials",
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

constexpr Diligent::ShaderResourceVariableDesc kSolveRigidContactConstraintsVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
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

constexpr Diligent::ShaderResourceVariableDesc kSolveRigidContactVelocitiesVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
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
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidContacts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocityCorrections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocityCorrections",
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
    "physics/physics_rigid_predict.cs.hlsl",
    "CRESSimNeo.Physics.RigidPredict.CS",
    "CRESSimNeo.Physics.RigidPredict.PSO",
    kPredictVars,
    std::size(kPredictVars),
};

const gpu::GpuComputePassDefinition kSoftPredict{
    "physics/physics_soft_predict.cs.hlsl",
    "CRESSimNeo.Physics.SoftPredict.CS",
    "CRESSimNeo.Physics.SoftPredict.PSO",
    kSoftPredictVars,
    std::size(kSoftPredictVars),
};

const gpu::GpuComputePassDefinition kBuildParticleBroadPhaseEntries{
    "physics/physics_soft_rigid_build_broad_phase_particles.cs.hlsl",
    "CRESSimNeo.Physics.ParticleBuildBroadPhaseEntries.CS",
    "CRESSimNeo.Physics.ParticleBuildBroadPhaseEntries.PSO",
    kBuildParticleBroadPhaseEntriesVars,
    std::size(kBuildParticleBroadPhaseEntriesVars),
};

const gpu::GpuComputePassDefinition kBuildParticleBroadPhaseKeys{
    "physics/physics_soft_rigid_build_broad_phase_keys.cs.hlsl",
    "CRESSimNeo.Physics.ParticleBuildBroadPhaseKeys.CS",
    "CRESSimNeo.Physics.ParticleBuildBroadPhaseKeys.PSO",
    kBuildParticleBroadPhaseKeysVars,
    std::size(kBuildParticleBroadPhaseKeysVars),
};

const gpu::GpuComputePassDefinition kMarkParticleCellRangeStarts{
    "physics/physics_soft_rigid_mark_cell_range_starts.cs.hlsl",
    "CRESSimNeo.Physics.ParticleMarkCellRangeStarts.CS",
    "CRESSimNeo.Physics.ParticleMarkCellRangeStarts.PSO",
    kMarkParticleCellRangeStartsVars,
    std::size(kMarkParticleCellRangeStartsVars),
};

const gpu::GpuComputePassDefinition kClearParticleCellRanges{
    "physics/physics_soft_rigid_clear_cell_ranges.cs.hlsl",
    "CRESSimNeo.Physics.ParticleClearCellRanges.CS",
    "CRESSimNeo.Physics.ParticleClearCellRanges.PSO",
    kClearParticleCellRangesVars,
    std::size(kClearParticleCellRangesVars),
};

const gpu::GpuComputePassDefinition kBuildParticleCellRanges{
    "physics/physics_soft_rigid_build_cell_ranges.cs.hlsl",
    "CRESSimNeo.Physics.ParticleBuildCellRanges.CS",
    "CRESSimNeo.Physics.ParticleBuildCellRanges.PSO",
    kBuildParticleCellRangesVars,
    std::size(kBuildParticleCellRangesVars),
};

const gpu::GpuComputePassDefinition kCountSoftSoftCandidatePairs{
    "physics/physics_soft_count_soft_soft_candidates.cs.hlsl",
    "CRESSimNeo.Physics.CountSoftSoftCandidatePairs.CS",
    "CRESSimNeo.Physics.CountSoftSoftCandidatePairs.PSO",
    kCountSoftSoftCandidatePairsVars,
    std::size(kCountSoftSoftCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kFinalizeSoftSoftCandidatePairs{
    "physics/physics_soft_finalize_soft_soft_candidates.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeSoftSoftCandidatePairs.CS",
    "CRESSimNeo.Physics.FinalizeSoftSoftCandidatePairs.PSO",
    kFinalizeSoftCandidatePairsVars,
    std::size(kFinalizeSoftCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kEmitSoftSoftCandidatePairs{
    "physics/physics_soft_emit_soft_soft_candidates.cs.hlsl",
    "CRESSimNeo.Physics.EmitSoftSoftCandidatePairs.CS",
    "CRESSimNeo.Physics.EmitSoftSoftCandidatePairs.PSO",
    kEmitSoftSoftCandidatePairsVars,
    std::size(kEmitSoftSoftCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kCountSoftRigidCandidatePairs{
    "physics/physics_soft_count_soft_rigid_candidates.cs.hlsl",
    "CRESSimNeo.Physics.CountSoftRigidCandidatePairs.CS",
    "CRESSimNeo.Physics.CountSoftRigidCandidatePairs.PSO",
    kCountSoftRigidCandidatePairsVars,
    std::size(kCountSoftRigidCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kFinalizeSoftRigidCandidatePairs{
    "physics/physics_soft_finalize_soft_rigid_candidates.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeSoftRigidCandidatePairs.CS",
    "CRESSimNeo.Physics.FinalizeSoftRigidCandidatePairs.PSO",
    kFinalizeSoftCandidatePairsVars,
    std::size(kFinalizeSoftCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kEmitSoftRigidCandidatePairs{
    "physics/physics_soft_emit_soft_rigid_candidates.cs.hlsl",
    "CRESSimNeo.Physics.EmitSoftRigidCandidatePairs.CS",
    "CRESSimNeo.Physics.EmitSoftRigidCandidatePairs.PSO",
    kEmitSoftRigidCandidatePairsVars,
    std::size(kEmitSoftRigidCandidatePairsVars),
};

const gpu::GpuComputePassDefinition kGenerateSoftRigidContacts{
    "physics/physics_soft_rigid_generate_contacts.cs.hlsl",
    "CRESSimNeo.Physics.GenerateSoftRigidContacts.CS",
    "CRESSimNeo.Physics.GenerateSoftRigidContacts.PSO",
    kGenerateSoftRigidContactsVars,
    std::size(kGenerateSoftRigidContactsVars),
};

const gpu::GpuComputePassDefinition kGenerateSoftContacts{
    "physics/physics_soft_generate_contacts.cs.hlsl",
    "CRESSimNeo.Physics.GenerateSoftContacts.CS",
    "CRESSimNeo.Physics.GenerateSoftContacts.PSO",
    kGenerateSoftContactsVars,
    std::size(kGenerateSoftContactsVars),
};

const gpu::GpuComputePassDefinition kPrepareSoftCandidateIndirectArgs{
    "physics/physics_prepare_soft_candidate_indirect_args.cs.hlsl",
    "CRESSimNeo.Physics.PrepareSoftCandidateIndirectArgs.CS",
    "CRESSimNeo.Physics.PrepareSoftCandidateIndirectArgs.PSO",
    kPrepareSoftIndirectArgsVars,
    std::size(kPrepareSoftIndirectArgsVars),
};

const gpu::GpuComputePassDefinition kPrepareSoftActiveIndirectArgs{
    "physics/physics_prepare_soft_active_indirect_args.cs.hlsl",
    "CRESSimNeo.Physics.PrepareSoftActiveIndirectArgs.CS",
    "CRESSimNeo.Physics.PrepareSoftActiveIndirectArgs.PSO",
    kPrepareSoftIndirectArgsVars,
    std::size(kPrepareSoftIndirectArgsVars),
};

const gpu::GpuComputePassDefinition kFinalizeActiveSoftContacts{
    "physics/physics_soft_finalize_active_contacts.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeActiveSoftContacts.CS",
    "CRESSimNeo.Physics.FinalizeActiveSoftContacts.PSO",
    kFinalizeActiveContactVars,
    std::size(kFinalizeActiveContactVars),
};

const gpu::GpuComputePassDefinition kCompactActiveSoftContacts{
    "physics/physics_soft_compact_active_contacts.cs.hlsl",
    "CRESSimNeo.Physics.CompactActiveSoftContacts.CS",
    "CRESSimNeo.Physics.CompactActiveSoftContacts.PSO",
    kCompactActiveSoftContactsVars,
    std::size(kCompactActiveSoftContactsVars),
};

const gpu::GpuComputePassDefinition kFinalizeActiveSoftRigidContacts{
    "physics/physics_soft_finalize_active_rigid_contacts.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeActiveSoftRigidContacts.CS",
    "CRESSimNeo.Physics.FinalizeActiveSoftRigidContacts.PSO",
    kFinalizeActiveContactVars,
    std::size(kFinalizeActiveContactVars),
};

const gpu::GpuComputePassDefinition kCompactActiveSoftRigidContacts{
    "physics/physics_soft_compact_active_rigid_contacts.cs.hlsl",
    "CRESSimNeo.Physics.CompactActiveSoftRigidContacts.CS",
    "CRESSimNeo.Physics.CompactActiveSoftRigidContacts.PSO",
    kCompactActiveSoftRigidContactsVars,
    std::size(kCompactActiveSoftRigidContactsVars),
};

const gpu::GpuComputePassDefinition kClearSoftConstraintState{
    "physics/physics_clear_soft_constraint_state.cs.hlsl",
    "CRESSimNeo.Physics.ClearSoftConstraintState.CS",
    "CRESSimNeo.Physics.ClearSoftConstraintState.PSO",
    kClearSoftConstraintStateVars,
    std::size(kClearSoftConstraintStateVars),
};

const gpu::GpuComputePassDefinition kSolveSoftEdgeConstraints{
    "physics/physics_soft_solve_edge_constraints.cs.hlsl",
    "CRESSimNeo.Physics.SolveSoftEdgeConstraints.CS",
    "CRESSimNeo.Physics.SolveSoftEdgeConstraints.PSO",
    kSolveSoftEdgeConstraintsVars,
    std::size(kSolveSoftEdgeConstraintsVars),
};

const gpu::GpuComputePassDefinition kSolveSoftTetConstraints{
    "physics/physics_soft_solve_tet_constraints.cs.hlsl",
    "CRESSimNeo.Physics.SolveSoftTetConstraints.CS",
    "CRESSimNeo.Physics.SolveSoftTetConstraints.PSO",
    kSolveSoftTetConstraintsVars,
    std::size(kSolveSoftTetConstraintsVars),
};

const gpu::GpuComputePassDefinition kApplySoftEdgeCorrections{
    "physics/physics_soft_apply_edge_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplySoftEdgeCorrections.CS",
    "CRESSimNeo.Physics.ApplySoftEdgeCorrections.PSO",
    kApplySoftEdgeCorrectionsVars,
    std::size(kApplySoftEdgeCorrectionsVars),
};

const gpu::GpuComputePassDefinition kApplySoftTetCorrections{
    "physics/physics_soft_apply_tet_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplySoftTetCorrections.CS",
    "CRESSimNeo.Physics.ApplySoftTetCorrections.PSO",
    kApplySoftTetCorrectionsVars,
    std::size(kApplySoftTetCorrectionsVars),
};

const gpu::GpuComputePassDefinition kSolveSoftRigidContacts{
    "physics/physics_soft_rigid_solve_contacts.cs.hlsl",
    "CRESSimNeo.Physics.SolveSoftRigidContacts.CS",
    "CRESSimNeo.Physics.SolveSoftRigidContacts.PSO",
    kSolveSoftRigidContactsVars,
    std::size(kSolveSoftRigidContactsVars),
};

const gpu::GpuComputePassDefinition kSolveSoftContacts{
    "physics/physics_soft_solve_contacts.cs.hlsl",
    "CRESSimNeo.Physics.SolveSoftContacts.CS",
    "CRESSimNeo.Physics.SolveSoftContacts.PSO",
    kSolveSoftContactsVars,
    std::size(kSolveSoftContactsVars),
};

const gpu::GpuComputePassDefinition kApplySoftPositionCorrections{
    "physics/physics_soft_apply_position_corrections.cs.hlsl",
    "CRESSimNeo.Physics.ApplySoftPositionCorrections.CS",
    "CRESSimNeo.Physics.ApplySoftPositionCorrections.PSO",
    kApplySoftPositionCorrectionsVars,
    std::size(kApplySoftPositionCorrectionsVars),
};

const gpu::GpuComputePassDefinition kUpdateSoftVelocities{
    "physics/physics_soft_update_velocities.cs.hlsl",
    "CRESSimNeo.Physics.UpdateSoftVelocities.CS",
    "CRESSimNeo.Physics.UpdateSoftVelocities.PSO",
    kUpdateSoftVelocitiesVars,
    std::size(kUpdateSoftVelocitiesVars),
};

const gpu::GpuComputePassDefinition kSolveSoftRigidContactVelocities{
    "physics/physics_soft_rigid_solve_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.SolveSoftRigidContactVelocities.CS",
    "CRESSimNeo.Physics.SolveSoftRigidContactVelocities.PSO",
    kSolveSoftRigidContactVelocitiesVars,
    std::size(kSolveSoftRigidContactVelocitiesVars),
};

const gpu::GpuComputePassDefinition kApplySoftContactVelocities{
    "physics/physics_soft_apply_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.ApplySoftContactVelocities.CS",
    "CRESSimNeo.Physics.ApplySoftContactVelocities.PSO",
    kApplySoftContactVelocitiesVars,
    std::size(kApplySoftContactVelocitiesVars),
};

const gpu::GpuComputePassDefinition kUpdateSoftTriangleNormals{
    "physics/physics_soft_update_triangle_normals.cs.hlsl",
    "CRESSimNeo.Physics.UpdateSoftTriangleNormals.CS",
    "CRESSimNeo.Physics.UpdateSoftTriangleNormals.PSO",
    kUpdateSoftTriangleNormalsVars,
    std::size(kUpdateSoftTriangleNormalsVars),
};

const gpu::GpuComputePassDefinition kUpdateSoftRenderNormals{
    "physics/physics_soft_update_render_normals.cs.hlsl",
    "CRESSimNeo.Physics.UpdateSoftRenderNormals.CS",
    "CRESSimNeo.Physics.UpdateSoftRenderNormals.PSO",
    kUpdateSoftRenderNormalsVars,
    std::size(kUpdateSoftRenderNormalsVars),
};

const gpu::GpuComputePassDefinition kUpdateSoftBodyBounds{
    "physics/physics_soft_update_body_bounds.cs.hlsl",
    "CRESSimNeo.Physics.UpdateSoftBodyBounds.CS",
    "CRESSimNeo.Physics.UpdateSoftBodyBounds.PSO",
    kUpdateSoftBodyBoundsVars,
    std::size(kUpdateSoftBodyBoundsVars),
};

const gpu::GpuComputePassDefinition kFinalizeSoftBodyBounds{
    "physics/physics_soft_finalize_body_bounds.cs.hlsl",
    "CRESSimNeo.Physics.FinalizeSoftBodyBounds.CS",
    "CRESSimNeo.Physics.FinalizeSoftBodyBounds.PSO",
    kFinalizeSoftBodyBoundsVars,
    std::size(kFinalizeSoftBodyBoundsVars),
};

const gpu::GpuComputePassDefinition kUpdateRigidWorldAabbs{
    "physics/physics_rigid_update_world_aabbs.cs.hlsl",
    "CRESSimNeo.Physics.RigidUpdateWorldAabbs.CS",
    "CRESSimNeo.Physics.RigidUpdateWorldAabbs.PSO",
    kUpdateWorldAabbsVars,
    std::size(kUpdateWorldAabbsVars),
};

const gpu::GpuComputePassDefinition kScanBlock{
    "physics/physics_scan_block.cs.hlsl",
    "CRESSimNeo.Physics.ScanBlock.CS",
    "CRESSimNeo.Physics.ScanBlock.PSO",
    kScanBlockVars,
    std::size(kScanBlockVars),
};

const gpu::GpuComputePassDefinition kScanAddOffsets{
    "physics/physics_scan_add_offsets.cs.hlsl",
    "CRESSimNeo.Physics.ScanAddOffsets.CS",
    "CRESSimNeo.Physics.ScanAddOffsets.PSO",
    kScanAddOffsetsVars,
    std::size(kScanAddOffsetsVars),
};

const gpu::GpuComputePassDefinition kCompactBodySet{
    "physics/physics_rigid_compact_body_set.cs.hlsl",
    "CRESSimNeo.Physics.RigidCompactBodySet.CS",
    "CRESSimNeo.Physics.RigidCompactBodySet.PSO",
    kCompactBodySetVars,
    std::size(kCompactBodySetVars),
};

const gpu::GpuComputePassDefinition kFinalizeActiveBodies{
    "physics/physics_rigid_finalize_active_bodies.cs.hlsl",
    "CRESSimNeo.Physics.RigidFinalizeActiveBodies.CS",
    "CRESSimNeo.Physics.RigidFinalizeActiveBodies.PSO",
    kFinalizeActiveBodiesVars,
    std::size(kFinalizeActiveBodiesVars),
};

const gpu::GpuComputePassDefinition kBuildBroadPhaseElements{
    "physics/physics_rigid_build_broad_phase_elements.cs.hlsl",
    "CRESSimNeo.Physics.RigidBuildBroadPhaseElements.CS",
    "CRESSimNeo.Physics.RigidBuildBroadPhaseElements.PSO",
    kBuildBroadPhaseElementsVars,
    std::size(kBuildBroadPhaseElementsVars),
};

const gpu::GpuComputePassDefinition kReduceExtentElements{
    "physics/physics_rigid_reduce_extent_elements.cs.hlsl",
    "CRESSimNeo.Physics.RigidReduceExtentElements.CS",
    "CRESSimNeo.Physics.RigidReduceExtentElements.PSO",
    kReduceExtentElementsVars,
    std::size(kReduceExtentElementsVars),
};

const gpu::GpuComputePassDefinition kReduceExtentExtents{
    "physics/physics_rigid_reduce_extent_extents.cs.hlsl",
    "CRESSimNeo.Physics.RigidReduceExtentExtents.CS",
    "CRESSimNeo.Physics.RigidReduceExtentExtents.PSO",
    kReduceExtentExtentsVars,
    std::size(kReduceExtentExtentsVars),
};

const gpu::GpuComputePassDefinition kMortonCodes{
    "physics/physics_rigid_morton_codes.cs.hlsl",
    "CRESSimNeo.Physics.RigidMortonCodes.CS",
    "CRESSimNeo.Physics.RigidMortonCodes.PSO",
    kMortonCodesVars,
    std::size(kMortonCodesVars),
};

const gpu::GpuComputePassDefinition kRadixClassify{
    "physics/physics_radix_classify.cs.hlsl",
    "CRESSimNeo.Physics.RigidRadixClassify.CS",
    "CRESSimNeo.Physics.RigidRadixClassify.PSO",
    kRadixClassifyVars,
    std::size(kRadixClassifyVars),
};

const gpu::GpuComputePassDefinition kRadixFinalize{
    "physics/physics_radix_finalize.cs.hlsl",
    "CRESSimNeo.Physics.RigidRadixFinalize.CS",
    "CRESSimNeo.Physics.RigidRadixFinalize.PSO",
    kRadixFinalizeVars,
    std::size(kRadixFinalizeVars),
};

const gpu::GpuComputePassDefinition kRadixScatter{
    "physics/physics_radix_scatter.cs.hlsl",
    "CRESSimNeo.Physics.RigidRadixScatter.CS",
    "CRESSimNeo.Physics.RigidRadixScatter.PSO",
    kRadixScatterVars,
    std::size(kRadixScatterVars),
};

const gpu::GpuComputePassDefinition kBvhHierarchy{
    "physics/physics_rigid_bvh_hierarchy.cs.hlsl",
    "CRESSimNeo.Physics.RigidBvhHierarchy.CS",
    "CRESSimNeo.Physics.RigidBvhHierarchy.PSO",
    kBvhHierarchyVars,
    std::size(kBvhHierarchyVars),
};
const gpu::GpuComputePassDefinition kBvhBoundingBoxes{
    "physics/physics_rigid_bvh_bounding_boxes.cs.hlsl",
    "CRESSimNeo.Physics.RigidBvhBoundingBoxes.CS",
    "CRESSimNeo.Physics.RigidBvhBoundingBoxes.PSO",
    kBvhBoundingBoxesVars,
    std::size(kBvhBoundingBoxesVars),
};

const gpu::GpuComputePassDefinition kCountPairs{
    "physics/physics_rigid_count_pairs.cs.hlsl",
    "CRESSimNeo.Physics.RigidCountPairs.CS",
    "CRESSimNeo.Physics.RigidCountPairs.PSO",
    kCountPairsVars,
    std::size(kCountPairsVars),
};

const gpu::GpuComputePassDefinition kFinalizePairs{
    "physics/physics_rigid_finalize_pairs.cs.hlsl",
    "CRESSimNeo.Physics.RigidFinalizePairs.CS",
    "CRESSimNeo.Physics.RigidFinalizePairs.PSO",
    kFinalizePairsVars,
    std::size(kFinalizePairsVars),
};

const gpu::GpuComputePassDefinition kEmitPairs{
    "physics/physics_rigid_emit_pairs.cs.hlsl",
    "CRESSimNeo.Physics.RigidEmitPairs.CS",
    "CRESSimNeo.Physics.RigidEmitPairs.PSO",
    kEmitPairsVars,
    std::size(kEmitPairsVars),
};

const gpu::GpuComputePassDefinition kBuildNarrowPhaseChunks{
    "physics/physics_rigid_build_narrow_phase_chunks.cs.hlsl",
    "CRESSimNeo.Physics.RigidBuildNarrowPhaseChunks.CS",
    "CRESSimNeo.Physics.RigidBuildNarrowPhaseChunks.PSO",
    kBuildNarrowPhaseChunksVars,
    std::size(kBuildNarrowPhaseChunksVars),
};

const gpu::GpuComputePassDefinition kPrepareRigidIndirectArgs{
    "physics/physics_prepare_rigid_indirect_args.cs.hlsl",
    "CRESSimNeo.Physics.PrepareRigidIndirectArgs.CS",
    "CRESSimNeo.Physics.PrepareRigidIndirectArgs.PSO",
    kPrepareRigidIndirectArgsVars,
    std::size(kPrepareRigidIndirectArgsVars),
};

const gpu::GpuComputePassDefinition kGenerateRigidContacts{
    "physics/physics_rigid_generate_contacts.cs.hlsl",
    "CRESSimNeo.Physics.RigidGenerateContacts.CS",
    "CRESSimNeo.Physics.RigidGenerateContacts.PSO",
    kGenerateContactsVars,
    std::size(kGenerateContactsVars),
};

const gpu::GpuComputePassDefinition kSolveRigidContactConstraints{
    "physics/physics_rigid_solve_contacts.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveContactConstraints.CS",
    "CRESSimNeo.Physics.RigidSolveContactConstraints.PSO",
    kSolveRigidContactConstraintsVars,
    std::size(kSolveRigidContactConstraintsVars),
};

const gpu::GpuComputePassDefinition kClearRigidCorrections{
    "physics/physics_rigid_clear_corrections.cs.hlsl",
    "CRESSimNeo.Physics.RigidClearCorrections.CS",
    "CRESSimNeo.Physics.RigidClearCorrections.PSO",
    kClearCorrectionsVars,
    std::size(kClearCorrectionsVars),
};

const gpu::GpuComputePassDefinition kApplyRigidCorrections{
    "physics/physics_rigid_apply_corrections.cs.hlsl",
    "CRESSimNeo.Physics.RigidApplyCorrections.CS",
    "CRESSimNeo.Physics.RigidApplyCorrections.PSO",
    kApplyCorrectionsVars,
    std::size(kApplyCorrectionsVars),
};

const gpu::GpuComputePassDefinition kUpdateRigidVelocities{
    "physics/physics_rigid_update_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidUpdateVelocities.CS",
    "CRESSimNeo.Physics.RigidUpdateVelocities.PSO",
    kUpdateVelocitiesVars,
    std::size(kUpdateVelocitiesVars),
};

const gpu::GpuComputePassDefinition kSolveRigidContactVelocities{
    "physics/physics_rigid_solve_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveContactVelocities.CS",
    "CRESSimNeo.Physics.RigidSolveContactVelocities.PSO",
    kSolveRigidContactVelocitiesVars,
    std::size(kSolveRigidContactVelocitiesVars),
};

const gpu::GpuComputePassDefinition kApplyRigidContactVelocities{
    "physics/physics_rigid_apply_contact_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidApplyContactVelocities.CS",
    "CRESSimNeo.Physics.RigidApplyContactVelocities.PSO",
    kApplyRigidContactVelocitiesVars,
    std::size(kApplyRigidContactVelocitiesVars),
};

} // namespace cressim::neo::physics::passdefs
