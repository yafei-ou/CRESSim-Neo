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

constexpr Diligent::ShaderResourceVariableDesc kSolveGatherVars[] = {
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

} // namespace

const gpu::GpuComputePassDefinition kPredict{
    "physics/physics_rigid_predict.cs.hlsl",
    "CRESSimNeo.Physics.RigidPredict.CS",
    "CRESSimNeo.Physics.RigidPredict.PSO",
    kPredictVars,
    std::size(kPredictVars),
};

const gpu::GpuComputePassDefinition kUpdateWorldAabbs{
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

const gpu::GpuComputePassDefinition kGenerateContacts{
    "physics/physics_rigid_generate_contacts.cs.hlsl",
    "CRESSimNeo.Physics.RigidGenerateContacts.CS",
    "CRESSimNeo.Physics.RigidGenerateContacts.PSO",
    kGenerateContactsVars,
    std::size(kGenerateContactsVars),
};

const gpu::GpuComputePassDefinition kSolveGather{
    "physics/physics_rigid_solve_gather.cs.hlsl",
    "CRESSimNeo.Physics.RigidSolveGather.CS",
    "CRESSimNeo.Physics.RigidSolveGather.PSO",
    kSolveGatherVars,
    std::size(kSolveGatherVars),
};

const gpu::GpuComputePassDefinition kClearCorrections{
    "physics/physics_rigid_clear_corrections.cs.hlsl",
    "CRESSimNeo.Physics.RigidClearCorrections.CS",
    "CRESSimNeo.Physics.RigidClearCorrections.PSO",
    kClearCorrectionsVars,
    std::size(kClearCorrectionsVars),
};

const gpu::GpuComputePassDefinition kApplyCorrections{
    "physics/physics_rigid_apply_corrections.cs.hlsl",
    "CRESSimNeo.Physics.RigidApplyCorrections.CS",
    "CRESSimNeo.Physics.RigidApplyCorrections.PSO",
    kApplyCorrectionsVars,
    std::size(kApplyCorrectionsVars),
};

const gpu::GpuComputePassDefinition kUpdateVelocities{
    "physics/physics_rigid_update_velocities.cs.hlsl",
    "CRESSimNeo.Physics.RigidUpdateVelocities.CS",
    "CRESSimNeo.Physics.RigidUpdateVelocities.PSO",
    kUpdateVelocitiesVars,
    std::size(kUpdateVelocitiesVars),
};

} // namespace cressim::neo::physics::passdefs
