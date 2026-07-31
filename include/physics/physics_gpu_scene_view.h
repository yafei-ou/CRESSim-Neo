#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H

#include "common/scene_primitives.h"

#include <cstdint>

namespace cressim::neo::physics
{

struct PhysicsGpuRigidSceneView
{
    common::PoseBufferView poses{};
    Diligent::IBuffer *statePositionsBuffer                = nullptr;
    Diligent::IBuffer *stateOrientationsBuffer             = nullptr;
    Diligent::IBuffer *stateLinearVelocitiesBuffer         = nullptr;
    Diligent::IBuffer *stateAngularVelocitiesBuffer        = nullptr;
    Diligent::IBuffer *inverseInertiaLocalBuffer           = nullptr;
    Diligent::IBuffer *bodyTypesBuffer                     = nullptr;
    Diligent::IBuffer *proxyParticleContactMaterialsBuffer = nullptr;
    Diligent::IBuffer *kinematicTargetPositionsBuffer      = nullptr;
    Diligent::IBuffer *kinematicTargetOrientationsBuffer   = nullptr;
    Diligent::IBuffer *kinematicTargetFlagsBuffer          = nullptr;
    Diligent::IBuffer *colliderOwnerBodyIndicesBuffer      = nullptr;
    Diligent::IBuffer *colliderBroadPhaseBuffer            = nullptr;
    Diligent::IBuffer *colliderGeometryBuffer              = nullptr;
    Diligent::IBuffer *colliderMaterialsBuffer             = nullptr;
    Diligent::IBuffer *colliderShapeTypesBuffer            = nullptr;
    Diligent::IBuffer *colliderEnabledFlagsBuffer          = nullptr;
    Diligent::IBuffer *bodyColliderOffsetsBuffer           = nullptr;
    Diligent::IBuffer *bodyColliderCountsBuffer            = nullptr;
    Diligent::IBuffer *bodyColliderRangesBuffer            = nullptr;
    Diligent::IBuffer *bodyColliderIndicesBuffer           = nullptr;
    // Constraint item buffers preserve prepare()/uploadWorld()-resolved slot ownership but allow
    // runtime mutation of supported per-slot fields through custom compute. For rigid-particle
    // attachments this includes the live descriptor payload read directly by the solver
    // (`particleIndex`, `rigidBodyIndex`, `compliance`, `enabled`) as long as the caller does not
    // assume prepared host-side layout mappings will reflect those GPU-only edits.
    Diligent::IBuffer *rigidParticleAttachmentsBuffer      = nullptr;
    Diligent::IBuffer *strandRigidAttachmentsBuffer        = nullptr;
    Diligent::IBuffer *rigidDistanceConstraintsBuffer      = nullptr;
    Diligent::IBuffer *routedCableDescriptorsBuffer        = nullptr;
    Diligent::IBuffer *routedCableRoutePointsBuffer        = nullptr;
    Diligent::IBuffer *routedCableDebugSegmentsBuffer      = nullptr;
    std::uint32_t bodyCount                                = 0;
    std::uint32_t rigidParticleAttachmentCount             = 0;
    std::uint32_t strandRigidAttachmentCount               = 0;
    std::uint32_t rigidDistanceConstraintCount             = 0;
    std::uint32_t routedCableCount                         = 0;
    std::uint32_t routedCableRoutePointCount               = 0;
    std::uint32_t routedCableDebugSegmentCount             = 0;
    std::uint32_t colliderCount                            = 0;
    std::uint64_t bindingGeneration                        = 0;
    std::uint64_t constraintBindingGeneration              = 0;
};

struct PhysicsGpuJointSceneView
{
    Diligent::IBuffer *ballJointsBuffer               = nullptr;
    Diligent::IBuffer *sphericalJointsBuffer          = nullptr;
    Diligent::IBuffer *hingeJointsBuffer              = nullptr;
    Diligent::IBuffer *hingeJointRuntimeStatesBuffer  = nullptr;
    Diligent::IBuffer *sliderJointsBuffer             = nullptr;
    Diligent::IBuffer *sliderJointRuntimeStatesBuffer = nullptr;
    std::uint32_t ballJointCount                      = 0u;
    std::uint32_t hingeJointCount                     = 0u;
    std::uint32_t sphericalJointCount                 = 0u;
    std::uint32_t sliderJointCount                    = 0u;
    std::uint64_t bindingGeneration                   = 0u;
};

struct PhysicsGpuParticleBufferView
{
    Diligent::IBuffer *positionsInvMassBuffer              = nullptr;
    Diligent::IBuffer *previousPositionsBuffer             = nullptr;
    Diligent::IBuffer *velocitiesBuffer                    = nullptr;
    Diligent::IBuffer *radiiBuffer                         = nullptr;
    Diligent::IBuffer *environmentIndicesBuffer            = nullptr;
    Diligent::IBuffer *particleKindsBuffer                 = nullptr;
    Diligent::IBuffer *ownerTypesBuffer                    = nullptr;
    Diligent::IBuffer *ownerIndicesBuffer                  = nullptr;
    Diligent::IBuffer *strandIdsBuffer                     = nullptr;
    Diligent::IBuffer *strandRolesBuffer                   = nullptr;
    Diligent::IBuffer *suturingNeighborLinksBuffer         = nullptr;
    Diligent::IBuffer *owningSoftBodyIndicesBuffer         = nullptr;
    Diligent::IBuffer *particleMaterialIndicesBuffer       = nullptr;
    Diligent::IBuffer *fluidMaterialIndicesBuffer          = nullptr;
    Diligent::IBuffer *fluidVisualsBuffer                  = nullptr;
    Diligent::IBuffer *particleContactMaterialsBuffer      = nullptr;
    Diligent::IBuffer *fluidMaterialsBuffer                = nullptr;
    Diligent::IBuffer *phasesBuffer                        = nullptr;
    Diligent::IBuffer *collisionLayersBuffer               = nullptr;
    Diligent::IBuffer *collisionMasksBuffer                = nullptr;
    Diligent::IBuffer *adjacencyOffsetsBuffer              = nullptr;
    Diligent::IBuffer *adjacencyCountsBuffer               = nullptr;
    Diligent::IBuffer *adjacencyIndicesBuffer              = nullptr;
    Diligent::IBuffer *fluidSurfaceNormalConstraintsBuffer = nullptr;
    Diligent::IBuffer *fluidAnisotropy1Buffer              = nullptr;
    Diligent::IBuffer *fluidAnisotropy2Buffer              = nullptr;
    Diligent::IBuffer *fluidAnisotropy3Buffer              = nullptr;
    std::uint32_t count                                    = 0;
    std::uint32_t fluidVisualCount                         = 0;
    std::uint32_t contactMaterialCount                     = 0;
    std::uint32_t fluidMaterialCount                       = 0;
};

struct PhysicsGpuSoftSceneView
{
    PhysicsGpuParticleBufferView particles{};
    Diligent::IBuffer *edgesBuffer                       = nullptr;
    Diligent::IBuffer *bendsBuffer                       = nullptr;
    Diligent::IBuffer *tetsBuffer                        = nullptr;
    Diligent::IBuffer *strandSegmentsBuffer              = nullptr;
    Diligent::IBuffer *strandJointsBuffer                = nullptr;
    Diligent::IBuffer *strandDistanceConstraintsBuffer   = nullptr;
    Diligent::IBuffer *strandSegmentStatesBuffer         = nullptr;
    Diligent::IBuffer *shapeClustersBuffer               = nullptr;
    Diligent::IBuffer *shapeClusterMembersBuffer         = nullptr;
    Diligent::IBuffer *particleShapeMembershipRangesBuffer  = nullptr;
    Diligent::IBuffer *particleShapeMembershipIndicesBuffer = nullptr;
    Diligent::IBuffer *membershipShapeClusterIndicesBuffer  = nullptr;
    Diligent::IBuffer *shapeClusterPosesBuffer              = nullptr;
    Diligent::IBuffer *shapeCorrectionMagnitudesBuffer      = nullptr;
    Diligent::IBuffer *segmentStrandJointRangesBuffer    = nullptr;
    Diligent::IBuffer *segmentIncidentStrandJointsBuffer = nullptr;
    Diligent::IBuffer *suturingPairsBuffer               = nullptr;
    Diligent::IBuffer *suturingParticleRefsBuffer        = nullptr;
    Diligent::IBuffer *suturingInsertionStatesBuffer     = nullptr;
    Diligent::IBuffer *suturingPathHeadersBuffer         = nullptr;
    Diligent::IBuffer *suturingPathNodesBuffer           = nullptr;
    Diligent::IBuffer *renderPositionsBuffer             = nullptr;
    Diligent::IBuffer *renderNormalsBuffer               = nullptr;
    Diligent::IBuffer *worldAabbsBuffer                  = nullptr;
    std::uint32_t softBodyCount                          = 0;
    std::uint32_t edgeCount                              = 0;
    std::uint32_t bendCount                              = 0;
    std::uint32_t tetCount                               = 0;
    std::uint32_t strandSegmentCount                     = 0;
    std::uint32_t strandJointCount                       = 0;
    std::uint32_t strandDistanceCount                    = 0;
    std::uint32_t shapeClusterCount                      = 0;
    std::uint32_t shapeClusterMemberCount                = 0;
    std::uint32_t particleShapeMembershipIndexCount      = 0;
    std::uint32_t suturingPairCount                      = 0;
    std::uint32_t suturingPathHeaderCount                = 0;
    std::uint32_t suturingPathNodeCount                  = 0;
    std::uint64_t bindingGeneration                      = 0;
};

struct PhysicsGpuCurveSceneView
{
    Diligent::IBuffer *descriptorsBuffer     = nullptr;
    Diligent::IBuffer *particleIndicesBuffer = nullptr;
    Diligent::IBuffer *positionsBuffer       = nullptr;
    Diligent::IBuffer *normalsBuffer         = nullptr;
    Diligent::IBuffer *worldAabbsBuffer      = nullptr;
    std::uint32_t curveCount                 = 0;
    std::uint64_t bindingGeneration          = 0;
};

struct PhysicsGpuSceneView
{
    PhysicsGpuRigidSceneView rigid{};
    PhysicsGpuJointSceneView joints{};
    PhysicsGpuSoftSceneView soft{};
    PhysicsGpuCurveSceneView curve{};
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
