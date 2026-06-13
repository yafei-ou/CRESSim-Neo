#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H

#include "physics/export.h"

#include "common/scene_primitives.h"

#include <cstdint>

namespace cressim::neo::physics
{

struct PhysicsGpuRigidSceneView
{
    common::PoseBufferView poses{};
    Diligent::IBuffer *rigidParticleAttachmentsBuffer = nullptr;
    Diligent::IBuffer *rigidDistanceConstraintsBuffer = nullptr;
    Diligent::IBuffer *routedCableDescriptorsBuffer   = nullptr;
    Diligent::IBuffer *routedCableRoutePointsBuffer   = nullptr;
    Diligent::IBuffer *routedCableDebugSegmentsBuffer = nullptr;
    std::uint32_t rigidParticleAttachmentCount        = 0;
    std::uint32_t rigidDistanceConstraintCount        = 0;
    std::uint32_t routedCableCount                    = 0;
    std::uint32_t routedCableDebugSegmentCount        = 0;
    std::uint32_t colliderCount                       = 0;
    std::uint64_t bindingGeneration                   = 0;
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
    Diligent::IBuffer *edgesBuffer                   = nullptr;
    Diligent::IBuffer *bendsBuffer                   = nullptr;
    Diligent::IBuffer *tetsBuffer                    = nullptr;
    Diligent::IBuffer *suturingPairsBuffer           = nullptr;
    Diligent::IBuffer *suturingParticleRefsBuffer    = nullptr;
    Diligent::IBuffer *suturingInsertionStatesBuffer = nullptr;
    Diligent::IBuffer *suturingPathHeadersBuffer     = nullptr;
    Diligent::IBuffer *suturingPathNodesBuffer       = nullptr;
    Diligent::IBuffer *renderPositionsBuffer         = nullptr;
    Diligent::IBuffer *renderNormalsBuffer           = nullptr;
    Diligent::IBuffer *worldAabbsBuffer              = nullptr;
    std::uint32_t softBodyCount                      = 0;
    std::uint32_t edgeCount                          = 0;
    std::uint32_t bendCount                          = 0;
    std::uint32_t tetCount                           = 0;
    std::uint32_t suturingPairCount                  = 0;
    std::uint32_t suturingPathHeaderCount            = 0;
    std::uint32_t suturingPathNodeCount              = 0;
    std::uint64_t bindingGeneration                  = 0;
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

struct CRESSIM_NEO_PHYSICS_API PhysicsGpuSceneView
{
    PhysicsGpuRigidSceneView rigid{};
    PhysicsGpuSoftSceneView soft{};
    PhysicsGpuCurveSceneView curve{};
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_GPU_SCENE_VIEW_H
