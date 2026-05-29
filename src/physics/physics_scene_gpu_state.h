#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_SCENE_GPU_STATE_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_SCENE_GPU_STATE_H

#include "gpu/shared_export_buffer.h"
#include "physics/physics_gpu_scene_view.h"
#include "physics/physics_world.h"
#include "physics/rigid_body_common.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

#include <array>
#include <cstdint>
#include <vector>

namespace cressim::neo::physics
{

class PhysicsSceneGpuState
{
public:
    struct PersistentRigidBodyBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> scalesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> inverseInertiaLocalBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> bodyTypesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> kinematicTargetPositionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> kinematicTargetOrientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> kinematicTargetFlagsBuffer;
    };

    struct PersistentColliderBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> ownerRigidBodyIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> broadPhaseDataBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> geometryDataBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> contactDataBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> shapeTypesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> enabledFlagsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> materialBuffer;
    };

    struct PersistentBodyColliderMappingBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderCountsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderIndicesBuffer;
    };

    struct PersistentJointCollisionSuppressionBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> neighborOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> neighborsBuffer;
    };

    struct PersistentJointBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> ballJointsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> hingeJointsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> sliderJointsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> hingePassiveJointIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> hingePositionDriveJointIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> hingeVelocityDriveJointIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> sliderPassiveJointIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> sliderPositionDriveJointIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> sliderVelocityDriveJointIndicesBuffer;
    };

    struct PredictedRigidBodyBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
    };

    struct PersistentParticleBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsInvMassBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> previousPositionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> velocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> radiiBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> environmentIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleKindsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> ownerTypesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> ownerIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> deformableObjectKindsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> deformableObjectIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> strandIdsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> strandOrdersBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> strandRolesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> owningSoftBodyIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleMaterialIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidMaterialIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidVisualsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleContactMaterialsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidMaterialsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> phasesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> collisionLayersBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> collisionMasksBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> adjacencyOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> adjacencyCountsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> adjacencyIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> broadPhaseMetadataBuffer;
    };

    struct PersistentSoftTopologyBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> edgesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> bendsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> tetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleEdgeRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleIncidentEdgesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleBendRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleIncidentBendsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleTetRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleIncidentTetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> renderVertexTriangleRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> renderVertexTriangleIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> renderTriangleParticleIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> renderTriangleNormalsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softBodyParticleRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softBodyChunkRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softBodyBoundsChunksBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softBodyFallbackNormalsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softBodyRenderNormalsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softBodyWorldAabbsBuffer;
    };

    struct PreviousRigidBodyBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
    };

    struct SolverTransientBuffers
    {
        PredictedRigidBodyBuffers predictedRigidBodies;
        PreviousRigidBodyBuffers previousRigidBodies;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleBroadPhaseEntriesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleBroadPhaseKeysBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleBroadPhaseKeysScratchBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleCellRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRadixBitFlagsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRadixBitOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRadixMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softNeighborMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> physicsIndirectArgsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softSoftCandidatePairsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidNeighborPairsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidBoundaryCandidateCountsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidBoundaryCandidateOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidBoundaryCandidateRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidBoundaryCandidatePairsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRigidCandidatePairsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRigidContactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softContactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> activeSoftRigidContactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> activeSoftContactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softPositionCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softVelocityCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidDeltaPositionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidIterationDeltaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidSurfaceNormalConstraintsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidAnisotropy1Buffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidAnisotropy2Buffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidAnisotropy3Buffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fluidVorticitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softEdgeLambdasBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softBendLambdasBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softTetLambdasBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softEdgeCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softBendCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softTetCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softBodyChunkAabbsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> bodyAabbsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> bodyMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> activeBodyFlagsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> activeBodyOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> activeBodyIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticBodyFlagsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticBodyOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticBodyIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> broadPhaseElementsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> mortonCodesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> mortonCodesScratchBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> globalBroadPhaseExtentBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticBroadPhaseElementsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticMortonCodesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticMortonCodesScratchBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticGlobalBroadPhaseExtentBuffer;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IBuffer>> scanBlockSumsBuffers;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IBuffer>> scanScannedBlockSumsBuffers;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IBuffer>> broadPhaseExtentScratchBuffers;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IBuffer>> staticScanBlockSumsBuffers;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IBuffer>> staticScanScannedBlockSumsBuffers;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IBuffer>>
            staticBroadPhaseExtentScratchBuffers;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> radixBitFlagsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> radixBitOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> radixMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> bvhBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> bvhConstructionInfoBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticRadixBitFlagsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticRadixBitOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticRadixMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticBvhBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> staticBvhConstructionInfoBuffer;
        std::array<Diligent::RefCntAutoPtr<Diligent::IBuffer>, kRigidPairTypeCount>
            pairCountBuffers;
        std::array<Diligent::RefCntAutoPtr<Diligent::IBuffer>, kRigidPairTypeCount>
            pairOffsetBuffers;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> rigidPairRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> candidatePairsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> broadPhaseMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> narrowPhaseChunksBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> narrowPhaseMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> narrowPhaseChunkCounterBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> rigidContactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> rigidBodyPairAggregateMapBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> rigidBodyPairAggregateActiveCountBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> rigidBodyPairAggregateHeadersBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> rigidBodyPairAggregateSlotsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> hingeJointLambdas0123Buffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> hingeJointLambdas45Buffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> sliderJointLambdas0123Buffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> sliderJointLambdas45Buffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> translationCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> rotationCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocityCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocityCorrectionsBuffer;
    };

    struct RigidBodyReadbackBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> broadPhaseMetaBuffer;
    };

    struct ParticleReadbackBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> previousPositionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> velocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> neighborMetaBuffer;
    };

    struct BroadPhaseCompactionTailReadback
    {
        std::uint32_t colliderCount = 0;
        std::uint32_t sampleBegin   = 0;
        std::uint32_t sampleCount   = 0;
        std::array<std::uint32_t, 4> activeFlags{};
        std::array<std::uint32_t, 4> activeOffsets{};
        std::array<std::uint32_t, 4> staticFlags{};
        std::array<std::uint32_t, 4> staticOffsets{};
    };

    struct BroadPhaseInputTailReadback
    {
        std::uint32_t colliderCount  = 0;
        std::uint32_t rigidBodyCount = 0;
        std::uint32_t sampleBegin    = 0;
        std::uint32_t sampleCount    = 0;
        std::array<std::uint32_t, 4> ownerRigidBodyIndices{};
        std::array<std::uint32_t, 4> enabledFlags{};
        std::array<std::uint32_t, 4> rigidBodyTypes{};
    };

    struct BroadPhaseOutputTailReadback
    {
        std::uint32_t colliderSampleBegin  = 0;
        std::uint32_t rigidBodySampleBegin = 0;
        std::uint32_t sampleCount          = 0;
        std::array<std::uint32_t, 4> bodyMetaTypes{};
        std::array<std::uint32_t, 4> bodyMetaIds{};
        std::array<std::uint32_t, 4> bodyMetaActiveIndices{};
        std::array<float, 4> aabbMinX{};
        std::array<float, 4> aabbMaxX{};
        std::array<float, 4> predictedPosX{};
        std::array<float, 4> predictedPosW{};
    };

    struct PredictedRigidTailReadback
    {
        std::uint32_t rigidBodyCount = 0;
        std::uint32_t sampleBegin    = 0;
        std::uint32_t sampleCount    = 0;
        std::array<float, 4> predictedPosX{};
        std::array<float, 4> predictedPosW{};
    };

    struct PersistentRigidTailReadback
    {
        std::uint32_t rigidBodyCount = 0;
        std::uint32_t sampleBegin    = 0;
        std::uint32_t sampleCount    = 0;
        std::array<float, 4> posX{};
        std::array<float, 4> posW{};
        std::array<std::uint32_t, 4> bodyTypes{};
    };

    bool ensureCapacity(Diligent::IRenderDevice *renderDevice, std::uint32_t bodyCount,
                        std::uint32_t colliderCount, std::uint32_t particleCount,
                        std::uint32_t fluidCount, std::uint32_t particleContactMaterialCount,
                        std::uint32_t fluidMaterialCount, std::uint32_t softEdgeCount,
                        std::uint32_t softBendCount, std::uint32_t softTetCount,
                        std::uint32_t ballJointCount, std::uint32_t hingeJointCount,
                        std::uint32_t sliderJointCount, std::uint32_t softRenderVertexCount,
                        std::uint32_t softRenderTriangleIndexCount,
                        std::uint32_t softRenderTriangleCount, std::uint32_t softBodyRangeCount,
                        std::uint32_t softBodyBoundsChunkCount, Diligent::Uint64 sharedContextMask,
                        const std::uint32_t *sharedQueueFamilyIndices,
                        std::uint32_t sharedQueueFamilyIndexCount, bool useNativeFloatAtomics);
    bool uploadWorldState(Diligent::IDeviceContext *computeContext, PhysicsWorld &world,
                          std::uint32_t bodyCount, std::uint32_t colliderCount);
    bool copyPredictedRigidBodiesToPersistentState(Diligent::IDeviceContext *computeContext,
                                                   std::uint32_t bodyCount);
    bool readbackBroadPhaseMetaBlocking(Diligent::IDeviceContext *computeContext,
                                        GpuBroadPhaseMeta &outMeta);
    bool readbackPredictedRigidStateBlocking(Diligent::IDeviceContext *computeContext,
                                             PhysicsWorld &world, std::uint32_t bodyCount);
    bool readbackPredictedParticleStateBlocking(Diligent::IDeviceContext *computeContext,
                                                PhysicsWorld &world, std::uint32_t particleCount);
    bool readbackSoftNeighborMetaBlocking(Diligent::IDeviceContext *computeContext,
                                          GpuParticleNeighborMeta &outMeta);

    const PersistentRigidBodyBuffers &persistentRigidBodies() const noexcept;
    const PersistentColliderBuffers &persistentColliders() const noexcept;
    const PersistentBodyColliderMappingBuffers &persistentBodyColliderMapping() const noexcept;
    const PersistentJointCollisionSuppressionBuffers &persistentJointCollisionSuppression()
        const noexcept;
    const PersistentJointBuffers &persistentJoints() const noexcept;
    const PersistentParticleBuffers &persistentParticles() const noexcept;
    const gpu::SharedExportBuffer &softPositionsInvMassSharedBuffer() const noexcept;
    const PersistentSoftTopologyBuffers &persistentSoftTopology() const noexcept;
    const SolverTransientBuffers &transientBuffers() const noexcept;
    std::uint32_t ballJointCount() const noexcept;
    std::uint32_t hingeJointCount() const noexcept;
    std::uint32_t sliderJointCount() const noexcept;
    std::uint32_t hingePassiveJointCount() const noexcept;
    std::uint32_t hingePositionDriveJointCount() const noexcept;
    std::uint32_t hingeVelocityDriveJointCount() const noexcept;
    std::uint32_t sliderPassiveJointCount() const noexcept;
    std::uint32_t sliderPositionDriveJointCount() const noexcept;
    std::uint32_t sliderVelocityDriveJointCount() const noexcept;
    std::uint32_t candidatePairCapacity() const noexcept;
    std::uint32_t particleCandidatePairCapacity() const noexcept;
    std::uint32_t fluidBoundaryCandidatePairCapacity() const noexcept;
    std::uint32_t maxFluidNeighborhood() const noexcept;
    bool correctionBuffersNeedClear() const noexcept;
    void setCorrectionBuffersNeedClear(bool needClear) noexcept;
    bool staticBroadPhaseDirty() const noexcept;
    void setStaticBroadPhaseDirty(bool dirty) noexcept;
    std::uint64_t rigidBindingGeneration() const noexcept;
    std::uint64_t softBindingGeneration() const noexcept;
    PhysicsGpuSceneView sceneView() const noexcept;

private:
    bool uploadRigidBodies(Diligent::IDeviceContext *computeContext, const PhysicsWorld &world,
                           const RigidBodySoAHost &rigidBodies, std::uint32_t bodyCount,
                           bool forceFullUpload);
    bool uploadColliders(Diligent::IDeviceContext *computeContext, const PhysicsWorld &world,
                         const ColliderSoAHost &colliders, std::uint32_t colliderCount,
                         bool forceFullUpload);
    bool uploadRigidBodyRange(Diligent::IDeviceContext *computeContext,
                              const RigidBodySoAHost &rigidBodies, std::uint32_t begin,
                              std::uint32_t count);
    bool uploadColliderRange(Diligent::IDeviceContext *computeContext,
                             const ColliderSoAHost &colliders, std::uint32_t begin,
                             std::uint32_t count);
    bool uploadColliderBroadPhaseRange(Diligent::IDeviceContext *computeContext,
                                       const ColliderSoAHost &colliders, std::uint32_t begin,
                                       std::uint32_t count);
    bool uploadBodyColliderMapping(Diligent::IDeviceContext *computeContext,
                                   const BodyColliderMappingHost &mapping, std::uint32_t bodyCount,
                                   std::uint32_t colliderCount);
    bool uploadJointCollisionSuppression(Diligent::IDeviceContext *computeContext,
                                         const JointCollisionSuppressionHost &suppression,
                                         std::uint32_t bodyCount);
    bool uploadRigidJoints(Diligent::IDeviceContext *computeContext, const PhysicsWorld &world);
    bool uploadParticles(Diligent::IDeviceContext *computeContext, const ParticleSoAHost &particles,
                         const std::vector<FluidState> &fluids,
                         const std::vector<Diligent::float4> &particleContactMaterials,
                         const std::vector<FluidMaterialGpu> &fluidMaterials);
    bool uploadSoftTopology(Diligent::IDeviceContext *computeContext, std::uint32_t particleCount,
                            const SoftRenderDataHost &softRenderData,
                            const std::vector<DeformableDistanceConstraint> &distanceConstraints,
                            const std::vector<DeformableBendConstraint> &bendConstraints,
                            const std::vector<DeformableVolumeConstraint> &volumeConstraints);
    gpu::SharedExportBuffer mSharedSoftPositionsInvMass;
    PersistentRigidBodyBuffers mPersistentRigidBodies;
    PersistentColliderBuffers mPersistentColliders;
    PersistentBodyColliderMappingBuffers mPersistentBodyColliderMapping;
    PersistentJointCollisionSuppressionBuffers mPersistentJointCollisionSuppression;
    PersistentJointBuffers mPersistentJoints;
    PersistentParticleBuffers mPersistentParticles;
    PersistentSoftTopologyBuffers mPersistentSoftTopology;
    SolverTransientBuffers mTransientState;
    RigidBodyReadbackBuffers mReadbackRigidBodies;
    ParticleReadbackBuffers mReadbackParticles;
    std::uint32_t mRigidBodyCapacity                         = 0;
    std::uint32_t mColliderCapacity                          = 0;
    std::uint32_t mSoftParticleCapacity                      = 0;
    std::uint32_t mFluidVisualCapacity                       = 0;
    std::uint32_t mParticleContactMaterialCapacity           = 0;
    std::uint32_t mFluidMaterialCapacity                     = 0;
    std::uint32_t mSoftEdgeCapacity                          = 0;
    std::uint32_t mSoftBendCapacity                          = 0;
    std::uint32_t mSoftTetCapacity                           = 0;
    std::uint32_t mParticleBroadPhaseEntryCapacity           = 0;
    std::uint32_t mSoftCandidatePairCapacity                 = 0;
    std::uint32_t mFluidBoundaryCandidatePairCapacity        = 0;
    std::uint32_t mFluidNeighborPairCapacity                 = 0;
    std::uint32_t mMaxFluidNeighborhood                      = 0;
    std::uint32_t mSoftParticleAdjacencyCapacity             = 0;
    std::uint32_t mRigidBodyCount                            = 0;
    std::uint32_t mColliderCount                             = 0;
    std::uint32_t mSoftBodyCount                             = 0;
    std::uint32_t mSoftParticleCount                         = 0;
    std::uint32_t mFluidCount                                = 0;
    std::uint32_t mParticleContactMaterialCount              = 0;
    std::uint32_t mFluidMaterialCount                        = 0;
    std::uint32_t mSoftEdgeCount                             = 0;
    std::uint32_t mSoftBendCount                             = 0;
    std::uint32_t mSoftTetCount                              = 0;
    std::uint32_t mBroadPhaseNodeCapacity                    = 0;
    std::uint32_t mCandidatePairCapacity                     = 0;
    std::uint32_t mContactCapacity                           = 0;
    std::uint32_t mSoftScanScratchCapacity                   = 0;
    std::uint32_t mSoftIncidentEdgeCapacity                  = 0;
    std::uint32_t mSoftIncidentBendCapacity                  = 0;
    std::uint32_t mSoftIncidentTetCapacity                   = 0;
    std::uint32_t mSoftRenderVertexCapacity                  = 0;
    std::uint32_t mSoftRenderTriangleIndexCapacity           = 0;
    std::uint32_t mSoftRenderTriangleCapacity                = 0;
    std::uint32_t mSoftBodyRangeCapacity                     = 0;
    std::uint32_t mSoftBodyBoundsChunkCapacity               = 0;
    std::uint32_t mJointCollisionSuppressionOffsetCapacity   = 0;
    std::uint32_t mJointCollisionSuppressionNeighborCapacity = 0;
    std::uint32_t mBallJointCapacity                         = 0;
    std::uint32_t mHingeJointCapacity                        = 0;
    std::uint32_t mSliderJointCapacity                       = 0;
    std::uint32_t mHingePassiveJointIndexCapacity            = 0;
    std::uint32_t mHingePositionDriveIndexCapacity           = 0;
    std::uint32_t mHingeVelocityDriveIndexCapacity           = 0;
    std::uint32_t mSliderPassiveJointIndexCapacity           = 0;
    std::uint32_t mSliderPositionDriveIndexCapacity          = 0;
    std::uint32_t mSliderVelocityDriveIndexCapacity          = 0;
    bool mCorrectionBuffersNeedClear                         = false;
    bool mStaticBroadPhaseDirty                              = true;
    bool mRigidBodyUploadResetRequired                       = true;
    bool mColliderUploadResetRequired                        = true;
    bool mRigidJointUploadResetRequired                      = true;
    bool mSoftParticleUploadResetRequired                    = true;
    bool mSoftTopologyUploadResetRequired                    = true;
    std::uint64_t mRigidBindingGeneration                    = 1u;
    std::uint64_t mSoftBindingGeneration                     = 1u;
    std::uint64_t mLastUploadedRigidJointSceneRevision       = 0;
    std::uint64_t mLastUploadedRigidJointModeRevision        = 0;
    std::uint64_t mLastUploadedSoftParticleRevision          = 0;
    std::uint64_t mLastUploadedSoftTopologyRevision          = 0;
    std::uint32_t mBallJointCount                            = 0;
    std::uint32_t mHingeJointCount                           = 0;
    std::uint32_t mSliderJointCount                          = 0;
    std::uint32_t mHingePassiveJointCount                    = 0;
    std::uint32_t mHingePositionDriveJointCount              = 0;
    std::uint32_t mHingeVelocityDriveJointCount              = 0;
    std::uint32_t mSliderPassiveJointCount                   = 0;
    std::uint32_t mSliderPositionDriveJointCount             = 0;
    std::uint32_t mSliderVelocityDriveJointCount             = 0;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_SCENE_GPU_STATE_H
