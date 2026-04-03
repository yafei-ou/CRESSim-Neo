#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_SCENE_GPU_STATE_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_SCENE_GPU_STATE_H

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
        Diligent::RefCntAutoPtr<Diligent::IBuffer> shapeTypesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> shapeParamsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> localPositionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> localOrientationsBuffer;
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

    struct PredictedRigidBodyBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
    };

    struct PersistentSoftParticleBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsInvMassBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> previousPositionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> velocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> radiiBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> environmentIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> owningSoftBodyIndicesBuffer;
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
        Diligent::RefCntAutoPtr<Diligent::IBuffer> tetsBuffer;
    };

    struct PersistentRigidSurfaceParticleBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> localPositionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> owningRigidBodyIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> owningColliderIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> sampleRadiiBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> environmentIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> collisionLayersBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> collisionMasksBuffer;
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
        Diligent::RefCntAutoPtr<Diligent::IBuffer> rigidSurfaceWorldPositionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleBroadPhaseEntriesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleBroadPhaseKeysBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleBroadPhaseKeysScratchBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> particleCellRangesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRadixBitFlagsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRadixBitOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRadixMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softNeighborMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softSoftCandidatePairsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRigidCandidatePairsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softRigidContactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softContactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> activeSoftRigidContactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> activeSoftContactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softPositionCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softEdgeLambdasBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> softTetLambdasBuffer;
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

    struct SoftParticleReadbackBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> previousPositionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> velocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> neighborMetaBuffer;
    };

    bool ensureCapacity(Diligent::IRenderDevice *renderDevice, std::uint32_t bodyCount,
                        std::uint32_t colliderCount, std::uint32_t softParticleCount,
                        std::uint32_t softEdgeCount, std::uint32_t softTetCount,
                        std::uint32_t rigidSurfaceParticleCount, std::uint32_t physicsContextId);
    bool uploadWorldState(Diligent::IDeviceContext *computeContext, PhysicsWorld &world,
                          std::uint32_t bodyCount, std::uint32_t colliderCount);
    bool copyPredictedRigidBodiesToPersistentState(Diligent::IDeviceContext *computeContext,
                                                   std::uint32_t bodyCount);
    bool readbackBroadPhaseMetaBlocking(Diligent::IDeviceContext *computeContext,
                                        GpuBroadPhaseMeta &outMeta);
    bool readbackPredictedRigidStateBlocking(Diligent::IDeviceContext *computeContext,
                                             PhysicsWorld &world, std::uint32_t bodyCount);
    bool readbackPredictedSoftStateBlocking(Diligent::IDeviceContext *computeContext,
                                            PhysicsWorld &world, std::uint32_t softParticleCount);
    bool readbackSoftNeighborMetaBlocking(Diligent::IDeviceContext *computeContext,
                                          GpuSoftNeighborMeta &outMeta);

    const PersistentRigidBodyBuffers &persistentRigidBodies() const noexcept;
    const PersistentColliderBuffers &persistentColliders() const noexcept;
    const PersistentBodyColliderMappingBuffers &persistentBodyColliderMapping() const noexcept;
    const PersistentSoftParticleBuffers &persistentSoftParticles() const noexcept;
    const PersistentSoftTopologyBuffers &persistentSoftTopology() const noexcept;
    const PersistentRigidSurfaceParticleBuffers &persistentRigidSurfaceParticles() const noexcept;
    const SolverTransientBuffers &transientBuffers() const noexcept;
    std::uint32_t candidatePairCapacity() const noexcept;
    std::uint32_t softCandidatePairCapacity() const noexcept;
    bool correctionBuffersNeedClear() const noexcept;
    void setCorrectionBuffersNeedClear(bool needClear) noexcept;
    bool staticBroadPhaseDirty() const noexcept;
    void setStaticBroadPhaseDirty(bool dirty) noexcept;
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
    bool uploadSoftParticles(Diligent::IDeviceContext *computeContext,
                             const SoftParticleSoAHost &softParticles);
    bool uploadSoftTopology(Diligent::IDeviceContext *computeContext,
                            const std::vector<SoftEdge> &softEdges,
                            const std::vector<SoftTet> &softTets);
    bool uploadRigidSurfaceParticles(Diligent::IDeviceContext *computeContext,
                                     const RigidSurfaceParticleSoAHost &surfaceParticles);
    PersistentRigidBodyBuffers mPersistentRigidBodies;
    PersistentColliderBuffers mPersistentColliders;
    PersistentBodyColliderMappingBuffers mPersistentBodyColliderMapping;
    PersistentSoftParticleBuffers mPersistentSoftParticles;
    PersistentSoftTopologyBuffers mPersistentSoftTopology;
    PersistentRigidSurfaceParticleBuffers mPersistentRigidSurfaceParticles;
    SolverTransientBuffers mTransientState;
    RigidBodyReadbackBuffers mReadbackRigidBodies;
    SoftParticleReadbackBuffers mReadbackSoftParticles;
    std::uint32_t mRigidBodyCapacity                        = 0;
    std::uint32_t mColliderCapacity                         = 0;
    std::uint32_t mSoftParticleCapacity                     = 0;
    std::uint32_t mSoftEdgeCapacity                         = 0;
    std::uint32_t mSoftTetCapacity                          = 0;
    std::uint32_t mRigidSurfaceParticleCapacity             = 0;
    std::uint32_t mParticleBroadPhaseEntryCapacity          = 0;
    std::uint32_t mSoftCandidatePairCapacity                = 0;
    std::uint32_t mSoftParticleAdjacencyCapacity            = 0;
    std::uint32_t mRigidBodyCount                           = 0;
    std::uint32_t mColliderCount                            = 0;
    std::uint32_t mSoftBodyCount                            = 0;
    std::uint32_t mSoftParticleCount                        = 0;
    std::uint32_t mSoftEdgeCount                            = 0;
    std::uint32_t mSoftTetCount                             = 0;
    std::uint32_t mRigidSurfaceParticleCount                = 0;
    std::uint32_t mBroadPhaseNodeCapacity                   = 0;
    std::uint32_t mCandidatePairCapacity                    = 0;
    std::uint32_t mContactCapacity                          = 0;
    std::uint32_t mSoftScanScratchCapacity                  = 0;
    bool mCorrectionBuffersNeedClear                        = false;
    bool mStaticBroadPhaseDirty                             = true;
    bool mRigidBodyUploadResetRequired                      = true;
    bool mColliderUploadResetRequired                       = true;
    bool mSoftParticleUploadResetRequired                   = true;
    bool mSoftTopologyUploadResetRequired                   = true;
    bool mRigidSurfaceParticleUploadResetRequired           = true;
    std::uint64_t mLastUploadedSoftParticleRevision         = 0;
    std::uint64_t mLastUploadedSoftTopologyRevision         = 0;
    std::uint64_t mLastUploadedRigidSurfaceParticleRevision = 0;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_SCENE_GPU_STATE_H
