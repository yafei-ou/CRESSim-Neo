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

    struct PredictedRigidBodyBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
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
        Diligent::RefCntAutoPtr<Diligent::IBuffer> contactsBuffer;
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

    bool ensureCapacity(Diligent::IRenderDevice *renderDevice, std::uint32_t bodyCount,
                        std::uint32_t colliderCount, std::uint32_t physicsContextId);
    bool uploadWorldState(Diligent::IDeviceContext *computeContext, PhysicsWorld &world,
                          std::uint32_t bodyCount, std::uint32_t colliderCount);
    bool copyPredictedRigidBodiesToPersistentState(Diligent::IDeviceContext *computeContext,
                                                   std::uint32_t bodyCount);
    bool readbackBroadPhaseMetaBlocking(Diligent::IDeviceContext *computeContext,
                                        GpuBroadPhaseMeta &outMeta);
    bool readbackPredictedRigidStateBlocking(Diligent::IDeviceContext *computeContext,
                                             PhysicsWorld &world, std::uint32_t bodyCount);

    const PersistentRigidBodyBuffers &persistentRigidBodies() const noexcept;
    const PersistentColliderBuffers &persistentColliders() const noexcept;
    const SolverTransientBuffers &transientBuffers() const noexcept;
    std::uint32_t candidatePairCapacity() const noexcept;
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
    PersistentRigidBodyBuffers mPersistentRigidBodies;
    PersistentColliderBuffers mPersistentColliders;
    SolverTransientBuffers mTransientState;
    RigidBodyReadbackBuffers mReadbackRigidBodies;
    std::uint32_t mRigidBodyCapacity      = 0;
    std::uint32_t mColliderCapacity       = 0;
    std::uint32_t mRigidBodyCount         = 0;
    std::uint32_t mColliderCount          = 0;
    std::uint32_t mBroadPhaseNodeCapacity = 0;
    std::uint32_t mCandidatePairCapacity  = 0;
    std::uint32_t mContactCapacity        = 0;
    bool mCorrectionBuffersNeedClear      = false;
    bool mStaticBroadPhaseDirty           = true;
    bool mRigidBodyUploadResetRequired    = true;
    bool mColliderUploadResetRequired     = true;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_SCENE_GPU_STATE_H
