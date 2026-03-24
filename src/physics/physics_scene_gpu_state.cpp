#include "physics/physics_scene_gpu_state.h"

#include "physics/rigid_body_common.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"

#include <algorithm>
#include <string>

namespace cressim::neo::physics
{

namespace
{

constexpr std::uint32_t kComputeThreadGroupSize = 64u;
constexpr std::uint32_t kNarrowPhaseChunkSize   = 128u;

Diligent::Uint64 contextMaskForId(std::uint32_t contextId)
{
    return static_cast<Diligent::Uint64>(1ull) << contextId;
}

bool ensureStructuredBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                            std::uint32_t elementStride, std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags, Diligent::USAGE usage,
                            Diligent::CPU_ACCESS_FLAGS cpuAccess,
                            Diligent::Uint64 immediateContextMask,
                            Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    Diligent::BufferDesc desc{};
    desc.Name                 = name;
    desc.Size                 = static_cast<Diligent::Uint64>(elementStride) * elementCount;
    desc.BindFlags            = bindFlags;
    desc.Usage                = usage;
    desc.CPUAccessFlags       = cpuAccess;
    desc.ImmediateContextMask = immediateContextMask;
    if (usage != Diligent::USAGE_STAGING)
    {
        desc.Mode              = Diligent::BUFFER_MODE_STRUCTURED;
        desc.ElementByteStride = elementStride;
    }

    renderDevice->CreateBuffer(desc, nullptr, &outBuffer);
    return outBuffer != nullptr;
}

std::uint32_t dispatchGroupCount(std::uint32_t threadCount)
{
    return (threadCount + kComputeThreadGroupSize - 1u) / kComputeThreadGroupSize;
}

std::vector<std::uint32_t> buildReductionLevelCounts(std::uint32_t elementCount)
{
    std::vector<std::uint32_t> counts;
    std::uint32_t levelCount = std::max<std::uint32_t>(elementCount, 1u);
    do
    {
        levelCount = dispatchGroupCount(levelCount);
        counts.push_back(std::max<std::uint32_t>(levelCount, 1u));
    } while (levelCount > 1u);
    return counts;
}

} // namespace

bool PhysicsSceneGpuState::ensureCapacity(Diligent::IRenderDevice *renderDevice,
                                          std::uint32_t bodyCount, std::uint32_t colliderCount,
                                          std::uint32_t physicsContextId)
{
    const bool hasAllBuffers =
        mPersistentRigidBodies.positionsBuffer != nullptr &&
        mPersistentRigidBodies.orientationsBuffer != nullptr &&
        mPersistentRigidBodies.scalesBuffer != nullptr &&
        mPersistentRigidBodies.linearVelocitiesBuffer != nullptr &&
        mPersistentRigidBodies.angularVelocitiesBuffer != nullptr &&
        mPersistentRigidBodies.inverseInertiaLocalBuffer != nullptr &&
        mPersistentRigidBodies.bodyTypesBuffer != nullptr &&
        mPersistentRigidBodies.kinematicTargetPositionsBuffer != nullptr &&
        mPersistentRigidBodies.kinematicTargetOrientationsBuffer != nullptr &&
        mPersistentRigidBodies.kinematicTargetFlagsBuffer != nullptr &&
        mPersistentColliders.ownerRigidBodyIndicesBuffer != nullptr &&
        mPersistentColliders.broadPhaseDataBuffer != nullptr &&
        mPersistentColliders.shapeTypesBuffer != nullptr &&
        mPersistentColliders.shapeParamsBuffer != nullptr &&
        mPersistentColliders.localPositionsBuffer != nullptr &&
        mPersistentColliders.localOrientationsBuffer != nullptr &&
        mPersistentColliders.enabledFlagsBuffer != nullptr &&
        mPersistentColliders.materialBuffer != nullptr &&
        mTransientState.predictedRigidBodies.positionsBuffer != nullptr &&
        mTransientState.predictedRigidBodies.orientationsBuffer != nullptr &&
        mTransientState.predictedRigidBodies.linearVelocitiesBuffer != nullptr &&
        mTransientState.predictedRigidBodies.angularVelocitiesBuffer != nullptr &&
        mTransientState.previousRigidBodies.positionsBuffer != nullptr &&
        mTransientState.previousRigidBodies.orientationsBuffer != nullptr &&
        mTransientState.bodyAabbsBuffer != nullptr && mTransientState.bodyMetaBuffer != nullptr &&
        mTransientState.activeBodyFlagsBuffer != nullptr &&
        mTransientState.activeBodyOffsetsBuffer != nullptr &&
        mTransientState.activeBodyIndicesBuffer != nullptr &&
        mTransientState.staticBodyFlagsBuffer != nullptr &&
        mTransientState.staticBodyOffsetsBuffer != nullptr &&
        mTransientState.staticBodyIndicesBuffer != nullptr &&
        mTransientState.broadPhaseElementsBuffer != nullptr &&
        mTransientState.mortonCodesBuffer != nullptr &&
        mTransientState.mortonCodesScratchBuffer != nullptr &&
        mTransientState.globalBroadPhaseExtentBuffer != nullptr &&
        mTransientState.staticBroadPhaseElementsBuffer != nullptr &&
        mTransientState.staticMortonCodesBuffer != nullptr &&
        mTransientState.staticMortonCodesScratchBuffer != nullptr &&
        mTransientState.staticGlobalBroadPhaseExtentBuffer != nullptr &&
        !mTransientState.scanBlockSumsBuffers.empty() &&
        mTransientState.scanBlockSumsBuffers.front() != nullptr &&
        !mTransientState.scanScannedBlockSumsBuffers.empty() &&
        mTransientState.scanScannedBlockSumsBuffers.front() != nullptr &&
        !mTransientState.broadPhaseExtentScratchBuffers.empty() &&
        mTransientState.broadPhaseExtentScratchBuffers.front() != nullptr &&
        !mTransientState.staticScanBlockSumsBuffers.empty() &&
        mTransientState.staticScanBlockSumsBuffers.front() != nullptr &&
        !mTransientState.staticScanScannedBlockSumsBuffers.empty() &&
        mTransientState.staticScanScannedBlockSumsBuffers.front() != nullptr &&
        !mTransientState.staticBroadPhaseExtentScratchBuffers.empty() &&
        mTransientState.staticBroadPhaseExtentScratchBuffers.front() != nullptr &&
        mTransientState.radixBitFlagsBuffer != nullptr &&
        mTransientState.radixBitOffsetsBuffer != nullptr &&
        mTransientState.radixMetaBuffer != nullptr && mTransientState.bvhBuffer != nullptr &&
        mTransientState.bvhConstructionInfoBuffer != nullptr &&
        mTransientState.staticRadixBitFlagsBuffer != nullptr &&
        mTransientState.staticRadixBitOffsetsBuffer != nullptr &&
        mTransientState.staticRadixMetaBuffer != nullptr &&
        mTransientState.staticBvhBuffer != nullptr &&
        mTransientState.staticBvhConstructionInfoBuffer != nullptr &&
        mTransientState.pairCountBuffers[0] != nullptr &&
        mTransientState.pairCountBuffers[1] != nullptr &&
        mTransientState.pairCountBuffers[2] != nullptr &&
        mTransientState.pairCountBuffers[3] != nullptr &&
        mTransientState.pairCountBuffers[4] != nullptr &&
        mTransientState.pairCountBuffers[5] != nullptr &&
        mTransientState.pairOffsetBuffers[0] != nullptr &&
        mTransientState.pairOffsetBuffers[1] != nullptr &&
        mTransientState.pairOffsetBuffers[2] != nullptr &&
        mTransientState.pairOffsetBuffers[3] != nullptr &&
        mTransientState.pairOffsetBuffers[4] != nullptr &&
        mTransientState.pairOffsetBuffers[5] != nullptr &&
        mTransientState.rigidPairRangesBuffer != nullptr &&
        mTransientState.candidatePairsBuffer != nullptr &&
        mTransientState.broadPhaseMetaBuffer != nullptr &&
        mTransientState.narrowPhaseChunksBuffer != nullptr &&
        mTransientState.narrowPhaseMetaBuffer != nullptr &&
        mTransientState.narrowPhaseChunkCounterBuffer != nullptr &&
        mTransientState.contactsBuffer != nullptr &&
        mTransientState.translationCorrectionsBuffer != nullptr &&
        mTransientState.rotationCorrectionsBuffer != nullptr &&
        mTransientState.linearVelocityCorrectionsBuffer != nullptr &&
        mTransientState.angularVelocityCorrectionsBuffer != nullptr &&
        mReadbackRigidBodies.positionsBuffer != nullptr &&
        mReadbackRigidBodies.orientationsBuffer != nullptr &&
        mReadbackRigidBodies.linearVelocitiesBuffer != nullptr &&
        mReadbackRigidBodies.angularVelocitiesBuffer != nullptr &&
        mReadbackRigidBodies.broadPhaseMetaBuffer != nullptr;
    if (hasAllBuffers && mRigidBodyCapacity >= bodyCount && mColliderCapacity >= colliderCount)
    {
        return true;
    }

    const std::uint32_t newRigidBodyCapacity = std::max<std::uint32_t>(bodyCount, 64u);
    const std::uint32_t newColliderCapacity  = std::max<std::uint32_t>(colliderCount, 64u);
    const std::uint32_t newNodeCapacity      = std::max<std::uint32_t>(
        newColliderCapacity > 0u ? (newColliderCapacity * 2u - 1u) : 1u, 1u);
    const std::uint32_t newCandidatePairCapacity =
        estimateRigidCandidatePairCapacity(newColliderCapacity);
    const std::uint32_t newChunkCapacity = std::max<std::uint32_t>(
        (newCandidatePairCapacity + kNarrowPhaseChunkSize - 1u) / kNarrowPhaseChunkSize, 1u);
    const std::uint32_t newContactCapacity = std::max<std::uint32_t>(
        newCandidatePairCapacity * kRigidContactsPerPair, kRigidContactsPerPair);
    const Diligent::Uint64 contextMask = contextMaskForId(physicsContextId);
    const std::vector<std::uint32_t> reductionLevelCounts =
        buildReductionLevelCounts(newColliderCapacity);

    if (!ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.PositionsInvMass", sizeof(Diligent::float4),
            newRigidBodyCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.Orientations", sizeof(Diligent::float4),
            newRigidBodyCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Scales", sizeof(Diligent::float4),
                                newRigidBodyCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.scalesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.LinearVelocities",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.AngularVelocities",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.angularVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.InverseInertiaLocal",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.inverseInertiaLocalBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BodyTypes", sizeof(std::uint32_t),
                                newRigidBodyCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.bodyTypesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.KinematicTargetPositions",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.kinematicTargetPositionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.KinematicTargetOrientations",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.kinematicTargetOrientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.KinematicTargetFlags",
                                sizeof(std::uint32_t), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.kinematicTargetFlagsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderOwnerBodyIndices",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentColliders.ownerRigidBodyIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderBroadPhaseData",
                                sizeof(GpuColliderBroadPhaseData), newColliderCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentColliders.broadPhaseDataBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ColliderShapeTypes", sizeof(std::uint32_t),
            newColliderCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentColliders.shapeTypesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ColliderShapeParams", sizeof(Diligent::float4),
            newColliderCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentColliders.shapeParamsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ColliderLocalPositions", sizeof(Diligent::float4),
            newColliderCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentColliders.localPositionsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ColliderLocalOrientations", sizeof(Diligent::float4),
            newColliderCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentColliders.localOrientationsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ColliderEnabledFlags", sizeof(std::uint32_t),
            newColliderCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentColliders.enabledFlagsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ColliderMaterials", sizeof(Diligent::float4),
            newColliderCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentColliders.materialBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PreviousPositionsInvMass",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.previousRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PreviousOrientations",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.previousRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedPositionsInvMass",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.predictedRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedOrientations",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.predictedRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedLinearVelocities",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.predictedRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedAngularVelocities",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.predictedRigidBodies.angularVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderAabbs",
                                sizeof(GpuBodyAabb), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.bodyAabbsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderMeta",
                                sizeof(GpuBodyMeta), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.bodyMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveBodyFlags",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.activeBodyFlagsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveBodyOffsets",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.activeBodyOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveBodyIndices",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.activeBodyIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticBodyFlags",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticBodyFlagsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticBodyOffsets",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticBodyOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticBodyIndices",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticBodyIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BroadPhaseElements",
                                sizeof(GpuBroadPhaseElement), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.broadPhaseElementsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.MortonCodes",
                                sizeof(GpuMortonCodeElement), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.mortonCodesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.MortonCodesScratch",
                                sizeof(GpuMortonCodeElement), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.mortonCodesScratchBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.GlobalBroadPhaseExtent",
                                sizeof(GpuBroadPhaseExtent), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.globalBroadPhaseExtentBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticBroadPhaseElements",
                                sizeof(GpuBroadPhaseElement), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticBroadPhaseElementsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticMortonCodes",
                                sizeof(GpuMortonCodeElement), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticMortonCodesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticMortonCodesScratch",
                                sizeof(GpuMortonCodeElement), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticMortonCodesScratchBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticGlobalBroadPhaseExtent",
                                sizeof(GpuBroadPhaseExtent), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticGlobalBroadPhaseExtentBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RadixBitFlags",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.radixBitFlagsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RadixBitOffsets",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.radixBitOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RadixMeta", sizeof(std::uint32_t),
                                1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.radixMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BvhNodes", sizeof(GpuBvhNode),
                                newNodeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.bvhBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BvhConstructionInfos",
                                sizeof(GpuBvhConstructionInfo), newNodeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.bvhConstructionInfoBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticRadixBitFlags",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticRadixBitFlagsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticRadixBitOffsets",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticRadixBitOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticRadixMeta",
                                sizeof(std::uint32_t), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticRadixMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticBvhNodes",
                                sizeof(GpuBvhNode), newNodeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticBvhBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StaticBvhConstructionInfos",
                                sizeof(GpuBvhConstructionInfo), newNodeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.staticBvhConstructionInfoBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairCountsSphereSphere",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairCountBuffers[0]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairCountsSphereBox",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairCountBuffers[1]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairCountsSphereCapsule",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairCountBuffers[2]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairCountsBoxBox",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairCountBuffers[3]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairCountsBoxCapsule",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairCountBuffers[4]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairCountsCapsuleCapsule",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairCountBuffers[5]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairOffsetsSphereSphere",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairOffsetBuffers[0]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairOffsetsSphereBox",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairOffsetBuffers[1]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairOffsetsSphereCapsule",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairOffsetBuffers[2]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairOffsetsBoxBox",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairOffsetBuffers[3]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairOffsetsBoxCapsule",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairOffsetBuffers[4]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairOffsetsCapsuleCapsule",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.pairOffsetBuffers[5]) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidPairRanges",
                                sizeof(GpuRigidPairRange), kRigidPairTypeCount,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rigidPairRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.CandidatePairs",
                                sizeof(GpuCandidatePair), newCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.candidatePairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BroadPhaseMeta",
                                sizeof(GpuBroadPhaseMeta), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.broadPhaseMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.NarrowPhaseChunks",
                                sizeof(GpuNarrowPhaseChunk), newChunkCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.narrowPhaseChunksBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.NarrowPhaseMeta",
                                sizeof(GpuNarrowPhaseMeta), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.narrowPhaseMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.NarrowPhaseChunkCounter",
                                sizeof(std::uint32_t), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.narrowPhaseChunkCounterBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidContacts",
                                sizeof(GpuRigidContact), newContactCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.contactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.TranslationCorrections",
                                sizeof(std::int32_t) * 4u, newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.translationCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RotationCorrections",
                                sizeof(std::int32_t) * 4u, newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rotationCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.LinearVelocityCorrections",
                                sizeof(std::int32_t) * 4u, newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.linearVelocityCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.AngularVelocityCorrections",
                                sizeof(std::int32_t) * 4u, newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.angularVelocityCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedPositions.Readback",
                                sizeof(Diligent::float4), newRigidBodyCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedOrientations.Readback",
                                sizeof(Diligent::float4), newRigidBodyCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice,
                                "CRESSimNeo.Physics.PredictedLinearVelocities.Readback",
                                sizeof(Diligent::float4), newRigidBodyCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice,
                                "CRESSimNeo.Physics.PredictedAngularVelocities.Readback",
                                sizeof(Diligent::float4), newRigidBodyCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackRigidBodies.angularVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BroadPhaseMeta.Readback",
                                sizeof(GpuBroadPhaseMeta), 1u, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackRigidBodies.broadPhaseMetaBuffer))
    {
        return false;
    }

    mTransientState.scanBlockSumsBuffers.resize(reductionLevelCounts.size());
    mTransientState.scanScannedBlockSumsBuffers.resize(reductionLevelCounts.size());
    mTransientState.broadPhaseExtentScratchBuffers.resize(reductionLevelCounts.size());
    mTransientState.staticScanBlockSumsBuffers.resize(reductionLevelCounts.size());
    mTransientState.staticScanScannedBlockSumsBuffers.resize(reductionLevelCounts.size());
    mTransientState.staticBroadPhaseExtentScratchBuffers.resize(reductionLevelCounts.size());
    for (std::size_t level = 0; level < reductionLevelCounts.size(); ++level)
    {
        const std::uint32_t levelCount = reductionLevelCounts[level];
        const std::string scanSumsName =
            "CRESSimNeo.Physics.ScanBlockSums." + std::to_string(level);
        const std::string scanOffsetsName =
            "CRESSimNeo.Physics.ScanScannedBlockSums." + std::to_string(level);
        const std::string extentName =
            "CRESSimNeo.Physics.BroadPhaseExtentScratch." + std::to_string(level);
        const std::string staticScanSumsName =
            "CRESSimNeo.Physics.StaticScanBlockSums." + std::to_string(level);
        const std::string staticScanOffsetsName =
            "CRESSimNeo.Physics.StaticScanScannedBlockSums." + std::to_string(level);
        const std::string staticExtentName =
            "CRESSimNeo.Physics.StaticBroadPhaseExtentScratch." + std::to_string(level);
        if (!ensureStructuredBuffer(
                renderDevice, scanSumsName.c_str(), sizeof(std::uint32_t), levelCount,
                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mTransientState.scanBlockSumsBuffers[level]) ||
            !ensureStructuredBuffer(
                renderDevice, scanOffsetsName.c_str(), sizeof(std::uint32_t), levelCount,
                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mTransientState.scanScannedBlockSumsBuffers[level]) ||
            !ensureStructuredBuffer(
                renderDevice, extentName.c_str(), sizeof(GpuBroadPhaseExtent), levelCount,
                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mTransientState.broadPhaseExtentScratchBuffers[level]) ||
            !ensureStructuredBuffer(
                renderDevice, staticScanSumsName.c_str(), sizeof(std::uint32_t), levelCount,
                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mTransientState.staticScanBlockSumsBuffers[level]) ||
            !ensureStructuredBuffer(
                renderDevice, staticScanOffsetsName.c_str(), sizeof(std::uint32_t), levelCount,
                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mTransientState.staticScanScannedBlockSumsBuffers[level]) ||
            !ensureStructuredBuffer(
                renderDevice, staticExtentName.c_str(), sizeof(GpuBroadPhaseExtent), levelCount,
                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                mTransientState.staticBroadPhaseExtentScratchBuffers[level]))
        {
            return false;
        }
    }

    mRigidBodyCapacity          = newRigidBodyCapacity;
    mColliderCapacity           = newColliderCapacity;
    mBroadPhaseNodeCapacity     = newNodeCapacity;
    mCandidatePairCapacity      = newCandidatePairCapacity;
    mContactCapacity            = newContactCapacity;
    mCorrectionBuffersNeedClear = true;
    mStaticBroadPhaseDirty      = true;
    return true;
}

bool PhysicsSceneGpuState::uploadWorldState(Diligent::IDeviceContext *computeContext,
                                            PhysicsWorld &world, std::uint32_t bodyCount,
                                            std::uint32_t colliderCount)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    world.ensureDerivedStateUpToDate();

    const RigidBodySoAHost &rigidBodies = world.rigidBodySoA();
    const ColliderSoAHost &colliders    = world.colliderSoA();
    if (static_cast<std::uint32_t>(rigidBodies.size()) != bodyCount ||
        static_cast<std::uint32_t>(colliders.size()) != colliderCount)
    {
        return false;
    }

    if (bodyCount == 0u && colliderCount == 0u)
    {
        mRigidBodyCount        = 0u;
        mColliderCount         = 0u;
        mStaticBroadPhaseDirty = mStaticBroadPhaseDirty || world.staticBroadPhaseDirty();
        world.clearStaticBroadPhaseDirty();
        return true;
    }

    const Diligent::Uint64 bodyFloat4Bytes =
        static_cast<Diligent::Uint64>(bodyCount) * sizeof(Diligent::float4);
    const Diligent::Uint64 bodyUintBytes =
        static_cast<Diligent::Uint64>(bodyCount) * sizeof(std::uint32_t);
    const Diligent::Uint64 colliderFloat4Bytes =
        static_cast<Diligent::Uint64>(colliderCount) * sizeof(Diligent::float4);
    const Diligent::Uint64 colliderUintBytes =
        static_cast<Diligent::Uint64>(colliderCount) * sizeof(std::uint32_t);

    if (bodyCount > 0u)
    {
        computeContext->UpdateBuffer(mPersistentRigidBodies.positionsBuffer, 0u, bodyFloat4Bytes,
                                     rigidBodies.positionsInvMass.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentRigidBodies.orientationsBuffer, 0u, bodyFloat4Bytes,
                                     rigidBodies.orientations.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentRigidBodies.scalesBuffer, 0u, bodyFloat4Bytes,
                                     rigidBodies.scales.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentRigidBodies.linearVelocitiesBuffer, 0u,
                                     bodyFloat4Bytes, rigidBodies.linearVelocities.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentRigidBodies.angularVelocitiesBuffer, 0u,
                                     bodyFloat4Bytes, rigidBodies.angularVelocities.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentRigidBodies.inverseInertiaLocalBuffer, 0u,
                                     bodyFloat4Bytes, rigidBodies.inverseInertiaLocal.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentRigidBodies.bodyTypesBuffer, 0u, bodyUintBytes,
                                     rigidBodies.bodyTypes.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentRigidBodies.kinematicTargetPositionsBuffer, 0u,
                                     bodyFloat4Bytes, rigidBodies.kinematicTargetPositions.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentRigidBodies.kinematicTargetOrientationsBuffer, 0u,
                                     bodyFloat4Bytes,
                                     rigidBodies.kinematicTargetOrientations.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentRigidBodies.kinematicTargetFlagsBuffer, 0u,
                                     bodyUintBytes, rigidBodies.kinematicTargetFlags.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    if (colliderCount > 0u)
    {
        std::vector<GpuColliderBroadPhaseData> broadPhaseData(colliderCount);
        for (std::uint32_t i = 0; i < colliderCount; ++i)
        {
            GpuColliderBroadPhaseData &entry = broadPhaseData[i];
            entry.ownerBody                  = colliders.ownerRigidBodyIndices[i];
            entry.shapeType                  = colliders.shapeTypes[i];
            entry.environmentIndex           = colliders.environmentIndices[i];
            entry.collisionLayer             = colliders.collisionLayers[i];
            entry.collisionMask              = colliders.collisionMasks[i];
        }

        computeContext->UpdateBuffer(mPersistentColliders.ownerRigidBodyIndicesBuffer, 0u,
                                     colliderUintBytes, colliders.ownerRigidBodyIndices.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(
            mPersistentColliders.broadPhaseDataBuffer, 0u,
            static_cast<Diligent::Uint64>(colliderCount) * sizeof(GpuColliderBroadPhaseData),
            broadPhaseData.data(), Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentColliders.shapeTypesBuffer, 0u, colliderUintBytes,
                                     colliders.shapeTypes.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentColliders.shapeParamsBuffer, 0u,
                                     colliderFloat4Bytes, colliders.shapeParams.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentColliders.localPositionsBuffer, 0u,
                                     colliderFloat4Bytes, colliders.localPositions.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentColliders.localOrientationsBuffer, 0u,
                                     colliderFloat4Bytes, colliders.localOrientations.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentColliders.enabledFlagsBuffer, 0u, colliderUintBytes,
                                     colliders.enabledFlags.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->UpdateBuffer(mPersistentColliders.materialBuffer, 0u, colliderFloat4Bytes,
                                     colliders.frictionRestitution.data(),
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    world.clearRigidBodyDirtyRange();
    world.clearColliderDirtyRange();
    mRigidBodyCount        = bodyCount;
    mColliderCount         = colliderCount;
    mStaticBroadPhaseDirty = mStaticBroadPhaseDirty || world.staticBroadPhaseDirty();
    world.clearStaticBroadPhaseDirty();
    return true;
}

bool PhysicsSceneGpuState::copyPredictedRigidBodiesToPersistentState(
    Diligent::IDeviceContext *computeContext, std::uint32_t bodyCount)
{
    if (computeContext == nullptr || bodyCount == 0u)
    {
        return false;
    }

    const Diligent::Uint64 bytes =
        static_cast<Diligent::Uint64>(bodyCount) * sizeof(Diligent::float4);
    computeContext->CopyBuffer(mTransientState.predictedRigidBodies.positionsBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mPersistentRigidBodies.positionsBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mTransientState.predictedRigidBodies.orientationsBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mPersistentRigidBodies.orientationsBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mTransientState.predictedRigidBodies.linearVelocitiesBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mPersistentRigidBodies.linearVelocitiesBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mTransientState.predictedRigidBodies.angularVelocitiesBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mPersistentRigidBodies.angularVelocitiesBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return true;
}

bool PhysicsSceneGpuState::readbackBroadPhaseMetaBlocking(Diligent::IDeviceContext *computeContext,
                                                          GpuBroadPhaseMeta &outMeta)
{
    if (computeContext == nullptr || mTransientState.broadPhaseMetaBuffer == nullptr ||
        mReadbackRigidBodies.broadPhaseMetaBuffer == nullptr)
    {
        return false;
    }

    computeContext->CopyBuffer(mTransientState.broadPhaseMetaBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackRigidBodies.broadPhaseMetaBuffer, 0u,
                               sizeof(GpuBroadPhaseMeta),
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->Flush();
    computeContext->WaitForIdle();

    void *mappedMeta = nullptr;
    computeContext->MapBuffer(mReadbackRigidBodies.broadPhaseMetaBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedMeta);
    if (mappedMeta == nullptr)
    {
        return false;
    }

    outMeta = *static_cast<const GpuBroadPhaseMeta *>(mappedMeta);
    computeContext->UnmapBuffer(mReadbackRigidBodies.broadPhaseMetaBuffer, Diligent::MAP_READ);
    return true;
}

bool PhysicsSceneGpuState::readbackPredictedRigidStateBlocking(
    Diligent::IDeviceContext *computeContext, PhysicsWorld &world, std::uint32_t bodyCount)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (bodyCount == 0u)
    {
        return true;
    }

    const Diligent::Uint64 bytes =
        static_cast<Diligent::Uint64>(bodyCount) * sizeof(Diligent::float4);
    computeContext->CopyBuffer(mTransientState.predictedRigidBodies.positionsBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackRigidBodies.positionsBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mTransientState.predictedRigidBodies.orientationsBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackRigidBodies.orientationsBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mTransientState.predictedRigidBodies.linearVelocitiesBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackRigidBodies.linearVelocitiesBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mTransientState.predictedRigidBodies.angularVelocitiesBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackRigidBodies.angularVelocitiesBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    computeContext->Flush();
    computeContext->WaitForIdle();

    void *mappedPositions    = nullptr;
    void *mappedOrientations = nullptr;
    void *mappedLinear       = nullptr;
    void *mappedAngular      = nullptr;

    computeContext->MapBuffer(mReadbackRigidBodies.positionsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedPositions);
    computeContext->MapBuffer(mReadbackRigidBodies.orientationsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedOrientations);
    computeContext->MapBuffer(mReadbackRigidBodies.linearVelocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedLinear);
    computeContext->MapBuffer(mReadbackRigidBodies.angularVelocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedAngular);

    if (mappedPositions == nullptr || mappedOrientations == nullptr || mappedLinear == nullptr ||
        mappedAngular == nullptr)
    {
        if (mappedPositions != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackRigidBodies.positionsBuffer, Diligent::MAP_READ);
        }
        if (mappedOrientations != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackRigidBodies.orientationsBuffer,
                                        Diligent::MAP_READ);
        }
        if (mappedLinear != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackRigidBodies.linearVelocitiesBuffer,
                                        Diligent::MAP_READ);
        }
        if (mappedAngular != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackRigidBodies.angularVelocitiesBuffer,
                                        Diligent::MAP_READ);
        }
        return false;
    }

    const auto *positions         = static_cast<const Diligent::float4 *>(mappedPositions);
    const auto *orientations      = static_cast<const Diligent::float4 *>(mappedOrientations);
    const auto *linearVelocities  = static_cast<const Diligent::float4 *>(mappedLinear);
    const auto *angularVelocities = static_cast<const Diligent::float4 *>(mappedAngular);
    for (std::uint32_t i = 0; i < bodyCount; ++i)
    {
        (void)world.writeBackRigidBodyState(i, positions[i], orientations[i], linearVelocities[i],
                                            angularVelocities[i]);
    }
    world.finalizeRigidBodyWriteback();

    computeContext->UnmapBuffer(mReadbackRigidBodies.positionsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackRigidBodies.orientationsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackRigidBodies.linearVelocitiesBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackRigidBodies.angularVelocitiesBuffer, Diligent::MAP_READ);
    return true;
}

const PhysicsSceneGpuState::PersistentRigidBodyBuffers &PhysicsSceneGpuState::
    persistentRigidBodies() const noexcept
{
    return mPersistentRigidBodies;
}

const PhysicsSceneGpuState::PersistentColliderBuffers &PhysicsSceneGpuState::persistentColliders()
    const noexcept
{
    return mPersistentColliders;
}

const PhysicsSceneGpuState::SolverTransientBuffers &PhysicsSceneGpuState::transientBuffers()
    const noexcept
{
    return mTransientState;
}

std::uint32_t PhysicsSceneGpuState::candidatePairCapacity() const noexcept
{
    return mCandidatePairCapacity;
}

bool PhysicsSceneGpuState::correctionBuffersNeedClear() const noexcept
{
    return mCorrectionBuffersNeedClear;
}

void PhysicsSceneGpuState::setCorrectionBuffersNeedClear(bool needClear) noexcept
{
    mCorrectionBuffersNeedClear = needClear;
}

bool PhysicsSceneGpuState::staticBroadPhaseDirty() const noexcept
{
    return mStaticBroadPhaseDirty;
}

void PhysicsSceneGpuState::setStaticBroadPhaseDirty(bool dirty) noexcept
{
    mStaticBroadPhaseDirty = dirty;
}

PhysicsGpuSceneView PhysicsSceneGpuState::sceneView() const noexcept
{
    PhysicsGpuSceneView view{};
    view.rigid.poses.positionsBuffer    = mTransientState.predictedRigidBodies.positionsBuffer;
    view.rigid.poses.orientationsBuffer = mTransientState.predictedRigidBodies.orientationsBuffer;
    view.rigid.poses.scalesBuffer       = mPersistentRigidBodies.scalesBuffer;
    view.rigid.poses.count              = mRigidBodyCount;
    view.rigid.colliderCount            = mColliderCount;
    return view;
}

} // namespace cressim::neo::physics
