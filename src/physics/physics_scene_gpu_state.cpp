#include "physics/physics_scene_gpu_state.h"

#include "gpu/gpu_buffer_utils.h"
#include "gpu/gpu_types.h"
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

std::uint32_t nextPowerOfTwo(std::uint32_t value) noexcept
{
    if (value <= 1u)
    {
        return 1u;
    }

    --value;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    return value + 1u;
}

bool ensureStructuredBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                            std::uint32_t elementStride, std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags, Diligent::USAGE usage,
                            Diligent::CPU_ACCESS_FLAGS cpuAccess,
                            Diligent::Uint64 immediateContextMask,
                            Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer)
{
    std::uint32_t capacity = 0u;
    return gpu::detail::ensureStructuredBufferCapacity(
        renderDevice, name, elementStride, elementCount, elementCount, bindFlags, usage, cpuAccess,
        immediateContextMask, outBuffer, capacity);
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

template <typename T>
bool updateStructuredBufferRange(Diligent::IDeviceContext *computeContext,
                                 Diligent::IBuffer *buffer, const std::vector<T> &source,
                                 std::uint32_t begin, std::uint32_t count)
{
    if (computeContext == nullptr || buffer == nullptr)
    {
        return false;
    }
    if (count == 0u)
    {
        return true;
    }

    computeContext->UpdateBuffer(buffer, static_cast<Diligent::Uint64>(begin) * sizeof(T),
                                 static_cast<Diligent::Uint64>(count) * sizeof(T),
                                 source.data() + begin,
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return true;
}

template <typename Uploader>
bool uploadContiguousRuns(const std::vector<std::uint32_t> &dirtyIndices, Uploader &&uploader)
{
    if (dirtyIndices.empty())
    {
        return true;
    }

    std::vector<std::uint32_t> sortedIndices = dirtyIndices;
    std::sort(sortedIndices.begin(), sortedIndices.end());

    std::uint32_t runBegin = sortedIndices.front();
    std::uint32_t runCount = 1u;
    for (std::size_t i = 1; i < sortedIndices.size(); ++i)
    {
        if (sortedIndices[i] == sortedIndices[i - 1] + 1u)
        {
            ++runCount;
            continue;
        }

        if (!uploader(runBegin, runCount))
        {
            return false;
        }
        runBegin = sortedIndices[i];
        runCount = 1u;
    }

    return uploader(runBegin, runCount);
}

} // namespace

bool PhysicsSceneGpuState::ensureCapacity(Diligent::IRenderDevice *renderDevice,
                                          std::uint32_t bodyCount, std::uint32_t colliderCount,
                                          std::uint32_t softParticleCount,
                                          std::uint32_t softEdgeCount, std::uint32_t softTetCount,
                                          std::uint32_t rigidSurfaceParticleCount,
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
        mPersistentBodyColliderMapping.colliderOffsetsBuffer != nullptr &&
        mPersistentBodyColliderMapping.colliderCountsBuffer != nullptr &&
        mPersistentBodyColliderMapping.colliderRangesBuffer != nullptr &&
        mPersistentBodyColliderMapping.colliderIndicesBuffer != nullptr &&
        mPersistentSoftParticles.positionsInvMassBuffer != nullptr &&
        mPersistentSoftParticles.previousPositionsBuffer != nullptr &&
        mPersistentSoftParticles.velocitiesBuffer != nullptr &&
        mPersistentSoftParticles.radiiBuffer != nullptr &&
        mPersistentSoftParticles.environmentIndicesBuffer != nullptr &&
        mPersistentSoftParticles.owningSoftBodyIndicesBuffer != nullptr &&
        mPersistentSoftParticles.phasesBuffer != nullptr &&
        mPersistentSoftParticles.collisionLayersBuffer != nullptr &&
        mPersistentSoftParticles.collisionMasksBuffer != nullptr &&
        mPersistentSoftParticles.adjacencyOffsetsBuffer != nullptr &&
        mPersistentSoftParticles.adjacencyCountsBuffer != nullptr &&
        mPersistentSoftParticles.adjacencyIndicesBuffer != nullptr &&
        mPersistentSoftParticles.broadPhaseMetadataBuffer != nullptr &&
        mPersistentSoftTopology.edgesBuffer != nullptr &&
        mPersistentSoftTopology.tetsBuffer != nullptr &&
        mPersistentRigidSurfaceParticles.localPositionsBuffer != nullptr &&
        mTransientState.rigidSurfaceWorldPositionsBuffer != nullptr &&
        mPersistentRigidSurfaceParticles.owningRigidBodyIndicesBuffer != nullptr &&
        mPersistentRigidSurfaceParticles.owningColliderIndicesBuffer != nullptr &&
        mPersistentRigidSurfaceParticles.sampleRadiiBuffer != nullptr &&
        mPersistentRigidSurfaceParticles.environmentIndicesBuffer != nullptr &&
        mPersistentRigidSurfaceParticles.collisionLayersBuffer != nullptr &&
        mPersistentRigidSurfaceParticles.collisionMasksBuffer != nullptr &&
        mTransientState.predictedRigidBodies.positionsBuffer != nullptr &&
        mTransientState.predictedRigidBodies.orientationsBuffer != nullptr &&
        mTransientState.predictedRigidBodies.linearVelocitiesBuffer != nullptr &&
        mTransientState.predictedRigidBodies.angularVelocitiesBuffer != nullptr &&
        mTransientState.previousRigidBodies.positionsBuffer != nullptr &&
        mTransientState.previousRigidBodies.orientationsBuffer != nullptr &&
        mTransientState.particleBroadPhaseEntriesBuffer != nullptr &&
        mTransientState.particleBroadPhaseKeysBuffer != nullptr &&
        mTransientState.particleBroadPhaseKeysScratchBuffer != nullptr &&
        mTransientState.particleCellRangesBuffer != nullptr &&
        mTransientState.softRadixBitFlagsBuffer != nullptr &&
        mTransientState.softRadixBitOffsetsBuffer != nullptr &&
        mTransientState.softRadixMetaBuffer != nullptr &&
        mTransientState.softNeighborMetaBuffer != nullptr &&
        mTransientState.softSoftCandidatePairsBuffer != nullptr &&
        mTransientState.softRigidCandidatePairsBuffer != nullptr &&
        mTransientState.softRigidContactsBuffer != nullptr &&
        mTransientState.softContactsBuffer != nullptr &&
        mTransientState.activeSoftRigidContactsBuffer != nullptr &&
        mTransientState.activeSoftContactsBuffer != nullptr &&
        mTransientState.softPositionCorrectionsBuffer != nullptr &&
        mTransientState.softEdgeLambdasBuffer != nullptr &&
        mTransientState.softTetLambdasBuffer != nullptr &&
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
        mTransientState.rigidContactsBuffer != nullptr &&
        mTransientState.translationCorrectionsBuffer != nullptr &&
        mTransientState.rotationCorrectionsBuffer != nullptr &&
        mTransientState.linearVelocityCorrectionsBuffer != nullptr &&
        mTransientState.angularVelocityCorrectionsBuffer != nullptr &&
        mReadbackRigidBodies.positionsBuffer != nullptr &&
        mReadbackRigidBodies.orientationsBuffer != nullptr &&
        mReadbackRigidBodies.linearVelocitiesBuffer != nullptr &&
        mReadbackRigidBodies.angularVelocitiesBuffer != nullptr &&
        mReadbackRigidBodies.broadPhaseMetaBuffer != nullptr &&
        mReadbackSoftParticles.positionsBuffer != nullptr &&
        mReadbackSoftParticles.previousPositionsBuffer != nullptr &&
        mReadbackSoftParticles.velocitiesBuffer != nullptr &&
        mReadbackSoftParticles.neighborMetaBuffer != nullptr;
    const std::uint32_t newRigidBodyCapacity    = std::max<std::uint32_t>(bodyCount, 64u);
    const std::uint32_t newColliderCapacity     = std::max<std::uint32_t>(colliderCount, 64u);
    const std::uint32_t newSoftParticleCapacity = std::max<std::uint32_t>(softParticleCount, 64u);
    const std::uint32_t newSoftEdgeCapacity     = std::max<std::uint32_t>(softEdgeCount, 64u);
    const std::uint32_t newSoftTetCapacity      = std::max<std::uint32_t>(softTetCount, 64u);
    const std::uint32_t newRigidSurfaceParticleCapacity =
        std::max<std::uint32_t>(rigidSurfaceParticleCount, 64u);
    const std::uint32_t newParticleBroadPhaseEntryCapacity =
        std::max<std::uint32_t>(softParticleCount + rigidSurfaceParticleCount, 64u);
    const std::uint32_t newParticleCellRangeCapacity =
        nextPowerOfTwo(std::max<std::uint32_t>(newParticleBroadPhaseEntryCapacity * 2u, 64u));
    const std::uint32_t newSoftCandidatePairCapacity =
        std::max<std::uint32_t>(softParticleCount * 8u, 64u);
    const std::uint32_t newSoftScanCapacity =
        std::max(newParticleBroadPhaseEntryCapacity, newSoftCandidatePairCapacity);
    const std::uint32_t newSoftParticleAdjacencyCapacity =
        std::max<std::uint32_t>(std::max(softEdgeCount * 2u, softTetCount * 12u), 64u);
    const std::uint32_t newNodeCapacity = std::max<std::uint32_t>(
        newColliderCapacity > 0u ? (newColliderCapacity * 2u - 1u) : 1u, 1u);
    const std::uint32_t newCandidatePairCapacity =
        estimateRigidCandidatePairCapacity(newColliderCapacity);
    const std::uint32_t newChunkCapacity = std::max<std::uint32_t>(
        (newCandidatePairCapacity + kNarrowPhaseChunkSize - 1u) / kNarrowPhaseChunkSize, 1u);
    const std::uint32_t newRigidContactCapacity = std::max<std::uint32_t>(
        newCandidatePairCapacity * kRigidContactsPerPair, kRigidContactsPerPair);
    const Diligent::Uint64 contextMask = gpu::contextMaskForId(physicsContextId);
    const std::vector<std::uint32_t> reductionLevelCounts =
        buildReductionLevelCounts(std::max(newColliderCapacity, newSoftScanCapacity));

    if (hasAllBuffers && mRigidBodyCapacity >= bodyCount && mColliderCapacity >= colliderCount &&
        mSoftParticleCapacity >= softParticleCount && mSoftEdgeCapacity >= softEdgeCount &&
        mSoftTetCapacity >= softTetCount &&
        mRigidSurfaceParticleCapacity >= rigidSurfaceParticleCount &&
        mParticleBroadPhaseEntryCapacity >= newParticleBroadPhaseEntryCapacity &&
        mSoftCandidatePairCapacity >= newSoftCandidatePairCapacity &&
        mSoftScanScratchCapacity >= newSoftScanCapacity)
    {
        return true;
    }

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
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BodyColliderOffsets",
                                sizeof(std::uint32_t), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentBodyColliderMapping.colliderOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BodyColliderCounts",
                                sizeof(std::uint32_t), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentBodyColliderMapping.colliderCountsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BodyColliderRanges",
                                sizeof(Diligent::uint2), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentBodyColliderMapping.colliderRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BodyColliderIndices",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentBodyColliderMapping.colliderIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftPositionsInvMass",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.positionsInvMassBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftPreviousPositions",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.previousPositionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftVelocities",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.velocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRadii", sizeof(float),
                                newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.radiiBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftEnvironmentIndices",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.environmentIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBodyIndices",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.owningSoftBodyIndicesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftPhases", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentSoftParticles.phasesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftCollisionLayers",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.collisionLayersBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftCollisionMasks",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.collisionMasksBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftAdjacencyOffsets",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.adjacencyOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftAdjacencyCounts",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.adjacencyCountsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftAdjacencyIndices",
                                sizeof(std::uint32_t), newSoftParticleAdjacencyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.adjacencyIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBroadPhaseMetadata",
                                sizeof(Diligent::uint4), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftParticles.broadPhaseMetadataBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftEdges", sizeof(SoftEdge),
                                newSoftEdgeCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.edgesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftTets", sizeof(SoftTet),
                                newSoftTetCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.tetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidSurfaceLocalPositions",
                                sizeof(Diligent::float4), newRigidSurfaceParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidSurfaceParticles.localPositionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidSurfaceBodyIndices",
                                sizeof(std::uint32_t), newRigidSurfaceParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidSurfaceParticles.owningRigidBodyIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidSurfaceColliderIndices",
                                sizeof(std::uint32_t), newRigidSurfaceParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidSurfaceParticles.owningColliderIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidSurfaceRadii", sizeof(float),
                                newRigidSurfaceParticleCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidSurfaceParticles.sampleRadiiBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidSurfaceEnvironmentIndices",
                                sizeof(std::uint32_t), newRigidSurfaceParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidSurfaceParticles.environmentIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidSurfaceCollisionLayers",
                                sizeof(std::uint32_t), newRigidSurfaceParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidSurfaceParticles.collisionLayersBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidSurfaceCollisionMasks",
                                sizeof(std::uint32_t), newRigidSurfaceParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidSurfaceParticles.collisionMasksBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidSurfaceWorldPositions",
                                sizeof(Diligent::float4), newRigidSurfaceParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rigidSurfaceWorldPositionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleBroadPhaseEntries",
                                sizeof(GpuParticleBroadPhaseEntry),
                                newParticleBroadPhaseEntryCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.particleBroadPhaseEntriesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleBroadPhaseKeys",
                                sizeof(GpuMortonCodeElement), newParticleBroadPhaseEntryCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.particleBroadPhaseKeysBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleBroadPhaseKeysScratch",
                                sizeof(GpuMortonCodeElement), newParticleBroadPhaseEntryCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.particleBroadPhaseKeysScratchBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleCellRanges",
                                sizeof(GpuParticleCellRange), newParticleCellRangeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.particleCellRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRadixBitFlags",
                                sizeof(std::uint32_t), newSoftScanCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softRadixBitFlagsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRadixBitOffsets",
                                sizeof(std::uint32_t), newSoftScanCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softRadixBitOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRadixMeta",
                                sizeof(std::uint32_t), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softRadixMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftNeighborMeta",
                                sizeof(GpuSoftNeighborMeta), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softNeighborMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftSoftCandidatePairs",
                                sizeof(GpuSoftCandidatePair), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softSoftCandidatePairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRigidCandidatePairs",
                                sizeof(GpuSoftCandidatePair), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softRigidCandidatePairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRigidContacts",
                                sizeof(GpuSoftRigidContact), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softRigidContactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftContacts",
                                sizeof(GpuSoftContact), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softContactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveSoftRigidContacts",
                                sizeof(GpuSoftRigidContact), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.activeSoftRigidContactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveSoftContacts",
                                sizeof(GpuSoftContact), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.activeSoftContactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftPositionCorrections",
                                sizeof(Diligent::int4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softPositionCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftEdgeLambdas", sizeof(float),
                                newSoftEdgeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softEdgeLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftTetLambdas", sizeof(float),
                                newSoftTetCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softTetLambdasBuffer) ||
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
                                sizeof(GpuRigidContact), newRigidContactCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rigidContactsBuffer) ||
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
                                mReadbackRigidBodies.broadPhaseMetaBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftPositions.Readback", sizeof(Diligent::float4),
            newSoftParticleCapacity, Diligent::BIND_NONE, Diligent::USAGE_STAGING,
            Diligent::CPU_ACCESS_READ, contextMask, mReadbackSoftParticles.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftPreviousPositions.Readback",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_NONE, Diligent::USAGE_STAGING,
                                Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackSoftParticles.previousPositionsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftVelocities.Readback", sizeof(Diligent::float4),
            newSoftParticleCapacity, Diligent::BIND_NONE, Diligent::USAGE_STAGING,
            Diligent::CPU_ACCESS_READ, contextMask, mReadbackSoftParticles.velocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftNeighborMeta.Readback",
                                sizeof(GpuSoftNeighborMeta), 1u, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackSoftParticles.neighborMetaBuffer))
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

    mRigidBodyCapacity               = newRigidBodyCapacity;
    mColliderCapacity                = newColliderCapacity;
    mSoftParticleCapacity            = newSoftParticleCapacity;
    mSoftEdgeCapacity                = newSoftEdgeCapacity;
    mSoftTetCapacity                 = newSoftTetCapacity;
    mRigidSurfaceParticleCapacity    = newRigidSurfaceParticleCapacity;
    mParticleBroadPhaseEntryCapacity = newParticleBroadPhaseEntryCapacity;
    mSoftCandidatePairCapacity       = newSoftCandidatePairCapacity;
    mSoftScanScratchCapacity         = newSoftScanCapacity;
    mSoftParticleAdjacencyCapacity   = newSoftParticleAdjacencyCapacity;
    mBroadPhaseNodeCapacity          = newNodeCapacity;
    mCandidatePairCapacity           = newCandidatePairCapacity;
    mContactCapacity                 = newRigidContactCapacity;
    mCorrectionBuffersNeedClear      = true;
    mStaticBroadPhaseDirty           = true;
    mRigidBodyUploadResetRequired    = true;
    mColliderUploadResetRequired     = true;
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

    const RigidBodySoAHost &rigidBodies                 = world.rigidBodySoA();
    const ColliderSoAHost &colliders                    = world.colliderSoA();
    const BodyColliderMappingHost &bodyColliderMapping  = world.bodyColliderMapping();
    const SoftParticleSoAHost &softParticles            = world.softParticles();
    const std::vector<SoftEdge> &softEdges              = world.softEdges();
    const std::vector<SoftTet> &softTets                = world.softTets();
    const RigidSurfaceParticleSoAHost &surfaceParticles = world.rigidSurfaceParticles();
    if (static_cast<std::uint32_t>(rigidBodies.size()) != bodyCount ||
        static_cast<std::uint32_t>(colliders.size()) != colliderCount)
    {
        return false;
    }

    if (bodyCount == 0u && colliderCount == 0u && softParticles.empty() && softEdges.empty() &&
        softTets.empty() && surfaceParticles.empty())
    {
        mRigidBodyCount            = 0u;
        mColliderCount             = 0u;
        mSoftBodyCount             = 0u;
        mSoftParticleCount         = 0u;
        mSoftEdgeCount             = 0u;
        mSoftTetCount              = 0u;
        mRigidSurfaceParticleCount = 0u;
        world.clearRigidBodyUploadState();
        world.clearColliderUploadState();
        mStaticBroadPhaseDirty = mStaticBroadPhaseDirty || world.staticBroadPhaseDirty();
        world.clearStaticBroadPhaseDirty();
        return true;
    }

    if (!uploadRigidBodies(computeContext, world, rigidBodies, bodyCount,
                           mRigidBodyUploadResetRequired))
    {
        return false;
    }
    if (!uploadColliders(computeContext, world, colliders, colliderCount,
                         mColliderUploadResetRequired))
    {
        return false;
    }
    if (!uploadBodyColliderMapping(computeContext, bodyColliderMapping, bodyCount, colliderCount))
    {
        return false;
    }
    if (!uploadSoftParticles(computeContext, softParticles) ||
        !uploadSoftTopology(computeContext, softEdges, softTets) ||
        !uploadRigidSurfaceParticles(computeContext, surfaceParticles))
    {
        return false;
    }

    world.clearRigidBodyUploadState();
    world.clearColliderUploadState();
    mRigidBodyCount               = bodyCount;
    mColliderCount                = colliderCount;
    mSoftBodyCount                = world.softBodyCount();
    mSoftParticleCount            = static_cast<std::uint32_t>(softParticles.size());
    mSoftEdgeCount                = static_cast<std::uint32_t>(softEdges.size());
    mSoftTetCount                 = static_cast<std::uint32_t>(softTets.size());
    mRigidSurfaceParticleCount    = static_cast<std::uint32_t>(surfaceParticles.size());
    mRigidBodyUploadResetRequired = false;
    mColliderUploadResetRequired  = false;
    mStaticBroadPhaseDirty        = mStaticBroadPhaseDirty || world.staticBroadPhaseDirty();
    world.clearStaticBroadPhaseDirty();
    return true;
}

bool PhysicsSceneGpuState::uploadRigidBodies(Diligent::IDeviceContext *computeContext,
                                             const PhysicsWorld &world,
                                             const RigidBodySoAHost &rigidBodies,
                                             std::uint32_t bodyCount, bool forceFullUpload)
{
    const bool needsFullUpload = forceFullUpload || world.fullRigidBodyUploadRequired();
    if (bodyCount == 0u)
    {
        return true;
    }
    if (needsFullUpload)
    {
        return uploadRigidBodyRange(computeContext, rigidBodies, 0u, bodyCount);
    }
    if (world.rigidBodyDirtyIndices().empty())
    {
        return true;
    }

    return uploadContiguousRuns(
        world.rigidBodyDirtyIndices(), [&](std::uint32_t begin, std::uint32_t count)
        { return uploadRigidBodyRange(computeContext, rigidBodies, begin, count); });
}

bool PhysicsSceneGpuState::uploadColliders(Diligent::IDeviceContext *computeContext,
                                           const PhysicsWorld &world,
                                           const ColliderSoAHost &colliders,
                                           std::uint32_t colliderCount, bool forceFullUpload)
{
    const bool needsFullUpload = forceFullUpload || world.fullColliderUploadRequired();
    if (colliderCount == 0u)
    {
        return true;
    }
    if (needsFullUpload)
    {
        return uploadColliderRange(computeContext, colliders, 0u, colliderCount) &&
               uploadColliderBroadPhaseRange(computeContext, colliders, 0u, colliderCount);
    }
    if (world.colliderDirtyIndices().empty())
    {
        return true;
    }

    return uploadContiguousRuns(
               world.colliderDirtyIndices(), [&](std::uint32_t begin, std::uint32_t count)
               { return uploadColliderRange(computeContext, colliders, begin, count); }) &&
           uploadContiguousRuns(
               world.colliderDirtyIndices(), [&](std::uint32_t begin, std::uint32_t count)
               { return uploadColliderBroadPhaseRange(computeContext, colliders, begin, count); });
}

bool PhysicsSceneGpuState::uploadRigidBodyRange(Diligent::IDeviceContext *computeContext,
                                                const RigidBodySoAHost &rigidBodies,
                                                std::uint32_t begin, std::uint32_t count)
{
    return updateStructuredBufferRange(computeContext, mPersistentRigidBodies.positionsBuffer,
                                       rigidBodies.positionsInvMass, begin, count) &&
           updateStructuredBufferRange(computeContext, mPersistentRigidBodies.orientationsBuffer,
                                       rigidBodies.orientations, begin, count) &&
           updateStructuredBufferRange(computeContext, mPersistentRigidBodies.scalesBuffer,
                                       rigidBodies.scales, begin, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidBodies.linearVelocitiesBuffer,
                                       rigidBodies.linearVelocities, begin, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidBodies.angularVelocitiesBuffer,
                                       rigidBodies.angularVelocities, begin, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidBodies.inverseInertiaLocalBuffer,
                                       rigidBodies.inverseInertiaLocal, begin, count) &&
           updateStructuredBufferRange(computeContext, mPersistentRigidBodies.bodyTypesBuffer,
                                       rigidBodies.bodyTypes, begin, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidBodies.kinematicTargetPositionsBuffer,
                                       rigidBodies.kinematicTargetPositions, begin, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidBodies.kinematicTargetOrientationsBuffer,
                                       rigidBodies.kinematicTargetOrientations, begin, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidBodies.kinematicTargetFlagsBuffer,
                                       rigidBodies.kinematicTargetFlags, begin, count);
}

bool PhysicsSceneGpuState::uploadColliderRange(Diligent::IDeviceContext *computeContext,
                                               const ColliderSoAHost &colliders,
                                               std::uint32_t begin, std::uint32_t count)
{
    return updateStructuredBufferRange(computeContext,
                                       mPersistentColliders.ownerRigidBodyIndicesBuffer,
                                       colliders.ownerRigidBodyIndices, begin, count) &&
           updateStructuredBufferRange(computeContext, mPersistentColliders.shapeTypesBuffer,
                                       colliders.shapeTypes, begin, count) &&
           updateStructuredBufferRange(computeContext, mPersistentColliders.shapeParamsBuffer,
                                       colliders.shapeParams, begin, count) &&
           updateStructuredBufferRange(computeContext, mPersistentColliders.localPositionsBuffer,
                                       colliders.localPositions, begin, count) &&
           updateStructuredBufferRange(computeContext, mPersistentColliders.localOrientationsBuffer,
                                       colliders.localOrientations, begin, count) &&
           updateStructuredBufferRange(computeContext, mPersistentColliders.enabledFlagsBuffer,
                                       colliders.enabledFlags, begin, count) &&
           updateStructuredBufferRange(computeContext, mPersistentColliders.materialBuffer,
                                       colliders.frictionRestitution, begin, count);
}

bool PhysicsSceneGpuState::uploadColliderBroadPhaseRange(Diligent::IDeviceContext *computeContext,
                                                         const ColliderSoAHost &colliders,
                                                         std::uint32_t begin, std::uint32_t count)
{
    if (computeContext == nullptr || mPersistentColliders.broadPhaseDataBuffer == nullptr)
    {
        return false;
    }
    if (count == 0u)
    {
        return true;
    }

    std::vector<GpuColliderBroadPhaseData> broadPhaseData(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        const std::uint32_t sourceIndex  = begin + i;
        GpuColliderBroadPhaseData &entry = broadPhaseData[i];
        entry.ownerBody                  = colliders.ownerRigidBodyIndices[sourceIndex];
        entry.shapeType                  = colliders.shapeTypes[sourceIndex];
        entry.environmentIndex           = colliders.environmentIndices[sourceIndex];
        entry.collisionLayer             = colliders.collisionLayers[sourceIndex];
        entry.collisionMask              = colliders.collisionMasks[sourceIndex];
        entry.enabledFlag                = colliders.enabledFlags[sourceIndex];
    }

    computeContext->UpdateBuffer(
        mPersistentColliders.broadPhaseDataBuffer,
        static_cast<Diligent::Uint64>(begin) * sizeof(GpuColliderBroadPhaseData),
        static_cast<Diligent::Uint64>(count) * sizeof(GpuColliderBroadPhaseData),
        broadPhaseData.data(), Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return true;
}

bool PhysicsSceneGpuState::uploadBodyColliderMapping(Diligent::IDeviceContext *computeContext,
                                                     const BodyColliderMappingHost &mapping,
                                                     std::uint32_t bodyCount,
                                                     std::uint32_t colliderCount)
{
    return updateStructuredBufferRange(computeContext,
                                       mPersistentBodyColliderMapping.colliderOffsetsBuffer,
                                       mapping.colliderOffsets, 0u, bodyCount) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentBodyColliderMapping.colliderCountsBuffer,
                                       mapping.colliderCounts, 0u, bodyCount) &&
           [&]()
    {
        std::vector<Diligent::uint2> colliderRanges(bodyCount);
        for (std::uint32_t i = 0; i < bodyCount; ++i)
        {
            colliderRanges[i] =
                Diligent::uint2{mapping.colliderOffsets[i], mapping.colliderCounts[i]};
        }
        return updateStructuredBufferRange(computeContext,
                                           mPersistentBodyColliderMapping.colliderRangesBuffer,
                                           colliderRanges, 0u, bodyCount);
    }() &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentBodyColliderMapping.colliderIndicesBuffer,
                                       mapping.colliderIndices, 0u, colliderCount);
}

bool PhysicsSceneGpuState::uploadSoftParticles(Diligent::IDeviceContext *computeContext,
                                               const SoftParticleSoAHost &softParticles)
{
    const std::uint32_t count = static_cast<std::uint32_t>(softParticles.size());
    if (count == 0u)
    {
        return true;
    }

    return updateStructuredBufferRange(computeContext,
                                       mPersistentSoftParticles.positionsInvMassBuffer,
                                       softParticles.positionsInvMass, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftParticles.previousPositionsBuffer,
                                       softParticles.previousPositions, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentSoftParticles.velocitiesBuffer,
                                       softParticles.velocities, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentSoftParticles.radiiBuffer,
                                       softParticles.radii, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftParticles.environmentIndicesBuffer,
                                       softParticles.environmentIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftParticles.owningSoftBodyIndicesBuffer,
                                       softParticles.owningSoftBodyIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentSoftParticles.phasesBuffer,
                                       softParticles.phases, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftParticles.collisionLayersBuffer,
                                       softParticles.collisionLayers, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftParticles.collisionMasksBuffer,
                                       softParticles.collisionMasks, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftParticles.adjacencyOffsetsBuffer,
                                       softParticles.adjacencyOffsets, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftParticles.adjacencyCountsBuffer,
                                       softParticles.adjacencyCounts, 0u, count) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftParticles.adjacencyIndicesBuffer,
               softParticles.adjacencyIndices, 0u,
               static_cast<std::uint32_t>(softParticles.adjacencyIndices.size())) &&
           [&]()
    {
        std::vector<Diligent::uint4> metadata(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            metadata[i] =
                Diligent::uint4{softParticles.environmentIndices[i], softParticles.phases[i],
                                softParticles.collisionLayers[i], softParticles.collisionMasks[i]};
        }
        return updateStructuredBufferRange(
            computeContext, mPersistentSoftParticles.broadPhaseMetadataBuffer, metadata, 0u, count);
    }();
}

bool PhysicsSceneGpuState::uploadSoftTopology(Diligent::IDeviceContext *computeContext,
                                              const std::vector<SoftEdge> &softEdges,
                                              const std::vector<SoftTet> &softTets)
{
    return updateStructuredBufferRange(computeContext, mPersistentSoftTopology.edgesBuffer,
                                       softEdges, 0u,
                                       static_cast<std::uint32_t>(softEdges.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentSoftTopology.tetsBuffer, softTets,
                                       0u, static_cast<std::uint32_t>(softTets.size()));
}

bool PhysicsSceneGpuState::uploadRigidSurfaceParticles(
    Diligent::IDeviceContext *computeContext, const RigidSurfaceParticleSoAHost &surfaceParticles)
{
    const std::uint32_t count = static_cast<std::uint32_t>(surfaceParticles.size());
    if (count == 0u)
    {
        return true;
    }

    return updateStructuredBufferRange(computeContext,
                                       mPersistentRigidSurfaceParticles.localPositionsBuffer,
                                       surfaceParticles.localPositions, 0u, count) &&
           updateStructuredBufferRange(
               computeContext, mPersistentRigidSurfaceParticles.owningRigidBodyIndicesBuffer,
               surfaceParticles.owningRigidBodyIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidSurfaceParticles.owningColliderIndicesBuffer,
                                       surfaceParticles.owningColliderIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidSurfaceParticles.sampleRadiiBuffer,
                                       surfaceParticles.sampleRadii, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidSurfaceParticles.environmentIndicesBuffer,
                                       surfaceParticles.environmentIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidSurfaceParticles.collisionLayersBuffer,
                                       surfaceParticles.collisionLayers, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentRigidSurfaceParticles.collisionMasksBuffer,
                                       surfaceParticles.collisionMasks, 0u, count);
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

bool PhysicsSceneGpuState::readbackPredictedSoftStateBlocking(
    Diligent::IDeviceContext *computeContext, PhysicsWorld &world, std::uint32_t softParticleCount)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (softParticleCount == 0u)
    {
        return true;
    }

    const Diligent::Uint64 bytes =
        static_cast<Diligent::Uint64>(softParticleCount) * sizeof(Diligent::float4);
    computeContext->CopyBuffer(mPersistentSoftParticles.positionsInvMassBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackSoftParticles.positionsBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mPersistentSoftParticles.previousPositionsBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackSoftParticles.previousPositionsBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mPersistentSoftParticles.velocitiesBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackSoftParticles.velocitiesBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    computeContext->Flush();
    computeContext->WaitForIdle();

    void *mappedPositions  = nullptr;
    void *mappedPrevious   = nullptr;
    void *mappedVelocities = nullptr;
    computeContext->MapBuffer(mReadbackSoftParticles.positionsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedPositions);
    computeContext->MapBuffer(mReadbackSoftParticles.previousPositionsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedPrevious);
    computeContext->MapBuffer(mReadbackSoftParticles.velocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedVelocities);

    if (mappedPositions == nullptr || mappedPrevious == nullptr || mappedVelocities == nullptr)
    {
        if (mappedPositions != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackSoftParticles.positionsBuffer, Diligent::MAP_READ);
        }
        if (mappedPrevious != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackSoftParticles.previousPositionsBuffer,
                                        Diligent::MAP_READ);
        }
        if (mappedVelocities != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackSoftParticles.velocitiesBuffer,
                                        Diligent::MAP_READ);
        }
        return false;
    }

    const auto *positions         = static_cast<const Diligent::float4 *>(mappedPositions);
    const auto *previousPositions = static_cast<const Diligent::float4 *>(mappedPrevious);
    const auto *velocities        = static_cast<const Diligent::float4 *>(mappedVelocities);
    for (std::uint32_t i = 0; i < softParticleCount; ++i)
    {
        (void)world.writeBackSoftParticleState(i, positions[i], previousPositions[i],
                                               velocities[i]);
    }
    world.finalizeSoftParticleWriteback();

    computeContext->UnmapBuffer(mReadbackSoftParticles.positionsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackSoftParticles.previousPositionsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackSoftParticles.velocitiesBuffer, Diligent::MAP_READ);
    return true;
}

bool PhysicsSceneGpuState::readbackSoftNeighborMetaBlocking(
    Diligent::IDeviceContext *computeContext, GpuSoftNeighborMeta &outMeta)
{
    if (computeContext == nullptr || mTransientState.softNeighborMetaBuffer == nullptr ||
        mReadbackSoftParticles.neighborMetaBuffer == nullptr)
    {
        return false;
    }

    computeContext->CopyBuffer(mTransientState.softNeighborMetaBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackSoftParticles.neighborMetaBuffer, 0u,
                               sizeof(GpuSoftNeighborMeta),
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->Flush();
    computeContext->WaitForIdle();

    void *mappedMeta = nullptr;
    computeContext->MapBuffer(mReadbackSoftParticles.neighborMetaBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedMeta);
    if (mappedMeta == nullptr)
    {
        return false;
    }

    outMeta = *static_cast<const GpuSoftNeighborMeta *>(mappedMeta);
    computeContext->UnmapBuffer(mReadbackSoftParticles.neighborMetaBuffer, Diligent::MAP_READ);
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

const PhysicsSceneGpuState::PersistentBodyColliderMappingBuffers &PhysicsSceneGpuState::
    persistentBodyColliderMapping() const noexcept
{
    return mPersistentBodyColliderMapping;
}

const PhysicsSceneGpuState::PersistentSoftParticleBuffers &PhysicsSceneGpuState::
    persistentSoftParticles() const noexcept
{
    return mPersistentSoftParticles;
}

const PhysicsSceneGpuState::PersistentSoftTopologyBuffers &PhysicsSceneGpuState::
    persistentSoftTopology() const noexcept
{
    return mPersistentSoftTopology;
}

const PhysicsSceneGpuState::PersistentRigidSurfaceParticleBuffers &PhysicsSceneGpuState::
    persistentRigidSurfaceParticles() const noexcept
{
    return mPersistentRigidSurfaceParticles;
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

std::uint32_t PhysicsSceneGpuState::softCandidatePairCapacity() const noexcept
{
    return mSoftCandidatePairCapacity;
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
    view.soft.particles.positionsInvMassBuffer  = mPersistentSoftParticles.positionsInvMassBuffer;
    view.soft.particles.previousPositionsBuffer = mPersistentSoftParticles.previousPositionsBuffer;
    view.soft.particles.velocitiesBuffer        = mPersistentSoftParticles.velocitiesBuffer;
    view.soft.particles.radiiBuffer             = mPersistentSoftParticles.radiiBuffer;
    view.soft.particles.environmentIndicesBuffer =
        mPersistentSoftParticles.environmentIndicesBuffer;
    view.soft.particles.owningSoftBodyIndicesBuffer =
        mPersistentSoftParticles.owningSoftBodyIndicesBuffer;
    view.soft.particles.phasesBuffer           = mPersistentSoftParticles.phasesBuffer;
    view.soft.particles.collisionLayersBuffer  = mPersistentSoftParticles.collisionLayersBuffer;
    view.soft.particles.collisionMasksBuffer   = mPersistentSoftParticles.collisionMasksBuffer;
    view.soft.particles.adjacencyOffsetsBuffer = mPersistentSoftParticles.adjacencyOffsetsBuffer;
    view.soft.particles.adjacencyCountsBuffer  = mPersistentSoftParticles.adjacencyCountsBuffer;
    view.soft.particles.adjacencyIndicesBuffer = mPersistentSoftParticles.adjacencyIndicesBuffer;
    view.soft.particles.count                  = mSoftParticleCount;
    view.soft.edgesBuffer                      = mPersistentSoftTopology.edgesBuffer;
    view.soft.tetsBuffer                       = mPersistentSoftTopology.tetsBuffer;
    view.soft.softBodyCount                    = mSoftBodyCount;
    view.soft.edgeCount                        = mSoftEdgeCount;
    view.soft.tetCount                         = mSoftTetCount;
    view.soft.rigidSurfaceParticles.localPositionsBuffer =
        mPersistentRigidSurfaceParticles.localPositionsBuffer;
    view.soft.rigidSurfaceParticles.worldPositionsBuffer =
        mTransientState.rigidSurfaceWorldPositionsBuffer;
    view.soft.rigidSurfaceParticles.owningRigidBodyIndicesBuffer =
        mPersistentRigidSurfaceParticles.owningRigidBodyIndicesBuffer;
    view.soft.rigidSurfaceParticles.owningColliderIndicesBuffer =
        mPersistentRigidSurfaceParticles.owningColliderIndicesBuffer;
    view.soft.rigidSurfaceParticles.sampleRadiiBuffer =
        mPersistentRigidSurfaceParticles.sampleRadiiBuffer;
    view.soft.rigidSurfaceParticles.environmentIndicesBuffer =
        mPersistentRigidSurfaceParticles.environmentIndicesBuffer;
    view.soft.rigidSurfaceParticles.collisionLayersBuffer =
        mPersistentRigidSurfaceParticles.collisionLayersBuffer;
    view.soft.rigidSurfaceParticles.collisionMasksBuffer =
        mPersistentRigidSurfaceParticles.collisionMasksBuffer;
    view.soft.rigidSurfaceParticles.count = mRigidSurfaceParticleCount;
    return view;
}

} // namespace cressim::neo::physics
