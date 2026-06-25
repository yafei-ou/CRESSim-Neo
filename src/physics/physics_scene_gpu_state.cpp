#include "physics/physics_scene_gpu_state.h"

#include "common/logger.h"
#include "gpu/gpu_buffer_utils.h"
#include "physics/rigid_body_common.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"

#include <algorithm>
#include <array>
#include <string>

namespace cressim::neo::physics
{

namespace
{

constexpr std::uint32_t kComputeThreadGroupSize  = 64u;
constexpr std::uint32_t kNarrowPhaseChunkSize    = 128u;
constexpr std::uint32_t kSoftBodyBoundsChunkSize = 64u;

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

bool ensureRawBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                     std::uint32_t elementStride, std::uint32_t elementCount,
                     Diligent::BIND_FLAGS bindFlags, Diligent::USAGE usage,
                     Diligent::CPU_ACCESS_FLAGS cpuAccess, Diligent::Uint64 immediateContextMask,
                     Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer)
{
    if (renderDevice == nullptr || elementStride == 0u)
    {
        return false;
    }

    const std::uint32_t requiredCapacity = std::max(elementCount, 1u);
    if (outBuffer != nullptr)
    {
        const Diligent::BufferDesc &desc    = outBuffer->GetDesc();
        const std::uint32_t currentCapacity = static_cast<std::uint32_t>(desc.Size / elementStride);
        if (desc.Mode == Diligent::BUFFER_MODE_RAW && currentCapacity >= requiredCapacity)
        {
            return true;
        }
    }

    Diligent::BufferDesc desc{};
    desc.Name                 = name;
    desc.Size                 = static_cast<Diligent::Uint64>(requiredCapacity) * elementStride;
    desc.BindFlags            = bindFlags;
    desc.Usage                = usage;
    desc.CPUAccessFlags       = cpuAccess;
    desc.ImmediateContextMask = immediateContextMask;
    desc.Mode                 = Diligent::BUFFER_MODE_RAW;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> buffer;
    renderDevice->CreateBuffer(desc, nullptr, &buffer);
    if (buffer == nullptr)
    {
        return false;
    }

    outBuffer = std::move(buffer);
    return true;
}

bool ensureAtomicFloatBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                             std::uint32_t elementCount, Diligent::BIND_FLAGS bindFlags,
                             Diligent::USAGE usage, Diligent::CPU_ACCESS_FLAGS cpuAccess,
                             Diligent::Uint64 immediateContextMask, bool useNativeFloatAtomics,
                             Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer)
{
    if (useNativeFloatAtomics)
    {
        return ensureStructuredBuffer(renderDevice, name, sizeof(Diligent::float4), elementCount,
                                      bindFlags, usage, cpuAccess, immediateContextMask, outBuffer);
    }

    return ensureRawBuffer(renderDevice, name, sizeof(Diligent::uint4), elementCount, bindFlags,
                           usage, cpuAccess, immediateContextMask, outBuffer);
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
bool readbackBufferRangeBlocking(Diligent::IDeviceContext *computeContext, Diligent::IBuffer *src,
                                 Diligent::IBuffer *dst, std::uint32_t begin, std::uint32_t count,
                                 T *outValues)
{
    if (computeContext == nullptr || src == nullptr || dst == nullptr || outValues == nullptr)
    {
        return false;
    }
    if (count == 0u)
    {
        return true;
    }

    const Diligent::Uint64 offset = static_cast<Diligent::Uint64>(begin) * sizeof(T);
    const Diligent::Uint64 bytes  = static_cast<Diligent::Uint64>(count) * sizeof(T);
    computeContext->CopyBuffer(src, offset, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               dst, 0u, bytes, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->Flush();
    computeContext->WaitForIdle();

    void *mapped = nullptr;
    computeContext->MapBuffer(dst, Diligent::MAP_READ, Diligent::MAP_FLAG_DO_NOT_WAIT, mapped);
    if (mapped == nullptr)
    {
        return false;
    }

    const T *typed = static_cast<const T *>(mapped);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        outValues[i] = typed[i];
    }
    computeContext->UnmapBuffer(dst, Diligent::MAP_READ);
    return true;
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

template <typename RefT>
void buildConstraintAdjacencyRanges(std::uint32_t particleCount,
                                    const std::vector<std::vector<RefT>> &refs,
                                    std::vector<GpuSoftConstraintRange> &ranges)
{
    ranges.resize(particleCount);
    std::uint32_t start = 0u;
    for (std::uint32_t particleIndex = 0u; particleIndex < particleCount; ++particleIndex)
    {
        GpuSoftConstraintRange &range = ranges[particleIndex];
        range.start                   = start;
        range.count                   = static_cast<std::uint32_t>(refs[particleIndex].size());
        start += range.count;
    }
}

} // namespace

bool PhysicsSceneGpuState::ensureCapacity(
    Diligent::IRenderDevice *renderDevice, std::uint32_t bodyCount, std::uint32_t colliderCount,
    std::uint32_t particleCount, std::uint32_t fluidCount,
    std::uint32_t particleContactMaterialCount, std::uint32_t fluidMaterialCount,
    std::uint32_t softEdgeCount, std::uint32_t softBendCount, std::uint32_t softTetCount,
    std::uint32_t strandSegmentCount, std::uint32_t strandJointCount,
    std::uint32_t strandDistanceCount, std::uint32_t ballJointCount,
    std::uint32_t sphericalJointCount, std::uint32_t hingeJointCount,
    std::uint32_t sliderJointCount, std::uint32_t rigidParticleAttachmentCount,
    std::uint32_t strandRigidAttachmentCount, std::uint32_t rigidDistanceConstraintCount,
    std::uint32_t softRenderVertexCount, std::uint32_t softRenderTriangleIndexCount,
    std::uint32_t softRenderTriangleCount, std::uint32_t softBodyRangeCount,
    std::uint32_t softBodyBoundsChunkCount, std::uint32_t suturingPairCount,
    std::uint32_t suturingPathHeaderCount, std::uint32_t suturingPathNodeCount,
    std::uint32_t routedCableCount, std::uint32_t routedCableRoutePointCount,
    std::uint32_t routedCableDebugSegmentCount, std::uint32_t curveRenderCount,
    std::uint32_t curveRenderParticleIndexCount, std::uint32_t curveRenderVertexCount,
    Diligent::Uint64 sharedContextMask, const std::uint32_t *sharedQueueFamilyIndices,
    std::uint32_t sharedQueueFamilyIndexCount, bool useNativeFloatAtomics)
{
    const auto rigidPositionsBefore    = mPersistentRigidBodies.positionsBuffer.RawPtr();
    const auto rigidOrientationsBefore = mPersistentRigidBodies.orientationsBuffer.RawPtr();
    const auto rigidBodyTypesBefore    = mPersistentRigidBodies.bodyTypesBuffer.RawPtr();
    const auto rigidProxyMaterialsBefore =
        mPersistentRigidBodies.proxyParticleContactMaterialsBuffer.RawPtr();
    const auto colliderOwnersBefore     = mPersistentColliders.ownerRigidBodyIndicesBuffer.RawPtr();
    const auto colliderBroadPhaseBefore = mPersistentColliders.broadPhaseDataBuffer.RawPtr();
    const auto colliderGeometryBefore   = mPersistentColliders.geometryDataBuffer.RawPtr();
    const auto bodyColliderRangesBefore =
        mPersistentBodyColliderMapping.colliderRangesBuffer.RawPtr();
    const auto bodyColliderIndicesBefore =
        mPersistentBodyColliderMapping.colliderIndicesBuffer.RawPtr();
    const auto jointSuppressionOffsetsBefore =
        mPersistentJointCollisionSuppression.neighborOffsetsBuffer.RawPtr();
    const auto jointSuppressionNeighborsBefore =
        mPersistentJointCollisionSuppression.neighborsBuffer.RawPtr();
    const auto predictedPositionsBefore =
        mTransientState.predictedRigidBodies.positionsBuffer.RawPtr();
    const auto predictedOrientationsBefore =
        mTransientState.predictedRigidBodies.orientationsBuffer.RawPtr();
    const auto predictedLinearBefore =
        mTransientState.predictedRigidBodies.linearVelocitiesBuffer.RawPtr();
    const auto predictedAngularBefore =
        mTransientState.predictedRigidBodies.angularVelocitiesBuffer.RawPtr();
    const auto bodyAabbsBefore             = mTransientState.bodyAabbsBuffer.RawPtr();
    const auto bodyMetaBefore              = mTransientState.bodyMetaBuffer.RawPtr();
    const auto activeFlagsBefore           = mTransientState.activeBodyFlagsBuffer.RawPtr();
    const auto activeOffsetsBefore         = mTransientState.activeBodyOffsetsBuffer.RawPtr();
    const auto staticFlagsBefore           = mTransientState.staticBodyFlagsBuffer.RawPtr();
    const auto staticOffsetsBefore         = mTransientState.staticBodyOffsetsBuffer.RawPtr();
    const auto rigidContactsBefore         = mTransientState.rigidContactsBuffer.RawPtr();
    const auto proxyRigidContactMetaBefore = mTransientState.proxyRigidContactMetaBuffer.RawPtr();
    const auto rigidAggregateMapBefore = mTransientState.rigidBodyPairAggregateMapBuffer.RawPtr();
    const auto rigidAggregateActiveCountBefore =
        mTransientState.rigidBodyPairAggregateActiveCountBuffer.RawPtr();
    const auto rigidAggregateHeadersBefore =
        mTransientState.rigidBodyPairAggregateHeadersBuffer.RawPtr();
    const auto rigidAggregateSlotsBefore =
        mTransientState.rigidBodyPairAggregateSlotsBuffer.RawPtr();
    const auto sphericalJointTranslationLambdasBefore =
        mTransientState.sphericalJointTranslationLambdasBuffer.RawPtr();
    const auto sphericalJointRotationLambdasBefore =
        mTransientState.sphericalJointRotationLambdasBuffer.RawPtr();
    const auto translationCorrBefore    = mTransientState.translationCorrectionsBuffer.RawPtr();
    const auto rotationCorrBefore       = mTransientState.rotationCorrectionsBuffer.RawPtr();
    const auto linearVelCorrBefore      = mTransientState.linearVelocityCorrectionsBuffer.RawPtr();
    const auto angularVelCorrBefore     = mTransientState.angularVelocityCorrectionsBuffer.RawPtr();
    const auto particlePositionsBefore  = mPersistentParticles.positionsInvMassBuffer.RawPtr();
    const auto particlePreviousBefore   = mPersistentParticles.previousPositionsBuffer.RawPtr();
    const auto particleVelocitiesBefore = mPersistentParticles.velocitiesBuffer.RawPtr();
    const auto ballJointsBefore         = mPersistentJoints.ballJointsBuffer.RawPtr();
    const auto sphericalJointsBefore    = mPersistentJoints.sphericalJointsBuffer.RawPtr();
    const auto hingeJointsBefore        = mPersistentJoints.hingeJointsBuffer.RawPtr();
    const auto sliderJointsBefore       = mPersistentJoints.sliderJointsBuffer.RawPtr();
    const auto hingePassiveIndicesBefore =
        mPersistentJoints.hingePassiveJointIndicesBuffer.RawPtr();
    const auto hingePositionDriveIndicesBefore =
        mPersistentJoints.hingePositionDriveJointIndicesBuffer.RawPtr();
    const auto hingeVelocityDriveIndicesBefore =
        mPersistentJoints.hingeVelocityDriveJointIndicesBuffer.RawPtr();
    const auto sliderPassiveIndicesBefore =
        mPersistentJoints.sliderPassiveJointIndicesBuffer.RawPtr();
    const auto sliderPositionDriveIndicesBefore =
        mPersistentJoints.sliderPositionDriveJointIndicesBuffer.RawPtr();
    const auto sliderVelocityDriveIndicesBefore =
        mPersistentJoints.sliderVelocityDriveJointIndicesBuffer.RawPtr();
    const auto rigidParticleAttachmentsBefore =
        mPersistentRoutedCables.rigidParticleAttachmentsBuffer.RawPtr();
    const auto strandRigidAttachmentsBefore =
        mPersistentRoutedCables.strandRigidAttachmentsBuffer.RawPtr();
    const auto rigidDistanceConstraintsBefore =
        mPersistentRoutedCables.rigidDistanceConstraintsBuffer.RawPtr();
    const auto routedCableDescriptorsBefore = mPersistentRoutedCables.descriptorsBuffer.RawPtr();
    const auto routedCableRoutePointsBefore = mPersistentRoutedCables.routePointsBuffer.RawPtr();
    const auto routedCableDebugSegmentsBefore =
        mPersistentRoutedCables.debugSegmentsBuffer.RawPtr();
    const auto softEdgesBefore      = mPersistentSoftTopology.edgesBuffer.RawPtr();
    const auto softBendsBefore      = mPersistentSoftTopology.bendsBuffer.RawPtr();
    const auto softTetsBefore       = mPersistentSoftTopology.tetsBuffer.RawPtr();
    const auto strandSegmentsBefore = mPersistentSoftTopology.strandSegmentsBuffer.RawPtr();
    const auto strandJointsBefore   = mPersistentSoftTopology.strandJointsBuffer.RawPtr();
    const auto strandDistanceBefore =
        mPersistentSoftTopology.strandDistanceConstraintsBuffer.RawPtr();
    const auto strandStatesBefore = mPersistentSoftTopology.strandSegmentStatesBuffer.RawPtr();
    const auto segmentAttachmentRangesBefore =
        mPersistentSoftTopology.segmentStrandRigidAttachmentRangesBuffer.RawPtr();
    const auto segmentIncidentAttachmentsBefore =
        mPersistentSoftTopology.segmentIncidentStrandRigidAttachmentsBuffer.RawPtr();
    const auto softRenderNormalsBefore =
        mPersistentSoftTopology.softBodyRenderNormalsBuffer.RawPtr();
    const auto softWorldAabbsBefore     = mPersistentSoftTopology.softBodyWorldAabbsBuffer.RawPtr();
    const auto suturingPairsBefore      = mPersistentSuturing.pairsBuffer.RawPtr();
    const auto suturingInsertionsBefore = mPersistentSuturing.insertionStatesBuffer.RawPtr();
    const auto suturingPathHeadersBefore = mPersistentSuturing.pathHeadersBuffer.RawPtr();
    const auto suturingPathNodesBefore   = mPersistentSuturing.pathNodesBuffer.RawPtr();
    const auto curveDescriptorsBefore    = mPersistentCurveRender.descriptorsBuffer.RawPtr();
    const auto curvePositionsBefore      = mPersistentCurveRender.positionsBuffer.RawPtr();
    const auto curveNormalsBefore        = mPersistentCurveRender.normalsBuffer.RawPtr();
    const auto curveWorldAabbsBefore     = mPersistentCurveRender.worldAabbsBuffer.RawPtr();

    const bool hasAllBuffers =
        mPersistentRigidBodies.positionsBuffer != nullptr &&
        mPersistentRigidBodies.orientationsBuffer != nullptr &&
        mPersistentRigidBodies.scalesBuffer != nullptr &&
        mPersistentRigidBodies.linearVelocitiesBuffer != nullptr &&
        mPersistentRigidBodies.angularVelocitiesBuffer != nullptr &&
        mPersistentRigidBodies.inverseInertiaLocalBuffer != nullptr &&
        mPersistentRigidBodies.bodyTypesBuffer != nullptr &&
        mPersistentRigidBodies.proxyParticleContactMaterialsBuffer != nullptr &&
        mPersistentRigidBodies.kinematicTargetPositionsBuffer != nullptr &&
        mPersistentRigidBodies.kinematicTargetOrientationsBuffer != nullptr &&
        mPersistentRigidBodies.kinematicTargetFlagsBuffer != nullptr &&
        mPersistentRoutedCables.rigidParticleAttachmentsBuffer != nullptr &&
        mPersistentRoutedCables.strandRigidAttachmentsBuffer != nullptr &&
        mPersistentRoutedCables.rigidDistanceConstraintsBuffer != nullptr &&
        mPersistentRoutedCables.descriptorsBuffer != nullptr &&
        mPersistentRoutedCables.routePointsBuffer != nullptr &&
        mPersistentRoutedCables.debugSegmentsBuffer != nullptr &&
        mPersistentColliders.ownerRigidBodyIndicesBuffer != nullptr &&
        mPersistentColliders.broadPhaseDataBuffer != nullptr &&
        mPersistentColliders.geometryDataBuffer != nullptr &&
        mPersistentColliders.contactDataBuffer != nullptr &&
        mPersistentColliders.shapeTypesBuffer != nullptr &&
        mPersistentColliders.enabledFlagsBuffer != nullptr &&
        mPersistentColliders.materialBuffer != nullptr &&
        mPersistentBodyColliderMapping.colliderOffsetsBuffer != nullptr &&
        mPersistentBodyColliderMapping.colliderCountsBuffer != nullptr &&
        mPersistentBodyColliderMapping.colliderRangesBuffer != nullptr &&
        mPersistentBodyColliderMapping.colliderIndicesBuffer != nullptr &&
        mPersistentJointCollisionSuppression.neighborOffsetsBuffer != nullptr &&
        mPersistentJointCollisionSuppression.neighborsBuffer != nullptr &&
        mPersistentJoints.ballJointsBuffer != nullptr &&
        mPersistentJoints.sphericalJointsBuffer != nullptr &&
        mPersistentJoints.hingeJointsBuffer != nullptr &&
        mPersistentJoints.sliderJointsBuffer != nullptr &&
        mPersistentJoints.hingePassiveJointIndicesBuffer != nullptr &&
        mPersistentJoints.hingePositionDriveJointIndicesBuffer != nullptr &&
        mPersistentJoints.hingeVelocityDriveJointIndicesBuffer != nullptr &&
        mPersistentJoints.sliderPassiveJointIndicesBuffer != nullptr &&
        mPersistentJoints.sliderPositionDriveJointIndicesBuffer != nullptr &&
        mPersistentJoints.sliderVelocityDriveJointIndicesBuffer != nullptr &&
        mPersistentParticles.positionsInvMassBuffer != nullptr &&
        mPersistentParticles.previousPositionsBuffer != nullptr &&
        mPersistentParticles.velocitiesBuffer != nullptr &&
        mPersistentParticles.radiiBuffer != nullptr &&
        mPersistentParticles.environmentIndicesBuffer != nullptr &&
        mPersistentParticles.particleKindsBuffer != nullptr &&
        mPersistentParticles.ownerTypesBuffer != nullptr &&
        mPersistentParticles.ownerIndicesBuffer != nullptr &&
        mPersistentParticles.strandIdsBuffer != nullptr &&
        mPersistentParticles.strandRolesBuffer != nullptr &&
        mPersistentParticles.suturingNeighborLinksBuffer != nullptr &&
        mPersistentParticles.owningSoftBodyIndicesBuffer != nullptr &&
        mPersistentParticles.particleMaterialIndicesBuffer != nullptr &&
        mPersistentParticles.fluidMaterialIndicesBuffer != nullptr &&
        mPersistentParticles.fluidVisualsBuffer != nullptr &&
        mPersistentParticles.particleContactMaterialsBuffer != nullptr &&
        mPersistentParticles.fluidMaterialsBuffer != nullptr &&
        mPersistentParticles.phasesBuffer != nullptr &&
        mPersistentParticles.collisionLayersBuffer != nullptr &&
        mPersistentParticles.collisionMasksBuffer != nullptr &&
        mPersistentParticles.adjacencyOffsetsBuffer != nullptr &&
        mPersistentParticles.adjacencyCountsBuffer != nullptr &&
        mPersistentParticles.adjacencyIndicesBuffer != nullptr &&
        mPersistentParticles.broadPhaseMetadataBuffer != nullptr &&
        mPersistentParticles.rigidProxyLocalPositionsBuffer != nullptr &&
        mPersistentSoftTopology.edgesBuffer != nullptr &&
        mPersistentSoftTopology.bendsBuffer != nullptr &&
        mPersistentSoftTopology.tetsBuffer != nullptr &&
        mPersistentSoftTopology.strandSegmentsBuffer != nullptr &&
        mPersistentSoftTopology.strandJointsBuffer != nullptr &&
        mPersistentSoftTopology.strandDistanceConstraintsBuffer != nullptr &&
        mPersistentSoftTopology.strandSegmentStatesBuffer != nullptr &&
        mPersistentSoftTopology.particleEdgeRangesBuffer != nullptr &&
        mPersistentSoftTopology.particleIncidentEdgesBuffer != nullptr &&
        mPersistentSoftTopology.particleBendRangesBuffer != nullptr &&
        mPersistentSoftTopology.particleIncidentBendsBuffer != nullptr &&
        mPersistentSoftTopology.particleTetRangesBuffer != nullptr &&
        mPersistentSoftTopology.particleIncidentTetsBuffer != nullptr &&
        mPersistentSoftTopology.particleStrandSegmentRangesBuffer != nullptr &&
        mPersistentSoftTopology.particleIncidentStrandSegmentsBuffer != nullptr &&
        mPersistentSoftTopology.particleStrandJointRangesBuffer != nullptr &&
        mPersistentSoftTopology.particleIncidentStrandJointsBuffer != nullptr &&
        mPersistentSoftTopology.segmentStrandJointRangesBuffer != nullptr &&
        mPersistentSoftTopology.segmentIncidentStrandJointsBuffer != nullptr &&
        mPersistentSoftTopology.segmentStrandRigidAttachmentRangesBuffer != nullptr &&
        mPersistentSoftTopology.segmentIncidentStrandRigidAttachmentsBuffer != nullptr &&
        mPersistentSoftTopology.renderVertexTriangleRangesBuffer != nullptr &&
        mPersistentSoftTopology.renderVertexTriangleIndicesBuffer != nullptr &&
        mPersistentSoftTopology.renderTriangleParticleIndicesBuffer != nullptr &&
        mPersistentSoftTopology.renderTriangleNormalsBuffer != nullptr &&
        mPersistentSoftTopology.softBodyParticleRangesBuffer != nullptr &&
        mPersistentSoftTopology.softBodyChunkRangesBuffer != nullptr &&
        mPersistentSoftTopology.softBodyBoundsChunksBuffer != nullptr &&
        mPersistentSoftTopology.softBodyFallbackNormalsBuffer != nullptr &&
        mPersistentSoftTopology.softBodyRenderNormalsBuffer != nullptr &&
        mPersistentSoftTopology.softBodyWorldAabbsBuffer != nullptr &&
        mPersistentSuturing.pairsBuffer != nullptr &&
        mPersistentSuturing.particleRefsBuffer != nullptr &&
        mPersistentSuturing.insertionStatesBuffer != nullptr &&
        mPersistentSuturing.pathHeadersBuffer != nullptr &&
        mPersistentSuturing.pathNodesBuffer != nullptr &&
        mPersistentCurveRender.descriptorsBuffer != nullptr &&
        mPersistentCurveRender.particleIndicesBuffer != nullptr &&
        mPersistentCurveRender.positionsBuffer != nullptr &&
        mPersistentCurveRender.normalsBuffer != nullptr &&
        mPersistentCurveRender.worldAabbsBuffer != nullptr &&
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
        mTransientState.physicsIndirectArgsBuffer != nullptr &&
        mTransientState.softSoftCandidatePairsBuffer != nullptr &&
        mTransientState.suturingCandidateCountsBuffer != nullptr &&
        mTransientState.suturingCandidateParticlesBuffer != nullptr &&
        mTransientState.fluidNeighborPairsBuffer != nullptr &&
        mTransientState.fluidBoundaryCandidateCountsBuffer != nullptr &&
        mTransientState.fluidBoundaryCandidateOffsetsBuffer != nullptr &&
        mTransientState.fluidBoundaryCandidateRangesBuffer != nullptr &&
        mTransientState.fluidBoundaryCandidatePairsBuffer != nullptr &&
        mTransientState.softRigidCandidatePairsBuffer != nullptr &&
        mTransientState.softRigidContactsBuffer != nullptr &&
        mTransientState.softContactsBuffer != nullptr &&
        mTransientState.activeSoftRigidContactsBuffer != nullptr &&
        mTransientState.activeSoftContactsBuffer != nullptr &&
        mTransientState.softPositionCorrectionsBuffer != nullptr &&
        mTransientState.softVelocityCorrectionsBuffer != nullptr &&
        mTransientState.fluidDeltaPositionsBuffer != nullptr &&
        mTransientState.fluidIterationDeltaBuffer != nullptr &&
        mTransientState.fluidSurfaceNormalConstraintsBuffer != nullptr &&
        mTransientState.fluidAnisotropy1Buffer != nullptr &&
        mTransientState.fluidAnisotropy2Buffer != nullptr &&
        mTransientState.fluidAnisotropy3Buffer != nullptr &&
        mTransientState.fluidVorticitiesBuffer != nullptr &&
        mTransientState.softEdgeLambdasBuffer != nullptr &&
        mTransientState.strandSegmentLambdasBuffer != nullptr &&
        mTransientState.strandDistanceLambdasBuffer != nullptr &&
        mTransientState.routedCableLambdasBuffer != nullptr &&
        mTransientState.softBendLambdasBuffer != nullptr &&
        mTransientState.strandJointLambdasBuffer != nullptr &&
        mTransientState.softTetLambdasBuffer != nullptr &&
        mTransientState.softEdgeCorrectionsBuffer != nullptr &&
        mTransientState.strandSegmentCorrectionsBuffer != nullptr &&
        mTransientState.softBendCorrectionsBuffer != nullptr &&
        mTransientState.strandJointCorrectionsBuffer != nullptr &&
        mTransientState.strandDistanceCorrectionsBuffer != nullptr &&
        mTransientState.softTetCorrectionsBuffer != nullptr &&
        mTransientState.softBodyChunkAabbsBuffer != nullptr &&
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
        mTransientState.proxyRigidContactMetaBuffer != nullptr &&
        mTransientState.rigidBodyPairAggregateMapBuffer != nullptr &&
        mTransientState.rigidBodyPairAggregateActiveCountBuffer != nullptr &&
        mTransientState.rigidBodyPairAggregateHeadersBuffer != nullptr &&
        mTransientState.rigidBodyPairAggregateSlotsBuffer != nullptr &&
        mTransientState.sphericalJointTranslationLambdasBuffer != nullptr &&
        mTransientState.sphericalJointRotationLambdasBuffer != nullptr &&
        mTransientState.hingeJointLambdas0123Buffer != nullptr &&
        mTransientState.hingeJointLambdas45Buffer != nullptr &&
        mTransientState.sliderJointLambdas0123Buffer != nullptr &&
        mTransientState.sliderJointLambdas45Buffer != nullptr &&
        mTransientState.rigidParticleAttachmentLambdasBuffer != nullptr &&
        mTransientState.strandRigidAttachmentLambdasBuffer != nullptr &&
        mTransientState.rigidDistanceConstraintLambdasBuffer != nullptr &&
        mTransientState.translationCorrectionsBuffer != nullptr &&
        mTransientState.rotationCorrectionsBuffer != nullptr &&
        mTransientState.linearVelocityCorrectionsBuffer != nullptr &&
        mTransientState.angularVelocityCorrectionsBuffer != nullptr &&
        mTransientState.strandRigidAttachmentCorrectionsBuffer != nullptr &&
        mReadbackRigidBodies.positionsBuffer != nullptr &&
        mReadbackRigidBodies.orientationsBuffer != nullptr &&
        mReadbackRigidBodies.linearVelocitiesBuffer != nullptr &&
        mReadbackRigidBodies.angularVelocitiesBuffer != nullptr &&
        mReadbackRigidBodies.broadPhaseMetaBuffer != nullptr &&
        mReadbackParticles.positionsBuffer != nullptr &&
        mReadbackParticles.previousPositionsBuffer != nullptr &&
        mReadbackParticles.velocitiesBuffer != nullptr &&
        mReadbackParticles.neighborMetaBuffer != nullptr;
    const std::uint32_t newRigidBodyCapacity    = std::max<std::uint32_t>(bodyCount, 64u);
    const std::uint32_t newColliderCapacity     = std::max<std::uint32_t>(colliderCount, 64u);
    const std::uint32_t newSoftParticleCapacity = std::max<std::uint32_t>(particleCount, 64u);
    const std::uint32_t newFluidVisualCapacity  = std::max<std::uint32_t>(fluidCount, 1u);
    const std::uint32_t newParticleContactMaterialCapacity =
        std::max<std::uint32_t>(particleContactMaterialCount, 1u);
    const std::uint32_t newFluidMaterialCapacity = std::max<std::uint32_t>(fluidMaterialCount, 1u);
    const std::uint32_t newSoftEdgeCapacity      = std::max<std::uint32_t>(softEdgeCount, 64u);
    const std::uint32_t newSoftBendCapacity      = std::max<std::uint32_t>(softBendCount, 64u);
    const std::uint32_t newSoftTetCapacity       = std::max<std::uint32_t>(softTetCount, 64u);
    const std::uint32_t newStrandSegmentCapacity = std::max<std::uint32_t>(strandSegmentCount, 64u);
    const std::uint32_t newStrandJointCapacity   = std::max<std::uint32_t>(strandJointCount, 64u);
    const std::uint32_t newStrandDistanceCapacity =
        std::max<std::uint32_t>(strandDistanceCount, 64u);
    const std::uint32_t newParticleBroadPhaseEntryCapacity =
        std::max<std::uint32_t>(particleCount, 64u);
    const std::uint32_t newParticleCellRangeCapacity =
        nextPowerOfTwo(std::max<std::uint32_t>(newParticleBroadPhaseEntryCapacity * 2u, 64u));
    const std::uint32_t newSoftCandidatePairCapacity =
        std::max<std::uint32_t>(particleCount * 8u, 64u);
    const std::uint32_t newFluidBoundaryCandidatePairCapacity =
        std::max<std::uint32_t>(particleCount * 32u, 64u);
    // Fluid neighbors are stored in fixed per-particle slots so the solver can
    // index them directly without a scan/offset pass. This is still a heuristic
    // and may truncate in very dense scenes until we add explicit overflow
    // reporting for the fluid path.
    const std::uint32_t newMaxFluidNeighborhood = kDefaultFluidMaxNeighborhood;
    const std::uint32_t newFluidNeighborPairCapacity =
        std::max<std::uint32_t>(particleCount * newMaxFluidNeighborhood, newMaxFluidNeighborhood);
    const std::uint32_t newSoftScanCapacity =
        std::max(newParticleBroadPhaseEntryCapacity, newSoftCandidatePairCapacity);
    const std::uint32_t newSoftParticleAdjacencyCapacity = std::max<std::uint32_t>(
        std::max(std::max(softEdgeCount * 2u, softBendCount * 3u), softTetCount * 12u), 64u);
    const std::uint32_t newSoftIncidentEdgeCapacity =
        std::max<std::uint32_t>(softEdgeCount * 2u, 64u);
    const std::uint32_t newSoftIncidentBendCapacity =
        std::max<std::uint32_t>(softBendCount * 3u, 64u);
    const std::uint32_t newSoftIncidentTetCapacity =
        std::max<std::uint32_t>(softTetCount * 4u, 64u);
    const std::uint32_t newStrandIncidentSegmentCapacity =
        std::max<std::uint32_t>(strandSegmentCount * 2u, 64u);
    const std::uint32_t newStrandIncidentJointCapacity =
        std::max<std::uint32_t>(strandJointCount * 3u, 64u);
    const std::uint32_t newStrandSegmentIncidentJointCapacity =
        std::max<std::uint32_t>(strandJointCount * 2u, 64u);
    const std::uint32_t newStrandSegmentIncidentAttachmentCapacity =
        std::max<std::uint32_t>(strandRigidAttachmentCount, 64u);
    const std::uint32_t newSoftRenderVertexCapacity =
        std::max<std::uint32_t>(softRenderVertexCount, 64u);
    const std::uint32_t newSoftRenderTriangleIndexCapacity =
        std::max<std::uint32_t>(softRenderTriangleIndexCount, 64u);
    const std::uint32_t newSoftRenderTriangleCapacity =
        std::max<std::uint32_t>(softRenderTriangleCount, 64u);
    const std::uint32_t newSoftBodyRangeCapacity = std::max<std::uint32_t>(softBodyRangeCount, 64u);
    const std::uint32_t newSoftBodyBoundsChunkCapacity =
        std::max<std::uint32_t>(softBodyBoundsChunkCount, 64u);
    const std::uint32_t newSuturingPairCapacity = std::max<std::uint32_t>(suturingPairCount, 1u);
    const std::uint32_t newSuturingParticleCapacity = std::max<std::uint32_t>(particleCount, 1u);
    const std::uint32_t newSuturingPathHeaderCapacity =
        std::max<std::uint32_t>(suturingPathHeaderCount, 1u);
    const std::uint32_t newSuturingPathNodeCapacity =
        std::max<std::uint32_t>(suturingPathNodeCount, 1u);
    const std::uint32_t newRoutedCableCapacity = std::max<std::uint32_t>(routedCableCount, 1u);
    const std::uint32_t newRoutedCableRoutePointCapacity =
        std::max<std::uint32_t>(routedCableRoutePointCount, 1u);
    const std::uint32_t newRoutedCableDebugSegmentCapacity =
        std::max<std::uint32_t>(routedCableDebugSegmentCount, 1u);
    const std::uint32_t newRigidParticleAttachmentCapacity =
        std::max<std::uint32_t>(rigidParticleAttachmentCount, 1u);
    const std::uint32_t newStrandRigidAttachmentCapacity =
        std::max<std::uint32_t>(strandRigidAttachmentCount, 1u);
    const std::uint32_t newRigidDistanceConstraintCapacity =
        std::max<std::uint32_t>(rigidDistanceConstraintCount, 1u);
    const std::uint32_t newCurveRenderCapacity = std::max<std::uint32_t>(curveRenderCount, 1u);
    const std::uint32_t newCurveRenderParticleIndexCapacity =
        std::max<std::uint32_t>(curveRenderParticleIndexCount, 1u);
    const std::uint32_t newCurveRenderVertexCapacity =
        std::max<std::uint32_t>(curveRenderVertexCount, 1u);
    const std::uint32_t newSuturingCandidateSlotCapacity = std::max<std::uint32_t>(
        newSuturingParticleCapacity * kMaxSuturingCandidatesPerParticle, 1u);
    const std::uint32_t newJointCollisionSuppressionOffsetCapacity =
        std::max<std::uint32_t>(bodyCount + 1u, 1u);
    const std::uint32_t newJointCollisionSuppressionNeighborCapacity = std::max<std::uint32_t>(
        (ballJointCount + sphericalJointCount + hingeJointCount + sliderJointCount) * 2u, 1u);
    const std::uint32_t newBallJointCapacity = std::max<std::uint32_t>(ballJointCount, 1u);
    const std::uint32_t newSphericalJointCapacity =
        std::max<std::uint32_t>(sphericalJointCount, 1u);
    const std::uint32_t newHingeJointCapacity  = std::max<std::uint32_t>(hingeJointCount, 1u);
    const std::uint32_t newSliderJointCapacity = std::max<std::uint32_t>(sliderJointCount, 1u);
    const std::uint32_t newHingePassiveJointIndexCapacity   = newHingeJointCapacity;
    const std::uint32_t newHingePositionDriveIndexCapacity  = newHingeJointCapacity;
    const std::uint32_t newHingeVelocityDriveIndexCapacity  = newHingeJointCapacity;
    const std::uint32_t newSliderPassiveJointIndexCapacity  = newSliderJointCapacity;
    const std::uint32_t newSliderPositionDriveIndexCapacity = newSliderJointCapacity;
    const std::uint32_t newSliderVelocityDriveIndexCapacity = newSliderJointCapacity;
    const std::uint32_t newNodeCapacity                     = std::max<std::uint32_t>(
        newColliderCapacity > 0u ? (newColliderCapacity * 2u - 1u) : 1u, 1u);
    const std::uint32_t newCandidatePairCapacity =
        estimateRigidCandidatePairCapacityFromColliderCount(newColliderCapacity);
    const std::uint32_t newChunkCapacity = std::max<std::uint32_t>(
        (newCandidatePairCapacity + kNarrowPhaseChunkSize - 1u) / kNarrowPhaseChunkSize, 1u);
    const std::uint32_t newRigidContactCapacity = std::max<std::uint32_t>(
        newCandidatePairCapacity * kRigidContactsPerPair + newSoftCandidatePairCapacity,
        kRigidContactsPerPair);
    const Diligent::Uint64 contextMask = sharedContextMask;
    const bool needsSharedSoftPositionsRecreate =
        mSharedSoftPositionsInvMass.buffer() != nullptr &&
        (mSharedSoftPositionsInvMass.capacity() < newSoftParticleCapacity ||
         mSharedSoftPositionsInvMass.elementStride() != sizeof(Diligent::float4));
    const std::vector<std::uint32_t> reductionLevelCounts =
        buildReductionLevelCounts(std::max(newColliderCapacity, newSoftScanCapacity));

    if (hasAllBuffers && mRigidBodyCapacity >= bodyCount && mColliderCapacity >= colliderCount &&
        mSoftParticleCapacity >= particleCount && mSoftEdgeCapacity >= softEdgeCount &&
        mSoftBendCapacity >= softBendCount && mSoftTetCapacity >= softTetCount &&
        mStrandSegmentCapacity >= strandSegmentCount && mStrandJointCapacity >= strandJointCount &&
        mStrandDistanceCapacity >= strandDistanceCount &&
        mFluidVisualCapacity >= newFluidVisualCapacity &&
        mParticleContactMaterialCapacity >= newParticleContactMaterialCapacity &&
        mFluidMaterialCapacity >= newFluidMaterialCapacity &&
        mParticleBroadPhaseEntryCapacity >= newParticleBroadPhaseEntryCapacity &&
        mSoftCandidatePairCapacity >= newSoftCandidatePairCapacity &&
        mFluidBoundaryCandidatePairCapacity >= newFluidBoundaryCandidatePairCapacity &&
        mFluidNeighborPairCapacity >= newFluidNeighborPairCapacity &&
        mSoftScanScratchCapacity >= newSoftScanCapacity &&
        mSoftIncidentEdgeCapacity >= newSoftIncidentEdgeCapacity &&
        mSoftIncidentBendCapacity >= newSoftIncidentBendCapacity &&
        mSoftIncidentTetCapacity >= newSoftIncidentTetCapacity &&
        mStrandIncidentSegmentCapacity >= newStrandIncidentSegmentCapacity &&
        mStrandIncidentJointCapacity >= newStrandIncidentJointCapacity &&
        mStrandSegmentIncidentJointCapacity >= newStrandSegmentIncidentJointCapacity &&
        mStrandSegmentIncidentAttachmentCapacity >= newStrandSegmentIncidentAttachmentCapacity &&
        mSoftRenderVertexCapacity >= newSoftRenderVertexCapacity &&
        mSoftRenderTriangleIndexCapacity >= newSoftRenderTriangleIndexCapacity &&
        mSoftRenderTriangleCapacity >= newSoftRenderTriangleCapacity &&
        mSoftBodyRangeCapacity >= newSoftBodyRangeCapacity &&
        mJointCollisionSuppressionOffsetCapacity >= newJointCollisionSuppressionOffsetCapacity &&
        mJointCollisionSuppressionNeighborCapacity >=
            newJointCollisionSuppressionNeighborCapacity &&
        mBallJointCapacity >= newBallJointCapacity &&
        mSphericalJointCapacity >= newSphericalJointCapacity &&
        mHingeJointCapacity >= newHingeJointCapacity &&
        mSliderJointCapacity >= newSliderJointCapacity &&
        mHingePassiveJointIndexCapacity >= newHingePassiveJointIndexCapacity &&
        mHingePositionDriveIndexCapacity >= newHingePositionDriveIndexCapacity &&
        mHingeVelocityDriveIndexCapacity >= newHingeVelocityDriveIndexCapacity &&
        mSliderPassiveJointIndexCapacity >= newSliderPassiveJointIndexCapacity &&
        mSliderPositionDriveIndexCapacity >= newSliderPositionDriveIndexCapacity &&
        mSliderVelocityDriveIndexCapacity >= newSliderVelocityDriveIndexCapacity &&
        mSuturingPairCapacity >= newSuturingPairCapacity &&
        mSuturingParticleCapacity >= newSuturingParticleCapacity &&
        mSuturingPathHeaderCapacity >= newSuturingPathHeaderCapacity &&
        mSuturingPathNodeCapacity >= newSuturingPathNodeCapacity &&
        mRoutedCableCapacity >= newRoutedCableCapacity &&
        mRoutedCableRoutePointCapacity >= newRoutedCableRoutePointCapacity &&
        mRoutedCableDebugSegmentCapacity >= newRoutedCableDebugSegmentCapacity &&
        mRigidParticleAttachmentCapacity >= newRigidParticleAttachmentCapacity &&
        mStrandRigidAttachmentCapacity >= newStrandRigidAttachmentCapacity &&
        mRigidDistanceConstraintCapacity >= newRigidDistanceConstraintCapacity &&
        mCurveRenderCapacity >= newCurveRenderCapacity &&
        mCurveRenderParticleIndexCapacity >= newCurveRenderParticleIndexCapacity &&
        mCurveRenderVertexCapacity >= newCurveRenderVertexCapacity &&
        mSoftBodyBoundsChunkCapacity >= newSoftBodyBoundsChunkCapacity)
    {
        return true;
    }

    if (needsSharedSoftPositionsRecreate)
    {
        mPersistentParticles.positionsInvMassBuffer = nullptr;
    }

    if (!ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PositionsInvMass",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Orientations",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Scales", sizeof(Diligent::float4),
                                newRigidBodyCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.scalesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.LinearVelocities",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.AngularVelocities",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
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
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidBodyProxyParticleMaterials",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.proxyParticleContactMaterialsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.KinematicTargetPositions",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.kinematicTargetPositionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.KinematicTargetOrientations",
                                sizeof(Diligent::float4), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.kinematicTargetOrientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.KinematicTargetFlags",
                                sizeof(std::uint32_t), newRigidBodyCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRigidBodies.kinematicTargetFlagsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderOwnerBodyIndices",
                                sizeof(std::uint32_t), newColliderCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentColliders.ownerRigidBodyIndicesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ColliderContactData", sizeof(GpuColliderContactData),
            newColliderCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentColliders.contactDataBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderBroadPhaseData",
                                sizeof(GpuColliderBroadPhaseData), newColliderCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentColliders.broadPhaseDataBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderGeometryData",
                                sizeof(GpuColliderGeometryData), newColliderCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentColliders.geometryDataBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ColliderShapeTypes", sizeof(std::uint32_t),
            newColliderCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentColliders.shapeTypesBuffer) ||
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
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.JointCollisionSuppressionOffsets",
                                sizeof(std::uint32_t), newJointCollisionSuppressionOffsetCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJointCollisionSuppression.neighborOffsetsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.JointCollisionSuppressionNeighbors",
            sizeof(std::uint32_t), newJointCollisionSuppressionNeighborCapacity,
            Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
            contextMask, mPersistentJointCollisionSuppression.neighborsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BallJoints", sizeof(GpuBallJoint),
                                newBallJointCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJoints.ballJointsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SphericalJoints", sizeof(GpuSphericalJoint),
            newSphericalJointCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentJoints.sphericalJointsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.HingeJoints",
                                sizeof(GpuHingeJoint), newHingeJointCapacity,
                                Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJoints.hingeJointsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SliderJoints",
                                sizeof(GpuSliderJoint), newSliderJointCapacity,
                                Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJoints.sliderJointsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.HingePassiveJointIndices",
                                sizeof(std::uint32_t), newHingePassiveJointIndexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJoints.hingePassiveJointIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.HingePositionDriveJointIndices",
                                sizeof(std::uint32_t), newHingePositionDriveIndexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJoints.hingePositionDriveJointIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.HingeVelocityDriveJointIndices",
                                sizeof(std::uint32_t), newHingeVelocityDriveIndexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJoints.hingeVelocityDriveJointIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SliderPassiveJointIndices",
                                sizeof(std::uint32_t), newSliderPassiveJointIndexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJoints.sliderPassiveJointIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SliderPositionDriveJointIndices",
                                sizeof(std::uint32_t), newSliderPositionDriveIndexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJoints.sliderPositionDriveJointIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SliderVelocityDriveJointIndices",
                                sizeof(std::uint32_t), newSliderVelocityDriveIndexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentJoints.sliderVelocityDriveJointIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidParticleAttachments",
                                sizeof(GpuRigidParticleAttachmentConstraint),
                                newRigidParticleAttachmentCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRoutedCables.rigidParticleAttachmentsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandRigidAttachments",
                                sizeof(GpuStrandRigidAttachmentConstraint),
                                newStrandRigidAttachmentCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRoutedCables.strandRigidAttachmentsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidDistanceConstraints",
                                sizeof(GpuRigidDistanceConstraint),
                                newRigidDistanceConstraintCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRoutedCables.rigidDistanceConstraintsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RoutedCableDescriptors",
                                sizeof(GpuRoutedCableConstraint), newRoutedCableCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRoutedCables.descriptorsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RoutedCableRoutePoints",
                                sizeof(GpuRoutedCableRoutePoint), newRoutedCableRoutePointCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRoutedCables.routePointsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RoutedCableDebugSegments",
                                sizeof(GpuRoutedCableDebugSegment),
                                newRoutedCableDebugSegmentCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentRoutedCables.debugSegmentsBuffer) ||
        !mSharedSoftPositionsInvMass.ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftPositionsInvMass", sizeof(Diligent::float4),
            newSoftParticleCapacity, newSoftParticleCapacity,
            Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
            sharedQueueFamilyIndices, sharedQueueFamilyIndexCount) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftPreviousPositions",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.previousPositionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftVelocities",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.velocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRadii", sizeof(float),
                                newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.radiiBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftEnvironmentIndices",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.environmentIndicesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ParticleKinds", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.particleKindsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ParticleOwnerTypes", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.ownerTypesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ParticleOwnerIndices", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.ownerIndicesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ParticleStrandIds", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.strandIdsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.ParticleStrandRoles", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.strandRolesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SuturingNeighborLinks",
                                sizeof(Diligent::uint4), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.suturingNeighborLinksBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBodyIndices",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.owningSoftBodyIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleMaterialIndices",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.particleMaterialIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidMaterialIndices",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.fluidMaterialIndicesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.FluidVisuals", sizeof(Diligent::float4),
            newFluidVisualCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.fluidVisualsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleContactMaterials",
                                sizeof(Diligent::float4), newParticleContactMaterialCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.particleContactMaterialsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.FluidMaterials", sizeof(FluidMaterialGpu),
            newFluidMaterialCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.fluidMaterialsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftPhases", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.phasesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftCollisionLayers", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.collisionLayersBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftCollisionMasks", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.collisionMasksBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftAdjacencyOffsets", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.adjacencyOffsetsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftAdjacencyCounts", sizeof(std::uint32_t),
            newSoftParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentParticles.adjacencyCountsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftAdjacencyIndices",
                                sizeof(std::uint32_t), newSoftParticleAdjacencyCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.adjacencyIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidProxyLocalPositions",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.rigidProxyLocalPositionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBroadPhaseMetadata",
                                sizeof(Diligent::uint4), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentParticles.broadPhaseMetadataBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftEdges", sizeof(SoftEdge),
                                newSoftEdgeCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.edgesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBends", sizeof(SoftBend),
                                newSoftBendCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.bendsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftTets", sizeof(SoftTet),
                                newSoftTetCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.tetsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.StrandSegments", sizeof(StrandSegmentConstraint),
            newStrandSegmentCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentSoftTopology.strandSegmentsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.StrandJoints", sizeof(StrandJointConstraint),
            newStrandJointCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentSoftTopology.strandJointsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandDistanceConstraints",
                                sizeof(StrandDistanceConstraint), newStrandDistanceCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.strandDistanceConstraintsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandSegmentStates",
                                sizeof(StrandSegmentState), newStrandSegmentCapacity,
                                Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.strandSegmentStatesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftParticleEdgeRanges",
                                sizeof(GpuSoftConstraintRange), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleEdgeRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftIncidentEdges",
                                sizeof(GpuSoftIncidentEdge), newSoftIncidentEdgeCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleIncidentEdgesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftParticleBendRanges",
                                sizeof(GpuSoftConstraintRange), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleBendRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftIncidentBends",
                                sizeof(GpuSoftIncidentBend), newSoftIncidentBendCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleIncidentBendsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftParticleTetRanges",
                                sizeof(GpuSoftConstraintRange), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleTetRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftIncidentTets",
                                sizeof(GpuSoftIncidentTet), newSoftIncidentTetCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleIncidentTetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleStrandSegmentRanges",
                                sizeof(GpuSoftConstraintRange), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleStrandSegmentRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleIncidentStrandSegments",
                                sizeof(GpuStrandIncidentSegment), newStrandIncidentSegmentCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleIncidentStrandSegmentsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleStrandJointRanges",
                                sizeof(GpuSoftConstraintRange), newSoftParticleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleStrandJointRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ParticleIncidentStrandJoints",
                                sizeof(GpuStrandIncidentJoint), newStrandIncidentJointCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.particleIncidentStrandJointsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SegmentStrandJointRanges",
                                sizeof(GpuSoftConstraintRange), newStrandSegmentCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.segmentStrandJointRangesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SegmentIncidentStrandJoints",
            sizeof(GpuStrandIncidentJoint), newStrandSegmentIncidentJointCapacity,
            Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
            contextMask, mPersistentSoftTopology.segmentIncidentStrandJointsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SegmentStrandRigidAttachmentRanges",
            sizeof(GpuSoftConstraintRange), newStrandSegmentCapacity,
            Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
            contextMask, mPersistentSoftTopology.segmentStrandRigidAttachmentRangesBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SegmentIncidentStrandRigidAttachments",
            sizeof(GpuStrandIncidentAttachment), newStrandSegmentIncidentAttachmentCapacity,
            Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
            contextMask, mPersistentSoftTopology.segmentIncidentStrandRigidAttachmentsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRenderVertexTriangleRanges",
                                sizeof(GpuSoftRenderVertexTriangleRange),
                                newSoftRenderVertexCapacity, Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.renderVertexTriangleRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRenderVertexTriangleIndices",
                                sizeof(std::uint32_t), newSoftRenderTriangleIndexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.renderVertexTriangleIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRenderVertexBindings",
                                sizeof(SoftRenderVertexBinding), newSoftRenderVertexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.renderVertexBindingsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRenderTriangleParticles",
                                sizeof(Diligent::uint4), newSoftRenderTriangleCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.renderTriangleParticleIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRenderTriangleNormals",
                                sizeof(Diligent::float4), newSoftRenderTriangleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.renderTriangleNormalsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBodyParticleRanges",
                                sizeof(GpuSoftBodyParticleRange), newSoftBodyRangeCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.softBodyParticleRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBodyChunkRanges",
                                sizeof(GpuSoftBodyChunkRange), newSoftBodyRangeCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.softBodyChunkRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBodyBoundsChunks",
                                sizeof(GpuSoftBodyBoundsChunk), newSoftBodyBoundsChunkCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.softBodyBoundsChunksBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftFallbackNormals",
                                sizeof(Diligent::float4), newSoftRenderVertexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.softBodyFallbackNormalsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRenderPositions",
                                sizeof(Diligent::float4), newSoftRenderVertexCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.softBodyRenderPositionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRenderNormals",
                                sizeof(Diligent::float4), newSoftRenderVertexCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.softBodyRenderNormalsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBodyWorldAabbs",
                                sizeof(GpuBodyAabb), newSoftBodyRangeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSoftTopology.softBodyWorldAabbsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SuturingPairs",
                                sizeof(GpuSuturingPair), newSuturingPairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSuturing.pairsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SuturingParticleRefs", sizeof(Diligent::uint4),
            newSuturingParticleCapacity, Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
            Diligent::CPU_ACCESS_NONE, contextMask, mPersistentSuturing.particleRefsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SuturingInsertionStates",
                                sizeof(GpuSuturingInsertionStateStorage), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSuturing.insertionStatesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SuturingPathHeaders",
                                sizeof(GpuSuturingPathHeader), newSuturingPathHeaderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSuturing.pathHeadersBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SuturingPathNodes",
                                sizeof(GpuSuturingPathNode), newSuturingPathNodeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentSuturing.pathNodesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.CurveRenderDescriptors",
                                sizeof(GpuCurveRenderDescriptor), newCurveRenderCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentCurveRender.descriptorsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.CurveRenderParticleIndices",
                                sizeof(std::uint32_t), newCurveRenderParticleIndexCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentCurveRender.particleIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.CurveRenderPositions",
                                sizeof(Diligent::float4), newCurveRenderVertexCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentCurveRender.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.CurveRenderNormals",
                                sizeof(Diligent::float4), newCurveRenderVertexCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentCurveRender.normalsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.CurveWorldAabbs",
                                sizeof(GpuBodyAabb), newCurveRenderCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mPersistentCurveRender.worldAabbsBuffer) ||
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
                                sizeof(GpuParticleNeighborMeta), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softNeighborMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.IndirectDispatchArgs",
                                sizeof(GpuDispatchIndirectArgs),
                                static_cast<std::uint32_t>(GpuPhysicsIndirectDispatchSlot::Count),
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE |
                                    Diligent::BIND_INDIRECT_DRAW_ARGS,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.physicsIndirectArgsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftSoftCandidatePairs",
                                sizeof(GpuParticleCandidatePair), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softSoftCandidatePairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SuturingCandidateCounts",
                                sizeof(std::uint32_t), newSuturingParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.suturingCandidateCountsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SuturingCandidateParticles",
                                sizeof(Diligent::uint4), newSuturingCandidateSlotCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.suturingCandidateParticlesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidNeighborPairs",
                                sizeof(GpuParticleCandidatePair), newFluidNeighborPairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidNeighborPairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidBoundaryCandidateCounts",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidBoundaryCandidateCountsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidBoundaryCandidateOffsets",
                                sizeof(std::uint32_t), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidBoundaryCandidateOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidBoundaryCandidateRanges",
                                sizeof(Diligent::uint2), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidBoundaryCandidateRangesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidBoundaryCandidatePairs",
                                sizeof(GpuParticleCandidatePair),
                                newFluidBoundaryCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidBoundaryCandidatePairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRigidCandidatePairs",
                                sizeof(GpuParticleCandidatePair), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softRigidCandidatePairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftRigidContacts",
                                sizeof(GpuParticleRigidContact), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softRigidContactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftContacts",
                                sizeof(GpuParticleContact), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softContactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveSoftRigidContacts",
                                sizeof(GpuParticleRigidContact), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.activeSoftRigidContactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveSoftContacts",
                                sizeof(GpuParticleContact), newSoftCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.activeSoftContactsBuffer) ||
        !ensureAtomicFloatBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftPositionCorrections", newSoftParticleCapacity,
            Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask, useNativeFloatAtomics,
            mTransientState.softPositionCorrectionsBuffer) ||
        !ensureAtomicFloatBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftVelocityCorrections", newSoftParticleCapacity,
            Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask, useNativeFloatAtomics,
            mTransientState.softVelocityCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidDeltaPositions",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidDeltaPositionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidIterationDelta",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidIterationDeltaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidSurfaceNormalConstraints",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidSurfaceNormalConstraintsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidAnisotropy1",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidAnisotropy1Buffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidAnisotropy2",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidAnisotropy2Buffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidAnisotropy3",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidAnisotropy3Buffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.FluidVorticities",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.fluidVorticitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftEdgeLambdas", sizeof(float),
                                newSoftEdgeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softEdgeLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandSegmentLambdas",
                                sizeof(Diligent::float4), newStrandSegmentCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.strandSegmentLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandDistanceLambdas",
                                sizeof(float), newStrandDistanceCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.strandDistanceLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidParticleAttachmentLambdas",
                                sizeof(Diligent::float4), newRigidParticleAttachmentCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rigidParticleAttachmentLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandRigidAttachmentLambdas",
                                sizeof(GpuStrandRigidAttachmentLambda),
                                newStrandRigidAttachmentCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.strandRigidAttachmentLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidDistanceConstraintLambdas",
                                sizeof(float), newRigidDistanceConstraintCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rigidDistanceConstraintLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RoutedCableLambdas",
                                sizeof(float), newRoutedCableCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.routedCableLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBendLambdas", sizeof(float),
                                newSoftBendCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softBendLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandJointLambdas",
                                sizeof(Diligent::float4), newStrandJointCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.strandJointLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftTetLambdas", sizeof(float),
                                newSoftTetCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softTetLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftEdgeCorrections",
                                sizeof(GpuSoftEdgeCorrection), newSoftEdgeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softEdgeCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandSegmentCorrections",
                                sizeof(GpuStrandSegmentCorrection), newStrandSegmentCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.strandSegmentCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBendCorrections",
                                sizeof(GpuSoftBendCorrection), newSoftBendCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softBendCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandJointCorrections",
                                sizeof(GpuStrandJointCorrection), newStrandJointCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.strandJointCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandRigidAttachmentCorrections",
                                sizeof(GpuStrandRigidAttachmentCorrection),
                                newStrandRigidAttachmentCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.strandRigidAttachmentCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.StrandDistanceCorrections",
                                sizeof(GpuSoftEdgeCorrection), newStrandDistanceCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.strandDistanceCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftTetCorrections",
                                sizeof(GpuSoftTetCorrection), newSoftTetCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softTetCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftBodyChunkAabbs",
                                sizeof(GpuBodyAabb), newSoftBodyBoundsChunkCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.softBodyChunkAabbsBuffer) ||
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
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ProxyRigidContactMeta",
                                sizeof(GpuProxyRigidContactMeta), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.proxyRigidContactMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ProxyRigidContactMeta.Readback",
                                sizeof(GpuProxyRigidContactMeta), 1u, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackRigidBodies.proxyRigidContactMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidBodyPairAggregateMap",
                                sizeof(GpuRigidBodyPairContactAggregateMapEntry),
                                newCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rigidBodyPairAggregateMapBuffer) ||
        !ensureStructuredBuffer(renderDevice,
                                "CRESSimNeo.Physics.RigidBodyPairAggregateActiveCount",
                                sizeof(std::uint32_t), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rigidBodyPairAggregateActiveCountBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidBodyPairAggregateHeaders",
                                sizeof(GpuRigidBodyPairContactAggregateHeader),
                                newCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rigidBodyPairAggregateHeadersBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidBodyPairAggregateSlots",
                                sizeof(GpuRigidBodyPairContactAggregateSlot),
                                newCandidatePairCapacity * kRigidBodyPairAggregateContacts,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.rigidBodyPairAggregateSlotsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SphericalJointTranslationLambdas",
                                sizeof(Diligent::float4), newSphericalJointCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.sphericalJointTranslationLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SphericalJointRotationLambdas",
                                sizeof(Diligent::float4), newSphericalJointCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.sphericalJointRotationLambdasBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.HingeJointLambdas0123",
                                sizeof(Diligent::float4), newHingeJointCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.hingeJointLambdas0123Buffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.HingeJointLambdas45",
                                sizeof(Diligent::float4), newHingeJointCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.hingeJointLambdas45Buffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SliderJointLambdas0123",
                                sizeof(Diligent::float4), newSliderJointCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.sliderJointLambdas0123Buffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SliderJointLambdas45",
                                sizeof(Diligent::float4), newSliderJointCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                mTransientState.sliderJointLambdas45Buffer) ||
        !ensureAtomicFloatBuffer(
            renderDevice, "CRESSimNeo.Physics.TranslationCorrections", newRigidBodyCapacity,
            Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask, useNativeFloatAtomics,
            mTransientState.translationCorrectionsBuffer) ||
        !ensureAtomicFloatBuffer(
            renderDevice, "CRESSimNeo.Physics.RotationCorrections", newRigidBodyCapacity,
            Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask, useNativeFloatAtomics,
            mTransientState.rotationCorrectionsBuffer) ||
        !ensureAtomicFloatBuffer(
            renderDevice, "CRESSimNeo.Physics.LinearVelocityCorrections", newRigidBodyCapacity,
            Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask, useNativeFloatAtomics,
            mTransientState.linearVelocityCorrectionsBuffer) ||
        !ensureAtomicFloatBuffer(
            renderDevice, "CRESSimNeo.Physics.AngularVelocityCorrections", newRigidBodyCapacity,
            Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask, useNativeFloatAtomics,
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
            Diligent::CPU_ACCESS_READ, contextMask, mReadbackParticles.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftPreviousPositions.Readback",
                                sizeof(Diligent::float4), newSoftParticleCapacity,
                                Diligent::BIND_NONE, Diligent::USAGE_STAGING,
                                Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackParticles.previousPositionsBuffer) ||
        !ensureStructuredBuffer(
            renderDevice, "CRESSimNeo.Physics.SoftVelocities.Readback", sizeof(Diligent::float4),
            newSoftParticleCapacity, Diligent::BIND_NONE, Diligent::USAGE_STAGING,
            Diligent::CPU_ACCESS_READ, contextMask, mReadbackParticles.velocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SoftNeighborMeta.Readback",
                                sizeof(GpuParticleNeighborMeta), 1u, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackParticles.neighborMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ProxyRigidContactMeta.Readback",
                                sizeof(GpuProxyRigidContactMeta), 1u, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                mReadbackRigidBodies.proxyRigidContactMetaBuffer))
    {
        return false;
    }

    mPersistentParticles.positionsInvMassBuffer = mSharedSoftPositionsInvMass.bufferRef();

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

    const bool softCapacityChanged =
        mSoftParticleCapacity != newSoftParticleCapacity ||
        mFluidVisualCapacity != newFluidVisualCapacity ||
        mParticleContactMaterialCapacity != newParticleContactMaterialCapacity ||
        mFluidMaterialCapacity != newFluidMaterialCapacity ||
        mSoftEdgeCapacity != newSoftEdgeCapacity || mSoftBendCapacity != newSoftBendCapacity ||
        mSoftTetCapacity != newSoftTetCapacity ||
        mStrandSegmentCapacity != newStrandSegmentCapacity ||
        mStrandJointCapacity != newStrandJointCapacity ||
        mStrandDistanceCapacity != newStrandDistanceCapacity ||
        mParticleBroadPhaseEntryCapacity != newParticleBroadPhaseEntryCapacity ||
        mSoftCandidatePairCapacity != newSoftCandidatePairCapacity ||
        mFluidBoundaryCandidatePairCapacity != newFluidBoundaryCandidatePairCapacity ||
        mFluidNeighborPairCapacity != newFluidNeighborPairCapacity ||
        mMaxFluidNeighborhood != newMaxFluidNeighborhood ||
        mSoftScanScratchCapacity != newSoftScanCapacity ||
        mSoftParticleAdjacencyCapacity != newSoftParticleAdjacencyCapacity ||
        mSoftIncidentEdgeCapacity != newSoftIncidentEdgeCapacity ||
        mSoftIncidentBendCapacity != newSoftIncidentBendCapacity ||
        mSoftIncidentTetCapacity != newSoftIncidentTetCapacity ||
        mStrandIncidentSegmentCapacity != newStrandIncidentSegmentCapacity ||
        mStrandIncidentJointCapacity != newStrandIncidentJointCapacity ||
        mStrandSegmentIncidentJointCapacity != newStrandSegmentIncidentJointCapacity ||
        mStrandSegmentIncidentAttachmentCapacity != newStrandSegmentIncidentAttachmentCapacity ||
        mSoftRenderVertexCapacity != newSoftRenderVertexCapacity ||
        mSoftRenderTriangleIndexCapacity != newSoftRenderTriangleIndexCapacity ||
        mSoftRenderTriangleCapacity != newSoftRenderTriangleCapacity ||
        mSoftBodyRangeCapacity != newSoftBodyRangeCapacity ||
        mSuturingPairCapacity != newSuturingPairCapacity ||
        mSuturingParticleCapacity != newSuturingParticleCapacity ||
        mSuturingPathHeaderCapacity != newSuturingPathHeaderCapacity ||
        mSuturingPathNodeCapacity != newSuturingPathNodeCapacity ||
        mRoutedCableCapacity != newRoutedCableCapacity ||
        mRoutedCableRoutePointCapacity != newRoutedCableRoutePointCapacity ||
        mRoutedCableDebugSegmentCapacity != newRoutedCableDebugSegmentCapacity ||
        mRigidParticleAttachmentCapacity != newRigidParticleAttachmentCapacity ||
        mStrandRigidAttachmentCapacity != newStrandRigidAttachmentCapacity ||
        mRigidDistanceConstraintCapacity != newRigidDistanceConstraintCapacity ||
        mCurveRenderCapacity != newCurveRenderCapacity ||
        mCurveRenderParticleIndexCapacity != newCurveRenderParticleIndexCapacity ||
        mCurveRenderVertexCapacity != newCurveRenderVertexCapacity ||
        mSoftBodyBoundsChunkCapacity != newSoftBodyBoundsChunkCapacity;

    mRigidBodyCapacity                         = newRigidBodyCapacity;
    mColliderCapacity                          = newColliderCapacity;
    mSoftParticleCapacity                      = newSoftParticleCapacity;
    mSuturingPairCapacity                      = newSuturingPairCapacity;
    mSuturingParticleCapacity                  = newSuturingParticleCapacity;
    mSuturingPathHeaderCapacity                = newSuturingPathHeaderCapacity;
    mSuturingPathNodeCapacity                  = newSuturingPathNodeCapacity;
    mRoutedCableCapacity                       = newRoutedCableCapacity;
    mRoutedCableRoutePointCapacity             = newRoutedCableRoutePointCapacity;
    mRoutedCableDebugSegmentCapacity           = newRoutedCableDebugSegmentCapacity;
    mRigidParticleAttachmentCapacity           = newRigidParticleAttachmentCapacity;
    mStrandRigidAttachmentCapacity             = newStrandRigidAttachmentCapacity;
    mRigidDistanceConstraintCapacity           = newRigidDistanceConstraintCapacity;
    mCurveRenderCapacity                       = newCurveRenderCapacity;
    mCurveRenderParticleIndexCapacity          = newCurveRenderParticleIndexCapacity;
    mCurveRenderVertexCapacity                 = newCurveRenderVertexCapacity;
    mFluidVisualCapacity                       = newFluidVisualCapacity;
    mParticleContactMaterialCapacity           = newParticleContactMaterialCapacity;
    mFluidMaterialCapacity                     = newFluidMaterialCapacity;
    mSoftEdgeCapacity                          = newSoftEdgeCapacity;
    mSoftBendCapacity                          = newSoftBendCapacity;
    mSoftTetCapacity                           = newSoftTetCapacity;
    mStrandSegmentCapacity                     = newStrandSegmentCapacity;
    mStrandJointCapacity                       = newStrandJointCapacity;
    mStrandDistanceCapacity                    = newStrandDistanceCapacity;
    mParticleBroadPhaseEntryCapacity           = newParticleBroadPhaseEntryCapacity;
    mSoftCandidatePairCapacity                 = newSoftCandidatePairCapacity;
    mFluidBoundaryCandidatePairCapacity        = newFluidBoundaryCandidatePairCapacity;
    mFluidNeighborPairCapacity                 = newFluidNeighborPairCapacity;
    mMaxFluidNeighborhood                      = newMaxFluidNeighborhood;
    mSoftScanScratchCapacity                   = newSoftScanCapacity;
    mSoftParticleAdjacencyCapacity             = newSoftParticleAdjacencyCapacity;
    mSoftIncidentEdgeCapacity                  = newSoftIncidentEdgeCapacity;
    mSoftIncidentBendCapacity                  = newSoftIncidentBendCapacity;
    mSoftIncidentTetCapacity                   = newSoftIncidentTetCapacity;
    mStrandIncidentSegmentCapacity             = newStrandIncidentSegmentCapacity;
    mStrandIncidentJointCapacity               = newStrandIncidentJointCapacity;
    mStrandSegmentIncidentJointCapacity        = newStrandSegmentIncidentJointCapacity;
    mStrandSegmentIncidentAttachmentCapacity   = newStrandSegmentIncidentAttachmentCapacity;
    mSoftRenderVertexCapacity                  = newSoftRenderVertexCapacity;
    mSoftRenderTriangleIndexCapacity           = newSoftRenderTriangleIndexCapacity;
    mSoftRenderTriangleCapacity                = newSoftRenderTriangleCapacity;
    mSoftBodyRangeCapacity                     = newSoftBodyRangeCapacity;
    mSoftBodyBoundsChunkCapacity               = newSoftBodyBoundsChunkCapacity;
    mJointCollisionSuppressionOffsetCapacity   = newJointCollisionSuppressionOffsetCapacity;
    mJointCollisionSuppressionNeighborCapacity = newJointCollisionSuppressionNeighborCapacity;
    mBallJointCapacity                         = newBallJointCapacity;
    mSphericalJointCapacity                    = newSphericalJointCapacity;
    mHingeJointCapacity                        = newHingeJointCapacity;
    mSliderJointCapacity                       = newSliderJointCapacity;
    mHingePassiveJointIndexCapacity            = newHingePassiveJointIndexCapacity;
    mHingePositionDriveIndexCapacity           = newHingePositionDriveIndexCapacity;
    mHingeVelocityDriveIndexCapacity           = newHingeVelocityDriveIndexCapacity;
    mSliderPassiveJointIndexCapacity           = newSliderPassiveJointIndexCapacity;
    mSliderPositionDriveIndexCapacity          = newSliderPositionDriveIndexCapacity;
    mSliderVelocityDriveIndexCapacity          = newSliderVelocityDriveIndexCapacity;
    mBroadPhaseNodeCapacity                    = newNodeCapacity;
    mCandidatePairCapacity                     = newCandidatePairCapacity;
    mContactCapacity                           = newRigidContactCapacity;
    const bool rigidBindingsChanged =
        rigidPositionsBefore != mPersistentRigidBodies.positionsBuffer.RawPtr() ||
        rigidOrientationsBefore != mPersistentRigidBodies.orientationsBuffer.RawPtr() ||
        rigidBodyTypesBefore != mPersistentRigidBodies.bodyTypesBuffer.RawPtr() ||
        rigidProxyMaterialsBefore !=
            mPersistentRigidBodies.proxyParticleContactMaterialsBuffer.RawPtr() ||
        colliderOwnersBefore != mPersistentColliders.ownerRigidBodyIndicesBuffer.RawPtr() ||
        colliderBroadPhaseBefore != mPersistentColliders.broadPhaseDataBuffer.RawPtr() ||
        colliderGeometryBefore != mPersistentColliders.geometryDataBuffer.RawPtr() ||
        bodyColliderRangesBefore != mPersistentBodyColliderMapping.colliderRangesBuffer.RawPtr() ||
        bodyColliderIndicesBefore !=
            mPersistentBodyColliderMapping.colliderIndicesBuffer.RawPtr() ||
        jointSuppressionOffsetsBefore !=
            mPersistentJointCollisionSuppression.neighborOffsetsBuffer.RawPtr() ||
        jointSuppressionNeighborsBefore !=
            mPersistentJointCollisionSuppression.neighborsBuffer.RawPtr() ||
        ballJointsBefore != mPersistentJoints.ballJointsBuffer.RawPtr() ||
        sphericalJointsBefore != mPersistentJoints.sphericalJointsBuffer.RawPtr() ||
        hingeJointsBefore != mPersistentJoints.hingeJointsBuffer.RawPtr() ||
        sliderJointsBefore != mPersistentJoints.sliderJointsBuffer.RawPtr() ||
        hingePassiveIndicesBefore != mPersistentJoints.hingePassiveJointIndicesBuffer.RawPtr() ||
        hingePositionDriveIndicesBefore !=
            mPersistentJoints.hingePositionDriveJointIndicesBuffer.RawPtr() ||
        hingeVelocityDriveIndicesBefore !=
            mPersistentJoints.hingeVelocityDriveJointIndicesBuffer.RawPtr() ||
        sliderPassiveIndicesBefore != mPersistentJoints.sliderPassiveJointIndicesBuffer.RawPtr() ||
        sliderPositionDriveIndicesBefore !=
            mPersistentJoints.sliderPositionDriveJointIndicesBuffer.RawPtr() ||
        sliderVelocityDriveIndicesBefore !=
            mPersistentJoints.sliderVelocityDriveJointIndicesBuffer.RawPtr() ||
        rigidParticleAttachmentsBefore !=
            mPersistentRoutedCables.rigidParticleAttachmentsBuffer.RawPtr() ||
        strandRigidAttachmentsBefore !=
            mPersistentRoutedCables.strandRigidAttachmentsBuffer.RawPtr() ||
        rigidDistanceConstraintsBefore !=
            mPersistentRoutedCables.rigidDistanceConstraintsBuffer.RawPtr() ||
        routedCableDescriptorsBefore != mPersistentRoutedCables.descriptorsBuffer.RawPtr() ||
        routedCableRoutePointsBefore != mPersistentRoutedCables.routePointsBuffer.RawPtr() ||
        routedCableDebugSegmentsBefore != mPersistentRoutedCables.debugSegmentsBuffer.RawPtr() ||
        predictedPositionsBefore != mTransientState.predictedRigidBodies.positionsBuffer.RawPtr() ||
        predictedOrientationsBefore !=
            mTransientState.predictedRigidBodies.orientationsBuffer.RawPtr() ||
        predictedLinearBefore !=
            mTransientState.predictedRigidBodies.linearVelocitiesBuffer.RawPtr() ||
        predictedAngularBefore !=
            mTransientState.predictedRigidBodies.angularVelocitiesBuffer.RawPtr() ||
        bodyAabbsBefore != mTransientState.bodyAabbsBuffer.RawPtr() ||
        bodyMetaBefore != mTransientState.bodyMetaBuffer.RawPtr() ||
        activeFlagsBefore != mTransientState.activeBodyFlagsBuffer.RawPtr() ||
        activeOffsetsBefore != mTransientState.activeBodyOffsetsBuffer.RawPtr() ||
        staticFlagsBefore != mTransientState.staticBodyFlagsBuffer.RawPtr() ||
        staticOffsetsBefore != mTransientState.staticBodyOffsetsBuffer.RawPtr() ||
        rigidContactsBefore != mTransientState.rigidContactsBuffer.RawPtr() ||
        proxyRigidContactMetaBefore != mTransientState.proxyRigidContactMetaBuffer.RawPtr() ||
        rigidAggregateMapBefore != mTransientState.rigidBodyPairAggregateMapBuffer.RawPtr() ||
        rigidAggregateActiveCountBefore !=
            mTransientState.rigidBodyPairAggregateActiveCountBuffer.RawPtr() ||
        rigidAggregateHeadersBefore !=
            mTransientState.rigidBodyPairAggregateHeadersBuffer.RawPtr() ||
        rigidAggregateSlotsBefore != mTransientState.rigidBodyPairAggregateSlotsBuffer.RawPtr() ||
        sphericalJointTranslationLambdasBefore !=
            mTransientState.sphericalJointTranslationLambdasBuffer.RawPtr() ||
        sphericalJointRotationLambdasBefore !=
            mTransientState.sphericalJointRotationLambdasBuffer.RawPtr() ||
        translationCorrBefore != mTransientState.translationCorrectionsBuffer.RawPtr() ||
        rotationCorrBefore != mTransientState.rotationCorrectionsBuffer.RawPtr() ||
        linearVelCorrBefore != mTransientState.linearVelocityCorrectionsBuffer.RawPtr() ||
        angularVelCorrBefore != mTransientState.angularVelocityCorrectionsBuffer.RawPtr();
    const bool softBindingsChanged =
        softCapacityChanged ||
        particlePositionsBefore != mPersistentParticles.positionsInvMassBuffer.RawPtr() ||
        particlePreviousBefore != mPersistentParticles.previousPositionsBuffer.RawPtr() ||
        particleVelocitiesBefore != mPersistentParticles.velocitiesBuffer.RawPtr() ||
        softEdgesBefore != mPersistentSoftTopology.edgesBuffer.RawPtr() ||
        softBendsBefore != mPersistentSoftTopology.bendsBuffer.RawPtr() ||
        softTetsBefore != mPersistentSoftTopology.tetsBuffer.RawPtr() ||
        strandSegmentsBefore != mPersistentSoftTopology.strandSegmentsBuffer.RawPtr() ||
        strandJointsBefore != mPersistentSoftTopology.strandJointsBuffer.RawPtr() ||
        strandDistanceBefore != mPersistentSoftTopology.strandDistanceConstraintsBuffer.RawPtr() ||
        strandStatesBefore != mPersistentSoftTopology.strandSegmentStatesBuffer.RawPtr() ||
        segmentAttachmentRangesBefore !=
            mPersistentSoftTopology.segmentStrandRigidAttachmentRangesBuffer.RawPtr() ||
        segmentIncidentAttachmentsBefore !=
            mPersistentSoftTopology.segmentIncidentStrandRigidAttachmentsBuffer.RawPtr() ||
        suturingPairsBefore != mPersistentSuturing.pairsBuffer.RawPtr() ||
        suturingInsertionsBefore != mPersistentSuturing.insertionStatesBuffer.RawPtr() ||
        suturingPathHeadersBefore != mPersistentSuturing.pathHeadersBuffer.RawPtr() ||
        suturingPathNodesBefore != mPersistentSuturing.pathNodesBuffer.RawPtr() ||
        curveDescriptorsBefore != mPersistentCurveRender.descriptorsBuffer.RawPtr() ||
        curvePositionsBefore != mPersistentCurveRender.positionsBuffer.RawPtr() ||
        curveNormalsBefore != mPersistentCurveRender.normalsBuffer.RawPtr() ||
        curveWorldAabbsBefore != mPersistentCurveRender.worldAabbsBuffer.RawPtr() ||
        softRenderNormalsBefore != mPersistentSoftTopology.softBodyRenderNormalsBuffer.RawPtr() ||
        softWorldAabbsBefore != mPersistentSoftTopology.softBodyWorldAabbsBuffer.RawPtr();
    if (rigidBindingsChanged)
    {
        ++mRigidBindingGeneration;
    }
    if (softBindingsChanged)
    {
        ++mSoftBindingGeneration;
    }
    mCorrectionBuffersNeedClear      = true;
    mStaticBroadPhaseDirty           = true;
    mRigidBodyUploadResetRequired    = true;
    mColliderUploadResetRequired     = true;
    mRigidJointUploadResetRequired   = true;
    mSoftParticleUploadResetRequired = true;
    mSoftTopologyUploadResetRequired = true;
    mRigidParticleAttachmentGpuDirty = true;
    mStrandRigidAttachmentGpuDirty   = true;
    mRigidDistanceConstraintGpuDirty = true;
    mRoutedCableGpuDirty             = true;
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

    const RigidBodySoAHost &rigidBodies                = world.rigidBodySoA();
    const ColliderSoAHost &colliders                   = world.colliderSoA();
    const BodyColliderMappingHost &bodyColliderMapping = world.bodyColliderMapping();
    const JointCollisionSuppressionHost &jointCollisionSuppression =
        world.jointCollisionSuppression();
    const ParticleSoAHost &particles      = world.particles();
    const std::vector<FluidState> &fluids = world.fluidSnapshot();
    const std::vector<Diligent::float4> &particleContactMaterials =
        world.particleContactMaterials();
    const std::vector<FluidMaterialGpu> &fluidMaterials = world.fluidMaterials();
    const SoftRenderDataHost &softRenderData            = world.softRenderData();
    const CurveRenderDataHost &curveRenderData          = world.curveRenderData();
    const std::vector<DeformableDistanceConstraint> &distanceConstraints =
        world.distanceConstraints();
    const std::vector<DeformableBendConstraint> &bendConstraints     = world.bendConstraints();
    const std::vector<DeformableVolumeConstraint> &volumeConstraints = world.volumeConstraints();
    const std::vector<StrandSegmentConstraint> &strandSegments       = world.strandSegments();
    const std::vector<StrandJointConstraint> &strandJoints           = world.strandJoints();
    const std::vector<StrandSegmentState> &strandSegmentStates       = world.strandSegmentStates();
    const std::vector<RigidParticleAttachmentConstraint> &rigidParticleAttachments =
        world.rigidParticleAttachments();
    const std::vector<StrandRigidAttachmentConstraint> &strandRigidAttachments =
        world.strandRigidAttachments();
    const std::vector<RigidDistanceConstraint> &rigidDistanceConstraints =
        world.rigidDistanceConstraints();
    const std::vector<RoutedCableConstraint> &routedCableConstraints =
        world.routedCableConstraints();
    const std::vector<RoutedCableRoutePoint> &routedCableRoutePoints =
        world.routedCableRoutePoints();
    const std::vector<StrandSoftSuturingPair> &suturingPairs  = world.suturingPairs();
    const std::vector<std::uint32_t> &suturingParticleIndices = world.suturingParticleIndices();
    const std::uint32_t suturingPathHeaderCount         = world.reservedSuturingPathHeaderCount();
    const std::uint32_t suturingPathNodeCount           = world.reservedSuturingPathNodeCount();
    const std::uint64_t softParticleRevision            = world.softParticleRevision();
    const std::uint64_t softTopologyRevision            = world.softTopologyRevision();
    const std::uint64_t softConstraintAdjacencyRevision = world.softConstraintAdjacencyRevision();
    const std::uint64_t rigidParticleAttachmentResolvedRevision =
        world.rigidParticleAttachmentResolvedRevision();
    const std::uint64_t strandRigidAttachmentResolvedRevision =
        world.strandRigidAttachmentResolvedRevision();
    const std::uint64_t rigidDistanceConstraintResolvedRevision =
        world.rigidDistanceConstraintResolvedRevision();
    const std::uint64_t routedCableResolvedRevision = world.routedCableResolvedRevision();
    const std::uint64_t curveRenderRevision         = world.curveRenderRevision();
    if (static_cast<std::uint32_t>(rigidBodies.size()) != bodyCount ||
        static_cast<std::uint32_t>(colliders.size()) != colliderCount)
    {
        return false;
    }

    const std::vector<StrandDistanceConstraint> &strandDistanceConstraints =
        world.strandDistanceConstraints();
    if (bodyCount == 0u && colliderCount == 0u && particles.empty() &&
        distanceConstraints.empty() && bendConstraints.empty() && volumeConstraints.empty() &&
        strandSegments.empty() && strandJoints.empty() && strandDistanceConstraints.empty())
    {
        clearPublishedSceneCounts();
        mRigidJointUploadResetRequired       = true;
        mSoftParticleUploadResetRequired     = true;
        mSoftTopologyUploadResetRequired     = true;
        mRigidParticleAttachmentGpuDirty     = true;
        mStrandRigidAttachmentGpuDirty       = true;
        mRigidDistanceConstraintGpuDirty     = true;
        mRoutedCableGpuDirty                 = true;
        mLastUploadedRigidJointSceneRevision = world.rigidJointSceneRevision();
        mLastUploadedRigidJointModeRevision  = world.rigidJointModeRevision();
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
    if (!uploadJointCollisionSuppression(computeContext, jointCollisionSuppression, bodyCount))
    {
        return false;
    }
    if (!uploadRigidJoints(computeContext, world))
    {
        return false;
    }

    const bool needsSoftParticleUpload = mSoftParticleUploadResetRequired ||
                                         mLastUploadedSoftParticleRevision != softParticleRevision;
    const bool needsSoftTopologyUpload = mSoftTopologyUploadResetRequired ||
                                         mLastUploadedSoftTopologyRevision != softTopologyRevision;
    const bool needsRoutedCableUpload =
        mRoutedCableGpuDirty ||
        mLastUploadedRoutedCableResolvedRevision != routedCableResolvedRevision;
    const bool needsRigidParticleAttachmentUpload =
        mRigidParticleAttachmentGpuDirty || mLastUploadedRigidParticleAttachmentResolvedRevision !=
                                                rigidParticleAttachmentResolvedRevision;
    const bool needsStrandRigidAttachmentUpload =
        mStrandRigidAttachmentGpuDirty ||
        mLastUploadedStrandRigidAttachmentResolvedRevision != strandRigidAttachmentResolvedRevision;
    const bool needsRigidDistanceConstraintUpload =
        mRigidDistanceConstraintGpuDirty || mLastUploadedRigidDistanceConstraintResolvedRevision !=
                                                rigidDistanceConstraintResolvedRevision;
    const bool needsSuturingStateUpload = mSoftTopologyUploadResetRequired ||
                                          mLastUploadedSoftTopologyRevision != softTopologyRevision;
    const bool needsCurveRenderUpload =
        mSoftTopologyUploadResetRequired || mLastUploadedCurveRenderRevision != curveRenderRevision;
    const bool needsSoftConstraintAdjacencyUpload =
        mSoftTopologyUploadResetRequired ||
        mLastUploadedSoftConstraintAdjacencyRevision != softConstraintAdjacencyRevision;

    if ((needsSoftParticleUpload && !uploadParticles(computeContext, particles, fluids,
                                                     particleContactMaterials, fluidMaterials)) ||
        ((needsSoftTopologyUpload || needsSoftConstraintAdjacencyUpload) &&
         !uploadSoftTopology(
             computeContext, static_cast<std::uint32_t>(particles.size()), softRenderData,
             distanceConstraints, bendConstraints, volumeConstraints, strandSegments, strandJoints,
             strandDistanceConstraints, strandSegmentStates, strandRigidAttachments)))
    {
        return false;
    }
    if (needsSuturingStateUpload &&
        !uploadSuturingState(computeContext, particles,
                             static_cast<std::uint32_t>(particles.size()), suturingParticleIndices,
                             suturingPairs, suturingPathHeaderCount, suturingPathNodeCount))
    {
        return false;
    }
    if (needsRigidParticleAttachmentUpload &&
        !uploadRigidParticleAttachments(computeContext, rigidParticleAttachments))
    {
        return false;
    }
    if (needsStrandRigidAttachmentUpload &&
        !uploadStrandRigidAttachments(computeContext, strandRigidAttachments))
    {
        return false;
    }
    if (needsRigidDistanceConstraintUpload &&
        !uploadRigidDistanceConstraints(computeContext, rigidDistanceConstraints))
    {
        return false;
    }
    if (needsRoutedCableUpload &&
        !uploadRoutedCableTopology(computeContext, world.rigidBodySnapshot(),
                                   routedCableConstraints, routedCableRoutePoints))
    {
        return false;
    }
    if (needsCurveRenderUpload && !uploadCurveRenderData(computeContext, curveRenderData))
    {
        return false;
    }

    world.clearRigidBodyUploadState();
    world.clearColliderUploadState();
    publishSceneCounts(world, bodyCount, colliderCount, fluids, particleContactMaterials,
                       fluidMaterials, distanceConstraints, bendConstraints, volumeConstraints,
                       strandSegments, strandJoints, strandDistanceConstraints,
                       rigidParticleAttachments, strandRigidAttachments, rigidDistanceConstraints,
                       routedCableConstraints, suturingPairs, suturingParticleIndices,
                       suturingPathHeaderCount, suturingPathNodeCount, curveRenderData);
    mRigidBodyUploadResetRequired                        = false;
    mColliderUploadResetRequired                         = false;
    mSoftParticleUploadResetRequired                     = false;
    mSoftTopologyUploadResetRequired                     = false;
    mRigidParticleAttachmentGpuDirty                     = false;
    mStrandRigidAttachmentGpuDirty                       = false;
    mRigidDistanceConstraintGpuDirty                     = false;
    mRoutedCableGpuDirty                                 = false;
    mLastUploadedSoftParticleRevision                    = softParticleRevision;
    mLastUploadedSoftTopologyRevision                    = softTopologyRevision;
    mLastUploadedSoftConstraintAdjacencyRevision         = softConstraintAdjacencyRevision;
    mLastUploadedRigidParticleAttachmentResolvedRevision = rigidParticleAttachmentResolvedRevision;
    mLastUploadedStrandRigidAttachmentResolvedRevision   = strandRigidAttachmentResolvedRevision;
    mLastUploadedRigidDistanceConstraintResolvedRevision = rigidDistanceConstraintResolvedRevision;
    mLastUploadedRoutedCableResolvedRevision             = routedCableResolvedRevision;
    mLastUploadedCurveRenderRevision                     = curveRenderRevision;
    mStaticBroadPhaseDirty = mStaticBroadPhaseDirty || world.staticBroadPhaseDirty();
    world.clearStaticBroadPhaseDirty();
    return true;
}

void PhysicsSceneGpuState::clearPublishedSceneCounts() noexcept
{
    mRigidBodyCount                = 0u;
    mColliderCount                 = 0u;
    mBallJointCount                = 0u;
    mHingeJointCount               = 0u;
    mSliderJointCount              = 0u;
    mSoftBodyCount                 = 0u;
    mSoftParticleCount             = 0u;
    mFluidCount                    = 0u;
    mParticleContactMaterialCount  = 0u;
    mFluidMaterialCount            = 0u;
    mSoftEdgeCount                 = 0u;
    mSoftBendCount                 = 0u;
    mSoftTetCount                  = 0u;
    mStrandSegmentCount            = 0u;
    mStrandJointCount              = 0u;
    mStrandDistanceCount           = 0u;
    mSuturingPairCount             = 0u;
    mSuturingParticleCount         = 0u;
    mSuturingPathHeaderCount       = 0u;
    mSuturingPathNodeCount         = 0u;
    mCurveRenderCount              = 0u;
    mRigidParticleAttachmentCount  = 0u;
    mStrandRigidAttachmentCount    = 0u;
    mRigidDistanceConstraintCount  = 0u;
    mRoutedCableCount              = 0u;
    mRoutedCableDebugSegmentCount  = 0u;
    mHingePassiveJointCount        = 0u;
    mHingePositionDriveJointCount  = 0u;
    mHingeVelocityDriveJointCount  = 0u;
    mSliderPassiveJointCount       = 0u;
    mSliderPositionDriveJointCount = 0u;
    mSliderVelocityDriveJointCount = 0u;
}

void PhysicsSceneGpuState::publishSceneCounts(
    const PhysicsWorld &world, std::uint32_t bodyCount, std::uint32_t colliderCount,
    const std::vector<FluidState> &fluids,
    const std::vector<Diligent::float4> &particleContactMaterials,
    const std::vector<FluidMaterialGpu> &fluidMaterials,
    const std::vector<DeformableDistanceConstraint> &distanceConstraints,
    const std::vector<DeformableBendConstraint> &bendConstraints,
    const std::vector<DeformableVolumeConstraint> &volumeConstraints,
    const std::vector<StrandSegmentConstraint> &strandSegments,
    const std::vector<StrandJointConstraint> &strandJoints,
    const std::vector<StrandDistanceConstraint> &strandDistanceConstraints,
    const std::vector<RigidParticleAttachmentConstraint> &rigidParticleAttachments,
    const std::vector<StrandRigidAttachmentConstraint> &strandRigidAttachments,
    const std::vector<RigidDistanceConstraint> &rigidDistanceConstraints,
    const std::vector<RoutedCableConstraint> &routedCableConstraints,
    const std::vector<StrandSoftSuturingPair> &suturingPairs,
    const std::vector<std::uint32_t> &suturingParticleIndices,
    std::uint32_t suturingPathHeaderCount, std::uint32_t suturingPathNodeCount,
    const CurveRenderDataHost &curveRenderData) noexcept
{
    mRigidBodyCount               = bodyCount;
    mColliderCount                = colliderCount;
    mSoftBodyCount                = world.softBodyCount();
    mSoftParticleCount            = static_cast<std::uint32_t>(world.particles().size());
    mFluidCount                   = static_cast<std::uint32_t>(fluids.size());
    mParticleContactMaterialCount = static_cast<std::uint32_t>(particleContactMaterials.size());
    mFluidMaterialCount           = static_cast<std::uint32_t>(fluidMaterials.size());
    mSoftEdgeCount                = static_cast<std::uint32_t>(distanceConstraints.size());
    mSoftBendCount                = static_cast<std::uint32_t>(bendConstraints.size());
    mSoftTetCount                 = static_cast<std::uint32_t>(volumeConstraints.size());
    mStrandSegmentCount           = static_cast<std::uint32_t>(strandSegments.size());
    mStrandJointCount             = static_cast<std::uint32_t>(strandJoints.size());
    mStrandDistanceCount          = static_cast<std::uint32_t>(strandDistanceConstraints.size());
    mRigidParticleAttachmentCount = static_cast<std::uint32_t>(rigidParticleAttachments.size());
    mStrandRigidAttachmentCount   = static_cast<std::uint32_t>(strandRigidAttachments.size());
    mRigidDistanceConstraintCount = static_cast<std::uint32_t>(rigidDistanceConstraints.size());
    mRoutedCableCount             = static_cast<std::uint32_t>(routedCableConstraints.size());
    mRoutedCableDebugSegmentCount = 0u;
    for (const RoutedCableConstraint &constraint : routedCableConstraints)
    {
        if (constraint.routePointCount > 1u)
        {
            mRoutedCableDebugSegmentCount += constraint.routePointCount - 1u;
        }
    }
    mSuturingPairCount       = static_cast<std::uint32_t>(suturingPairs.size());
    mSuturingParticleCount   = static_cast<std::uint32_t>(suturingParticleIndices.size());
    mSuturingPathHeaderCount = suturingPathHeaderCount;
    mSuturingPathNodeCount   = suturingPathNodeCount;
    mCurveRenderCount        = static_cast<std::uint32_t>(curveRenderData.descriptors.size());
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
                                       mPersistentRigidBodies.proxyParticleContactMaterialsBuffer,
                                       rigidBodies.proxyParticleContactMaterials, begin, count) &&
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
    if (!updateStructuredBufferRange(computeContext,
                                     mPersistentColliders.ownerRigidBodyIndicesBuffer,
                                     colliders.ownerRigidBodyIndices, begin, count) ||
        !updateStructuredBufferRange(computeContext, mPersistentColliders.shapeTypesBuffer,
                                     colliders.shapeTypes, begin, count) ||
        !updateStructuredBufferRange(computeContext, mPersistentColliders.enabledFlagsBuffer,
                                     colliders.enabledFlags, begin, count) ||
        !updateStructuredBufferRange(computeContext, mPersistentColliders.materialBuffer,
                                     colliders.frictionRestitution, begin, count))
    {
        return false;
    }

    std::vector<GpuColliderContactData> contactData(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        const std::uint32_t sourceIndex = begin + i;
        GpuColliderContactData &entry   = contactData[i];
        entry.ownerBody                 = colliders.ownerRigidBodyIndices[sourceIndex];
        entry.shapeType                 = colliders.shapeTypes[sourceIndex];
        entry.reserved0                 = colliders.enabledFlags[sourceIndex];
        entry.reserved1                 = 0u;
        entry.shapeParams               = colliders.shapeParams[sourceIndex];
        entry.localPosition             = colliders.localPositions[sourceIndex];
        entry.localOrientation          = colliders.localOrientations[sourceIndex];
        entry.material                  = colliders.frictionRestitution[sourceIndex];
    }

    std::vector<GpuColliderGeometryData> geometryData(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        const std::uint32_t sourceIndex = begin + i;
        GpuColliderGeometryData &entry  = geometryData[i];
        entry.shapeParams               = colliders.shapeParams[sourceIndex];
        entry.localPosition             = colliders.localPositions[sourceIndex];
        entry.localOrientation          = colliders.localOrientations[sourceIndex];
    }

    computeContext->UpdateBuffer(
        mPersistentColliders.geometryDataBuffer,
        static_cast<Diligent::Uint64>(begin) * sizeof(GpuColliderGeometryData),
        static_cast<Diligent::Uint64>(count) * sizeof(GpuColliderGeometryData), geometryData.data(),
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(
        mPersistentColliders.contactDataBuffer,
        static_cast<Diligent::Uint64>(begin) * sizeof(GpuColliderContactData),
        static_cast<Diligent::Uint64>(count) * sizeof(GpuColliderContactData), contactData.data(),
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return true;
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

bool PhysicsSceneGpuState::uploadJointCollisionSuppression(
    Diligent::IDeviceContext *computeContext, const JointCollisionSuppressionHost &suppression,
    std::uint32_t bodyCount)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const std::vector<std::uint32_t> &neighborOffsets = suppression.neighborOffsets;
    const std::vector<std::uint32_t> &neighbors       = suppression.neighbors;
    if (neighborOffsets.size() != static_cast<std::size_t>(bodyCount + 1u))
    {
        CRESSIM_LOG_ERROR("PhysicsSceneGpuState: joint collision suppression offsets size ",
                          neighborOffsets.size(),
                          " did not match expected bodyCount+1=", (bodyCount + 1u), ".");
        return false;
    }

    return updateStructuredBufferRange(
               computeContext, mPersistentJointCollisionSuppression.neighborOffsetsBuffer,
               neighborOffsets, 0u, static_cast<std::uint32_t>(neighborOffsets.size())) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentJointCollisionSuppression.neighborsBuffer,
                                       neighbors, 0u, static_cast<std::uint32_t>(neighbors.size()));
}

bool PhysicsSceneGpuState::uploadRigidJoints(Diligent::IDeviceContext *computeContext,
                                             const PhysicsWorld &world)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const std::uint64_t sceneRevision = world.rigidJointSceneRevision();
    const std::uint64_t modeRevision  = world.rigidJointModeRevision();
    const bool needsJointBufferUpload =
        mRigidJointUploadResetRequired || mLastUploadedRigidJointSceneRevision != sceneRevision;
    const bool needsModeIndexUpload =
        mRigidJointUploadResetRequired || mLastUploadedRigidJointModeRevision != modeRevision;
    if (!needsJointBufferUpload && !needsModeIndexUpload)
    {
        return true;
    }
    const RigidJointSceneHost &jointScene = world.rigidJointScene();

    std::vector<GpuBallJoint> ballJoints(jointScene.ball.size());
    for (std::size_t i = 0; i < ballJoints.size(); ++i)
    {
        GpuBallJoint &dst = ballJoints[i];
        dst.bodyA         = jointScene.ball.bodyIndicesA[i];
        dst.bodyB         = jointScene.ball.bodyIndicesB[i];
        dst.enabled       = jointScene.ball.enabledFlags[i];
        dst.localAnchorA  = jointScene.ball.localAnchorsA[i];
        dst.localAnchorB  = jointScene.ball.localAnchorsB[i];
    }

    std::vector<GpuSphericalJoint> sphericalJoints(jointScene.spherical.size());
    for (std::size_t i = 0; i < sphericalJoints.size(); ++i)
    {
        GpuSphericalJoint &dst = sphericalJoints[i];
        dst.bodyA              = jointScene.spherical.bodyIndicesA[i];
        dst.bodyB              = jointScene.spherical.bodyIndicesB[i];
        dst.enabled            = jointScene.spherical.enabledFlags[i];
        dst.driveMode          = jointScene.spherical.driveModes[i];
        dst.localAnchorA       = jointScene.spherical.localAnchorsA[i];
        dst.localAnchorB       = jointScene.spherical.localAnchorsB[i];
        dst.localRotationA     = jointScene.spherical.localRotationsA[i];
        dst.localRotationB     = jointScene.spherical.localRotationsB[i];
        dst.limitParams0       = Diligent::float4{
            static_cast<float>(jointScene.spherical.limitEnabledFlags[i]),
            jointScene.spherical.swingLimitYs[i], jointScene.spherical.swingLimitZs[i],
            jointScene.spherical.constraintCompliances[i]};
        dst.limitParams1 = Diligent::float4{
            jointScene.spherical.twistLimitMins[i], jointScene.spherical.twistLimitMaxs[i],
            jointScene.spherical.swingCompliances[i], jointScene.spherical.twistCompliances[i]};
        dst.driveTargetOrientation = jointScene.spherical.driveTargetOrientations[i];
        dst.driveParams =
            Diligent::float4{jointScene.spherical.driveCompliances[i], 0.0f, 0.0f, 0.0f};
    }

    std::vector<GpuHingeJoint> hingeJoints(jointScene.hinge.size());
    std::vector<std::uint32_t> hingePassiveJointIndices;
    std::vector<std::uint32_t> hingePositionDriveJointIndices;
    std::vector<std::uint32_t> hingeVelocityDriveJointIndices;
    if (needsModeIndexUpload)
    {
        hingePassiveJointIndices.reserve(jointScene.hinge.size());
        hingePositionDriveJointIndices.reserve(jointScene.hinge.size());
        hingeVelocityDriveJointIndices.reserve(jointScene.hinge.size());
    }
    for (std::size_t i = 0; i < hingeJoints.size(); ++i)
    {
        GpuHingeJoint &dst = hingeJoints[i];
        dst.bodyA          = jointScene.hinge.bodyIndicesA[i];
        dst.bodyB          = jointScene.hinge.bodyIndicesB[i];
        dst.enabled        = jointScene.hinge.enabledFlags[i];
        dst.driveMode      = jointScene.hinge.driveModes[i];
        dst.localAnchorA   = jointScene.hinge.localAnchorsA[i];
        dst.localAnchorB   = jointScene.hinge.localAnchorsB[i];
        dst.localAxisA0    = jointScene.hinge.localAxesA0[i];
        dst.projectionRow0 = jointScene.hinge.projectionRow0[i];
        dst.projectionRow1 = jointScene.hinge.projectionRow1[i];
        dst.projectionRow2 = jointScene.hinge.projectionRow2[i];
        dst.limitParams =
            Diligent::float4{static_cast<float>(jointScene.hinge.limitEnabledFlags[i]),
                             jointScene.hinge.limitMins[i], jointScene.hinge.limitMaxs[i],
                             jointScene.hinge.constraintCompliances[i]};
        dst.driveTargetParams = Diligent::float4{jointScene.hinge.driveTargetAngles[i],
                                                 jointScene.hinge.driveTargetAngularVelocities[i],
                                                 jointScene.hinge.driveCompliances[i], 0.0f};

        if (!needsModeIndexUpload || jointScene.hinge.enabledFlags[i] == 0u)
        {
            continue;
        }

        if (jointScene.hinge.driveModes[i] ==
            static_cast<std::uint32_t>(RigidJointDriveMode::TargetPosition))
        {
            hingePositionDriveJointIndices.push_back(static_cast<std::uint32_t>(i));
        }
        else
        {
            hingePassiveJointIndices.push_back(static_cast<std::uint32_t>(i));
            if (jointScene.hinge.driveModes[i] ==
                static_cast<std::uint32_t>(RigidJointDriveMode::TargetVelocity))
            {
                hingeVelocityDriveJointIndices.push_back(static_cast<std::uint32_t>(i));
            }
        }
    }

    std::vector<GpuSliderJoint> sliderJoints(jointScene.slider.size());
    std::vector<std::uint32_t> sliderPassiveJointIndices;
    std::vector<std::uint32_t> sliderPositionDriveJointIndices;
    std::vector<std::uint32_t> sliderVelocityDriveJointIndices;
    if (needsModeIndexUpload)
    {
        sliderPassiveJointIndices.reserve(jointScene.slider.size());
        sliderPositionDriveJointIndices.reserve(jointScene.slider.size());
        sliderVelocityDriveJointIndices.reserve(jointScene.slider.size());
    }
    for (std::size_t i = 0; i < sliderJoints.size(); ++i)
    {
        GpuSliderJoint &dst = sliderJoints[i];
        dst.bodyA           = jointScene.slider.bodyIndicesA[i];
        dst.bodyB           = jointScene.slider.bodyIndicesB[i];
        dst.enabled         = jointScene.slider.enabledFlags[i];
        dst.driveMode       = jointScene.slider.driveModes[i];
        dst.localAnchorA    = jointScene.slider.localAnchorsA[i];
        dst.localAnchorB    = jointScene.slider.localAnchorsB[i];
        dst.localAxisA0     = jointScene.slider.localAxesA0[i];
        dst.localAxisA1     = jointScene.slider.localAxesA1[i];
        dst.localAxisA2     = jointScene.slider.localAxesA2[i];
        dst.projectionRow0  = jointScene.slider.projectionRow0[i];
        dst.projectionRow1  = jointScene.slider.projectionRow1[i];
        dst.projectionRow2  = jointScene.slider.projectionRow2[i];
        dst.limitParams =
            Diligent::float4{static_cast<float>(jointScene.slider.limitEnabledFlags[i]),
                             jointScene.slider.limitMins[i], jointScene.slider.limitMaxs[i],
                             jointScene.slider.constraintCompliances[i]};
        dst.driveTargetParams = Diligent::float4{
            jointScene.slider.driveTargetPositions[i], jointScene.slider.driveRestOffsets[i],
            jointScene.slider.driveTargetVelocities[i], jointScene.slider.driveCompliances[i]};

        if (!needsModeIndexUpload || jointScene.slider.enabledFlags[i] == 0u)
        {
            continue;
        }

        if (jointScene.slider.driveModes[i] ==
            static_cast<std::uint32_t>(RigidJointDriveMode::TargetPosition))
        {
            sliderPositionDriveJointIndices.push_back(static_cast<std::uint32_t>(i));
        }
        else
        {
            sliderPassiveJointIndices.push_back(static_cast<std::uint32_t>(i));
            if (jointScene.slider.driveModes[i] ==
                static_cast<std::uint32_t>(RigidJointDriveMode::TargetVelocity))
            {
                sliderVelocityDriveJointIndices.push_back(static_cast<std::uint32_t>(i));
            }
        }
    }

    if (needsJointBufferUpload &&
        (!updateStructuredBufferRange(computeContext, mPersistentJoints.ballJointsBuffer,
                                      ballJoints, 0u,
                                      static_cast<std::uint32_t>(ballJoints.size())) ||
         !updateStructuredBufferRange(computeContext, mPersistentJoints.sphericalJointsBuffer,
                                      sphericalJoints, 0u,
                                      static_cast<std::uint32_t>(sphericalJoints.size())) ||
         !updateStructuredBufferRange(computeContext, mPersistentJoints.hingeJointsBuffer,
                                      hingeJoints, 0u,
                                      static_cast<std::uint32_t>(hingeJoints.size())) ||
         !updateStructuredBufferRange(computeContext, mPersistentJoints.sliderJointsBuffer,
                                      sliderJoints, 0u,
                                      static_cast<std::uint32_t>(sliderJoints.size()))))
    {
        return false;
    }

    if (needsModeIndexUpload &&
        (!updateStructuredBufferRange(
             computeContext, mPersistentJoints.hingePassiveJointIndicesBuffer,
             hingePassiveJointIndices, 0u,
             static_cast<std::uint32_t>(hingePassiveJointIndices.size())) ||
         !updateStructuredBufferRange(
             computeContext, mPersistentJoints.hingePositionDriveJointIndicesBuffer,
             hingePositionDriveJointIndices, 0u,
             static_cast<std::uint32_t>(hingePositionDriveJointIndices.size())) ||
         !updateStructuredBufferRange(
             computeContext, mPersistentJoints.hingeVelocityDriveJointIndicesBuffer,
             hingeVelocityDriveJointIndices, 0u,
             static_cast<std::uint32_t>(hingeVelocityDriveJointIndices.size())) ||
         !updateStructuredBufferRange(
             computeContext, mPersistentJoints.sliderPassiveJointIndicesBuffer,
             sliderPassiveJointIndices, 0u,
             static_cast<std::uint32_t>(sliderPassiveJointIndices.size())) ||
         !updateStructuredBufferRange(
             computeContext, mPersistentJoints.sliderPositionDriveJointIndicesBuffer,
             sliderPositionDriveJointIndices, 0u,
             static_cast<std::uint32_t>(sliderPositionDriveJointIndices.size())) ||
         !updateStructuredBufferRange(
             computeContext, mPersistentJoints.sliderVelocityDriveJointIndicesBuffer,
             sliderVelocityDriveJointIndices, 0u,
             static_cast<std::uint32_t>(sliderVelocityDriveJointIndices.size()))))
    {
        return false;
    }

    mBallJointCount      = static_cast<std::uint32_t>(ballJoints.size());
    mSphericalJointCount = static_cast<std::uint32_t>(sphericalJoints.size());
    mHingeJointCount     = static_cast<std::uint32_t>(hingeJoints.size());
    mSliderJointCount    = static_cast<std::uint32_t>(sliderJoints.size());
    if (needsModeIndexUpload)
    {
        mHingePassiveJointCount = static_cast<std::uint32_t>(hingePassiveJointIndices.size());
        mHingePositionDriveJointCount =
            static_cast<std::uint32_t>(hingePositionDriveJointIndices.size());
        mHingeVelocityDriveJointCount =
            static_cast<std::uint32_t>(hingeVelocityDriveJointIndices.size());
        mSliderPassiveJointCount = static_cast<std::uint32_t>(sliderPassiveJointIndices.size());
        mSliderPositionDriveJointCount =
            static_cast<std::uint32_t>(sliderPositionDriveJointIndices.size());
        mSliderVelocityDriveJointCount =
            static_cast<std::uint32_t>(sliderVelocityDriveJointIndices.size());
    }
    mLastUploadedRigidJointSceneRevision = sceneRevision;
    mLastUploadedRigidJointModeRevision  = modeRevision;
    mRigidJointUploadResetRequired       = false;
    return true;
}

bool PhysicsSceneGpuState::uploadParticles(
    Diligent::IDeviceContext *computeContext, const ParticleSoAHost &particles,
    const std::vector<FluidState> &fluids,
    const std::vector<Diligent::float4> &particleContactMaterials,
    const std::vector<FluidMaterialGpu> &fluidMaterials)
{
    const std::uint32_t count = static_cast<std::uint32_t>(particles.size());
    if (count == 0u && fluids.empty() && particleContactMaterials.empty() && fluidMaterials.empty())
    {
        return true;
    }

    std::vector<Diligent::float4> fluidVisuals(fluids.size());
    for (std::size_t i = 0; i < fluids.size(); ++i)
    {
        fluidVisuals[i] = fluids[i].visualColor;
    }

    return updateStructuredBufferRange(computeContext, mPersistentParticles.positionsInvMassBuffer,
                                       particles.positionsInvMass, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.previousPositionsBuffer,
                                       particles.previousPositions, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.velocitiesBuffer,
                                       particles.velocities, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.radiiBuffer,
                                       particles.radii, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentParticles.environmentIndicesBuffer,
                                       particles.environmentIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.particleKindsBuffer,
                                       particles.particleKinds, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.ownerTypesBuffer,
                                       particles.ownerTypes, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.ownerIndicesBuffer,
                                       particles.ownerIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.strandIdsBuffer,
                                       particles.strandIds, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.strandRolesBuffer,
                                       particles.strandRoles, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentParticles.suturingNeighborLinksBuffer,
                                       particles.suturingNeighborLinks, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentParticles.owningSoftBodyIndicesBuffer,
                                       particles.owningSoftBodyIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentParticles.particleMaterialIndicesBuffer,
                                       particles.particleMaterialIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentParticles.fluidMaterialIndicesBuffer,
                                       particles.fluidMaterialIndices, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.fluidVisualsBuffer,
                                       fluidVisuals, 0u,
                                       static_cast<std::uint32_t>(fluidVisuals.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentParticles.particleContactMaterialsBuffer,
               particleContactMaterials, 0u,
               static_cast<std::uint32_t>(particleContactMaterials.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.fluidMaterialsBuffer,
                                       fluidMaterials, 0u,
                                       static_cast<std::uint32_t>(fluidMaterials.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.phasesBuffer,
                                       particles.phases, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.collisionLayersBuffer,
                                       particles.collisionLayers, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.collisionMasksBuffer,
                                       particles.collisionMasks, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.adjacencyOffsetsBuffer,
                                       particles.adjacencyOffsets, 0u, count) &&
           updateStructuredBufferRange(computeContext, mPersistentParticles.adjacencyCountsBuffer,
                                       particles.adjacencyCounts, 0u, count) &&
           updateStructuredBufferRange(
               computeContext, mPersistentParticles.adjacencyIndicesBuffer,
               particles.adjacencyIndices, 0u,
               static_cast<std::uint32_t>(particles.adjacencyIndices.size())) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentParticles.rigidProxyLocalPositionsBuffer,
                                       particles.rigidProxyLocalPositions, 0u, count) &&
           [&]()
    {
        std::vector<Diligent::uint4> metadata(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            metadata[i] =
                Diligent::uint4{particles.environmentIndices[i], particles.phases[i],
                                particles.collisionLayers[i], particles.collisionMasks[i]};
        }
        return updateStructuredBufferRange(
            computeContext, mPersistentParticles.broadPhaseMetadataBuffer, metadata, 0u, count);
    }();
}

bool PhysicsSceneGpuState::uploadSoftTopology(
    Diligent::IDeviceContext *computeContext, std::uint32_t particleCount,
    const SoftRenderDataHost &softRenderData,
    const std::vector<DeformableDistanceConstraint> &distanceConstraints,
    const std::vector<DeformableBendConstraint> &bendConstraints,
    const std::vector<DeformableVolumeConstraint> &volumeConstraints,
    const std::vector<StrandSegmentConstraint> &strandSegments,
    const std::vector<StrandJointConstraint> &strandJoints,
    const std::vector<StrandDistanceConstraint> &strandDistanceConstraints,
    const std::vector<StrandSegmentState> &strandSegmentStates,
    const std::vector<StrandRigidAttachmentConstraint> &strandRigidAttachments)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    std::vector<std::vector<GpuSoftIncidentEdge>> particleEdgeRefs(particleCount);
    for (std::uint32_t edgeIndex = 0u;
         edgeIndex < static_cast<std::uint32_t>(distanceConstraints.size()); ++edgeIndex)
    {
        const DeformableDistanceConstraint &edge = distanceConstraints[edgeIndex];
        if (edge.particleA < particleCount)
        {
            particleEdgeRefs[edge.particleA].push_back(GpuSoftIncidentEdge{edgeIndex, 0u, 0u, 0u});
        }
        if (edge.particleB < particleCount)
        {
            particleEdgeRefs[edge.particleB].push_back(GpuSoftIncidentEdge{edgeIndex, 1u, 0u, 0u});
        }
    }

    std::vector<std::vector<GpuSoftIncidentBend>> particleBendRefs(particleCount);
    for (std::uint32_t bendIndex = 0u;
         bendIndex < static_cast<std::uint32_t>(bendConstraints.size()); ++bendIndex)
    {
        const DeformableBendConstraint &bend = bendConstraints[bendIndex];
        const std::array<std::uint32_t, 3u> particleIndices{bend.particle0, bend.particle1,
                                                            bend.particle2};
        for (std::uint32_t slot = 0u; slot < particleIndices.size(); ++slot)
        {
            const std::uint32_t particleIndex = particleIndices[slot];
            if (particleIndex < particleCount)
            {
                particleBendRefs[particleIndex].push_back(
                    GpuSoftIncidentBend{bendIndex, slot, 0u, 0u});
            }
        }
    }

    std::vector<std::vector<GpuSoftIncidentTet>> particleTetRefs(particleCount);
    for (std::uint32_t tetIndex = 0u;
         tetIndex < static_cast<std::uint32_t>(volumeConstraints.size()); ++tetIndex)
    {
        const DeformableVolumeConstraint &tet = volumeConstraints[tetIndex];
        const std::array<std::uint32_t, 4u> particleIndices{
            tet.particleIndices[0], tet.particleIndices[1], tet.particleIndices[2],
            tet.particleIndices[3]};
        for (std::uint32_t slot = 0u; slot < particleIndices.size(); ++slot)
        {
            const std::uint32_t particleIndex = particleIndices[slot];
            if (particleIndex < particleCount)
            {
                particleTetRefs[particleIndex].push_back(
                    GpuSoftIncidentTet{tetIndex, slot, 0u, 0u});
            }
        }
    }

    std::vector<std::vector<GpuStrandIncidentSegment>> particleStrandSegmentRefs(particleCount);
    for (std::uint32_t segmentIndex = 0u;
         segmentIndex < static_cast<std::uint32_t>(strandSegments.size()); ++segmentIndex)
    {
        const StrandSegmentConstraint &segment = strandSegments[segmentIndex];
        if (segment.particleA < particleCount)
        {
            particleStrandSegmentRefs[segment.particleA].push_back(
                GpuStrandIncidentSegment{segmentIndex, 0u, 0u, 0u});
        }
        if (segment.particleB < particleCount)
        {
            particleStrandSegmentRefs[segment.particleB].push_back(
                GpuStrandIncidentSegment{segmentIndex, 1u, 0u, 0u});
        }
    }

    std::vector<std::vector<GpuStrandIncidentJoint>> particleStrandJointRefs(particleCount);
    std::vector<std::vector<GpuStrandIncidentJoint>> segmentStrandJointRefs(strandSegments.size());
    for (std::uint32_t jointIndex = 0u;
         jointIndex < static_cast<std::uint32_t>(strandJoints.size()); ++jointIndex)
    {
        const StrandJointConstraint &joint = strandJoints[jointIndex];
        if (joint.segmentA >= strandSegments.size() || joint.segmentB >= strandSegments.size())
        {
            continue;
        }

        const StrandSegmentConstraint &segmentA = strandSegments[joint.segmentA];
        const StrandSegmentConstraint &segmentB = strandSegments[joint.segmentB];
        const std::array<std::uint32_t, 3u> particleIndices{segmentA.particleA, segmentA.particleB,
                                                            segmentB.particleB};
        for (std::uint32_t slot = 0u; slot < particleIndices.size(); ++slot)
        {
            const std::uint32_t particleIndex = particleIndices[slot];
            if (particleIndex < particleCount)
            {
                particleStrandJointRefs[particleIndex].push_back(
                    GpuStrandIncidentJoint{jointIndex, slot, 0u, 0u});
            }
        }

        segmentStrandJointRefs[joint.segmentA].push_back(
            GpuStrandIncidentJoint{jointIndex, 0u, 0u, 0u});
        segmentStrandJointRefs[joint.segmentB].push_back(
            GpuStrandIncidentJoint{jointIndex, 1u, 0u, 0u});
    }

    std::vector<GpuSoftConstraintRange> particleEdgeRanges;
    buildConstraintAdjacencyRanges(particleCount, particleEdgeRefs, particleEdgeRanges);
    std::vector<GpuSoftIncidentEdge> incidentEdges;
    incidentEdges.reserve(distanceConstraints.size() * 2u);
    for (const auto &refs : particleEdgeRefs)
    {
        incidentEdges.insert(incidentEdges.end(), refs.begin(), refs.end());
    }

    std::vector<GpuSoftConstraintRange> particleBendRanges;
    buildConstraintAdjacencyRanges(particleCount, particleBendRefs, particleBendRanges);
    std::vector<GpuSoftIncidentBend> incidentBends;
    incidentBends.reserve(bendConstraints.size() * 3u);
    for (const auto &refs : particleBendRefs)
    {
        incidentBends.insert(incidentBends.end(), refs.begin(), refs.end());
    }

    std::vector<GpuSoftConstraintRange> particleTetRanges;
    buildConstraintAdjacencyRanges(particleCount, particleTetRefs, particleTetRanges);
    std::vector<GpuSoftIncidentTet> incidentTets;
    incidentTets.reserve(volumeConstraints.size() * 4u);
    for (const auto &refs : particleTetRefs)
    {
        incidentTets.insert(incidentTets.end(), refs.begin(), refs.end());
    }

    std::vector<GpuSoftConstraintRange> particleStrandSegmentRanges;
    buildConstraintAdjacencyRanges(particleCount, particleStrandSegmentRefs,
                                   particleStrandSegmentRanges);
    std::vector<GpuStrandIncidentSegment> incidentStrandSegments;
    incidentStrandSegments.reserve(strandSegments.size() * 2u);
    for (const auto &refs : particleStrandSegmentRefs)
    {
        incidentStrandSegments.insert(incidentStrandSegments.end(), refs.begin(), refs.end());
    }

    std::vector<GpuSoftConstraintRange> particleStrandJointRanges;
    buildConstraintAdjacencyRanges(particleCount, particleStrandJointRefs,
                                   particleStrandJointRanges);
    std::vector<GpuStrandIncidentJoint> incidentStrandJoints;
    incidentStrandJoints.reserve(strandJoints.size() * 3u);
    for (const auto &refs : particleStrandJointRefs)
    {
        incidentStrandJoints.insert(incidentStrandJoints.end(), refs.begin(), refs.end());
    }

    std::vector<GpuSoftConstraintRange> segmentStrandJointRanges;
    buildConstraintAdjacencyRanges(static_cast<std::uint32_t>(strandSegments.size()),
                                   segmentStrandJointRefs, segmentStrandJointRanges);
    std::vector<GpuStrandIncidentJoint> segmentIncidentStrandJoints;
    segmentIncidentStrandJoints.reserve(strandJoints.size() * 2u);
    for (const auto &refs : segmentStrandJointRefs)
    {
        segmentIncidentStrandJoints.insert(segmentIncidentStrandJoints.end(), refs.begin(),
                                           refs.end());
    }

    std::vector<std::vector<GpuStrandIncidentAttachment>> segmentStrandAttachmentRefs(
        strandSegments.size());
    for (std::uint32_t attachmentIndex = 0u;
         attachmentIndex < static_cast<std::uint32_t>(strandRigidAttachments.size());
         ++attachmentIndex)
    {
        const StrandRigidAttachmentConstraint &attachment = strandRigidAttachments[attachmentIndex];
        if (attachment.segmentIndex >= strandSegments.size())
        {
            continue;
        }
        segmentStrandAttachmentRefs[attachment.segmentIndex].push_back(
            GpuStrandIncidentAttachment{attachmentIndex, 0u, 0u, 0u});
    }

    std::vector<GpuSoftConstraintRange> segmentStrandAttachmentRanges;
    buildConstraintAdjacencyRanges(static_cast<std::uint32_t>(strandSegments.size()),
                                   segmentStrandAttachmentRefs, segmentStrandAttachmentRanges);
    std::vector<GpuStrandIncidentAttachment> segmentIncidentStrandAttachments;
    segmentIncidentStrandAttachments.reserve(strandRigidAttachments.size());
    for (const auto &refs : segmentStrandAttachmentRefs)
    {
        segmentIncidentStrandAttachments.insert(segmentIncidentStrandAttachments.end(),
                                                refs.begin(), refs.end());
    }

    std::vector<GpuSoftRenderVertexTriangleRange> renderVertexTriangleRanges(
        softRenderData.vertexTriangleRanges.size());
    for (std::size_t i = 0; i < softRenderData.vertexTriangleRanges.size(); ++i)
    {
        const SoftRenderVertexTriangleRange &src = softRenderData.vertexTriangleRanges[i];
        renderVertexTriangleRanges[i] =
            GpuSoftRenderVertexTriangleRange{src.start, src.count, src.reserved0, src.reserved1};
    }

    std::vector<GpuSoftBodyParticleRange> softBodyParticleRanges(
        softRenderData.softBodyParticleRanges.size());
    std::vector<GpuSoftBodyChunkRange> softBodyChunkRanges(
        softRenderData.softBodyParticleRanges.size());
    std::vector<Diligent::float4> initialRenderPositions(softRenderData.vertexBindings.size(),
                                                         Diligent::float4{});
    std::vector<GpuSoftBodyBoundsChunk> softBodyBoundsChunks;
    for (std::size_t i = 0; i < softRenderData.softBodyParticleRanges.size(); ++i)
    {
        const Diligent::uint2 &src        = softRenderData.softBodyParticleRanges[i];
        softBodyParticleRanges[i]         = GpuSoftBodyParticleRange{src.x, src.y, 0u, 0u};
        GpuSoftBodyChunkRange &chunkRange = softBodyChunkRanges[i];
        chunkRange.start                  = static_cast<std::uint32_t>(softBodyBoundsChunks.size());
        for (std::uint32_t offset = 0u; offset < src.y; offset += kSoftBodyBoundsChunkSize)
        {
            softBodyBoundsChunks.push_back(
                GpuSoftBodyBoundsChunk{static_cast<std::uint32_t>(i), src.x + offset,
                                       std::min(kSoftBodyBoundsChunkSize, src.y - offset), 0u});
        }
        chunkRange.count =
            static_cast<std::uint32_t>(softBodyBoundsChunks.size()) - chunkRange.start;
    }

    return updateStructuredBufferRange(computeContext, mPersistentSoftTopology.edgesBuffer,
                                       distanceConstraints, 0u,
                                       static_cast<std::uint32_t>(distanceConstraints.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentSoftTopology.bendsBuffer,
                                       bendConstraints, 0u,
                                       static_cast<std::uint32_t>(bendConstraints.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentSoftTopology.tetsBuffer,
                                       volumeConstraints, 0u,
                                       static_cast<std::uint32_t>(volumeConstraints.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentSoftTopology.strandSegmentsBuffer,
                                       strandSegments, 0u,
                                       static_cast<std::uint32_t>(strandSegments.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentSoftTopology.strandJointsBuffer,
                                       strandJoints, 0u,
                                       static_cast<std::uint32_t>(strandJoints.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.strandDistanceConstraintsBuffer,
               strandDistanceConstraints, 0u,
               static_cast<std::uint32_t>(strandDistanceConstraints.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.strandSegmentStatesBuffer,
               strandSegmentStates, 0u, static_cast<std::uint32_t>(strandSegmentStates.size())) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftTopology.particleEdgeRangesBuffer,
                                       particleEdgeRanges, 0u, particleCount) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.particleIncidentEdgesBuffer, incidentEdges,
               0u, static_cast<std::uint32_t>(incidentEdges.size())) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftTopology.particleBendRangesBuffer,
                                       particleBendRanges, 0u, particleCount) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.particleIncidentBendsBuffer, incidentBends,
               0u, static_cast<std::uint32_t>(incidentBends.size())) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftTopology.particleTetRangesBuffer,
                                       particleTetRanges, 0u, particleCount) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.particleIncidentTetsBuffer, incidentTets, 0u,
               static_cast<std::uint32_t>(incidentTets.size())) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftTopology.particleStrandSegmentRangesBuffer,
                                       particleStrandSegmentRanges, 0u, particleCount) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftTopology.particleIncidentStrandSegmentsBuffer,
                                       incidentStrandSegments, 0u,
                                       static_cast<std::uint32_t>(incidentStrandSegments.size())) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftTopology.particleStrandJointRangesBuffer,
                                       particleStrandJointRanges, 0u, particleCount) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.particleIncidentStrandJointsBuffer,
               incidentStrandJoints, 0u, static_cast<std::uint32_t>(incidentStrandJoints.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.segmentStrandJointRangesBuffer,
               segmentStrandJointRanges, 0u,
               static_cast<std::uint32_t>(segmentStrandJointRanges.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.segmentIncidentStrandJointsBuffer,
               segmentIncidentStrandJoints, 0u,
               static_cast<std::uint32_t>(segmentIncidentStrandJoints.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.segmentStrandRigidAttachmentRangesBuffer,
               segmentStrandAttachmentRanges, 0u,
               static_cast<std::uint32_t>(segmentStrandAttachmentRanges.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.segmentIncidentStrandRigidAttachmentsBuffer,
               segmentIncidentStrandAttachments, 0u,
               static_cast<std::uint32_t>(segmentIncidentStrandAttachments.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.renderVertexTriangleRangesBuffer,
               renderVertexTriangleRanges, 0u,
               static_cast<std::uint32_t>(renderVertexTriangleRanges.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.renderVertexTriangleIndicesBuffer,
               softRenderData.vertexTriangleIndices, 0u,
               static_cast<std::uint32_t>(softRenderData.vertexTriangleIndices.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.renderVertexBindingsBuffer,
               softRenderData.vertexBindings, 0u,
               static_cast<std::uint32_t>(softRenderData.vertexBindings.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.renderTriangleParticleIndicesBuffer,
               softRenderData.triangleParticleIndices, 0u,
               static_cast<std::uint32_t>(softRenderData.triangleParticleIndices.size())) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftTopology.softBodyParticleRangesBuffer,
                                       softBodyParticleRanges, 0u,
                                       static_cast<std::uint32_t>(softBodyParticleRanges.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.softBodyChunkRangesBuffer,
               softBodyChunkRanges, 0u, static_cast<std::uint32_t>(softBodyChunkRanges.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.softBodyBoundsChunksBuffer,
               softBodyBoundsChunks, 0u, static_cast<std::uint32_t>(softBodyBoundsChunks.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.softBodyFallbackNormalsBuffer,
               softRenderData.fallbackNormals, 0u,
               static_cast<std::uint32_t>(softRenderData.fallbackNormals.size())) &&
           updateStructuredBufferRange(computeContext,
                                       mPersistentSoftTopology.softBodyRenderPositionsBuffer,
                                       initialRenderPositions, 0u,
                                       static_cast<std::uint32_t>(initialRenderPositions.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentSoftTopology.softBodyRenderNormalsBuffer,
               softRenderData.fallbackNormals, 0u,
               static_cast<std::uint32_t>(softRenderData.fallbackNormals.size()));
}

bool PhysicsSceneGpuState::uploadRoutedCableTopology(
    Diligent::IDeviceContext *computeContext, const std::vector<RigidBodyState> &rigidBodies,
    const std::vector<RoutedCableConstraint> &constraints,
    const std::vector<RoutedCableRoutePoint> &routePoints)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    std::vector<GpuRoutedCableConstraint> descriptors(constraints.size());
    for (std::size_t i = 0; i < constraints.size(); ++i)
    {
        const RoutedCableConstraint &src = constraints[i];
        descriptors[i]                   = GpuRoutedCableConstraint{
            src.routePointStart, src.routePointCount, src.targetLength, src.compliance,
            src.tensionOnly,     src.reserved0,       src.reserved1,    src.reserved2};
    }

    std::vector<GpuRoutedCableRoutePoint> gpuRoutePoints(routePoints.size());
    for (std::size_t i = 0; i < routePoints.size(); ++i)
    {
        const RoutedCableRoutePoint &src = routePoints[i];
        gpuRoutePoints[i]                = GpuRoutedCableRoutePoint{
            src.rigidBodyIndex, src.reserved0, src.reserved1, src.reserved2, src.localGuideOffset};
    }

    std::vector<GpuRoutedCableDebugSegment> debugSegments;
    for (std::size_t cableIndex = 0; cableIndex < constraints.size(); ++cableIndex)
    {
        const RoutedCableConstraint &constraint = constraints[cableIndex];
        if (constraint.routePointCount < 2u)
        {
            continue;
        }

        const std::uint32_t firstRoutePointIndex = constraint.routePointStart;
        if (firstRoutePointIndex >= routePoints.size())
        {
            continue;
        }

        const std::uint32_t rigidBodyIndex = routePoints[firstRoutePointIndex].rigidBodyIndex;
        const std::uint32_t envIndex =
            rigidBodyIndex < rigidBodies.size() ? rigidBodies[rigidBodyIndex].environmentIndex : 0u;
        for (std::uint32_t routeOffset = 0u; routeOffset + 1u < constraint.routePointCount;
             ++routeOffset)
        {
            debugSegments.push_back(
                GpuRoutedCableDebugSegment{constraint.routePointStart + routeOffset,
                                           constraint.routePointStart + routeOffset + 1u, envIndex,
                                           static_cast<std::uint32_t>(cableIndex)});
        }
    }

    return updateStructuredBufferRange(computeContext, mPersistentRoutedCables.descriptorsBuffer,
                                       descriptors, 0u,
                                       static_cast<std::uint32_t>(descriptors.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentRoutedCables.routePointsBuffer,
                                       gpuRoutePoints, 0u,
                                       static_cast<std::uint32_t>(gpuRoutePoints.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentRoutedCables.debugSegmentsBuffer,
                                       debugSegments, 0u,
                                       static_cast<std::uint32_t>(debugSegments.size()));
}

bool PhysicsSceneGpuState::uploadRigidDistanceConstraints(
    Diligent::IDeviceContext *computeContext,
    const std::vector<RigidDistanceConstraint> &constraints)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    std::vector<GpuRigidDistanceConstraint> gpuConstraints(constraints.size());
    for (std::size_t i = 0; i < constraints.size(); ++i)
    {
        const RigidDistanceConstraint &constraint = constraints[i];
        gpuConstraints[i]                         = GpuRigidDistanceConstraint{
            constraint.rigidBodyIndexA, constraint.rigidBodyIndexB, constraint.restDistance,
            constraint.compliance,      constraint.localAnchorA,    constraint.localAnchorB,
        };
    }

    return updateStructuredBufferRange(
        computeContext, mPersistentRoutedCables.rigidDistanceConstraintsBuffer, gpuConstraints, 0u,
        static_cast<std::uint32_t>(gpuConstraints.size()));
}

bool PhysicsSceneGpuState::uploadRigidParticleAttachments(
    Diligent::IDeviceContext *computeContext,
    const std::vector<RigidParticleAttachmentConstraint> &constraints)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    std::vector<GpuRigidParticleAttachmentConstraint> gpuConstraints(constraints.size());
    for (std::size_t i = 0; i < constraints.size(); ++i)
    {
        const RigidParticleAttachmentConstraint &constraint = constraints[i];
        gpuConstraints[i]                                   = GpuRigidParticleAttachmentConstraint{
            constraint.particleIndex, constraint.rigidBodyIndex, constraint.compliance,
            constraint.reserved0,     constraint.localAnchor,
        };
    }

    return updateStructuredBufferRange(
        computeContext, mPersistentRoutedCables.rigidParticleAttachmentsBuffer, gpuConstraints, 0u,
        static_cast<std::uint32_t>(gpuConstraints.size()));
}

bool PhysicsSceneGpuState::uploadStrandRigidAttachments(
    Diligent::IDeviceContext *computeContext,
    const std::vector<StrandRigidAttachmentConstraint> &constraints)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    std::vector<GpuStrandRigidAttachmentConstraint> gpuConstraints(constraints.size());
    for (std::size_t i = 0; i < constraints.size(); ++i)
    {
        const StrandRigidAttachmentConstraint &constraint = constraints[i];
        gpuConstraints[i]                                 = GpuStrandRigidAttachmentConstraint{
            constraint.segmentIndex,       constraint.rigidBodyIndex,
            constraint.segmentT,           constraint.translationCompliance,
            constraint.rotationCompliance, constraint.reserved0,
            constraint.reserved1,          constraint.reserved2,
            constraint.localAnchor,        constraint.localRotation,
        };
    }

    return updateStructuredBufferRange(
        computeContext, mPersistentRoutedCables.strandRigidAttachmentsBuffer, gpuConstraints, 0u,
        static_cast<std::uint32_t>(gpuConstraints.size()));
}

bool PhysicsSceneGpuState::uploadCurveRenderData(Diligent::IDeviceContext *computeContext,
                                                 const CurveRenderDataHost &curveRenderData)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    std::vector<GpuCurveRenderDescriptor> descriptors(curveRenderData.descriptors.size());
    for (std::size_t i = 0; i < curveRenderData.descriptors.size(); ++i)
    {
        const CurveRenderDescriptorHost &src = curveRenderData.descriptors[i];
        descriptors[i]                       = GpuCurveRenderDescriptor{
            src.particleIndexStart, src.particleCount,    src.vertexBase, src.vertexCount,
            src.radialResolution,   src.environmentIndex, src.radius};
    }

    std::vector<Diligent::float4> zeroedVertices;
    std::uint32_t totalVertexCount = 0u;
    for (const CurveRenderDescriptorHost &descriptor : curveRenderData.descriptors)
    {
        totalVertexCount =
            std::max(totalVertexCount, descriptor.vertexBase + descriptor.vertexCount);
    }
    zeroedVertices.assign(totalVertexCount, Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<GpuBodyAabb> emptyAabbs(curveRenderData.descriptors.size());
    for (GpuBodyAabb &aabb : emptyAabbs)
    {
        aabb.minBounds = Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f};
        aabb.maxBounds = Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f};
    }

    return updateStructuredBufferRange(computeContext, mPersistentCurveRender.descriptorsBuffer,
                                       descriptors, 0u,
                                       static_cast<std::uint32_t>(descriptors.size())) &&
           updateStructuredBufferRange(
               computeContext, mPersistentCurveRender.particleIndicesBuffer,
               curveRenderData.particleIndices, 0u,
               static_cast<std::uint32_t>(curveRenderData.particleIndices.size())) &&
           updateStructuredBufferRange(computeContext, mPersistentCurveRender.positionsBuffer,
                                       zeroedVertices, 0u, totalVertexCount) &&
           updateStructuredBufferRange(computeContext, mPersistentCurveRender.normalsBuffer,
                                       zeroedVertices, 0u, totalVertexCount) &&
           updateStructuredBufferRange(computeContext, mPersistentCurveRender.worldAabbsBuffer,
                                       emptyAabbs, 0u,
                                       static_cast<std::uint32_t>(emptyAabbs.size()));
}

bool PhysicsSceneGpuState::uploadSuturingState(
    Diligent::IDeviceContext *computeContext, const ParticleSoAHost &particles,
    std::uint32_t particleCount, const std::vector<std::uint32_t> &suturingParticleIndices,
    const std::vector<StrandSoftSuturingPair> &pairs, std::uint32_t pathHeaderCount,
    std::uint32_t pathNodeCount)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    std::vector<GpuSuturingPair> gpuPairs(pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i)
    {
        const StrandSoftSuturingPair &pair = pairs[i];
        gpuPairs[i] = GpuSuturingPair{pair.suturingGroupId,     pair.softBodyIndex,
                                      pair.strandParticleStart, pair.strandParticleCount,
                                      pair.tipParticleIndex,    pair.softTetStart,
                                      pair.softTetCount,        pair.pathStart,
                                      pair.pathCount,           pair.nodeStart,
                                      pair.nodeCount,           pair.activePathIndex,
                                      pair.environmentIndex,    pair.pathNodeSpacing,
                                      pair.reserved0,           pair.reserved1};
    }

    std::vector<GpuSuturingInsertionStateStorage> insertionStates(particleCount);
    for (GpuSuturingInsertionStateStorage &state : insertionStates)
    {
        state.state            = static_cast<std::uint32_t>(SuturingInsertionState::Outside);
        state.softBodyIndex    = kInvalidSuturingIndex;
        state.tetIndex         = kInvalidSuturingIndex;
        state.pathIndex        = kInvalidSuturingIndex;
        state.nearestNodeIndex = kInvalidSuturingIndex;
    }

    std::vector<GpuSuturingPathHeader> pathHeaders(pathHeaderCount);
    for (GpuSuturingPathHeader &header : pathHeaders)
    {
        header.suturingGroupId = kInvalidSuturingIndex;
        header.softBodyIndex   = kInvalidSuturingIndex;
    }

    std::vector<GpuSuturingPathNode> pathNodes(pathNodeCount);
    for (GpuSuturingPathNode &node : pathNodes)
    {
        node.softBodyIndex = kInvalidSuturingIndex;
        node.tetIndex      = kInvalidSuturingIndex;
    }

    return updateStructuredBufferRange(computeContext, mPersistentSuturing.pairsBuffer, gpuPairs,
                                       0u, static_cast<std::uint32_t>(gpuPairs.size())) &&
           [&]()
    {
        std::vector<Diligent::uint4> suturingParticleRefs(suturingParticleIndices.size());
        for (std::size_t i = 0; i < suturingParticleIndices.size(); ++i)
        {
            const std::uint32_t particleIndex = suturingParticleIndices[i];
            const std::uint32_t role = particleIndex < particles.strandRoles.size()
                                           ? particles.strandRoles[particleIndex]
                                           : static_cast<std::uint32_t>(ParticleStrandRole::None);
            const std::uint32_t ownerType = particleIndex < particles.ownerTypes.size()
                                                ? particles.ownerTypes[particleIndex]
                                                : 0u;
            const std::uint32_t strandId  = particleIndex < particles.strandIds.size()
                                                ? particles.strandIds[particleIndex]
                                                : kInvalidSuturingIndex;
            const std::uint32_t environmentIndex =
                particleIndex < particles.environmentIndices.size()
                    ? particles.environmentIndices[particleIndex]
                    : 0u;
            suturingParticleRefs[i] = Diligent::uint4{particleIndex, role | (ownerType << 16u),
                                                      strandId, environmentIndex};
        }
        return updateStructuredBufferRange(computeContext, mPersistentSuturing.particleRefsBuffer,
                                           suturingParticleRefs, 0u,
                                           static_cast<std::uint32_t>(suturingParticleRefs.size()));
    }() &&
           updateStructuredBufferRange(computeContext, mPersistentSuturing.insertionStatesBuffer,
                                       insertionStates, 0u, particleCount) &&
           updateStructuredBufferRange(computeContext, mPersistentSuturing.pathHeadersBuffer,
                                       pathHeaders, 0u, pathHeaderCount) &&
           updateStructuredBufferRange(computeContext, mPersistentSuturing.pathNodesBuffer,
                                       pathNodes, 0u, pathNodeCount);
}

bool PhysicsSceneGpuState::copyPredictedRigidBodiesToPersistentState(
    Diligent::IDeviceContext *computeContext, std::uint32_t bodyCount)
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

bool PhysicsSceneGpuState::readbackProxyRigidContactMetaBlocking(
    Diligent::IDeviceContext *computeContext, GpuProxyRigidContactMeta &outMeta)
{
    if (computeContext == nullptr || mTransientState.proxyRigidContactMetaBuffer == nullptr ||
        mReadbackRigidBodies.proxyRigidContactMetaBuffer == nullptr)
    {
        return false;
    }

    computeContext->CopyBuffer(mTransientState.proxyRigidContactMetaBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackRigidBodies.proxyRigidContactMetaBuffer, 0u,
                               sizeof(GpuProxyRigidContactMeta),
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->Flush();
    computeContext->WaitForIdle();

    void *mappedMeta = nullptr;
    computeContext->MapBuffer(mReadbackRigidBodies.proxyRigidContactMetaBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedMeta);
    if (mappedMeta == nullptr)
    {
        return false;
    }

    outMeta = *static_cast<const GpuProxyRigidContactMeta *>(mappedMeta);
    computeContext->UnmapBuffer(mReadbackRigidBodies.proxyRigidContactMetaBuffer,
                                Diligent::MAP_READ);
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
        (void)world.syncRigidBodyStateFromSimulation(i, positions[i], orientations[i],
                                                     linearVelocities[i], angularVelocities[i]);
    }
    world.finalizeRigidBodyWriteback();

    computeContext->UnmapBuffer(mReadbackRigidBodies.positionsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackRigidBodies.orientationsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackRigidBodies.linearVelocitiesBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackRigidBodies.angularVelocitiesBuffer, Diligent::MAP_READ);
    return true;
}

bool PhysicsSceneGpuState::readbackPredictedParticleStateBlocking(
    Diligent::IDeviceContext *computeContext, PhysicsWorld &world, std::uint32_t particleCount)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (particleCount == 0u)
    {
        return true;
    }

    const Diligent::Uint64 bytes =
        static_cast<Diligent::Uint64>(particleCount) * sizeof(Diligent::float4);
    computeContext->CopyBuffer(mPersistentParticles.positionsInvMassBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackParticles.positionsBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mPersistentParticles.previousPositionsBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackParticles.previousPositionsBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(mPersistentParticles.velocitiesBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               mReadbackParticles.velocitiesBuffer, 0u, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    computeContext->Flush();
    computeContext->WaitForIdle();

    void *mappedPositions  = nullptr;
    void *mappedPrevious   = nullptr;
    void *mappedVelocities = nullptr;
    computeContext->MapBuffer(mReadbackParticles.positionsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedPositions);
    computeContext->MapBuffer(mReadbackParticles.previousPositionsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedPrevious);
    computeContext->MapBuffer(mReadbackParticles.velocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedVelocities);

    if (mappedPositions == nullptr || mappedPrevious == nullptr || mappedVelocities == nullptr)
    {
        if (mappedPositions != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackParticles.positionsBuffer, Diligent::MAP_READ);
        }
        if (mappedPrevious != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackParticles.previousPositionsBuffer,
                                        Diligent::MAP_READ);
        }
        if (mappedVelocities != nullptr)
        {
            computeContext->UnmapBuffer(mReadbackParticles.velocitiesBuffer, Diligent::MAP_READ);
        }
        return false;
    }

    const auto *positions         = static_cast<const Diligent::float4 *>(mappedPositions);
    const auto *previousPositions = static_cast<const Diligent::float4 *>(mappedPrevious);
    const auto *velocities        = static_cast<const Diligent::float4 *>(mappedVelocities);
    for (std::uint32_t i = 0; i < particleCount; ++i)
    {
        (void)world.syncParticleStateFromSimulation(i, positions[i], previousPositions[i],
                                                    velocities[i]);
    }
    world.finalizeParticleWriteback();

    computeContext->UnmapBuffer(mReadbackParticles.positionsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackParticles.previousPositionsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(mReadbackParticles.velocitiesBuffer, Diligent::MAP_READ);
    return true;
}

bool PhysicsSceneGpuState::readbackSoftNeighborMetaBlocking(
    Diligent::IDeviceContext *computeContext, GpuParticleNeighborMeta &outMeta)
{
    if (computeContext == nullptr || mTransientState.softNeighborMetaBuffer == nullptr ||
        mReadbackParticles.neighborMetaBuffer == nullptr)
    {
        return false;
    }

    computeContext->CopyBuffer(
        mTransientState.softNeighborMetaBuffer, 0u,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION, mReadbackParticles.neighborMetaBuffer,
        0u, sizeof(GpuParticleNeighborMeta), Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->Flush();
    computeContext->WaitForIdle();

    void *mappedMeta = nullptr;
    computeContext->MapBuffer(mReadbackParticles.neighborMetaBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedMeta);
    if (mappedMeta == nullptr)
    {
        return false;
    }

    outMeta = *static_cast<const GpuParticleNeighborMeta *>(mappedMeta);
    computeContext->UnmapBuffer(mReadbackParticles.neighborMetaBuffer, Diligent::MAP_READ);
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

const PhysicsSceneGpuState::PersistentJointCollisionSuppressionBuffers &PhysicsSceneGpuState::
    persistentJointCollisionSuppression() const noexcept
{
    return mPersistentJointCollisionSuppression;
}

const PhysicsSceneGpuState::PersistentJointBuffers &PhysicsSceneGpuState::persistentJoints()
    const noexcept
{
    return mPersistentJoints;
}

const PhysicsSceneGpuState::PersistentRoutedCableBuffers &PhysicsSceneGpuState::
    persistentRoutedCables() const noexcept
{
    return mPersistentRoutedCables;
}

const PhysicsSceneGpuState::PersistentParticleBuffers &PhysicsSceneGpuState::persistentParticles()
    const noexcept
{
    return mPersistentParticles;
}

const gpu::SharedExportBuffer &PhysicsSceneGpuState::softPositionsInvMassSharedBuffer()
    const noexcept
{
    return mSharedSoftPositionsInvMass;
}

const PhysicsSceneGpuState::PersistentSoftTopologyBuffers &PhysicsSceneGpuState::
    persistentSoftTopology() const noexcept
{
    return mPersistentSoftTopology;
}

const PhysicsSceneGpuState::PersistentCurveRenderBuffers &PhysicsSceneGpuState::
    persistentCurveRender() const noexcept
{
    return mPersistentCurveRender;
}

const PhysicsSceneGpuState::SolverTransientBuffers &PhysicsSceneGpuState::transientBuffers()
    const noexcept
{
    return mTransientState;
}

std::uint32_t PhysicsSceneGpuState::ballJointCount() const noexcept
{
    return mBallJointCount;
}

std::uint32_t PhysicsSceneGpuState::sphericalJointCount() const noexcept
{
    return mSphericalJointCount;
}

std::uint32_t PhysicsSceneGpuState::hingeJointCount() const noexcept
{
    return mHingeJointCount;
}

std::uint32_t PhysicsSceneGpuState::sliderJointCount() const noexcept
{
    return mSliderJointCount;
}

std::uint32_t PhysicsSceneGpuState::hingePassiveJointCount() const noexcept
{
    return mHingePassiveJointCount;
}

std::uint32_t PhysicsSceneGpuState::hingePositionDriveJointCount() const noexcept
{
    return mHingePositionDriveJointCount;
}

std::uint32_t PhysicsSceneGpuState::hingeVelocityDriveJointCount() const noexcept
{
    return mHingeVelocityDriveJointCount;
}

std::uint32_t PhysicsSceneGpuState::sliderPassiveJointCount() const noexcept
{
    return mSliderPassiveJointCount;
}

std::uint32_t PhysicsSceneGpuState::sliderPositionDriveJointCount() const noexcept
{
    return mSliderPositionDriveJointCount;
}

std::uint32_t PhysicsSceneGpuState::sliderVelocityDriveJointCount() const noexcept
{
    return mSliderVelocityDriveJointCount;
}

std::uint32_t PhysicsSceneGpuState::candidatePairCapacity() const noexcept
{
    return mCandidatePairCapacity;
}

std::uint32_t PhysicsSceneGpuState::rigidContactCapacity() const noexcept
{
    return mContactCapacity;
}

std::uint32_t PhysicsSceneGpuState::particleCandidatePairCapacity() const noexcept
{
    return mSoftCandidatePairCapacity;
}

std::uint32_t PhysicsSceneGpuState::fluidBoundaryCandidatePairCapacity() const noexcept
{
    return mFluidBoundaryCandidatePairCapacity;
}

std::uint32_t PhysicsSceneGpuState::maxFluidNeighborhood() const noexcept
{
    return mMaxFluidNeighborhood;
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

std::uint64_t PhysicsSceneGpuState::rigidBindingGeneration() const noexcept
{
    return mRigidBindingGeneration;
}

std::uint64_t PhysicsSceneGpuState::softBindingGeneration() const noexcept
{
    return mSoftBindingGeneration;
}

PhysicsGpuSceneView PhysicsSceneGpuState::sceneView() const noexcept
{
    PhysicsGpuSceneView view{};
    view.rigid.poses.positionsBuffer    = mTransientState.predictedRigidBodies.positionsBuffer;
    view.rigid.poses.orientationsBuffer = mTransientState.predictedRigidBodies.orientationsBuffer;
    view.rigid.poses.scalesBuffer       = mPersistentRigidBodies.scalesBuffer;
    view.rigid.poses.count              = mRigidBodyCount;
    view.rigid.poses.bindingGeneration  = mRigidBindingGeneration;
    view.rigid.statePositionsBuffer     = mPersistentRigidBodies.positionsBuffer;
    view.rigid.stateOrientationsBuffer  = mPersistentRigidBodies.orientationsBuffer;
    view.rigid.stateLinearVelocitiesBuffer  = mPersistentRigidBodies.linearVelocitiesBuffer;
    view.rigid.stateAngularVelocitiesBuffer = mPersistentRigidBodies.angularVelocitiesBuffer;
    view.rigid.kinematicTargetPositionsBuffer =
        mPersistentRigidBodies.kinematicTargetPositionsBuffer;
    view.rigid.kinematicTargetOrientationsBuffer =
        mPersistentRigidBodies.kinematicTargetOrientationsBuffer;
    view.rigid.kinematicTargetFlagsBuffer = mPersistentRigidBodies.kinematicTargetFlagsBuffer;
    view.rigid.bodyCount                  = mRigidBodyCount;
    view.rigid.rigidParticleAttachmentsBuffer =
        mPersistentRoutedCables.rigidParticleAttachmentsBuffer;
    view.rigid.rigidParticleAttachmentCount = mRigidParticleAttachmentCount;
    view.rigid.rigidDistanceConstraintsBuffer =
        mPersistentRoutedCables.rigidDistanceConstraintsBuffer;
    view.rigid.rigidDistanceConstraintCount    = mRigidDistanceConstraintCount;
    view.rigid.routedCableDescriptorsBuffer    = mPersistentRoutedCables.descriptorsBuffer;
    view.rigid.routedCableRoutePointsBuffer    = mPersistentRoutedCables.routePointsBuffer;
    view.rigid.routedCableDebugSegmentsBuffer  = mPersistentRoutedCables.debugSegmentsBuffer;
    view.rigid.routedCableCount                = mRoutedCableCount;
    view.rigid.routedCableDebugSegmentCount    = mRoutedCableDebugSegmentCount;
    view.rigid.colliderCount                   = mColliderCount;
    view.rigid.bindingGeneration               = mRigidBindingGeneration;
    view.joints.hingeJointsBuffer              = mPersistentJoints.hingeJointsBuffer;
    view.joints.sliderJointsBuffer             = mPersistentJoints.sliderJointsBuffer;
    view.joints.hingePassiveJointIndicesBuffer = mPersistentJoints.hingePassiveJointIndicesBuffer;
    view.joints.hingePositionDriveJointIndicesBuffer =
        mPersistentJoints.hingePositionDriveJointIndicesBuffer;
    view.joints.hingeVelocityDriveJointIndicesBuffer =
        mPersistentJoints.hingeVelocityDriveJointIndicesBuffer;
    view.joints.sliderPassiveJointIndicesBuffer = mPersistentJoints.sliderPassiveJointIndicesBuffer;
    view.joints.sliderPositionDriveJointIndicesBuffer =
        mPersistentJoints.sliderPositionDriveJointIndicesBuffer;
    view.joints.sliderVelocityDriveJointIndicesBuffer =
        mPersistentJoints.sliderVelocityDriveJointIndicesBuffer;
    view.joints.hingeJointCount               = mHingeJointCount;
    view.joints.sliderJointCount              = mSliderJointCount;
    view.joints.hingePassiveJointCount        = mHingePassiveJointCount;
    view.joints.hingePositionDriveJointCount  = mHingePositionDriveJointCount;
    view.joints.hingeVelocityDriveJointCount  = mHingeVelocityDriveJointCount;
    view.joints.sliderPassiveJointCount       = mSliderPassiveJointCount;
    view.joints.sliderPositionDriveJointCount = mSliderPositionDriveJointCount;
    view.joints.sliderVelocityDriveJointCount = mSliderVelocityDriveJointCount;
    view.joints.bindingGeneration =
        std::max(mLastUploadedRigidJointSceneRevision, mLastUploadedRigidJointModeRevision);
    view.joints.modeBindingGeneration            = mLastUploadedRigidJointModeRevision;
    view.soft.particles.positionsInvMassBuffer   = mPersistentParticles.positionsInvMassBuffer;
    view.soft.particles.previousPositionsBuffer  = mPersistentParticles.previousPositionsBuffer;
    view.soft.particles.velocitiesBuffer         = mPersistentParticles.velocitiesBuffer;
    view.soft.particles.radiiBuffer              = mPersistentParticles.radiiBuffer;
    view.soft.particles.environmentIndicesBuffer = mPersistentParticles.environmentIndicesBuffer;
    view.soft.particles.particleKindsBuffer      = mPersistentParticles.particleKindsBuffer;
    view.soft.particles.ownerTypesBuffer         = mPersistentParticles.ownerTypesBuffer;
    view.soft.particles.ownerIndicesBuffer       = mPersistentParticles.ownerIndicesBuffer;
    view.soft.particles.strandIdsBuffer          = mPersistentParticles.strandIdsBuffer;
    view.soft.particles.strandRolesBuffer        = mPersistentParticles.strandRolesBuffer;
    view.soft.particles.suturingNeighborLinksBuffer =
        mPersistentParticles.suturingNeighborLinksBuffer;
    view.soft.particles.owningSoftBodyIndicesBuffer =
        mPersistentParticles.owningSoftBodyIndicesBuffer;
    view.soft.particles.particleMaterialIndicesBuffer =
        mPersistentParticles.particleMaterialIndicesBuffer;
    view.soft.particles.fluidMaterialIndicesBuffer =
        mPersistentParticles.fluidMaterialIndicesBuffer;
    view.soft.particles.fluidVisualsBuffer = mPersistentParticles.fluidVisualsBuffer;
    view.soft.particles.particleContactMaterialsBuffer =
        mPersistentParticles.particleContactMaterialsBuffer;
    view.soft.particles.fluidMaterialsBuffer   = mPersistentParticles.fluidMaterialsBuffer;
    view.soft.particles.phasesBuffer           = mPersistentParticles.phasesBuffer;
    view.soft.particles.collisionLayersBuffer  = mPersistentParticles.collisionLayersBuffer;
    view.soft.particles.collisionMasksBuffer   = mPersistentParticles.collisionMasksBuffer;
    view.soft.particles.adjacencyOffsetsBuffer = mPersistentParticles.adjacencyOffsetsBuffer;
    view.soft.particles.adjacencyCountsBuffer  = mPersistentParticles.adjacencyCountsBuffer;
    view.soft.particles.adjacencyIndicesBuffer = mPersistentParticles.adjacencyIndicesBuffer;
    view.soft.particles.fluidSurfaceNormalConstraintsBuffer =
        mTransientState.fluidSurfaceNormalConstraintsBuffer;
    view.soft.particles.fluidAnisotropy1Buffer = mTransientState.fluidAnisotropy1Buffer;
    view.soft.particles.fluidAnisotropy2Buffer = mTransientState.fluidAnisotropy2Buffer;
    view.soft.particles.fluidAnisotropy3Buffer = mTransientState.fluidAnisotropy3Buffer;
    view.soft.particles.count                  = mSoftParticleCount;
    view.soft.particles.fluidVisualCount       = mFluidCount;
    view.soft.particles.contactMaterialCount   = mParticleContactMaterialCount;
    view.soft.particles.fluidMaterialCount     = mFluidMaterialCount;
    view.soft.edgesBuffer                      = mPersistentSoftTopology.edgesBuffer;
    view.soft.bendsBuffer                      = mPersistentSoftTopology.bendsBuffer;
    view.soft.tetsBuffer                       = mPersistentSoftTopology.tetsBuffer;
    view.soft.strandSegmentsBuffer             = mPersistentSoftTopology.strandSegmentsBuffer;
    view.soft.strandJointsBuffer               = mPersistentSoftTopology.strandJointsBuffer;
    view.soft.strandDistanceConstraintsBuffer =
        mPersistentSoftTopology.strandDistanceConstraintsBuffer;
    view.soft.strandSegmentStatesBuffer = mPersistentSoftTopology.strandSegmentStatesBuffer;
    view.soft.segmentStrandJointRangesBuffer =
        mPersistentSoftTopology.segmentStrandJointRangesBuffer;
    view.soft.segmentIncidentStrandJointsBuffer =
        mPersistentSoftTopology.segmentIncidentStrandJointsBuffer;
    view.soft.suturingPairsBuffer           = mPersistentSuturing.pairsBuffer;
    view.soft.suturingParticleRefsBuffer    = mPersistentSuturing.particleRefsBuffer;
    view.soft.suturingInsertionStatesBuffer = mPersistentSuturing.insertionStatesBuffer;
    view.soft.suturingPathHeadersBuffer     = mPersistentSuturing.pathHeadersBuffer;
    view.soft.suturingPathNodesBuffer       = mPersistentSuturing.pathNodesBuffer;
    view.soft.renderPositionsBuffer         = mPersistentSoftTopology.softBodyRenderPositionsBuffer;
    view.soft.renderNormalsBuffer           = mPersistentSoftTopology.softBodyRenderNormalsBuffer;
    view.soft.worldAabbsBuffer              = mPersistentSoftTopology.softBodyWorldAabbsBuffer;
    view.soft.softBodyCount                 = mSoftBodyCount;
    view.soft.edgeCount                     = mSoftEdgeCount;
    view.soft.bendCount                     = mSoftBendCount;
    view.soft.tetCount                      = mSoftTetCount;
    view.soft.strandSegmentCount            = mStrandSegmentCount;
    view.soft.strandJointCount              = mStrandJointCount;
    view.soft.strandDistanceCount           = mStrandDistanceCount;
    view.soft.suturingPairCount             = mSuturingPairCount;
    view.soft.suturingPathHeaderCount       = mSuturingPathHeaderCount;
    view.soft.suturingPathNodeCount         = mSuturingPathNodeCount;
    view.soft.bindingGeneration             = mSoftBindingGeneration;
    view.curve.descriptorsBuffer            = mPersistentCurveRender.descriptorsBuffer;
    view.curve.particleIndicesBuffer        = mPersistentCurveRender.particleIndicesBuffer;
    view.curve.positionsBuffer              = mPersistentCurveRender.positionsBuffer;
    view.curve.normalsBuffer                = mPersistentCurveRender.normalsBuffer;
    view.curve.worldAabbsBuffer             = mPersistentCurveRender.worldAabbsBuffer;
    view.curve.curveCount                   = mCurveRenderCount;
    view.curve.bindingGeneration            = mSoftBindingGeneration;
    return view;
}

} // namespace cressim::neo::physics
