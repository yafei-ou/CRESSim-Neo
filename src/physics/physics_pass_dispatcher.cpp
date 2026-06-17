#include "physics/physics_pass_dispatcher.h"
#include "physics/physics_pass_definitions.h"

#include "gpu/gpu_types.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"
#include "common/logger.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace cressim::neo::physics
{

namespace
{

constexpr std::uint32_t kComputeThreadGroupSize = 64u;
constexpr std::uint32_t kNarrowPhaseChunkSize   = 128u;
constexpr std::uint32_t kMaxPreparedScanLevels  = 8u;

constexpr std::size_t kDefaultVariant = 0u;
constexpr std::size_t kAltVariant     = 1u;

std::uint32_t dispatchGroupCount(std::uint32_t threadCount)
{
    return (threadCount + kComputeThreadGroupSize - 1u) / kComputeThreadGroupSize;
}

Diligent::Uint64 indirectArgsOffset(GpuPhysicsIndirectDispatchSlot slot)
{
    return static_cast<Diligent::Uint64>(static_cast<std::uint32_t>(slot)) *
           sizeof(GpuDispatchIndirectArgs);
}

Diligent::Uint64 paddedIndirectArgsOffset(std::uint32_t slot)
{
    return static_cast<Diligent::Uint64>(slot) * sizeof(GpuPaddedDispatchIndirectArgs);
}

GpuPhysicsRadixConstants makeRadixConstants(std::uint32_t elementCount, std::uint32_t bitIndex)
{
    GpuPhysicsRadixConstants result{};
    result.elementCount = elementCount;
    result.bitIndex     = bitIndex;
    return result;
}

GpuBroadPhaseBuildConstants makeBroadPhaseBuildConstants(std::uint32_t elementCount)
{
    GpuBroadPhaseBuildConstants result{};
    result.elementCount = elementCount;
    return result;
}

GpuBroadPhaseReductionConstants makeBroadPhaseReductionConstants(std::uint32_t elementCount)
{
    GpuBroadPhaseReductionConstants result{};
    result.elementCount = elementCount;
    return result;
}

} // namespace

bool PhysicsPassDispatcher::writeConstantsBuffer(Diligent::IDeviceContext *computeContext,
                                                 Diligent::IBuffer *buffer, const void *constants,
                                                 std::size_t constantsSize)
{
    if (computeContext == nullptr || buffer == nullptr || constants == nullptr ||
        constantsSize == 0u)
    {
        return false;
    }

    void *mapped = nullptr;
    computeContext->MapBuffer(buffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }

    std::memcpy(mapped, constants, constantsSize);
    computeContext->UnmapBuffer(buffer, Diligent::MAP_WRITE);
    return true;
}

bool PhysicsPassDispatcher::writeRigidDispatchConstants(Diligent::IDeviceContext *computeContext,
                                                        const GpuRigidDispatchConstants &constants)
{
    return writeConstantsBuffer(computeContext, mRigidDispatchConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeRigidJointDispatchConstants(
    Diligent::IDeviceContext *computeContext, const GpuRigidJointDispatchConstants &constants)
{
    return writeConstantsBuffer(computeContext, mRigidJointDispatchConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeParticleDispatchConstants(
    Diligent::IDeviceContext *computeContext, const GpuParticleDispatchConstants &constants)
{
    return writeConstantsBuffer(computeContext, mParticleDispatchConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeSoftRenderDispatchConstants(
    Diligent::IDeviceContext *computeContext, const GpuSoftRenderDispatchConstants &constants)
{
    return writeConstantsBuffer(computeContext, mSoftRenderDispatchConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeCurveRenderDispatchConstants(
    Diligent::IDeviceContext *computeContext, const GpuCurveRenderDispatchConstants &constants)
{
    return writeConstantsBuffer(computeContext, mCurveRenderDispatchConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeScanDispatchConstants(
    Diligent::IDeviceContext *computeContext, const GpuPhysicsScanDispatchConstants &constants)
{
    return writeConstantsBuffer(computeContext, mScanDispatchConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeRadixConstants(Diligent::IDeviceContext *computeContext,
                                                const GpuPhysicsRadixConstants &constants)
{
    return writeConstantsBuffer(computeContext, mRadixConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeBroadPhaseBuildConstants(
    Diligent::IDeviceContext *computeContext, const GpuBroadPhaseBuildConstants &constants)
{
    return writeConstantsBuffer(computeContext, mBroadPhaseBuildConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeBroadPhaseReductionConstants(
    Diligent::IDeviceContext *computeContext, const GpuBroadPhaseReductionConstants &constants)
{
    return writeConstantsBuffer(computeContext, mBroadPhaseReductionConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::initialize(gpu::GpuDevice &device, std::uint32_t physicsContextId)
{
    gpu::GpuComputeBackendContext backendContext{};
    if (!device.tryGetPhysicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        return false;
    }

    mShaderLibrary      = gpu::ShaderLibrary(device.shaderSourceDirectory());
    mPhysicsContextMask = gpu::contextMaskForId(physicsContextId);

    Diligent::IShaderSourceInputStreamFactory *streamFactory = mShaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        CRESSIM_LOG_ERROR("PhysicsPassDispatcher: shader stream factory is null.");
        return false;
    }

    auto initPass = [&](gpu::GpuComputePass &pass, const gpu::GpuComputePassDefinition &definition,
                        std::size_t variantCount = 1u) -> bool
    {
        if (!pass.initialize(device, streamFactory, mPhysicsContextMask, definition))
        {
            return false;
        }
        return pass.createVariants(variantCount);
    };

    using namespace passdefs;

    if (!initPass(mRigidPredictPass, kPredictRigid) || !initPass(mSoftPredictPass, kSoftPredict) ||
        !initPass(mSyncRigidProxyParticlesPass, kSyncRigidProxyParticles) ||
        !initPass(mBuildParticleBroadPhaseEntriesPass, kBuildParticleBroadPhaseEntries) ||
        !initPass(mBuildParticleBroadPhaseKeysPass, kBuildParticleBroadPhaseKeys) ||
        !initPass(mMarkParticleCellRangeStartsPass, kMarkParticleCellRangeStarts) ||
        !initPass(mClearParticleCellRangesPass, kClearParticleCellRanges) ||
        !initPass(mBuildParticleCellRangesPass, kBuildParticleCellRanges) ||
        !initPass(mCountParticleParticleCandidatePairsPass, kCountParticleParticleCandidatePairs) ||
        !initPass(mFinalizeParticleParticleCandidatePairsPass,
                  kFinalizeParticleParticleCandidatePairs) ||
        !initPass(mEmitParticleParticleCandidatePairsPass, kEmitParticleParticleCandidatePairs) ||
        !initPass(mCountParticleRigidCandidatePairsPass, kCountParticleRigidCandidatePairs) ||
        !initPass(mFinalizeParticleRigidCandidatePairsPass, kFinalizeParticleRigidCandidatePairs) ||
        !initPass(mEmitParticleRigidCandidatePairsPass, kEmitParticleRigidCandidatePairs) ||
        !initPass(mCountFluidBoundaryCandidatePairsPass, kCountFluidBoundaryCandidatePairs) ||
        !initPass(mFinalizeFluidBoundaryCandidatePairsPass, kFinalizeFluidBoundaryCandidatePairs) ||
        !initPass(mEmitFluidBoundaryCandidatePairsPass, kEmitFluidBoundaryCandidatePairs) ||
        !initPass(mGenerateParticleExplicitContactsPass, kGenerateParticleExplicitContacts) ||
        !initPass(mGenerateParticleRigidContactsPass, kGenerateParticleRigidContacts) ||
        !initPass(mPrepareExplicitContactScanPass, kPrepareExplicitContactScan) ||
        !initPass(mPrepareRigidContactScanPass, kPrepareRigidContactScan) ||
        !initPass(mPrepareParticleCandidateIndirectArgsPass,
                  kPrepareParticleCandidateIndirectArgs) ||
        !initPass(mPrepareParticleActiveIndirectArgsPass, kPrepareParticleActiveIndirectArgs) ||
        !initPass(mFinalizeActiveParticleExplicitContactsPass,
                  kFinalizeActiveParticleExplicitContacts) ||
        !initPass(mCompactActiveParticleExplicitContactsPass,
                  kCompactActiveParticleExplicitContacts) ||
        !initPass(mFinalizeActiveParticleRigidContactsPass, kFinalizeActiveParticleRigidContacts) ||
        !initPass(mCompactActiveParticleRigidContactsPass, kCompactActiveParticleRigidContacts) ||
        !initPass(mClearSoftConstraintStatePass, kClearSoftConstraintState) ||
        !initPass(mClearRigidParticleAttachmentConstraintStatePass,
                  kClearRigidParticleAttachmentConstraintState) ||
        !initPass(mClearStrandRigidAttachmentConstraintStatePass,
                  kClearStrandRigidAttachmentConstraintState) ||
        !initPass(mClearRigidDistanceConstraintStatePass, kClearRigidDistanceConstraintState) ||
        !initPass(mClearRoutedCableConstraintStatePass, kClearRoutedCableConstraintState) ||
        !initPass(mClearSuturingCandidatesPass, kClearSuturingCandidates) ||
        !initPass(mGatherSuturingCandidatesPass, kGatherSuturingCandidates) ||
        !initPass(mClassifySuturingParticlesPass, kClassifySuturingParticles) ||
        !initPass(mUpdateSuturingTipPathsPass, kUpdateSuturingTipPaths) ||
        !initPass(mAssignSuturingInsideParticlesPass, kAssignSuturingInsideParticles) ||
        !initPass(mSolveSuturingNodePathConstraintsPass, kSolveSuturingNodePathConstraints) ||
        !initPass(mSolveSoftEdgeConstraintsPass, kSolveSoftEdgeConstraints) ||
        !initPass(mSolveSoftBendConstraintsPass, kSolveSoftBendConstraints) ||
        !initPass(mSolveSoftTetConstraintsPass, kSolveSoftTetConstraints) ||
        !initPass(mApplySoftEdgeCorrectionsPass, kApplySoftEdgeCorrections) ||
        !initPass(mApplySoftBendCorrectionsPass, kApplySoftBendCorrections) ||
        !initPass(mApplySoftTetCorrectionsPass, kApplySoftTetCorrections) ||
        !initPass(mSolveStrandSegmentConstraintsPass, kSolveStrandSegmentConstraints) ||
        !initPass(mApplyStrandSegmentCorrectionsPass, kApplyStrandSegmentCorrections) ||
        !initPass(mSolveStrandJointConstraintsPass, kSolveStrandJointConstraints) ||
        !initPass(mApplyStrandJointCorrectionsPass, kApplyStrandJointCorrections) ||
        !initPass(mApplyStrandRigidAttachmentCorrectionsPass,
                  kApplyStrandRigidAttachmentCorrections) ||
        !initPass(mSolveStrandDistanceConstraintsPass, kSolveStrandDistanceConstraints) ||
        !initPass(mApplyStrandDistanceCorrectionsPass, kApplyStrandDistanceCorrections) ||
        !initPass(mSolveParticleExplicitContactsPass, kSolveParticleExplicitContacts) ||
        !initPass(mSolveParticleRigidContactsPass, kSolveParticleRigidContacts) ||
        !initPass(mApplyParticlePositionCorrectionsPass, kApplyParticlePositionCorrections) ||
        !initPass(mUpdateParticleVelocitiesPass, kUpdateParticleVelocities) ||
        !initPass(mBuildFluidNeighborPairsPass, kBuildFluidNeighborPairs) ||
        !initPass(mComputeFluidDensityConstraintsPass, kComputeFluidDensityConstraints) ||
        !initPass(mComputeFluidDeltaPositionsPass, kComputeFluidDeltaPositions) ||
        !initPass(mApplyFluidDeltaPositionsPass, kApplyFluidDeltaPositions) ||
        !initPass(mClampFluidBoundaryPass, kClampFluidBoundary) ||
        !initPass(mProjectFluidBoundaryVelocitiesPass, kProjectFluidBoundaryVelocities) ||
        !initPass(mComputeFluidVorticityPass, kComputeFluidVorticity) ||
        !initPass(mApplyFluidVorticityConfinementPass, kApplyFluidVorticityConfinement) ||
        !initPass(mBuildFluidRenderAnisotropyPass, kBuildFluidRenderAnisotropy) ||
        !initPass(mSolveParticleContactVelocitiesPass, kSolveParticleContactVelocities) ||
        !initPass(mSolveParticleRigidContactVelocitiesPass, kSolveParticleRigidContactVelocities) ||
        !initPass(mApplyParticleContactVelocitiesPass, kApplyParticleContactVelocities) ||
        !initPass(mSkinSoftRenderVerticesPass, kSkinSoftRenderVertices) ||
        !initPass(mUpdateSoftTriangleNormalsPass, kUpdateSoftTriangleNormals) ||
        !initPass(mUpdateSoftRenderNormalsPass, kUpdateSoftRenderNormals) ||
        !initPass(mUpdateCurveRenderDataPass, kUpdateCurveRenderData) ||
        !initPass(mUpdateSoftBodyBoundsPass, kUpdateSoftBodyBounds) ||
        !initPass(mFinalizeSoftBodyBoundsPass, kFinalizeSoftBodyBounds) ||
        !initPass(mUpdateRigidWorldAabbsPass, kUpdateRigidWorldAabbs) ||
        !initPass(mScanBlockPass, kScanBlock) || !initPass(mScanAddOffsetsPass, kScanAddOffsets) ||
        !initPass(mCompactBodySetPass, kCompactBodySet, 2u) ||
        !initPass(mFinalizeActiveBodiesPass, kFinalizeActiveBodies) ||
        !initPass(mBuildBroadPhaseElementsPass, kBuildBroadPhaseElements, 2u) ||
        !initPass(mReduceExtentElementsPass, kReduceExtentElements) ||
        !initPass(mReduceExtentExtentsPass, kReduceExtentExtents) ||
        !initPass(mMortonCodesPass, kMortonCodes, 2u) ||
        !initPass(mRadixClassifyPass, kRadixClassify) ||
        !initPass(mRadixFinalizePass, kRadixFinalize) ||
        !initPass(mRadixScatterPass, kRadixScatter) ||
        !initPass(mBvhHierarchyPass, kBvhHierarchy, 2u) ||
        !initPass(mBvhBoundingBoxesPass, kBvhBoundingBoxes, 2u) ||
        !initPass(mCountPairsPass, kCountPairs, 2u) ||
        !initPass(mFinalizePairsPass, kFinalizePairs) ||
        !initPass(mEmitPairsPass, kEmitPairs, 2u) ||
        !initPass(mBuildNarrowPhaseChunksPass, kBuildNarrowPhaseChunks) ||
        !initPass(mPrepareRigidIndirectArgsPass, kPrepareRigidIndirectArgs) ||
        !initPass(mGenerateRigidContactsPass, kGenerateRigidContacts) ||
        !initPass(mGenerateProxyRigidContactsPass, kGenerateProxyRigidContacts) ||
        !initPass(mFinalRigidContactDepenetrationPass, kFinalRigidContactDepenetration) ||
        !initPass(mClearRigidBodyPairContactAggregatesPass, kClearRigidBodyPairContactAggregates) ||
        !initPass(mInitRigidContactVelocitiesPass, kInitRigidContactVelocities) ||
        !initPass(mPrepareRigidContactVelocityIndirectArgsPass,
                  kPrepareRigidContactVelocityIndirectArgs) ||
        !initPass(mSolveRigidContactVelocitiesPass, kSolveRigidContactVelocities) ||
        !initPass(mSolveBallJointConstraintsPass, kSolveBallJointConstraints) ||
        !initPass(mSolveHingeJointConstraintsPassivePass, kSolveHingeJointConstraintsPassive) ||
        !initPass(mSolveHingeJointConstraintsTargetPositionPass,
                  kSolveHingeJointConstraintsTargetPosition) ||
        !initPass(mSolveSliderJointConstraintsPassivePass, kSolveSliderJointConstraintsPassive) ||
        !initPass(mSolveSliderJointConstraintsTargetPositionPass,
                  kSolveSliderJointConstraintsTargetPosition) ||
        !initPass(mSolveRigidParticleAttachmentConstraintsPass,
                  kSolveRigidParticleAttachmentConstraints) ||
        !initPass(mSolveStrandRigidAttachmentConstraintsPass,
                  kSolveStrandRigidAttachmentConstraints) ||
        !initPass(mSolveRigidDistanceConstraintsPass, kSolveRigidDistanceConstraints) ||
        !initPass(mSolveRoutedCableConstraintsPass, kSolveRoutedCableConstraints) ||
        !initPass(mClearHingeJointConstraintStatePass, kClearHingeJointConstraintState) ||
        !initPass(mClearSliderJointConstraintStatePass, kClearSliderJointConstraintState) ||
        !initPass(mSolveHingeJointTargetVelocitiesPass, kSolveHingeJointTargetVelocities) ||
        !initPass(mSolveSliderJointTargetVelocitiesPass, kSolveSliderJointTargetVelocities) ||
        !initPass(mClearRigidCorrectionsPass, kClearRigidCorrections) ||
        !initPass(mApplyRigidCorrectionsPass, kApplyRigidCorrections) ||
        !initPass(mUpdateRigidVelocitiesPass, kUpdateRigidVelocities) ||
        !initPass(mApplyRigidContactVelocitiesPass, kApplyRigidContactVelocities))
    {
        CRESSIM_LOG_ERROR("PhysicsPassDispatcher: failed to initialize compute passes.");
        return false;
    }

    auto createConstantsBuffer = [&](const char *name, std::size_t size,
                                     Diligent::RefCntAutoPtr<Diligent::IBuffer> &buffer) -> bool
    {
        Diligent::BufferDesc constantsDesc{};
        constantsDesc.Name                 = name;
        constantsDesc.Size                 = static_cast<Diligent::Uint64>(size);
        constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
        constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
        constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
        constantsDesc.ImmediateContextMask = mPhysicsContextMask;
        backendContext.renderDevice->CreateBuffer(constantsDesc, nullptr, &buffer);
        return buffer != nullptr;
    };

    return createConstantsBuffer("CRESSimNeo.Physics.RigidDispatchConstants",
                                 sizeof(GpuRigidDispatchConstants),
                                 mRigidDispatchConstantsBuffer) &&
               createConstantsBuffer("CRESSimNeo.Physics.RigidJointDispatchConstants",
                                     sizeof(GpuRigidJointDispatchConstants),
                                     mRigidJointDispatchConstantsBuffer) &&
               createConstantsBuffer("CRESSimNeo.Physics.SoftDispatchConstants",
                                     sizeof(GpuParticleDispatchConstants),
                                     mParticleDispatchConstantsBuffer) &&
               createConstantsBuffer("CRESSimNeo.Physics.SoftRenderDispatchConstants",
                                     sizeof(GpuSoftRenderDispatchConstants),
                                     mSoftRenderDispatchConstantsBuffer) &&
               createConstantsBuffer("CRESSimNeo.Physics.CurveRenderDispatchConstants",
                                     sizeof(GpuCurveRenderDispatchConstants),
                                     mCurveRenderDispatchConstantsBuffer) &&
               createConstantsBuffer("CRESSimNeo.Physics.ScanDispatchConstants",
                                     sizeof(GpuPhysicsScanDispatchConstants),
                                     mScanDispatchConstantsBuffer) &&
               [&]() -> bool
    {
        Diligent::BufferDesc scanConstantsDesc{};
        scanConstantsDesc.Name = "CRESSimNeo.Physics.ScanConstants";
        scanConstantsDesc.Size =
            static_cast<Diligent::Uint64>(sizeof(GpuPhysicsScanConstants) * kMaxPreparedScanLevels);
        scanConstantsDesc.Usage = Diligent::USAGE_DEFAULT;
        scanConstantsDesc.BindFlags =
            Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE;
        scanConstantsDesc.Mode                 = Diligent::BUFFER_MODE_STRUCTURED;
        scanConstantsDesc.ElementByteStride    = sizeof(GpuPhysicsScanConstants);
        scanConstantsDesc.ImmediateContextMask = mPhysicsContextMask;
        backendContext.renderDevice->CreateBuffer(scanConstantsDesc, nullptr,
                                                  &mScanConstantsBuffer);
        if (mScanConstantsBuffer == nullptr)
        {
            return false;
        }

        Diligent::BufferDesc scanArgsDesc{};
        scanArgsDesc.Name  = "CRESSimNeo.Physics.ScanIndirectArgs";
        scanArgsDesc.Size  = static_cast<Diligent::Uint64>(sizeof(GpuPaddedDispatchIndirectArgs) *
                                                           kMaxPreparedScanLevels);
        scanArgsDesc.Usage = Diligent::USAGE_DEFAULT;
        scanArgsDesc.BindFlags = Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE |
                                 Diligent::BIND_INDIRECT_DRAW_ARGS;
        scanArgsDesc.Mode      = Diligent::BUFFER_MODE_STRUCTURED;
        scanArgsDesc.ElementByteStride    = sizeof(GpuPaddedDispatchIndirectArgs);
        scanArgsDesc.ImmediateContextMask = mPhysicsContextMask;
        backendContext.renderDevice->CreateBuffer(scanArgsDesc, nullptr, &mScanIndirectArgsBuffer);
        return mScanIndirectArgsBuffer != nullptr;
    }() &&
                            createConstantsBuffer("CRESSimNeo.Physics.RadixConstants",
                                                  sizeof(GpuPhysicsRadixConstants),
                                                  mRadixConstantsBuffer) &&
                            createConstantsBuffer("CRESSimNeo.Physics.BroadPhaseBuildConstants",
                                                  sizeof(GpuBroadPhaseBuildConstants),
                                                  mBroadPhaseBuildConstantsBuffer) &&
                            createConstantsBuffer("CRESSimNeo.Physics.BroadPhaseReductionConstants",
                                                  sizeof(GpuBroadPhaseReductionConstants),
                                                  mBroadPhaseReductionConstantsBuffer);
}

bool PhysicsPassDispatcher::dispatchPreparedScanBlockPass(
    Diligent::IDeviceContext *computeContext, Diligent::IBuffer *input, Diligent::IBuffer *output,
    Diligent::IBuffer *blockSums, Diligent::IBuffer *indirectArgsBuffer,
    std::uint32_t scanLevelIndex, std::uint32_t dispatchElementCount, bool useIndirect)
{
    if (!useIndirect && dispatchElementCount == 0u)
    {
        return true;
    }

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsScanDispatchConstantsBuffer", mScanDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScanConstants", mScanConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScanInput", input, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScanOutput", output, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_BlockSums", blockSums, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const GpuPhysicsScanDispatchConstants scanDispatchConstants{scanLevelIndex, 0u, 0u, 0u};
    return writeScanDispatchConstants(computeContext, scanDispatchConstants) &&
           (useIndirect ? mScanBlockPass.dispatchIndirect(computeContext, kDefaultVariant, bindings,
                                                          indirectArgsBuffer,
                                                          paddedIndirectArgsOffset(scanLevelIndex))
                        : mScanBlockPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                  dispatchGroupCount(dispatchElementCount)));
}

bool PhysicsPassDispatcher::dispatchPreparedScanAddOffsetsPass(
    Diligent::IDeviceContext *computeContext, Diligent::IBuffer *output,
    Diligent::IBuffer *scannedBlockOffsets, Diligent::IBuffer *indirectArgsBuffer,
    std::uint32_t scanLevelIndex, std::uint32_t dispatchElementCount, bool useIndirect)
{
    if (!useIndirect && dispatchElementCount == 0u)
    {
        return true;
    }

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsScanDispatchConstantsBuffer", mScanDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScanConstants", mScanConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScannedBlockOffsets", scannedBlockOffsets,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScanOutput", output, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const GpuPhysicsScanDispatchConstants scanDispatchConstants{scanLevelIndex, 0u, 0u, 0u};
    return writeScanDispatchConstants(computeContext, scanDispatchConstants) &&
           (useIndirect ? mScanAddOffsetsPass.dispatchIndirect(
                              computeContext, kDefaultVariant, bindings, indirectArgsBuffer,
                              paddedIndirectArgsOffset(scanLevelIndex))
                        : mScanAddOffsetsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                       dispatchGroupCount(dispatchElementCount)));
}

bool PhysicsPassDispatcher::dispatchExclusiveScanPrepared(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    Diligent::IBuffer *input, Diligent::IBuffer *output, Diligent::IBuffer *indirectArgsBuffer,
    bool useIndirect, const std::uint32_t *directCounts)
{
    const auto &transientState = sceneState.transientBuffers();
    const std::size_t maxLevels =
        std::min<std::size_t>(kMaxPreparedScanLevels, transientState.scanBlockSumsBuffers.size());
    if (maxLevels == 0u || maxLevels > transientState.scanScannedBlockSumsBuffers.size())
    {
        return false;
    }

    if (!dispatchPreparedScanBlockPass(
            computeContext, input, output, transientState.scanBlockSumsBuffers[0],
            indirectArgsBuffer, 0u, directCounts != nullptr ? directCounts[0] : 1u, useIndirect))
    {
        return false;
    }

    for (std::size_t level = 1u; level < maxLevels; ++level)
    {
        if (!dispatchPreparedScanBlockPass(
                computeContext, transientState.scanBlockSumsBuffers[level - 1u],
                transientState.scanScannedBlockSumsBuffers[level - 1u],
                transientState.scanBlockSumsBuffers[level], indirectArgsBuffer,
                static_cast<std::uint32_t>(level),
                directCounts != nullptr ? directCounts[level] : 1u, useIndirect))
        {
            return false;
        }
    }

    for (std::size_t scanLevelIndex = maxLevels; scanLevelIndex-- > 1u;)
    {
        if (!dispatchPreparedScanAddOffsetsPass(
                computeContext, transientState.scanScannedBlockSumsBuffers[scanLevelIndex - 1u],
                transientState.scanScannedBlockSumsBuffers[scanLevelIndex], indirectArgsBuffer,
                static_cast<std::uint32_t>(scanLevelIndex),
                directCounts != nullptr ? directCounts[scanLevelIndex] : 1u, useIndirect))
        {
            return false;
        }
    }

    return dispatchPreparedScanAddOffsetsPass(
        computeContext, output, transientState.scanScannedBlockSumsBuffers[0], indirectArgsBuffer,
        0u, directCounts != nullptr ? directCounts[0] : 1u, useIndirect);
}

bool PhysicsPassDispatcher::dispatchExclusiveScanWithCpuCount(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    Diligent::IBuffer *input, Diligent::IBuffer *output, std::uint32_t count)
{
    if (count == 0u)
    {
        return true;
    }

    std::array<GpuPhysicsScanConstants, kMaxPreparedScanLevels> scanConstants{};
    std::array<std::uint32_t, kMaxPreparedScanLevels> directCounts{};
    std::uint32_t levelCount = count;
    for (std::size_t level = 0u; level < scanConstants.size(); ++level)
    {
        scanConstants[level].elementCount = levelCount;
        const std::uint32_t groupCount    = levelCount == 0u ? 0u : dispatchGroupCount(levelCount);
        scanConstants[level].hasParentOffsets = groupCount > 1u ? 1u : 0u;
        directCounts[level]                   = levelCount;
        levelCount = scanConstants[level].hasParentOffsets != 0u ? groupCount : 0u;
    }

    computeContext->UpdateBuffer(
        mScanConstantsBuffer, 0u, static_cast<Diligent::Uint32>(sizeof(scanConstants)),
        scanConstants.data(), Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return dispatchExclusiveScanPrepared(computeContext, sceneState, input, output, nullptr, false,
                                         directCounts.data());
}

bool PhysicsPassDispatcher::dispatchExclusiveScanWithGpuCount(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    Diligent::IBuffer *input, Diligent::IBuffer *output, bool particleRigidCandidates)
{
    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScanConstantsRW", mScanConstantsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ScanIndirectArgsRW", mScanIndirectArgsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const bool prepared =
        particleRigidCandidates
            ? mPrepareRigidContactScanPass.dispatch(computeContext, kDefaultVariant, bindings, 1u)
            : mPrepareExplicitContactScanPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                       1u);
    return prepared && dispatchExclusiveScanPrepared(computeContext, sceneState, input, output,
                                                     mScanIndirectArgsBuffer, true, nullptr);
}

bool PhysicsPassDispatcher::clearRigidCorrections(Diligent::IDeviceContext *computeContext,
                                                  PhysicsSceneGpuState &sceneState,
                                                  std::uint32_t bodyCount,
                                                  const GpuRigidDispatchConstants &constants)
{
    if (bodyCount == 0u)
    {
        sceneState.setCorrectionBuffersNeedClear(false);
        return true;
    }

    const auto &transientState = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transientState.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections",
                              transientState.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transientState.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transientState.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!writeRigidDispatchConstants(computeContext, constants) ||
        !mClearRigidCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                             dispatchGroupCount(bodyCount)))
    {
        return false;
    }

    sceneState.setCorrectionBuffersNeedClear(false);
    return true;
}

bool PhysicsPassDispatcher::clearSliderJointConstraintState(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t jointCount)
{
    if (jointCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidJointDispatchConstantsBuffer",
                              mRigidJointDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SliderJointLambdas0123", transient.sliderJointLambdas0123Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SliderJointLambdas45", transient.sliderJointLambdas45Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const GpuRigidJointDispatchConstants jointConstants{jointCount, 0u, 0u, 0u};
    return writeRigidJointDispatchConstants(computeContext, jointConstants) &&
           mClearSliderJointConstraintStatePass.dispatch(computeContext, kDefaultVariant, bindings,
                                                         dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::clearHingeJointConstraintState(Diligent::IDeviceContext *computeContext,
                                                           const PhysicsSceneGpuState &sceneState,
                                                           std::uint32_t jointCount)
{
    if (jointCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidJointDispatchConstantsBuffer",
                              mRigidJointDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_HingeJointLambdas0123", transient.hingeJointLambdas0123Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_HingeJointLambdas45", transient.hingeJointLambdas45Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const GpuRigidJointDispatchConstants jointConstants{jointCount, 0u, 0u, 0u};
    return writeRigidJointDispatchConstants(computeContext, jointConstants) &&
           mClearHingeJointConstraintStatePass.dispatch(computeContext, kDefaultVariant, bindings,
                                                        dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::predictRigid(Diligent::IDeviceContext *computeContext,
                                         const PhysicsSceneGpuState &sceneState,
                                         std::uint32_t bodyCount,
                                         const GpuRigidDispatchConstants &constants)
{
    if (bodyCount == 0u)
    {
        return true;
    }

    const auto &persistent = sceneState.persistentRigidBodies();
    const auto &transient  = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyPositionsInvMass", persistent.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyOrientations", persistent.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocities", persistent.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocities", persistent.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyKinematicTargetPositions",
                              persistent.kinematicTargetPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyKinematicTargetOrientations",
                              persistent.kinematicTargetOrientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyKinematicTargetFlags",
                              persistent.kinematicTargetFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                              transient.previousRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyOrientations",
                              transient.previousRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeRigidDispatchConstants(computeContext, constants) &&
           mRigidPredictPass.dispatch(computeContext, kDefaultVariant, bindings,
                                      dispatchGroupCount(bodyCount));
}

bool PhysicsPassDispatcher::predictSoft(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        std::uint32_t particleCount,
                                        const GpuParticleDispatchConstants &constants)
{
    if (particleCount == 0u)
    {
        return true;
    }

    const auto &particles = sceneState.persistentParticles();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticlePreviousPositions", particles.previousPositionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleVelocities", particles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleKinds", particles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterialIndices", particles.fluidMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterials", particles.fluidMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mSoftPredictPass.dispatch(computeContext, kDefaultVariant, bindings,
                                     dispatchGroupCount(particleCount));
}

bool PhysicsPassDispatcher::syncRigidProxyParticles(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t particleCount,
                                                    const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (particleCount == 0u)
    {
        return true;
    }

    const auto &particles = sceneState.persistentParticles();
    const auto &rigid     = sceneState.persistentRigidBodies();
    const auto &predicted = sceneState.transientBuffers().predictedRigidBodies;
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", particles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerIndices", particles.ownerIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidProxyLocalPositions",
                              particles.rigidProxyLocalPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass", predicted.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations", predicted.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mSyncRigidProxyParticlesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(particleCount));
}

bool PhysicsPassDispatcher::buildParticleBroadPhaseEntries(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t totalParticleLikeCount, const GpuParticleDispatchConstants &constants)
{
    if (totalParticleLikeCount == 0u)
    {
        return true;
    }

    const auto &particles = sceneState.persistentParticles();
    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseEntries",
                              transient.particleBroadPhaseEntriesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mBuildParticleBroadPhaseEntriesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                        dispatchGroupCount(totalParticleLikeCount));
}

bool PhysicsPassDispatcher::buildParticleBroadPhaseKeys(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t totalParticleLikeCount, const GpuParticleDispatchConstants &constants)
{
    if (totalParticleLikeCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseEntries",
                              transient.particleBroadPhaseEntriesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseKeys", transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mBuildParticleBroadPhaseKeysPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                     dispatchGroupCount(totalParticleLikeCount));
}

bool PhysicsPassDispatcher::markParticleCellRangeStarts(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t totalParticleLikeCount, const GpuParticleDispatchConstants &constants)
{
    if (totalParticleLikeCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedParticleBroadPhaseKeys",
                              transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRangeStartFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mMarkParticleCellRangeStartsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                     dispatchGroupCount(totalParticleLikeCount));
}

bool PhysicsPassDispatcher::sortParticleBroadPhase(Diligent::IDeviceContext *computeContext,
                                                   const PhysicsSceneGpuState &sceneState,
                                                   std::uint32_t count)
{
    return dispatchSoftRadixSortPass(computeContext, sceneState, count);
}

bool PhysicsPassDispatcher::clearParticleCellRanges(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t cellRangeCapacity,
                                                    const GpuParticleDispatchConstants &constants)
{
    if (cellRangeCapacity == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRanges", transient.particleCellRangesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mClearParticleCellRangesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(cellRangeCapacity));
}

bool PhysicsPassDispatcher::buildParticleCellRanges(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t totalParticleLikeCount,
                                                    const GpuParticleDispatchConstants &constants)
{
    if (totalParticleLikeCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    if (!markParticleCellRangeStarts(computeContext, sceneState, totalParticleLikeCount,
                                     constants) ||
        !dispatchExclusiveScanWithCpuCount(
            computeContext, sceneState, transient.softRadixBitFlagsBuffer,
            transient.softRadixBitOffsetsBuffer, totalParticleLikeCount))
    {
        return false;
    }

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedParticleBroadPhaseKeys",
                              transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRangeStartFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRangeStartOffsets",
                              transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRanges", transient.particleCellRangesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mBuildParticleCellRangesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(totalParticleLikeCount));
}

bool PhysicsPassDispatcher::clearParticleNeighborMeta(Diligent::IDeviceContext *computeContext,
                                                      const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const GpuParticleNeighborMeta zeroMeta{};
    computeContext->UpdateBuffer(sceneState.transientBuffers().softNeighborMetaBuffer, 0u,
                                 sizeof(GpuParticleNeighborMeta), &zeroMeta,
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return true;
}

bool PhysicsPassDispatcher::buildParticleParticleCandidatePairs(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t particleCount, const GpuParticleDispatchConstants &constants)
{
    if (particleCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const auto &particles = sceneState.persistentParticles();
    const std::array countBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseEntries",
                              transient.particleBroadPhaseEntriesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRanges", transient.particleCellRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedParticleBroadPhaseKeys",
                              transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", particles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseMetadata", particles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", particles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleAdjacencyOffsets", particles.adjacencyOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleAdjacencyCounts", particles.adjacencyCountsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleAdjacencyIndices", particles.adjacencyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array emitBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseEntries",
                              transient.particleBroadPhaseEntriesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRanges", transient.particleCellRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedParticleBroadPhaseKeys",
                              transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", particles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseMetadata", particles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", particles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleAdjacencyOffsets", particles.adjacencyOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleAdjacencyCounts", particles.adjacencyCountsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleAdjacencyIndices", particles.adjacencyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCandidatePairs", transient.softSoftCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mCountParticleParticleCandidatePairsPass.dispatch(
               computeContext, kDefaultVariant, countBindings, dispatchGroupCount(particleCount)) &&
           dispatchExclusiveScanWithCpuCount(computeContext, sceneState,
                                             transient.softRadixBitFlagsBuffer,
                                             transient.softRadixBitOffsetsBuffer, particleCount) &&
           mFinalizeParticleParticleCandidatePairsPass.dispatch(computeContext, kDefaultVariant,
                                                                finalizeBindings, 1u) &&
           mEmitParticleParticleCandidatePairsPass.dispatch(
               computeContext, kDefaultVariant, emitBindings, dispatchGroupCount(particleCount));
}

bool PhysicsPassDispatcher::buildParticleRigidCandidatePairs(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t particleCount, const GpuParticleDispatchConstants &constants)
{
    if (particleCount == 0u)
    {
        return true;
    }

    const auto &transient           = sceneState.transientBuffers();
    const auto &softParticles       = sceneState.persistentParticles();
    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &persistentRigid     = sceneState.persistentRigidBodies();
    const auto &bodyColliderMapping = sceneState.persistentBodyColliderMapping();
    const std::array countBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", softParticles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerIndices", softParticles.ownerIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseMetadata",
                              softParticles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", persistentColliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyColliderRanges", bodyColliderMapping.colliderRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array emitBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", softParticles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerIndices", softParticles.ownerIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseMetadata",
                              softParticles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", persistentColliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyColliderRanges", bodyColliderMapping.colliderRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCandidatePairs", transient.softRigidCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mCountParticleRigidCandidatePairsPass.dispatch(
               computeContext, kDefaultVariant, countBindings, dispatchGroupCount(particleCount)) &&
           dispatchExclusiveScanWithCpuCount(computeContext, sceneState,
                                             transient.softRadixBitFlagsBuffer,
                                             transient.softRadixBitOffsetsBuffer, particleCount) &&
           mFinalizeParticleRigidCandidatePairsPass.dispatch(computeContext, kDefaultVariant,
                                                             finalizeBindings, 1u) &&
           mEmitParticleRigidCandidatePairsPass.dispatch(
               computeContext, kDefaultVariant, emitBindings, dispatchGroupCount(particleCount));
}

bool PhysicsPassDispatcher::buildFluidBoundaryCandidatePairs(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t particleCount, const GpuParticleDispatchConstants &constants)
{
    if (particleCount == 0u)
    {
        return true;
    }

    const auto &transient           = sceneState.transientBuffers();
    const auto &softParticles       = sceneState.persistentParticles();
    const auto &persistentRigid     = sceneState.persistentRigidBodies();
    const auto &persistentColliders = sceneState.persistentColliders();
    const std::array countBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseMetadata",
                              softParticles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterialIndices", softParticles.fluidMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterials", softParticles.fluidMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", persistentColliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.fluidBoundaryCandidateCountsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.fluidBoundaryCandidateCountsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.fluidBoundaryCandidateOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array emitBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseMetadata",
                              softParticles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterialIndices", softParticles.fluidMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterials", softParticles.fluidMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", persistentColliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.fluidBoundaryCandidateCountsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.fluidBoundaryCandidateOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateRanges", transient.fluidBoundaryCandidateRangesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleCandidatePairs",
                              transient.fluidBoundaryCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mCountFluidBoundaryCandidatePairsPass.dispatch(
               computeContext, kDefaultVariant, countBindings, dispatchGroupCount(particleCount)) &&
           dispatchExclusiveScanWithCpuCount(
               computeContext, sceneState, transient.fluidBoundaryCandidateCountsBuffer,
               transient.fluidBoundaryCandidateOffsetsBuffer, particleCount) &&
           mFinalizeFluidBoundaryCandidatePairsPass.dispatch(computeContext, kDefaultVariant,
                                                             finalizeBindings, 1u) &&
           mEmitFluidBoundaryCandidatePairsPass.dispatch(
               computeContext, kDefaultVariant, emitBindings, dispatchGroupCount(particleCount));
}

bool PhysicsPassDispatcher::generateParticleExplicitContacts(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles           = sceneState.persistentParticles();
    const auto &transient               = sceneState.transientBuffers();
    const PhysicsGpuSceneView sceneView = sceneState.sceneView();
    const auto &suturing                = sceneView.soft;
    const std::array bindings{
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", softParticles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleStrandRoles", softParticles.strandRolesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingNeighborLinks", softParticles.suturingNeighborLinksBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwningSoftBodyIndices",
                              softParticles.owningSoftBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleMaterialIndices",
                              softParticles.particleMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleContactMaterials",
                              softParticles.particleContactMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCandidatePairs", transient.softSoftCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingInsertionStates", suturing.suturingInsertionStatesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleContacts", transient.softContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mGenerateParticleExplicitContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftGenerateContacts));
}

bool PhysicsPassDispatcher::generateParticleRigidContacts(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &rigidBodies   = sceneState.persistentRigidBodies();
    const auto &mapping       = sceneState.persistentBodyColliderMapping();
    const auto &colliders     = sceneState.persistentColliders();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseMetadata",
                              softParticles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", softParticles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyScales", rigidBodies.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyColliderRanges", mapping.colliderRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyColliderIndices", mapping.colliderIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderGeometryData", colliders.geometryDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", colliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCandidatePairs", transient.softRigidCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleRigidContacts", transient.softRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mGenerateParticleRigidContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftGenerateRigidContacts));
}

bool PhysicsPassDispatcher::prepareParticleCandidateIndirectArgs(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PhysicsIndirectDispatchArgs", transient.physicsIndirectArgsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return mPrepareParticleCandidateIndirectArgsPass.dispatch(computeContext, kDefaultVariant,
                                                              bindings, 1u);
}

bool PhysicsPassDispatcher::prepareParticleActiveIndirectArgs(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PhysicsIndirectDispatchArgs", transient.physicsIndirectArgsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return mPrepareParticleActiveIndirectArgsPass.dispatch(computeContext, kDefaultVariant,
                                                           bindings, 1u);
}

bool PhysicsPassDispatcher::compactParticleExplicitContacts(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array compactBindings{
        gpu::GpuBufferBinding{"g_ParticleContacts", transient.softContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ActiveSoftContacts", transient.activeSoftContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return dispatchExclusiveScanWithGpuCount(computeContext, sceneState,
                                             transient.softRadixBitFlagsBuffer,
                                             transient.softRadixBitOffsetsBuffer, false) &&
           writeParticleDispatchConstants(computeContext, constants) &&
           mFinalizeActiveParticleExplicitContactsPass.dispatch(computeContext, kDefaultVariant,
                                                                finalizeBindings, 1u) &&
           mCompactActiveParticleExplicitContactsPass.dispatchIndirect(
               computeContext, kDefaultVariant, compactBindings,
               transient.physicsIndirectArgsBuffer,
               indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftCompactContacts));
}

bool PhysicsPassDispatcher::compactParticleRigidContacts(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array compactBindings{
        gpu::GpuBufferBinding{"g_ParticleRigidContacts", transient.softRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ActiveSoftRigidContacts", transient.activeSoftRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return dispatchExclusiveScanWithGpuCount(computeContext, sceneState,
                                             transient.softRadixBitFlagsBuffer,
                                             transient.softRadixBitOffsetsBuffer, true) &&
           writeParticleDispatchConstants(computeContext, constants) &&
           mFinalizeActiveParticleRigidContactsPass.dispatch(computeContext, kDefaultVariant,
                                                             finalizeBindings, 1u) &&
           mCompactActiveParticleRigidContactsPass.dispatchIndirect(
               computeContext, kDefaultVariant, compactBindings,
               transient.physicsIndirectArgsBuffer,
               indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftCompactRigidContacts));
}

bool PhysicsPassDispatcher::clearSoftConstraintState(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     std::uint32_t threadCount,
                                                     const GpuParticleDispatchConstants &constants)
{
    if (threadCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionCorrections",
                              transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleVelocityCorrections",
                              transient.softVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftEdgeLambdas", transient.softEdgeLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftBendLambdas", transient.softBendLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftTetLambdas", transient.softTetLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_StrandSegmentLambdas", transient.strandSegmentLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_StrandJointLambdas", transient.strandJointLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_StrandDistanceLambdas", transient.strandDistanceLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mClearSoftConstraintStatePass.dispatch(computeContext, kDefaultVariant, bindings,
                                                  dispatchGroupCount(threadCount));
}

bool PhysicsPassDispatcher::clearRoutedCableConstraintState(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t routedCableCount, const GpuRigidDispatchConstants &constants)
{
    if (routedCableCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RoutedCableLambdas", transient.routedCableLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeRigidDispatchConstants(computeContext, constants) &&
           mClearRoutedCableConstraintStatePass.dispatch(
               computeContext, kDefaultVariant, bindings, dispatchGroupCount(routedCableCount));
}

bool PhysicsPassDispatcher::clearRigidParticleAttachmentConstraintState(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t constraintCount, const GpuRigidDispatchConstants &constants)
{
    if (constraintCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidParticleAttachmentLambdas",
                              transient.rigidParticleAttachmentLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    GpuRigidDispatchConstants dispatchConstants = constants;
    dispatchConstants.reserved0 = constraintCount;
    return writeRigidDispatchConstants(computeContext, dispatchConstants) &&
           mClearRigidParticleAttachmentConstraintStatePass.dispatch(
               computeContext, kDefaultVariant, bindings, dispatchGroupCount(constraintCount));
}

bool PhysicsPassDispatcher::clearStrandRigidAttachmentConstraintState(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t constraintCount, const GpuRigidDispatchConstants &constants)
{
    if (constraintCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandRigidAttachmentLambdas",
                              transient.strandRigidAttachmentLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    GpuRigidDispatchConstants dispatchConstants = constants;
    dispatchConstants.reserved0 = constraintCount;
    return writeRigidDispatchConstants(computeContext, dispatchConstants) &&
           mClearStrandRigidAttachmentConstraintStatePass.dispatch(
               computeContext, kDefaultVariant, bindings, dispatchGroupCount(constraintCount));
}

bool PhysicsPassDispatcher::clearRigidDistanceConstraintState(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t constraintCount, const GpuRigidDispatchConstants &constants)
{
    if (constraintCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidDistanceConstraintLambdas",
                              transient.rigidDistanceConstraintLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    GpuRigidDispatchConstants dispatchConstants = constants;
    dispatchConstants.reserved0 = constraintCount;
    return writeRigidDispatchConstants(computeContext, dispatchConstants) &&
           mClearRigidDistanceConstraintStatePass.dispatch(
               computeContext, kDefaultVariant, bindings, dispatchGroupCount(constraintCount));
}

bool PhysicsPassDispatcher::dispatchClearSuturingCandidatesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t suturingParticleCount, const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || suturingParticleCount == 0u ||
        constants.suturingPairCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingCandidateCounts", transient.suturingCandidateCountsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SuturingCandidateParticles",
                              transient.suturingCandidateParticlesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mClearSuturingCandidatesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(suturingParticleCount));
}

bool PhysicsPassDispatcher::clearSuturingCandidates(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t suturingParticleCount,
                                                    const GpuParticleDispatchConstants &constants)
{
    return dispatchClearSuturingCandidatesPass(computeContext, sceneState, suturingParticleCount,
                                               constants);
}

bool PhysicsPassDispatcher::dispatchGatherSuturingCandidatesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.suturingPairCount == 0u ||
        constants.suturingParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", softParticles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingNeighborLinks", softParticles.suturingNeighborLinksBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCandidatePairs", transient.softSoftCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingCandidateCounts", transient.suturingCandidateCountsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SuturingCandidateParticles",
                              transient.suturingCandidateParticlesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mGatherSuturingCandidatesPass.dispatchIndirect(
               computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
               indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftGenerateContacts));
}

bool PhysicsPassDispatcher::gatherSuturingCandidates(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     const GpuParticleDispatchConstants &constants)
{
    return dispatchGatherSuturingCandidatesPass(computeContext, sceneState, constants);
}

bool PhysicsPassDispatcher::dispatchClassifySuturingParticlesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t suturingParticleCount, const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || suturingParticleCount == 0u ||
        constants.suturingPairCount == 0u)
    {
        return true;
    }

    const auto &softParticles           = sceneState.persistentParticles();
    const auto &softTopology            = sceneState.persistentSoftTopology();
    const auto &transient               = sceneState.transientBuffers();
    const PhysicsGpuSceneView sceneView = sceneState.sceneView();
    const auto &suturing                = sceneView.soft;
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingParticleRefs", suturing.suturingParticleRefsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", softParticles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwningSoftBodyIndices",
                              softParticles.owningSoftBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingCandidateParticles",
                              transient.suturingCandidateParticlesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleTetRanges", softTopology.particleTetRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleIncidentTets", softTopology.particleIncidentTetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTets", softTopology.tetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingPairs", suturing.suturingPairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingInsertionStates", suturing.suturingInsertionStatesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mClassifySuturingParticlesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                   dispatchGroupCount(suturingParticleCount));
}

bool PhysicsPassDispatcher::classifySuturingParticles(Diligent::IDeviceContext *computeContext,
                                                      const PhysicsSceneGpuState &sceneState,
                                                      std::uint32_t suturingParticleCount,
                                                      const GpuParticleDispatchConstants &constants)
{
    return dispatchClassifySuturingParticlesPass(computeContext, sceneState, suturingParticleCount,
                                                 constants);
}

bool PhysicsPassDispatcher::dispatchUpdateSuturingTipPathsPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t suturingPairCount, const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || suturingPairCount == 0u)
    {
        return true;
    }

    const auto &softParticles           = sceneState.persistentParticles();
    const auto &softTopology            = sceneState.persistentSoftTopology();
    const PhysicsGpuSceneView sceneView = sceneState.sceneView();
    const auto &suturing                = sceneView.soft;
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTets", softTopology.tetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingPairs", suturing.suturingPairsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SuturingInsertionStates", suturing.suturingInsertionStatesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingPathHeaders", suturing.suturingPathHeadersBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SuturingPathNodes", suturing.suturingPathNodesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mUpdateSuturingTipPathsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                dispatchGroupCount(suturingPairCount));
}

bool PhysicsPassDispatcher::updateSuturingTipPaths(Diligent::IDeviceContext *computeContext,
                                                   const PhysicsSceneGpuState &sceneState,
                                                   std::uint32_t suturingPairCount,
                                                   const GpuParticleDispatchConstants &constants)
{
    return dispatchUpdateSuturingTipPathsPass(computeContext, sceneState, suturingPairCount,
                                              constants);
}

bool PhysicsPassDispatcher::dispatchAssignSuturingInsideParticlesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t suturingParticleCount, const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || suturingParticleCount == 0u ||
        constants.suturingPairCount == 0u)
    {
        return true;
    }

    const auto &softParticles           = sceneState.persistentParticles();
    const auto &softTopology            = sceneState.persistentSoftTopology();
    const PhysicsGpuSceneView sceneView = sceneState.sceneView();
    const auto &suturing                = sceneView.soft;
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingParticleRefs", suturing.suturingParticleRefsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingPairs", suturing.suturingPairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingInsertionStates", suturing.suturingInsertionStatesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SuturingPathHeaders", suturing.suturingPathHeadersBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingPathNodes", suturing.suturingPathNodesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTets", softTopology.tetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mAssignSuturingInsideParticlesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                       dispatchGroupCount(suturingParticleCount));
}

bool PhysicsPassDispatcher::assignSuturingInsideParticles(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t suturingParticleCount, const GpuParticleDispatchConstants &constants)
{
    return dispatchAssignSuturingInsideParticlesPass(computeContext, sceneState,
                                                     suturingParticleCount, constants);
}

bool PhysicsPassDispatcher::dispatchSolveSuturingNodePathConstraintsPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t suturingParticleCount, const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || suturingParticleCount == 0u ||
        constants.suturingPairCount == 0u)
    {
        return true;
    }

    const auto &softParticles           = sceneState.persistentParticles();
    const auto &softTopology            = sceneState.persistentSoftTopology();
    const auto &transient               = sceneState.transientBuffers();
    const auto &persistentRigid         = sceneState.persistentRigidBodies();
    const PhysicsGpuSceneView sceneView = sceneState.sceneView();
    const auto &suturing                = sceneView.soft;
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingParticleRefs", suturing.suturingParticleRefsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePreviousPositions", softParticles.previousPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerIndices", softParticles.ownerIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidProxyLocalPositions",
                              softParticles.rigidProxyLocalPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingInsertionStates", suturing.suturingInsertionStatesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SuturingPathNodes", suturing.suturingPathNodesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTets", softTopology.tetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                              transient.previousRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyOrientations",
                              transient.previousRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentRigid.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionCorrections",
                              transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mSolveSuturingNodePathConstraintsPass.dispatch(
               computeContext, kDefaultVariant, bindings,
               dispatchGroupCount(suturingParticleCount));
}

bool PhysicsPassDispatcher::solveSuturingNodePathConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t suturingParticleCount, const GpuParticleDispatchConstants &constants)
{
    return dispatchSolveSuturingNodePathConstraintsPass(computeContext, sceneState,
                                                        suturingParticleCount, constants);
}

bool PhysicsPassDispatcher::solveSoftEdgeConstraints(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     std::uint32_t softEdgeCount,
                                                     const GpuParticleDispatchConstants &constants)
{
    if (softEdgeCount == 0u || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftEdges", softTopology.edgesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftEdgeLambdas", transient.softEdgeLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftEdgeCorrections", transient.softEdgeCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mSolveSoftEdgeConstraintsPass.dispatch(computeContext, kDefaultVariant, solveBindings,
                                                  dispatchGroupCount(softEdgeCount));
}

bool PhysicsPassDispatcher::solveSoftBendConstraints(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     std::uint32_t softBendCount,
                                                     const GpuParticleDispatchConstants &constants)
{
    if (softBendCount == 0u || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBends", softTopology.bendsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBendLambdas", transient.softBendLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftBendCorrections", transient.softBendCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mSolveSoftBendConstraintsPass.dispatch(computeContext, kDefaultVariant, solveBindings,
                                                  dispatchGroupCount(softBendCount));
}

bool PhysicsPassDispatcher::solveSoftTetConstraints(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t softTetCount,
                                                    const GpuParticleDispatchConstants &constants)
{
    if (softTetCount == 0u || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTets", softTopology.tetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTetLambdas", transient.softTetLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftTetCorrections", transient.softTetCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mSolveSoftTetConstraintsPass.dispatch(computeContext, kDefaultVariant, solveBindings,
                                                 dispatchGroupCount(softTetCount));
}

bool PhysicsPassDispatcher::applySoftEdgeCorrections(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleEdgeRanges", softTopology.particleEdgeRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleIncidentEdges", softTopology.particleIncidentEdgesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftEdgeCorrections", transient.softEdgeCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplySoftEdgeCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                  dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::applySoftBendCorrections(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleBendRanges", softTopology.particleBendRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleIncidentBends", softTopology.particleIncidentBendsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBendCorrections", transient.softBendCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplySoftBendCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                  dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::applySoftTetCorrections(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleTetRanges", softTopology.particleTetRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleIncidentTets", softTopology.particleIncidentTetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTetCorrections", transient.softTetCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplySoftTetCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::solveStrandSegmentConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t strandSegmentCount, const GpuParticleDispatchConstants &constants)
{
    if (strandSegmentCount == 0u || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegments", softTopology.strandSegmentsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegmentStates", softTopology.strandSegmentStatesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegmentLambdas", transient.strandSegmentLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_StrandSegmentCorrections",
                              transient.strandSegmentCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mSolveStrandSegmentConstraintsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                       dispatchGroupCount(strandSegmentCount));
}

bool PhysicsPassDispatcher::applyStrandSegmentCorrections(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t dispatchCount, const GpuParticleDispatchConstants &constants)
{
    if (dispatchCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleStrandSegmentRanges",
                              softTopology.particleStrandSegmentRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleIncidentStrandSegments",
                              softTopology.particleIncidentStrandSegmentsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegmentCorrections",
                              transient.strandSegmentCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegmentStates", softTopology.strandSegmentStatesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplyStrandSegmentCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                       dispatchGroupCount(dispatchCount));
}

bool PhysicsPassDispatcher::solveStrandJointConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t strandJointCount, const GpuParticleDispatchConstants &constants)
{
    if (strandJointCount == 0u || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandJoints", softTopology.strandJointsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegments", softTopology.strandSegmentsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegmentStates", softTopology.strandSegmentStatesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandJointLambdas", transient.strandJointLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_StrandJointCorrections",
                              transient.strandJointCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mSolveStrandJointConstraintsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                     dispatchGroupCount(strandJointCount));
}

bool PhysicsPassDispatcher::applyStrandJointCorrections(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t dispatchCount, const GpuParticleDispatchConstants &constants)
{
    if (dispatchCount == 0u)
    {
        return true;
    }

    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandJointCorrections", transient.strandJointCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SegmentStrandJointRanges",
                              softTopology.segmentStrandJointRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SegmentIncidentStrandJoints",
                              softTopology.segmentIncidentStrandJointsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegmentStates", softTopology.strandSegmentStatesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplyStrandJointCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                     dispatchGroupCount(dispatchCount));
}

bool PhysicsPassDispatcher::applyStrandRigidAttachmentCorrections(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t dispatchCount, const GpuParticleDispatchConstants &constants)
{
    if (dispatchCount == 0u)
    {
        return true;
    }

    const auto &softTopology = sceneState.persistentSoftTopology();
    const auto &transient    = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandRigidAttachmentCorrections",
                              transient.strandRigidAttachmentCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SegmentStrandRigidAttachmentRanges",
                              softTopology.segmentStrandRigidAttachmentRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SegmentIncidentStrandRigidAttachments",
                              softTopology.segmentIncidentStrandRigidAttachmentsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegmentStates", softTopology.strandSegmentStatesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplyStrandRigidAttachmentCorrectionsPass.dispatch(
               computeContext, kDefaultVariant, bindings, dispatchGroupCount(dispatchCount));
}

bool PhysicsPassDispatcher::solveStrandDistanceConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t strandDistanceCount, const GpuParticleDispatchConstants &constants)
{
    if (strandDistanceCount == 0u || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandDistanceConstraints",
                              softTopology.strandDistanceConstraintsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandDistanceLambdas", transient.strandDistanceLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_StrandDistanceCorrections",
                              transient.strandDistanceCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mSolveStrandDistanceConstraintsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                        dispatchGroupCount(strandDistanceCount));
}

bool PhysicsPassDispatcher::applyStrandDistanceCorrections(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t particleCount, const GpuParticleDispatchConstants &constants)
{
    if (particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleStrandSegmentRanges",
                              softTopology.particleStrandSegmentRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleIncidentStrandSegments",
                              softTopology.particleIncidentStrandSegmentsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandDistanceCorrections",
                              transient.strandDistanceCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplyStrandDistanceCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                        dispatchGroupCount(particleCount));
}

bool PhysicsPassDispatcher::solveParticleExplicitContacts(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles   = sceneState.persistentParticles();
    const auto &persistentRigid = sceneState.persistentRigidBodies();
    const auto &transient       = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePreviousPositions", softParticles.previousPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", softParticles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerIndices", softParticles.ownerIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidProxyLocalPositions",
                              softParticles.rigidProxyLocalPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                              persistentRigid.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyOrientations", persistentRigid.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentRigid.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleContacts", transient.activeSoftContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionCorrections",
                              transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveParticleExplicitContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, solveBindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftSolveContacts));
}

bool PhysicsPassDispatcher::solveParticleRigidContacts(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles       = sceneState.persistentParticles();
    const auto &persistentRigid     = sceneState.persistentRigidBodies();
    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &transient           = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePreviousPositions", softParticles.previousPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleMaterialIndices",
                              softParticles.particleMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleContactMaterials",
                              softParticles.particleContactMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                              persistentRigid.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyOrientations", persistentRigid.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentRigid.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderMaterials", persistentColliders.materialBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRigidContacts", transient.activeSoftRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionCorrections",
                              transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveParticleRigidContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, solveBindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftSolveRigidContacts));
}

bool PhysicsPassDispatcher::applyParticlePositionCorrections(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticlePositionCorrections",
                              transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplyParticlePositionCorrectionsPass.dispatch(
               computeContext, kDefaultVariant, bindings,
               dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::updateParticleVelocities(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     std::uint32_t particleCount,
                                                     const GpuParticleDispatchConstants &constants)
{
    if (particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticlePreviousPositions", softParticles.previousPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleMaterialIndices",
                              softParticles.particleMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleContactMaterials",
                              softParticles.particleContactMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_FluidIterationDelta", transient.fluidIterationDeltaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mUpdateParticleVelocitiesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                  dispatchGroupCount(particleCount));
}

bool PhysicsPassDispatcher::computeFluidDensityConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles   = sceneState.persistentParticles();
    const auto &persistentRigid = sceneState.persistentRigidBodies();
    const auto &colliders       = sceneState.persistentColliders();
    const auto &transient       = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterialIndices", softParticles.fluidMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterials", softParticles.fluidMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidIterationDelta", transient.fluidIterationDeltaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborPairs", transient.fluidNeighborPairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidBoundaryCandidateRanges",
                              transient.fluidBoundaryCandidateRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidBoundaryCandidatePairs",
                              transient.fluidBoundaryCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyScales", persistentRigid.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderGeometryData", colliders.geometryDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", colliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidSurfaceNormalConstraints",
                              transient.fluidSurfaceNormalConstraintsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mComputeFluidDensityConstraintsPass.dispatch(
               computeContext, kDefaultVariant, bindings,
               dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::buildFluidNeighborPairs(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &particles = sceneState.persistentParticles();
    const auto &transient = sceneState.transientBuffers();
    const std::array buildBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseEntries",
                              transient.particleBroadPhaseEntriesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRanges", transient.particleCellRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedParticleBroadPhaseKeys",
                              transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseMetadata", particles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", particles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterialIndices", particles.fluidMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterials", particles.fluidMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidIterationDelta", transient.fluidIterationDeltaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_FluidNeighborPairs", transient.fluidNeighborPairsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mBuildFluidNeighborPairsPass.dispatch(computeContext, kDefaultVariant, buildBindings,
                                                 dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::computeFluidDeltaPositions(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles   = sceneState.persistentParticles();
    const auto &persistentRigid = sceneState.persistentRigidBodies();
    const auto &colliders       = sceneState.persistentColliders();
    const auto &transient       = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterialIndices", softParticles.fluidMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterials", softParticles.fluidMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidIterationDelta", transient.fluidIterationDeltaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidSurfaceNormalConstraints",
                              transient.fluidSurfaceNormalConstraintsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborPairs", transient.fluidNeighborPairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidBoundaryCandidateRanges",
                              transient.fluidBoundaryCandidateRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidBoundaryCandidatePairs",
                              transient.fluidBoundaryCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyScales", persistentRigid.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderGeometryData", colliders.geometryDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", colliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidDeltaPositions", transient.fluidDeltaPositionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mComputeFluidDeltaPositionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                    dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::applyFluidDeltaPositions(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidDeltaPositions", transient.fluidDeltaPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidIterationDelta", transient.fluidIterationDeltaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplyFluidDeltaPositionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                  dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::clampFluidBoundary(Diligent::IDeviceContext *computeContext,
                                               const PhysicsSceneGpuState &sceneState,
                                               const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles   = sceneState.persistentParticles();
    const auto &persistentRigid = sceneState.persistentRigidBodies();
    const auto &colliders       = sceneState.persistentColliders();
    const auto &transient       = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidIterationDelta", transient.fluidIterationDeltaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidBoundaryCandidateRanges",
                              transient.fluidBoundaryCandidateRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidBoundaryCandidatePairs",
                              transient.fluidBoundaryCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyScales", persistentRigid.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderGeometryData", colliders.geometryDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", colliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidIterationDeltaRW", transient.fluidIterationDeltaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mClampFluidBoundaryPass.dispatch(computeContext, kDefaultVariant, bindings,
                                            dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::projectFluidBoundaryVelocities(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &softParticles       = sceneState.persistentParticles();
    const auto &persistentRigid     = sceneState.persistentRigidBodies();
    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &transient           = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", softParticles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleMaterialIndices",
                              softParticles.particleMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleContactMaterials",
                              softParticles.particleContactMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_FluidBoundaryCandidateRanges",
                              transient.fluidBoundaryCandidateRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidBoundaryCandidatePairs",
                              transient.fluidBoundaryCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyScales", persistentRigid.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderMaterials", persistentColliders.materialBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderGeometryData", persistentColliders.geometryDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", persistentColliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mProjectFluidBoundaryVelocitiesPass.dispatch(
               computeContext, kDefaultVariant, bindings,
               dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::computeFluidVorticity(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState,
                                                  const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &particles = sceneState.persistentParticles();
    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", particles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterialIndices", particles.fluidMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterials", particles.fluidMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocitiesRW", particles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborPairs", transient.fluidNeighborPairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidVorticities", transient.fluidVorticitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mComputeFluidVorticityPass.dispatch(computeContext, kDefaultVariant, bindings,
                                               dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::applyFluidVorticityConfinement(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &particles = sceneState.persistentParticles();
    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", particles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterialIndices", particles.fluidMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterials", particles.fluidMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidVorticities", transient.fluidVorticitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborPairs", transient.fluidNeighborPairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocitiesRW", particles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mApplyFluidVorticityConfinementPass.dispatch(
               computeContext, kDefaultVariant, bindings,
               dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::buildFluidRenderAnisotropy(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (constants.particleCount == 0u)
    {
        return true;
    }

    const auto &particles = sceneState.persistentParticles();
    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleKinds", particles.particleKindsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", particles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterialIndices", particles.fluidMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidMaterials", particles.fluidMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidNeighborPairs", transient.fluidNeighborPairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_FluidAnisotropy1RW", transient.fluidAnisotropy1Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_FluidAnisotropy2RW", transient.fluidAnisotropy2Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_FluidAnisotropy3RW", transient.fluidAnisotropy3Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeParticleDispatchConstants(computeContext, constants) &&
           mBuildFluidRenderAnisotropyPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                    dispatchGroupCount(constants.particleCount));
}

bool PhysicsPassDispatcher::dispatchSolveParticleContactVelocitiesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &softParticles   = sceneState.persistentParticles();
    const auto &persistentRigid = sceneState.persistentRigidBodies();
    const auto &transient       = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", softParticles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerIndices", softParticles.ownerIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidProxyLocalPositions",
                              softParticles.rigidProxyLocalPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentRigid.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleContacts", transient.activeSoftContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocityCorrections",
                              transient.softVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveParticleContactVelocitiesPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftSolveContactVelocities));
}

bool PhysicsPassDispatcher::solveParticleContactVelocities(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t particleCount, std::uint32_t rigidBodyCount, std::uint32_t iterations,
    const GpuRigidDispatchConstants &rigidConstants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (particleCount == 0u || iterations == 0u)
    {
        return true;
    }

    const auto &softParticles   = sceneState.persistentParticles();
    const auto &persistentRigid = sceneState.persistentRigidBodies();
    const auto &transient       = sceneState.transientBuffers();
    const std::array applyBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleVelocityCorrections",
                              transient.softVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array applyRigidBindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!mApplyParticleContactVelocitiesPass.bindVariant(kDefaultVariant, applyBindings) ||
        !mApplyRigidContactVelocitiesPass.bindVariant(kDefaultVariant, applyRigidBindings))
    {
        return false;
    }

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        if (!dispatchSolveParticleContactVelocitiesPass(computeContext, sceneState) ||
            !mApplyParticleContactVelocitiesPass.dispatch(
                computeContext, kDefaultVariant, applyBindings, dispatchGroupCount(particleCount)))
        {
            return false;
        }
        if (rigidBodyCount > 0u && (!writeRigidDispatchConstants(computeContext, rigidConstants) ||
                                    !mApplyRigidContactVelocitiesPass.dispatch(
                                        computeContext, kDefaultVariant, applyRigidBindings,
                                        dispatchGroupCount(rigidBodyCount))))
        {
            return false;
        }
    }

    return true;
}

bool PhysicsPassDispatcher::dispatchSolveParticleRigidContactVelocitiesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &softParticles       = sceneState.persistentParticles();
    const auto &persistentRigid     = sceneState.persistentRigidBodies();
    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &transient           = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleMaterialIndices",
                              softParticles.particleMaterialIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleContactMaterials",
                              softParticles.particleContactMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentRigid.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderMaterials", persistentColliders.materialBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRigidContacts", transient.activeSoftRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocityCorrections",
                              transient.softVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveParticleRigidContactVelocitiesPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftSolveRigidContacts));
}

bool PhysicsPassDispatcher::solveParticleRigidContactVelocities(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t particleCount, std::uint32_t rigidBodyCount, std::uint32_t iterations,
    const GpuRigidDispatchConstants &rigidConstants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (particleCount == 0u || iterations == 0u)
    {
        return true;
    }

    const auto &softParticles   = sceneState.persistentParticles();
    const auto &transient       = sceneState.transientBuffers();
    const auto &persistentRigid = sceneState.persistentRigidBodies();
    const std::array applySoftBindings{
        gpu::GpuBufferBinding{"PhysicsParticleDispatchConstantsBuffer",
                              mParticleDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleVelocityCorrections",
                              transient.softVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array applyRigidBindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentRigid.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!mApplyParticleContactVelocitiesPass.bindVariant(kDefaultVariant, applySoftBindings) ||
        !mApplyRigidContactVelocitiesPass.bindVariant(kDefaultVariant, applyRigidBindings))
    {
        return false;
    }

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        if (!dispatchSolveParticleRigidContactVelocitiesPass(computeContext, sceneState))
        {
            return false;
        }

        if (!mApplyParticleContactVelocitiesPass.dispatch(computeContext, kDefaultVariant,
                                                          applySoftBindings,
                                                          dispatchGroupCount(particleCount)) ||
            !mApplyRigidContactVelocitiesPass.dispatch(computeContext, kDefaultVariant,
                                                       applyRigidBindings,
                                                       dispatchGroupCount(rigidBodyCount)))
        {
            return false;
        }
    }

    return true;
}

bool PhysicsPassDispatcher::skinSoftRenderVertices(Diligent::IDeviceContext *computeContext,
                                                   const PhysicsSceneGpuState &sceneState,
                                                   std::uint32_t renderVertexCount)
{
    if (renderVertexCount == 0u)
    {
        return true;
    }

    const GpuSoftRenderDispatchConstants constants{renderVertexCount, 0u, 0u, 0u};
    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftRenderDispatchConstantsBuffer",
                              mSoftRenderDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftRenderVertexBindings",
                              softTopology.renderVertexBindingsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBodyRenderPositionsRW",
                              softTopology.softBodyRenderPositionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftRenderDispatchConstants(computeContext, constants) &&
           mSkinSoftRenderVerticesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                dispatchGroupCount(renderVertexCount));
}

bool PhysicsPassDispatcher::updateSoftTriangleNormals(Diligent::IDeviceContext *computeContext,
                                                      const PhysicsSceneGpuState &sceneState,
                                                      std::uint32_t renderTriangleCount)
{
    if (renderTriangleCount == 0u)
    {
        return true;
    }

    const GpuSoftRenderDispatchConstants constants{0u, renderTriangleCount, 0u, 0u};
    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftRenderDispatchConstantsBuffer",
                              mSoftRenderDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftRenderTriangleParticleIndices",
                              softTopology.renderTriangleParticleIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftRenderTriangleNormalsRW",
                              softTopology.renderTriangleNormalsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftRenderDispatchConstants(computeContext, constants) &&
           mUpdateSoftTriangleNormalsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                   dispatchGroupCount(renderTriangleCount));
}

bool PhysicsPassDispatcher::updateSoftRenderNormals(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t renderVertexCount)
{
    if (renderVertexCount == 0u)
    {
        return true;
    }

    const GpuSoftRenderDispatchConstants constants{renderVertexCount, 0u, 0u, 0u};
    const auto &softTopology = sceneState.persistentSoftTopology();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftRenderDispatchConstantsBuffer",
                              mSoftRenderDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftRenderTriangleNormals",
                              softTopology.renderTriangleNormalsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftRenderVertexTriangleRanges",
                              softTopology.renderVertexTriangleRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftRenderVertexTriangleIndices",
                              softTopology.renderVertexTriangleIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftRenderFallbackNormals",
                              softTopology.softBodyFallbackNormalsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBodyRenderNormalsRW", softTopology.softBodyRenderNormalsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftRenderDispatchConstants(computeContext, constants) &&
           mUpdateSoftRenderNormalsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(renderVertexCount));
}

bool PhysicsPassDispatcher::updateCurveRenderData(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState,
                                                  std::uint32_t curveCount)
{
    if (curveCount == 0u)
    {
        return true;
    }

    const GpuCurveRenderDispatchConstants constants{curveCount, 0u, 0u, 0u};
    const auto &softParticles = sceneState.persistentParticles();
    const auto &curveRender   = sceneState.persistentCurveRender();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsCurveRenderDispatchConstantsBuffer",
                              mCurveRenderDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CurveRenderDescriptors", curveRender.descriptorsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CurveRenderParticleIndices", curveRender.particleIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CurveRenderPositionsRW", curveRender.positionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_CurveRenderNormalsRW", curveRender.normalsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_CurveWorldAabbsRW", curveRender.worldAabbsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeCurveRenderDispatchConstants(computeContext, constants) &&
           mUpdateCurveRenderDataPass.dispatch(computeContext, kDefaultVariant, bindings,
                                               dispatchGroupCount(curveCount));
}

bool PhysicsPassDispatcher::updateSoftBodyBounds(Diligent::IDeviceContext *computeContext,
                                                 const PhysicsSceneGpuState &sceneState,
                                                 std::uint32_t softBodyCount,
                                                 std::uint32_t softBodyBoundsChunkCount)
{
    if (softBodyCount == 0u || softBodyBoundsChunkCount == 0u)
    {
        return true;
    }

    const GpuSoftRenderDispatchConstants chunkConstants{0u, 0u, 0u, softBodyBoundsChunkCount};
    const GpuSoftRenderDispatchConstants finalizeConstants{0u, 0u, softBodyCount, 0u};
    const auto &softParticles = sceneState.persistentParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array chunkBindings{
        gpu::GpuBufferBinding{"PhysicsSoftRenderDispatchConstantsBuffer",
                              mSoftRenderDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBodyBoundsChunks", softTopology.softBodyBoundsChunksBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBodyChunkAabbsRW", transient.softBodyChunkAabbsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"PhysicsSoftRenderDispatchConstantsBuffer",
                              mSoftRenderDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBodyChunkRanges", softTopology.softBodyChunkRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBodyChunkAabbs", transient.softBodyChunkAabbsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBodyWorldAabbsRW", softTopology.softBodyWorldAabbsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftRenderDispatchConstants(computeContext, chunkConstants) &&
           mUpdateSoftBodyBoundsPass.dispatch(computeContext, kDefaultVariant, chunkBindings,
                                              softBodyBoundsChunkCount) &&
           writeSoftRenderDispatchConstants(computeContext, finalizeConstants) &&
           mFinalizeSoftBodyBoundsPass.dispatch(computeContext, kDefaultVariant, finalizeBindings,
                                                softBodyCount);
}

bool PhysicsPassDispatcher::updateRigidWorldAabbs(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState,
                                                  std::uint32_t bodyCount,
                                                  const GpuRigidDispatchConstants &constants)
{
    if (constants.colliderCount == 0u)
    {
        return true;
    }

    const auto &persistentBodies    = sceneState.persistentRigidBodies();
    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &jointSuppression    = sceneState.persistentJointCollisionSuppression();
    const auto &transient           = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                              transient.previousRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyOrientations",
                              transient.previousRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyScales", persistentBodies.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderOwnerRigidBodyIndices",
                              persistentColliders.ownerRigidBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderShapeTypes", persistentColliders.shapeTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderGeometryData", persistentColliders.geometryDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderEnabledFlags", persistentColliders.enabledFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_BodyMeta", transient.bodyMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ActiveBodyFlags", transient.activeBodyFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_StaticBodyFlags", transient.staticBodyFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeRigidDispatchConstants(computeContext, constants) &&
           mUpdateRigidWorldAabbsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                               dispatchGroupCount(constants.colliderCount));
}

bool PhysicsPassDispatcher::compactBroadPhaseBodySets(Diligent::IDeviceContext *computeContext,
                                                      const PhysicsSceneGpuState &sceneState,
                                                      std::uint32_t bodyCount,
                                                      const GpuRigidDispatchConstants &constants)
{
    // Conceptual example:
    //
    // full body domain:
    // - bodyIndex:          0 1 2 3 4 5
    // - active flags:       0 1 0 1 1 0
    // - exclusive offsets:  0 0 1 1 2 3
    //
    // compact domain:
    // - activeIndex:        0 1 2
    // - activeBodyIndices:  1 3 4

    if (!dispatchExclusiveScanWithCpuCount(
            computeContext, sceneState, sceneState.transientBuffers().activeBodyFlagsBuffer,
            sceneState.transientBuffers().activeBodyOffsetsBuffer, constants.colliderCount))
    {
        return false;
    }
    if (!dispatchExclusiveScanWithCpuCount(
            computeContext, sceneState, sceneState.transientBuffers().staticBodyFlagsBuffer,
            sceneState.transientBuffers().staticBodyOffsetsBuffer, constants.colliderCount))
    {
        return false;
    }

    const auto &transient = sceneState.transientBuffers();

    const std::array movingSetCompactBindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodySetFlags", transient.activeBodyFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodySetOffsets", transient.activeBodyOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseBodyIndices", transient.activeBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_BodyMeta", transient.bodyMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::array staticSetCompactBindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodySetFlags", transient.staticBodyFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodySetOffsets", transient.staticBodyOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseBodyIndices", transient.staticBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_BodyMeta", transient.bodyMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ActiveBodyFlags", transient.activeBodyFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ActiveBodyOffsets", transient.activeBodyOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StaticBodyFlags", transient.staticBodyFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StaticBodyOffsets", transient.staticBodyOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!writeRigidDispatchConstants(computeContext, constants) ||
        !mCompactBodySetPass.dispatch(computeContext, kDefaultVariant, movingSetCompactBindings,
                                      dispatchGroupCount(constants.colliderCount)))
    {
        return false;
    }

    if (!writeRigidDispatchConstants(computeContext, constants) ||
        !mCompactBodySetPass.dispatch(computeContext, kAltVariant, staticSetCompactBindings,
                                      dispatchGroupCount(constants.colliderCount)))
    {
        return false;
    }

    return writeRigidDispatchConstants(computeContext, constants) &&
           mFinalizeActiveBodiesPass.dispatch(computeContext, kDefaultVariant, finalizeBindings,
                                              1u);
}

bool PhysicsPassDispatcher::dispatchReduceBroadPhaseExtentPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t bodyCount, bool useStaticSet)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (bodyCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();

    const auto &scratchBuffers = useStaticSet ? transient.staticBroadPhaseExtentScratchBuffers
                                              : transient.broadPhaseExtentScratchBuffers;

    Diligent::IBuffer *broadPhaseElementsBuffer = useStaticSet
                                                      ? transient.staticBroadPhaseElementsBuffer
                                                      : transient.broadPhaseElementsBuffer;

    Diligent::IBuffer *globalBroadPhaseExtentBuffer =
        useStaticSet ? transient.staticGlobalBroadPhaseExtentBuffer
                     : transient.globalBroadPhaseExtentBuffer;

    if (scratchBuffers.empty() || globalBroadPhaseExtentBuffer == nullptr ||
        broadPhaseElementsBuffer == nullptr)
    {
        return false;
    }

    const std::uint32_t initialGroupCount = dispatchGroupCount(bodyCount);
    Diligent::IBuffer *currentOutput =
        (initialGroupCount <= 1u) ? globalBroadPhaseExtentBuffer : scratchBuffers.front();

    GpuBroadPhaseReductionConstants reductionConstants =
        makeBroadPhaseReductionConstants(bodyCount);

    const std::size_t variantIndex = kDefaultVariant;

    const std::array firstBindings{
        gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseReductionConstantsBuffer",
                              mBroadPhaseReductionConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseElements", broadPhaseElementsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_GroupExtents", currentOutput,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!writeBroadPhaseReductionConstants(computeContext, reductionConstants) ||
        !mReduceExtentElementsPass.dispatch(computeContext, variantIndex, firstBindings,
                                            initialGroupCount))
    {
        return false;
    }

    std::uint32_t currentCount      = initialGroupCount;
    std::uint32_t level             = 1u;
    Diligent::IBuffer *currentInput = currentOutput;

    while (currentCount > 1u)
    {
        const std::uint32_t nextGroupCount = dispatchGroupCount(currentCount);

        if (nextGroupCount > 1u && level >= scratchBuffers.size())
        {
            return false;
        }

        currentOutput =
            (nextGroupCount <= 1u) ? globalBroadPhaseExtentBuffer : scratchBuffers[level];

        reductionConstants = makeBroadPhaseReductionConstants(currentCount);

        const std::array reduceBindings{
            gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseReductionConstantsBuffer",
                                  mBroadPhaseReductionConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_InputExtents", currentInput,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_OutputExtents", currentOutput,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeBroadPhaseReductionConstants(computeContext, reductionConstants) ||
            !mReduceExtentExtentsPass.dispatch(computeContext, variantIndex, reduceBindings,
                                               currentCount))
        {
            return false;
        }

        currentInput = currentOutput;
        currentCount = nextGroupCount;
        ++level;
    }

    return true;
}

bool PhysicsPassDispatcher::dispatchRadixSortPass(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState,
                                                  std::uint32_t count, bool useStaticSet)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (count == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();

    Diligent::IBuffer *finalMortonBuffer = nullptr;
    Diligent::IBuffer *currentInput      = nullptr;
    Diligent::IBuffer *currentOutput     = nullptr;
    Diligent::IBuffer *radixBitFlags     = nullptr;
    Diligent::IBuffer *radixBitOffsets   = nullptr;
    Diligent::IBuffer *radixMeta         = nullptr;

    if (useStaticSet)
    {
        finalMortonBuffer = transient.staticMortonCodesBuffer;
        currentInput      = transient.staticMortonCodesBuffer;
        currentOutput     = transient.staticMortonCodesScratchBuffer;
        radixBitFlags     = transient.staticRadixBitFlagsBuffer;
        radixBitOffsets   = transient.staticRadixBitOffsetsBuffer;
        radixMeta         = transient.staticRadixMetaBuffer;
    }
    else
    {
        finalMortonBuffer = transient.mortonCodesBuffer;
        currentInput      = transient.mortonCodesBuffer;
        currentOutput     = transient.mortonCodesScratchBuffer;
        radixBitFlags     = transient.radixBitFlagsBuffer;
        radixBitOffsets   = transient.radixBitOffsetsBuffer;
        radixMeta         = transient.radixMetaBuffer;
    }

    constexpr std::uint32_t kRadixBitCount = 32u;

    for (std::uint32_t bit = 0u; bit < kRadixBitCount; ++bit)
    {
        const GpuPhysicsRadixConstants radixConstants = makeRadixConstants(count, bit);

        const std::array classifyBindings{
            gpu::GpuBufferBinding{"PhysicsRadixConstantsBuffer", mRadixConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_MortonCodesIn", currentInput,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitFlags", radixBitFlags,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeRadixConstants(computeContext, radixConstants) ||
            !mRadixClassifyPass.dispatch(computeContext, kDefaultVariant, classifyBindings,
                                         dispatchGroupCount(count)))
        {
            return false;
        }

        if (!dispatchExclusiveScanWithCpuCount(computeContext, sceneState, radixBitFlags,
                                               radixBitOffsets, count))
        {
            return false;
        }

        const std::array finalizeBindings{
            gpu::GpuBufferBinding{"PhysicsRadixConstantsBuffer", mRadixConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitFlags", radixBitFlags,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitOffsets", radixBitOffsets,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixMeta", radixMeta, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeRadixConstants(computeContext, radixConstants) ||
            !mRadixFinalizePass.dispatch(computeContext, kDefaultVariant, finalizeBindings, 1u))
        {
            return false;
        }

        const std::array scatterBindings{
            gpu::GpuBufferBinding{"PhysicsRadixConstantsBuffer", mRadixConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_MortonCodesIn", currentInput,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitFlags", radixBitFlags,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitOffsets", radixBitOffsets,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixMeta", radixMeta, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_MortonCodesOut", currentOutput,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeRadixConstants(computeContext, radixConstants) ||
            !mRadixScatterPass.dispatch(computeContext, kDefaultVariant, scatterBindings,
                                        dispatchGroupCount(count)))
        {
            return false;
        }

        std::swap(currentInput, currentOutput);
    }

    if (currentInput != finalMortonBuffer)
    {
        const Diligent::Uint64 bytes =
            static_cast<Diligent::Uint64>(count) * sizeof(GpuMortonCodeElement);

        computeContext->CopyBuffer(
            currentInput, 0u, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            finalMortonBuffer, 0u, bytes, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    return true;
}

bool PhysicsPassDispatcher::dispatchSoftRadixSortPass(Diligent::IDeviceContext *computeContext,
                                                      const PhysicsSceneGpuState &sceneState,
                                                      std::uint32_t count)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (count == 0u)
    {
        return true;
    }

    const auto &transient                = sceneState.transientBuffers();
    Diligent::IBuffer *finalMortonBuffer = transient.particleBroadPhaseKeysBuffer;
    Diligent::IBuffer *currentInput      = transient.particleBroadPhaseKeysBuffer;
    Diligent::IBuffer *currentOutput     = transient.particleBroadPhaseKeysScratchBuffer;
    Diligent::IBuffer *radixBitFlags     = transient.softRadixBitFlagsBuffer;
    Diligent::IBuffer *radixBitOffsets   = transient.softRadixBitOffsetsBuffer;
    Diligent::IBuffer *radixMeta         = transient.softRadixMetaBuffer;

    constexpr std::uint32_t kRadixBitCount = 32u;

    for (std::uint32_t bit = 0u; bit < kRadixBitCount; ++bit)
    {
        const GpuPhysicsRadixConstants radixConstants = makeRadixConstants(count, bit);

        const std::array classifyBindings{
            gpu::GpuBufferBinding{"PhysicsRadixConstantsBuffer", mRadixConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_MortonCodesIn", currentInput,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitFlags", radixBitFlags,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeRadixConstants(computeContext, radixConstants) ||
            !mRadixClassifyPass.dispatch(computeContext, kDefaultVariant, classifyBindings,
                                         dispatchGroupCount(count)))
        {
            return false;
        }

        if (!dispatchExclusiveScanWithCpuCount(computeContext, sceneState, radixBitFlags,
                                               radixBitOffsets, count))
        {
            return false;
        }

        const std::array finalizeBindings{
            gpu::GpuBufferBinding{"PhysicsRadixConstantsBuffer", mRadixConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitFlags", radixBitFlags,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitOffsets", radixBitOffsets,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixMeta", radixMeta, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeRadixConstants(computeContext, radixConstants) ||
            !mRadixFinalizePass.dispatch(computeContext, kDefaultVariant, finalizeBindings, 1u))
        {
            return false;
        }

        const std::array scatterBindings{
            gpu::GpuBufferBinding{"PhysicsRadixConstantsBuffer", mRadixConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_MortonCodesIn", currentInput,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitFlags", radixBitFlags,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixBitOffsets", radixBitOffsets,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RadixMeta", radixMeta, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_MortonCodesOut", currentOutput,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeRadixConstants(computeContext, radixConstants) ||
            !mRadixScatterPass.dispatch(computeContext, kDefaultVariant, scatterBindings,
                                        dispatchGroupCount(count)))
        {
            return false;
        }

        std::swap(currentInput, currentOutput);
    }

    if (currentInput != finalMortonBuffer)
    {
        const Diligent::Uint64 bytes =
            static_cast<Diligent::Uint64>(count) * sizeof(GpuMortonCodeElement);

        computeContext->CopyBuffer(
            currentInput, 0u, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            finalMortonBuffer, 0u, bytes, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    return true;
}

bool PhysicsPassDispatcher::buildBroadPhase(Diligent::IDeviceContext *computeContext,
                                            const PhysicsSceneGpuState &sceneState,
                                            std::uint32_t activeMovingCount,
                                            const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    const auto &transient = sceneState.transientBuffers();
    if (activeMovingCount > 0u)
    {
        const GpuBroadPhaseBuildConstants dynamicBuildConstants =
            makeBroadPhaseBuildConstants(activeMovingCount);

        const std::array buildElementsBindings{
            gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseBuildConstantsBuffer",
                                  mBroadPhaseBuildConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_BroadPhaseBodyIndices", transient.activeBodyIndicesBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_BroadPhaseElements", transient.broadPhaseElementsBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!writeBroadPhaseBuildConstants(computeContext, dynamicBuildConstants) ||
            !mBuildBroadPhaseElementsPass.dispatch(computeContext, kDefaultVariant,
                                                   buildElementsBindings,
                                                   dispatchGroupCount(activeMovingCount)))
        {
            return false;
        }

        if (!dispatchReduceBroadPhaseExtentPass(computeContext, sceneState, activeMovingCount,
                                                false))
        {
            return false;
        }

        const std::array mortonBindings{
            gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseBuildConstantsBuffer",
                                  mBroadPhaseBuildConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_BroadPhaseElements", transient.broadPhaseElementsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_GlobalExtent", transient.globalBroadPhaseExtentBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_MortonCodes", transient.mortonCodesBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!writeBroadPhaseBuildConstants(computeContext, dynamicBuildConstants) ||
            !mMortonCodesPass.dispatch(computeContext, kDefaultVariant, mortonBindings,
                                       dispatchGroupCount(activeMovingCount)))
        {
            return false;
        }

        if (!dispatchRadixSortPass(computeContext, sceneState, activeMovingCount, false))
        {
            return false;
        }

        const std::array bvhHierarchyBindings{
            gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseBuildConstantsBuffer",
                                  mBroadPhaseBuildConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_SortedMortonCodes", transient.mortonCodesBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_BroadPhaseElements", transient.broadPhaseElementsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            gpu::GpuBufferBinding{"g_BvhConstructionInfos", transient.bvhConstructionInfoBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!writeBroadPhaseBuildConstants(computeContext, dynamicBuildConstants) ||
            !mBvhHierarchyPass.dispatch(computeContext, kDefaultVariant, bvhHierarchyBindings,
                                        dispatchGroupCount(activeMovingCount)))
        {
            return false;
        }

        const std::array bvhBoundsBindings{
            gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseBuildConstantsBuffer",
                                  mBroadPhaseBuildConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            gpu::GpuBufferBinding{"g_BvhConstructionInfos", transient.bvhConstructionInfoBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!writeBroadPhaseBuildConstants(computeContext, dynamicBuildConstants) ||
            !mBvhBoundingBoxesPass.dispatch(computeContext, kDefaultVariant, bvhBoundsBindings,
                                            dispatchGroupCount(activeMovingCount)))
        {
            return false;
        }
    }

    if (constants.staticBodyCount == 0u || !sceneState.staticBroadPhaseDirty())
    {
        return true;
    }

    const GpuBroadPhaseBuildConstants staticBuildConstants =
        makeBroadPhaseBuildConstants(constants.staticBodyCount);

    const std::array staticBuildElementsBindings{
        gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseBuildConstantsBuffer",
                              mBroadPhaseBuildConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseBodyIndices", transient.staticBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseElements", transient.staticBroadPhaseElementsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeBroadPhaseBuildConstants(computeContext, staticBuildConstants) ||
        !mBuildBroadPhaseElementsPass.dispatch(computeContext, kAltVariant,
                                               staticBuildElementsBindings,
                                               dispatchGroupCount(constants.staticBodyCount)))
    {
        return false;
    }

    if (!dispatchReduceBroadPhaseExtentPass(computeContext, sceneState, constants.staticBodyCount,
                                            true))
    {
        return false;
    }

    const std::array staticMortonBindings{
        gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseBuildConstantsBuffer",
                              mBroadPhaseBuildConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseElements", transient.staticBroadPhaseElementsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_GlobalExtent", transient.staticGlobalBroadPhaseExtentBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_MortonCodes", transient.staticMortonCodesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeBroadPhaseBuildConstants(computeContext, staticBuildConstants) ||
        !mMortonCodesPass.dispatch(computeContext, kAltVariant, staticMortonBindings,
                                   dispatchGroupCount(constants.staticBodyCount)))
    {
        return false;
    }

    if (!dispatchRadixSortPass(computeContext, sceneState, constants.staticBodyCount, true))
    {
        return false;
    }

    const std::array staticBvhHierarchyBindings{
        gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseBuildConstantsBuffer",
                              mBroadPhaseBuildConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedMortonCodes", transient.staticMortonCodesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseElements", transient.staticBroadPhaseElementsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BvhNodes", transient.staticBvhBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_BvhConstructionInfos", transient.staticBvhConstructionInfoBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeBroadPhaseBuildConstants(computeContext, staticBuildConstants) ||
        !mBvhHierarchyPass.dispatch(computeContext, kAltVariant, staticBvhHierarchyBindings,
                                    dispatchGroupCount(constants.staticBodyCount)))
    {
        return false;
    }

    const std::array staticBvhBoundsBindings{
        gpu::GpuBufferBinding{"PhysicsRigidBroadPhaseBuildConstantsBuffer",
                              mBroadPhaseBuildConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BvhNodes", transient.staticBvhBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_BvhConstructionInfos", transient.staticBvhConstructionInfoBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return writeBroadPhaseBuildConstants(computeContext, staticBuildConstants) &&
           mBvhBoundingBoxesPass.dispatch(computeContext, kAltVariant, staticBvhBoundsBindings,
                                          dispatchGroupCount(constants.staticBodyCount));
}

bool PhysicsPassDispatcher::finalizeBroadPhasePairs(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t activeMovingCount,
                                                    const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (activeMovingCount == 0u)
    {
        return true;
    }

    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &jointSuppression    = sceneState.persistentJointCollisionSuppression();
    const auto &transient           = sceneState.transientBuffers();

    const std::array countBindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseBodyIndices", transient.activeBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", persistentColliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_JointCollisionSuppressionOffsets",
                              jointSuppression.neighborOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_JointCollisionSuppressionNeighbors",
                              jointSuppression.neighborsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairCountsSphereSphere", transient.pairCountBuffers[0],
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PairCountsSphereBox", transient.pairCountBuffers[1],
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PairCountsSphereCapsule", transient.pairCountBuffers[2],
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PairCountsBoxBox", transient.pairCountBuffers[3],
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PairCountsBoxCapsule", transient.pairCountBuffers[4],
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PairCountsCapsuleCapsule", transient.pairCountBuffers[5],
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeRigidDispatchConstants(computeContext, constants) ||
        !mCountPairsPass.dispatch(computeContext, kAltVariant, countBindings,
                                  dispatchGroupCount(activeMovingCount)))
    {
        return false;
    }

    for (std::uint32_t pairType = 0u; pairType < kRigidPairTypeCount; ++pairType)
    {
        if (!dispatchExclusiveScanWithCpuCount(
                computeContext, sceneState, transient.pairCountBuffers[pairType],
                transient.pairOffsetBuffers[pairType], activeMovingCount))
        {
            return false;
        }
    }

    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairCountsSphereSphere", transient.pairCountBuffers[0],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairCountsSphereBox", transient.pairCountBuffers[1],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairCountsSphereCapsule", transient.pairCountBuffers[2],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairCountsBoxBox", transient.pairCountBuffers[3],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairCountsBoxCapsule", transient.pairCountBuffers[4],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairCountsCapsuleCapsule", transient.pairCountBuffers[5],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsSphereSphere", transient.pairOffsetBuffers[0],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsSphereBox", transient.pairOffsetBuffers[1],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsSphereCapsule", transient.pairOffsetBuffers[2],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsBoxBox", transient.pairOffsetBuffers[3],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsBoxCapsule", transient.pairOffsetBuffers[4],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsCapsuleCapsule", transient.pairOffsetBuffers[5],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeRigidDispatchConstants(computeContext, constants) &&
           mFinalizePairsPass.dispatch(computeContext, kDefaultVariant, finalizeBindings, 1u);
}

bool PhysicsPassDispatcher::emitBroadPhasePairs(Diligent::IDeviceContext *computeContext,
                                                const PhysicsSceneGpuState &sceneState,
                                                std::uint32_t activeMovingCount,
                                                const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (activeMovingCount == 0u)
    {
        return true;
    }

    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &jointSuppression    = sceneState.persistentJointCollisionSuppression();
    const auto &transient           = sceneState.transientBuffers();

    const std::array emitBindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseBodyIndices", transient.activeBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", persistentColliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_JointCollisionSuppressionOffsets",
                              jointSuppression.neighborOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_JointCollisionSuppressionNeighbors",
                              jointSuppression.neighborsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsSphereSphere", transient.pairOffsetBuffers[0],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsSphereBox", transient.pairOffsetBuffers[1],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsSphereCapsule", transient.pairOffsetBuffers[2],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsBoxBox", transient.pairOffsetBuffers[3],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsBoxCapsule", transient.pairOffsetBuffers[4],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PairOffsetsCapsuleCapsule", transient.pairOffsetBuffers[5],
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidatePairs", transient.candidatePairsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeRigidDispatchConstants(computeContext, constants) ||
        !mEmitPairsPass.dispatch(computeContext, kAltVariant, emitBindings,
                                 dispatchGroupCount(activeMovingCount)))
    {
        return false;
    }

    return buildRigidNarrowPhaseChunks(computeContext, sceneState);
}

bool PhysicsPassDispatcher::dispatchBuildRigidNarrowPhaseChunksPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_NarrowPhaseChunks", transient.narrowPhaseChunksBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_NarrowPhaseMeta", transient.narrowPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_NarrowPhaseChunkCounter", transient.narrowPhaseChunkCounterBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return mBuildNarrowPhaseChunksPass.dispatch(computeContext, kDefaultVariant, bindings, 1u);
}

bool PhysicsPassDispatcher::buildRigidNarrowPhaseChunks(Diligent::IDeviceContext *computeContext,
                                                        const PhysicsSceneGpuState &sceneState)
{
    return dispatchBuildRigidNarrowPhaseChunksPass(computeContext, sceneState);
}

bool PhysicsPassDispatcher::prepareRigidIndirectArgs(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_NarrowPhaseMeta", transient.narrowPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ProxyRigidContactMeta", transient.proxyRigidContactMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PhysicsIndirectDispatchArgs", transient.physicsIndirectArgsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return mPrepareRigidIndirectArgsPass.dispatch(computeContext, kDefaultVariant, bindings, 1u);
}

bool PhysicsPassDispatcher::dispatchGenerateRigidContactsPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &persistent          = sceneState.persistentRigidBodies();
    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &transient           = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyScales", persistent.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderContactData", persistentColliders.contactDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidatePairs", transient.candidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_NarrowPhaseChunks", transient.narrowPhaseChunksBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_NarrowPhaseMeta", transient.narrowPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_NarrowPhaseChunkCounter", transient.narrowPhaseChunkCounterBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidContacts", transient.rigidContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mGenerateRigidContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::RigidGenerateContacts));
}

bool PhysicsPassDispatcher::generateRigidContacts(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState)
{
    return dispatchGenerateRigidContactsPass(computeContext, sceneState);
}

bool PhysicsPassDispatcher::dispatchGenerateProxyRigidContactsPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.particleCount == 0u)
    {
        return true;
    }

    const GpuProxyRigidContactMeta zeroMeta{};
    computeContext->UpdateBuffer(sceneState.transientBuffers().proxyRigidContactMetaBuffer, 0u,
                                 sizeof(GpuProxyRigidContactMeta), &zeroMeta,
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    const auto &particles           = sceneState.persistentParticles();
    const auto &rigidBodies         = sceneState.persistentRigidBodies();
    const auto &bodyColliderMapping = sceneState.persistentBodyColliderMapping();
    const auto &colliders           = sceneState.persistentColliders();
    const auto &transient           = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ProxyRigidContactMeta", transient.proxyRigidContactMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass", particles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleRadii", particles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerTypes", particles.ownerTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleOwnerIndices", particles.ownerIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyScales", rigidBodies.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyProxyParticleContactMaterials",
                              rigidBodies.proxyParticleContactMaterialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyColliderRanges", bodyColliderMapping.colliderRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BodyColliderIndices", bodyColliderMapping.colliderIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderContactData", colliders.contactDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCandidatePairs", transient.softRigidCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidContacts", transient.rigidContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mGenerateProxyRigidContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftGenerateRigidContacts));
}

bool PhysicsPassDispatcher::generateProxyRigidContacts(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuParticleDispatchConstants &constants)
{
    return dispatchGenerateProxyRigidContactsPass(computeContext, sceneState, constants);
}

bool PhysicsPassDispatcher::dispatchFinalRigidContactDepenetrationPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &persistent = sceneState.persistentRigidBodies();
    const auto &transient  = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ProxyRigidContactMeta", transient.proxyRigidContactMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                              transient.previousRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyOrientations",
                              transient.previousRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistent.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidContacts", transient.rigidContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mFinalRigidContactDepenetrationPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::RigidSolveContacts));
}

bool PhysicsPassDispatcher::finalRigidContactDepenetration(Diligent::IDeviceContext *computeContext,
                                                           const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    return dispatchFinalRigidContactDepenetrationPass(computeContext, sceneState);
}

bool PhysicsPassDispatcher::dispatchInitRigidContactVelocitiesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return true;
    }
    if (sceneState.candidatePairCapacity() == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ProxyRigidContactMeta", transient.proxyRigidContactMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidContacts", transient.rigidContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateMap",
                              transient.rigidBodyPairAggregateMapBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateActiveCount",
                              transient.rigidBodyPairAggregateActiveCountBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateHeaders",
                              transient.rigidBodyPairAggregateHeadersBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateSlots",
                              transient.rigidBodyPairAggregateSlotsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mInitRigidContactVelocitiesPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::RigidSolveContactVelocities));
}

bool PhysicsPassDispatcher::dispatchClearRigidBodyPairContactAggregatesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t candidatePairCapacity)
{
    if (computeContext == nullptr)
    {
        return true;
    }
    if (candidatePairCapacity == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateMap",
                              transient.rigidBodyPairAggregateMapBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateActiveCount",
                              transient.rigidBodyPairAggregateActiveCountBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateHeaders",
                              transient.rigidBodyPairAggregateHeadersBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mClearRigidBodyPairContactAggregatesPass.dispatch(
        computeContext, kDefaultVariant, bindings, dispatchGroupCount(candidatePairCapacity));
}

bool PhysicsPassDispatcher::dispatchPrepareRigidContactVelocityIndirectArgsPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateActiveCount",
                              transient.rigidBodyPairAggregateActiveCountBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PhysicsIndirectDispatchArgs", transient.physicsIndirectArgsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mPrepareRigidContactVelocityIndirectArgsPass.dispatch(computeContext, kDefaultVariant,
                                                                 bindings, 1u);
}

bool PhysicsPassDispatcher::dispatchSolveRigidContactVelocitiesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return true;
    }
    if (sceneState.candidatePairCapacity() == 0u)
    {
        return true;
    }

    const auto &persistent = sceneState.persistentRigidBodies();
    const auto &transient  = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistent.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateActiveCount",
                              transient.rigidBodyPairAggregateActiveCountBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateHeaders",
                              transient.rigidBodyPairAggregateHeadersBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyPairAggregateSlots",
                              transient.rigidBodyPairAggregateSlotsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveRigidContactVelocitiesPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::RigidSolveContactVelocities));
}

bool PhysicsPassDispatcher::updateRigidDispatchConstants(Diligent::IDeviceContext *computeContext,
                                                         const GpuRigidDispatchConstants &constants)
{
    return writeRigidDispatchConstants(computeContext, constants);
}

bool PhysicsPassDispatcher::dispatchSolveBallJointConstraintsPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t jointCount)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &persistentBodies = sceneState.persistentRigidBodies();
    const auto &persistentJoints = sceneState.persistentJoints();
    const auto &transient        = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"PhysicsRigidJointDispatchConstantsBuffer",
                              mRigidJointDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentBodies.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_BallJoints", persistentJoints.ballJointsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveBallJointConstraintsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                   dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::solveBallJointConstraints(Diligent::IDeviceContext *computeContext,
                                                      const PhysicsSceneGpuState &sceneState)
{
    const std::uint32_t jointCount = sceneState.ballJointCount();
    if (computeContext == nullptr)
    {
        return false;
    }
    if (jointCount == 0u)
    {
        return true;
    }

    const GpuRigidJointDispatchConstants jointConstants{jointCount, 0u, 0u, 0u};
    return writeRigidJointDispatchConstants(computeContext, jointConstants) &&
           dispatchSolveBallJointConstraintsPass(computeContext, sceneState, jointCount);
}

bool PhysicsPassDispatcher::dispatchSolveHingeJointConstraintsPass(
    Diligent::IDeviceContext *computeContext, gpu::GpuComputePass &pass,
    const PhysicsSceneGpuState &sceneState, Diligent::IBuffer *jointIndicesBuffer,
    std::uint32_t jointCount)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &persistentBodies = sceneState.persistentRigidBodies();
    const auto &persistentJoints = sceneState.persistentJoints();
    const auto &transient        = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"PhysicsRigidJointDispatchConstantsBuffer",
                              mRigidJointDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentBodies.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_HingeJoints", persistentJoints.hingeJointsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_HingeJointIndices", jointIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_HingeJointLambdas0123", transient.hingeJointLambdas0123Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_HingeJointLambdas45", transient.hingeJointLambdas45Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return pass.dispatch(computeContext, kDefaultVariant, bindings, dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::solveHingeJointConstraints(Diligent::IDeviceContext *computeContext,
                                                       const PhysicsSceneGpuState &sceneState)
{
    const std::uint32_t passiveJointCount       = sceneState.hingePassiveJointCount();
    const std::uint32_t positionDriveJointCount = sceneState.hingePositionDriveJointCount();
    if (computeContext == nullptr)
    {
        return false;
    }
    if (passiveJointCount == 0u && positionDriveJointCount == 0u)
    {
        return true;
    }

    const auto &persistentJoints = sceneState.persistentJoints();
    if (passiveJointCount > 0u)
    {
        const GpuRigidJointDispatchConstants jointConstants{passiveJointCount, 0u, 0u, 0u};
        if (!writeRigidJointDispatchConstants(computeContext, jointConstants) ||
            !dispatchSolveHingeJointConstraintsPass(
                computeContext, mSolveHingeJointConstraintsPassivePass, sceneState,
                persistentJoints.hingePassiveJointIndicesBuffer, passiveJointCount))
        {
            return false;
        }
    }

    if (positionDriveJointCount > 0u)
    {
        const GpuRigidJointDispatchConstants jointConstants{positionDriveJointCount, 0u, 0u, 0u};
        if (!writeRigidJointDispatchConstants(computeContext, jointConstants) ||
            !dispatchSolveHingeJointConstraintsPass(
                computeContext, mSolveHingeJointConstraintsTargetPositionPass, sceneState,
                persistentJoints.hingePositionDriveJointIndicesBuffer, positionDriveJointCount))
        {
            return false;
        }
    }

    return true;
}

bool PhysicsPassDispatcher::dispatchSolveSliderJointConstraintsPass(
    Diligent::IDeviceContext *computeContext, gpu::GpuComputePass &pass,
    const PhysicsSceneGpuState &sceneState, Diligent::IBuffer *jointIndicesBuffer,
    std::uint32_t jointCount)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &persistentBodies = sceneState.persistentRigidBodies();
    const auto &persistentJoints = sceneState.persistentJoints();
    const auto &transient        = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"PhysicsRigidJointDispatchConstantsBuffer",
                              mRigidJointDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentBodies.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SliderJoints", persistentJoints.sliderJointsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SliderJointIndices", jointIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SliderJointLambdas0123", transient.sliderJointLambdas0123Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SliderJointLambdas45", transient.sliderJointLambdas45Buffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return pass.dispatch(computeContext, kDefaultVariant, bindings, dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::solveSliderJointConstraints(Diligent::IDeviceContext *computeContext,
                                                        const PhysicsSceneGpuState &sceneState)
{
    const std::uint32_t passiveJointCount       = sceneState.sliderPassiveJointCount();
    const std::uint32_t positionDriveJointCount = sceneState.sliderPositionDriveJointCount();
    if (computeContext == nullptr)
    {
        return false;
    }
    if (passiveJointCount == 0u && positionDriveJointCount == 0u)
    {
        return true;
    }

    const auto &persistentJoints = sceneState.persistentJoints();
    if (passiveJointCount > 0u)
    {
        const GpuRigidJointDispatchConstants jointConstants{passiveJointCount, 0u, 0u, 0u};
        if (!writeRigidJointDispatchConstants(computeContext, jointConstants) ||
            !dispatchSolveSliderJointConstraintsPass(
                computeContext, mSolveSliderJointConstraintsPassivePass, sceneState,
                persistentJoints.sliderPassiveJointIndicesBuffer, passiveJointCount))
        {
            return false;
        }
    }

    if (positionDriveJointCount > 0u)
    {
        const GpuRigidJointDispatchConstants jointConstants{positionDriveJointCount, 0u, 0u, 0u};
        if (!writeRigidJointDispatchConstants(computeContext, jointConstants) ||
            !dispatchSolveSliderJointConstraintsPass(
                computeContext, mSolveSliderJointConstraintsTargetPositionPass, sceneState,
                persistentJoints.sliderPositionDriveJointIndicesBuffer, positionDriveJointCount))
        {
            return false;
        }
    }

    return true;
}

bool PhysicsPassDispatcher::solveRigidParticleAttachmentConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t constraintCount, const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (constraintCount == 0u)
    {
        return true;
    }

    const auto &persistentBodies    = sceneState.persistentRigidBodies();
    const auto &persistentParticles = sceneState.persistentParticles();
    const auto &persistentCables    = sceneState.persistentRoutedCables();
    const auto &transient           = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass",
                              persistentParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentBodies.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidParticleAttachments",
                              persistentCables.rigidParticleAttachmentsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidParticleAttachmentLambdas",
                              transient.rigidParticleAttachmentLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticlePositionCorrections",
                              transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    GpuRigidDispatchConstants dispatchConstants = constants;
    dispatchConstants.reserved0 = constraintCount;
    dispatchConstants.reserved1 = sceneState.sceneView().soft.particles.count;
    return writeRigidDispatchConstants(computeContext, dispatchConstants) &&
           mSolveRigidParticleAttachmentConstraintsPass.dispatch(
               computeContext, kDefaultVariant, bindings, dispatchGroupCount(constraintCount));
}

bool PhysicsPassDispatcher::solveStrandRigidAttachmentConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t constraintCount, const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (constraintCount == 0u)
    {
        return true;
    }

    const auto &persistentParticles = sceneState.persistentParticles();
    const auto &persistentCables    = sceneState.persistentRoutedCables();
    const auto &softTopology        = sceneState.persistentSoftTopology();
    const auto &transient           = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticlePositionsInvMass",
                              persistentParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegments", softTopology.strandSegmentsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandSegmentStates", softTopology.strandSegmentStatesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandRigidAttachments",
                              persistentCables.strandRigidAttachmentsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_StrandRigidAttachmentLambdas",
                              transient.strandRigidAttachmentLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_StrandRigidAttachmentCorrections",
                              transient.strandRigidAttachmentCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticlePositionCorrections",
                              transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    GpuRigidDispatchConstants dispatchConstants = constants;
    dispatchConstants.reserved0 = constraintCount;
    dispatchConstants.reserved1 = sceneState.sceneView().soft.particles.count;
    return writeRigidDispatchConstants(computeContext, dispatchConstants) &&
           mSolveStrandRigidAttachmentConstraintsPass.dispatch(
               computeContext, kDefaultVariant, bindings, dispatchGroupCount(constraintCount));
}

bool PhysicsPassDispatcher::solveRoutedCableConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t routedCableCount, const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (routedCableCount == 0u)
    {
        return true;
    }

    const auto &persistentBodies = sceneState.persistentRigidBodies();
    const auto &persistentCables = sceneState.persistentRoutedCables();
    const auto &transient        = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentBodies.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RoutedCableConstraints",
                              persistentCables.descriptorsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RoutedCableRoutePoints",
                              persistentCables.routePointsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RoutedCableLambdas", transient.routedCableLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeRigidDispatchConstants(computeContext, constants) &&
           mSolveRoutedCableConstraintsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                     dispatchGroupCount(routedCableCount));
}

bool PhysicsPassDispatcher::solveRigidDistanceConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t constraintCount, const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (constraintCount == 0u)
    {
        return true;
    }

    const auto &persistentBodies = sceneState.persistentRigidBodies();
    const auto &persistentCables = sceneState.persistentRoutedCables();
    const auto &transient        = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentBodies.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidDistanceConstraints",
                              persistentCables.rigidDistanceConstraintsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidDistanceConstraintLambdas",
                              transient.rigidDistanceConstraintLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    GpuRigidDispatchConstants dispatchConstants = constants;
    dispatchConstants.reserved0 = constraintCount;
    return writeRigidDispatchConstants(computeContext, dispatchConstants) &&
           mSolveRigidDistanceConstraintsPass.dispatch(
               computeContext, kDefaultVariant, bindings, dispatchGroupCount(constraintCount));
}

bool PhysicsPassDispatcher::dispatchSolveHingeJointVelocityTargetsPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    Diligent::IBuffer *jointIndicesBuffer, std::uint32_t jointCount)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &persistentBodies = sceneState.persistentRigidBodies();
    const auto &persistentJoints = sceneState.persistentJoints();
    const auto &transient        = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"PhysicsRigidJointDispatchConstantsBuffer",
                              mRigidJointDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentBodies.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_HingeJoints", persistentJoints.hingeJointsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_HingeJointIndices", jointIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveHingeJointTargetVelocitiesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                         dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::dispatchSolveSliderJointVelocityTargetsPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    Diligent::IBuffer *jointIndicesBuffer, std::uint32_t jointCount)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &persistentBodies = sceneState.persistentRigidBodies();
    const auto &persistentJoints = sceneState.persistentJoints();
    const auto &transient        = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"PhysicsRigidJointDispatchConstantsBuffer",
                              mRigidJointDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyInverseInertiaLocal",
                              persistentBodies.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SliderJoints", persistentJoints.sliderJointsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SliderJointIndices", jointIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveSliderJointTargetVelocitiesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                          dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::solveHingeJointTargetVelocities(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    const std::uint32_t jointCount = sceneState.hingeVelocityDriveJointCount();
    if (computeContext == nullptr)
    {
        return false;
    }
    if (jointCount == 0u)
    {
        return true;
    }

    const auto &persistentJoints = sceneState.persistentJoints();
    const GpuRigidJointDispatchConstants jointConstants{jointCount, 0u, 0u, 0u};
    return writeRigidJointDispatchConstants(computeContext, jointConstants) &&
           dispatchSolveHingeJointVelocityTargetsPass(
               computeContext, sceneState, persistentJoints.hingeVelocityDriveJointIndicesBuffer,
               jointCount);
}

bool PhysicsPassDispatcher::solveSliderJointTargetVelocities(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    const std::uint32_t jointCount = sceneState.sliderVelocityDriveJointCount();
    if (computeContext == nullptr)
    {
        return false;
    }
    if (jointCount == 0u)
    {
        return true;
    }

    const auto &persistentJoints = sceneState.persistentJoints();
    const GpuRigidJointDispatchConstants jointConstants{jointCount, 0u, 0u, 0u};
    return writeRigidJointDispatchConstants(computeContext, jointConstants) &&
           dispatchSolveSliderJointVelocityTargetsPass(
               computeContext, sceneState, persistentJoints.sliderVelocityDriveJointIndicesBuffer,
               jointCount);
}

bool PhysicsPassDispatcher::dispatchApplyRigidCorrectionsPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t rigidBodyCount)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (rigidBodyCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTypes",
                              sceneState.persistentRigidBodies().bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mApplyRigidCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                               dispatchGroupCount(rigidBodyCount));
}

bool PhysicsPassDispatcher::applyRigidCorrections(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState,
                                                  std::uint32_t rigidBodyCount,
                                                  const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (rigidBodyCount == 0u)
    {
        return true;
    }

    return writeRigidDispatchConstants(computeContext, constants) &&
           dispatchApplyRigidCorrectionsPass(computeContext, sceneState, rigidBodyCount);
}

bool PhysicsPassDispatcher::updateRigidVelocities(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState,
                                                  std::uint32_t bodyCount,
                                                  const GpuRigidDispatchConstants &constants)
{
    if (bodyCount == 0u)
    {
        return true;
    }

    const auto &transient  = sceneState.transientBuffers();
    const auto &persistent = sceneState.persistentRigidBodies();

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                              transient.previousRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyOrientations",
                              transient.previousRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeRigidDispatchConstants(computeContext, constants) &&
           mUpdateRigidVelocitiesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                               dispatchGroupCount(bodyCount));
}

bool PhysicsPassDispatcher::resetRigidContactVelocityAggregates(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuRigidDispatchConstants &constants)
{
    if (constants.candidatePairCapacity == 0u)
    {
        return true;
    }

    return writeRigidDispatchConstants(computeContext, constants) &&
           dispatchClearRigidBodyPairContactAggregatesPass(computeContext, sceneState,
                                                           constants.candidatePairCapacity) &&
           dispatchPrepareRigidContactVelocityIndirectArgsPass(computeContext, sceneState);
}

bool PhysicsPassDispatcher::initRigidContactVelocities(Diligent::IDeviceContext *computeContext,
                                                       const PhysicsSceneGpuState &sceneState,
                                                       const GpuRigidDispatchConstants &constants)
{
    if (constants.candidatePairCapacity == 0u)
    {
        return true;
    }

    return writeRigidDispatchConstants(computeContext, constants) &&
           dispatchClearRigidBodyPairContactAggregatesPass(computeContext, sceneState,
                                                           constants.candidatePairCapacity) &&
           dispatchInitRigidContactVelocitiesPass(computeContext, sceneState) &&
           dispatchPrepareRigidContactVelocityIndirectArgsPass(computeContext, sceneState);
}

bool PhysicsPassDispatcher::solveRigidContactVelocities(Diligent::IDeviceContext *computeContext,
                                                        const PhysicsSceneGpuState &sceneState,
                                                        std::uint32_t rigidBodyCount,
                                                        std::uint32_t rigidContactIterations,
                                                        std::uint32_t rigidJointIterations,
                                                        const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    const std::uint32_t iterations = std::max(rigidContactIterations, rigidJointIterations);
    if (rigidBodyCount == 0u || iterations == 0u)
    {
        return true;
    }
    if (!writeRigidDispatchConstants(computeContext, constants))
    {
        return false;
    }

    const std::uint32_t hingeVelocityJointCount  = sceneState.hingeVelocityDriveJointCount();
    const std::uint32_t sliderVelocityJointCount = sceneState.sliderVelocityDriveJointCount();
    const auto &transient                        = sceneState.transientBuffers();
    const auto &persistent                       = sceneState.persistentRigidBodies();
    const std::array applyBindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                              transient.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                              transient.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!mApplyRigidContactVelocitiesPass.bindVariant(kDefaultVariant, applyBindings))
    {
        return false;
    }

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        if (iteration < rigidContactIterations &&
            !dispatchSolveRigidContactVelocitiesPass(computeContext, sceneState))
        {
            return false;
        }

        if (iteration < rigidJointIterations)
        {
            if (hingeVelocityJointCount > 0u &&
                !solveHingeJointTargetVelocities(computeContext, sceneState))
            {
                return false;
            }
            if (sliderVelocityJointCount > 0u &&
                !solveSliderJointTargetVelocities(computeContext, sceneState))
            {
                return false;
            }
        }

        if (!mApplyRigidContactVelocitiesPass.dispatch(
                computeContext, kDefaultVariant, applyBindings, dispatchGroupCount(rigidBodyCount)))
        {
            return false;
        }
    }

    return true;
}

bool PhysicsPassDispatcher::recreateSceneBindingVariants()
{
    return mRigidPredictPass.forceRecreateAllVariants() &&
           mSoftPredictPass.forceRecreateAllVariants() &&
           mSyncRigidProxyParticlesPass.forceRecreateAllVariants() &&
           mBuildParticleBroadPhaseEntriesPass.forceRecreateAllVariants() &&
           mBuildParticleBroadPhaseKeysPass.forceRecreateAllVariants() &&
           mMarkParticleCellRangeStartsPass.forceRecreateAllVariants() &&
           mClearParticleCellRangesPass.forceRecreateAllVariants() &&
           mBuildParticleCellRangesPass.forceRecreateAllVariants() &&
           mCountParticleParticleCandidatePairsPass.forceRecreateAllVariants() &&
           mFinalizeParticleParticleCandidatePairsPass.forceRecreateAllVariants() &&
           mEmitParticleParticleCandidatePairsPass.forceRecreateAllVariants() &&
           mCountParticleRigidCandidatePairsPass.forceRecreateAllVariants() &&
           mFinalizeParticleRigidCandidatePairsPass.forceRecreateAllVariants() &&
           mEmitParticleRigidCandidatePairsPass.forceRecreateAllVariants() &&
           mCountFluidBoundaryCandidatePairsPass.forceRecreateAllVariants() &&
           mFinalizeFluidBoundaryCandidatePairsPass.forceRecreateAllVariants() &&
           mEmitFluidBoundaryCandidatePairsPass.forceRecreateAllVariants() &&
           mGenerateParticleExplicitContactsPass.forceRecreateAllVariants() &&
           mGenerateParticleRigidContactsPass.forceRecreateAllVariants() &&
           mPrepareExplicitContactScanPass.forceRecreateAllVariants() &&
           mPrepareRigidContactScanPass.forceRecreateAllVariants() &&
           mPrepareParticleCandidateIndirectArgsPass.forceRecreateAllVariants() &&
           mPrepareParticleActiveIndirectArgsPass.forceRecreateAllVariants() &&
           mFinalizeActiveParticleExplicitContactsPass.forceRecreateAllVariants() &&
           mCompactActiveParticleExplicitContactsPass.forceRecreateAllVariants() &&
           mFinalizeActiveParticleRigidContactsPass.forceRecreateAllVariants() &&
           mCompactActiveParticleRigidContactsPass.forceRecreateAllVariants() &&
           mClearSoftConstraintStatePass.forceRecreateAllVariants() &&
           mClearRigidParticleAttachmentConstraintStatePass.forceRecreateAllVariants() &&
           mClearStrandRigidAttachmentConstraintStatePass.forceRecreateAllVariants() &&
           mClearRigidDistanceConstraintStatePass.forceRecreateAllVariants() &&
           mClearRoutedCableConstraintStatePass.forceRecreateAllVariants() &&
           mClearSuturingCandidatesPass.forceRecreateAllVariants() &&
           mGatherSuturingCandidatesPass.forceRecreateAllVariants() &&
           mClassifySuturingParticlesPass.forceRecreateAllVariants() &&
           mUpdateSuturingTipPathsPass.forceRecreateAllVariants() &&
           mAssignSuturingInsideParticlesPass.forceRecreateAllVariants() &&
           mSolveSuturingNodePathConstraintsPass.forceRecreateAllVariants() &&
           mSolveSoftEdgeConstraintsPass.forceRecreateAllVariants() &&
           mSolveSoftBendConstraintsPass.forceRecreateAllVariants() &&
           mSolveSoftTetConstraintsPass.forceRecreateAllVariants() &&
           mApplySoftEdgeCorrectionsPass.forceRecreateAllVariants() &&
           mApplySoftBendCorrectionsPass.forceRecreateAllVariants() &&
           mApplySoftTetCorrectionsPass.forceRecreateAllVariants() &&
           mSolveStrandSegmentConstraintsPass.forceRecreateAllVariants() &&
           mApplyStrandSegmentCorrectionsPass.forceRecreateAllVariants() &&
           mSolveStrandJointConstraintsPass.forceRecreateAllVariants() &&
           mApplyStrandJointCorrectionsPass.forceRecreateAllVariants() &&
           mApplyStrandRigidAttachmentCorrectionsPass.forceRecreateAllVariants() &&
           mSolveStrandDistanceConstraintsPass.forceRecreateAllVariants() &&
           mApplyStrandDistanceCorrectionsPass.forceRecreateAllVariants() &&
           mSolveParticleExplicitContactsPass.forceRecreateAllVariants() &&
           mSolveParticleRigidContactsPass.forceRecreateAllVariants() &&
           mApplyParticlePositionCorrectionsPass.forceRecreateAllVariants() &&
           mUpdateParticleVelocitiesPass.forceRecreateAllVariants() &&
           mBuildFluidNeighborPairsPass.forceRecreateAllVariants() &&
           mComputeFluidDensityConstraintsPass.forceRecreateAllVariants() &&
           mComputeFluidDeltaPositionsPass.forceRecreateAllVariants() &&
           mApplyFluidDeltaPositionsPass.forceRecreateAllVariants() &&
           mClampFluidBoundaryPass.forceRecreateAllVariants() &&
           mProjectFluidBoundaryVelocitiesPass.forceRecreateAllVariants() &&
           mComputeFluidVorticityPass.forceRecreateAllVariants() &&
           mApplyFluidVorticityConfinementPass.forceRecreateAllVariants() &&
           mBuildFluidRenderAnisotropyPass.forceRecreateAllVariants() &&
           mSolveParticleContactVelocitiesPass.forceRecreateAllVariants() &&
           mSolveParticleRigidContactVelocitiesPass.forceRecreateAllVariants() &&
           mApplyParticleContactVelocitiesPass.forceRecreateAllVariants() &&
           mUpdateSoftTriangleNormalsPass.forceRecreateAllVariants() &&
           mUpdateSoftRenderNormalsPass.forceRecreateAllVariants() &&
           mUpdateCurveRenderDataPass.forceRecreateAllVariants() &&
           mUpdateSoftBodyBoundsPass.forceRecreateAllVariants() &&
           mFinalizeSoftBodyBoundsPass.forceRecreateAllVariants() &&
           mClearRigidCorrectionsPass.forceRecreateAllVariants() &&
           mUpdateRigidWorldAabbsPass.forceRecreateAllVariants() &&
           mScanBlockPass.forceRecreateAllVariants() &&
           mScanAddOffsetsPass.forceRecreateAllVariants() &&
           mCompactBodySetPass.forceRecreateAllVariants() &&
           mFinalizeActiveBodiesPass.forceRecreateAllVariants() &&
           mBuildBroadPhaseElementsPass.forceRecreateAllVariants() &&
           mReduceExtentElementsPass.forceRecreateAllVariants() &&
           mReduceExtentExtentsPass.forceRecreateAllVariants() &&
           mMortonCodesPass.forceRecreateAllVariants() &&
           mRadixClassifyPass.forceRecreateAllVariants() &&
           mRadixFinalizePass.forceRecreateAllVariants() &&
           mRadixScatterPass.forceRecreateAllVariants() &&
           mBvhHierarchyPass.forceRecreateAllVariants() &&
           mBvhBoundingBoxesPass.forceRecreateAllVariants() &&
           mCountPairsPass.forceRecreateAllVariants() &&
           mFinalizePairsPass.forceRecreateAllVariants() &&
           mEmitPairsPass.forceRecreateAllVariants() &&
           mBuildNarrowPhaseChunksPass.forceRecreateAllVariants() &&
           mPrepareRigidIndirectArgsPass.forceRecreateAllVariants() &&
           mGenerateRigidContactsPass.forceRecreateAllVariants() &&
           mGenerateProxyRigidContactsPass.forceRecreateAllVariants() &&
           mFinalRigidContactDepenetrationPass.forceRecreateAllVariants() &&
           mClearRigidBodyPairContactAggregatesPass.forceRecreateAllVariants() &&
           mInitRigidContactVelocitiesPass.forceRecreateAllVariants() &&
           mPrepareRigidContactVelocityIndirectArgsPass.forceRecreateAllVariants() &&
           mSolveRigidContactVelocitiesPass.forceRecreateAllVariants() &&
           mSolveBallJointConstraintsPass.forceRecreateAllVariants() &&
           mSolveHingeJointConstraintsPassivePass.forceRecreateAllVariants() &&
           mSolveHingeJointConstraintsTargetPositionPass.forceRecreateAllVariants() &&
           mSolveSliderJointConstraintsPassivePass.forceRecreateAllVariants() &&
           mSolveSliderJointConstraintsTargetPositionPass.forceRecreateAllVariants() &&
           mSolveRigidParticleAttachmentConstraintsPass.forceRecreateAllVariants() &&
           mSolveStrandRigidAttachmentConstraintsPass.forceRecreateAllVariants() &&
           mSolveRigidDistanceConstraintsPass.forceRecreateAllVariants() &&
           mSolveRoutedCableConstraintsPass.forceRecreateAllVariants() &&
           mClearHingeJointConstraintStatePass.forceRecreateAllVariants() &&
           mClearSliderJointConstraintStatePass.forceRecreateAllVariants() &&
           mSolveHingeJointTargetVelocitiesPass.forceRecreateAllVariants() &&
           mSolveSliderJointTargetVelocitiesPass.forceRecreateAllVariants() &&
           mApplyRigidCorrectionsPass.forceRecreateAllVariants() &&
           mUpdateRigidVelocitiesPass.forceRecreateAllVariants() &&
           mApplyRigidContactVelocitiesPass.forceRecreateAllVariants();
}

} // namespace cressim::neo::physics
