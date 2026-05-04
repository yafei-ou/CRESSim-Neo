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

GpuPhysicsScanConstants makeScanConstants(std::uint32_t elementCount)
{
    GpuPhysicsScanConstants result{};
    result.elementCount = elementCount;
    return result;
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

bool PhysicsPassDispatcher::writeSoftDispatchConstants(Diligent::IDeviceContext *computeContext,
                                                       const GpuSoftDispatchConstants &constants)
{
    return writeConstantsBuffer(computeContext, mSoftDispatchConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeSoftRenderDispatchConstants(
    Diligent::IDeviceContext *computeContext, const GpuSoftRenderDispatchConstants &constants)
{
    return writeConstantsBuffer(computeContext, mSoftRenderDispatchConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeScanConstants(Diligent::IDeviceContext *computeContext,
                                               const GpuPhysicsScanConstants &constants)
{
    return writeConstantsBuffer(computeContext, mScanConstantsBuffer, &constants,
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
        !initPass(mBuildParticleBroadPhaseEntriesPass, kBuildParticleBroadPhaseEntries) ||
        !initPass(mBuildParticleBroadPhaseKeysPass, kBuildParticleBroadPhaseKeys) ||
        !initPass(mMarkParticleCellRangeStartsPass, kMarkParticleCellRangeStarts) ||
        !initPass(mClearParticleCellRangesPass, kClearParticleCellRanges) ||
        !initPass(mBuildParticleCellRangesPass, kBuildParticleCellRanges) ||
        !initPass(mCountSoftSoftCandidatePairsPass, kCountSoftSoftCandidatePairs) ||
        !initPass(mFinalizeSoftSoftCandidatePairsPass, kFinalizeSoftSoftCandidatePairs) ||
        !initPass(mEmitSoftSoftCandidatePairsPass, kEmitSoftSoftCandidatePairs) ||
        !initPass(mCountSoftRigidCandidatePairsPass, kCountSoftRigidCandidatePairs) ||
        !initPass(mFinalizeSoftRigidCandidatePairsPass, kFinalizeSoftRigidCandidatePairs) ||
        !initPass(mEmitSoftRigidCandidatePairsPass, kEmitSoftRigidCandidatePairs) ||
        !initPass(mGenerateSoftContactsPass, kGenerateSoftContacts) ||
        !initPass(mGenerateSoftRigidContactsPass, kGenerateSoftRigidContacts) ||
        !initPass(mPrepareSoftCandidateIndirectArgsPass, kPrepareSoftCandidateIndirectArgs) ||
        !initPass(mPrepareSoftActiveIndirectArgsPass, kPrepareSoftActiveIndirectArgs) ||
        !initPass(mFinalizeActiveSoftContactsPass, kFinalizeActiveSoftContacts) ||
        !initPass(mCompactActiveSoftContactsPass, kCompactActiveSoftContacts) ||
        !initPass(mFinalizeActiveSoftRigidContactsPass, kFinalizeActiveSoftRigidContacts) ||
        !initPass(mCompactActiveSoftRigidContactsPass, kCompactActiveSoftRigidContacts) ||
        !initPass(mClearSoftConstraintStatePass, kClearSoftConstraintState) ||
        !initPass(mSolveSoftEdgeConstraintsPass, kSolveSoftEdgeConstraints) ||
        !initPass(mSolveSoftTetConstraintsPass, kSolveSoftTetConstraints) ||
        !initPass(mApplySoftEdgeCorrectionsPass, kApplySoftEdgeCorrections) ||
        !initPass(mApplySoftTetCorrectionsPass, kApplySoftTetCorrections) ||
        !initPass(mSolveSoftContactsPass, kSolveSoftContacts) ||
        !initPass(mSolveSoftRigidContactsPass, kSolveSoftRigidContacts) ||
        !initPass(mApplySoftPositionCorrectionsPass, kApplySoftPositionCorrections) ||
        !initPass(mUpdateSoftVelocitiesPass, kUpdateSoftVelocities) ||
        !initPass(mSolveSoftContactVelocitiesPass, kSolveSoftContactVelocities) ||
        !initPass(mSolveSoftRigidContactVelocitiesPass, kSolveSoftRigidContactVelocities) ||
        !initPass(mApplySoftContactVelocitiesPass, kApplySoftContactVelocities) ||
        !initPass(mUpdateSoftTriangleNormalsPass, kUpdateSoftTriangleNormals) ||
        !initPass(mUpdateSoftRenderNormalsPass, kUpdateSoftRenderNormals) ||
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
        !initPass(mSolveRigidContactConstraintsPass, kSolveRigidContactConstraints) ||
        !initPass(mSolveBallJointConstraintsPass, kSolveBallJointConstraints) ||
        !initPass(mSolveHingeJointConstraintsPassivePass,
                  kSolveHingeJointConstraintsPassive) ||
        !initPass(mSolveHingeJointConstraintsTargetPositionPass,
                  kSolveHingeJointConstraintsTargetPosition) ||
        !initPass(mSolveSliderJointConstraintsPassivePass,
                  kSolveSliderJointConstraintsPassive) ||
        !initPass(mSolveSliderJointConstraintsTargetPositionPass,
                  kSolveSliderJointConstraintsTargetPosition) ||
        !initPass(mSolveHingeJointTargetVelocitiesPass,
                  kSolveHingeJointTargetVelocities) ||
        !initPass(mSolveSliderJointTargetVelocitiesPass,
                  kSolveSliderJointTargetVelocities) ||
        !initPass(mClearRigidCorrectionsPass, kClearRigidCorrections) ||
        !initPass(mApplyRigidCorrectionsPass, kApplyRigidCorrections) ||
        !initPass(mUpdateRigidVelocitiesPass, kUpdateRigidVelocities) ||
        !initPass(mSolveRigidContactVelocitiesPass, kSolveRigidContactVelocities) ||
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
                                 sizeof(GpuSoftDispatchConstants), mSoftDispatchConstantsBuffer) &&
           createConstantsBuffer("CRESSimNeo.Physics.SoftRenderDispatchConstants",
                                 sizeof(GpuSoftRenderDispatchConstants),
                                 mSoftRenderDispatchConstantsBuffer) &&
           createConstantsBuffer("CRESSimNeo.Physics.ScanConstants",
                                 sizeof(GpuPhysicsScanConstants), mScanConstantsBuffer) &&
           createConstantsBuffer("CRESSimNeo.Physics.RadixConstants",
                                 sizeof(GpuPhysicsRadixConstants), mRadixConstantsBuffer) &&
           createConstantsBuffer("CRESSimNeo.Physics.BroadPhaseBuildConstants",
                                 sizeof(GpuBroadPhaseBuildConstants),
                                 mBroadPhaseBuildConstantsBuffer) &&
           createConstantsBuffer("CRESSimNeo.Physics.BroadPhaseReductionConstants",
                                 sizeof(GpuBroadPhaseReductionConstants),
                                 mBroadPhaseReductionConstantsBuffer);
}

bool PhysicsPassDispatcher::dispatchScanBlockPass(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &,
                                                  Diligent::IBuffer *input,
                                                  Diligent::IBuffer *output,
                                                  Diligent::IBuffer *blockSums, std::uint32_t count)
{
    if (count == 0u)
    {
        return false;
    }

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsScanConstantsBuffer", mScanConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScanInput", input, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScanOutput", output, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_BlockSums", blockSums, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const GpuPhysicsScanConstants scanConstants = makeScanConstants(count);
    return writeScanConstants(computeContext, scanConstants) &&
           mScanBlockPass.dispatch(computeContext, kDefaultVariant, bindings,
                                   dispatchGroupCount(count));
}

bool PhysicsPassDispatcher::dispatchScanAddOffsetsPass(Diligent::IDeviceContext *computeContext,
                                                       Diligent::IBuffer *output,
                                                       Diligent::IBuffer *scannedBlockOffsets,
                                                       std::uint32_t count)
{
    if (count == 0u)
    {
        return false;
    }

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsScanConstantsBuffer", mScanConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScannedBlockOffsets", scannedBlockOffsets,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ScanOutput", output, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const GpuPhysicsScanConstants scanConstants = makeScanConstants(count);
    return writeScanConstants(computeContext, scanConstants) &&
           mScanAddOffsetsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                        dispatchGroupCount(count));
}

bool PhysicsPassDispatcher::dispatchExclusiveScanPass(Diligent::IDeviceContext *computeContext,
                                                      const PhysicsSceneGpuState &sceneState,
                                                      Diligent::IBuffer *input,
                                                      Diligent::IBuffer *output,
                                                      std::uint32_t count,
                                                      std::uint32_t recursionLevel)
{
    if (count == 0u)
    {
        return true;
    }

    const auto &transientState = sceneState.transientBuffers();
    if (recursionLevel >= transientState.scanBlockSumsBuffers.size() ||
        recursionLevel >= transientState.scanScannedBlockSumsBuffers.size())
    {
        return false;
    }

    Diligent::IBuffer *blockSums = transientState.scanBlockSumsBuffers[recursionLevel];
    Diligent::IBuffer *scannedBlockSums =
        transientState.scanScannedBlockSumsBuffers[recursionLevel];
    if (!dispatchScanBlockPass(computeContext, sceneState, input, output, blockSums, count))
    {
        return false;
    }

    const std::uint32_t groupCount = dispatchGroupCount(count);
    if (groupCount <= 1u)
    {
        return true;
    }

    if (!dispatchExclusiveScanPass(computeContext, sceneState, blockSums, scannedBlockSums,
                                   groupCount, recursionLevel + 1u))
    {
        return false;
    }

    return dispatchScanAddOffsetsPass(computeContext, output, scannedBlockSums, count);
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
                                        std::uint32_t softParticleCount,
                                        const GpuSoftDispatchConstants &constants)
{
    if (softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftParticlePreviousPositions",
                              softParticles.previousPositionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mSoftPredictPass.dispatch(computeContext, kDefaultVariant, bindings,
                                     dispatchGroupCount(softParticleCount));
}

bool PhysicsPassDispatcher::buildParticleBroadPhaseEntries(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t totalParticleLikeCount, const GpuSoftDispatchConstants &constants)
{
    if (totalParticleLikeCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleOwningSoftBodyIndices",
                              softParticles.owningSoftBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseEntries",
                              transient.particleBroadPhaseEntriesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mBuildParticleBroadPhaseEntriesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                        dispatchGroupCount(totalParticleLikeCount));
}

bool PhysicsPassDispatcher::buildParticleBroadPhaseKeys(Diligent::IDeviceContext *computeContext,
                                                        const PhysicsSceneGpuState &sceneState,
                                                        std::uint32_t totalParticleLikeCount,
                                                        const GpuSoftDispatchConstants &constants)
{
    if (totalParticleLikeCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseEntries",
                              transient.particleBroadPhaseEntriesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseKeys", transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mBuildParticleBroadPhaseKeysPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                     dispatchGroupCount(totalParticleLikeCount));
}

bool PhysicsPassDispatcher::markParticleCellRangeStarts(Diligent::IDeviceContext *computeContext,
                                                        const PhysicsSceneGpuState &sceneState,
                                                        std::uint32_t totalParticleLikeCount,
                                                        const GpuSoftDispatchConstants &constants)
{
    if (totalParticleLikeCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedParticleBroadPhaseKeys",
                              transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRangeStartFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
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
                                                    const GpuSoftDispatchConstants &constants)
{
    if (cellRangeCapacity == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRanges", transient.particleCellRangesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mClearParticleCellRangesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(cellRangeCapacity));
}

bool PhysicsPassDispatcher::buildParticleCellRanges(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t totalParticleLikeCount,
                                                    const GpuSoftDispatchConstants &constants)
{
    if (totalParticleLikeCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    if (!markParticleCellRangeStarts(computeContext, sceneState, totalParticleLikeCount,
                                     constants) ||
        !dispatchExclusiveScanPass(computeContext, sceneState, transient.softRadixBitFlagsBuffer,
                                   transient.softRadixBitOffsetsBuffer, totalParticleLikeCount))
    {
        return false;
    }

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
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

    return writeSoftDispatchConstants(computeContext, constants) &&
           mBuildParticleCellRangesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(totalParticleLikeCount));
}

bool PhysicsPassDispatcher::clearSoftNeighborMeta(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const GpuSoftNeighborMeta zeroMeta{};
    computeContext->UpdateBuffer(sceneState.transientBuffers().softNeighborMetaBuffer, 0u,
                                 sizeof(GpuSoftNeighborMeta), &zeroMeta,
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return true;
}

bool PhysicsPassDispatcher::buildSoftSoftCandidatePairs(Diligent::IDeviceContext *computeContext,
                                                        const PhysicsSceneGpuState &sceneState,
                                                        std::uint32_t softParticleCount,
                                                        const GpuSoftDispatchConstants &constants)
{
    if (softParticleCount == 0u)
    {
        return true;
    }

    const auto &transient     = sceneState.transientBuffers();
    const auto &softParticles = sceneState.persistentSoftParticles();
    const std::array countBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseEntries",
                              transient.particleBroadPhaseEntriesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRanges", transient.particleCellRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedParticleBroadPhaseKeys",
                              transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleBroadPhaseMetadata",
                              softParticles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleAdjacencyOffsets",
                              softParticles.adjacencyOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleAdjacencyCounts", softParticles.adjacencyCountsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleAdjacencyIndices",
                              softParticles.adjacencyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array emitBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleBroadPhaseEntries",
                              transient.particleBroadPhaseEntriesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleCellRanges", transient.particleCellRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedParticleBroadPhaseKeys",
                              transient.particleBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleBroadPhaseMetadata",
                              softParticles.broadPhaseMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleAdjacencyOffsets",
                              softParticles.adjacencyOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleAdjacencyCounts", softParticles.adjacencyCountsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleAdjacencyIndices",
                              softParticles.adjacencyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftCandidatePairs", transient.softSoftCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mCountSoftSoftCandidatePairsPass.dispatch(computeContext, kDefaultVariant, countBindings,
                                                     dispatchGroupCount(softParticleCount)) &&
           dispatchExclusiveScanPass(computeContext, sceneState, transient.softRadixBitFlagsBuffer,
                                     transient.softRadixBitOffsetsBuffer, softParticleCount) &&
           mFinalizeSoftSoftCandidatePairsPass.dispatch(computeContext, kDefaultVariant,
                                                        finalizeBindings, 1u) &&
           mEmitSoftSoftCandidatePairsPass.dispatch(computeContext, kDefaultVariant, emitBindings,
                                                    dispatchGroupCount(softParticleCount));
}

bool PhysicsPassDispatcher::buildSoftRigidCandidatePairs(Diligent::IDeviceContext *computeContext,
                                                         const PhysicsSceneGpuState &sceneState,
                                                         std::uint32_t softParticleCount,
                                                         const GpuSoftDispatchConstants &constants)
{
    if (softParticleCount == 0u)
    {
        return true;
    }

    const auto &transient           = sceneState.transientBuffers();
    const auto &softParticles       = sceneState.persistentSoftParticles();
    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &bodyColliderMapping = sceneState.persistentBodyColliderMapping();
    const std::array countBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleBroadPhaseMetadata",
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
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array emitBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleBroadPhaseMetadata",
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
        gpu::GpuBufferBinding{"g_CandidateCounts", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidateOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftCandidatePairs", transient.softRigidCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mCountSoftRigidCandidatePairsPass.dispatch(computeContext, kDefaultVariant,
                                                      countBindings,
                                                      dispatchGroupCount(softParticleCount)) &&
           dispatchExclusiveScanPass(computeContext, sceneState, transient.softRadixBitFlagsBuffer,
                                     transient.softRadixBitOffsetsBuffer, softParticleCount) &&
           mFinalizeSoftRigidCandidatePairsPass.dispatch(computeContext, kDefaultVariant,
                                                         finalizeBindings, 1u) &&
           mEmitSoftRigidCandidatePairsPass.dispatch(computeContext, kDefaultVariant, emitBindings,
                                                     dispatchGroupCount(softParticleCount));
}

bool PhysicsPassDispatcher::generateSoftContacts(Diligent::IDeviceContext *computeContext,
                                                 const PhysicsSceneGpuState &sceneState,
                                                 const GpuSoftDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftCandidatePairs", transient.softSoftCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftContacts", transient.softContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mGenerateSoftContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftGenerateContacts));
}

bool PhysicsPassDispatcher::generateSoftRigidContacts(Diligent::IDeviceContext *computeContext,
                                                      const PhysicsSceneGpuState &sceneState,
                                                      const GpuSoftDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &rigidBodies   = sceneState.persistentRigidBodies();
    const auto &mapping       = sceneState.persistentBodyColliderMapping();
    const auto &colliders     = sceneState.persistentColliders();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleBroadPhaseMetadata",
                              softParticles.broadPhaseMetadataBuffer,
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
        gpu::GpuBufferBinding{"g_ColliderShapeParams", colliders.shapeParamsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderLocalPositions", colliders.localPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderLocalOrientations", colliders.localOrientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderBroadPhaseData", colliders.broadPhaseDataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftCandidatePairs", transient.softRigidCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftRigidContacts", transient.softRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mGenerateSoftRigidContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftGenerateRigidContacts));
}

bool PhysicsPassDispatcher::prepareSoftCandidateIndirectArgs(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PhysicsIndirectDispatchArgs", transient.physicsIndirectArgsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return mPrepareSoftCandidateIndirectArgsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                          1u);
}

bool PhysicsPassDispatcher::prepareSoftActiveIndirectArgs(Diligent::IDeviceContext *computeContext,
                                                          const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PhysicsIndirectDispatchArgs", transient.physicsIndirectArgsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return mPrepareSoftActiveIndirectArgsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                       1u);
}

bool PhysicsPassDispatcher::compactSoftContacts(Diligent::IDeviceContext *computeContext,
                                                const PhysicsSceneGpuState &sceneState,
                                                const GpuSoftDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array compactBindings{
        gpu::GpuBufferBinding{"g_SoftContacts", transient.softContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ActiveSoftContacts", transient.activeSoftContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return dispatchExclusiveScanPass(computeContext, sceneState, transient.softRadixBitFlagsBuffer,
                                     transient.softRadixBitOffsetsBuffer,
                                     sceneState.softCandidatePairCapacity()) &&
           writeSoftDispatchConstants(computeContext, constants) &&
           mFinalizeActiveSoftContactsPass.dispatch(computeContext, kDefaultVariant,
                                                    finalizeBindings, 1u) &&
           mCompactActiveSoftContactsPass.dispatchIndirect(
               computeContext, kDefaultVariant, compactBindings,
               transient.physicsIndirectArgsBuffer,
               indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftCompactContacts));
}

bool PhysicsPassDispatcher::compactSoftRigidContacts(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     const GpuSoftDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array finalizeBindings{
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array compactBindings{
        gpu::GpuBufferBinding{"g_SoftRigidContacts", transient.softRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveFlags", transient.softRadixBitFlagsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ContactActiveOffsets", transient.softRadixBitOffsetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ActiveSoftRigidContacts", transient.activeSoftRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return dispatchExclusiveScanPass(computeContext, sceneState, transient.softRadixBitFlagsBuffer,
                                     transient.softRadixBitOffsetsBuffer,
                                     sceneState.softCandidatePairCapacity()) &&
           writeSoftDispatchConstants(computeContext, constants) &&
           mFinalizeActiveSoftRigidContactsPass.dispatch(computeContext, kDefaultVariant,
                                                         finalizeBindings, 1u) &&
           mCompactActiveSoftRigidContactsPass.dispatchIndirect(
               computeContext, kDefaultVariant, compactBindings,
               transient.physicsIndirectArgsBuffer,
               indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftCompactRigidContacts));
}

bool PhysicsPassDispatcher::clearSoftConstraintState(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     std::uint32_t threadCount,
                                                     const GpuSoftDispatchConstants &constants)
{
    if (threadCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftParticleVelocityCorrections",
                              transient.softVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftEdgeLambdas", transient.softEdgeLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftTetLambdas", transient.softTetLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mClearSoftConstraintStatePass.dispatch(computeContext, kDefaultVariant, bindings,
                                                  dispatchGroupCount(threadCount));
}

bool PhysicsPassDispatcher::solveSoftEdgeConstraints(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     std::uint32_t softEdgeCount,
                                                     const GpuSoftDispatchConstants &constants)
{
    if (softEdgeCount == 0u || constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftEdges", softTopology.edgesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftEdgeLambdas", transient.softEdgeLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftEdgeCorrections", transient.softEdgeCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mSolveSoftEdgeConstraintsPass.dispatch(computeContext, kDefaultVariant, solveBindings,
                                                  dispatchGroupCount(softEdgeCount));
}

bool PhysicsPassDispatcher::solveSoftTetConstraints(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t softTetCount,
                                                    const GpuSoftDispatchConstants &constants)
{
    if (softTetCount == 0u || constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTets", softTopology.tetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTetLambdas", transient.softTetLambdasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftTetCorrections", transient.softTetCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mSolveSoftTetConstraintsPass.dispatch(computeContext, kDefaultVariant, solveBindings,
                                                 dispatchGroupCount(softTetCount));
}

bool PhysicsPassDispatcher::applySoftEdgeCorrections(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     const GpuSoftDispatchConstants &constants)
{
    if (constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleEdgeRanges", softTopology.particleEdgeRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleIncidentEdges", softTopology.particleIncidentEdgesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftEdgeCorrections", transient.softEdgeCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mApplySoftEdgeCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                  dispatchGroupCount(constants.softParticleCount));
}

bool PhysicsPassDispatcher::applySoftTetCorrections(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    const GpuSoftDispatchConstants &constants)
{
    if (constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_ParticleTetRanges", softTopology.particleTetRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ParticleIncidentTets", softTopology.particleIncidentTetsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftTetCorrections", transient.softTetCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mApplySoftTetCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(constants.softParticleCount));
}

bool PhysicsPassDispatcher::solveSoftContacts(Diligent::IDeviceContext *computeContext,
                                              const PhysicsSceneGpuState &sceneState,
                                              const GpuSoftDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePreviousPositions",
                              softParticles.previousPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleMaterials", softParticles.materialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftContacts", transient.activeSoftContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveSoftContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, solveBindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftSolveContacts));
}

bool PhysicsPassDispatcher::solveSoftRigidContacts(Diligent::IDeviceContext *computeContext,
                                                   const PhysicsSceneGpuState &sceneState,
                                                   const GpuSoftDispatchConstants &constants)
{
    if (computeContext == nullptr || constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles       = sceneState.persistentSoftParticles();
    const auto &persistentRigid     = sceneState.persistentRigidBodies();
    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &transient           = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"g_SoftParticlePreviousPositions",
                              softParticles.previousPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleMaterials", softParticles.materialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                              persistentRigid.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyOrientations", persistentRigid.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
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
        gpu::GpuBufferBinding{"g_SoftRigidContacts", transient.activeSoftRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveSoftRigidContactsPass.dispatchIndirect(
        computeContext, kDefaultVariant, solveBindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftSolveRigidContacts));
}

bool PhysicsPassDispatcher::applySoftPositionCorrections(Diligent::IDeviceContext *computeContext,
                                                         const PhysicsSceneGpuState &sceneState,
                                                         const GpuSoftDispatchConstants &constants)
{
    if (constants.softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mApplySoftPositionCorrectionsPass.dispatch(
               computeContext, kDefaultVariant, bindings,
               dispatchGroupCount(constants.softParticleCount));
}

bool PhysicsPassDispatcher::updateSoftVelocities(Diligent::IDeviceContext *computeContext,
                                                 const PhysicsSceneGpuState &sceneState,
                                                 std::uint32_t softParticleCount,
                                                 const GpuSoftDispatchConstants &constants)
{
    if (softParticleCount == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftParticlePreviousPositions",
                              softParticles.previousPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleMaterials", softParticles.materialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mUpdateSoftVelocitiesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                              dispatchGroupCount(softParticleCount));
}

bool PhysicsPassDispatcher::dispatchSolveSoftContactVelocitiesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &transient     = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleMaterials", softParticles.materialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftContacts", transient.activeSoftContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleVelocityCorrections",
                              transient.softVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveSoftContactVelocitiesPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftSolveContactVelocities));
}

bool PhysicsPassDispatcher::solveSoftContactVelocities(Diligent::IDeviceContext *computeContext,
                                                       const PhysicsSceneGpuState &sceneState,
                                                       std::uint32_t softParticleCount,
                                                       std::uint32_t iterations)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (softParticleCount == 0u || iterations == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array applyBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftParticleVelocityCorrections",
                              transient.softVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!mApplySoftContactVelocitiesPass.bindVariant(kDefaultVariant, applyBindings))
    {
        return false;
    }

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        if (!dispatchSolveSoftContactVelocitiesPass(computeContext, sceneState) ||
            !mApplySoftContactVelocitiesPass.dispatch(computeContext, kDefaultVariant,
                                                      applyBindings,
                                                      dispatchGroupCount(softParticleCount)))
        {
            return false;
        }
    }

    return true;
}

bool PhysicsPassDispatcher::dispatchSolveSoftRigidContactVelocitiesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState)
{
    if (computeContext == nullptr)
    {
        return true;
    }

    const auto &softParticles       = sceneState.persistentSoftParticles();
    const auto &persistentRigid     = sceneState.persistentRigidBodies();
    const auto &persistentColliders = sceneState.persistentColliders();
    const auto &transient           = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleMaterials", softParticles.materialsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleVelocities", softParticles.velocitiesBuffer,
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
        gpu::GpuBufferBinding{"g_SoftRigidContacts", transient.activeSoftRigidContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftNeighborMeta", transient.softNeighborMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleVelocityCorrections",
                              transient.softVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mSolveSoftRigidContactVelocitiesPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::SoftSolveRigidContacts));
}

bool PhysicsPassDispatcher::solveSoftRigidContactVelocities(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t softParticleCount, std::uint32_t rigidBodyCount, std::uint32_t iterations,
    const GpuRigidDispatchConstants &rigidConstants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (softParticleCount == 0u || iterations == 0u)
    {
        return true;
    }

    const auto &softParticles   = sceneState.persistentSoftParticles();
    const auto &transient       = sceneState.transientBuffers();
    const auto &persistentRigid = sceneState.persistentRigidBodies();
    const std::array applySoftBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftParticleVelocityCorrections",
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

    if (!mApplySoftContactVelocitiesPass.bindVariant(kDefaultVariant, applySoftBindings) ||
        !mApplyRigidContactVelocitiesPass.bindVariant(kDefaultVariant, applyRigidBindings))
    {
        return false;
    }

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        GpuRigidDispatchConstants iterationRigidConstants = rigidConstants;
        iterationRigidConstants.iterationIndex            = iteration;

        if (!dispatchSolveSoftRigidContactVelocitiesPass(computeContext, sceneState))
        {
            return false;
        }

        if (!mApplySoftContactVelocitiesPass.dispatch(computeContext, kDefaultVariant,
                                                      applySoftBindings,
                                                      dispatchGroupCount(softParticleCount)) ||
            !writeRigidDispatchConstants(computeContext, iterationRigidConstants) ||
            !mApplyRigidContactVelocitiesPass.dispatch(computeContext, kDefaultVariant,
                                                       applyRigidBindings,
                                                       dispatchGroupCount(rigidBodyCount)))
        {
            return false;
        }
    }

    return true;
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
    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftRenderDispatchConstantsBuffer",
                              mSoftRenderDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
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
    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &softTopology  = sceneState.persistentSoftTopology();
    const auto &transient     = sceneState.transientBuffers();
    const std::array chunkBindings{
        gpu::GpuBufferBinding{"PhysicsSoftRenderDispatchConstantsBuffer",
                              mSoftRenderDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
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
        gpu::GpuBufferBinding{"g_RigidBodyScales", persistentBodies.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistentBodies.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderOwnerRigidBodyIndices",
                              persistentColliders.ownerRigidBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderShapeTypes", persistentColliders.shapeTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderShapeParams", persistentColliders.shapeParamsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderLocalPositions", persistentColliders.localPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderLocalOrientations",
                              persistentColliders.localOrientationsBuffer,
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

    if (!dispatchExclusiveScanPass(
            computeContext, sceneState, sceneState.transientBuffers().activeBodyFlagsBuffer,
            sceneState.transientBuffers().activeBodyOffsetsBuffer, constants.colliderCount))
    {
        return false;
    }
    if (!dispatchExclusiveScanPass(
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

        if (!dispatchExclusiveScanPass(computeContext, sceneState, radixBitFlags, radixBitOffsets,
                                       count))
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

        if (!dispatchExclusiveScanPass(computeContext, sceneState, radixBitFlags, radixBitOffsets,
                                       count))
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
        if (!dispatchExclusiveScanPass(computeContext, sceneState,
                                       transient.pairCountBuffers[pairType],
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

    const std::array chunkBindings{
        gpu::GpuBufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_NarrowPhaseChunks", transient.narrowPhaseChunksBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_NarrowPhaseMeta", transient.narrowPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_NarrowPhaseChunkCounter", transient.narrowPhaseChunkCounterBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return mBuildNarrowPhaseChunksPass.dispatch(computeContext, kDefaultVariant, chunkBindings, 1u);
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

bool PhysicsPassDispatcher::dispatchSolveRigidContactConstraintsPass(
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
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyPositionsInvMass", persistent.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreviousRigidBodyOrientations", persistent.orientationsBuffer,
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

    return mSolveRigidContactConstraintsPass.dispatchIndirect(
        computeContext, kDefaultVariant, bindings, transient.physicsIndirectArgsBuffer,
        indirectArgsOffset(GpuPhysicsIndirectDispatchSlot::RigidSolveContacts));
}

bool PhysicsPassDispatcher::solveRigidContactConstraints(Diligent::IDeviceContext *computeContext,
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
           dispatchSolveRigidContactConstraintsPass(computeContext, sceneState);
}

bool PhysicsPassDispatcher::updateRigidDispatchConstants(
    Diligent::IDeviceContext *computeContext, const GpuRigidDispatchConstants &constants)
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

bool PhysicsPassDispatcher::solveBallJointConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuRigidDispatchConstants &constants)
{
    (void)constants;
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
    const PhysicsSceneGpuState &sceneState,
    Diligent::IBuffer *jointIndicesBuffer,
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
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return pass.dispatch(computeContext, kDefaultVariant, bindings, dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::solveHingeJointConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuRigidDispatchConstants &constants)
{
    (void)constants;
    const std::uint32_t passiveJointCount = sceneState.hingePassiveJointCount();
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
                persistentJoints.hingePassiveJointIndicesBuffer,
                passiveJointCount))
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
                persistentJoints.hingePositionDriveJointIndicesBuffer,
                positionDriveJointCount))
        {
            return false;
        }
    }

    return true;
}

bool PhysicsPassDispatcher::dispatchSolveSliderJointConstraintsPass(
    Diligent::IDeviceContext *computeContext, gpu::GpuComputePass &pass,
    const PhysicsSceneGpuState &sceneState,
    Diligent::IBuffer *jointIndicesBuffer,
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
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return pass.dispatch(computeContext, kDefaultVariant, bindings, dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::solveSliderJointConstraints(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuRigidDispatchConstants &constants)
{
    (void)constants;
    const std::uint32_t passiveJointCount = sceneState.sliderPassiveJointCount();
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
                persistentJoints.sliderPassiveJointIndicesBuffer,
                passiveJointCount))
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
                persistentJoints.sliderPositionDriveJointIndicesBuffer,
                positionDriveJointCount))
        {
            return false;
        }
    }

    return true;
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

    return mSolveSliderJointTargetVelocitiesPass.dispatch(
        computeContext, kDefaultVariant, bindings, dispatchGroupCount(jointCount));
}

bool PhysicsPassDispatcher::dispatchApplyRigidVelocityCorrectionsPass(
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

    const auto &transient  = sceneState.transientBuffers();
    const auto &persistent = sceneState.persistentRigidBodies();
    const std::array bindings{
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

    return mApplyRigidContactVelocitiesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                     dispatchGroupCount(rigidBodyCount));
}

bool PhysicsPassDispatcher::solveHingeJointTargetVelocities(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuRigidDispatchConstants &constants)
{
    (void)constants;
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
               computeContext, sceneState,
               persistentJoints.hingeVelocityDriveJointIndicesBuffer, jointCount);
}

bool PhysicsPassDispatcher::solveSliderJointTargetVelocities(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    const GpuRigidDispatchConstants &constants)
{
    (void)constants;
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
               computeContext, sceneState,
               persistentJoints.sliderVelocityDriveJointIndicesBuffer, jointCount);
}

bool PhysicsPassDispatcher::applyRigidVelocityCorrections(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t rigidBodyCount, const GpuRigidDispatchConstants &constants)
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
           dispatchApplyRigidVelocityCorrectionsPass(computeContext, sceneState, rigidBodyCount);
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

bool PhysicsPassDispatcher::dispatchSolveRigidContactVelocitiesPass(
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
        gpu::GpuBufferBinding{"g_RigidContacts", transient.rigidContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
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

bool PhysicsPassDispatcher::solveRigidContactVelocities(Diligent::IDeviceContext *computeContext,
                                                        const PhysicsSceneGpuState &sceneState,
                                                        std::uint32_t rigidBodyCount,
                                                        std::uint32_t iterations,
                                                        const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (rigidBodyCount == 0u || iterations == 0u)
    {
        return true;
    }

    const std::uint32_t hingeVelocityJointCount = sceneState.hingeVelocityDriveJointCount();
    const std::uint32_t sliderVelocityJointCount = sceneState.sliderVelocityDriveJointCount();
    const bool hasVelocityMotorJoints =
        hingeVelocityJointCount > 0u || sliderVelocityJointCount > 0u;

    const auto &transient  = sceneState.transientBuffers();
    const auto &persistent = sceneState.persistentRigidBodies();
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
        GpuRigidDispatchConstants iterationConstants = constants;
        iterationConstants.iterationIndex            = iteration;

        if (hasVelocityMotorJoints)
        {
            if (hingeVelocityJointCount > 0u &&
                !solveHingeJointTargetVelocities(computeContext, sceneState, iterationConstants))
            {
                return false;
            }
            if (sliderVelocityJointCount > 0u &&
                !solveSliderJointTargetVelocities(computeContext, sceneState, iterationConstants))
            {
                return false;
            }
            if (!mApplyRigidContactVelocitiesPass.dispatch(computeContext, kDefaultVariant,
                                                           applyBindings,
                                                           dispatchGroupCount(rigidBodyCount)))
            {
                return false;
            }
        }

        if (!writeRigidDispatchConstants(computeContext, iterationConstants) ||
            !dispatchSolveRigidContactVelocitiesPass(computeContext, sceneState))
        {
            return false;
        }

        if (!mApplyRigidContactVelocitiesPass.dispatch(computeContext, kDefaultVariant, applyBindings,
                                                       dispatchGroupCount(rigidBodyCount)))
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
           mBuildParticleBroadPhaseEntriesPass.forceRecreateAllVariants() &&
           mBuildParticleBroadPhaseKeysPass.forceRecreateAllVariants() &&
           mMarkParticleCellRangeStartsPass.forceRecreateAllVariants() &&
           mClearParticleCellRangesPass.forceRecreateAllVariants() &&
           mBuildParticleCellRangesPass.forceRecreateAllVariants() &&
           mCountSoftSoftCandidatePairsPass.forceRecreateAllVariants() &&
           mFinalizeSoftSoftCandidatePairsPass.forceRecreateAllVariants() &&
           mEmitSoftSoftCandidatePairsPass.forceRecreateAllVariants() &&
           mCountSoftRigidCandidatePairsPass.forceRecreateAllVariants() &&
           mFinalizeSoftRigidCandidatePairsPass.forceRecreateAllVariants() &&
           mEmitSoftRigidCandidatePairsPass.forceRecreateAllVariants() &&
           mGenerateSoftContactsPass.forceRecreateAllVariants() &&
           mGenerateSoftRigidContactsPass.forceRecreateAllVariants() &&
           mPrepareSoftCandidateIndirectArgsPass.forceRecreateAllVariants() &&
           mPrepareSoftActiveIndirectArgsPass.forceRecreateAllVariants() &&
           mFinalizeActiveSoftContactsPass.forceRecreateAllVariants() &&
           mCompactActiveSoftContactsPass.forceRecreateAllVariants() &&
           mFinalizeActiveSoftRigidContactsPass.forceRecreateAllVariants() &&
           mCompactActiveSoftRigidContactsPass.forceRecreateAllVariants() &&
           mClearSoftConstraintStatePass.forceRecreateAllVariants() &&
           mSolveSoftEdgeConstraintsPass.forceRecreateAllVariants() &&
           mSolveSoftTetConstraintsPass.forceRecreateAllVariants() &&
           mApplySoftEdgeCorrectionsPass.forceRecreateAllVariants() &&
           mApplySoftTetCorrectionsPass.forceRecreateAllVariants() &&
           mSolveSoftContactsPass.forceRecreateAllVariants() &&
           mSolveSoftRigidContactsPass.forceRecreateAllVariants() &&
           mApplySoftPositionCorrectionsPass.forceRecreateAllVariants() &&
           mUpdateSoftVelocitiesPass.forceRecreateAllVariants() &&
           mSolveSoftContactVelocitiesPass.forceRecreateAllVariants() &&
           mSolveSoftRigidContactVelocitiesPass.forceRecreateAllVariants() &&
           mApplySoftContactVelocitiesPass.forceRecreateAllVariants() &&
           mUpdateSoftTriangleNormalsPass.forceRecreateAllVariants() &&
           mUpdateSoftRenderNormalsPass.forceRecreateAllVariants() &&
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
           mSolveRigidContactConstraintsPass.forceRecreateAllVariants() &&
           mSolveBallJointConstraintsPass.forceRecreateAllVariants() &&
           mSolveHingeJointConstraintsPassivePass.forceRecreateAllVariants() &&
           mSolveHingeJointConstraintsTargetPositionPass.forceRecreateAllVariants() &&
           mSolveSliderJointConstraintsPassivePass.forceRecreateAllVariants() &&
           mSolveSliderJointConstraintsTargetPositionPass.forceRecreateAllVariants() &&
           mSolveHingeJointTargetVelocitiesPass.forceRecreateAllVariants() &&
           mSolveSliderJointTargetVelocitiesPass.forceRecreateAllVariants() &&
           mApplyRigidCorrectionsPass.forceRecreateAllVariants() &&
           mUpdateRigidVelocitiesPass.forceRecreateAllVariants() &&
           mSolveRigidContactVelocitiesPass.forceRecreateAllVariants() &&
           mApplyRigidContactVelocitiesPass.forceRecreateAllVariants();
}

} // namespace cressim::neo::physics
