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

bool PhysicsPassDispatcher::writeSoftDispatchConstants(Diligent::IDeviceContext *computeContext,
                                                       const GpuSoftDispatchConstants &constants)
{
    return writeConstantsBuffer(computeContext, mSoftDispatchConstantsBuffer, &constants,
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

    if (!initPass(mPredictPass, kPredict) || !initPass(mSoftPredictPass, kSoftPredict) ||
        !initPass(mUpdateRigidSurfaceWorldPositionsPass, kUpdateRigidSurfaceWorldPositions) ||
        !initPass(mBuildSoftBroadPhaseParticlesPass, kBuildSoftBroadPhaseParticles) ||
        !initPass(mBuildSoftBroadPhaseKeysPass, kBuildSoftBroadPhaseKeys) ||
        !initPass(mEmitSoftCandidatePairsPass, kEmitSoftCandidatePairs) ||
        !initPass(mGenerateSoftContactsPass, kGenerateSoftContacts) ||
        !initPass(mClearSoftConstraintStatePass, kClearSoftConstraintState) ||
        !initPass(mSolveSoftEdgeConstraintsPass, kSolveSoftEdgeConstraints) ||
        !initPass(mSolveSoftTetConstraintsPass, kSolveSoftTetConstraints) ||
        !initPass(mSolveSoftContactsPass, kSolveSoftContacts) ||
        !initPass(mApplySoftContactCorrectionsPass, kApplySoftContactCorrections) ||
        !initPass(mUpdateSoftVelocitiesPass, kUpdateSoftVelocities) ||
        !initPass(mUpdateWorldAabbsPass, kUpdateWorldAabbs) ||
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
        !initPass(mGenerateContactsPass, kGenerateContacts) ||
        !initPass(mSolveGatherPass, kSolveGather) ||
        !initPass(mClearCorrectionsPass, kClearCorrections) ||
        !initPass(mApplyCorrectionsPass, kApplyCorrections) ||
        !initPass(mUpdateVelocitiesPass, kUpdateVelocities) ||
        !initPass(mSolveContactVelocitiesPass, kSolveContactVelocities) ||
        !initPass(mApplyContactVelocitiesPass, kApplyContactVelocities))
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
           createConstantsBuffer("CRESSimNeo.Physics.SoftDispatchConstants",
                                 sizeof(GpuSoftDispatchConstants), mSoftDispatchConstantsBuffer) &&
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

bool PhysicsPassDispatcher::clearCorrections(Diligent::IDeviceContext *computeContext,
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
        !mClearCorrectionsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                        dispatchGroupCount(bodyCount)))
    {
        return false;
    }

    sceneState.setCorrectionBuffersNeedClear(false);
    return true;
}

bool PhysicsPassDispatcher::predict(Diligent::IDeviceContext *computeContext,
                                    const PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
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
           mPredictPass.dispatch(computeContext, kDefaultVariant, bindings,
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

bool PhysicsPassDispatcher::updateRigidSurfaceWorldPositions(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t rigidSurfaceParticleCount, const GpuSoftDispatchConstants &constants)
{
    if (rigidSurfaceParticleCount == 0u)
    {
        return true;
    }

    const auto &surfaceParticles = sceneState.persistentRigidSurfaceParticles();
    const auto &transient        = sceneState.transientBuffers();
    const auto &colliders        = sceneState.persistentColliders();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleLocalPositions",
                              surfaceParticles.localPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleOwningRigidBodyIndices",
                              surfaceParticles.owningRigidBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleOwningColliderIndices",
                              surfaceParticles.owningColliderIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                              transient.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PredictedRigidBodyOrientations",
                              transient.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderLocalPositions", colliders.localPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_ColliderLocalOrientations", colliders.localOrientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleWorldPositions",
                              surfaceParticles.worldPositionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mUpdateRigidSurfaceWorldPositionsPass.dispatch(
               computeContext, kDefaultVariant, bindings,
               dispatchGroupCount(rigidSurfaceParticleCount));
}

bool PhysicsPassDispatcher::buildSoftBroadPhaseParticles(Diligent::IDeviceContext *computeContext,
                                                         const PhysicsSceneGpuState &sceneState,
                                                         std::uint32_t totalParticleLikeCount,
                                                         const GpuSoftDispatchConstants &constants)
{
    if (totalParticleLikeCount == 0u)
    {
        return true;
    }

    const auto &softParticles    = sceneState.persistentSoftParticles();
    const auto &surfaceParticles = sceneState.persistentRigidSurfaceParticles();
    const auto &transient        = sceneState.transientBuffers();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleOwningSoftBodyIndices",
                              softParticles.owningSoftBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleWorldPositions",
                              surfaceParticles.worldPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleOwningRigidBodyIndices",
                              surfaceParticles.owningRigidBodyIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBroadPhaseParticles", transient.softBroadPhaseParticlesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mBuildSoftBroadPhaseParticlesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                      dispatchGroupCount(totalParticleLikeCount));
}

bool PhysicsPassDispatcher::buildSoftBroadPhaseKeys(Diligent::IDeviceContext *computeContext,
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
        gpu::GpuBufferBinding{"g_SoftBroadPhaseParticles", transient.softBroadPhaseParticlesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBroadPhaseKeys", transient.softBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mBuildSoftBroadPhaseKeysPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                 dispatchGroupCount(totalParticleLikeCount));
}

bool PhysicsPassDispatcher::sortSoftBroadPhase(Diligent::IDeviceContext *computeContext,
                                               const PhysicsSceneGpuState &sceneState,
                                               std::uint32_t count)
{
    return dispatchSoftRadixSortPass(computeContext, sceneState, count);
}

bool PhysicsPassDispatcher::emitSoftCandidatePairs(Diligent::IDeviceContext *computeContext,
                                                   const PhysicsSceneGpuState &sceneState,
                                                   std::uint32_t softParticleCount,
                                                   const GpuSoftDispatchConstants &constants)
{
    if (softParticleCount == 0u)
    {
        return true;
    }

    const std::uint32_t zero = 0u;
    computeContext->UpdateBuffer(sceneState.transientBuffers().softCandidatePairCountBuffer, 0u,
                                 sizeof(std::uint32_t), &zero,
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    const auto &transient        = sceneState.transientBuffers();
    const auto &softParticles    = sceneState.persistentSoftParticles();
    const auto &surfaceParticles = sceneState.persistentRigidSurfaceParticles();
    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBroadPhaseParticles", transient.softBroadPhaseParticlesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SortedSoftBroadPhaseKeys", transient.softBroadPhaseKeysBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleRadii", softParticles.radiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleEnvironmentIndices",
                              softParticles.environmentIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleCollisionLayers", softParticles.collisionLayersBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticleCollisionMasks", softParticles.collisionMasksBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleWorldPositions",
                              surfaceParticles.worldPositionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleSampleRadii",
                              surfaceParticles.sampleRadiiBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleEnvironmentIndices",
                              surfaceParticles.environmentIndicesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleCollisionLayers",
                              surfaceParticles.collisionLayersBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidSurfaceParticleCollisionMasks",
                              surfaceParticles.collisionMasksBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftCandidatePairs", transient.softCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftCandidatePairCount", transient.softCandidatePairCountBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mEmitSoftCandidatePairsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                dispatchGroupCount(softParticleCount));
}

bool PhysicsPassDispatcher::generateSoftContacts(Diligent::IDeviceContext *computeContext,
                                                 const PhysicsSceneGpuState &sceneState,
                                                 const GpuSoftDispatchConstants &constants)
{
    if (constants.softCandidatePairCapacity == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &rigidBodies   = sceneState.persistentRigidBodies();
    const auto &mapping       = sceneState.persistentBodyColliderMapping();
    const auto &colliders     = sceneState.persistentColliders();
    const auto &transient     = sceneState.transientBuffers();
    const std::array bindings{
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
        gpu::GpuBufferBinding{"g_SoftCandidatePairs", transient.softCandidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftCandidatePairCount", transient.softCandidatePairCountBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftContacts", transient.softContactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mGenerateSoftContactsPass.dispatch(
               computeContext, kDefaultVariant, bindings,
               dispatchGroupCount(constants.softCandidatePairCapacity));
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
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array applyBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mSolveSoftEdgeConstraintsPass.dispatch(computeContext, kDefaultVariant, solveBindings,
                                                  dispatchGroupCount(softEdgeCount)) &&
           mApplySoftContactCorrectionsPass.dispatch(
               computeContext, kDefaultVariant, applyBindings,
               dispatchGroupCount(constants.softParticleCount));
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
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array applyBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mSolveSoftTetConstraintsPass.dispatch(computeContext, kDefaultVariant, solveBindings,
                                                 dispatchGroupCount(softTetCount)) &&
           mApplySoftContactCorrectionsPass.dispatch(
               computeContext, kDefaultVariant, applyBindings,
               dispatchGroupCount(constants.softParticleCount));
}

bool PhysicsPassDispatcher::solveSoftContacts(Diligent::IDeviceContext *computeContext,
                                              const PhysicsSceneGpuState &sceneState,
                                              std::uint32_t iterations,
                                              const GpuSoftDispatchConstants &constants)
{
    if (iterations == 0u || constants.softParticleCount == 0u ||
        constants.softCandidatePairCapacity == 0u)
    {
        return true;
    }

    const auto &softParticles = sceneState.persistentSoftParticles();
    const auto &transient     = sceneState.transientBuffers();
    const std::array solveBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftContacts", transient.softContactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    const std::array applyBindings{
        gpu::GpuBufferBinding{"PhysicsSoftDispatchConstantsBuffer", mSoftDispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftParticlePositionsInvMass",
                              softParticles.positionsInvMassBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_SoftPositionCorrections", transient.softPositionCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!mApplySoftContactCorrectionsPass.bindVariant(kDefaultVariant, applyBindings))
    {
        return false;
    }

    for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration)
    {
        if (!writeSoftDispatchConstants(computeContext, constants) ||
            !mSolveSoftContactsPass.dispatch(
                computeContext, kDefaultVariant, solveBindings,
                dispatchGroupCount(constants.softCandidatePairCapacity)))
        {
            return false;
        }

        if (!mApplySoftContactCorrectionsPass.dispatch(
                computeContext, kDefaultVariant, applyBindings,
                dispatchGroupCount(constants.softParticleCount)))
        {
            return false;
        }
    }

    return true;
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
        gpu::GpuBufferBinding{"g_SoftParticleVelocities", softParticles.velocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeSoftDispatchConstants(computeContext, constants) &&
           mUpdateSoftVelocitiesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                              dispatchGroupCount(softParticleCount));
}

bool PhysicsPassDispatcher::updateWorldAabbs(Diligent::IDeviceContext *computeContext,
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
           mUpdateWorldAabbsPass.dispatch(computeContext, kDefaultVariant, bindings,
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
    Diligent::IBuffer *finalMortonBuffer = transient.softBroadPhaseKeysBuffer;
    Diligent::IBuffer *currentInput      = transient.softBroadPhaseKeysBuffer;
    Diligent::IBuffer *currentOutput     = transient.softBroadPhaseKeysScratchBuffer;
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
    if (activeMovingCount == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
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

    if (!dispatchReduceBroadPhaseExtentPass(computeContext, sceneState, activeMovingCount, false))
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

bool PhysicsPassDispatcher::dispatchGenerateContactsPass(Diligent::IDeviceContext *computeContext,
                                                         const PhysicsSceneGpuState &sceneState,
                                                         std::uint32_t pairCount)
{
    if (pairCount == 0u)
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
        gpu::GpuBufferBinding{"g_ColliderMaterials", persistentColliders.materialBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CandidatePairs", transient.candidatePairsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_NarrowPhaseChunks", transient.narrowPhaseChunksBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_NarrowPhaseMeta", transient.narrowPhaseMetaBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_NarrowPhaseChunkCounter", transient.narrowPhaseChunkCounterBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidContacts", transient.contactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::uint32_t dispatchGroupUpperBound =
        ((pairCount + kNarrowPhaseChunkSize - 1u) / kNarrowPhaseChunkSize) +
        (kRigidPairTypeCount - 1u);

    return mGenerateContactsPass.dispatch(computeContext, kDefaultVariant, bindings,
                                          dispatchGroupUpperBound);
}

bool PhysicsPassDispatcher::generateContacts(Diligent::IDeviceContext *computeContext,
                                             const PhysicsSceneGpuState &sceneState,
                                             std::uint32_t pairCount)
{
    return dispatchGenerateContactsPass(computeContext, sceneState, pairCount);
}

bool PhysicsPassDispatcher::dispatchSolveGatherPass(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    std::uint32_t pairCount)
{
    if (pairCount == 0u)
    {
        return false;
    }

    const auto &persistent = sceneState.persistentRigidBodies();
    const auto &transient  = sceneState.transientBuffers();

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
                              persistent.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidContacts", transient.contactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyTranslationCorrections",
                              transient.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::uint32_t contactSlotCount = pairCount * kRigidContactsPerPair;
    return mSolveGatherPass.dispatch(computeContext, kDefaultVariant, bindings,
                                     dispatchGroupCount(contactSlotCount));
}

bool PhysicsPassDispatcher::solveConstraints(Diligent::IDeviceContext *computeContext,
                                             const PhysicsSceneGpuState &sceneState,
                                             std::uint32_t rigidBodyCount, std::uint32_t pairCount,
                                             std::uint32_t iterations,
                                             const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (pairCount == 0u || iterations == 0u)
    {
        return true;
    }

    const auto &transient = sceneState.transientBuffers();
    const std::array applyBindings{
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

    if (!mApplyCorrectionsPass.bindVariant(kDefaultVariant, applyBindings))
    {
        return false;
    }

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        GpuRigidDispatchConstants iterationConstants = constants;
        iterationConstants.iterationIndex            = iteration;

        if (!writeRigidDispatchConstants(computeContext, iterationConstants) ||
            !dispatchSolveGatherPass(computeContext, sceneState, pairCount))
        {
            return false;
        }

        if (!mApplyCorrectionsPass.dispatch(computeContext, kDefaultVariant, applyBindings,
                                            dispatchGroupCount(rigidBodyCount)))
        {
            return false;
        }
    }

    return true;
}

bool PhysicsPassDispatcher::updateVelocities(Diligent::IDeviceContext *computeContext,
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
           mUpdateVelocitiesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                          dispatchGroupCount(bodyCount));
}

bool PhysicsPassDispatcher::dispatchSolveContactVelocitiesPass(
    Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
    std::uint32_t pairCount)
{
    if (pairCount == 0u)
    {
        return true;
    }

    const auto &persistent = sceneState.persistentRigidBodies();
    const auto &transient  = sceneState.transientBuffers();

    const std::array bindings{
        gpu::GpuBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
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
        gpu::GpuBufferBinding{"g_RigidContacts", transient.contactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RigidBodyLinearVelocityCorrections",
                              transient.linearVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RigidBodyAngularVelocityCorrections",
                              transient.angularVelocityCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::uint32_t slotCount = pairCount * kRigidContactsPerPair;
    return mSolveContactVelocitiesPass.dispatch(computeContext, kDefaultVariant, bindings,
                                                dispatchGroupCount(slotCount));
}

bool PhysicsPassDispatcher::solveContactVelocities(Diligent::IDeviceContext *computeContext,
                                                   const PhysicsSceneGpuState &sceneState,
                                                   std::uint32_t rigidBodyCount,
                                                   std::uint32_t pairCount,
                                                   std::uint32_t iterations,
                                                   const GpuRigidDispatchConstants &constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (rigidBodyCount == 0u || pairCount == 0u || iterations == 0u)
    {
        return true;
    }

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

    if (!mApplyContactVelocitiesPass.bindVariant(kDefaultVariant, applyBindings))
    {
        return false;
    }

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        GpuRigidDispatchConstants iterationConstants = constants;
        iterationConstants.iterationIndex            = iteration;

        if (!writeRigidDispatchConstants(computeContext, iterationConstants) ||
            !dispatchSolveContactVelocitiesPass(computeContext, sceneState, pairCount))
        {
            return false;
        }

        if (!mApplyContactVelocitiesPass.dispatch(computeContext, kDefaultVariant, applyBindings,
                                                  dispatchGroupCount(rigidBodyCount)))
        {
            return false;
        }
    }

    return true;
}

} // namespace cressim::neo::physics
