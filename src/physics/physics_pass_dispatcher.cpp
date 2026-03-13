#include "physics/physics_pass_dispatcher.h"
#include "physics/physics_pass_definitions.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"
#include "DiligentEngine/DiligentCore/Primitives/interface/Errors.hpp"

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

} // namespace

template <std::size_t N>
bool PhysicsPassDispatcher::writeAndDispatchRigid(
    Diligent::IDeviceContext* computeContext, const ComputePass& pass, std::size_t variantIndex,
    const std::array<ComputeBufferBinding, N>& bindings,
    const GpuRigidDispatchConstants* constants, std::uint32_t groupCountX,
    std::uint32_t groupCountY, std::uint32_t groupCountZ)
{
    if (constants != nullptr && !writeRigidDispatchConstants(computeContext, *constants))
    {
        return false;
    }

    return pass.dispatch(computeContext, variantIndex, bindings, groupCountX, groupCountY,
                         groupCountZ);
}

template <std::size_t N>
bool PhysicsPassDispatcher::writeAndDispatchScan(
    Diligent::IDeviceContext* computeContext, const ComputePass& pass, std::size_t variantIndex,
    const std::array<ComputeBufferBinding, N>& bindings,
    const GpuPhysicsScanConstants* constants, std::uint32_t groupCountX,
    std::uint32_t groupCountY, std::uint32_t groupCountZ)
{
    if (constants != nullptr && !writeScanConstants(computeContext, *constants))
    {
        return false;
    }

    return pass.dispatch(computeContext, variantIndex, bindings, groupCountX, groupCountY,
                         groupCountZ);
}

template <std::size_t N>
bool PhysicsPassDispatcher::writeAndDispatchRadix(
    Diligent::IDeviceContext* computeContext, const ComputePass& pass, std::size_t variantIndex,
    const std::array<ComputeBufferBinding, N>& bindings,
    const GpuPhysicsRadixConstants* constants, std::uint32_t groupCountX,
    std::uint32_t groupCountY, std::uint32_t groupCountZ)
{
    if (constants != nullptr && !writeRadixConstants(computeContext, *constants))
    {
        return false;
    }

    return pass.dispatch(computeContext, variantIndex, bindings, groupCountX, groupCountY,
                         groupCountZ);
}

bool PhysicsPassDispatcher::writeConstantsBuffer(Diligent::IDeviceContext* computeContext,
                                                 Diligent::IBuffer* buffer, const void* constants,
                                                 std::size_t constantsSize)
{
    if (computeContext == nullptr || buffer == nullptr || constants == nullptr || constantsSize == 0u)
    {
        return false;
    }

    void* mapped = nullptr;
    computeContext->MapBuffer(buffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }

    std::memcpy(mapped, constants, constantsSize);
    computeContext->UnmapBuffer(buffer, Diligent::MAP_WRITE);
    return true;
}

bool PhysicsPassDispatcher::writeRigidDispatchConstants(
    Diligent::IDeviceContext* computeContext, const GpuRigidDispatchConstants& constants)
{
    return writeConstantsBuffer(computeContext, mRigidDispatchConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::writeScanConstants(Diligent::IDeviceContext* computeContext,
                                               const GpuPhysicsScanConstants& constants)
{
    return writeConstantsBuffer(computeContext, mScanConstantsBuffer, &constants, sizeof(constants));
}

bool PhysicsPassDispatcher::writeRadixConstants(Diligent::IDeviceContext* computeContext,
                                                const GpuPhysicsRadixConstants& constants)
{
    return writeConstantsBuffer(computeContext, mRadixConstantsBuffer, &constants,
                                sizeof(constants));
}

bool PhysicsPassDispatcher::initialize(Diligent::IRenderDevice* renderDevice,
                                       std::uint32_t physicsContextId,
                                       const char* shaderSourceDirectory)
{
    if (renderDevice == nullptr || shaderSourceDirectory == nullptr)
    {
        return false;
    }

    mShaderLibrary      = gpu::ShaderLibrary(shaderSourceDirectory);
    mPhysicsContextMask = static_cast<Diligent::Uint64>(1ull) << physicsContextId;

    Diligent::IShaderSourceInputStreamFactory* streamFactory = mShaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: shader stream factory is null.");
        return false;
    }

    auto initPass = [&](ComputePass& pass, const ComputePassDefinition& definition,
                        std::size_t variantCount = 1u) -> bool
    {
        if (!pass.initialize(renderDevice, streamFactory, mPhysicsContextMask, definition))
        {
            return false;
        }
        return pass.createVariants(variantCount);
    };

    using namespace passdefs;

    if (!initPass(mPredictPass, kPredict) || !initPass(mUpdateWorldAabbsPass, kUpdateWorldAabbs) ||
        !initPass(mScanBlockPass, kScanBlock) || !initPass(mScanAddOffsetsPass, kScanAddOffsets) ||
        !initPass(mCompactActiveBodiesPass, kCompactActiveBodies, 2u) ||
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
        !initPass(mUpdateVelocitiesPass, kUpdateVelocities))
    {
        LOG_ERROR_MESSAGE("PhysicsPassDispatcher: failed to initialize compute passes.");
        return false;
    }

    auto createConstantsBuffer = [&](const char* name, std::size_t size,
                                     Diligent::RefCntAutoPtr<Diligent::IBuffer>& buffer) -> bool
    {
        Diligent::BufferDesc constantsDesc{};
        constantsDesc.Name                 = name;
        constantsDesc.Size                 = static_cast<Diligent::Uint64>(size);
        constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
        constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
        constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
        constantsDesc.ImmediateContextMask = mPhysicsContextMask;
        renderDevice->CreateBuffer(constantsDesc, nullptr, &buffer);
        return buffer != nullptr;
    };

    return createConstantsBuffer("CRESSimNeo.Physics.RigidDispatchConstants",
                                 sizeof(GpuRigidDispatchConstants),
                                 mRigidDispatchConstantsBuffer) &&
           createConstantsBuffer("CRESSimNeo.Physics.ScanConstants",
                                 sizeof(GpuPhysicsScanConstants), mScanConstantsBuffer) &&
           createConstantsBuffer("CRESSimNeo.Physics.RadixConstants",
                                 sizeof(GpuPhysicsRadixConstants), mRadixConstantsBuffer);
}

bool PhysicsPassDispatcher::dispatchScanBlockPass(Diligent::IDeviceContext* computeContext,
                                                  const PhysicsSceneGpuState&,
                                                  Diligent::IBuffer* input,
                                                  Diligent::IBuffer* output,
                                                  Diligent::IBuffer* blockSums,
                                                  std::uint32_t count)
{
    if (count == 0u)
    {
        return false;
    }

    const std::array bindings{
        ComputeBufferBinding{"PhysicsScanConstantsBuffer", mScanConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ScanInput", input, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ScanOutput", output, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_BlockSums", blockSums, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const GpuPhysicsScanConstants scanConstants = makeScanConstants(count);
    return writeAndDispatchScan(computeContext, mScanBlockPass, kDefaultVariant, bindings,
                                &scanConstants, dispatchGroupCount(count));
}

bool PhysicsPassDispatcher::dispatchScanAddOffsetsPass(Diligent::IDeviceContext* computeContext,
                                                       Diligent::IBuffer* output,
                                                       Diligent::IBuffer* scannedBlockOffsets,
                                                       std::uint32_t count)
{
    if (count == 0u)
    {
        return false;
    }

    const std::array bindings{
        ComputeBufferBinding{"PhysicsScanConstantsBuffer", mScanConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ScannedBlockOffsets", scannedBlockOffsets,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ScanOutput", output, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const GpuPhysicsScanConstants scanConstants = makeScanConstants(count);
    return writeAndDispatchScan(computeContext, mScanAddOffsetsPass, kDefaultVariant, bindings,
                                &scanConstants, dispatchGroupCount(count));
}

bool PhysicsPassDispatcher::dispatchExclusiveScanPass(
    Diligent::IDeviceContext* computeContext, const PhysicsSceneGpuState& sceneState,
    Diligent::IBuffer* input, Diligent::IBuffer* output, std::uint32_t count,
    std::uint32_t recursionLevel)
{
    if (count == 0u)
    {
        return true;
    }

    const auto& transientState = sceneState.transientBuffers();
    if (recursionLevel >= transientState.scanBlockSumsBuffers.size() ||
        recursionLevel >= transientState.scanScannedBlockSumsBuffers.size())
    {
        return false;
    }

    Diligent::IBuffer* blockSums = transientState.scanBlockSumsBuffers[recursionLevel];
    Diligent::IBuffer* scannedBlockSums =
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

bool PhysicsPassDispatcher::clearCorrections(Diligent::IDeviceContext* computeContext,
                                             PhysicsSceneGpuState& sceneState,
                                             std::uint32_t bodyCount,
                                             const GpuRigidDispatchConstants& constants)
{
    if (bodyCount == 0u)
    {
        sceneState.setCorrectionBuffersNeedClear(false);
        return true;
    }

    const auto& transientState = sceneState.transientBuffers();
    const std::array bindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyTranslationCorrections",
                             transientState.translationCorrectionsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_RigidBodyRotationCorrections",
                             transientState.rotationCorrectionsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!writeAndDispatchRigid(computeContext, mClearCorrectionsPass, kDefaultVariant, bindings,
                               &constants, dispatchGroupCount(bodyCount)))
    {
        return false;
    }

    sceneState.setCorrectionBuffersNeedClear(false);
    return true;
}

bool PhysicsPassDispatcher::predict(Diligent::IDeviceContext* computeContext,
                                    const PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                                    const GpuRigidDispatchConstants& constants)
{
    if (bodyCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();
    const std::array bindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyPositionsInvMass", persistent.positionsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyOrientations", persistent.orientationsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyLinearVelocities", persistent.linearVelocitiesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyAngularVelocities", persistent.angularVelocitiesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyKinematicTargetPositions",
                             persistent.kinematicTargetPositionsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyKinematicTargetOrientations",
                             persistent.kinematicTargetOrientationsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyKinematicTargetFlags",
                             persistent.kinematicTargetFlagsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                             transient.previousRigidBodies.positionsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PreviousRigidBodyOrientations",
                             transient.previousRigidBodies.orientationsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                             transient.predictedRigidBodies.positionsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PredictedRigidBodyOrientations",
                             transient.predictedRigidBodies.orientationsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                             transient.predictedRigidBodies.linearVelocitiesBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                             transient.predictedRigidBodies.angularVelocitiesBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeAndDispatchRigid(computeContext, mPredictPass, kDefaultVariant, bindings,
                                 &constants, dispatchGroupCount(bodyCount));
}

bool PhysicsPassDispatcher::updateWorldAabbs(Diligent::IDeviceContext* computeContext,
                                             const PhysicsSceneGpuState& sceneState,
                                             std::uint32_t bodyCount,
                                             const GpuRigidDispatchConstants& constants)
{
    if (bodyCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();
    const std::array bindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                             transient.predictedRigidBodies.positionsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PredictedRigidBodyOrientations",
                             transient.predictedRigidBodies.orientationsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyScales", persistent.scalesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyColliderShapeTypes", persistent.colliderShapeTypesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyColliderParams", persistent.colliderParamsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_BodyMeta", transient.bodyMetaBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_ActiveBodyFlags", transient.activeBodyFlagsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_StaticBodyFlags", transient.staticBodyFlagsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeAndDispatchRigid(computeContext, mUpdateWorldAabbsPass, kDefaultVariant, bindings,
                                 &constants, dispatchGroupCount(bodyCount));
}

bool PhysicsPassDispatcher::compactActiveBodies(Diligent::IDeviceContext* computeContext,
                                                const PhysicsSceneGpuState& sceneState,
                                                std::uint32_t bodyCount,
                                                const GpuRigidDispatchConstants& constants)
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

    if (!dispatchExclusiveScanPass(computeContext, sceneState,
                                   sceneState.transientBuffers().activeBodyFlagsBuffer,
                                   sceneState.transientBuffers().activeBodyOffsetsBuffer,
                                   bodyCount))
    {
        return false;
    }
    if (!dispatchExclusiveScanPass(computeContext, sceneState,
                                   sceneState.transientBuffers().staticBodyFlagsBuffer,
                                   sceneState.transientBuffers().staticBodyOffsetsBuffer,
                                   bodyCount))
    {
        return false;
    }

    const auto& transient = sceneState.transientBuffers();

    const std::array compactBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyFlags", transient.activeBodyFlagsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyOffsets", transient.activeBodyOffsetsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyIndices", transient.activeBodyIndicesBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_BodyMeta", transient.bodyMetaBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::array staticCompactBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyFlags", transient.staticBodyFlagsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyOffsets", transient.staticBodyOffsetsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyIndices", transient.staticBodyIndicesBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_BodyMeta", transient.bodyMetaBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::array finalizeBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyFlags", transient.activeBodyFlagsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyOffsets", transient.activeBodyOffsetsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_StaticBodyFlags", transient.staticBodyFlagsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_StaticBodyOffsets", transient.staticBodyOffsetsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!writeAndDispatchRigid(computeContext, mCompactActiveBodiesPass, kDefaultVariant,
                               compactBindings, &constants, dispatchGroupCount(bodyCount)))
    {
        return false;
    }

    if (!writeAndDispatchRigid(computeContext, mCompactActiveBodiesPass, kAltVariant,
                               staticCompactBindings, &constants, dispatchGroupCount(bodyCount)))
    {
        return false;
    }

    return writeAndDispatchRigid(computeContext, mFinalizeActiveBodiesPass, kDefaultVariant,
                                 finalizeBindings, &constants, 1u);
}

bool PhysicsPassDispatcher::dispatchReduceBroadPhaseExtentPass(
    Diligent::IDeviceContext* computeContext, const PhysicsSceneGpuState& sceneState,
    std::uint32_t activeMovingCount, bool useStaticSet)
{
    const auto& transient = sceneState.transientBuffers();
    if (computeContext == nullptr || activeMovingCount == 0u ||
        (useStaticSet ? transient.staticBroadPhaseExtentScratchBuffers.empty()
                      : transient.broadPhaseExtentScratchBuffers.empty()) ||
        (useStaticSet ? transient.staticGlobalBroadPhaseExtentBuffer == nullptr
                      : transient.globalBroadPhaseExtentBuffer == nullptr))
    {
        return false;
    }

    const std::uint32_t initialGroupCount = dispatchGroupCount(activeMovingCount);
    Diligent::IBuffer* currentOutput =
        (initialGroupCount <= 1u)
            ? (useStaticSet ? transient.staticGlobalBroadPhaseExtentBuffer
                            : transient.globalBroadPhaseExtentBuffer)
            : (useStaticSet ? transient.staticBroadPhaseExtentScratchBuffers.front()
                            : transient.broadPhaseExtentScratchBuffers.front());

    GpuRigidDispatchConstants reductionConstants{};
    reductionConstants.activeMovingCount     = activeMovingCount;
    reductionConstants.staticBodyCount       = activeMovingCount;
    reductionConstants.candidatePairCapacity = activeMovingCount;

    const std::size_t variantIndex = kDefaultVariant;

    const std::array firstBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BroadPhaseElements",
                             useStaticSet ? transient.staticBroadPhaseElementsBuffer
                                          : transient.broadPhaseElementsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_GroupExtents", currentOutput,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    if (!writeAndDispatchRigid(computeContext, mReduceExtentElementsPass, variantIndex,
                               firstBindings, &reductionConstants,
                               dispatchGroupCount(activeMovingCount)))
    {
        return false;
    }

    std::uint32_t currentCount      = dispatchGroupCount(activeMovingCount);
    std::uint32_t level             = 1u;
    Diligent::IBuffer* currentInput = currentOutput;

    while (currentCount > 1u)
    {
        const std::uint32_t nextGroupCount = dispatchGroupCount(currentCount);
        const auto& scratchBuffers = useStaticSet ? transient.staticBroadPhaseExtentScratchBuffers
                                                  : transient.broadPhaseExtentScratchBuffers;

        if (nextGroupCount > 1u && level >= scratchBuffers.size())
        {
            return false;
        }

        currentOutput = (nextGroupCount <= 1u)
                            ? (useStaticSet ? transient.staticGlobalBroadPhaseExtentBuffer
                                            : transient.globalBroadPhaseExtentBuffer)
                            : scratchBuffers[level];

        reductionConstants.candidatePairCapacity = currentCount;

        const std::array reduceBindings{
            ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer",
                                 mRigidDispatchConstantsBuffer,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_InputExtents", currentInput,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_OutputExtents", currentOutput,
                                 Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeAndDispatchRigid(computeContext, mReduceExtentExtentsPass, variantIndex,
                                   reduceBindings, &reductionConstants,
                                   dispatchGroupCount(currentCount)))
        {
            return false;
        }

        currentInput = currentOutput;
        currentCount = nextGroupCount;
        ++level;
    }

    return true;
}

bool PhysicsPassDispatcher::dispatchRadixSortPass(Diligent::IDeviceContext* computeContext,
                                                  const PhysicsSceneGpuState& sceneState,
                                                  std::uint32_t activeMovingCount,
                                                  bool useStaticSet,
                                                  const GpuRigidDispatchConstants& constants)
{
    (void)constants;

    if (computeContext == nullptr || activeMovingCount == 0u)
    {
        return false;
    }

    const auto& transient = sceneState.transientBuffers();
    Diligent::IBuffer* currentInput =
        useStaticSet ? transient.staticMortonCodesBuffer : transient.mortonCodesBuffer;
    Diligent::IBuffer* currentOutput = useStaticSet ? transient.staticMortonCodesScratchBuffer
                                                    : transient.mortonCodesScratchBuffer;

    for (std::uint32_t bit = 0u; bit < 32u; ++bit)
    {
        const GpuPhysicsRadixConstants radixConstants = makeRadixConstants(activeMovingCount, bit);

        const std::array classifyBindings{
            ComputeBufferBinding{"PhysicsRadixConstantsBuffer", mRadixConstantsBuffer,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_MortonCodesIn", currentInput,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_RadixBitFlags",
                                 useStaticSet ? transient.staticRadixBitFlagsBuffer
                                              : transient.radixBitFlagsBuffer,
                                 Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeAndDispatchRadix(computeContext, mRadixClassifyPass, kDefaultVariant,
                                   classifyBindings, &radixConstants,
                                   dispatchGroupCount(activeMovingCount)))
        {
            return false;
        }

        if (!dispatchExclusiveScanPass(computeContext, sceneState,
                                       useStaticSet ? transient.staticRadixBitFlagsBuffer
                                                    : transient.radixBitFlagsBuffer,
                                       useStaticSet ? transient.staticRadixBitOffsetsBuffer
                                                    : transient.radixBitOffsetsBuffer,
                                       activeMovingCount))
        {
            return false;
        }

        const std::array finalizeBindings{
            ComputeBufferBinding{"PhysicsRadixConstantsBuffer", mRadixConstantsBuffer,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_RadixBitFlags",
                                 useStaticSet ? transient.staticRadixBitFlagsBuffer
                                              : transient.radixBitFlagsBuffer,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_RadixBitOffsets",
                                 useStaticSet ? transient.staticRadixBitOffsetsBuffer
                                              : transient.radixBitOffsetsBuffer,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_RadixMeta",
                                 useStaticSet ? transient.staticRadixMetaBuffer
                                              : transient.radixMetaBuffer,
                                 Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeAndDispatchRadix(computeContext, mRadixFinalizePass, kDefaultVariant,
                                   finalizeBindings, &radixConstants, 1u))
        {
            return false;
        }

        const std::array scatterBindings{
            ComputeBufferBinding{"PhysicsRadixConstantsBuffer", mRadixConstantsBuffer,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_MortonCodesIn", currentInput,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_RadixBitFlags",
                                 useStaticSet ? transient.staticRadixBitFlagsBuffer
                                              : transient.radixBitFlagsBuffer,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_RadixBitOffsets",
                                 useStaticSet ? transient.staticRadixBitOffsetsBuffer
                                              : transient.radixBitOffsetsBuffer,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_RadixMeta",
                                 useStaticSet ? transient.staticRadixMetaBuffer
                                              : transient.radixMetaBuffer,
                                 Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            ComputeBufferBinding{"g_MortonCodesOut", currentOutput,
                                 Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };

        if (!writeAndDispatchRadix(computeContext, mRadixScatterPass, kDefaultVariant,
                                   scatterBindings, &radixConstants,
                                   dispatchGroupCount(activeMovingCount)))
        {
            return false;
        }

        std::swap(currentInput, currentOutput);
    }

    Diligent::IBuffer* finalMortonBuffer =
        useStaticSet ? transient.staticMortonCodesBuffer : transient.mortonCodesBuffer;
    if (currentInput != finalMortonBuffer)
    {
        const Diligent::Uint64 bytes =
            static_cast<Diligent::Uint64>(activeMovingCount) * sizeof(GpuMortonCodeElement);
        computeContext->CopyBuffer(
            currentInput, 0u, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
            finalMortonBuffer, 0u, bytes, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    return true;
}

bool PhysicsPassDispatcher::buildBroadPhase(Diligent::IDeviceContext* computeContext,
                                            const PhysicsSceneGpuState& sceneState,
                                            std::uint32_t activeMovingCount,
                                            const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (activeMovingCount == 0u)
    {
        return true;
    }

    const auto& transient = sceneState.transientBuffers();

    const std::array buildElementsBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyIndices", transient.activeBodyIndicesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BroadPhaseElements", transient.broadPhaseElementsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeAndDispatchRigid(computeContext, mBuildBroadPhaseElementsPass, kDefaultVariant,
                               buildElementsBindings, &constants,
                               dispatchGroupCount(activeMovingCount)))
    {
        return false;
    }

    if (!dispatchReduceBroadPhaseExtentPass(computeContext, sceneState, activeMovingCount, false))
    {
        return false;
    }

    const std::array mortonBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BroadPhaseElements", transient.broadPhaseElementsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_GlobalExtent", transient.globalBroadPhaseExtentBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_MortonCodes", transient.mortonCodesBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeAndDispatchRigid(computeContext, mMortonCodesPass, kDefaultVariant, mortonBindings,
                               &constants, dispatchGroupCount(activeMovingCount)))
    {
        return false;
    }

    if (!dispatchRadixSortPass(computeContext, sceneState, activeMovingCount, false, constants))
    {
        return false;
    }

    const std::array bvhHierarchyBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_SortedMortonCodes", transient.mortonCodesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BroadPhaseElements", transient.broadPhaseElementsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_BvhConstructionInfos", transient.bvhConstructionInfoBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeAndDispatchRigid(computeContext, mBvhHierarchyPass, kDefaultVariant,
                               bvhHierarchyBindings, &constants,
                               dispatchGroupCount(activeMovingCount)))
    {
        return false;
    }

    const std::array bvhBoundsBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_BvhConstructionInfos", transient.bvhConstructionInfoBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeAndDispatchRigid(computeContext, mBvhBoundingBoxesPass, kDefaultVariant,
                               bvhBoundsBindings, &constants,
                               dispatchGroupCount(activeMovingCount)))
    {
        return false;
    }

    if (constants.staticBodyCount == 0u || !sceneState.staticBroadPhaseDirty())
    {
        return true;
    }

    GpuRigidDispatchConstants staticConstants = constants;
    staticConstants.activeMovingCount         = constants.staticBodyCount;

    const std::array staticBuildElementsBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyIndices", transient.staticBodyIndicesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BroadPhaseElements", transient.staticBroadPhaseElementsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeAndDispatchRigid(computeContext, mBuildBroadPhaseElementsPass, kAltVariant,
                               staticBuildElementsBindings, &staticConstants,
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
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BroadPhaseElements", transient.staticBroadPhaseElementsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_GlobalExtent", transient.staticGlobalBroadPhaseExtentBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_MortonCodes", transient.staticMortonCodesBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeAndDispatchRigid(computeContext, mMortonCodesPass, kAltVariant,
                               staticMortonBindings, &staticConstants,
                               dispatchGroupCount(constants.staticBodyCount)))
    {
        return false;
    }

    if (!dispatchRadixSortPass(computeContext, sceneState, constants.staticBodyCount, true,
                               staticConstants))
    {
        return false;
    }

    const std::array staticBvhHierarchyBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_SortedMortonCodes", transient.staticMortonCodesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BroadPhaseElements", transient.staticBroadPhaseElementsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BvhNodes", transient.staticBvhBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_BvhConstructionInfos", transient.staticBvhConstructionInfoBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeAndDispatchRigid(computeContext, mBvhHierarchyPass, kAltVariant,
                               staticBvhHierarchyBindings, &staticConstants,
                               dispatchGroupCount(constants.staticBodyCount)))
    {
        return false;
    }

    const std::array staticBvhBoundsBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BvhNodes", transient.staticBvhBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_BvhConstructionInfos", transient.staticBvhConstructionInfoBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return writeAndDispatchRigid(computeContext, mBvhBoundingBoxesPass, kAltVariant,
                                 staticBvhBoundsBindings, &staticConstants,
                                 dispatchGroupCount(constants.staticBodyCount));
}

bool PhysicsPassDispatcher::finalizeBroadPhasePairs(Diligent::IDeviceContext* computeContext,
                                                    const PhysicsSceneGpuState& sceneState,
                                                    std::uint32_t activeMovingCount,
                                                    const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (activeMovingCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();

    const std::array countBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyIndices", transient.activeBodyIndicesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyColliderShapeTypes", persistent.colliderShapeTypesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairCountsSphereSphere", transient.pairCountBuffers[0],
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PairCountsSphereBox", transient.pairCountBuffers[1],
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PairCountsSphereCapsule", transient.pairCountBuffers[2],
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PairCountsBoxBox", transient.pairCountBuffers[3],
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PairCountsBoxCapsule", transient.pairCountBuffers[4],
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PairCountsCapsuleCapsule", transient.pairCountBuffers[5],
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeAndDispatchRigid(computeContext, mCountPairsPass, kAltVariant, countBindings,
                               &constants, dispatchGroupCount(activeMovingCount)))
    {
        return false;
    }

    for (std::uint32_t pairType = 0u; pairType < kRigidPairTypeCount; ++pairType)
    {
        if (!dispatchExclusiveScanPass(computeContext, sceneState,
                                       transient.pairCountBuffers[pairType],
                                       transient.pairOffsetBuffers[pairType],
                                       activeMovingCount))
        {
            return false;
        }
    }

    const std::array finalizeBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairCountsSphereSphere", transient.pairCountBuffers[0],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairCountsSphereBox", transient.pairCountBuffers[1],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairCountsSphereCapsule", transient.pairCountBuffers[2],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairCountsBoxBox", transient.pairCountBuffers[3],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairCountsBoxCapsule", transient.pairCountBuffers[4],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairCountsCapsuleCapsule", transient.pairCountBuffers[5],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsSphereSphere", transient.pairOffsetBuffers[0],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsSphereBox", transient.pairOffsetBuffers[1],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsSphereCapsule", transient.pairOffsetBuffers[2],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsBoxBox", transient.pairOffsetBuffers[3],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsBoxCapsule", transient.pairOffsetBuffers[4],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsCapsuleCapsule", transient.pairOffsetBuffers[5],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_BroadPhaseMeta", transient.broadPhaseMetaBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeAndDispatchRigid(computeContext, mFinalizePairsPass, kDefaultVariant,
                                 finalizeBindings, &constants, 1u);
}

bool PhysicsPassDispatcher::emitBroadPhasePairs(Diligent::IDeviceContext* computeContext,
                                                const PhysicsSceneGpuState& sceneState,
                                                std::uint32_t activeMovingCount,
                                                const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (activeMovingCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();

    const std::array emitBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_ActiveBodyIndices", transient.activeBodyIndicesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BodyAabbs", transient.bodyAabbsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_BvhNodes", transient.bvhBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_StaticBvhNodes", transient.staticBvhBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyColliderShapeTypes", persistent.colliderShapeTypesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsSphereSphere", transient.pairOffsetBuffers[0],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsSphereBox", transient.pairOffsetBuffers[1],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsSphereCapsule", transient.pairOffsetBuffers[2],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsBoxBox", transient.pairOffsetBuffers[3],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsBoxCapsule", transient.pairOffsetBuffers[4],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PairOffsetsCapsuleCapsule", transient.pairOffsetBuffers[5],
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_CandidatePairs", transient.candidatePairsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!writeAndDispatchRigid(computeContext, mEmitPairsPass, kAltVariant, emitBindings,
                               &constants, dispatchGroupCount(activeMovingCount)))
    {
        return false;
    }

    const std::array chunkBindings{
        ComputeBufferBinding{"g_RigidPairRanges", transient.rigidPairRangesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_NarrowPhaseChunks", transient.narrowPhaseChunksBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_NarrowPhaseMeta", transient.narrowPhaseMetaBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_NarrowPhaseChunkCounter", transient.narrowPhaseChunkCounterBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return writeAndDispatchRigid(computeContext, mBuildNarrowPhaseChunksPass, kDefaultVariant,
                                 chunkBindings, nullptr, 1u);
}

bool PhysicsPassDispatcher::dispatchGenerateContactsPass(Diligent::IDeviceContext* computeContext,
                                                         const PhysicsSceneGpuState& sceneState,
                                                         std::uint32_t pairCount)
{
    if (pairCount == 0u)
    {
        return true;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();

    const std::array bindings{
        ComputeBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                             transient.predictedRigidBodies.positionsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PredictedRigidBodyOrientations",
                             transient.predictedRigidBodies.orientationsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyScales", persistent.scalesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyColliderParams", persistent.colliderParamsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_CandidatePairs", transient.candidatePairsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_NarrowPhaseChunks", transient.narrowPhaseChunksBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_NarrowPhaseMeta", transient.narrowPhaseMetaBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_NarrowPhaseChunkCounter", transient.narrowPhaseChunkCounterBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_RigidContacts", transient.contactsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::uint32_t dispatchGroupUpperBound =
        ((pairCount + kNarrowPhaseChunkSize - 1u) / kNarrowPhaseChunkSize) +
        (kRigidPairTypeCount - 1u);

    return writeAndDispatchRigid(computeContext, mGenerateContactsPass, kDefaultVariant, bindings,
                                 nullptr, dispatchGroupUpperBound);
}

bool PhysicsPassDispatcher::generateContacts(Diligent::IDeviceContext* computeContext,
                                             const PhysicsSceneGpuState& sceneState,
                                             std::uint32_t pairCount,
                                             const GpuRigidDispatchConstants& constants)
{
    (void)constants;

    return dispatchGenerateContactsPass(computeContext, sceneState, pairCount);
}

bool PhysicsPassDispatcher::dispatchSolveGatherPass(Diligent::IDeviceContext* computeContext,
                                                    const PhysicsSceneGpuState& sceneState,
                                                    std::uint32_t pairCount)
{
    if (pairCount == 0u)
    {
        return false;
    }

    const auto& persistent = sceneState.persistentRigidBodies();
    const auto& transient  = sceneState.transientBuffers();

    const std::array bindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                             transient.predictedRigidBodies.positionsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PredictedRigidBodyOrientations",
                             transient.predictedRigidBodies.orientationsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyInverseInertiaLocal", persistent.inverseInertiaLocalBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidContacts", transient.contactsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyTranslationCorrections",
                             transient.translationCorrectionsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    const std::uint32_t contactSlotCount = pairCount * kRigidContactsPerPair;
    return writeAndDispatchRigid(computeContext, mSolveGatherPass, kDefaultVariant, bindings,
                                 nullptr, dispatchGroupCount(contactSlotCount));
}

bool PhysicsPassDispatcher::solveConstraints(Diligent::IDeviceContext* computeContext,
                                             const PhysicsSceneGpuState& sceneState,
                                             std::uint32_t rigidBodyCount, std::uint32_t pairCount,
                                             std::uint32_t iterations,
                                             const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr)
    {
        return false;
    }
    if (pairCount == 0u || iterations == 0u)
    {
        return true;
    }

    const auto& transient = sceneState.transientBuffers();
    const std::array applyBindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                             transient.predictedRigidBodies.positionsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PredictedRigidBodyOrientations",
                             transient.predictedRigidBodies.orientationsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_RigidBodyTypes", sceneState.persistentRigidBodies().bodyTypesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyTranslationCorrections",
                             transient.translationCorrectionsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_RigidBodyRotationCorrections", transient.rotationCorrectionsBuffer,
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

bool PhysicsPassDispatcher::updateVelocities(Diligent::IDeviceContext* computeContext,
                                             const PhysicsSceneGpuState& sceneState,
                                             std::uint32_t bodyCount,
                                             const GpuRigidDispatchConstants& constants)
{
    if (bodyCount == 0u)
    {
        return true;
    }

    const auto& transient  = sceneState.transientBuffers();
    const auto& persistent = sceneState.persistentRigidBodies();

    const std::array bindings{
        ComputeBufferBinding{"PhysicsRigidDispatchConstantsBuffer", mRigidDispatchConstantsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                             transient.previousRigidBodies.positionsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PreviousRigidBodyOrientations",
                             transient.previousRigidBodies.orientationsBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_RigidBodyTypes", persistent.bodyTypesBuffer,
                             Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        ComputeBufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                             transient.predictedRigidBodies.positionsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PredictedRigidBodyOrientations",
                             transient.predictedRigidBodies.orientationsBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PredictedRigidBodyLinearVelocities",
                             transient.predictedRigidBodies.linearVelocitiesBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        ComputeBufferBinding{"g_PredictedRigidBodyAngularVelocities",
                             transient.predictedRigidBodies.angularVelocitiesBuffer,
                             Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return writeAndDispatchRigid(computeContext, mUpdateVelocitiesPass, kDefaultVariant, bindings,
                                 &constants, dispatchGroupCount(bodyCount));
}

} // namespace cressim::neo::physics
