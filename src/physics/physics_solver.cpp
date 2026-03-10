#include "physics/physics_solver.h"

#include "physics/rigid_body_common.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/ShaderResourceVariable.h"
#include "gpu/shader_library.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"
#include "DiligentEngine/DiligentCore/Primitives/interface/Errors.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace cressim::neo::physics
{

namespace
{

constexpr std::uint32_t kComputeThreadGroupSize = 64u;

constexpr std::size_t stageIndex(PhysicsSolverStage stage)
{
    return static_cast<std::size_t>(stage);
}

void markStage(PhysicsSolverStageStats& stats, PhysicsSolverStage stage, bool executed)
{
    stats.executed[stageIndex(stage)] = executed;
    if (executed)
    {
        ++stats.dispatchedStages;
    }
    else
    {
        ++stats.skippedStages;
    }
}

Diligent::Uint64 contextMaskForId(std::uint32_t contextId)
{
    return static_cast<Diligent::Uint64>(1ull) << contextId;
}

bool ensureStructuredBuffer(Diligent::IRenderDevice* renderDevice, const char* name,
                            std::uint32_t elementStride, std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags, Diligent::USAGE usage,
                            Diligent::CPU_ACCESS_FLAGS cpuAccess,
                            Diligent::Uint64 immediateContextMask,
                            Diligent::RefCntAutoPtr<Diligent::IBuffer>& outBuffer)
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

bool writeDispatchConstants(Diligent::IDeviceContext* computeContext,
                            Diligent::IBuffer* constantsBuffer,
                            const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr || constantsBuffer == nullptr)
    {
        return false;
    }

    void* mapped = nullptr;
    computeContext->MapBuffer(constantsBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                              mapped);
    if (mapped == nullptr)
    {
        return false;
    }

    std::memcpy(mapped, &constants, sizeof(constants));
    computeContext->UnmapBuffer(constantsBuffer, Diligent::MAP_WRITE);
    return true;
}

bool bindBufferVariable(Diligent::IShaderResourceBinding* srb, const char* variableName,
                        Diligent::IBuffer* buffer, Diligent::BUFFER_VIEW_TYPE viewType)
{
    if (srb == nullptr || buffer == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: bindBufferVariable invalid input for '", variableName,
                          "' (srb=", srb != nullptr ? "set" : "null", ", buffer=",
                          buffer != nullptr ? "set" : "null", ").");
        return false;
    }

    Diligent::IShaderResourceVariable* variable =
        srb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE, variableName);
    if (variable == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: shader variable not found: '", variableName,
                          "'. It may have been optimized out of the shader.");
        return false;
    }

    Diligent::IBufferView* view = buffer->GetDefaultView(viewType);
    if (view != nullptr)
    {
        variable->Set(view);
        return true;
    }

    variable->Set(buffer);
    return true;
}

struct BufferBinding
{
    const char* variableName;
    Diligent::IBuffer* buffer;
    Diligent::BUFFER_VIEW_TYPE viewType;
};

template <std::size_t N>
bool bindBufferVariables(Diligent::IShaderResourceBinding* srb,
                         const std::array<BufferBinding, N>& bindings)
{
    for (const BufferBinding& binding : bindings)
    {
        if (!bindBufferVariable(srb, binding.variableName, binding.buffer, binding.viewType))
        {
            return false;
        }
    }
    return true;
}

bool createComputePipeline(Diligent::IRenderDevice* renderDevice,
                           Diligent::IShaderSourceInputStreamFactory* streamFactory,
                           const char* shaderPath, const char* shaderName, const char* psoName,
                           Diligent::Uint64 immediateContextMask,
                           const Diligent::ShaderResourceVariableDesc* variables,
                           std::size_t variableCount,
                           Diligent::RefCntAutoPtr<Diligent::IPipelineState>& outPso,
                           Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>& outSrb)
{
    if (renderDevice == nullptr || streamFactory == nullptr)
    {
        return false;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.Desc.ShaderType                 = Diligent::SHADER_TYPE_COMPUTE;
    shaderCreateInfo.Desc.Name                       = shaderName;
    shaderCreateInfo.FilePath                        = shaderPath;
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::RefCntAutoPtr<Diligent::IShader> computeShader;
    renderDevice->CreateShader(shaderCreateInfo, &computeShader);
    if (computeShader == nullptr)
    {
        return false;
    }

    Diligent::ComputePipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name         = psoName;
    psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_COMPUTE;
    psoCreateInfo.PSODesc.ImmediateContextMask = immediateContextMask;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
    psoCreateInfo.PSODesc.ResourceLayout.Variables    = variables;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(variableCount);
    psoCreateInfo.pCS = computeShader;

    renderDevice->CreateComputePipelineState(psoCreateInfo, &outPso);
    if (outPso == nullptr)
    {
        return false;
    }

    outPso->CreateShaderResourceBinding(&outSrb, true);
    return outSrb != nullptr;
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

// TODO: Impl is handling too much work.

struct PhysicsSolver::Impl
{
    struct PersistentRigidBodyBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> scalesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> inverseInertiaLocalBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderShapeTypesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderParamsBuffer;
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
        Diligent::RefCntAutoPtr<Diligent::IBuffer> broadPhaseElementsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> mortonCodesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> mortonCodesScratchBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> globalBroadPhaseExtentBuffer;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IBuffer>> scanBlockSumsBuffers;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IBuffer>> scanScannedBlockSumsBuffers;
        std::vector<Diligent::RefCntAutoPtr<Diligent::IBuffer>> broadPhaseExtentScratchBuffers;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> radixBitFlagsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> radixBitOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> radixMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> bvhBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> bvhConstructionInfoBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> pairCountsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> pairOffsetsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> candidatePairsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> broadPhaseMetaBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> contactsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> translationCorrectionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> rotationCorrectionsBuffer;
    };

    struct RigidBodyReadbackBuffers
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> broadPhaseMetaBuffer;
    };

    gpu::ShaderLibrary shaderLibrary{""};

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> predictPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> predictSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> updateWorldAabbsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> updateWorldAabbsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> scanBlockPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> scanBlockSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> scanAddOffsetsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> scanAddOffsetsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> compactActiveBodiesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> compactActiveBodiesSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> finalizeActiveBodiesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> finalizeActiveBodiesSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> buildBroadPhaseElementsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> buildBroadPhaseElementsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> reduceExtentElementsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> reduceExtentElementsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> reduceExtentExtentsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> reduceExtentExtentsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mortonCodesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mortonCodesSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> radixClassifyPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> radixClassifySrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> radixFinalizePso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> radixFinalizeSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> radixScatterPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> radixScatterSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> bvhHierarchyPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> bvhHierarchySrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> bvhBoundingBoxesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> bvhBoundingBoxesSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> countPairsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> countPairsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> finalizePairsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> finalizePairsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> emitPairsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> emitPairsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> generateContactsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> generateContactsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> solveGatherPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> solveGatherSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> applyCorrectionsPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> applyCorrectionsSrb;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> updateVelocitiesPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> updateVelocitiesSrb;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> dispatchConstantsBuffer;

    PersistentRigidBodyBuffers persistentRigidBodies;
    SolverTransientBuffers transientState;
    RigidBodyReadbackBuffers readbackRigidBodies;

    std::uint32_t bufferCapacity  = 0;
    std::uint32_t broadPhaseNodeCapacity = 0;
    std::uint32_t candidatePairCapacity = 0;
    std::uint32_t contactCapacity = 0;
    PhysicsSolverStageStats stageStats{};

    bool bindPredictBuffers();
    bool bindUpdateWorldAabbsBuffers();
    bool bindScanBlockBuffers(Diligent::IBuffer* inputBuffer, Diligent::IBuffer* outputBuffer,
                              Diligent::IBuffer* blockSumsBuffer);
    bool bindScanAddOffsetsBuffers(Diligent::IBuffer* outputBuffer,
                                   Diligent::IBuffer* scannedBlockOffsetsBuffer);
    bool bindCompactActiveBodiesBuffers();
    bool bindFinalizeActiveBodiesBuffers();
    bool bindBuildBroadPhaseElementsBuffers();
    bool bindReduceExtentElementsBuffers(Diligent::IBuffer* outputBuffer);
    bool bindReduceExtentExtentsBuffers(Diligent::IBuffer* inputBuffer,
                                        Diligent::IBuffer* outputBuffer);
    bool bindMortonCodesBuffers();
    bool bindRadixClassifyBuffers(Diligent::IBuffer* mortonInputBuffer);
    bool bindRadixFinalizeBuffers();
    bool bindRadixScatterBuffers(Diligent::IBuffer* mortonInputBuffer,
                                 Diligent::IBuffer* mortonOutputBuffer);
    bool bindBvhHierarchyBuffers();
    bool bindBvhBoundingBoxesBuffers();
    bool bindCountPairsBuffers();
    bool bindFinalizePairsBuffers();
    bool bindEmitPairsBuffers();
    bool bindGenerateContactsBuffers();
    bool bindSolveGatherBuffers();
    bool bindApplyCorrectionsBuffers();
    bool bindUpdateVelocitiesBuffers();
    bool bindAllPassBuffers();
    bool ensureCapacity(Diligent::IRenderDevice* renderDevice, std::uint32_t bodyCount,
                        std::uint32_t physicsContextId);
    bool uploadPersistentRigidBodyState(Diligent::IDeviceContext* computeContext,
                                        PhysicsWorld& world, std::uint32_t bodyCount);
    bool copyPredictedRigidBodiesToPersistentState(Diligent::IDeviceContext* computeContext,
                                                   std::uint32_t bodyCount);
    bool dispatchPredictPass(Diligent::IDeviceContext* computeContext, std::uint32_t bodyCount);
    bool dispatchUpdateWorldAabbsPass(Diligent::IDeviceContext* computeContext,
                                      std::uint32_t bodyCount);
    bool dispatchScanBlockPass(Diligent::IDeviceContext* computeContext, Diligent::IBuffer* input,
                               Diligent::IBuffer* output, Diligent::IBuffer* blockSums,
                               std::uint32_t count, const GpuRigidDispatchConstants& constants);
    bool dispatchScanAddOffsetsPass(Diligent::IDeviceContext* computeContext,
                                    Diligent::IBuffer* output,
                                    Diligent::IBuffer* scannedBlockOffsets, std::uint32_t count,
                                    const GpuRigidDispatchConstants& constants);
    bool dispatchExclusiveScanPass(Diligent::IDeviceContext* computeContext, Diligent::IBuffer* input,
                                   Diligent::IBuffer* output, std::uint32_t count,
                                   const GpuRigidDispatchConstants& constants,
                                   std::uint32_t recursionLevel = 0u);
    bool dispatchCompactActiveBodiesPass(Diligent::IDeviceContext* computeContext,
                                         std::uint32_t bodyCount);
    bool dispatchFinalizeActiveBodiesPass(Diligent::IDeviceContext* computeContext);
    bool dispatchBuildBroadPhaseElementsPass(Diligent::IDeviceContext* computeContext,
                                             std::uint32_t activeDynamicCount);
    bool dispatchReduceBroadPhaseExtentPass(Diligent::IDeviceContext* computeContext,
                                            std::uint32_t activeDynamicCount);
    bool dispatchMortonCodesPass(Diligent::IDeviceContext* computeContext,
                                 std::uint32_t activeDynamicCount);
    bool dispatchRadixSortPass(Diligent::IDeviceContext* computeContext,
                               std::uint32_t activeDynamicCount,
                               const GpuRigidDispatchConstants& constants);
    bool dispatchBvhHierarchyPass(Diligent::IDeviceContext* computeContext,
                                  std::uint32_t activeDynamicCount);
    bool dispatchBvhBoundingBoxesPass(Diligent::IDeviceContext* computeContext,
                                      std::uint32_t activeDynamicCount);
    bool dispatchCountPairsPass(Diligent::IDeviceContext* computeContext,
                                std::uint32_t activeDynamicCount);
    bool dispatchFinalizePairsPass(Diligent::IDeviceContext* computeContext);
    bool dispatchEmitPairsPass(Diligent::IDeviceContext* computeContext,
                               std::uint32_t activeDynamicCount);
    bool dispatchGenerateContactsPass(Diligent::IDeviceContext* computeContext,
                                      std::uint32_t pairCount);
    bool dispatchSolveGatherPass(Diligent::IDeviceContext* computeContext,
                                 std::uint32_t bodyCount);
    bool dispatchApplyCorrectionsPass(Diligent::IDeviceContext* computeContext,
                                      std::uint32_t bodyCount);
    bool dispatchUpdateVelocitiesPass(Diligent::IDeviceContext* computeContext,
                                      std::uint32_t bodyCount);
    bool readbackBroadPhaseMetaBlocking(Diligent::IDeviceContext* computeContext,
                                        GpuBroadPhaseMeta& outMeta);
    bool readbackPredictedRigidStateBlocking(Diligent::IDeviceContext* computeContext,
                                             PhysicsWorld& world, std::uint32_t bodyCount);
};

PhysicsSolver::PhysicsSolver(gpu::GpuDevice& device, const PhysicsSolverDesc& desc)
    : mDevice(device), mDesc(desc), mImpl(std::make_unique<Impl>())
{
}

PhysicsSolver::~PhysicsSolver() = default;

bool PhysicsSolver::Impl::bindPredictBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyPositionsInvMass", persistentRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyOrientations", persistentRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyLinearVelocities", persistentRigidBodies.linearVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyAngularVelocities", persistentRigidBodies.angularVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                      transientState.previousRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PreviousRigidBodyOrientations",
                      transientState.previousRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transientState.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transientState.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyLinearVelocities",
                      transientState.predictedRigidBodies.linearVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyAngularVelocities",
                      transientState.predictedRigidBodies.angularVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(predictSrb, bindings);
}

bool PhysicsSolver::Impl::bindUpdateWorldAabbsBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transientState.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transientState.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyScales", persistentRigidBodies.scalesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyColliderShapeTypes",
                      persistentRigidBodies.colliderShapeTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyColliderParams", persistentRigidBodies.colliderParamsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyAabbs", transientState.bodyAabbsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BodyMeta", transientState.bodyMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_ActiveBodyFlags", transientState.activeBodyFlagsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(updateWorldAabbsSrb, bindings);
}

bool PhysicsSolver::Impl::bindScanBlockBuffers(Diligent::IBuffer* inputBuffer,
                                               Diligent::IBuffer* outputBuffer,
                                               Diligent::IBuffer* blockSumsBuffer)
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ScanInput", inputBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ScanOutput", outputBuffer, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BlockSums", blockSumsBuffer, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(scanBlockSrb, bindings);
}

bool PhysicsSolver::Impl::bindScanAddOffsetsBuffers(Diligent::IBuffer* outputBuffer,
                                                    Diligent::IBuffer* scannedBlockOffsetsBuffer)
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ScannedBlockOffsets", scannedBlockOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ScanOutput", outputBuffer, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(scanAddOffsetsSrb, bindings);
}

bool PhysicsSolver::Impl::bindCompactActiveBodiesBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyFlags", transientState.activeBodyFlagsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyOffsets", transientState.activeBodyOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transientState.activeBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BodyMeta", transientState.bodyMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(compactActiveBodiesSrb, bindings);
}

bool PhysicsSolver::Impl::bindFinalizeActiveBodiesBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyFlags", transientState.activeBodyFlagsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyOffsets", transientState.activeBodyOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseMeta", transientState.broadPhaseMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(finalizeActiveBodiesSrb, bindings);
}

bool PhysicsSolver::Impl::bindBuildBroadPhaseElementsBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transientState.activeBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyAabbs", transientState.bodyAabbsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transientState.broadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(buildBroadPhaseElementsSrb, bindings);
}

bool PhysicsSolver::Impl::bindReduceExtentElementsBuffers(Diligent::IBuffer* outputBuffer)
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transientState.broadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_GroupExtents", outputBuffer, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(reduceExtentElementsSrb, bindings);
}

bool PhysicsSolver::Impl::bindReduceExtentExtentsBuffers(Diligent::IBuffer* inputBuffer,
                                                         Diligent::IBuffer* outputBuffer)
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_InputExtents", inputBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_OutputExtents", outputBuffer, Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(reduceExtentExtentsSrb, bindings);
}

bool PhysicsSolver::Impl::bindMortonCodesBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transientState.broadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_GlobalExtent", transientState.globalBroadPhaseExtentBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_MortonCodes", transientState.mortonCodesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(mortonCodesSrb, bindings);
}

bool PhysicsSolver::Impl::bindRadixClassifyBuffers(Diligent::IBuffer* mortonInputBuffer)
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_MortonCodesIn", mortonInputBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RadixBitFlags", transientState.radixBitFlagsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(radixClassifySrb, bindings);
}

bool PhysicsSolver::Impl::bindRadixFinalizeBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RadixBitFlags", transientState.radixBitFlagsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RadixBitOffsets", transientState.radixBitOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RadixMeta", transientState.radixMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(radixFinalizeSrb, bindings);
}

bool PhysicsSolver::Impl::bindRadixScatterBuffers(Diligent::IBuffer* mortonInputBuffer,
                                                  Diligent::IBuffer* mortonOutputBuffer)
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_MortonCodesIn", mortonInputBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RadixBitFlags", transientState.radixBitFlagsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RadixBitOffsets", transientState.radixBitOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RadixMeta", transientState.radixMetaBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_MortonCodesOut", mortonOutputBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(radixScatterSrb, bindings);
}

bool PhysicsSolver::Impl::bindBvhHierarchyBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_SortedMortonCodes", transientState.mortonCodesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseElements", transientState.broadPhaseElementsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transientState.bvhBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BvhConstructionInfos", transientState.bvhConstructionInfoBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(bvhHierarchySrb, bindings);
}

bool PhysicsSolver::Impl::bindBvhBoundingBoxesBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transientState.bvhBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_BvhConstructionInfos", transientState.bvhConstructionInfoBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(bvhBoundingBoxesSrb, bindings);
}

bool PhysicsSolver::Impl::bindCountPairsBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transientState.activeBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyAabbs", transientState.bodyAabbsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyMeta", transientState.bodyMetaBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transientState.bvhBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairCounts", transientState.pairCountsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(countPairsSrb, bindings);
}

bool PhysicsSolver::Impl::bindFinalizePairsBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairCounts", transientState.pairCountsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsets", transientState.pairOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BroadPhaseMeta", transientState.broadPhaseMetaBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(finalizePairsSrb, bindings);
}

bool PhysicsSolver::Impl::bindEmitPairsBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_ActiveBodyIndices", transientState.activeBodyIndicesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyAabbs", transientState.bodyAabbsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BodyMeta", transientState.bodyMetaBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_BvhNodes", transientState.bvhBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PairOffsets", transientState.pairOffsetsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_CandidatePairs", transientState.candidatePairsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(emitPairsSrb, bindings);
}

bool PhysicsSolver::Impl::bindGenerateContactsBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transientState.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transientState.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyScales", persistentRigidBodies.scalesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyColliderShapeTypes",
                      persistentRigidBodies.colliderShapeTypesBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyColliderParams", persistentRigidBodies.colliderParamsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_CandidatePairs", transientState.candidatePairsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidContacts", transientState.contactsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(generateContactsSrb, bindings);
}

bool PhysicsSolver::Impl::bindSolveGatherBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transientState.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transientState.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyInverseInertiaLocal",
                      persistentRigidBodies.inverseInertiaLocalBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidContacts", transientState.contactsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_RigidBodyTranslationCorrections",
                      transientState.translationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_RigidBodyRotationCorrections", transientState.rotationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(solveGatherSrb, bindings);
}

bool PhysicsSolver::Impl::bindApplyCorrectionsBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transientState.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transientState.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_RigidBodyTranslationCorrections",
                      transientState.translationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_RigidBodyRotationCorrections", transientState.rotationCorrectionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(applyCorrectionsSrb, bindings);
}

bool PhysicsSolver::Impl::bindUpdateVelocitiesBuffers()
{
    const std::array bindings{
        BufferBinding{"PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PreviousRigidBodyPositionsInvMass",
                      transientState.previousRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PreviousRigidBodyOrientations",
                      transientState.previousRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        BufferBinding{"g_PredictedRigidBodyPositionsInvMass",
                      transientState.predictedRigidBodies.positionsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyOrientations",
                      transientState.predictedRigidBodies.orientationsBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyLinearVelocities",
                      transientState.predictedRigidBodies.linearVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        BufferBinding{"g_PredictedRigidBodyAngularVelocities",
                      transientState.predictedRigidBodies.angularVelocitiesBuffer,
                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    return bindBufferVariables(updateVelocitiesSrb, bindings);
}

bool PhysicsSolver::Impl::bindAllPassBuffers()
{
    return bindPredictBuffers() && bindUpdateWorldAabbsBuffers() &&
           bindCompactActiveBodiesBuffers() && bindFinalizeActiveBodiesBuffers() &&
           bindBuildBroadPhaseElementsBuffers() &&
           bindMortonCodesBuffers() &&
           bindRadixFinalizeBuffers() &&
           bindBvhHierarchyBuffers() &&
           bindBvhBoundingBoxesBuffers() && bindCountPairsBuffers() &&
           bindFinalizePairsBuffers() && bindEmitPairsBuffers() &&
           bindGenerateContactsBuffers() && bindSolveGatherBuffers() &&
           bindApplyCorrectionsBuffers() && bindUpdateVelocitiesBuffers();
}

bool PhysicsSolver::Impl::ensureCapacity(Diligent::IRenderDevice* renderDevice,
                                         std::uint32_t bodyCount, std::uint32_t physicsContextId)
{
    const bool hasAllBuffers =
        persistentRigidBodies.positionsBuffer != nullptr &&
        persistentRigidBodies.orientationsBuffer != nullptr &&
        persistentRigidBodies.scalesBuffer != nullptr &&
        persistentRigidBodies.linearVelocitiesBuffer != nullptr &&
        persistentRigidBodies.angularVelocitiesBuffer != nullptr &&
        persistentRigidBodies.inverseInertiaLocalBuffer != nullptr &&
        persistentRigidBodies.colliderShapeTypesBuffer != nullptr &&
        persistentRigidBodies.colliderParamsBuffer != nullptr &&
        transientState.predictedRigidBodies.positionsBuffer != nullptr &&
        transientState.predictedRigidBodies.orientationsBuffer != nullptr &&
        transientState.predictedRigidBodies.linearVelocitiesBuffer != nullptr &&
        transientState.predictedRigidBodies.angularVelocitiesBuffer != nullptr &&
        transientState.previousRigidBodies.positionsBuffer != nullptr &&
        transientState.previousRigidBodies.orientationsBuffer != nullptr &&
        transientState.bodyAabbsBuffer != nullptr &&
        transientState.bodyMetaBuffer != nullptr &&
        transientState.activeBodyFlagsBuffer != nullptr &&
        transientState.activeBodyOffsetsBuffer != nullptr &&
        transientState.activeBodyIndicesBuffer != nullptr &&
        transientState.broadPhaseElementsBuffer != nullptr &&
        transientState.mortonCodesBuffer != nullptr &&
        transientState.mortonCodesScratchBuffer != nullptr &&
        transientState.globalBroadPhaseExtentBuffer != nullptr &&
        !transientState.scanBlockSumsBuffers.empty() &&
        transientState.scanBlockSumsBuffers.front() != nullptr &&
        !transientState.scanScannedBlockSumsBuffers.empty() &&
        transientState.scanScannedBlockSumsBuffers.front() != nullptr &&
        !transientState.broadPhaseExtentScratchBuffers.empty() &&
        transientState.broadPhaseExtentScratchBuffers.front() != nullptr &&
        transientState.radixBitFlagsBuffer != nullptr &&
        transientState.radixBitOffsetsBuffer != nullptr &&
        transientState.radixMetaBuffer != nullptr &&
        transientState.bvhBuffer != nullptr &&
        transientState.bvhConstructionInfoBuffer != nullptr &&
        transientState.pairCountsBuffer != nullptr &&
        transientState.pairOffsetsBuffer != nullptr &&
        transientState.candidatePairsBuffer != nullptr &&
        transientState.broadPhaseMetaBuffer != nullptr &&
        transientState.contactsBuffer != nullptr &&
        transientState.translationCorrectionsBuffer != nullptr &&
        transientState.rotationCorrectionsBuffer != nullptr &&
        readbackRigidBodies.positionsBuffer != nullptr &&
        readbackRigidBodies.orientationsBuffer != nullptr &&
        readbackRigidBodies.linearVelocitiesBuffer != nullptr &&
        readbackRigidBodies.angularVelocitiesBuffer != nullptr &&
        readbackRigidBodies.broadPhaseMetaBuffer != nullptr;
    if (hasAllBuffers && bufferCapacity >= bodyCount)
    {
        return bindAllPassBuffers();
    }

    const std::uint32_t newCapacity = std::max<std::uint32_t>(bodyCount, 64u);
    const std::uint32_t newNodeCapacity =
        std::max<std::uint32_t>(newCapacity > 0u ? (newCapacity * 2u - 1u) : 1u, 1u);
    const std::uint32_t newCandidatePairCapacity =
        estimateRigidCandidatePairCapacity(newCapacity);
    const std::uint32_t newContactCap =
        std::max<std::uint32_t>(newCandidatePairCapacity * kRigidContactsPerPair,
                                kRigidContactsPerPair);
    const Diligent::Uint64 contextMask = contextMaskForId(physicsContextId);
    const std::vector<std::uint32_t> reductionLevelCounts =
        buildReductionLevelCounts(newCapacity);

    if (!ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PositionsInvMass",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Orientations",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Scales",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.scalesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.LinearVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.AngularVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.angularVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.InverseInertiaLocal",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.inverseInertiaLocalBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderShapeTypes",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.colliderShapeTypesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderParams",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT,
                                Diligent::CPU_ACCESS_NONE, contextMask,
                                persistentRigidBodies.colliderParamsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PreviousPositionsInvMass",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.previousRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PreviousOrientations",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.previousRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedPositionsInvMass",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.predictedRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedOrientations",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.predictedRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedLinearVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.predictedRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedAngularVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.predictedRigidBodies.angularVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BodyAabbs",
                                sizeof(GpuBodyAabb), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.bodyAabbsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BodyMeta",
                                sizeof(GpuBodyMeta), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.bodyMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveBodyFlags",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.activeBodyFlagsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveBodyOffsets",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.activeBodyOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ActiveBodyIndices",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.activeBodyIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BroadPhaseElements",
                                sizeof(GpuBroadPhaseElement), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.broadPhaseElementsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.MortonCodes",
                                sizeof(GpuMortonCodeElement), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.mortonCodesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.MortonCodesScratch",
                                sizeof(GpuMortonCodeElement), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.mortonCodesScratchBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.GlobalBroadPhaseExtent",
                                sizeof(GpuBroadPhaseExtent), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.globalBroadPhaseExtentBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RadixBitFlags",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.radixBitFlagsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RadixBitOffsets",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.radixBitOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RadixMeta",
                                sizeof(std::uint32_t), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.radixMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BvhNodes",
                                sizeof(GpuBvhNode), newNodeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.bvhBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BvhConstructionInfos",
                                sizeof(GpuBvhConstructionInfo), newNodeCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.bvhConstructionInfoBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairCounts",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.pairCountsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PairOffsets",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.pairOffsetsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.CandidatePairs",
                                sizeof(GpuCandidatePair), newCandidatePairCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.candidatePairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BroadPhaseMeta",
                                sizeof(GpuBroadPhaseMeta), 1u,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.broadPhaseMetaBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RigidContacts",
                                sizeof(GpuRigidContact), newContactCap,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.contactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.TranslationCorrections",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.translationCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.RotationCorrections",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                transientState.rotationCorrectionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedPositions.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                readbackRigidBodies.positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PredictedOrientations.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                readbackRigidBodies.orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice,
                                "CRESSimNeo.Physics.PredictedLinearVelocities.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                readbackRigidBodies.linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice,
                                "CRESSimNeo.Physics.PredictedAngularVelocities.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                readbackRigidBodies.angularVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BroadPhaseMeta.Readback",
                                sizeof(GpuBroadPhaseMeta), 1u, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                readbackRigidBodies.broadPhaseMetaBuffer))
    {
        return false;
    }

    transientState.scanBlockSumsBuffers.resize(reductionLevelCounts.size());
    transientState.scanScannedBlockSumsBuffers.resize(reductionLevelCounts.size());
    transientState.broadPhaseExtentScratchBuffers.resize(reductionLevelCounts.size());
    for (std::size_t level = 0; level < reductionLevelCounts.size(); ++level)
    {
        const std::uint32_t levelCount = reductionLevelCounts[level];
        const std::string scanSumsName =
            "CRESSimNeo.Physics.ScanBlockSums." + std::to_string(level);
        const std::string scanOffsetsName =
            "CRESSimNeo.Physics.ScanScannedBlockSums." + std::to_string(level);
        const std::string extentName =
            "CRESSimNeo.Physics.BroadPhaseExtentScratch." + std::to_string(level);
        if (!ensureStructuredBuffer(renderDevice, scanSumsName.c_str(), sizeof(std::uint32_t),
                                    levelCount,
                                    Diligent::BIND_UNORDERED_ACCESS |
                                        Diligent::BIND_SHADER_RESOURCE,
                                    Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
                                    contextMask, transientState.scanBlockSumsBuffers[level]) ||
            !ensureStructuredBuffer(renderDevice, scanOffsetsName.c_str(),
                                    sizeof(std::uint32_t), levelCount,
                                    Diligent::BIND_UNORDERED_ACCESS |
                                        Diligent::BIND_SHADER_RESOURCE,
                                    Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
                                    contextMask,
                                    transientState.scanScannedBlockSumsBuffers[level]) ||
            !ensureStructuredBuffer(renderDevice, extentName.c_str(),
                                    sizeof(GpuBroadPhaseExtent), levelCount,
                                    Diligent::BIND_UNORDERED_ACCESS |
                                        Diligent::BIND_SHADER_RESOURCE,
                                    Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
                                    contextMask,
                                    transientState.broadPhaseExtentScratchBuffers[level]))
        {
            return false;
        }
    }

    bufferCapacity = newCapacity;
    broadPhaseNodeCapacity = newNodeCapacity;
    candidatePairCapacity = newCandidatePairCapacity;
    contactCapacity = newContactCap;
    return bindAllPassBuffers();
}

bool PhysicsSolver::Impl::uploadPersistentRigidBodyState(Diligent::IDeviceContext* computeContext,
                                                         PhysicsWorld& world,
                                                         std::uint32_t bodyCount)
{
    if (computeContext == nullptr)
    {
        return false;
    }

    const RigidBodySoAHost& rigidBodies = world.rigidBodySoA();
    if (static_cast<std::uint32_t>(rigidBodies.size()) != bodyCount)
    {
        return false;
    }

    if (bodyCount == 0u)
    {
        return true;
    }

    const Diligent::Uint64 float4Bytes =
        static_cast<Diligent::Uint64>(bodyCount) * sizeof(Diligent::float4);
    const Diligent::Uint64 shapeTypeBytes =
        static_cast<Diligent::Uint64>(bodyCount) * sizeof(std::uint32_t);

    computeContext->UpdateBuffer(persistentRigidBodies.positionsBuffer, 0u, float4Bytes,
                                 rigidBodies.positionsInvMass.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.orientationsBuffer, 0u, float4Bytes,
                                 rigidBodies.orientations.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.scalesBuffer, 0u, float4Bytes,
                                 rigidBodies.scales.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.linearVelocitiesBuffer, 0u, float4Bytes,
                                 rigidBodies.linearVelocities.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.angularVelocitiesBuffer, 0u, float4Bytes,
                                 rigidBodies.angularVelocities.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.inverseInertiaLocalBuffer, 0u, float4Bytes,
                                 rigidBodies.inverseInertiaLocal.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.colliderShapeTypesBuffer, 0u, shapeTypeBytes,
                                 rigidBodies.colliderShapeTypes.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(persistentRigidBodies.colliderParamsBuffer, 0u, float4Bytes,
                                 rigidBodies.colliderParams.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    world.clearRigidBodyDirtyRange();
    return true;
}

bool PhysicsSolver::Impl::copyPredictedRigidBodiesToPersistentState(
    Diligent::IDeviceContext* computeContext, std::uint32_t bodyCount)
{
    if (computeContext == nullptr || bodyCount == 0u)
    {
        return false;
    }

    const Diligent::Uint64 bytes =
        static_cast<Diligent::Uint64>(bodyCount) * sizeof(Diligent::float4);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.positionsBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               persistentRigidBodies.positionsBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.orientationsBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               persistentRigidBodies.orientationsBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.linearVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               persistentRigidBodies.linearVelocitiesBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.angularVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               persistentRigidBodies.angularVelocitiesBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return true;
}

bool PhysicsSolver::Impl::dispatchPredictPass(Diligent::IDeviceContext* computeContext,
                                              std::uint32_t bodyCount)
{
    if (computeContext == nullptr || predictPso == nullptr || predictSrb == nullptr ||
        bodyCount == 0u)
    {
        return false;
    }

    computeContext->SetPipelineState(predictPso);
    computeContext->CommitShaderResources(predictSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(bodyCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchUpdateWorldAabbsPass(Diligent::IDeviceContext* computeContext,
                                                       std::uint32_t bodyCount)
{
    if (computeContext == nullptr || updateWorldAabbsPso == nullptr ||
        updateWorldAabbsSrb == nullptr || bodyCount == 0u)
    {
        return false;
    }

    computeContext->SetPipelineState(updateWorldAabbsPso);
    computeContext->CommitShaderResources(updateWorldAabbsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(bodyCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchScanBlockPass(Diligent::IDeviceContext* computeContext,
                                                Diligent::IBuffer* input,
                                                Diligent::IBuffer* output,
                                                Diligent::IBuffer* blockSums,
                                                std::uint32_t count,
                                                const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr || scanBlockPso == nullptr || scanBlockSrb == nullptr ||
        count == 0u || !bindScanBlockBuffers(input, output, blockSums))
    {
        return false;
    }

    GpuRigidDispatchConstants scanConstants = constants;
    scanConstants.candidatePairCapacity = count;
    if (!writeDispatchConstants(computeContext, dispatchConstantsBuffer, scanConstants))
    {
        return false;
    }

    computeContext->SetPipelineState(scanBlockPso);
    computeContext->CommitShaderResources(scanBlockSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(count), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchScanAddOffsetsPass(
    Diligent::IDeviceContext* computeContext, Diligent::IBuffer* output,
    Diligent::IBuffer* scannedBlockOffsets, std::uint32_t count,
    const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr || scanAddOffsetsPso == nullptr || scanAddOffsetsSrb == nullptr ||
        count == 0u || !bindScanAddOffsetsBuffers(output, scannedBlockOffsets))
    {
        return false;
    }

    GpuRigidDispatchConstants scanConstants = constants;
    scanConstants.candidatePairCapacity = count;
    if (!writeDispatchConstants(computeContext, dispatchConstantsBuffer, scanConstants))
    {
        return false;
    }

    computeContext->SetPipelineState(scanAddOffsetsPso);
    computeContext->CommitShaderResources(scanAddOffsetsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(count), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchExclusiveScanPass(Diligent::IDeviceContext* computeContext,
                                                    Diligent::IBuffer* input,
                                                    Diligent::IBuffer* output,
                                                    std::uint32_t count,
                                                    const GpuRigidDispatchConstants& constants,
                                                    std::uint32_t recursionLevel)
{
    if (count == 0u)
    {
        return true;
    }
    if (recursionLevel >= transientState.scanBlockSumsBuffers.size() ||
        recursionLevel >= transientState.scanScannedBlockSumsBuffers.size())
    {
        return false;
    }

    Diligent::IBuffer* blockSums = transientState.scanBlockSumsBuffers[recursionLevel];
    Diligent::IBuffer* scannedBlockSums =
        transientState.scanScannedBlockSumsBuffers[recursionLevel];
    if (!dispatchScanBlockPass(computeContext, input, output, blockSums, count, constants))
    {
        return false;
    }

    const std::uint32_t groupCount = dispatchGroupCount(count);
    if (groupCount <= 1u)
    {
        return true;
    }

    if (!dispatchExclusiveScanPass(computeContext, blockSums, scannedBlockSums, groupCount,
                                   constants, recursionLevel + 1u))
    {
        return false;
    }

    return dispatchScanAddOffsetsPass(computeContext, output, scannedBlockSums, count, constants);
}

bool PhysicsSolver::Impl::dispatchCompactActiveBodiesPass(Diligent::IDeviceContext* computeContext,
                                                          std::uint32_t bodyCount)
{
    if (computeContext == nullptr || compactActiveBodiesPso == nullptr ||
        compactActiveBodiesSrb == nullptr || bodyCount == 0u)
    {
        return false;
    }

    computeContext->SetPipelineState(compactActiveBodiesPso);
    computeContext->CommitShaderResources(compactActiveBodiesSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(bodyCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchFinalizeActiveBodiesPass(
    Diligent::IDeviceContext* computeContext)
{
    if (computeContext == nullptr || finalizeActiveBodiesPso == nullptr ||
        finalizeActiveBodiesSrb == nullptr)
    {
        return false;
    }

    computeContext->SetPipelineState(finalizeActiveBodiesPso);
    computeContext->CommitShaderResources(finalizeActiveBodiesSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(Diligent::DispatchComputeAttribs{1u, 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchBuildBroadPhaseElementsPass(
    Diligent::IDeviceContext* computeContext, std::uint32_t activeDynamicCount)
{
    if (computeContext == nullptr || buildBroadPhaseElementsPso == nullptr ||
        buildBroadPhaseElementsSrb == nullptr || activeDynamicCount == 0u)
    {
        return false;
    }

    computeContext->SetPipelineState(buildBroadPhaseElementsPso);
    computeContext->CommitShaderResources(buildBroadPhaseElementsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeDynamicCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchReduceBroadPhaseExtentPass(
    Diligent::IDeviceContext* computeContext, std::uint32_t activeDynamicCount)
{
    if (computeContext == nullptr || activeDynamicCount == 0u ||
        transientState.broadPhaseExtentScratchBuffers.empty() ||
        transientState.globalBroadPhaseExtentBuffer == nullptr)
    {
        return false;
    }

    const std::uint32_t initialGroupCount = dispatchGroupCount(activeDynamicCount);
    Diligent::IBuffer* currentOutput =
        (initialGroupCount <= 1u) ? transientState.globalBroadPhaseExtentBuffer
                                  : transientState.broadPhaseExtentScratchBuffers.front();
    GpuRigidDispatchConstants reductionConstants{};
    reductionConstants.activeDynamicCount = activeDynamicCount;
    reductionConstants.candidatePairCapacity = activeDynamicCount;
    if (!writeDispatchConstants(computeContext, dispatchConstantsBuffer, reductionConstants) ||
        reduceExtentElementsPso == nullptr || reduceExtentElementsSrb == nullptr ||
        !bindReduceExtentElementsBuffers(currentOutput))
    {
        return false;
    }

    computeContext->SetPipelineState(reduceExtentElementsPso);
    computeContext->CommitShaderResources(reduceExtentElementsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    std::uint32_t currentCount = activeDynamicCount;
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(currentCount), 1u, 1u});

    currentCount = dispatchGroupCount(currentCount);
    std::uint32_t level = 1u;
    Diligent::IBuffer* currentInput = currentOutput;
    while (currentCount > 1u)
    {
        const std::uint32_t nextGroupCount = dispatchGroupCount(currentCount);
        if (nextGroupCount > 1u && level >= transientState.broadPhaseExtentScratchBuffers.size())
        {
            return false;
        }

        currentOutput = (nextGroupCount <= 1u) ? transientState.globalBroadPhaseExtentBuffer
                                               : transientState.broadPhaseExtentScratchBuffers[level];
        reductionConstants.candidatePairCapacity = currentCount;
        if (!writeDispatchConstants(computeContext, dispatchConstantsBuffer, reductionConstants) ||
            reduceExtentExtentsPso == nullptr || reduceExtentExtentsSrb == nullptr ||
            !bindReduceExtentExtentsBuffers(currentInput, currentOutput))
        {
            return false;
        }

        computeContext->SetPipelineState(reduceExtentExtentsPso);
        computeContext->CommitShaderResources(reduceExtentExtentsSrb,
                                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->DispatchCompute(
            Diligent::DispatchComputeAttribs{dispatchGroupCount(currentCount), 1u, 1u});

        currentInput = currentOutput;
        currentCount = nextGroupCount;
        ++level;
    }

    return true;
}

bool PhysicsSolver::Impl::dispatchMortonCodesPass(Diligent::IDeviceContext* computeContext,
                                                  std::uint32_t activeDynamicCount)
{
    if (computeContext == nullptr || mortonCodesPso == nullptr || mortonCodesSrb == nullptr ||
        activeDynamicCount == 0u || !bindMortonCodesBuffers())
    {
        return false;
    }

    computeContext->SetPipelineState(mortonCodesPso);
    computeContext->CommitShaderResources(mortonCodesSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeDynamicCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchRadixSortPass(Diligent::IDeviceContext* computeContext,
                                                std::uint32_t activeDynamicCount,
                                                const GpuRigidDispatchConstants& constants)
{
    if (computeContext == nullptr || activeDynamicCount == 0u || radixClassifyPso == nullptr ||
        radixClassifySrb == nullptr || radixFinalizePso == nullptr ||
        radixFinalizeSrb == nullptr || radixScatterPso == nullptr || radixScatterSrb == nullptr)
    {
        return false;
    }

    // TODO: Replace this 1-bit classify/scan/finalize/scatter loop with a 4-bit
    // histogram/scan/scatter radix pipeline to reduce pass count and improve throughput.
    Diligent::IBuffer* currentInput = transientState.mortonCodesBuffer;
    Diligent::IBuffer* currentOutput = transientState.mortonCodesScratchBuffer;
    for (std::uint32_t bit = 0u; bit < 32u; ++bit)
    {
        GpuRigidDispatchConstants radixConstants = constants;
        radixConstants.activeDynamicCount = activeDynamicCount;
        radixConstants.candidatePairCapacity = activeDynamicCount;
        radixConstants.iterationIndex = bit;
        if (!writeDispatchConstants(computeContext, dispatchConstantsBuffer, radixConstants))
        {
            return false;
        }

        if (!bindRadixClassifyBuffers(currentInput))
        {
            return false;
        }
        computeContext->SetPipelineState(radixClassifyPso);
        computeContext->CommitShaderResources(radixClassifySrb,
                                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->DispatchCompute(
            Diligent::DispatchComputeAttribs{dispatchGroupCount(activeDynamicCount), 1u, 1u});

        if (!dispatchExclusiveScanPass(computeContext, transientState.radixBitFlagsBuffer,
                                       transientState.radixBitOffsetsBuffer, activeDynamicCount,
                                       radixConstants))
        {
            return false;
        }

        if (!bindRadixFinalizeBuffers())
        {
            return false;
        }
        computeContext->SetPipelineState(radixFinalizePso);
        computeContext->CommitShaderResources(radixFinalizeSrb,
                                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->DispatchCompute(Diligent::DispatchComputeAttribs{1u, 1u, 1u});

        if (!bindRadixScatterBuffers(currentInput, currentOutput))
        {
            return false;
        }
        computeContext->SetPipelineState(radixScatterPso);
        computeContext->CommitShaderResources(radixScatterSrb,
                                              Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        computeContext->DispatchCompute(
            Diligent::DispatchComputeAttribs{dispatchGroupCount(activeDynamicCount), 1u, 1u});

        std::swap(currentInput, currentOutput);
    }

    if (currentInput != transientState.mortonCodesBuffer)
    {
        const Diligent::Uint64 bytes =
            static_cast<Diligent::Uint64>(activeDynamicCount) * sizeof(GpuMortonCodeElement);
        computeContext->CopyBuffer(currentInput, 0u,
                                   Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                   transientState.mortonCodesBuffer, 0u, bytes,
                                   Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    return true;
}

bool PhysicsSolver::Impl::dispatchBvhHierarchyPass(Diligent::IDeviceContext* computeContext,
                                                   std::uint32_t activeDynamicCount)
{
    if (computeContext == nullptr || bvhHierarchyPso == nullptr || bvhHierarchySrb == nullptr ||
        activeDynamicCount < 2u)
    {
        return activeDynamicCount < 2u;
    }

    computeContext->SetPipelineState(bvhHierarchyPso);
    computeContext->CommitShaderResources(bvhHierarchySrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeDynamicCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchBvhBoundingBoxesPass(Diligent::IDeviceContext* computeContext,
                                                       std::uint32_t activeDynamicCount)
{
    if (computeContext == nullptr || bvhBoundingBoxesPso == nullptr ||
        bvhBoundingBoxesSrb == nullptr || activeDynamicCount < 2u)
    {
        return activeDynamicCount < 2u;
    }

    computeContext->SetPipelineState(bvhBoundingBoxesPso);
    computeContext->CommitShaderResources(bvhBoundingBoxesSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeDynamicCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchCountPairsPass(Diligent::IDeviceContext* computeContext,
                                                 std::uint32_t activeDynamicCount)
{
    if (computeContext == nullptr || countPairsPso == nullptr || countPairsSrb == nullptr ||
        activeDynamicCount == 0u)
    {
        return activeDynamicCount == 0u;
    }

    computeContext->SetPipelineState(countPairsPso);
    computeContext->CommitShaderResources(countPairsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeDynamicCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchFinalizePairsPass(Diligent::IDeviceContext* computeContext)
{
    if (computeContext == nullptr || finalizePairsPso == nullptr || finalizePairsSrb == nullptr)
    {
        return false;
    }

    computeContext->SetPipelineState(finalizePairsPso);
    computeContext->CommitShaderResources(finalizePairsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(Diligent::DispatchComputeAttribs{1u, 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchEmitPairsPass(Diligent::IDeviceContext* computeContext,
                                                std::uint32_t activeDynamicCount)
{
    if (computeContext == nullptr || emitPairsPso == nullptr || emitPairsSrb == nullptr ||
        activeDynamicCount == 0u)
    {
        return activeDynamicCount == 0u;
    }

    computeContext->SetPipelineState(emitPairsPso);
    computeContext->CommitShaderResources(emitPairsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(activeDynamicCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchGenerateContactsPass(Diligent::IDeviceContext* computeContext,
                                                       std::uint32_t pairCount)
{
    if (computeContext == nullptr || generateContactsPso == nullptr || generateContactsSrb == nullptr)
    {
        return false;
    }
    if (pairCount == 0u)
    {
        return true;
    }

    computeContext->SetPipelineState(generateContactsPso);
    computeContext->CommitShaderResources(generateContactsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(pairCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchSolveGatherPass(Diligent::IDeviceContext* computeContext,
                                                  std::uint32_t bodyCount)
{
    if (computeContext == nullptr || solveGatherPso == nullptr || solveGatherSrb == nullptr ||
        bodyCount == 0u)
    {
        return false;
    }

    computeContext->SetPipelineState(solveGatherPso);
    computeContext->CommitShaderResources(solveGatherSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(bodyCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchApplyCorrectionsPass(Diligent::IDeviceContext* computeContext,
                                                       std::uint32_t bodyCount)
{
    if (computeContext == nullptr || applyCorrectionsPso == nullptr ||
        applyCorrectionsSrb == nullptr || bodyCount == 0u)
    {
        return false;
    }

    computeContext->SetPipelineState(applyCorrectionsPso);
    computeContext->CommitShaderResources(applyCorrectionsSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(bodyCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::dispatchUpdateVelocitiesPass(Diligent::IDeviceContext* computeContext,
                                                       std::uint32_t bodyCount)
{
    if (computeContext == nullptr || updateVelocitiesPso == nullptr ||
        updateVelocitiesSrb == nullptr || bodyCount == 0u)
    {
        return false;
    }

    computeContext->SetPipelineState(updateVelocitiesPso);
    computeContext->CommitShaderResources(updateVelocitiesSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{dispatchGroupCount(bodyCount), 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::readbackBroadPhaseMetaBlocking(Diligent::IDeviceContext* computeContext,
                                                         GpuBroadPhaseMeta& outMeta)
{
    if (computeContext == nullptr || transientState.broadPhaseMetaBuffer == nullptr ||
        readbackRigidBodies.broadPhaseMetaBuffer == nullptr)
    {
        return false;
    }

    computeContext->CopyBuffer(transientState.broadPhaseMetaBuffer, 0u,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               readbackRigidBodies.broadPhaseMetaBuffer, 0u,
                               sizeof(GpuBroadPhaseMeta),
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->Flush();
    computeContext->WaitForIdle();

    void* mappedMeta = nullptr;
    computeContext->MapBuffer(readbackRigidBodies.broadPhaseMetaBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedMeta);
    if (mappedMeta == nullptr)
    {
        return false;
    }

    outMeta = *static_cast<const GpuBroadPhaseMeta*>(mappedMeta);
    computeContext->UnmapBuffer(readbackRigidBodies.broadPhaseMetaBuffer, Diligent::MAP_READ);
    return true;
}

// TODO: readback is stalling GPU; use fences or let engine do this

bool PhysicsSolver::Impl::readbackPredictedRigidStateBlocking(
    Diligent::IDeviceContext* computeContext, PhysicsWorld& world, std::uint32_t bodyCount)
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
    computeContext->CopyBuffer(transientState.predictedRigidBodies.positionsBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               readbackRigidBodies.positionsBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.orientationsBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               readbackRigidBodies.orientationsBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.linearVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               readbackRigidBodies.linearVelocitiesBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(transientState.predictedRigidBodies.angularVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               readbackRigidBodies.angularVelocitiesBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    computeContext->Flush();
    computeContext->WaitForIdle();

    void* mappedPositions    = nullptr;
    void* mappedOrientations = nullptr;
    void* mappedLinear       = nullptr;
    void* mappedAngular      = nullptr;

    // Use MAP_FLAG_DO_NOT_WAIT. Vulkan won't wait anyway.

    computeContext->MapBuffer(readbackRigidBodies.positionsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedPositions);
    computeContext->MapBuffer(readbackRigidBodies.orientationsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedOrientations);
    computeContext->MapBuffer(readbackRigidBodies.linearVelocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedLinear);
    computeContext->MapBuffer(readbackRigidBodies.angularVelocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedAngular);

    if (mappedPositions == nullptr || mappedOrientations == nullptr || mappedLinear == nullptr ||
        mappedAngular == nullptr)
    {
        if (mappedPositions != nullptr)
        {
            computeContext->UnmapBuffer(readbackRigidBodies.positionsBuffer, Diligent::MAP_READ);
        }
        if (mappedOrientations != nullptr)
        {
            computeContext->UnmapBuffer(readbackRigidBodies.orientationsBuffer, Diligent::MAP_READ);
        }
        if (mappedLinear != nullptr)
        {
            computeContext->UnmapBuffer(readbackRigidBodies.linearVelocitiesBuffer,
                                        Diligent::MAP_READ);
        }
        if (mappedAngular != nullptr)
        {
            computeContext->UnmapBuffer(readbackRigidBodies.angularVelocitiesBuffer,
                                        Diligent::MAP_READ);
        }
        return false;
    }

    const auto* positions         = static_cast<const Diligent::float4*>(mappedPositions);
    const auto* orientations      = static_cast<const Diligent::float4*>(mappedOrientations);
    const auto* linearVelocities  = static_cast<const Diligent::float4*>(mappedLinear);
    const auto* angularVelocities = static_cast<const Diligent::float4*>(mappedAngular);
    for (std::uint32_t i = 0; i < bodyCount; ++i)
    {
        (void)world.writeBackRigidBodyState(i, positions[i], orientations[i], linearVelocities[i],
                                            angularVelocities[i]);
    }
    world.finalizeRigidBodyWriteback();

    computeContext->UnmapBuffer(readbackRigidBodies.positionsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(readbackRigidBodies.orientationsBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(readbackRigidBodies.linearVelocitiesBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(readbackRigidBodies.angularVelocitiesBuffer, Diligent::MAP_READ);
    return true;
}

bool PhysicsSolver::initialize()
{
    shutdown();

    if (!mDesc.enableGpuCompute)
    {
        mInitialized = true;
        return true;
    }

    gpu::GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to get physics GPU context.");
        return false;
    }

    mImpl                = std::make_unique<Impl>();
    mImpl->shaderLibrary = gpu::ShaderLibrary(mDevice.shaderSourceDirectory());

    Diligent::IShaderSourceInputStreamFactory* streamFactory = mImpl->shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: shader stream factory is null.");
        return false;
    }
    const Diligent::Uint64 physicsContextMask = contextMaskForId(computeContext.contextId);

    constexpr Diligent::ShaderResourceVariableDesc kPredictVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_predict.cs.hlsl",
                               "CRESSimNeo.Physics.RigidPredict.CS",
                               "CRESSimNeo.Physics.RigidPredict.PSO", physicsContextMask,
                               kPredictVars,
                               std::size(kPredictVars), mImpl->predictPso, mImpl->predictSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create rigid predict compute pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kUpdateWorldAabbsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyColliderShapeTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyColliderParams",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_update_world_aabbs.cs.hlsl",
                               "CRESSimNeo.Physics.RigidUpdateWorldAabbs.CS",
                               "CRESSimNeo.Physics.RigidUpdateWorldAabbs.PSO",
                               physicsContextMask, kUpdateWorldAabbsVars,
                               std::size(kUpdateWorldAabbsVars), mImpl->updateWorldAabbsPso,
                               mImpl->updateWorldAabbsSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create rigid world AABB pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kScanBlockVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ScanInput",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ScanOutput",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BlockSums",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_scan_block.cs.hlsl",
                               "CRESSimNeo.Physics.ScanBlock.CS",
                               "CRESSimNeo.Physics.ScanBlock.PSO", physicsContextMask,
                               kScanBlockVars, std::size(kScanBlockVars),
                               mImpl->scanBlockPso, mImpl->scanBlockSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create scan block pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kScanAddOffsetsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ScannedBlockOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ScanOutput",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_scan_add_offsets.cs.hlsl",
                               "CRESSimNeo.Physics.ScanAddOffsets.CS",
                               "CRESSimNeo.Physics.ScanAddOffsets.PSO", physicsContextMask,
                               kScanAddOffsetsVars, std::size(kScanAddOffsetsVars),
                               mImpl->scanAddOffsetsPso, mImpl->scanAddOffsetsSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create scan add-offsets pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kCompactActiveBodiesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_compact_active_bodies.cs.hlsl",
                               "CRESSimNeo.Physics.RigidCompactActiveBodies.CS",
                               "CRESSimNeo.Physics.RigidCompactActiveBodies.PSO",
                               physicsContextMask, kCompactActiveBodiesVars,
                               std::size(kCompactActiveBodiesVars),
                               mImpl->compactActiveBodiesPso, mImpl->compactActiveBodiesSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create active body compaction pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kFinalizeActiveBodiesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_finalize_active_bodies.cs.hlsl",
                               "CRESSimNeo.Physics.RigidFinalizeActiveBodies.CS",
                               "CRESSimNeo.Physics.RigidFinalizeActiveBodies.PSO",
                               physicsContextMask, kFinalizeActiveBodiesVars,
                               std::size(kFinalizeActiveBodiesVars),
                               mImpl->finalizeActiveBodiesPso,
                               mImpl->finalizeActiveBodiesSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create active body finalize pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kBuildBroadPhaseElementsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_build_broad_phase_elements.cs.hlsl",
                               "CRESSimNeo.Physics.RigidBuildBroadPhaseElements.CS",
                               "CRESSimNeo.Physics.RigidBuildBroadPhaseElements.PSO",
                               physicsContextMask, kBuildBroadPhaseElementsVars,
                               std::size(kBuildBroadPhaseElementsVars),
                               mImpl->buildBroadPhaseElementsPso,
                               mImpl->buildBroadPhaseElementsSrb))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsSolver: failed to create broad-phase element build pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kReduceExtentElementsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_GroupExtents",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_reduce_extent_elements.cs.hlsl",
                               "CRESSimNeo.Physics.RigidReduceExtentElements.CS",
                               "CRESSimNeo.Physics.RigidReduceExtentElements.PSO",
                               physicsContextMask, kReduceExtentElementsVars,
                               std::size(kReduceExtentElementsVars),
                               mImpl->reduceExtentElementsPso,
                               mImpl->reduceExtentElementsSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create element extent reduction pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kReduceExtentExtentsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_InputExtents",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_OutputExtents",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_reduce_extent_extents.cs.hlsl",
                               "CRESSimNeo.Physics.RigidReduceExtentExtents.CS",
                               "CRESSimNeo.Physics.RigidReduceExtentExtents.PSO",
                               physicsContextMask, kReduceExtentExtentsVars,
                               std::size(kReduceExtentExtentsVars),
                               mImpl->reduceExtentExtentsPso,
                               mImpl->reduceExtentExtentsSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create extent reduction pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kMortonCodesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_GlobalExtent",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_morton_codes.cs.hlsl",
                               "CRESSimNeo.Physics.RigidMortonCodes.CS",
                               "CRESSimNeo.Physics.RigidMortonCodes.PSO",
                               physicsContextMask, kMortonCodesVars,
                               std::size(kMortonCodesVars), mImpl->mortonCodesPso,
                               mImpl->mortonCodesSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create morton code pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kRadixClassifyVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodesIn",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_radix_classify.cs.hlsl",
                               "CRESSimNeo.Physics.RigidRadixClassify.CS",
                               "CRESSimNeo.Physics.RigidRadixClassify.PSO",
                               physicsContextMask, kRadixClassifyVars,
                               std::size(kRadixClassifyVars), mImpl->radixClassifyPso,
                               mImpl->radixClassifySrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create radix classify pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kRadixFinalizeVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_radix_finalize.cs.hlsl",
                               "CRESSimNeo.Physics.RigidRadixFinalize.CS",
                               "CRESSimNeo.Physics.RigidRadixFinalize.PSO",
                               physicsContextMask, kRadixFinalizeVars,
                               std::size(kRadixFinalizeVars), mImpl->radixFinalizePso,
                               mImpl->radixFinalizeSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create radix finalize pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kRadixScatterVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodesIn",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixBitOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RadixMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_COMPUTE, "g_MortonCodesOut",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_radix_scatter.cs.hlsl",
                               "CRESSimNeo.Physics.RigidRadixScatter.CS",
                               "CRESSimNeo.Physics.RigidRadixScatter.PSO",
                               physicsContextMask, kRadixScatterVars,
                               std::size(kRadixScatterVars), mImpl->radixScatterPso,
                               mImpl->radixScatterSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create radix scatter pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kBvhHierarchyVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_SortedMortonCodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseElements",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhConstructionInfos",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_bvh_hierarchy.cs.hlsl",
                               "CRESSimNeo.Physics.RigidBvhHierarchy.CS",
                               "CRESSimNeo.Physics.RigidBvhHierarchy.PSO",
                               physicsContextMask, kBvhHierarchyVars,
                               std::size(kBvhHierarchyVars), mImpl->bvhHierarchyPso,
                               mImpl->bvhHierarchySrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create BVH hierarchy pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kBvhBoundingBoxesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhConstructionInfos",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_bvh_bounding_boxes.cs.hlsl",
                               "CRESSimNeo.Physics.RigidBvhBoundingBoxes.CS",
                               "CRESSimNeo.Physics.RigidBvhBoundingBoxes.PSO",
                               physicsContextMask, kBvhBoundingBoxesVars,
                               std::size(kBvhBoundingBoxesVars),
                               mImpl->bvhBoundingBoxesPso, mImpl->bvhBoundingBoxesSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create BVH bounding box pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kCountPairsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCounts",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_count_pairs.cs.hlsl",
                               "CRESSimNeo.Physics.RigidCountPairs.CS",
                               "CRESSimNeo.Physics.RigidCountPairs.PSO",
                               physicsContextMask, kCountPairsVars,
                               std::size(kCountPairsVars), mImpl->countPairsPso,
                               mImpl->countPairsSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create pair count pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kFinalizePairsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairCounts",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BroadPhaseMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_finalize_pairs.cs.hlsl",
                               "CRESSimNeo.Physics.RigidFinalizePairs.CS",
                               "CRESSimNeo.Physics.RigidFinalizePairs.PSO",
                               physicsContextMask, kFinalizePairsVars,
                               std::size(kFinalizePairsVars), mImpl->finalizePairsPso,
                               mImpl->finalizePairsSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create pair finalize pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kEmitPairsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_ActiveBodyIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyAabbs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BodyMeta",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_BvhNodes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PairOffsets",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_CandidatePairs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_emit_pairs.cs.hlsl",
                               "CRESSimNeo.Physics.RigidEmitPairs.CS",
                               "CRESSimNeo.Physics.RigidEmitPairs.PSO",
                               physicsContextMask, kEmitPairsVars,
                               std::size(kEmitPairsVars), mImpl->emitPairsPso,
                               mImpl->emitPairsSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create pair emit pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kGenerateContactsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyScales",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyColliderShapeTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyColliderParams",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_CandidatePairs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidContacts",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_generate_contacts.cs.hlsl",
                               "CRESSimNeo.Physics.RigidGenerateContacts.CS",
                               "CRESSimNeo.Physics.RigidGenerateContacts.PSO",
                               physicsContextMask,
                               kGenerateContactsVars, std::size(kGenerateContactsVars),
                               mImpl->generateContactsPso, mImpl->generateContactsSrb))
    {
        LOG_ERROR_MESSAGE(
            "PhysicsSolver: failed to create rigid contact generation compute pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kSolveGatherVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyInverseInertiaLocal",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidContacts",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_solve_gather.cs.hlsl",
                               "CRESSimNeo.Physics.RigidSolveGather.CS",
                               "CRESSimNeo.Physics.RigidSolveGather.PSO", physicsContextMask,
                               kSolveGatherVars,
                               std::size(kSolveGatherVars), mImpl->solveGatherPso,
                               mImpl->solveGatherSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create rigid solve gather pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kApplyCorrectionsVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyTranslationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyRotationCorrections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_apply_corrections.cs.hlsl",
                               "CRESSimNeo.Physics.RigidApplyCorrections.CS",
                               "CRESSimNeo.Physics.RigidApplyCorrections.PSO",
                               physicsContextMask,
                               kApplyCorrectionsVars, std::size(kApplyCorrectionsVars),
                               mImpl->applyCorrectionsPso, mImpl->applyCorrectionsSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create rigid correction apply pipeline.");
        return false;
    }

    constexpr Diligent::ShaderResourceVariableDesc kUpdateVelocitiesVars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsDispatchConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PreviousRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyLinearVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_PredictedRigidBodyAngularVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    if (!createComputePipeline(computeContext.renderDevice, streamFactory,
                               "physics/physics_rigid_update_velocities.cs.hlsl",
                               "CRESSimNeo.Physics.RigidUpdateVelocities.CS",
                               "CRESSimNeo.Physics.RigidUpdateVelocities.PSO",
                               physicsContextMask,
                               kUpdateVelocitiesVars, std::size(kUpdateVelocitiesVars),
                               mImpl->updateVelocitiesPso, mImpl->updateVelocitiesSrb))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create rigid velocity update pipeline.");
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Name                 = "CRESSimNeo.Physics.DispatchConstants";
    constantsDesc.Size                 = sizeof(GpuRigidDispatchConstants);
    constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    constantsDesc.ImmediateContextMask = contextMaskForId(computeContext.contextId);
    computeContext.renderDevice->CreateBuffer(constantsDesc, nullptr,
                                              &mImpl->dispatchConstantsBuffer);
    if (mImpl->dispatchConstantsBuffer == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create dispatch constants buffer.");
        return false;
    }

    mInitialized = true;
    return true;
}

void PhysicsSolver::shutdown()
{
    mImpl        = std::make_unique<Impl>();
    mInitialized = false;
}

bool PhysicsSolver::step(const common::FrameContext& frameContext, PhysicsWorld& world)
{
    if (!mInitialized)
    {
        return false;
    }

    mImpl->stageStats = PhysicsSolverStageStats{};

    if (!mDesc.enableGpuCompute)
    {
        markStage(mImpl->stageStats, PhysicsSolverStage::PredictState, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateWorldAabbs, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::BuildBroadPhase, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::GenerateBroadPhasePairs, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::GenerateContacts, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::SolveConstraints, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateVelocities, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults, false);
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.renderDevice == nullptr || computeBackend.computeContext == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver::step failed: missing physics backend context.");
        return false;
    }

    const std::uint32_t rigidBodyCount = world.rigidBodyCount();
    if (rigidBodyCount == 0u)
    {
        markStage(mImpl->stageStats, PhysicsSolverStage::PredictState, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateWorldAabbs, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::BuildBroadPhase, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::GenerateBroadPhasePairs, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::GenerateContacts, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::SolveConstraints, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateVelocities, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults, false);
        return true;
    }

    if (!mImpl->ensureCapacity(computeBackend.renderDevice, rigidBodyCount,
                               computeBackend.contextId))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver::step failed: ensureCapacity.");
        return false;
    }
    if (!mImpl->uploadPersistentRigidBodyState(computeBackend.computeContext, world,
                                               rigidBodyCount))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver::step failed: uploadPersistentRigidBodyState.");
        return false;
    }

    const std::uint32_t substeps   = std::max<std::uint32_t>(mDesc.substeps, 1u);
    const std::uint32_t iterations = std::max<std::uint32_t>(mDesc.solverIterations, 1u);
    const float substepDt          = frameContext.deltaSeconds / static_cast<float>(substeps);

    for (std::uint32_t substep = 0; substep < substeps; ++substep)
    {
        GpuRigidDispatchConstants constants{};
        constants.dt                    = substepDt;
        constants.rigidBodyCount        = rigidBodyCount;
        constants.activeDynamicCount    = 0u;
        constants.candidatePairCount    = 0u;
        constants.candidatePairCapacity = mImpl->candidatePairCapacity;
        constants.substepIndex          = substep;
        constants.iterationIndex        = 0u;
        constants.solverIterations      = iterations;

        // Rigid body prediction

        if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                    constants) ||
            !mImpl->dispatchPredictPass(computeBackend.computeContext, rigidBodyCount))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: PredictState dispatch.");
            return false;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::PredictState, true);

        // Rigid body broad phase

        if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                    constants) ||
            !mImpl->dispatchUpdateWorldAabbsPass(computeBackend.computeContext, rigidBodyCount))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: UpdateWorldAabbs dispatch.");
            return false;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateWorldAabbs, true);

        if (!mImpl->dispatchExclusiveScanPass(computeBackend.computeContext,
                                              mImpl->transientState.activeBodyFlagsBuffer,
                                              mImpl->transientState.activeBodyOffsetsBuffer,
                                              rigidBodyCount, constants) ||
            !mImpl->dispatchCompactActiveBodiesPass(computeBackend.computeContext, rigidBodyCount) ||
            !mImpl->dispatchFinalizeActiveBodiesPass(computeBackend.computeContext))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: BuildBroadPhase compaction dispatch.");
            return false;
        }

        // TODO: this is blocking; use indirect dispatch
        GpuBroadPhaseMeta broadPhaseMeta{};
        if (!mImpl->readbackBroadPhaseMetaBlocking(computeBackend.computeContext, broadPhaseMeta))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: readbackBroadPhaseMetaBlocking.");
            return false;
        }

        const std::uint32_t activeDynamicCount = broadPhaseMeta.activeDynamicCount;
        bool builtBroadPhase = false;
        if (activeDynamicCount > 0u)
        {
            constants.activeDynamicCount = activeDynamicCount;
            constants.candidatePairCapacity = mImpl->candidatePairCapacity;
            constants.iterationIndex = 0u;
            if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                        constants) ||
                !mImpl->dispatchBuildBroadPhaseElementsPass(computeBackend.computeContext,
                                                            activeDynamicCount) ||
                !mImpl->dispatchReduceBroadPhaseExtentPass(computeBackend.computeContext,
                                                          activeDynamicCount) ||
                !mImpl->dispatchMortonCodesPass(computeBackend.computeContext,
                                               activeDynamicCount) ||
                !mImpl->dispatchRadixSortPass(computeBackend.computeContext, activeDynamicCount,
                                             constants) ||
                !mImpl->dispatchBvhHierarchyPass(computeBackend.computeContext,
                                                 activeDynamicCount) ||
                !mImpl->dispatchBvhBoundingBoxesPass(computeBackend.computeContext,
                                                     activeDynamicCount))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: BuildBroadPhase dispatch.");
                return false;
            }
            builtBroadPhase = true;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::BuildBroadPhase, builtBroadPhase);

        std::uint32_t pairCount = 0u;
        if (activeDynamicCount > 0u)
        {
            constants.activeDynamicCount = activeDynamicCount;
            constants.candidatePairCapacity = mImpl->candidatePairCapacity;
            if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                        constants) ||
                !mImpl->dispatchCountPairsPass(computeBackend.computeContext, activeDynamicCount))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: CountPairs dispatch.");
                return false;
            }

            if (!mImpl->dispatchExclusiveScanPass(computeBackend.computeContext,
                                                  mImpl->transientState.pairCountsBuffer,
                                                  mImpl->transientState.pairOffsetsBuffer,
                                                  activeDynamicCount, constants))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: Pair offset scan dispatch.");
                return false;
            }

            constants.candidatePairCapacity = mImpl->candidatePairCapacity;
            if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                        constants) ||
                !mImpl->dispatchFinalizePairsPass(computeBackend.computeContext))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: FinalizePairs dispatch.");
                return false;
            }

            // This is blocking and early fails the simulation
            // TODO: use GPU fallback
            if (!mImpl->readbackBroadPhaseMetaBlocking(computeBackend.computeContext, broadPhaseMeta))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: pair meta readback.");
                return false;
            }
            if (broadPhaseMeta.overflow != 0u)
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: candidate pair overflow (required=",
                                  broadPhaseMeta.requiredPairCount, ", capacity=",
                                  mImpl->candidatePairCapacity, ").");
                return false;
            }

            pairCount = broadPhaseMeta.candidatePairCount;
            constants.candidatePairCount = pairCount;
            if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                        constants) ||
                !mImpl->dispatchEmitPairsPass(computeBackend.computeContext, activeDynamicCount))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: EmitPairs dispatch.");
                return false;
            }
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateBroadPhasePairs, true);
        }
        else
        {
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateBroadPhasePairs, false);
        }

        if (pairCount > 0u)
        {
            constants.activeDynamicCount = activeDynamicCount;
            constants.candidatePairCount = pairCount;
            constants.candidatePairCapacity = mImpl->candidatePairCapacity;
            if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                        constants) ||
                !mImpl->dispatchGenerateContactsPass(computeBackend.computeContext, pairCount))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: GenerateContacts dispatch.");
                return false;
            }
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateContacts, true);

            // PBD solve iteration

            for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
            {
                constants.iterationIndex = iteration;
                if (!writeDispatchConstants(computeBackend.computeContext,
                                            mImpl->dispatchConstantsBuffer, constants) ||
                    !mImpl->dispatchSolveGatherPass(computeBackend.computeContext, rigidBodyCount) ||
                    !mImpl->dispatchApplyCorrectionsPass(computeBackend.computeContext,
                                                         rigidBodyCount))
                {
                    LOG_ERROR_MESSAGE(
                        "PhysicsSolver::step failed: SolveConstraints dispatch loop.");
                    return false;
                }
            }
            markStage(mImpl->stageStats, PhysicsSolverStage::SolveConstraints, true);
        }
        else
        {
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateContacts, false);
            markStage(mImpl->stageStats, PhysicsSolverStage::SolveConstraints, false);
        }

        constants.iterationIndex = iterations;
        if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                    constants) ||
            !mImpl->dispatchUpdateVelocitiesPass(computeBackend.computeContext, rigidBodyCount))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: UpdateVelocities dispatch.");
            return false;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateVelocities, true);

        if (substep + 1u < substeps &&
            !mImpl->copyPredictedRigidBodiesToPersistentState(computeBackend.computeContext,
                                                              rigidBodyCount))
        {
            LOG_ERROR_MESSAGE(
                "PhysicsSolver::step failed: copyPredictedRigidBodiesToPersistentState.");
            return false;
        }
    }

    if (!mDesc.enableBlockingReadback)
    {
        markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults, false);
        return false;
    }

    if (!mImpl->readbackPredictedRigidStateBlocking(computeBackend.computeContext, world,
                                                    rigidBodyCount))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver::step failed: readbackPredictedRigidStateBlocking.");
        return false;
    }

    markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults, true);
    return true;
}

const PhysicsSolverStageStats& PhysicsSolver::lastStageStats() const noexcept
{
    return mImpl->stageStats;
}

} // namespace cressim::neo::physics
