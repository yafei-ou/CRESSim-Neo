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

} // namespace

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
    };

    gpu::ShaderLibrary shaderLibrary{""};

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> predictPso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> predictSrb;
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
    std::uint32_t contactCapacity = 0;
    PhysicsSolverStageStats stageStats{};

    bool bindPredictBuffers();
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
    bool dispatchGenerateContactsPass(Diligent::IDeviceContext* computeContext,
                                      std::uint32_t pairCount);
    bool dispatchSolveGatherPass(Diligent::IDeviceContext* computeContext,
                                 std::uint32_t bodyCount);
    bool dispatchApplyCorrectionsPass(Diligent::IDeviceContext* computeContext,
                                      std::uint32_t bodyCount);
    bool dispatchUpdateVelocitiesPass(Diligent::IDeviceContext* computeContext,
                                      std::uint32_t bodyCount);
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
    return bindBufferVariable(predictSrb, "PhysicsDispatchConstantsBuffer", dispatchConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(predictSrb, "g_RigidBodyPositionsInvMass",
                              persistentRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(predictSrb, "g_RigidBodyOrientations",
                              persistentRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(predictSrb, "g_RigidBodyLinearVelocities",
                              persistentRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(predictSrb, "g_RigidBodyAngularVelocities",
                              persistentRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(predictSrb, "g_PreviousRigidBodyPositionsInvMass",
                              transientState.previousRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(predictSrb, "g_PreviousRigidBodyOrientations",
                              transientState.previousRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(predictSrb, "g_PredictedRigidBodyPositionsInvMass",
                              transientState.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(predictSrb, "g_PredictedRigidBodyOrientations",
                              transientState.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(predictSrb, "g_PredictedRigidBodyLinearVelocities",
                              transientState.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(predictSrb, "g_PredictedRigidBodyAngularVelocities",
                              transientState.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS);
}

bool PhysicsSolver::Impl::bindGenerateContactsBuffers()
{
    return bindBufferVariable(generateContactsSrb, "PhysicsDispatchConstantsBuffer",
                              dispatchConstantsBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(generateContactsSrb, "g_PredictedRigidBodyPositionsInvMass",
                              transientState.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(generateContactsSrb, "g_PredictedRigidBodyOrientations",
                              transientState.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(generateContactsSrb, "g_RigidBodyScales",
                              persistentRigidBodies.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(generateContactsSrb, "g_RigidBodyColliderShapeTypes",
                              persistentRigidBodies.colliderShapeTypesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(generateContactsSrb, "g_RigidBodyColliderParams",
                              persistentRigidBodies.colliderParamsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(generateContactsSrb, "g_RigidContacts",
                              transientState.contactsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS);
}

bool PhysicsSolver::Impl::bindSolveGatherBuffers()
{
    return bindBufferVariable(solveGatherSrb, "PhysicsDispatchConstantsBuffer",
                              dispatchConstantsBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(solveGatherSrb, "g_PredictedRigidBodyPositionsInvMass",
                              transientState.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(solveGatherSrb, "g_PredictedRigidBodyOrientations",
                              transientState.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(solveGatherSrb, "g_RigidBodyInverseInertiaLocal",
                              persistentRigidBodies.inverseInertiaLocalBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(solveGatherSrb, "g_RigidContacts", transientState.contactsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(solveGatherSrb, "g_RigidBodyTranslationCorrections",
                              transientState.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(solveGatherSrb, "g_RigidBodyRotationCorrections",
                              transientState.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS);
}

bool PhysicsSolver::Impl::bindApplyCorrectionsBuffers()
{
    return bindBufferVariable(applyCorrectionsSrb, "PhysicsDispatchConstantsBuffer",
                              dispatchConstantsBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(applyCorrectionsSrb, "g_PredictedRigidBodyPositionsInvMass",
                              transientState.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(applyCorrectionsSrb, "g_PredictedRigidBodyOrientations",
                              transientState.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(applyCorrectionsSrb, "g_RigidBodyTranslationCorrections",
                              transientState.translationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(applyCorrectionsSrb, "g_RigidBodyRotationCorrections",
                              transientState.rotationCorrectionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS);
}

bool PhysicsSolver::Impl::bindUpdateVelocitiesBuffers()
{
    return bindBufferVariable(updateVelocitiesSrb, "PhysicsDispatchConstantsBuffer",
                              dispatchConstantsBuffer, Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(updateVelocitiesSrb, "g_PreviousRigidBodyPositionsInvMass",
                              transientState.previousRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(updateVelocitiesSrb, "g_PreviousRigidBodyOrientations",
                              transientState.previousRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE) &&
           bindBufferVariable(updateVelocitiesSrb, "g_PredictedRigidBodyPositionsInvMass",
                              transientState.predictedRigidBodies.positionsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(updateVelocitiesSrb, "g_PredictedRigidBodyOrientations",
                              transientState.predictedRigidBodies.orientationsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(updateVelocitiesSrb, "g_PredictedRigidBodyLinearVelocities",
                              transientState.predictedRigidBodies.linearVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindBufferVariable(updateVelocitiesSrb, "g_PredictedRigidBodyAngularVelocities",
                              transientState.predictedRigidBodies.angularVelocitiesBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS);
}

bool PhysicsSolver::Impl::bindAllPassBuffers()
{
    return bindPredictBuffers() && bindGenerateContactsBuffers() && bindSolveGatherBuffers() &&
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
        transientState.contactsBuffer != nullptr &&
        transientState.translationCorrectionsBuffer != nullptr &&
        transientState.rotationCorrectionsBuffer != nullptr &&
        readbackRigidBodies.positionsBuffer != nullptr &&
        readbackRigidBodies.orientationsBuffer != nullptr &&
        readbackRigidBodies.linearVelocitiesBuffer != nullptr &&
        readbackRigidBodies.angularVelocitiesBuffer != nullptr;
    if (hasAllBuffers && bufferCapacity >= bodyCount)
    {
        return bindAllPassBuffers();
    }

    const std::uint32_t newCapacity    = std::max<std::uint32_t>(bodyCount, 64u);
    const std::uint32_t newPairCount   = computeRigidPairCount(newCapacity);
    const std::uint32_t newContactCap  = std::max<std::uint32_t>(
        newPairCount * kRigidContactsPerPair, kRigidContactsPerPair);
    const Diligent::Uint64 contextMask = contextMaskForId(physicsContextId);

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
                                readbackRigidBodies.angularVelocitiesBuffer))
    {
        return false;
    }

    bufferCapacity  = newCapacity;
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

    computeContext->MapBuffer(readbackRigidBodies.positionsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedPositions);
    computeContext->MapBuffer(readbackRigidBodies.orientationsBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedOrientations);
    computeContext->MapBuffer(readbackRigidBodies.linearVelocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedLinear);
    computeContext->MapBuffer(readbackRigidBodies.angularVelocitiesBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_DO_NOT_WAIT, mappedAngular);
    computeContext->Flush();
    computeContext->WaitForIdle();

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
        markStage(mImpl->stageStats, PhysicsSolverStage::BuildSpatialIndices, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::SortSpatialIndices, false);
        markStage(mImpl->stageStats, PhysicsSolverStage::BuildConstraintData, false);
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

    markStage(mImpl->stageStats, PhysicsSolverStage::BuildSpatialIndices, false);
    markStage(mImpl->stageStats, PhysicsSolverStage::SortSpatialIndices, false);
    markStage(mImpl->stageStats, PhysicsSolverStage::BuildConstraintData, false);

    const std::uint32_t substeps   = std::max<std::uint32_t>(mDesc.substeps, 1u);
    const std::uint32_t iterations = std::max<std::uint32_t>(mDesc.solverIterations, 1u);
    const float substepDt          = frameContext.deltaSeconds / static_cast<float>(substeps);
    const std::uint32_t pairCount  = computeRigidPairCount(rigidBodyCount);

    for (std::uint32_t substep = 0; substep < substeps; ++substep)
    {
        GpuRigidDispatchConstants constants{};
        constants.dt               = substepDt;
        constants.rigidBodyCount   = rigidBodyCount;
        constants.pairCount        = pairCount;
        constants.substepIndex     = substep;
        constants.solverIterations = iterations;

        if (!writeDispatchConstants(computeBackend.computeContext, mImpl->dispatchConstantsBuffer,
                                    constants) ||
            !mImpl->dispatchPredictPass(computeBackend.computeContext, rigidBodyCount))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: PredictState dispatch.");
            return false;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::PredictState, true);

        if (pairCount > 0u)
        {
            if (!mImpl->dispatchGenerateContactsPass(computeBackend.computeContext, pairCount))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: GenerateContacts dispatch.");
                return false;
            }
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateContacts, true);

            for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
            {
                constants.iterationIndex = iteration;

                if (!mImpl->dispatchGenerateContactsPass(computeBackend.computeContext, pairCount))
                {
                    LOG_ERROR_MESSAGE("PhysicsSolver::step failed: GenerateContacts dispatch.");
                    return false;
                }

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
