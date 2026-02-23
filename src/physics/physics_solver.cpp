#include "physics/physics_solver.h"

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
#include <iterator>
#include <limits>

namespace cressim::neo::physics
{

namespace
{

struct PhysicsStepConstants
{
    float dt                = 0.0f;
    std::uint32_t bodyCount = 0;
    std::uint32_t substep   = 0;
    std::uint32_t iteration = 0;
};

struct GpuPair
{
    std::uint32_t a = 0;
    std::uint32_t b = 0;
};

constexpr std::uint32_t kComputeThreadGroupSize = 64;

constexpr std::size_t stageIndex(RigidPbdSolverStage stage)
{
    return static_cast<std::size_t>(stage);
}

void markStage(PhysicsSolverStageStats& stats, RigidPbdSolverStage stage, bool executed)
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
    if (contextId >= 64u)
    {
        return std::numeric_limits<Diligent::Uint64>::max();
    }
    return static_cast<Diligent::Uint64>(1ull) << contextId;
}

} // namespace

struct PhysicsSolver::Impl
{
    gpu::ShaderLibrary shaderLibrary{""};
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> integratePso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> integrateSrb;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> stepConstantsBuffer;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderShapeTypesBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> colliderParamsBuffer;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> broadphaseKeysBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> sortedIndicesBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> candidatePairsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> contactsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> constraintScratchBuffer;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> positionsReadbackBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> orientationsReadbackBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> linearVelocitiesReadbackBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> angularVelocitiesReadbackBuffer;

    std::uint32_t bufferCapacity = 0;
    std::uint32_t scratchCapacity = 0;
    PhysicsSolverStageStats stageStats{};

    bool bindIntegrateBuffers();
    bool ensureCapacity(Diligent::IRenderDevice* renderDevice, std::uint32_t bodyCount,
                        std::uint32_t physicsContextId);
    bool uploadRigidBodySoA(Diligent::IDeviceContext* computeContext, PhysicsWorld& world,
                            std::uint32_t bodyCount);
    bool dispatchIntegratePass(Diligent::IDeviceContext* computeContext, std::uint32_t bodyCount);
    bool readbackRigidTransformsBlocking(Diligent::IDeviceContext* computeContext, PhysicsWorld& world,
                                         std::uint32_t bodyCount);
};

PhysicsSolver::PhysicsSolver(gpu::GpuDevice& device, const PhysicsSolverDesc& desc)
    : mDevice(device), mDesc(desc), mImpl(std::make_unique<Impl>())
{
}

PhysicsSolver::~PhysicsSolver() = default;

namespace
{

bool ensureStructuredBuffer(Diligent::IRenderDevice* renderDevice,
                            const char* name,
                            std::uint32_t elementStride,
                            std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags,
                            Diligent::USAGE usage,
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

} // namespace

bool PhysicsSolver::Impl::bindIntegrateBuffers()
{
    auto bindView = [&](const char* varName, Diligent::IBuffer* buffer,
                        Diligent::BUFFER_VIEW_TYPE viewType)
    {
        Diligent::IShaderResourceVariable* variable =
            integrateSrb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE, varName);
        if (variable == nullptr || buffer == nullptr)
        {
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
    };

    return bindView("g_RigidBodyPositionsInvMass", positionsBuffer,
                    Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindView("g_RigidBodyOrientations", orientationsBuffer,
                    Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindView("g_RigidBodyLinearVelocities", linearVelocitiesBuffer,
                    Diligent::BUFFER_VIEW_UNORDERED_ACCESS) &&
           bindView("g_RigidBodyAngularVelocities", angularVelocitiesBuffer,
                    Diligent::BUFFER_VIEW_UNORDERED_ACCESS);
}

bool PhysicsSolver::Impl::ensureCapacity(Diligent::IRenderDevice* renderDevice,
                                         std::uint32_t bodyCount,
                                         std::uint32_t physicsContextId)
{
    const bool hasAllCoreBuffers = positionsBuffer != nullptr && orientationsBuffer != nullptr &&
                                   linearVelocitiesBuffer != nullptr &&
                                   angularVelocitiesBuffer != nullptr &&
                                   colliderShapeTypesBuffer != nullptr &&
                                   colliderParamsBuffer != nullptr &&
                                   positionsReadbackBuffer != nullptr &&
                                   orientationsReadbackBuffer != nullptr &&
                                   linearVelocitiesReadbackBuffer != nullptr &&
                                   angularVelocitiesReadbackBuffer != nullptr;
    if (hasAllCoreBuffers && bufferCapacity >= bodyCount)
    {
        return true;
    }

    const std::uint32_t newCapacity = std::max<std::uint32_t>(bodyCount, 64u);
    const Diligent::Uint64 contextMask = contextMaskForId(physicsContextId);
    if (!ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PositionsInvMass",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                positionsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Orientations",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                orientationsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.LinearVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                linearVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.AngularVelocities",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                angularVelocitiesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderShapeTypes",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                colliderShapeTypesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ColliderParams",
                                sizeof(Diligent::float4), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                colliderParamsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.PositionsInvMass.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                positionsReadbackBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Orientations.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                orientationsReadbackBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.LinearVelocities.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                linearVelocitiesReadbackBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.AngularVelocities.Readback",
                                sizeof(Diligent::float4), newCapacity, Diligent::BIND_NONE,
                                Diligent::USAGE_STAGING, Diligent::CPU_ACCESS_READ, contextMask,
                                angularVelocitiesReadbackBuffer))
    {
        return false;
    }

    bufferCapacity = newCapacity;

    const std::uint32_t newScratchCapacity = std::max<std::uint32_t>(newCapacity * 8u, 64u);
    if (!ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.BroadphaseKeys",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                broadphaseKeysBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.SortedIndices",
                                sizeof(std::uint32_t), newCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                sortedIndicesBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.CandidatePairs", sizeof(GpuPair),
                                newScratchCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                candidatePairsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.Contacts",
                                sizeof(Diligent::float4), newScratchCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                contactsBuffer) ||
        !ensureStructuredBuffer(renderDevice, "CRESSimNeo.Physics.ConstraintScratch",
                                sizeof(Diligent::float4), newScratchCapacity,
                                Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE,
                                Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask,
                                constraintScratchBuffer))
    {
        return false;
    }
    scratchCapacity = newScratchCapacity;

    return bindIntegrateBuffers();
}

bool PhysicsSolver::Impl::uploadRigidBodySoA(Diligent::IDeviceContext* computeContext,
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

    const Diligent::Uint64 float4Bytes = static_cast<Diligent::Uint64>(bodyCount) *
                                         sizeof(Diligent::float4);
    const Diligent::Uint64 shapeTypeBytes = static_cast<Diligent::Uint64>(bodyCount) *
                                            sizeof(std::uint32_t);

    // TODO(PBD-GPU): use world.rigidBodyDirtyRange() to upload only changed slices.
    computeContext->UpdateBuffer(positionsBuffer, 0u, float4Bytes,
                                 rigidBodies.positionsInvMass.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(orientationsBuffer, 0u, float4Bytes,
                                 rigidBodies.orientations.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(linearVelocitiesBuffer, 0u, float4Bytes,
                                 rigidBodies.linearVelocities.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(angularVelocitiesBuffer, 0u, float4Bytes,
                                 rigidBodies.angularVelocities.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(colliderShapeTypesBuffer, 0u, shapeTypeBytes,
                                 rigidBodies.colliderShapeTypes.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->UpdateBuffer(colliderParamsBuffer, 0u, float4Bytes,
                                 rigidBodies.colliderParams.data(),
                                 Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    world.clearRigidBodyDirtyRange();
    return true;
}

namespace
{

bool writeStepConstants(Diligent::IDeviceContext* computeContext,
                        Diligent::IBuffer* constantsBuffer,
                        const PhysicsStepConstants& constants)
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

} // namespace

bool PhysicsSolver::Impl::dispatchIntegratePass(Diligent::IDeviceContext* computeContext,
                                                std::uint32_t bodyCount)
{
    if (computeContext == nullptr || integratePso == nullptr || integrateSrb == nullptr ||
        bodyCount == 0u)
    {
        return false;
    }

    computeContext->SetPipelineState(integratePso);
    computeContext->CommitShaderResources(integrateSrb,
                                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    const std::uint32_t groupCountX = (bodyCount + kComputeThreadGroupSize - 1u) /
                                      kComputeThreadGroupSize;
    computeContext->DispatchCompute(Diligent::DispatchComputeAttribs{groupCountX, 1u, 1u});
    return true;
}

bool PhysicsSolver::Impl::readbackRigidTransformsBlocking(Diligent::IDeviceContext* computeContext,
                                                          PhysicsWorld& world,
                                                          std::uint32_t bodyCount)
{
    if (computeContext == nullptr || bodyCount == 0u)
    {
        return false;
    }

    const Diligent::Uint64 bytes = static_cast<Diligent::Uint64>(bodyCount) * sizeof(Diligent::float4);
    computeContext->CopyBuffer(positionsBuffer, 0, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               positionsReadbackBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(orientationsBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               orientationsReadbackBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(linearVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               linearVelocitiesReadbackBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    computeContext->CopyBuffer(angularVelocitiesBuffer, 0,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                               angularVelocitiesReadbackBuffer, 0, bytes,
                               Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    computeContext->Flush();
    computeContext->WaitForIdle();

    void* mappedPositions = nullptr;
    void* mappedOrientations = nullptr;
    void* mappedLinear = nullptr;
    void* mappedAngular = nullptr;

    computeContext->MapBuffer(positionsReadbackBuffer, Diligent::MAP_READ, Diligent::MAP_FLAG_NONE,
                              mappedPositions);
    computeContext->MapBuffer(orientationsReadbackBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_NONE, mappedOrientations);
    computeContext->MapBuffer(linearVelocitiesReadbackBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_NONE, mappedLinear);
    computeContext->MapBuffer(angularVelocitiesReadbackBuffer, Diligent::MAP_READ,
                              Diligent::MAP_FLAG_NONE, mappedAngular);
    if (mappedPositions == nullptr || mappedOrientations == nullptr || mappedLinear == nullptr ||
        mappedAngular == nullptr)
    {
        if (mappedPositions != nullptr)
        {
            computeContext->UnmapBuffer(positionsReadbackBuffer, Diligent::MAP_READ);
        }
        if (mappedOrientations != nullptr)
        {
            computeContext->UnmapBuffer(orientationsReadbackBuffer, Diligent::MAP_READ);
        }
        if (mappedLinear != nullptr)
        {
            computeContext->UnmapBuffer(linearVelocitiesReadbackBuffer, Diligent::MAP_READ);
        }
        if (mappedAngular != nullptr)
        {
            computeContext->UnmapBuffer(angularVelocitiesReadbackBuffer, Diligent::MAP_READ);
        }
        return false;
    }

    const auto* positions = static_cast<const Diligent::float4*>(mappedPositions);
    const auto* orientations = static_cast<const Diligent::float4*>(mappedOrientations);
    const auto* linearVelocities = static_cast<const Diligent::float4*>(mappedLinear);
    const auto* angularVelocities = static_cast<const Diligent::float4*>(mappedAngular);
    for (std::uint32_t i = 0; i < bodyCount; ++i)
    {
        (void)world.writeBackRigidBodyState(i, positions[i], orientations[i], linearVelocities[i],
                                            angularVelocities[i]);
    }
    world.finalizeRigidBodyWriteback();

    computeContext->UnmapBuffer(positionsReadbackBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(orientationsReadbackBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(linearVelocitiesReadbackBuffer, Diligent::MAP_READ);
    computeContext->UnmapBuffer(angularVelocitiesReadbackBuffer, Diligent::MAP_READ);
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
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) || computeContext.renderDevice == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to get physics GPU context.");
        return false;
    }

    mImpl                = std::make_unique<Impl>();
    mImpl->shaderLibrary = gpu::ShaderLibrary(mDevice.shaderSourceDirectory());

    std::string shaderPath;
    constexpr const char* kShaderRelativePath = "physics/physics_placeholder_integrate.cs.hlsl";
    if (!mImpl->shaderLibrary.resolveShaderPath(kShaderRelativePath, shaderPath))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to resolve compute shader path.");
        return false;
    }

    Diligent::IShaderSourceInputStreamFactory* streamFactory = mImpl->shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: shader stream factory is null.");
        return false;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.Desc.ShaderType                 = Diligent::SHADER_TYPE_COMPUTE;
    shaderCreateInfo.Desc.Name                       = "CRESSimNeo.Physics.PlaceholderIntegrate.CS";
    shaderCreateInfo.FilePath                        = kShaderRelativePath;
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::RefCntAutoPtr<Diligent::IShader> computeShader;
    computeContext.renderDevice->CreateShader(shaderCreateInfo, &computeShader);
    if (computeShader == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to compile compute shader.");
        return false;
    }

    Diligent::ComputePipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name         = "CRESSimNeo.Physics.PlaceholderIntegrate.PSO";
    psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_COMPUTE;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;
    Diligent::ShaderResourceVariableDesc vars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsStepConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyPositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyLinearVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodyAngularVelocities",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    psoCreateInfo.PSODesc.ResourceLayout.Variables    = vars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(std::size(vars));
    psoCreateInfo.pCS = computeShader;

    computeContext.renderDevice->CreateComputePipelineState(psoCreateInfo, &mImpl->integratePso);
    if (mImpl->integratePso == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create compute PSO.");
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Name                 = "CRESSimNeo.Physics.StepConstants";
    constantsDesc.Size                 = sizeof(PhysicsStepConstants);
    constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    constantsDesc.ImmediateContextMask = contextMaskForId(computeContext.contextId);
    computeContext.renderDevice->CreateBuffer(constantsDesc, nullptr, &mImpl->stepConstantsBuffer);
    if (mImpl->stepConstantsBuffer == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create step constants buffer.");
        return false;
    }

    mImpl->integratePso->CreateShaderResourceBinding(&mImpl->integrateSrb, true);
    if (mImpl->integrateSrb == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create SRB.");
        return false;
    }

    Diligent::IShaderResourceVariable* constantsVar =
        mImpl->integrateSrb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE,
                                               "PhysicsStepConstantsBuffer");
    if (constantsVar == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: variable PhysicsStepConstantsBuffer not found.");
        return false;
    }
    constantsVar->Set(mImpl->stepConstantsBuffer);

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
        world.integrateRigidBodiesCpu(frameContext.deltaSeconds);
        markStage(mImpl->stageStats, RigidPbdSolverStage::IntegrateExternalForces, true);
        markStage(mImpl->stageStats, RigidPbdSolverStage::WritebackTransforms, true);
        return true;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) || computeBackend.renderDevice == nullptr ||
        computeBackend.computeContext == nullptr)
    {
        return false;
    }

    const std::uint32_t bodyCount = world.rigidBodyCount();
    if (bodyCount == 0u)
    {
        return true;
    }

    if (!mImpl->ensureCapacity(computeBackend.renderDevice, bodyCount, computeBackend.contextId))
    {
        return false;
    }
    if (!mImpl->uploadRigidBodySoA(computeBackend.computeContext, world, bodyCount))
    {
        return false;
    }
    if (!mImpl->bindIntegrateBuffers())
    {
        return false;
    }

    const std::uint32_t substeps = std::max<std::uint32_t>(mDesc.substeps, 1u);
    const std::uint32_t iterations = std::max<std::uint32_t>(mDesc.solverIterations, 1u);
    const float substepDt = frameContext.deltaSeconds / static_cast<float>(substeps);

    for (std::uint32_t substep = 0; substep < substeps; ++substep)
    {
        const PhysicsStepConstants constants{substepDt, bodyCount, substep, 0u};
        if (!writeStepConstants(computeBackend.computeContext, mImpl->stepConstantsBuffer, constants) ||
            !mImpl->dispatchIntegratePass(computeBackend.computeContext, bodyCount))
        {
            return false;
        }
        markStage(mImpl->stageStats, RigidPbdSolverStage::IntegrateExternalForces, true);

        // TODO(PBD-GPU): implement broadphase key generation dispatch and key layout.
        // TODO(PBD-GPU): implement broadphase sort/bucket stage (radix sort or uniform grid bucket build).
        // TODO(PBD-GPU): implement contact generation stage and contact pair schema.
        // TODO(PBD-GPU): implement rigid constraint construction stage.
        if (mDesc.enableRigidBroadphaseScaffold)
        {
            markStage(mImpl->stageStats, RigidPbdSolverStage::BuildBroadphaseKeys, false);
            markStage(mImpl->stageStats, RigidPbdSolverStage::SortOrBucket, false);
            markStage(mImpl->stageStats, RigidPbdSolverStage::GenerateContacts, false);
            markStage(mImpl->stageStats, RigidPbdSolverStage::BuildRigidConstraints, false);
        }

        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            (void)iteration;
            // TODO(PBD-GPU): implement iterative rigid constraints solve kernels and lambda buffers.
        }
        markStage(mImpl->stageStats, RigidPbdSolverStage::SolveConstraints, false);

        // TODO(PBD-GPU): implement velocity update pass from predicted pose deltas.
        markStage(mImpl->stageStats, RigidPbdSolverStage::UpdateVelocities, false);
    }

    if (!mDesc.enableBlockingReadback)
    {
        // TODO(PBD-GPU): replace blocking readback with async readback ring and fence polling.
        // TODO(PBD-GPU): add direct physics->render interop path to bypass CPU readback.
        markStage(mImpl->stageStats, RigidPbdSolverStage::WritebackTransforms, false);
        return true;
    }

    if (!mImpl->readbackRigidTransformsBlocking(computeBackend.computeContext, world, bodyCount))
    {
        return false;
    }
    markStage(mImpl->stageStats, RigidPbdSolverStage::WritebackTransforms, true);
    return true;
}

const PhysicsSolverStageStats& PhysicsSolver::lastStageStats() const noexcept
{
    return mImpl->stageStats;
}

} // namespace cressim::neo::physics
