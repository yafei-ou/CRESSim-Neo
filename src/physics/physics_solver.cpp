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
#include <cstring>
#include <vector>

namespace cressim::neo::physics
{

namespace
{

struct GpuRigidBodyState
{
    Diligent::float4 positionInvMass{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::float4 rotation{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::float4 linearVelocity{0.0f, 0.0f, 0.0f, 0.0f};
};

struct PhysicsStepConstants
{
    float dt                = 0.0f;
    std::uint32_t bodyCount = 0;
    float padding[2]        = {0.0f, 0.0f};
};

constexpr std::uint32_t kComputeThreadGroupSize = 64;

GpuRigidBodyState toGpuState(const RigidBodyState& rb)
{
    GpuRigidBodyState out{};
    out.positionInvMass =
        Diligent::float4{rb.position.x, rb.position.y, rb.position.z, rb.inverseMass};
    out.rotation =
        Diligent::float4{rb.rotation.q.x, rb.rotation.q.y, rb.rotation.q.z, rb.rotation.q.w};
    out.linearVelocity =
        Diligent::float4{rb.linearVelocity.x, rb.linearVelocity.y, rb.linearVelocity.z, 0.0f};
    return out;
}

void applyGpuState(const GpuRigidBodyState& src, RigidBodyState& dst)
{
    dst.position =
        Diligent::float3{src.positionInvMass.x, src.positionInvMass.y, src.positionInvMass.z};
    dst.inverseMass = src.positionInvMass.w;
    dst.rotation =
        Diligent::QuaternionF{src.rotation.x, src.rotation.y, src.rotation.z, src.rotation.w};
    dst.linearVelocity =
        Diligent::float3{src.linearVelocity.x, src.linearVelocity.y, src.linearVelocity.z};
}

} // namespace

struct PhysicsSolver::Impl
{
    gpu::ShaderLibrary shaderLibrary{""};
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> pso;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> rigidBodyBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> stepConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> readbackBuffer;
    std::vector<GpuRigidBodyState> stagingStates;
    std::uint32_t bufferCapacity = 0;
};

PhysicsSolver::PhysicsSolver(gpu::GpuDevice& device, const PhysicsSolverDesc& desc)
    : mDevice(device), mDesc(desc), mImpl(std::make_unique<Impl>())
{
}

PhysicsSolver::~PhysicsSolver() = default;

bool PhysicsSolver::initialize()
{
    shutdown();

    if (!mDesc.enableGpuCompute)
    {
        mInitialized = true;
        return true;
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetBackendContext(backendContext) || backendContext.renderDevice == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to get GPU backend context.");
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
    backendContext.renderDevice->CreateShader(shaderCreateInfo, &computeShader);
    if (computeShader == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to compile compute shader.");
        return false;
    }

    // PSO
    Diligent::ComputePipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name         = "CRESSimNeo.Physics.PlaceholderIntegrate.PSO";
    psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_COMPUTE;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE;

    Diligent::ShaderResourceVariableDesc vars[] = {
        {Diligent::SHADER_TYPE_COMPUTE, "PhysicsStepConstantsBuffer",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodies",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    psoCreateInfo.PSODesc.ResourceLayout.Variables    = vars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables = _countof(vars);

    psoCreateInfo.pCS = computeShader;

    backendContext.renderDevice->CreateComputePipelineState(psoCreateInfo, &mImpl->pso);
    if (mImpl->pso == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create compute PSO.");
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Name           = "CRESSimNeo.Physics.StepConstants";
    constantsDesc.Size           = sizeof(PhysicsStepConstants);
    constantsDesc.Usage          = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
    backendContext.renderDevice->CreateBuffer(constantsDesc, nullptr, &mImpl->stepConstantsBuffer);
    if (mImpl->stepConstantsBuffer == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create step constant buffer.");
        return false;
    }

    mImpl->pso->CreateShaderResourceBinding(&mImpl->srb, true);
    if (mImpl->srb == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to create SRB.");
        return false;
    }
    Diligent::IShaderResourceVariable* stepConstantsVar =
        mImpl->srb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE, "PhysicsStepConstantsBuffer");
    if (stepConstantsVar == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: variable PhysicsStepConstantsBuffer not found.");
        return false;
    }
    stepConstantsVar->Set(mImpl->stepConstantsBuffer);

    mInitialized = true;
    return true;
}

void PhysicsSolver::shutdown()
{
    mImpl        = std::make_unique<Impl>();
    mBuffers     = {};
    mInitialized = false;
}

bool PhysicsSolver::step(const common::FrameContext& frameContext, PhysicsWorld& world)
{
    if (!mInitialized)
    {
        return false;
    }

    if (!mDesc.enableGpuCompute)
    {
        for (RigidBodyState& rb : world.rigidBodies())
        {
            rb.position += rb.linearVelocity * frameContext.deltaSeconds;
        }
        return true;
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetBackendContext(backendContext) || backendContext.renderDevice == nullptr ||
        backendContext.immediateContext == nullptr)
    {
        return false;
    }

    const std::vector<RigidBodyState>& rigidBodies = world.rigidBodies();
    const std::uint32_t bodyCount                  = static_cast<std::uint32_t>(rigidBodies.size());
    if (bodyCount == 0)
    {
        return true;
    }

    if (mImpl->bufferCapacity < bodyCount || mImpl->rigidBodyBuffer == nullptr ||
        mImpl->readbackBuffer == nullptr)
    {
        const std::uint32_t newCapacity = std::max<std::uint32_t>(bodyCount, 64u);
        const Diligent::Uint64 byteSize =
            static_cast<Diligent::Uint64>(newCapacity) * sizeof(GpuRigidBodyState);

        Diligent::BufferDesc simBufferDesc{};
        simBufferDesc.Name      = "CRESSimNeo.Physics.RigidBodies";
        simBufferDesc.Size      = byteSize;
        simBufferDesc.Usage     = Diligent::USAGE_DEFAULT;
        simBufferDesc.BindFlags = Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_SHADER_RESOURCE;
        simBufferDesc.Mode      = Diligent::BUFFER_MODE_STRUCTURED;
        simBufferDesc.ElementByteStride = sizeof(GpuRigidBodyState);

        Diligent::RefCntAutoPtr<Diligent::IBuffer> newSimBuffer;
        backendContext.renderDevice->CreateBuffer(simBufferDesc, nullptr, &newSimBuffer);
        if (newSimBuffer == nullptr)
        {
            return false;
        }

        Diligent::BufferDesc readbackDesc{};
        readbackDesc.Name              = "CRESSimNeo.Physics.RigidBodies.Readback";
        readbackDesc.Size              = byteSize;
        readbackDesc.Usage             = Diligent::USAGE_STAGING;
        readbackDesc.CPUAccessFlags    = Diligent::CPU_ACCESS_READ;
        readbackDesc.BindFlags         = Diligent::BIND_NONE;

        Diligent::RefCntAutoPtr<Diligent::IBuffer> newReadbackBuffer;
        backendContext.renderDevice->CreateBuffer(readbackDesc, nullptr, &newReadbackBuffer);
        if (newReadbackBuffer == nullptr)
        {
            return false;
        }

        Diligent::IShaderResourceVariable* rigidBodiesVar =
            mImpl->srb->GetVariableByName(Diligent::SHADER_TYPE_COMPUTE, "g_RigidBodies");
        if (rigidBodiesVar == nullptr)
        {
            return false;
        }

        rigidBodiesVar->Set(newSimBuffer->GetDefaultView(Diligent::BUFFER_VIEW_UNORDERED_ACCESS));
        mImpl->rigidBodyBuffer = std::move(newSimBuffer);
        mImpl->readbackBuffer  = std::move(newReadbackBuffer);
        mImpl->bufferCapacity  = newCapacity;
    }

    mImpl->stagingStates.resize(bodyCount);
    for (std::uint32_t i = 0; i < bodyCount; ++i)
    {
        mImpl->stagingStates[i] = toGpuState(rigidBodies[i]);
    }

    const Diligent::Uint64 uploadSize =
        static_cast<Diligent::Uint64>(bodyCount) * sizeof(GpuRigidBodyState);
    backendContext.immediateContext->UpdateBuffer(
        mImpl->rigidBodyBuffer, 0, uploadSize, mImpl->stagingStates.data(),
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    PhysicsStepConstants stepConstants{};
    stepConstants.dt        = frameContext.deltaSeconds;
    stepConstants.bodyCount = bodyCount;
    void* mappedConstants   = nullptr;
    backendContext.immediateContext->MapBuffer(mImpl->stepConstantsBuffer, Diligent::MAP_WRITE,
                                               Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &stepConstants, sizeof(stepConstants));
    backendContext.immediateContext->UnmapBuffer(mImpl->stepConstantsBuffer, Diligent::MAP_WRITE);

    // Set PSO and dispatch
    backendContext.immediateContext->SetPipelineState(mImpl->pso);
    backendContext.immediateContext->CommitShaderResources(
        mImpl->srb, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    const std::uint32_t groupCountX =
        (bodyCount + kComputeThreadGroupSize - 1u) / kComputeThreadGroupSize;
    backendContext.immediateContext->DispatchCompute(
        Diligent::DispatchComputeAttribs{groupCountX, 1u, 1u});

    backendContext.immediateContext->CopyBuffer(
        mImpl->rigidBodyBuffer, 0, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        mImpl->readbackBuffer, 0, uploadSize, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    backendContext.immediateContext->Flush();
    backendContext.immediateContext->WaitForIdle();

    void* mappedReadback = nullptr;
    backendContext.immediateContext->MapBuffer(mImpl->readbackBuffer, Diligent::MAP_READ,
                                               Diligent::MAP_FLAG_DO_NOT_WAIT, mappedReadback);

    if (mappedReadback == nullptr)
    {
        return false;
    }

    const auto* gpuStates = static_cast<const GpuRigidBodyState*>(mappedReadback);
    for (std::uint32_t i = 0; i < bodyCount; ++i)
    {
        applyGpuState(gpuStates[i], world.rigidBodies()[i]);
    }
    backendContext.immediateContext->UnmapBuffer(mImpl->readbackBuffer, Diligent::MAP_READ);

    return true;
}

} // namespace cressim::neo::physics
