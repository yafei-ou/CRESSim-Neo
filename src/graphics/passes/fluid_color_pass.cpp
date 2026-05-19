#include "graphics/passes/fluid_color_pass.h"

#include "gpu/shader_library.h"
#include "graphics/gpu_scene.h"
#include "graphics/passes/render_pass_types.h"
#include "physics/physics_gpu_scene_view.h"

#include <cstring>

namespace cressim::neo::graphics::detail
{

FluidColorPass::FluidColorPass(gpu::GpuDevice &device) : mDevice(device) {}

std::size_t FluidColorPass::PipelineKeyHasher::operator()(const PipelineKey &key) const noexcept
{
    return std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat));
}

bool FluidColorPass::initialize()
{
    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        return false;
    }

    Diligent::SamplerDesc samplerDesc{};
    samplerDesc.MinFilter = Diligent::FILTER_TYPE_LINEAR;
    samplerDesc.MagFilter = Diligent::FILTER_TYPE_LINEAR;
    samplerDesc.MipFilter = Diligent::FILTER_TYPE_LINEAR;
    samplerDesc.AddressU  = Diligent::TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV  = Diligent::TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW  = Diligent::TEXTURE_ADDRESS_CLAMP;
    backendContext.renderDevice->CreateSampler(samplerDesc, &mLinearClampSampler);

    mInitialized = ensureConstants(backendContext.renderDevice);
    return mInitialized;
}

bool FluidColorPass::ensureConstants(Diligent::IRenderDevice *renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }
    if (mConstantsBuffer != nullptr)
    {
        return true;
    }

    Diligent::BufferDesc constantBufferDesc{};
    constantBufferDesc.Name           = "CRESSimNeo.FluidColorPass.Constants";
    constantBufferDesc.Size           = sizeof(DrawConstants);
    constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
    constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
    constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
    renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mConstantsBuffer);
    return mConstantsBuffer != nullptr;
}

Diligent::IPipelineState *FluidColorPass::getOrCreatePipeline(Diligent::IRenderDevice *renderDevice,
                                                              const PipelineKey &key)
{
    auto it = mPipelines.find(key);
    if (it != mPipelines.end())
    {
        return it->second;
    }
    if (renderDevice == nullptr || key.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return nullptr;
    }

    gpu::ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        return nullptr;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.FluidColorPass.VS";
    shaderCreateInfo.FilePath        = "graphics/fluid_color.vs.hlsl";
    if (!mDevice.createShader(shaderCreateInfo, &vertexShader))
    {
        return nullptr;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.FluidColorPass.PS";
    shaderCreateInfo.FilePath        = "graphics/fluid_color.ps.hlsl";
    if (!mDevice.createShader(shaderCreateInfo, &pixelShader))
    {
        return nullptr;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                       = "CRESSimNeo.FluidColorPass.PSO";
    psoCreateInfo.PSODesc.PipelineType               = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]     = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = Diligent::TEX_FORMAT_UNKNOWN;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode      = Diligent::CULL_MODE_NONE;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = Diligent::False;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    constexpr Diligent::ShaderResourceVariableDesc kVars[] = {
        {Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_CameraInputs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticlePositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticleRadii",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticleEnvironmentIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticleKinds",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticleOwnerTypes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticleOwnerIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_FluidVisuals",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_CameraInputs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_FilteredFluidDepth",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    psoCreateInfo.PSODesc.ResourceLayout.Variables = kVars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(std::size(kVars));
    psoCreateInfo.pVS = vertexShader;
    psoCreateInfo.pPS = pixelShader;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipeline;
    if (!mDevice.createGraphicsPipelineState(psoCreateInfo, &pipeline))
    {
        return nullptr;
    }

    if (Diligent::IShaderResourceVariable *vsConstants =
            pipeline->GetStaticVariableByName(Diligent::SHADER_TYPE_VERTEX, "GraphicsFluidColor"))
    {
        vsConstants->Set(mConstantsBuffer);
    }
    if (Diligent::IShaderResourceVariable *psConstants =
            pipeline->GetStaticVariableByName(Diligent::SHADER_TYPE_PIXEL, "GraphicsFluidColor"))
    {
        psConstants->Set(mConstantsBuffer);
    }

    auto insertResult = mPipelines.emplace(key, pipeline);
    return insertResult.first->second;
}

Diligent::IShaderResourceBinding *FluidColorPass::getOrCreateBinding(
    Diligent::IPipelineState *pipeline)
{
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    auto it = mBindings.find(pipeline);
    if (it != mBindings.end())
    {
        return it->second;
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    pipeline->CreateShaderResourceBinding(&srb, true);
    if (srb == nullptr)
    {
        return nullptr;
    }

    auto insertResult = mBindings.emplace(pipeline, std::move(srb));
    return insertResult.first->second;
}

bool FluidColorPass::draw(const gpu::GpuRenderTargetBinding &targetBinding,
                          const gpu::GpuRenderTargetDesc &targetDesc,
                          const GpuEntitySceneView &gpuScene,
                          const cressim::neo::physics::PhysicsGpuSceneView &physicsScene,
                          const ResolvedCameraView &camera, std::uint32_t fluidDepthLayer,
                          Diligent::ITextureView *filteredDepthSrv)
{
    if (!mInitialized || filteredDepthSrv == nullptr)
    {
        return false;
    }

    const auto &particles = physicsScene.soft.particles;
    if (particles.count == 0u || gpuScene.preparedCamerasBuffer == nullptr ||
        gpuScene.cameraInputsBuffer == nullptr || particles.positionsInvMassBuffer == nullptr ||
        particles.radiiBuffer == nullptr || particles.environmentIndicesBuffer == nullptr ||
        particles.particleKindsBuffer == nullptr || particles.ownerTypesBuffer == nullptr ||
        particles.ownerIndicesBuffer == nullptr || particles.fluidVisualsBuffer == nullptr ||
        particles.fluidVisualCount == 0u)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr || backendContext.graphicsContext == nullptr ||
        !backendContext.hasActiveRenderTarget ||
        !(backendContext.activeRenderTargetBinding == targetBinding))
    {
        return false;
    }

    Diligent::IPipelineState *pipeline =
        getOrCreatePipeline(backendContext.renderDevice, PipelineKey{targetDesc.colorFormat});
    if (pipeline == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceBinding *binding = getOrCreateBinding(pipeline);
    if (binding == nullptr)
    {
        return false;
    }

    const auto setBufferView = [&](Diligent::SHADER_TYPE shaderType, const char *name,
                                   Diligent::IBuffer *buffer) -> bool
    {
        Diligent::IShaderResourceVariable *variable = binding->GetVariableByName(shaderType, name);
        if (variable == nullptr || buffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *srv = buffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (srv == nullptr)
        {
            return false;
        }
        variable->Set(srv);
        return true;
    };

    if (!setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
                       gpuScene.preparedCamerasBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_CameraInputs",
                       gpuScene.cameraInputsBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_PIXEL, "g_CameraInputs",
                       gpuScene.cameraInputsBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticlePositionsInvMass",
                       particles.positionsInvMassBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticleRadii", particles.radiiBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticleEnvironmentIndices",
                       particles.environmentIndicesBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticleKinds",
                       particles.particleKindsBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticleOwnerTypes",
                       particles.ownerTypesBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticleOwnerIndices",
                       particles.ownerIndicesBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_FluidVisuals",
                       particles.fluidVisualsBuffer))
    {
        return false;
    }

    if (Diligent::IShaderResourceVariable *depthVar =
            binding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_FilteredFluidDepth"))
    {
        if (mLinearClampSampler != nullptr)
        {
            filteredDepthSrv->SetSampler(mLinearClampSampler);
        }
        depthVar->Set(filteredDepthSrv);
    }
    else
    {
        return false;
    }

    DrawConstants constants{};
    constants.cameraIndex     = camera.globalCameraIndex;
    constants.fluidDepthLayer = fluidDepthLayer;
    constants.envIndex        = camera.envIndex;

    void *mapped = nullptr;
    backendContext.graphicsContext->MapBuffer(mConstantsBuffer, Diligent::MAP_WRITE,
                                              Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &constants, sizeof(constants));
    backendContext.graphicsContext->UnmapBuffer(mConstantsBuffer, Diligent::MAP_WRITE);

    backendContext.graphicsContext->SetPipelineState(pipeline);
    backendContext.graphicsContext->CommitShaderResources(
        binding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawAttribs drawAttrs{};
    drawAttrs.NumVertices = particles.count * 6u;
    drawAttrs.Flags       = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.graphicsContext->Draw(drawAttrs);
    return true;
}

} // namespace cressim::neo::graphics::detail
