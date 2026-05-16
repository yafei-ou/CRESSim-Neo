#include "graphics/passes/fluid_composite_pass.h"

#include "gpu/shader_library.h"
#include "graphics/gpu_scene_pipeline_types.h"
#include "graphics/passes/render_pass_types.h"

#include <cstring>

namespace cressim::neo::graphics::detail
{

FluidCompositePass::FluidCompositePass(gpu::GpuDevice &device) : mDevice(device) {}

std::size_t FluidCompositePass::PipelineKeyHasher::operator()(const PipelineKey &key) const noexcept
{
    const std::size_t colorHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat));
    const std::size_t depthHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.depthFormat));
    const std::size_t refractionHash = std::hash<bool>{}(key.enableBackgroundRefraction);
    return colorHash ^ (depthHash << 1u) ^ (refractionHash << 2u);
}

bool FluidCompositePass::initialize()
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

bool FluidCompositePass::ensureConstants(Diligent::IRenderDevice *renderDevice)
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
    constantBufferDesc.Name           = "CRESSimNeo.FluidCompositePass.Constants";
    constantBufferDesc.Size           = sizeof(CompositeConstants);
    constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
    constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
    constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
    renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mConstantsBuffer);
    return mConstantsBuffer != nullptr;
}

Diligent::IPipelineState *FluidCompositePass::getOrCreatePipeline(
    Diligent::IRenderDevice *renderDevice, const PipelineKey &key)
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
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.FluidCompositePass.VS";
    shaderCreateInfo.FilePath        = "graphics/display_resolve.vs.hlsl";
    if (!mDevice.createShader(shaderCreateInfo, &vertexShader))
    {
        return nullptr;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType         = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name               = "CRESSimNeo.FluidCompositePass.PS";
    shaderCreateInfo.FilePath                = "graphics/fluid_composite.ps.hlsl";
    Diligent::ShaderMacro refractionMacros[] = {
        {"CRESSIM_FLUID_ENABLE_BACKGROUND_REFRACTION", key.enableBackgroundRefraction ? "1" : "0"},
    };
    shaderCreateInfo.Macros =
        Diligent::ShaderMacroArray{refractionMacros, static_cast<Diligent::Uint32>(1u)};
    if (!mDevice.createShader(shaderCreateInfo, &pixelShader))
    {
        return nullptr;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                       = "CRESSimNeo.FluidCompositePass.PSO";
    psoCreateInfo.PSODesc.PipelineType               = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]     = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = key.depthFormat;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode           = Diligent::CULL_MODE_NONE;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable      = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = Diligent::False;
    auto &blend       = psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0];
    blend.BlendEnable = Diligent::True;
    blend.SrcBlend    = Diligent::BLEND_FACTOR_SRC_ALPHA;
    blend.DestBlend   = Diligent::BLEND_FACTOR_INV_SRC_ALPHA;
    blend.BlendOp     = Diligent::BLEND_OPERATION_ADD;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    constexpr Diligent::ShaderResourceVariableDesc kVarsNoRefraction[] = {
        {Diligent::SHADER_TYPE_PIXEL, "g_CameraInputs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_LightInputs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_LocalLightSelections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_FluidSurfaceColor",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_FilteredFluidDepth",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_SceneDepth",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    constexpr Diligent::ShaderResourceVariableDesc kVarsWithRefraction[] = {
        {Diligent::SHADER_TYPE_PIXEL, "g_CameraInputs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_LightInputs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_LocalLightSelections",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_FluidSurfaceColor",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_FilteredFluidDepth",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_SceneColor",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_SceneDepth",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    if (key.enableBackgroundRefraction)
    {
        psoCreateInfo.PSODesc.ResourceLayout.Variables = kVarsWithRefraction;
        psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
            static_cast<Diligent::Uint32>(std::size(kVarsWithRefraction));
    }
    else
    {
        psoCreateInfo.PSODesc.ResourceLayout.Variables = kVarsNoRefraction;
        psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
            static_cast<Diligent::Uint32>(std::size(kVarsNoRefraction));
    }
    psoCreateInfo.pVS = vertexShader;
    psoCreateInfo.pPS = pixelShader;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipeline;
    if (!mDevice.createGraphicsPipelineState(psoCreateInfo, &pipeline))
    {
        return nullptr;
    }

    if (Diligent::IShaderResourceVariable *constantsVar = pipeline->GetStaticVariableByName(
            Diligent::SHADER_TYPE_PIXEL, "GraphicsFluidComposite"))
    {
        constantsVar->Set(mConstantsBuffer);
    }

    auto insertResult = mPipelines.emplace(key, pipeline);
    return insertResult.first->second;
}

Diligent::IShaderResourceBinding *FluidCompositePass::getOrCreateBinding(
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

bool FluidCompositePass::composite(
    const gpu::GpuRenderTargetBinding &targetBinding, const gpu::GpuRenderTargetDesc &targetDesc,
    const GpuEntitySceneView &gpuScene, const ResolvedCameraView &camera,
    std::uint32_t fluidDepthLayer, std::uint32_t sceneDepthLayer,
    Diligent::ITextureView *filteredDepthSrv, Diligent::ITextureView *surfaceColorSrv,
    Diligent::ITextureView *sceneColorSrv, Diligent::ITextureView *sceneDepthSrv,
    const RenderFrameOptions::FluidRenderingOptions &options)
{
    if (!mInitialized || filteredDepthSrv == nullptr || surfaceColorSrv == nullptr ||
        sceneDepthSrv == nullptr ||
        (options.enableBackgroundRefraction && sceneColorSrv == nullptr) ||
        gpuScene.cameraInputsBuffer == nullptr || gpuScene.lightInputsBuffer == nullptr ||
        gpuScene.localLightSelectionBuffer == nullptr)
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

    Diligent::IPipelineState *pipeline = getOrCreatePipeline(
        backendContext.renderDevice, PipelineKey{targetDesc.colorFormat, targetDesc.depthFormat,
                                                 options.enableBackgroundRefraction});
    if (pipeline == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceBinding *binding = getOrCreateBinding(pipeline);
    if (binding == nullptr)
    {
        return false;
    }

    if (Diligent::IShaderResourceVariable *cameraVar =
            binding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_CameraInputs"))
    {
        cameraVar->Set(
            gpuScene.cameraInputsBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE));
    }
    if (Diligent::IShaderResourceVariable *lightVar =
            binding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_LightInputs"))
    {
        lightVar->Set(
            gpuScene.lightInputsBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE));
    }
    if (Diligent::IShaderResourceVariable *localLightsVar =
            binding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_LocalLightSelections"))
    {
        localLightsVar->Set(gpuScene.localLightSelectionBuffer->GetDefaultView(
            Diligent::BUFFER_VIEW_SHADER_RESOURCE));
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
    if (Diligent::IShaderResourceVariable *surfaceColorVar =
            binding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_FluidSurfaceColor"))
    {
        if (mLinearClampSampler != nullptr)
        {
            surfaceColorSrv->SetSampler(mLinearClampSampler);
        }
        surfaceColorVar->Set(surfaceColorSrv);
    }
    if (options.enableBackgroundRefraction)
    {
        if (Diligent::IShaderResourceVariable *sceneColorVar =
                binding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_SceneColor"))
        {
            if (mLinearClampSampler != nullptr)
            {
                sceneColorSrv->SetSampler(mLinearClampSampler);
            }
            sceneColorVar->Set(sceneColorSrv);
        }
    }
    if (Diligent::IShaderResourceVariable *sceneDepthVar =
            binding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_SceneDepth"))
    {
        if (mLinearClampSampler != nullptr)
        {
            sceneDepthSrv->SetSampler(mLinearClampSampler);
        }
        sceneDepthVar->Set(sceneDepthSrv);
    }

    CompositeConstants constants{};
    constants.specularSmoothness      = Diligent::float4{options.specular.x, options.specular.y,
                                                         options.specular.z, options.smoothness};
    constants.cameraIndex             = camera.globalCameraIndex;
    constants.fluidDepthLayer         = fluidDepthLayer;
    constants.sceneDepthLayer         = sceneDepthLayer;
    constants.mainLightIndex          = mainDirectionalLightIndex(gpuScene.layout, camera.envIndex);
    constants.fresnel                 = options.fresnel;
    constants.refractionIor           = options.refractionIor;
    constants.refractionViewThickness = options.refractionViewThickness;
    constants.normalReconstructionDepthThreshold = options.depthEdgeThreshold;

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
    drawAttrs.NumVertices = 3u;
    drawAttrs.Flags       = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.graphicsContext->Draw(drawAttrs);
    return true;
}

} // namespace cressim::neo::graphics::detail
