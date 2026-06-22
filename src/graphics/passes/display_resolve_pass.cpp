#include "graphics/passes/display_resolve_pass.h"

#include "common/math_utils_runtime.h"
#include "gpu/shader_library.h"
#include <algorithm>

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

#include <array>
#include <cstring>

namespace cressim::neo::graphics::detail
{

namespace
{

enum class ResolveOutputMode : std::uint32_t
{
    Sdr       = 0u,
    HdrLinear = 1u,
};

Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> createResolveBinding(
    Diligent::IPipelineState *pipeline)
{
    if (pipeline == nullptr)
    {
        return {};
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    pipeline->CreateShaderResourceBinding(&srb, true);
    return srb;
}

ResolveOutputMode resolveOutputModeForFormat(Diligent::TEXTURE_FORMAT colorFormat)
{
    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(colorFormat);
    switch (formatAttribs.ComponentType)
    {
    case Diligent::COMPONENT_TYPE_FLOAT:
    case Diligent::COMPONENT_TYPE_COMPOUND:
        return ResolveOutputMode::HdrLinear;
    default:
        return ResolveOutputMode::Sdr;
    }
}

void computePresentationViewport(const DisplayResolveRequest &request,
                                 Diligent::Viewport &outViewport) noexcept
{
    outViewport.TopLeftX = 0.0f;
    outViewport.TopLeftY = 0.0f;
    outViewport.Width    = static_cast<float>(request.presentationTarget.width);
    outViewport.Height   = static_cast<float>(request.presentationTarget.height);
    outViewport.MinDepth = 0.0f;
    outViewport.MaxDepth = 1.0f;

    if (!request.preserveAspectRatio || request.sourceTargetDesc.width == 0u ||
        request.sourceTargetDesc.height == 0u || request.presentationTarget.width == 0u ||
        request.presentationTarget.height == 0u)
    {
        return;
    }

    const float sourceAspect       = static_cast<float>(request.sourceTargetDesc.width) /
                                     static_cast<float>(request.sourceTargetDesc.height);
    const float presentationAspect = static_cast<float>(request.presentationTarget.width) /
                                     static_cast<float>(request.presentationTarget.height);
    if (sourceAspect <= 0.0f || presentationAspect <= 0.0f)
    {
        return;
    }

    if (sourceAspect > presentationAspect)
    {
        const float viewportHeight =
            static_cast<float>(request.presentationTarget.width) / sourceAspect;
        outViewport.Height = std::max(viewportHeight, 1.0f);
        outViewport.TopLeftY =
            0.5f * (static_cast<float>(request.presentationTarget.height) - outViewport.Height);
    }
    else
    {
        const float viewportWidth =
            static_cast<float>(request.presentationTarget.height) * sourceAspect;
        outViewport.Width = std::max(viewportWidth, 1.0f);
        outViewport.TopLeftX =
            0.5f * (static_cast<float>(request.presentationTarget.width) - outViewport.Width);
    }
}

} // namespace

DisplayResolvePass::DisplayResolvePass(gpu::GpuDevice &device) : mDevice(device) {}

std::size_t DisplayResolvePass::PipelineKeyHasher::operator()(const PipelineKey &key) const noexcept
{
    const std::size_t colorHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat));
    const std::size_t depthHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.depthFormat));
    return colorHash ^ (depthHash << 1u);
}

bool DisplayResolvePass::initialize()
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
    backendContext.renderDevice->CreateSampler(samplerDesc, &mSampler);

    mInitialized = ensureConstants(backendContext.renderDevice);
    return mInitialized;
}

bool DisplayResolvePass::ensureConstants(Diligent::IRenderDevice *renderDevice)
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
    constantBufferDesc.Name           = "CRESSimNeo.DisplayResolve.GraphicsDisplayResolve";
    constantBufferDesc.Size           = sizeof(ResolveConstants);
    constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
    constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
    constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
    renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mConstantsBuffer);
    return mConstantsBuffer != nullptr;
}

Diligent::IPipelineState *DisplayResolvePass::getOrCreatePipeline(
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
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.DisplayResolve.VS";
    shaderCreateInfo.FilePath        = "graphics/display_resolve.vs.hlsl";
    if (!mDevice.createShader(shaderCreateInfo, &vertexShader))
    {
        vertexShader = nullptr;
    }
    if (vertexShader == nullptr)
    {
        return nullptr;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.DisplayResolve.PS";
    shaderCreateInfo.FilePath        = "graphics/display_resolve.ps.hlsl";
    if (!mDevice.createShader(shaderCreateInfo, &pixelShader))
    {
        pixelShader = nullptr;
    }
    if (pixelShader == nullptr)
    {
        return nullptr;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                       = "CRESSimNeo.DisplayResolve.PSO";
    psoCreateInfo.PSODesc.PipelineType               = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]     = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = key.depthFormat;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode      = Diligent::CULL_MODE_NONE;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = Diligent::False;
    psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0].BlendEnable = Diligent::False;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    constexpr Diligent::ShaderResourceVariableDesc kVars[] = {
        {Diligent::SHADER_TYPE_PIXEL, "g_SourceColor",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_SourceDepth",
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
        pipeline = nullptr;
    }
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    if (Diligent::IShaderResourceVariable *constantsVar = pipeline->GetStaticVariableByName(
            Diligent::SHADER_TYPE_PIXEL, "GraphicsDisplayResolve"))
    {
        constantsVar->Set(mConstantsBuffer);
    }

    auto insertResult = mPipelines.emplace(key, pipeline);
    return insertResult.first->second;
}

Diligent::IShaderResourceBinding *DisplayResolvePass::getOrCreateResolveBinding(
    Diligent::IPipelineState *pipeline)
{
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    auto it = mResolveBindings.find(pipeline);
    if (it != mResolveBindings.end())
    {
        return it->second;
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb = createResolveBinding(pipeline);
    if (srb == nullptr)
    {
        return nullptr;
    }

    auto insertResult = mResolveBindings.emplace(pipeline, std::move(srb));
    return insertResult.first->second;
}

Diligent::RefCntAutoPtr<Diligent::ITextureView> DisplayResolvePass::createSourceSrv(
    Diligent::ITexture *texture) const
{
    if (texture == nullptr)
    {
        return {};
    }

    const Diligent::TextureDesc &textureDesc = texture->GetDesc();
    Diligent::TextureViewDesc viewDesc{};
    viewDesc.ViewType        = Diligent::TEXTURE_VIEW_SHADER_RESOURCE;
    viewDesc.TextureDim      = Diligent::RESOURCE_DIM_TEX_2D_ARRAY;
    viewDesc.MostDetailedMip = 0u;
    viewDesc.NumMipLevels    = 1u;
    viewDesc.FirstArraySlice = 0u;
    viewDesc.NumArraySlices =
        textureDesc.Type == Diligent::RESOURCE_DIM_TEX_2D_ARRAY ? textureDesc.ArraySize : 1u;

    Diligent::RefCntAutoPtr<Diligent::ITextureView> srv;
    texture->CreateView(viewDesc, &srv);
    if (srv != nullptr && mSampler != nullptr)
    {
        srv->SetSampler(mSampler);
    }
    return srv;
}

bool DisplayResolvePass::resolve(const common::FrameContext &frameContext,
                                 const DisplayResolveRequest &request)
{
    (void)frameContext;
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr || backendContext.graphicsContext == nullptr)
    {
        return false;
    }

    Diligent::ITexture *sourceTexture = nullptr;
    const bool useDepthSource =
        request.sourceKind == RenderFrameOptions::PresentedExplicitOutput::SourceKind::Depth;
    const bool hasSourceTexture = useDepthSource
                                      ? mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(
                                            request.sourceBinding.target, sourceTexture)
                                      : mDevice.renderTargetSystem().tryGetRenderTargetColorTexture(
                                            request.sourceBinding.target, sourceTexture);
    if (!hasSourceTexture || sourceTexture == nullptr)
    {
        return false;
    }

    if (backendContext.primarySwapChain == nullptr)
    {
        return false;
    }

    const Diligent::TEXTURE_FORMAT depthFormat = request.presentationTarget.hasDepth
                                                     ? request.presentationTarget.depthFormat
                                                     : Diligent::TEX_FORMAT_UNKNOWN;
    Diligent::IPipelineState *pipeline =
        getOrCreatePipeline(backendContext.renderDevice,
                            PipelineKey{request.presentationTarget.colorFormat, depthFormat});
    if (pipeline == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::ITextureView> sourceSrv = createSourceSrv(sourceTexture);
    if (sourceSrv == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceBinding *resolveBinding = getOrCreateResolveBinding(pipeline);
    if (resolveBinding == nullptr)
    {
        return false;
    }
    Diligent::IShaderResourceVariable *sourceColorVar =
        resolveBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_SourceColor");
    Diligent::IShaderResourceVariable *sourceDepthVar =
        resolveBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_SourceDepth");
    if (sourceColorVar == nullptr || sourceDepthVar == nullptr)
    {
        return false;
    }
    sourceColorVar->Set(useDepthSource ? nullptr : sourceSrv);
    sourceDepthVar->Set(useDepthSource ? sourceSrv : nullptr);

    ResolveConstants constants{};
    constants.layer      = request.sourceBinding.firstLayer;
    constants.outputMode = static_cast<std::uint32_t>(
        resolveOutputModeForFormat(request.presentationTarget.colorFormat));
    constants.toneMapper             = static_cast<std::uint32_t>(request.toneMapper);
    constants.sourceIsDisplayEncoded = request.sourceIsDisplayEncoded ? 1u : 0u;
    constants.sourceKind             = static_cast<std::uint32_t>(request.sourceKind);
    constants.resolveParams[0]       = request.exposure;
    constants.resolveParams[1]       = request.nearClip;
    constants.resolveParams[2]       = request.farClip;
    void *mapped                     = nullptr;
    backendContext.graphicsContext->MapBuffer(mConstantsBuffer, Diligent::MAP_WRITE,
                                              Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &constants, sizeof(constants));
    backendContext.graphicsContext->UnmapBuffer(mConstantsBuffer, Diligent::MAP_WRITE);

    const auto &swapChainDesc = backendContext.primarySwapChain->GetDesc();
    if (swapChainDesc.Width != request.presentationTarget.width ||
        swapChainDesc.Height != request.presentationTarget.height)
    {
        backendContext.primarySwapChain->Resize(
            common::runtime_math::clampExtent(request.presentationTarget.width),
            common::runtime_math::clampExtent(request.presentationTarget.height),
            Diligent::SURFACE_TRANSFORM_OPTIMAL);
    }

    Diligent::ITextureView *backBufferRtv =
        backendContext.primarySwapChain->GetCurrentBackBufferRTV();
    Diligent::ITextureView *depthBufferDsv =
        request.presentationTarget.hasDepth ? backendContext.primarySwapChain->GetDepthBufferDSV()
                                            : nullptr;
    if (backBufferRtv == nullptr)
    {
        return false;
    }

    backendContext.graphicsContext->SetRenderTargets(
        1, &backBufferRtv, depthBufferDsv, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    if (request.clearColor)
    {
        const float clearColor[4] = {request.clearColorValue.x, request.clearColorValue.y,
                                     request.clearColorValue.z, request.clearColorValue.w};
        backendContext.graphicsContext->ClearRenderTarget(
            backBufferRtv, clearColor, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    if (depthBufferDsv != nullptr && request.clearDepth)
    {
        backendContext.graphicsContext->ClearDepthStencil(
            depthBufferDsv, Diligent::CLEAR_DEPTH_FLAG, request.clearDepthValue, 0,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    Diligent::Viewport diligentViewport{};
    computePresentationViewport(request, diligentViewport);
    backendContext.graphicsContext->SetViewports(
        1, &diligentViewport, request.presentationTarget.width, request.presentationTarget.height);

    backendContext.graphicsContext->SetPipelineState(pipeline);
    backendContext.graphicsContext->CommitShaderResources(
        resolveBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawAttribs drawAttrs{};
    drawAttrs.NumVertices = 3u;
    drawAttrs.Flags       = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.graphicsContext->Draw(drawAttrs);

    backendContext.graphicsContext->SetRenderTargets(0, nullptr, nullptr,
                                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
    return true;
}

} // namespace cressim::neo::graphics::detail
