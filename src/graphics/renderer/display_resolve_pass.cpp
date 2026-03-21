#include "graphics/renderer/display_resolve_pass.h"

#include "gpu/shader_library.h"

#include <array>
#include <cstring>

namespace cressim::neo::graphics::detail
{

namespace
{

Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> createResolveBinding(
    Diligent::IPipelineState* pipeline)
{
    if (pipeline == nullptr)
    {
        return {};
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    pipeline->CreateShaderResourceBinding(&srb, true);
    return srb;
}

} // namespace

DisplayResolvePass::DisplayResolvePass(gpu::GpuDevice& device) : mDevice(device) {}

std::size_t DisplayResolvePass::PipelineKeyHasher::operator()(const PipelineKey& key) const noexcept
{
    const std::size_t colorHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat));
    const std::size_t depthHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.depthFormat));
    return colorHash ^ (depthHash << 1u);
}

bool DisplayResolvePass::initialize()
{
    gpu::GpuBackendContext backendContext{};
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

bool DisplayResolvePass::ensureConstants(Diligent::IRenderDevice* renderDevice)
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

Diligent::IPipelineState* DisplayResolvePass::getOrCreatePipeline(
    Diligent::IRenderDevice* renderDevice, const PipelineKey& key)
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
    Diligent::IShaderSourceInputStreamFactory* streamFactory = shaderLibrary.streamFactory();
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
    renderDevice->CreateShader(shaderCreateInfo, &vertexShader);
    if (vertexShader == nullptr)
    {
        return nullptr;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.DisplayResolve.PS";
    shaderCreateInfo.FilePath        = "graphics/display_resolve.ps.hlsl";
    renderDevice->CreateShader(shaderCreateInfo, &pixelShader);
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
    };
    psoCreateInfo.PSODesc.ResourceLayout.Variables = kVars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(std::size(kVars));
    psoCreateInfo.pVS = vertexShader;
    psoCreateInfo.pPS = pixelShader;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipeline;
    renderDevice->CreateGraphicsPipelineState(psoCreateInfo, &pipeline);
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    if (Diligent::IShaderResourceVariable* constantsVar = pipeline->GetStaticVariableByName(
            Diligent::SHADER_TYPE_PIXEL, "GraphicsDisplayResolve"))
    {
        constantsVar->Set(mConstantsBuffer);
    }

    auto insertResult = mPipelines.emplace(key, pipeline);
    return insertResult.first->second;
}

Diligent::IShaderResourceBinding* DisplayResolvePass::getOrCreateResolveBinding(
    Diligent::IPipelineState* pipeline)
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

Diligent::RefCntAutoPtr<Diligent::ITextureView> DisplayResolvePass::createArraySrv(
    Diligent::ITexture* texture) const
{
    if (texture == nullptr)
    {
        return {};
    }

    const Diligent::TextureDesc& textureDesc = texture->GetDesc();
    Diligent::TextureViewDesc viewDesc{};
    viewDesc.ViewType        = Diligent::TEXTURE_VIEW_SHADER_RESOURCE;
    viewDesc.TextureDim      = textureDesc.Type;
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

bool DisplayResolvePass::resolve(const common::FrameContext& frameContext,
                                 const DisplayResolveRequest& request)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr || backendContext.immediateContext == nullptr)
    {
        return false;
    }

    Diligent::ITexture* sourceTexture = nullptr;
    if (!mDevice.renderTargetSystem().tryGetRenderTargetColorTexture(request.sourceBinding.target,
                                                                     sourceTexture) ||
        sourceTexture == nullptr)
    {
        return false;
    }

    const Diligent::TEXTURE_FORMAT depthFormat = request.targetTargetDesc.depth
                                                     ? request.targetTargetDesc.depthFormat
                                                     : Diligent::TEX_FORMAT_UNKNOWN;
    Diligent::IPipelineState* pipeline =
        getOrCreatePipeline(backendContext.renderDevice,
                            PipelineKey{request.targetTargetDesc.colorFormat, depthFormat});
    if (pipeline == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::ITextureView> sourceSrv = createArraySrv(sourceTexture);
    if (sourceSrv == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceBinding* resolveBinding = getOrCreateResolveBinding(pipeline);
    if (resolveBinding == nullptr)
    {
        return false;
    }
    Diligent::IShaderResourceVariable* sourceColorVar =
        resolveBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_SourceColor");
    if (sourceColorVar == nullptr)
    {
        return false;
    }
    sourceColorVar->Set(sourceSrv);

    ResolveConstants constants{};
    constants.layer = request.sourceBinding.firstLayer;
    void* mapped    = nullptr;
    backendContext.immediateContext->MapBuffer(mConstantsBuffer, Diligent::MAP_WRITE,
                                               Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &constants, sizeof(constants));
    backendContext.immediateContext->UnmapBuffer(mConstantsBuffer, Diligent::MAP_WRITE);

    mDevice.renderTargetSystem().setRenderTargetViewport(request.targetBinding,
                                                         gpu::GpuRenderViewport{});
    gpu::GpuRenderPassBeginDesc beginDesc{};
    beginDesc.clearColor         = request.clearColor;
    beginDesc.clearDepth         = request.clearDepth;
    beginDesc.clearColorValue[0] = request.clearColorValue.x;
    beginDesc.clearColorValue[1] = request.clearColorValue.y;
    beginDesc.clearColorValue[2] = request.clearColorValue.z;
    beginDesc.clearColorValue[3] = request.clearColorValue.w;
    beginDesc.clearDepthValue    = request.clearDepthValue;
    mDevice.renderTargetSystem().beginRenderTarget(request.targetBinding, frameContext, beginDesc);

    backendContext.immediateContext->SetPipelineState(pipeline);
    backendContext.immediateContext->CommitShaderResources(
        resolveBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawAttribs drawAttrs{};
    drawAttrs.NumVertices = 3u;
    drawAttrs.Flags       = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.immediateContext->Draw(drawAttrs);

    mDevice.renderTargetSystem().endRenderTarget(request.targetBinding, frameContext);
    return true;
}

} // namespace cressim::neo::graphics::detail
