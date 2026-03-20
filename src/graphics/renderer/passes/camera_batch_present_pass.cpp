#include "graphics/renderer/passes/camera_batch_present_pass.h"

#include "gpu/shader_library.h"

#include <array>
#include <cstring>

namespace cressim::neo::graphics::detail
{

CameraBatchPresentPass::CameraBatchPresentPass(gpu::GpuDevice& device) : mDevice(device) {}

std::size_t CameraBatchPresentPass::PipelineKeyHasher::operator()(const PipelineKey& key) const noexcept
{
    const std::size_t colorHash = std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat));
    const std::size_t depthHash = std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.depthFormat));
    return colorHash ^ (depthHash << 1u);
}

bool CameraBatchPresentPass::initialize()
{
    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) || backendContext.renderDevice == nullptr)
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

bool CameraBatchPresentPass::ensureConstants(Diligent::IRenderDevice* renderDevice)
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
    constantBufferDesc.Name           = "CRESSimNeo.CameraBatchPresent.GraphicsBatchPresent";
    constantBufferDesc.Size           = sizeof(PresentConstants);
    constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
    constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
    constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
    renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mConstantsBuffer);
    return mConstantsBuffer != nullptr;
}

Diligent::IPipelineState* CameraBatchPresentPass::getOrCreatePipeline(Diligent::IRenderDevice* renderDevice,
                                                                      const PipelineKey& key)
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
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.CameraBatchPresent.VS";
    shaderCreateInfo.FilePath        = "graphics/camera_batch_present.vs.hlsl";
    renderDevice->CreateShader(shaderCreateInfo, &vertexShader);
    if (vertexShader == nullptr)
    {
        return nullptr;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.CameraBatchPresent.PS";
    shaderCreateInfo.FilePath        = "graphics/camera_batch_present.ps.hlsl";
    renderDevice->CreateShader(shaderCreateInfo, &pixelShader);
    if (pixelShader == nullptr)
    {
        return nullptr;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                      = "CRESSimNeo.CameraBatchPresent.PSO";
    psoCreateInfo.PSODesc.PipelineType              = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]    = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat        = key.depthFormat;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology =
        Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    // This pass draws a generated fullscreen triangle; disabling culling avoids
    // relying on backend-specific default front-face winding.
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = Diligent::CULL_MODE_NONE;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = Diligent::False;
    psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0].BlendEnable = Diligent::False;

    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    constexpr Diligent::ShaderResourceVariableDesc kVars[] = {
        {Diligent::SHADER_TYPE_PIXEL, "g_BatchColor",
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

    if (Diligent::IShaderResourceVariable* constantsVar =
            pipeline->GetStaticVariableByName(Diligent::SHADER_TYPE_PIXEL, "GraphicsBatchPresent"))
    {
        constantsVar->Set(mConstantsBuffer);
    }

    auto insertResult = mPipelines.emplace(key, pipeline);
    return insertResult.first->second;
}

Diligent::RefCntAutoPtr<Diligent::ITextureView> CameraBatchPresentPass::createArraySrv(
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
    viewDesc.NumArraySlices  =
        textureDesc.Type == Diligent::RESOURCE_DIM_TEX_2D_ARRAY ? textureDesc.ArraySize : 1u;

    Diligent::RefCntAutoPtr<Diligent::ITextureView> srv;
    texture->CreateView(viewDesc, &srv);
    if (srv != nullptr && mSampler != nullptr)
    {
        srv->SetSampler(mSampler);
    }
    return srv;
}

bool CameraBatchPresentPass::present(const common::FrameContext& frameContext,
                                     gpu::GpuRenderTargetHandle target, Diligent::ITexture* sourceTexture,
                                     const gpu::GpuRenderTargetDesc& targetDesc,
                                     const std::vector<PresentRect>& rects, bool clearColor,
                                     bool clearDepth, const Diligent::float4& clearColorValue,
                                     float clearDepthValue)
{
    if (!mInitialized || sourceTexture == nullptr || rects.empty())
    {
        return false;
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) || backendContext.renderDevice == nullptr ||
        backendContext.immediateContext == nullptr)
    {
        return false;
    }

    const Diligent::TEXTURE_FORMAT depthFormat =
        targetDesc.depth ? targetDesc.depthFormat : Diligent::TEX_FORMAT_UNKNOWN;
    Diligent::IPipelineState* pipeline =
        getOrCreatePipeline(backendContext.renderDevice, PipelineKey{targetDesc.colorFormat, depthFormat});
    if (pipeline == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::ITextureView> sourceSrv = createArraySrv(sourceTexture);
    if (sourceSrv == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    pipeline->CreateShaderResourceBinding(&srb, true);
    if (srb == nullptr)
    {
        return false;
    }
    Diligent::IShaderResourceVariable* batchColorVar =
        srb->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_BatchColor");
    if (batchColorVar == nullptr)
    {
        return false;
    }
    batchColorVar->Set(sourceSrv);

    mDevice.renderTargetSystem().setRenderTargetViewport(target, gpu::GpuRenderViewport{});
    gpu::GpuRenderPassBeginDesc beginDesc{};
    beginDesc.clearColor         = clearColor;
    beginDesc.clearDepth         = clearDepth;
    beginDesc.clearColorValue[0] = clearColorValue.x;
    beginDesc.clearColorValue[1] = clearColorValue.y;
    beginDesc.clearColorValue[2] = clearColorValue.z;
    beginDesc.clearColorValue[3] = clearColorValue.w;
    beginDesc.clearDepthValue    = clearDepthValue;
    mDevice.renderTargetSystem().beginRenderTarget(target, frameContext, beginDesc);

    backendContext.immediateContext->SetPipelineState(pipeline);
    backendContext.immediateContext->CommitShaderResources(
        srb, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    for (const PresentRect& rect : rects)
    {
        PresentConstants constants{};
        constants.layer = rect.layer;
        void* mapped = nullptr;
        backendContext.immediateContext->MapBuffer(mConstantsBuffer, Diligent::MAP_WRITE,
                                                   Diligent::MAP_FLAG_DISCARD, mapped);
        if (mapped == nullptr)
        {
            continue;
        }
        std::memcpy(mapped, &constants, sizeof(constants));
        backendContext.immediateContext->UnmapBuffer(mConstantsBuffer, Diligent::MAP_WRITE);

        Diligent::Viewport viewport{};
        viewport.TopLeftX = rect.viewport.x * static_cast<float>(targetDesc.width);
        viewport.TopLeftY = rect.viewport.y * static_cast<float>(targetDesc.height);
        viewport.Width    = rect.viewport.width * static_cast<float>(targetDesc.width);
        viewport.Height   = rect.viewport.height * static_cast<float>(targetDesc.height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        backendContext.immediateContext->SetViewports(1, &viewport, targetDesc.width,
                                                      targetDesc.height);

        Diligent::DrawAttribs drawAttrs{};
        drawAttrs.NumVertices = 3u;
        drawAttrs.Flags       = Diligent::DRAW_FLAG_VERIFY_ALL;
        backendContext.immediateContext->Draw(drawAttrs);
    }

    mDevice.renderTargetSystem().endRenderTarget(target, frameContext);
    return true;
}

} // namespace cressim::neo::graphics::detail
