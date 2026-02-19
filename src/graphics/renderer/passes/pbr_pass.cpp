#include "graphics/renderer/passes/pbr_pass.h"

#include "graphics/device/graphics_device_impl.h"
#include "graphics/device/shaders/shader_fallback_sources.h"

#include <cstring>
#include <string>

namespace cressim::neo::graphics::detail
{

namespace
{

std::uint64_t makePipelineCacheKey(bool hasDepthTarget, Diligent::TEXTURE_FORMAT colorFormat)
{
    const std::uint64_t depthBit = hasDepthTarget ? (1ull << 32ull) : 0ull;
    const std::uint64_t colorBits = static_cast<std::uint64_t>(static_cast<std::uint32_t>(colorFormat));
    return depthBit | colorBits;
}

} // namespace

PbrPass::PbrPass(GraphicsDeviceImpl& device) :
    mDevice(device),
    mShaderSourceProvider("", true)
{
}

bool PbrPass::initialize()
{
    mShaderSourceProvider = ShaderSourceProvider(mDevice.shaderSourceDirectory(), mDevice.allowShaderFallback());

    GraphicsDeviceImpl::VulkanBackendContext backendContext{};
    if (mDevice.tryGetVulkanContext(backendContext) && backendContext.renderDevice != nullptr)
    {
        Diligent::SamplerDesc shadowSamplerDesc{};
        shadowSamplerDesc.MinFilter = Diligent::FILTER_TYPE_LINEAR;
        shadowSamplerDesc.MagFilter = Diligent::FILTER_TYPE_LINEAR;
        shadowSamplerDesc.MipFilter = Diligent::FILTER_TYPE_LINEAR;
        shadowSamplerDesc.AddressU = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.AddressV = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.AddressW = Diligent::TEXTURE_ADDRESS_CLAMP;
        backendContext.renderDevice->CreateSampler(shadowSamplerDesc, &mShadowSampler);

        Diligent::TextureDesc textureDesc{};
        textureDesc.Name = "CRESSimNeo.PbrPass.FallbackShadow";
        textureDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        textureDesc.Width = 1;
        textureDesc.Height = 1;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = Diligent::TEX_FORMAT_R32_FLOAT;
        textureDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE;
        textureDesc.Usage = Diligent::USAGE_IMMUTABLE;

        const float whiteShadow = 1.0f;
        Diligent::TextureSubResData subresource{&whiteShadow, sizeof(float)};
        Diligent::TextureData textureData{};
        textureData.pSubResources = &subresource;
        textureData.NumSubresources = 1;

        Diligent::RefCntAutoPtr<Diligent::ITexture> fallbackTexture;
        backendContext.renderDevice->CreateTexture(textureDesc, &textureData, &fallbackTexture);
        if (fallbackTexture != nullptr)
        {
            mFallbackShadowMapSrv = fallbackTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
            if (mFallbackShadowMapSrv != nullptr && mShadowSampler != nullptr)
            {
                mFallbackShadowMapSrv->SetSampler(mShadowSampler);
            }
        }
    }

    mInitialized = true;
    return true;
}

void PbrPass::setShadowMapTarget(RenderTargetHandle shadowMapTarget)
{
    mShadowMapTarget = shadowMapTarget;
}

bool PbrPass::draw(RenderTargetHandle target, const ForwardDrawCommand& drawCommand)
{
    if (!mInitialized)
    {
        return false;
    }

    GraphicsDeviceImpl::VulkanBackendContext backendContext{};
    if (!mDevice.tryGetVulkanContext(backendContext))
    {
        return false;
    }
    if (!backendContext.hasActiveRenderTarget || backendContext.activeRenderTargetId != target.id)
    {
        return false;
    }
    if (backendContext.renderDevice == nullptr || backendContext.immediateContext == nullptr)
    {
        return false;
    }
    if (backendContext.activeRenderTargetColorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return false;
    }

    if (drawCommand.meshId == common::kInvalidResourceId || drawCommand.vertexData == nullptr || drawCommand.indexData == nullptr)
    {
        return false;
    }
    if (drawCommand.vertexCount == 0 || drawCommand.indexCount < 3 || drawCommand.vertexStrideBytes == 0)
    {
        return false;
    }

    CachedMeshGpuData* meshBuffers = getOrCreateMeshBuffers(drawCommand, backendContext.renderDevice);
    if (meshBuffers == nullptr || meshBuffers->vertexBuffer == nullptr || meshBuffers->indexBuffer == nullptr || meshBuffers->indexCount == 0)
    {
        return false;
    }

    PipelineResources* pipeline = getOrCreatePipeline(
        backendContext.renderDevice,
        backendContext.activeRenderTargetHasDepth,
        backendContext.activeRenderTargetColorFormat);
    if (pipeline == nullptr || pipeline->pipelineState == nullptr || pipeline->shaderResourceBinding == nullptr || mConstantBuffer == nullptr)
    {
        return false;
    }

    Diligent::ITextureView* shadowMapSrv = mFallbackShadowMapSrv;
    bool hasShadowMap = false;
    if (mShadowMapTarget.id != common::kInvalidResourceId)
    {
        Diligent::ITexture* depthTexture = nullptr;
        if (mDevice.tryGetRenderTargetDepthTexture(mShadowMapTarget, depthTexture) && depthTexture != nullptr)
        {
            Diligent::ITextureView* depthSrv = depthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
            if (depthSrv != nullptr)
            {
                if (mShadowSampler != nullptr)
                {
                    depthSrv->SetSampler(mShadowSampler);
                }
                shadowMapSrv = depthSrv;
                hasShadowMap = true;
            }
        }
    }

    if (shadowMapSrv == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceVariable* shadowMapVar =
        pipeline->shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap");
    if (shadowMapVar == nullptr)
    {
        return false;
    }
    shadowMapVar->Set(shadowMapSrv);

    DrawConstants constants{};
    std::memcpy(constants.modelMatrix, drawCommand.modelMatrix, sizeof(constants.modelMatrix));
    std::memcpy(constants.viewProjectionMatrix, drawCommand.viewProjectionMatrix, sizeof(constants.viewProjectionMatrix));
    std::memcpy(constants.lightViewProjectionMatrix, drawCommand.lightViewProjectionMatrix, sizeof(constants.lightViewProjectionMatrix));
    constants.cameraPositionMetallic[0] = drawCommand.cameraPosition[0];
    constants.cameraPositionMetallic[1] = drawCommand.cameraPosition[1];
    constants.cameraPositionMetallic[2] = drawCommand.cameraPosition[2];
    constants.cameraPositionMetallic[3] = drawCommand.material.metallic;
    constants.lightDirectionIntensity[0] = drawCommand.light.direction[0];
    constants.lightDirectionIntensity[1] = drawCommand.light.direction[1];
    constants.lightDirectionIntensity[2] = drawCommand.light.direction[2];
    constants.lightDirectionIntensity[3] = drawCommand.light.intensity;
    constants.lightColorRoughness[0] = drawCommand.light.color[0];
    constants.lightColorRoughness[1] = drawCommand.light.color[1];
    constants.lightColorRoughness[2] = drawCommand.light.color[2];
    constants.lightColorRoughness[3] = drawCommand.material.roughness;
    constants.baseColor[0] = drawCommand.material.baseColor[0];
    constants.baseColor[1] = drawCommand.material.baseColor[1];
    constants.baseColor[2] = drawCommand.material.baseColor[2];
    constants.baseColor[3] = drawCommand.material.opacity;
    constants.shadowParams[0] = drawCommand.shadowBias;
    constants.shadowParams[1] = hasShadowMap ? 1.0f : 0.0f;
    constants.shadowParams[2] = drawCommand.material.receivesShadows;

    void* mappedConstants = nullptr;
    backendContext.immediateContext->MapBuffer(mConstantBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &constants, sizeof(constants));
    backendContext.immediateContext->UnmapBuffer(mConstantBuffer, Diligent::MAP_WRITE);

    const Diligent::Uint64 vertexOffset = 0;
    Diligent::IBuffer* vertexBuffers[] = {meshBuffers->vertexBuffer};
    backendContext.immediateContext->SetVertexBuffers(
        0,
        1,
        vertexBuffers,
        &vertexOffset,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        Diligent::SET_VERTEX_BUFFERS_FLAG_RESET);
    backendContext.immediateContext->SetIndexBuffer(meshBuffers->indexBuffer, 0, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    backendContext.immediateContext->SetPipelineState(pipeline->pipelineState);
    backendContext.immediateContext->CommitShaderResources(pipeline->shaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedAttribs drawAttrs{};
    drawAttrs.IndexType = Diligent::VT_UINT32;
    drawAttrs.NumIndices = meshBuffers->indexCount;
    drawAttrs.Flags = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.immediateContext->DrawIndexed(drawAttrs);
    return true;
}

PbrPass::CachedMeshGpuData* PbrPass::getOrCreateMeshBuffers(
    const ForwardDrawCommand& drawCommand,
    Diligent::IRenderDevice* renderDevice)
{
    if (renderDevice == nullptr)
    {
        return nullptr;
    }

    auto& mesh = mCachedMeshes[drawCommand.meshId];
    const bool recreate =
        mesh.vertexBuffer == nullptr ||
        mesh.indexBuffer == nullptr ||
        mesh.version != drawCommand.meshVersion;
    if (!recreate)
    {
        return &mesh;
    }

    Diligent::BufferDesc vertexBufferDesc{};
    vertexBufferDesc.Name = "CRESSimNeo.PbrPass.VertexBuffer";
    vertexBufferDesc.Usage = Diligent::USAGE_IMMUTABLE;
    vertexBufferDesc.BindFlags = Diligent::BIND_VERTEX_BUFFER;
    vertexBufferDesc.Size = static_cast<Diligent::Uint64>(drawCommand.vertexCount) * drawCommand.vertexStrideBytes;

    Diligent::BufferData vertexData{};
    vertexData.pData = drawCommand.vertexData;
    vertexData.DataSize = vertexBufferDesc.Size;
    renderDevice->CreateBuffer(vertexBufferDesc, &vertexData, &mesh.vertexBuffer);
    if (mesh.vertexBuffer == nullptr)
    {
        mCachedMeshes.erase(drawCommand.meshId);
        return nullptr;
    }

    Diligent::BufferDesc indexBufferDesc{};
    indexBufferDesc.Name = "CRESSimNeo.PbrPass.IndexBuffer";
    indexBufferDesc.Usage = Diligent::USAGE_IMMUTABLE;
    indexBufferDesc.BindFlags = Diligent::BIND_INDEX_BUFFER;
    indexBufferDesc.Size = static_cast<Diligent::Uint64>(drawCommand.indexCount) * sizeof(std::uint32_t);

    Diligent::BufferData indexData{};
    indexData.pData = drawCommand.indexData;
    indexData.DataSize = indexBufferDesc.Size;
    renderDevice->CreateBuffer(indexBufferDesc, &indexData, &mesh.indexBuffer);
    if (mesh.indexBuffer == nullptr)
    {
        mCachedMeshes.erase(drawCommand.meshId);
        return nullptr;
    }

    mesh.version = drawCommand.meshVersion;
    mesh.indexCount = drawCommand.indexCount;
    return &mesh;
}

bool PbrPass::createPipeline(
    Diligent::IRenderDevice* renderDevice,
    bool hasDepthTarget,
    Diligent::TEXTURE_FORMAT colorFormat,
    PipelineResources& outResources)
{
    if (renderDevice == nullptr || colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return false;
    }

    std::string pbrVsSource;
    if (!mShaderSourceProvider.loadSource("pbr.vs.hlsl", shaders::pbrVertex(), pbrVsSource))
    {
        return false;
    }

    std::string pbrPsSource;
    if (!mShaderSourceProvider.loadSource("pbr.ps.hlsl", shaders::pbrPixel(), pbrPsSource))
    {
        return false;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.CompileFlags = Diligent::SHADER_COMPILE_FLAG_PACK_MATRIX_ROW_MAJOR;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint = "main";

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name = hasDepthTarget ? "CRESSimNeo.PbrPass.VS.Depth" : "CRESSimNeo.PbrPass.VS.NoDepth";
    shaderCreateInfo.Source = pbrVsSource.c_str();
    renderDevice->CreateShader(shaderCreateInfo, &vertexShader);
    if (vertexShader == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name = hasDepthTarget ? "CRESSimNeo.PbrPass.PS.Depth" : "CRESSimNeo.PbrPass.PS.NoDepth";
    shaderCreateInfo.Source = pbrPsSource.c_str();
    renderDevice->CreateShader(shaderCreateInfo, &pixelShader);
    if (pixelShader == nullptr)
    {
        return false;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name = hasDepthTarget ? "CRESSimNeo.PbrPass.PSO.Depth" : "CRESSimNeo.PbrPass.PSO.NoDepth";
    psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0] = colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat = hasDepthTarget ? Diligent::TEX_FORMAT_D32_FLOAT : Diligent::TEX_FORMAT_UNKNOWN;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = Diligent::CULL_MODE_BACK;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.FrontCounterClockwise = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = hasDepthTarget ? Diligent::True : Diligent::False;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = hasDepthTarget ? Diligent::True : Diligent::False;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType = Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    constexpr Diligent::ShaderResourceVariableDesc kVars[] = {
        {Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE}};
    psoCreateInfo.PSODesc.ResourceLayout.Variables = kVars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables = 1;

    constexpr Diligent::LayoutElement kLayoutElements[] = {
        Diligent::LayoutElement{0, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{1, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{2, 0, 2, Diligent::VT_FLOAT32, Diligent::False}};
    psoCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = kLayoutElements;
    psoCreateInfo.GraphicsPipeline.InputLayout.NumElements = 3;
    psoCreateInfo.pVS = vertexShader;
    psoCreateInfo.pPS = pixelShader;

    renderDevice->CreateGraphicsPipelineState(psoCreateInfo, &outResources.pipelineState);
    if (outResources.pipelineState == nullptr)
    {
        return false;
    }

    if (mConstantBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name = "CRESSimNeo.PbrPass.Constants";
        constantBufferDesc.Size = sizeof(DrawConstants);
        constantBufferDesc.Usage = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
        renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mConstantBuffer);
        if (mConstantBuffer == nullptr)
        {
            return false;
        }
    }

    Diligent::IShaderResourceVariable* vertexConstants =
        outResources.pipelineState->GetStaticVariableByName(Diligent::SHADER_TYPE_VERTEX, "PbrConstants");
    Diligent::IShaderResourceVariable* pixelConstants =
        outResources.pipelineState->GetStaticVariableByName(Diligent::SHADER_TYPE_PIXEL, "PbrConstants");
    if (vertexConstants == nullptr || pixelConstants == nullptr)
    {
        return false;
    }
    vertexConstants->Set(mConstantBuffer);
    pixelConstants->Set(mConstantBuffer);
    outResources.pipelineState->CreateShaderResourceBinding(&outResources.shaderResourceBinding, true);
    return outResources.shaderResourceBinding != nullptr;
}

PbrPass::PipelineResources* PbrPass::getOrCreatePipeline(
    Diligent::IRenderDevice* renderDevice,
    bool hasDepthTarget,
    Diligent::TEXTURE_FORMAT colorFormat)
{
    if (renderDevice == nullptr || colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return nullptr;
    }

    const std::uint64_t key = makePipelineCacheKey(hasDepthTarget, colorFormat);
    auto cacheIt = mPipelineCache.find(key);
    if (cacheIt != mPipelineCache.end())
    {
        if (cacheIt->second.pipelineState != nullptr && cacheIt->second.shaderResourceBinding != nullptr)
        {
            return &cacheIt->second;
        }
    }

    PipelineResources resources{};
    if (!createPipeline(renderDevice, hasDepthTarget, colorFormat, resources))
    {
        return nullptr;
    }

    auto insertResult = mPipelineCache.emplace(key, std::move(resources));
    return &insertResult.first->second;
}

} // namespace cressim::neo::graphics::detail
