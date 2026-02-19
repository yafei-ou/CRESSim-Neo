#include "graphics/renderer/passes/pbr_pass.h"

#include "graphics/device/graphics_device_impl.h"
#include "graphics/device/shaders/shader_fallback_sources.h"

#include <algorithm>
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
        shadowSamplerDesc.MinFilter = Diligent::FILTER_TYPE_COMPARISON_LINEAR;
        shadowSamplerDesc.MagFilter = Diligent::FILTER_TYPE_COMPARISON_LINEAR;
        shadowSamplerDesc.MipFilter = Diligent::FILTER_TYPE_LINEAR;
        shadowSamplerDesc.AddressU = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.AddressV = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.AddressW = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.ComparisonFunc = Diligent::COMPARISON_FUNC_LESS_EQUAL;
        backendContext.renderDevice->CreateSampler(shadowSamplerDesc, &mShadowSampler);

        Diligent::TextureDesc textureDesc{};
        textureDesc.Name = "CRESSimNeo.PbrPass.FallbackShadow";
        textureDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        textureDesc.Width = 1;
        textureDesc.Height = 1;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = Diligent::TEX_FORMAT_D32_FLOAT;
        textureDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_DEPTH_STENCIL;
        textureDesc.Usage = Diligent::USAGE_DEFAULT;

        Diligent::RefCntAutoPtr<Diligent::ITexture> fallbackTexture;
        backendContext.renderDevice->CreateTexture(textureDesc, nullptr, &fallbackTexture);
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

void PbrPass::setShadowMapTargets(const std::array<RenderTargetHandle, kShadowCascadeCount>& shadowMapTargets, std::uint32_t shadowMapCount)
{
    mShadowMapTargets = shadowMapTargets;
    mShadowMapCount = std::min<std::uint32_t>(shadowMapCount, kShadowCascadeCount);
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

    std::array<Diligent::ITextureView*, kShadowCascadeCount> shadowMapSrvs{};
    bool hasShadowMap = false;
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        Diligent::ITextureView* shadowMapSrv = mFallbackShadowMapSrv;
        if (cascadeIdx < mShadowMapCount && mShadowMapTargets[cascadeIdx].id != common::kInvalidResourceId)
        {
            Diligent::ITexture* depthTexture = nullptr;
            if (mDevice.tryGetRenderTargetDepthTexture(mShadowMapTargets[cascadeIdx], depthTexture) && depthTexture != nullptr)
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
        shadowMapSrvs[cascadeIdx] = shadowMapSrv;
    }

    constexpr const char* kShadowMapVarNames[kShadowCascadeCount] = {
        "g_ShadowMap0",
        "g_ShadowMap1",
        "g_ShadowMap2",
        "g_ShadowMap3"};
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        Diligent::IShaderResourceVariable* shadowMapVar =
            pipeline->shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, kShadowMapVarNames[cascadeIdx]);
        if (shadowMapVar == nullptr)
        {
            return false;
        }
        shadowMapVar->Set(shadowMapSrvs[cascadeIdx]);
    }

    DrawConstants constants{};
    constants.modelMatrix = drawCommand.modelMatrix.Transpose();
    constants.viewMatrix = drawCommand.viewMatrix.Transpose();
    constants.viewProjectionMatrix = drawCommand.viewProjectionMatrix.Transpose();
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        constants.lightViewProjectionMatrices[cascadeIdx] = drawCommand.lightViewProjectionMatrices[cascadeIdx].Transpose();
    }
    constants.normalMatrix = drawCommand.normalMatrix.Transpose();
    constants.cameraPositionMetallic = Diligent::float4{
        drawCommand.cameraPosition.x,
        drawCommand.cameraPosition.y,
        drawCommand.cameraPosition.z,
        drawCommand.material.metallic};
    constants.lightDirectionIntensity = Diligent::float4{
        drawCommand.light.direction.x,
        drawCommand.light.direction.y,
        drawCommand.light.direction.z,
        drawCommand.light.intensity};
    constants.lightColorRoughness = Diligent::float4{
        drawCommand.light.color.x,
        drawCommand.light.color.y,
        drawCommand.light.color.z,
        drawCommand.material.roughness};
    constants.baseColor = Diligent::float4{
        drawCommand.material.baseColor.x,
        drawCommand.material.baseColor.y,
        drawCommand.material.baseColor.z,
        drawCommand.material.opacity};
    constants.cascadeSplits = Diligent::float4{
        drawCommand.cascadeSplits[0],
        drawCommand.cascadeSplits[1],
        drawCommand.cascadeSplits[2],
        drawCommand.cascadeSplits[3]};
    constants.shadowTexelSizeCascadeCount = Diligent::float4{
        drawCommand.shadowMapInvSizeX,
        drawCommand.shadowMapInvSizeY,
        std::min(drawCommand.shadowCascadeCount, static_cast<float>(mShadowMapCount)),
        std::max(drawCommand.light.shadowFadeDistance, 0.001f)};
    constants.shadowParams = Diligent::float4{
        drawCommand.shadowBias,
        hasShadowMap ? 1.0f : 0.0f,
        drawCommand.material.receivesShadows,
        0.35f};

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
        {Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap0", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap1", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap2", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_PIXEL, "g_ShadowMap3", Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE}};
    psoCreateInfo.PSODesc.ResourceLayout.Variables = kVars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables = 4;

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
