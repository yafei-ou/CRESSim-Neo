#include "graphics/renderer/passes/forward_opaque_pass.h"

#include "graphics/device/graphics_device_impl.h"

#include <algorithm>
#include <cstring>

namespace cressim::neo::graphics::detail
{

ForwardOpaquePass::ForwardOpaquePass(GraphicsDeviceImpl& device) :
    mDevice(device),
    mShaderSourceProvider("", true)
{
}

bool ForwardOpaquePass::initialize()
{
    mShaderSourceProvider = ShaderSourceProvider(mDevice.shaderSourceDirectory(), mDevice.allowShaderFallback());
    mProgramRegistry = std::make_unique<MaterialProgramRegistry>(mShaderSourceProvider);

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
        textureDesc.Name = "CRESSimNeo.ForwardOpaquePass.FallbackShadow";
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

void ForwardOpaquePass::setShadowMapTargets(
    const std::array<RenderTargetHandle, kShadowCascadeCount>& shadowMapTargets,
    std::uint32_t shadowMapCount)
{
    mShadowMapTargets = shadowMapTargets;
    mShadowMapCount = std::min<std::uint32_t>(shadowMapCount, kShadowCascadeCount);
}

bool ForwardOpaquePass::draw(RenderTargetHandle target, const ForwardDrawCommand& drawCommand)
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

    if (!ensureConstantBuffer(backendContext.renderDevice) || mProgramRegistry == nullptr)
    {
        return false;
    }

    const Diligent::TEXTURE_FORMAT depthFormat =
        backendContext.activeRenderTargetHasDepth ? Diligent::TEX_FORMAT_D32_FLOAT : Diligent::TEX_FORMAT_UNKNOWN;
    const MaterialProgramRegistry::ProgramKey key = MaterialProgramRegistry::buildProgramKey(
        MainPassClass::ForwardOpaque,
        drawCommand.programFamily,
        drawCommand.materialFeatureFlags,
        backendContext.activeRenderTargetColorFormat,
        depthFormat,
        backendContext.activeRenderTargetHasDepth,
        backendContext.activeRenderTargetHasDepth,
        false);
    MaterialProgramRegistry::ProgramResources* program = mProgramRegistry->getOrCreateProgram(backendContext.renderDevice, key);
    if (program == nullptr || program->pipelineState == nullptr)
    {
        return false;
    }
    if (!bindProgramConstants(*program))
    {
        return false;
    }
    if (program->shaderResourceBinding == nullptr)
    {
        program->pipelineState->CreateShaderResourceBinding(&program->shaderResourceBinding, true);
    }
    if (program->shaderResourceBinding == nullptr)
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
            program->shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, kShadowMapVarNames[cascadeIdx]);
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
    constants.pipelineParams = Diligent::float4{
        drawCommand.material.alphaCutoff,
        0.0f,
        0.0f,
        0.0f};

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

    backendContext.immediateContext->SetPipelineState(program->pipelineState);
    backendContext.immediateContext->CommitShaderResources(program->shaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedAttribs drawAttrs{};
    drawAttrs.IndexType = Diligent::VT_UINT32;
    drawAttrs.NumIndices = meshBuffers->indexCount;
    drawAttrs.Flags = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.immediateContext->DrawIndexed(drawAttrs);
    return true;
}

std::size_t ForwardOpaquePass::cachedProgramCount() const noexcept
{
    return mProgramRegistry != nullptr ? mProgramRegistry->cachedProgramCount() : 0u;
}

ForwardOpaquePass::CachedMeshGpuData* ForwardOpaquePass::getOrCreateMeshBuffers(
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
    vertexBufferDesc.Name = "CRESSimNeo.ForwardOpaquePass.VertexBuffer";
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
    indexBufferDesc.Name = "CRESSimNeo.ForwardOpaquePass.IndexBuffer";
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

bool ForwardOpaquePass::ensureConstantBuffer(Diligent::IRenderDevice* renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }
    if (mConstantBuffer != nullptr)
    {
        return true;
    }

    Diligent::BufferDesc constantBufferDesc{};
    constantBufferDesc.Name = "CRESSimNeo.ForwardOpaquePass.Constants";
    constantBufferDesc.Size = sizeof(DrawConstants);
    constantBufferDesc.Usage = Diligent::USAGE_DYNAMIC;
    constantBufferDesc.BindFlags = Diligent::BIND_UNIFORM_BUFFER;
    constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
    renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mConstantBuffer);
    return mConstantBuffer != nullptr;
}

bool ForwardOpaquePass::bindProgramConstants(MaterialProgramRegistry::ProgramResources& program)
{
    if (program.pipelineState == nullptr || mConstantBuffer == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceVariable* vertexConstants =
        program.pipelineState->GetStaticVariableByName(Diligent::SHADER_TYPE_VERTEX, "PbrConstants");
    Diligent::IShaderResourceVariable* pixelConstants =
        program.pipelineState->GetStaticVariableByName(Diligent::SHADER_TYPE_PIXEL, "PbrConstants");
    if (vertexConstants == nullptr || pixelConstants == nullptr)
    {
        return false;
    }
    vertexConstants->Set(mConstantBuffer);
    pixelConstants->Set(mConstantBuffer);
    return true;
}

} // namespace cressim::neo::graphics::detail
