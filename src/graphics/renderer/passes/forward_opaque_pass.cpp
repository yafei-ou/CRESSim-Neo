#include "graphics/renderer/passes/forward_opaque_pass.h"

#include <algorithm>
#include <cstring>

namespace cressim::neo::graphics::detail
{

ForwardOpaquePass::ForwardOpaquePass(gpu::GpuDevice& device)
    : mDevice(device), mShaderLibrary(""), mMeshGpuCache("CRESSimNeo.ForwardOpaquePass")
{
}

bool ForwardOpaquePass::initialize()
{
    mShaderLibrary   = gpu::ShaderLibrary(mDevice.shaderSourceDirectory());
    mProgramRegistry = std::make_unique<MaterialProgramRegistry>(mShaderLibrary);

    gpu::GpuBackendContext backendContext{};
    if (mDevice.tryGetGraphicsBackendContext(backendContext) &&
        backendContext.renderDevice != nullptr)
    {
        Diligent::SamplerDesc shadowSamplerDesc{};
        shadowSamplerDesc.MinFilter      = Diligent::FILTER_TYPE_COMPARISON_LINEAR;
        shadowSamplerDesc.MagFilter      = Diligent::FILTER_TYPE_COMPARISON_LINEAR;
        shadowSamplerDesc.MipFilter      = Diligent::FILTER_TYPE_LINEAR;
        shadowSamplerDesc.AddressU       = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.AddressV       = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.AddressW       = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.ComparisonFunc = Diligent::COMPARISON_FUNC_LESS_EQUAL;
        backendContext.renderDevice->CreateSampler(shadowSamplerDesc, &mShadowSampler);

        Diligent::TextureDesc textureDesc{};
        textureDesc.Name      = "CRESSimNeo.ForwardOpaquePass.FallbackShadow";
        textureDesc.Type      = Diligent::RESOURCE_DIM_TEX_2D;
        textureDesc.Width     = 1;
        textureDesc.Height    = 1;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format    = Diligent::TEX_FORMAT_D32_FLOAT;
        textureDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_DEPTH_STENCIL;
        textureDesc.Usage     = Diligent::USAGE_DEFAULT;

        Diligent::RefCntAutoPtr<Diligent::ITexture> fallbackTexture;
        backendContext.renderDevice->CreateTexture(textureDesc, nullptr, &fallbackTexture);
        if (fallbackTexture != nullptr)
        {
            mFallbackShadowMapSrv =
                fallbackTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
            if (mFallbackShadowMapSrv != nullptr && mShadowSampler != nullptr)
            {
                mFallbackShadowMapSrv->SetSampler(mShadowSampler);
            }
        }
    }

    mInitialized = true;
    return true;
}

bool ForwardOpaquePass::beginCameraFrame(const FrameViewData& frameView)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext))
    {
        return false;
    }
    if (backendContext.renderDevice == nullptr || backendContext.immediateContext == nullptr)
    {
        return false;
    }
    if (!ensureConstantBuffers(backendContext.renderDevice))
    {
        return false;
    }

    ForwardPerFrameConstants frameConstants{};
    frameConstants.viewMatrix           = frameView.viewMatrix.Transpose();
    frameConstants.viewProjectionMatrix = frameView.viewProjectionMatrix.Transpose();
    frameConstants.cameraPosition =
        Diligent::float4{frameView.cameraWorldPosition.x, frameView.cameraWorldPosition.y,
                         frameView.cameraWorldPosition.z, 0.0f};
    frameConstants.lightDirectionIntensity =
        Diligent::float4{frameView.light.direction.x, frameView.light.direction.y,
                         frameView.light.direction.z, frameView.light.intensity};
    frameConstants.lightColor = Diligent::float4{frameView.light.color.x, frameView.light.color.y,
                                                 frameView.light.color.z, 0.0f};
    frameConstants.shadowParams =
        Diligent::float4{0.0015f, hasAnyShadowMap() ? 1.0f : 0.0f, 0.35f, 0.0f};
    frameConstants.currentCameraIndex =
        frameView.envIndex * std::max(mSceneView.layout.maxCamerasPerEnv, 1u) + frameView.cameraSlot;

    void* mappedConstants = nullptr;
    backendContext.immediateContext->MapBuffer(mForwardPerFrameBuffer, Diligent::MAP_WRITE,
                                               Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &frameConstants, sizeof(frameConstants));
    backendContext.immediateContext->UnmapBuffer(mForwardPerFrameBuffer, Diligent::MAP_WRITE);
    return true;
}

void ForwardOpaquePass::setGpuSceneView(const gpu::GpuEntitySceneView& sceneView) noexcept
{
    mSceneView = sceneView;
}

void ForwardOpaquePass::setVisibleObjectIndexBuffer(Diligent::IBuffer* buffer) noexcept
{
    mVisibleObjectIndexBuffer = buffer;
}

void ForwardOpaquePass::setShadowMapTargets(
    const std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount>& shadowMapTargets,
    std::uint32_t shadowMapCount)
{
    mShadowMapTargets = shadowMapTargets;
    mShadowMapCount   = std::min<std::uint32_t>(shadowMapCount, kShadowCascadeCount);
}

bool ForwardOpaquePass::prepareDraw(gpu::GpuRenderTargetHandle target,
                                    const ForwardDrawCommand& drawCommand, DrawSetup& outSetup)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext))
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

    if (drawCommand.meshId == common::kInvalidResourceId || drawCommand.vertexData == nullptr ||
        drawCommand.indexData == nullptr)
    {
        return false;
    }
    if (drawCommand.vertexCount == 0 || drawCommand.indexCount < 3 ||
        drawCommand.vertexStrideBytes == 0)
    {
        return false;
    }

    MeshGpuCache::CachedBuffers* meshBuffers =
        mMeshGpuCache.getOrCreate(drawCommand, backendContext.renderDevice);
    if (meshBuffers == nullptr || meshBuffers->vertexBuffer == nullptr ||
        meshBuffers->indexBuffer == nullptr || meshBuffers->indexCount == 0)
    {
        return false;
    }

    if (!ensureConstantBuffers(backendContext.renderDevice) || mProgramRegistry == nullptr)
    {
        return false;
    }

    const Diligent::TEXTURE_FORMAT depthFormat    = backendContext.activeRenderTargetHasDepth
                                                        ? Diligent::TEX_FORMAT_D32_FLOAT
                                                        : Diligent::TEX_FORMAT_UNKNOWN;
    const MaterialProgramRegistry::ProgramKey key = MaterialProgramRegistry::buildProgramKey(
        MainPassClass::ForwardOpaque, drawCommand.programFamily, drawCommand.materialFeatureFlags,
        backendContext.activeRenderTargetColorFormat, depthFormat,
        backendContext.activeRenderTargetHasDepth, backendContext.activeRenderTargetHasDepth,
        false);
    MaterialProgramRegistry::ProgramResources* program =
        mProgramRegistry->getOrCreateProgram(backendContext.renderDevice, key);
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

    outSetup.backendContext = backendContext;
    outSetup.meshBuffers = meshBuffers;
    outSetup.program = program;
    outSetup.useSceneBuffers = drawCommand.instanceIndex != 0xffffffffu &&
                               mSceneView.poses.positionsBuffer != nullptr &&
                               mSceneView.poses.orientationsBuffer != nullptr &&
                               mSceneView.poses.scalesBuffer != nullptr &&
                               mSceneView.renderableMetadataBuffer != nullptr &&
                               mSceneView.renderableVisibilityFlagsBuffer != nullptr &&
                               mSceneView.preparedCamerasBuffer != nullptr;
    return true;
}

bool ForwardOpaquePass::bindShadowMaps(MaterialProgramRegistry::ProgramResources& program)
{
    std::array<Diligent::ITextureView*, kShadowCascadeCount> shadowMapSrvs{};
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        Diligent::ITextureView* shadowMapSrv = mFallbackShadowMapSrv;
        if (cascadeIdx < mShadowMapCount &&
            mShadowMapTargets[cascadeIdx].id != common::kInvalidResourceId)
        {
            Diligent::ITexture* depthTexture = nullptr;
            if (mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(
                    mShadowMapTargets[cascadeIdx], depthTexture) &&
                depthTexture != nullptr)
            {
                Diligent::ITextureView* depthSrv =
                    depthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
                if (depthSrv != nullptr)
                {
                    if (mShadowSampler != nullptr)
                    {
                        depthSrv->SetSampler(mShadowSampler);
                    }
                    shadowMapSrv = depthSrv;
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
        "g_ShadowMap0", "g_ShadowMap1", "g_ShadowMap2", "g_ShadowMap3"};
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        Diligent::IShaderResourceVariable* shadowMapVar =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                             kShadowMapVarNames[cascadeIdx]);
        if (shadowMapVar == nullptr)
        {
            return false;
        }
        shadowMapVar->Set(shadowMapSrvs[cascadeIdx]);
    }
    return true;
}

bool ForwardOpaquePass::bindSceneBuffers(MaterialProgramRegistry::ProgramResources& program) const
{
    if (program.shaderResourceBinding == nullptr)
    {
        return false;
    }
    struct VariableBinding
    {
        const char* name;
        Diligent::IBuffer* buffer;
    };
    const VariableBinding bindings[] = {
        {"g_EntityPositions", mSceneView.poses.positionsBuffer},
        {"g_EntityOrientations", mSceneView.poses.orientationsBuffer},
        {"g_EntityScales", mSceneView.poses.scalesBuffer},
        {"g_RenderableMetadata", mSceneView.renderableMetadataBuffer},
        {"g_RenderableVisibilityFlags", mSceneView.renderableVisibilityFlagsBuffer},
    };
    for (const VariableBinding& binding : bindings)
    {
        Diligent::IShaderResourceVariable* variable =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                             binding.name);
        if (variable == nullptr || binding.buffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView* srv =
            binding.buffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (srv == nullptr)
        {
            return false;
        }
        variable->Set(srv);
    }

    Diligent::IShaderResourceVariable* visibleObjectsVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                         "g_VisibleObjectIndices");
    if (visibleObjectsVar != nullptr)
    {
        Diligent::IBuffer* visibleObjectBuffer = mVisibleObjectIndexBuffer != nullptr
                                                     ? mVisibleObjectIndexBuffer
                                                     : mSceneView.renderableVisibilityFlagsBuffer;
        if (visibleObjectBuffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView* visibleObjectsSrv =
            visibleObjectBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (visibleObjectsSrv == nullptr)
        {
            return false;
        }
        visibleObjectsVar->Set(visibleObjectsSrv);
    }

    Diligent::IShaderResourceVariable* preparedCameraVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                         "g_PreparedCameras");
    if (preparedCameraVar == nullptr)
    {
        return false;
    }
    Diligent::IBufferView* preparedCameraSrv =
        mSceneView.preparedCamerasBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
    if (preparedCameraSrv == nullptr)
    {
        return false;
    }
    preparedCameraVar->Set(preparedCameraSrv);
    return true;
}

bool ForwardOpaquePass::updatePerDrawConstants(Diligent::IDeviceContext* immediateContext,
                                               const ForwardDrawCommand& drawCommand,
                                               bool useSceneBuffers)
{
    PerObjectConstants objectConstants{};
    objectConstants.modelMatrix  = drawCommand.modelMatrix.Transpose();
    objectConstants.normalMatrix = drawCommand.normalMatrix.Transpose();
    objectConstants.instanceIndex = drawCommand.instanceIndex;
    objectConstants.useSceneBuffers = useSceneBuffers ? 1u : 0u;
    objectConstants.drawListOffset = drawCommand.drawListOffset;
    objectConstants.useDrawListBuffer = drawCommand.useDrawListBuffer;

    ForwardPerMaterialConstants materialConstants{};
    materialConstants.baseColor =
        Diligent::float4{drawCommand.material.baseColor.x, drawCommand.material.baseColor.y,
                         drawCommand.material.baseColor.z, drawCommand.material.opacity};
    materialConstants.materialParams =
        Diligent::float4{drawCommand.material.metallic, drawCommand.material.roughness,
                         drawCommand.material.alphaCutoff, drawCommand.material.receivesShadows};

    void* mappedConstants = nullptr;
    immediateContext->MapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                                mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &objectConstants, sizeof(objectConstants));
    immediateContext->UnmapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE);

    mappedConstants = nullptr;
    immediateContext->MapBuffer(mForwardPerMaterialBuffer, Diligent::MAP_WRITE,
                                Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &materialConstants, sizeof(materialConstants));
    immediateContext->UnmapBuffer(mForwardPerMaterialBuffer, Diligent::MAP_WRITE);
    return true;
}

void ForwardOpaquePass::bindGeometry(Diligent::IDeviceContext* immediateContext,
                                     const MeshGpuCache::CachedBuffers& meshBuffers) const
{
    const Diligent::Uint64 vertexOffset = 0;
    Diligent::IBuffer* vertexBuffers[]  = {meshBuffers.vertexBuffer};
    immediateContext->SetVertexBuffers(
        0, 1, vertexBuffers, &vertexOffset, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        Diligent::SET_VERTEX_BUFFERS_FLAG_RESET);
    immediateContext->SetIndexBuffer(
        meshBuffers.indexBuffer, 0, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

bool ForwardOpaquePass::draw(gpu::GpuRenderTargetHandle target,
                             const ForwardDrawCommand& drawCommand)
{
    DrawSetup setup{};
    if (!prepareDraw(target, drawCommand, setup) || !bindShadowMaps(*setup.program))
    {
        return false;
    }
    if (setup.useSceneBuffers && !bindSceneBuffers(*setup.program))
    {
        return false;
    }
    if (!updatePerDrawConstants(setup.backendContext.immediateContext, drawCommand,
                                setup.useSceneBuffers))
    {
        return false;
    }
    bindGeometry(setup.backendContext.immediateContext, *setup.meshBuffers);

    setup.backendContext.immediateContext->SetPipelineState(setup.program->pipelineState);
    setup.backendContext.immediateContext->CommitShaderResources(
        setup.program->shaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedAttribs drawAttrs{};
    drawAttrs.IndexType  = Diligent::VT_UINT32;
    drawAttrs.NumIndices = setup.meshBuffers->indexCount;
    drawAttrs.Flags      = Diligent::DRAW_FLAG_VERIFY_ALL;
    setup.backendContext.immediateContext->DrawIndexed(drawAttrs);
    return true;
}

bool ForwardOpaquePass::drawIndirect(gpu::GpuRenderTargetHandle target,
                                     const ForwardDrawCommand& drawCommand,
                                     Diligent::IBuffer* indirectArgsBuffer,
                                     Diligent::Uint64 argsOffsetBytes)
{
    DrawSetup setup{};
    if (!prepareDraw(target, drawCommand, setup) || indirectArgsBuffer == nullptr ||
        !bindShadowMaps(*setup.program))
    {
        return false;
    }
    if (setup.useSceneBuffers && !bindSceneBuffers(*setup.program))
    {
        return false;
    }
    if (!updatePerDrawConstants(setup.backendContext.immediateContext, drawCommand,
                                setup.useSceneBuffers))
    {
        return false;
    }
    bindGeometry(setup.backendContext.immediateContext, *setup.meshBuffers);
    setup.backendContext.immediateContext->SetPipelineState(setup.program->pipelineState);
    setup.backendContext.immediateContext->CommitShaderResources(
        setup.program->shaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedIndirectAttribs drawAttrs{};
    drawAttrs.IndexType = Diligent::VT_UINT32;
    drawAttrs.pAttribsBuffer = indirectArgsBuffer;
    drawAttrs.DrawArgsOffset = argsOffsetBytes;
    drawAttrs.Flags = Diligent::DRAW_FLAG_VERIFY_ALL;
    drawAttrs.AttribsBufferStateTransitionMode = Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
    setup.backendContext.immediateContext->DrawIndexedIndirect(drawAttrs);
    return true;
}

std::size_t ForwardOpaquePass::cachedProgramCount() const noexcept
{
    return mProgramRegistry != nullptr ? mProgramRegistry->cachedProgramCount() : 0u;
}

bool ForwardOpaquePass::ensureConstantBuffers(Diligent::IRenderDevice* renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    if (mForwardPerFrameBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name           = "CRESSimNeo.ForwardOpaquePass.GriphicsForwardPerFrame";
        constantBufferDesc.Size           = sizeof(ForwardPerFrameConstants);
        constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
        renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mForwardPerFrameBuffer);
        if (mForwardPerFrameBuffer == nullptr)
        {
            return false;
        }
    }

    if (mPerObjectBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name           = "CRESSimNeo.ForwardOpaquePass.GraphicsPerObject";
        constantBufferDesc.Size           = sizeof(PerObjectConstants);
        constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
        renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mPerObjectBuffer);
        if (mPerObjectBuffer == nullptr)
        {
            return false;
        }
    }

    if (mForwardPerMaterialBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name      = "CRESSimNeo.ForwardOpaquePass.GraphicsForwardPerMaterial";
        constantBufferDesc.Size      = sizeof(ForwardPerMaterialConstants);
        constantBufferDesc.Usage     = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
        renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mForwardPerMaterialBuffer);
        if (mForwardPerMaterialBuffer == nullptr)
        {
            return false;
        }
    }

    return true;
}

bool ForwardOpaquePass::bindProgramConstants(MaterialProgramRegistry::ProgramResources& program)
{
    if (program.pipelineState == nullptr || mForwardPerFrameBuffer == nullptr ||
        mPerObjectBuffer == nullptr || mForwardPerMaterialBuffer == nullptr)
    {
        return false;
    }

    auto bindIfPresent = [&](Diligent::SHADER_TYPE shaderType, const char* varName,
                             Diligent::IBuffer* buffer, bool required)
    {
        Diligent::IShaderResourceVariable* variable =
            program.pipelineState->GetStaticVariableByName(shaderType, varName);
        if (variable == nullptr)
        {
            return !required;
        }
        variable->Set(buffer);
        return true;
    };

    if (!bindIfPresent(Diligent::SHADER_TYPE_VERTEX, "GriphicsForwardPerFrame",
                       mForwardPerFrameBuffer, true) ||
        !bindIfPresent(Diligent::SHADER_TYPE_PIXEL, "GriphicsForwardPerFrame",
                       mForwardPerFrameBuffer, true) ||
        !bindIfPresent(Diligent::SHADER_TYPE_VERTEX, "GraphicsPerObject", mPerObjectBuffer, true) ||
        !bindIfPresent(Diligent::SHADER_TYPE_PIXEL, "GraphicsPerObject", mPerObjectBuffer, false) ||
        !bindIfPresent(Diligent::SHADER_TYPE_VERTEX, "GraphicsForwardPerMaterial",
                       mForwardPerMaterialBuffer, false) ||
        !bindIfPresent(Diligent::SHADER_TYPE_PIXEL, "GraphicsForwardPerMaterial",
                       mForwardPerMaterialBuffer, true))
    {
        return false;
    }

    return true;
}

bool ForwardOpaquePass::hasAnyShadowMap() const
{
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < mShadowMapCount; ++cascadeIdx)
    {
        Diligent::ITexture* depthTexture = nullptr;
        if (mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(
                mShadowMapTargets[cascadeIdx], depthTexture) &&
            depthTexture != nullptr)
        {
            return true;
        }
    }
    return false;
}

} // namespace cressim::neo::graphics::detail
