#include "graphics/passes/shadow_pass.h"

#include "common/logger.h"

#include <cstring>
#include <string>

namespace cressim::neo::graphics::detail
{

ShadowPass::ShadowPass(gpu::GpuDevice &device, RenderResourceManager &resourceManager)
    : mDevice(device), mResourceManager(resourceManager), mShaderLibrary(""),
      mMeshGpuCache("CRESSimNeo.ShadowPass")
{
}

bool ShadowPass::initialize()
{
    mShaderLibrary = gpu::ShaderLibrary(mDevice.shaderSourceDirectory());
    mInitialized   = true;
    return true;
}

void ShadowPass::setGpuSceneView(const gpu::GpuEntitySceneView &sceneView) noexcept
{
    mSceneView = sceneView;
}

void ShadowPass::setVisiblePairBuffer(Diligent::IBuffer *buffer) noexcept
{
    mVisiblePairBuffer = buffer;
}

void ShadowPass::setBatchCameraBuffer(Diligent::IBuffer *buffer) noexcept
{
    mBatchCameraBuffer = buffer;
}

void ShadowPass::setLocalShadowViewBuffer(Diligent::IBuffer *buffer) noexcept
{
    mLocalShadowViewBuffer = buffer;
}

bool ShadowPass::prepareDraw(const gpu::GpuRenderTargetBinding &targetBinding,
                             const ForwardDrawCommand &drawCommand, DrawSetup &outSetup)
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
    if (!backendContext.hasActiveRenderTarget ||
        !(backendContext.activeRenderTargetBinding == targetBinding))
    {
        return false;
    }
    if (backendContext.renderDevice == nullptr || backendContext.immediateContext == nullptr ||
        !backendContext.activeRenderTargetHasDepth)
    {
        return false;
    }

    if (drawCommand.meshId == common::kInvalidResourceId)
    {
        return false;
    }
    if (drawCommand.indexCount < 3)
    {
        return false;
    }

    MeshGpuCache::CachedBuffers *meshBuffers =
        mMeshGpuCache.getOrCreate(mResourceManager, drawCommand, backendContext.renderDevice);
    if (meshBuffers == nullptr || meshBuffers->vertexBuffer == nullptr ||
        meshBuffers->indexBuffer == nullptr || meshBuffers->indexCount == 0)
    {
        return false;
    }

    if (mPipelineState == nullptr || mShaderResourceBinding == nullptr)
    {
        if (!createPipeline(backendContext.renderDevice))
        {
            return false;
        }
    }

    if (!ensureConstantBuffers(backendContext.renderDevice))
    {
        return false;
    }

    if (mSceneView.poses.positionsBuffer == nullptr ||
        mSceneView.poses.orientationsBuffer == nullptr ||
        mSceneView.poses.scalesBuffer == nullptr ||
        mSceneView.renderableMetadataBuffer == nullptr ||
        mSceneView.renderableShadowCascadeMasksBuffer == nullptr ||
        mSceneView.preparedCamerasBuffer == nullptr)
    {
        return false;
    }

    outSetup.backendContext = backendContext;
    outSetup.meshBuffers    = meshBuffers;
    return true;
}

bool ShadowPass::bindSceneBuffers() const
{
    if (mShaderResourceBinding == nullptr)
    {
        return false;
    }
    struct VariableBinding
    {
        const char *name;
        Diligent::IBuffer *buffer;
    };
    const VariableBinding bindings[] = {
        {"g_EntityPositions", mSceneView.poses.positionsBuffer},
        {"g_EntityOrientations", mSceneView.poses.orientationsBuffer},
        {"g_EntityScales", mSceneView.poses.scalesBuffer},
        {"g_RenderableMetadata", mSceneView.renderableMetadataBuffer},
        {"g_RenderableShadowCascadeMasks", mSceneView.renderableShadowCascadeMasksBuffer},
        {"g_PreparedCameras", mSceneView.preparedCamerasBuffer},
    };
    for (const VariableBinding &binding : bindings)
    {
        Diligent::IShaderResourceVariable *variable =
            mShaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, binding.name);
        if (variable == nullptr || binding.buffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *srv =
            binding.buffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (srv == nullptr)
        {
            return false;
        }
        variable->Set(srv);
    }

    if (mBatchCameraBuffer == nullptr && mLocalShadowViewBuffer == nullptr)
    {
        return false;
    }
    Diligent::IShaderResourceVariable *batchCamerasVar =
        mShaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, "g_BatchCameras");
    Diligent::IBufferView *batchCamerasSrv =
        mBatchCameraBuffer != nullptr
            ? mBatchCameraBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE)
            : nullptr;
    if (batchCamerasVar == nullptr || batchCamerasSrv == nullptr)
    {
        if (batchCamerasVar != nullptr && mSceneView.preparedCamerasBuffer != nullptr)
        {
            batchCamerasVar->Set(mSceneView.preparedCamerasBuffer->GetDefaultView(
                Diligent::BUFFER_VIEW_SHADER_RESOURCE));
        }
    }
    else
    {
        batchCamerasVar->Set(batchCamerasSrv);
    }

    Diligent::IShaderResourceVariable *localShadowViewsVar =
        mShaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                  "g_LocalShadowViews");
    if (localShadowViewsVar != nullptr)
    {
        Diligent::IBuffer *buffer = mLocalShadowViewBuffer != nullptr
                                        ? mLocalShadowViewBuffer
                                        : mSceneView.preparedCamerasBuffer;
        if (buffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *srv = buffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (srv == nullptr)
        {
            return false;
        }
        localShadowViewsVar->Set(srv);
    }

    Diligent::IShaderResourceVariable *visiblePairsVar =
        mShaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, "g_VisiblePairs");
    if (visiblePairsVar != nullptr)
    {
        Diligent::IBuffer *visiblePairBuffer = mVisiblePairBuffer != nullptr
                                                   ? mVisiblePairBuffer
                                                   : mSceneView.renderableShadowCascadeMasksBuffer;
        if (visiblePairBuffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *visiblePairsSrv =
            visiblePairBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (visiblePairsSrv == nullptr)
        {
            return false;
        }
        visiblePairsVar->Set(visiblePairsSrv);
    }
    return true;
}

bool ShadowPass::updatePerDrawConstants(Diligent::IDeviceContext *immediateContext,
                                        const ForwardDrawCommand &drawCommand,
                                        std::uint32_t shadowMatrixIndex,
                                        std::uint32_t shadowPassMode)
{
    PerObjectConstants objectConstants{};
    objectConstants.instanceIndex     = drawCommand.instanceIndex;
    objectConstants.drawListOffset    = drawCommand.drawListOffset;
    objectConstants.useDrawListBuffer = drawCommand.useDrawListBuffer;

    ShadowPerPassConstants shadowPassConstants{};
    shadowPassConstants.shadowPassParams[0] = shadowMatrixIndex;
    shadowPassConstants.shadowPassParams[1] = shadowPassMode;

    void *mappedConstants = nullptr;
    immediateContext->MapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                                mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &objectConstants, sizeof(objectConstants));
    immediateContext->UnmapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE);

    mappedConstants = nullptr;
    immediateContext->MapBuffer(mShadowPerPassBuffer, Diligent::MAP_WRITE,
                                Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &shadowPassConstants, sizeof(shadowPassConstants));
    immediateContext->UnmapBuffer(mShadowPerPassBuffer, Diligent::MAP_WRITE);
    return true;
}

void ShadowPass::bindGeometry(Diligent::IDeviceContext *immediateContext,
                              const MeshGpuCache::CachedBuffers &meshBuffers) const
{
    const Diligent::Uint64 vertexOffset = 0;
    Diligent::IBuffer *vertexBuffers[]  = {meshBuffers.vertexBuffer};
    immediateContext->SetVertexBuffers(0, 1, vertexBuffers, &vertexOffset,
                                       Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                       Diligent::SET_VERTEX_BUFFERS_FLAG_RESET);
    immediateContext->SetIndexBuffer(meshBuffers.indexBuffer, 0,
                                     Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

bool ShadowPass::drawIndirect(const gpu::GpuRenderTargetBinding &targetBinding,
                              const ForwardDrawCommand &drawCommand,
                              std::uint32_t shadowMatrixIndex, std::uint32_t shadowPassMode,
                              Diligent::IBuffer *indirectArgsBuffer,
                              Diligent::Uint64 argsOffsetBytes)
{
    DrawSetup setup{};
    if (!prepareDraw(targetBinding, drawCommand, setup) || indirectArgsBuffer == nullptr)
    {
        return false;
    }
    if (!bindSceneBuffers())
    {
        return false;
    }
    if (!updatePerDrawConstants(setup.backendContext.immediateContext, drawCommand,
                                shadowMatrixIndex, shadowPassMode))
    {
        return false;
    }
    bindGeometry(setup.backendContext.immediateContext, *setup.meshBuffers);
    setup.backendContext.immediateContext->SetPipelineState(mPipelineState);
    setup.backendContext.immediateContext->CommitShaderResources(
        mShaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedIndirectAttribs drawAttrs{};
    drawAttrs.IndexType      = Diligent::VT_UINT32;
    drawAttrs.pAttribsBuffer = indirectArgsBuffer;
    drawAttrs.DrawArgsOffset = argsOffsetBytes;
    drawAttrs.Flags          = Diligent::DRAW_FLAG_VERIFY_ALL;
    drawAttrs.AttribsBufferStateTransitionMode =
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
    setup.backendContext.immediateContext->DrawIndexedIndirect(drawAttrs);
    return true;
}

bool ShadowPass::createPipeline(Diligent::IRenderDevice *renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    constexpr const char *kShadowVsRelativePath = "graphics/shadow_depth.vs.hlsl";

    std::string shadowVsPath;
    if (!mShaderLibrary.resolveShaderPath(kShadowVsRelativePath, shadowVsPath))
    {
        CRESSIM_LOG_ERROR("ShadowPass shader path resolution failed for relative path '",
                          kShadowVsRelativePath, "'.");
        return false;
    }

    Diligent::IShaderSourceInputStreamFactory *streamFactory = mShaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        CRESSIM_LOG_ERROR("ShadowPass could not acquire shader source stream factory.");
        return false;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.Desc.ShaderType                 = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name                       = "CRESSimNeo.ShadowPass.VS";
    shaderCreateInfo.FilePath                        = kShadowVsRelativePath;
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;
    Diligent::ShaderMacro shadowMacros[]             = {
        {"MANUAL_LAYER_EXPORT", "1"},
    };
    shaderCreateInfo.Macros = Diligent::ShaderMacroArray{shadowMacros, 1};

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    renderDevice->CreateShader(shaderCreateInfo, &vertexShader);
    if (vertexShader == nullptr)
    {
        CRESSIM_LOG_ERROR("ShadowPass failed to compile shader: '", shadowVsPath, "'.");
        return false;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                       = "CRESSimNeo.ShadowPass.PSO";
    psoCreateInfo.PSODesc.PipelineType               = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 0;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = Diligent::TEX_FORMAT_D32_FLOAT;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode              = Diligent::CULL_MODE_BACK;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.FrontCounterClockwise = Diligent::True;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.DepthBias             = 8;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.SlopeScaledDepthBias  = 2.0f;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable         = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable    = Diligent::True;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    constexpr Diligent::ShaderResourceVariableDesc kVars[] = {
        {Diligent::SHADER_TYPE_VERTEX, "g_EntityPositions",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_EntityOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_EntityScales",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_RenderableMetadata",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_RenderableVisibilityFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_RenderableShadowCascadeMasks",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_VisiblePairs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_BatchCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_LocalShadowViews",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    psoCreateInfo.PSODesc.ResourceLayout.Variables = kVars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(std::size(kVars));

    constexpr Diligent::LayoutElement kLayoutElements[] = {
        Diligent::LayoutElement{0, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{1, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{2, 0, 2, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{3, 0, 4, Diligent::VT_FLOAT32, Diligent::False}};
    psoCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = kLayoutElements;
    psoCreateInfo.GraphicsPipeline.InputLayout.NumElements    = 4;
    psoCreateInfo.pVS                                         = vertexShader;

    renderDevice->CreateGraphicsPipelineState(psoCreateInfo, &mPipelineState);
    if (mPipelineState == nullptr)
    {
        CRESSIM_LOG_ERROR("ShadowPass failed to create PSO.");
        return false;
    }

    if (!ensureConstantBuffers(renderDevice))
    {
        CRESSIM_LOG_ERROR("ShadowPass failed to allocate constant buffers.");
        return false;
    }

    Diligent::IShaderResourceVariable *perObjectVar =
        mPipelineState->GetStaticVariableByName(Diligent::SHADER_TYPE_VERTEX, "GraphicsPerObject");
    Diligent::IShaderResourceVariable *shadowPerPassVar = mPipelineState->GetStaticVariableByName(
        Diligent::SHADER_TYPE_VERTEX, "GraphicsShadowPerPass");
    if (perObjectVar == nullptr || shadowPerPassVar == nullptr)
    {
        CRESSIM_LOG_ERROR(
            "ShadowPass static constant bindings are missing from shader reflection.");
        return false;
    }

    perObjectVar->Set(mPerObjectBuffer);
    shadowPerPassVar->Set(mShadowPerPassBuffer);

    mPipelineState->CreateShaderResourceBinding(&mShaderResourceBinding, true);
    return mShaderResourceBinding != nullptr;
}

bool ShadowPass::ensureConstantBuffers(Diligent::IRenderDevice *renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    if (mPerObjectBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name           = "CRESSimNeo.ShadowPass.GraphicsPerObject";
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

    if (mShadowPerPassBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name           = "CRESSimNeo.ShadowPass.GraphicsShadowPerPass";
        constantBufferDesc.Size           = sizeof(ShadowPerPassConstants);
        constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
        renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mShadowPerPassBuffer);
        if (mShadowPerPassBuffer == nullptr)
        {
            return false;
        }
    }

    return true;
}

} // namespace cressim::neo::graphics::detail
