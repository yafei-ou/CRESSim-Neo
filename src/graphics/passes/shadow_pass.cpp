#include "graphics/passes/shadow_pass.h"

#include "common/logger.h"
#include "physics/physics_gpu_scene_view.h"

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

void ShadowPass::setGpuSceneView(const GpuEntitySceneView &sceneView) noexcept
{
    const bool sceneBindingsChanged = mSceneView.bindingGeneration != 0u &&
                                      mSceneView.bindingGeneration != sceneView.bindingGeneration;
    mSceneView = sceneView;
    if (sceneBindingsChanged)
    {
        mShaderResourceBinding         = nullptr;
        mSoftBodyShaderResourceBinding = nullptr;
        mCurveShaderResourceBinding    = nullptr;
    }
}

void ShadowPass::setPhysicsSceneView(const physics::PhysicsGpuSceneView *physicsScene) noexcept
{
    mPhysicsScene = physicsScene;
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

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext))
    {
        return false;
    }
    if (!backendContext.hasActiveRenderTarget ||
        !(backendContext.activeRenderTargetBinding == targetBinding))
    {
        return false;
    }
    if (backendContext.renderDevice == nullptr || backendContext.graphicsContext == nullptr ||
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

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> &pipelineState =
        drawCommand.programFamily == MaterialProgramFamily::SoftBodyLit
            ? mSoftBodyPipelineState
            : (drawCommand.programFamily == MaterialProgramFamily::CurveLit
                   ? mCurvePipelineState
                   : mPipelineState);
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> &shaderBinding =
        drawCommand.programFamily == MaterialProgramFamily::SoftBodyLit
            ? mSoftBodyShaderResourceBinding
            : (drawCommand.programFamily == MaterialProgramFamily::CurveLit
                   ? mCurveShaderResourceBinding
                   : mShaderResourceBinding);
    if (pipelineState == nullptr || shaderBinding == nullptr)
    {
        if (!createPipeline(backendContext.renderDevice, drawCommand.programFamily))
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

bool ShadowPass::bindSceneBuffers(MaterialProgramFamily programFamily) const
{
    Diligent::IShaderResourceBinding *shaderBinding =
        programFamily == MaterialProgramFamily::SoftBodyLit
            ? mSoftBodyShaderResourceBinding
            : (programFamily == MaterialProgramFamily::CurveLit ? mCurveShaderResourceBinding
                                                                : mShaderResourceBinding);
    if (shaderBinding == nullptr)
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
            shaderBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, binding.name);
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
        shaderBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, "g_BatchCameras");
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
        shaderBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, "g_LocalShadowViews");
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
        shaderBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, "g_VisiblePairs");
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

    if (programFamily == MaterialProgramFamily::SoftBodyLit)
    {
        if (mPhysicsScene == nullptr ||
            mPhysicsScene->soft.particles.positionsInvMassBuffer == nullptr ||
            mSceneView.softBodyVertexBindingBuffer == nullptr)
        {
            return false;
        }
        Diligent::IShaderResourceVariable *softParticleVar =
            shaderBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, "g_ParticlePositions");
        Diligent::IShaderResourceVariable *bindingVar = shaderBinding->GetVariableByName(
            Diligent::SHADER_TYPE_VERTEX, "g_SoftBodyVertexBindings");
        if (softParticleVar == nullptr || bindingVar == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *softParticleSrv =
            mPhysicsScene->soft.particles.positionsInvMassBuffer->GetDefaultView(
                Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        Diligent::IBufferView *bindingSrv = mSceneView.softBodyVertexBindingBuffer->GetDefaultView(
            Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (softParticleSrv == nullptr || bindingSrv == nullptr)
        {
            return false;
        }
        softParticleVar->Set(softParticleSrv);
        bindingVar->Set(bindingSrv);
    }
    else if (programFamily == MaterialProgramFamily::CurveLit)
    {
        if (mPhysicsScene == nullptr || mPhysicsScene->curve.positionsBuffer == nullptr)
        {
            return false;
        }
        Diligent::IShaderResourceVariable *positionVar =
            shaderBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, "g_CurveRenderPositions");
        if (positionVar == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *positionSrv =
            mPhysicsScene->curve.positionsBuffer->GetDefaultView(
                Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (positionSrv == nullptr)
        {
            return false;
        }
        positionVar->Set(positionSrv);
    }
    return true;
}

bool ShadowPass::updatePerDrawConstants(Diligent::IDeviceContext *graphicsContext,
                                        const ForwardDrawCommand &drawCommand,
                                        std::uint32_t shadowMatrixIndex,
                                        std::uint32_t shadowPassMode)
{
    PerObjectConstants objectConstants{};
    objectConstants.instanceIndex     = drawCommand.instanceIndex;
    objectConstants.useDrawListBuffer = drawCommand.useDrawListBuffer;
    objectConstants.drawListOffset    = shadowPassMode == 0u ? drawCommand.drawListOffset : 0u;

    ShadowPerPassConstants shadowPassConstants{};
    shadowPassConstants.shadowPassParams[0] = shadowMatrixIndex;
    shadowPassConstants.shadowPassParams[1] = shadowPassMode;

    void *mappedConstants = nullptr;
    graphicsContext->MapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                               mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &objectConstants, sizeof(objectConstants));
    graphicsContext->UnmapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE);

    mappedConstants = nullptr;
    graphicsContext->MapBuffer(mShadowPerPassBuffer, Diligent::MAP_WRITE,
                               Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &shadowPassConstants, sizeof(shadowPassConstants));
    graphicsContext->UnmapBuffer(mShadowPerPassBuffer, Diligent::MAP_WRITE);
    return true;
}

void ShadowPass::bindGeometry(Diligent::IDeviceContext *graphicsContext,
                              const MeshGpuCache::CachedBuffers &meshBuffers) const
{
    const Diligent::Uint64 vertexOffset = 0;
    Diligent::IBuffer *vertexBuffers[]  = {meshBuffers.vertexBuffer};
    graphicsContext->SetVertexBuffers(0, 1, vertexBuffers, &vertexOffset,
                                      Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                      Diligent::SET_VERTEX_BUFFERS_FLAG_RESET);
    graphicsContext->SetIndexBuffer(meshBuffers.indexBuffer, 0,
                                    Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

bool ShadowPass::drawIndirect(const gpu::GpuRenderTargetBinding &targetBinding,
                              const ForwardDrawCommand &drawCommand,
                              std::uint32_t shadowMatrixIndex, std::uint32_t shadowPassMode,
                              Diligent::IBuffer *indirectArgsBuffer,
                              Diligent::Uint64 argsOffsetBytes, std::uint32_t drawCount,
                              std::uint32_t drawArgsStride)
{
    DrawSetup setup{};
    if (!prepareDraw(targetBinding, drawCommand, setup) || indirectArgsBuffer == nullptr)
    {
        return false;
    }
    if (!bindSceneBuffers(drawCommand.programFamily))
    {
        return false;
    }
    if (!updatePerDrawConstants(setup.backendContext.graphicsContext, drawCommand,
                                shadowMatrixIndex, shadowPassMode))
    {
        return false;
    }
    bindGeometry(setup.backendContext.graphicsContext, *setup.meshBuffers);
    Diligent::IPipelineState *pipeline =
        drawCommand.programFamily == MaterialProgramFamily::SoftBodyLit
            ? mSoftBodyPipelineState
            : (drawCommand.programFamily == MaterialProgramFamily::CurveLit
                   ? mCurvePipelineState
                   : mPipelineState);
    Diligent::IShaderResourceBinding *shaderBinding =
        drawCommand.programFamily == MaterialProgramFamily::SoftBodyLit
            ? mSoftBodyShaderResourceBinding
            : (drawCommand.programFamily == MaterialProgramFamily::CurveLit
                   ? mCurveShaderResourceBinding
                   : mShaderResourceBinding);
    setup.backendContext.graphicsContext->SetPipelineState(pipeline);
    setup.backendContext.graphicsContext->CommitShaderResources(
        shaderBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedIndirectAttribs drawAttrs{};
    drawAttrs.IndexType      = Diligent::VT_UINT32;
    drawAttrs.pAttribsBuffer = indirectArgsBuffer;
    drawAttrs.DrawArgsOffset = argsOffsetBytes;
    drawAttrs.DrawCount      = drawCount;
    drawAttrs.DrawArgsStride = drawArgsStride;
    drawAttrs.Flags          = Diligent::DRAW_FLAG_VERIFY_ALL;
    drawAttrs.AttribsBufferStateTransitionMode =
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
    setup.backendContext.graphicsContext->DrawIndexedIndirect(drawAttrs);
    return true;
}

bool ShadowPass::createPipeline(Diligent::IRenderDevice *renderDevice,
                                MaterialProgramFamily programFamily)
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
    shaderCreateInfo.Desc.Name =
        programFamily == MaterialProgramFamily::SoftBodyLit
            ? "CRESSimNeo.ShadowPass.SoftBody.VS"
            : (programFamily == MaterialProgramFamily::CurveLit
                   ? "CRESSimNeo.ShadowPass.Curve.VS"
                   : "CRESSimNeo.ShadowPass.VS");
    shaderCreateInfo.FilePath  = kShadowVsRelativePath;
    shaderCreateInfo.pShaderSourceStreamFactory = streamFactory;
    Diligent::ShaderMacro shadowMacros[]        = {
        {"MANUAL_LAYER_EXPORT", "1"},
        {programFamily == MaterialProgramFamily::SoftBodyLit
             ? "CRESSIM_PROGRAM_FAMILY_SOFT_BODY"
             : (programFamily == MaterialProgramFamily::CurveLit
                    ? "CRESSIM_PROGRAM_FAMILY_CURVE"
                    : ""),
         programFamily != MaterialProgramFamily::StandardLit ? "1" : ""},
    };
    shaderCreateInfo.Macros = Diligent::ShaderMacroArray{
        shadowMacros,
        static_cast<Diligent::Uint32>(programFamily == MaterialProgramFamily::StandardLit ? 1 : 2)};

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    if (!mDevice.createShader(shaderCreateInfo, &vertexShader))
    {
        vertexShader = nullptr;
    }
    if (vertexShader == nullptr)
    {
        CRESSIM_LOG_ERROR("ShadowPass failed to compile shader: '", shadowVsPath, "'.");
        return false;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name =
        programFamily == MaterialProgramFamily::SoftBodyLit
            ? "CRESSimNeo.ShadowPass.SoftBody.PSO"
            : (programFamily == MaterialProgramFamily::CurveLit
                   ? "CRESSimNeo.ShadowPass.Curve.PSO"
                   : "CRESSimNeo.ShadowPass.PSO");
    psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 0;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = Diligent::TEX_FORMAT_D32_FLOAT;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode              = Diligent::CULL_MODE_BACK;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.FrontCounterClockwise = Diligent::True;
    // These rasterizer bias values remain renderer-global. Per-light sample bias is applied in
    // the forward lighting shaders after shadow-map lookup.
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.DepthBias             = 8;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.SlopeScaledDepthBias  = 2.0f;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable         = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable    = Diligent::True;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;
    constexpr Diligent::ShaderResourceVariableDesc kStandardVars[] = {
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
    constexpr Diligent::ShaderResourceVariableDesc kSoftBodyVars[] = {
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
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticlePositions",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_SoftBodyVertexBindings",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    constexpr Diligent::ShaderResourceVariableDesc kCurveVars[] = {
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
        {Diligent::SHADER_TYPE_VERTEX, "g_CurveRenderPositions",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    psoCreateInfo.PSODesc.ResourceLayout.Variables =
        programFamily == MaterialProgramFamily::SoftBodyLit
            ? kSoftBodyVars
            : (programFamily == MaterialProgramFamily::CurveLit ? kCurveVars : kStandardVars);
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables = static_cast<Diligent::Uint32>(
        programFamily == MaterialProgramFamily::SoftBodyLit
            ? std::size(kSoftBodyVars)
            : (programFamily == MaterialProgramFamily::CurveLit ? std::size(kCurveVars)
                                                                : std::size(kStandardVars)));

    constexpr Diligent::LayoutElement kLayoutElements[] = {
        Diligent::LayoutElement{0, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{1, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{2, 0, 2, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{3, 0, 4, Diligent::VT_FLOAT32, Diligent::False}};
    psoCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = kLayoutElements;
    psoCreateInfo.GraphicsPipeline.InputLayout.NumElements    = 4;
    psoCreateInfo.pVS                                         = vertexShader;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> &pipelineState =
        programFamily == MaterialProgramFamily::SoftBodyLit
            ? mSoftBodyPipelineState
            : (programFamily == MaterialProgramFamily::CurveLit ? mCurvePipelineState
                                                                : mPipelineState);
    if (!mDevice.createGraphicsPipelineState(psoCreateInfo, &pipelineState))
    {
        pipelineState = nullptr;
    }
    if (pipelineState == nullptr)
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
        pipelineState->GetStaticVariableByName(Diligent::SHADER_TYPE_VERTEX, "GraphicsPerObject");
    Diligent::IShaderResourceVariable *shadowPerPassVar = pipelineState->GetStaticVariableByName(
        Diligent::SHADER_TYPE_VERTEX, "GraphicsShadowPerPass");
    if (perObjectVar == nullptr || shadowPerPassVar == nullptr)
    {
        CRESSIM_LOG_ERROR(
            "ShadowPass static constant bindings are missing from shader reflection.");
        return false;
    }

    perObjectVar->Set(mPerObjectBuffer);
    shadowPerPassVar->Set(mShadowPerPassBuffer);

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> &shaderBinding =
        programFamily == MaterialProgramFamily::SoftBodyLit
            ? mSoftBodyShaderResourceBinding
            : (programFamily == MaterialProgramFamily::CurveLit ? mCurveShaderResourceBinding
                                                                : mShaderResourceBinding);
    pipelineState->CreateShaderResourceBinding(&shaderBinding, true);
    return shaderBinding != nullptr;
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
