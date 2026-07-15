#include "graphics/passes/camera_segmentation_pass.h"

#include "physics/physics_gpu_scene_view.h"

#include <cstring>
#include <string>

namespace cressim::neo::graphics::detail
{

CameraSegmentationPass::CameraSegmentationPass(gpu::GpuDevice &device,
                                               RenderResourceManager &resourceManager)
    : mDevice(device), mResourceManager(resourceManager), mShaderSourceProvider(""),
      mMeshGpuCache("CRESSimNeo.CameraSegmentationPass"),
      mMaterialHelper(resourceManager, "CRESSimNeo.CameraSegmentationPass.Material")
{
}

bool CameraSegmentationPass::initialize()
{
    mShaderSourceProvider = gpu::ShaderSourceProvider(mDevice.shaderSourceDirectory());
    mInitialized   = true;
    return true;
}

void CameraSegmentationPass::setGpuSceneView(const GpuEntitySceneView &sceneView) noexcept
{
    const bool sceneBindingsChanged = mSceneView.bindingGeneration != 0u &&
                                      mSceneView.bindingGeneration != sceneView.bindingGeneration;
    mSceneView                      = sceneView;
    if (sceneBindingsChanged)
    {
        mShaderBindings.clear();
    }
}

void CameraSegmentationPass::setPhysicsSceneView(
    const physics::PhysicsGpuSceneView *physicsScene) noexcept
{
    mPhysicsScene = physicsScene;
}

void CameraSegmentationPass::setVisiblePairBuffer(Diligent::IBuffer *buffer) noexcept
{
    mVisiblePairBuffer = buffer;
}

void CameraSegmentationPass::setBatchCameraBuffer(Diligent::IBuffer *buffer) noexcept
{
    mBatchCameraBuffer = buffer;
}

std::size_t CameraSegmentationPass::PipelineKeyHasher::operator()(
    const PipelineKey &key) const noexcept
{
    const std::size_t programHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.programFamily));
    const std::size_t featureHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.materialFeatureFlags));
    const std::size_t colorHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat));
    const std::size_t depthHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.depthFormat));
    return programHash ^ (featureHash << 1u) ^ (colorHash << 2u) ^ (depthHash << 3u);
}

bool CameraSegmentationPass::ensureConstantBuffers(Diligent::IRenderDevice *renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    if (mPerObjectBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name           = "CRESSimNeo.CameraSegmentationPass.GraphicsPerObject";
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

    return true;
}

Diligent::IPipelineState *CameraSegmentationPass::getOrCreatePipeline(
    Diligent::IRenderDevice *renderDevice, const PipelineKey &key)
{
    auto it = mPipelines.find(key);
    if (it != mPipelines.end())
    {
        return it->second;
    }
    if (renderDevice == nullptr || key.colorFormat != Diligent::TEX_FORMAT_R32_UINT ||
        key.depthFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return nullptr;
    }

    Diligent::IShaderSourceInputStreamFactory *streamFactory = mShaderSourceProvider.streamFactory();
    if (streamFactory == nullptr)
    {
        return nullptr;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.ShaderCompiler                  = Diligent::SHADER_COMPILER_GLSLANG;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::ShaderMacro shaderMacros[] = {
        {"MANUAL_LAYER_EXPORT", "1"},
        {"CRESSIM_CAMERA_DEPTH_PASS", "1"},
        {"CRESSIM_CAMERA_SEGMENTATION_PASS", "1"},
        {"CRESSIM_FEATURE_ALPHA_TEST",
         hasFlag(key.materialFeatureFlags, MaterialFeatureFlags::AlphaTest) ? "1" : ""},
        {key.programFamily == MaterialProgramFamily::SoftBodyLit
             ? "CRESSIM_PROGRAM_FAMILY_SOFT_BODY"
             : (key.programFamily == MaterialProgramFamily::CurveLit
                    ? "CRESSIM_PROGRAM_FAMILY_CURVE"
                    : ""),
         key.programFamily != MaterialProgramFamily::StandardLit ? "1" : ""},
    };

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.FilePath        = "graphics/shadow_depth.vs.hlsl";
    shaderCreateInfo.Desc.Name       = key.programFamily == MaterialProgramFamily::SoftBodyLit
                                           ? "CRESSimNeo.CameraSegmentationPass.SoftBody.VS"
                                           : (key.programFamily == MaterialProgramFamily::CurveLit
                                                  ? "CRESSimNeo.CameraSegmentationPass.Curve.VS"
                                                  : "CRESSimNeo.CameraSegmentationPass.VS");
    shaderCreateInfo.Macros          = Diligent::ShaderMacroArray{
        shaderMacros,
        static_cast<Diligent::Uint32>(
            hasFlag(key.materialFeatureFlags, MaterialFeatureFlags::AlphaTest)
                ? (key.programFamily == MaterialProgramFamily::StandardLit ? 4u : 5u)
                : (key.programFamily == MaterialProgramFamily::StandardLit ? 3u : 4u))};
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
    shaderCreateInfo.FilePath        = "graphics/camera_segmentation.ps.hlsl";
    shaderCreateInfo.Desc.Name       = key.programFamily == MaterialProgramFamily::SoftBodyLit
                                           ? "CRESSimNeo.CameraSegmentationPass.SoftBody.PS"
                                           : (key.programFamily == MaterialProgramFamily::CurveLit
                                                  ? "CRESSimNeo.CameraSegmentationPass.Curve.PS"
                                                  : "CRESSimNeo.CameraSegmentationPass.PS");
    shaderCreateInfo.Macros          = Diligent::ShaderMacroArray{
        shaderMacros,
        static_cast<Diligent::Uint32>(
            hasFlag(key.materialFeatureFlags, MaterialFeatureFlags::AlphaTest)
                ? (key.programFamily == MaterialProgramFamily::StandardLit ? 4u : 5u)
                : (key.programFamily == MaterialProgramFamily::StandardLit ? 3u : 4u))};
    if (!mDevice.createShader(shaderCreateInfo, &pixelShader))
    {
        pixelShader = nullptr;
    }
    if (pixelShader == nullptr)
    {
        return nullptr;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name         = key.programFamily == MaterialProgramFamily::SoftBodyLit
                                             ? "CRESSimNeo.CameraSegmentationPass.SoftBody.PSO"
                                             : (key.programFamily == MaterialProgramFamily::CurveLit
                                                    ? "CRESSimNeo.CameraSegmentationPass.Curve.PSO"
                                                    : "CRESSimNeo.CameraSegmentationPass.PSO");
    psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]     = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = key.depthFormat;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode =
        hasFlag(key.materialFeatureFlags, MaterialFeatureFlags::DoubleSided)
            ? Diligent::CULL_MODE_NONE
            : Diligent::CULL_MODE_BACK;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.FrontCounterClockwise = Diligent::True;
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
        {Diligent::SHADER_TYPE_VERTEX, "g_VisiblePairs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_BatchCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_BaseColorTexture",
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
        {Diligent::SHADER_TYPE_VERTEX, "g_VisiblePairs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_BatchCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_SoftBodyRenderPositions",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_BaseColorTexture",
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
        {Diligent::SHADER_TYPE_VERTEX, "g_VisiblePairs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_BatchCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_CurveRenderPositions",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_BaseColorTexture",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    psoCreateInfo.PSODesc.ResourceLayout.Variables =
        key.programFamily == MaterialProgramFamily::SoftBodyLit
            ? kSoftBodyVars
            : (key.programFamily == MaterialProgramFamily::CurveLit ? kCurveVars : kStandardVars);
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables = static_cast<Diligent::Uint32>(
        key.programFamily == MaterialProgramFamily::SoftBodyLit
            ? std::size(kSoftBodyVars)
            : (key.programFamily == MaterialProgramFamily::CurveLit ? std::size(kCurveVars)
                                                                    : std::size(kStandardVars)));

    constexpr Diligent::LayoutElement kLayoutElements[] = {
        Diligent::LayoutElement{0, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{1, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{2, 0, 2, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{3, 0, 4, Diligent::VT_FLOAT32, Diligent::False},
    };
    psoCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = kLayoutElements;
    psoCreateInfo.GraphicsPipeline.InputLayout.NumElements    = 4;
    psoCreateInfo.pVS                                         = vertexShader;
    psoCreateInfo.pPS                                         = pixelShader;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipeline;
    if (!mDevice.createGraphicsPipelineState(psoCreateInfo, &pipeline))
    {
        pipeline = nullptr;
    }
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    if (!ensureConstantBuffers(renderDevice))
    {
        return nullptr;
    }

    Diligent::IShaderResourceVariable *perObjectVar =
        pipeline->GetStaticVariableByName(Diligent::SHADER_TYPE_VERTEX, "GraphicsPerObject");
    if (perObjectVar == nullptr)
    {
        return nullptr;
    }
    perObjectVar->Set(mPerObjectBuffer);
    if (!mMaterialHelper.initialize(renderDevice) || !mMaterialHelper.bindStaticResources(pipeline))
    {
        return nullptr;
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> shaderBinding;
    pipeline->CreateShaderResourceBinding(&shaderBinding, true);
    if (shaderBinding == nullptr)
    {
        return nullptr;
    }

    Diligent::IPipelineState *pipelinePtr = pipeline;
    mShaderBindings.emplace(pipelinePtr, std::move(shaderBinding));
    auto insertResult = mPipelines.emplace(key, std::move(pipeline));
    return insertResult.first->second;
}

Diligent::IShaderResourceBinding *CameraSegmentationPass::getOrCreateShaderBinding(
    Diligent::IPipelineState *pipeline) noexcept
{
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    const auto it = mShaderBindings.find(pipeline);
    return it != mShaderBindings.end() ? it->second : nullptr;
}

bool CameraSegmentationPass::prepareDraw(const gpu::GpuRenderTargetBinding &targetBinding,
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
        !backendContext.activeRenderTargetHasDepth ||
        backendContext.activeRenderTargetColorFormat != Diligent::TEX_FORMAT_R32_UINT ||
        drawCommand.meshId == common::kInvalidResourceId || drawCommand.indexCount < 3u)
    {
        return false;
    }

    MeshGpuCache::CachedBuffers *meshBuffers =
        mMeshGpuCache.getOrCreate(mResourceManager, drawCommand, backendContext.renderDevice);
    if (meshBuffers == nullptr || meshBuffers->vertexBuffer == nullptr ||
        meshBuffers->indexBuffer == nullptr || meshBuffers->indexCount == 0u)
    {
        return false;
    }

    outSetup.backendContext = backendContext;
    outSetup.meshBuffers    = meshBuffers;
    return true;
}

bool CameraSegmentationPass::bindSceneBuffers(Diligent::IShaderResourceBinding *shaderBinding,
                                              MaterialProgramFamily programFamily) const
{
    if (shaderBinding == nullptr || mSceneView.poses.positionsBuffer == nullptr ||
        mSceneView.poses.orientationsBuffer == nullptr ||
        mSceneView.poses.scalesBuffer == nullptr ||
        mSceneView.renderableMetadataBuffer == nullptr ||
        mSceneView.renderableVisibilityFlagsBuffer == nullptr ||
        mSceneView.preparedCamerasBuffer == nullptr || mVisiblePairBuffer == nullptr ||
        mBatchCameraBuffer == nullptr)
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
        {"g_RenderableVisibilityFlags", mSceneView.renderableVisibilityFlagsBuffer},
        {"g_VisiblePairs", mVisiblePairBuffer},
        {"g_BatchCameras", mBatchCameraBuffer},
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

    if (programFamily == MaterialProgramFamily::SoftBodyLit)
    {
        if (mPhysicsScene == nullptr || mPhysicsScene->soft.renderPositionsBuffer == nullptr)
        {
            return false;
        }
        Diligent::IShaderResourceVariable *positionVar = shaderBinding->GetVariableByName(
            Diligent::SHADER_TYPE_VERTEX, "g_SoftBodyRenderPositions");
        if (positionVar == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *positionSrv =
            mPhysicsScene->soft.renderPositionsBuffer->GetDefaultView(
                Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (positionSrv == nullptr)
        {
            return false;
        }
        positionVar->Set(positionSrv);
    }
    else if (programFamily == MaterialProgramFamily::CurveLit)
    {
        if (mPhysicsScene == nullptr || mPhysicsScene->curve.positionsBuffer == nullptr)
        {
            return false;
        }
        Diligent::IShaderResourceVariable *positionVar = shaderBinding->GetVariableByName(
            Diligent::SHADER_TYPE_VERTEX, "g_CurveRenderPositions");
        if (positionVar == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *positionSrv = mPhysicsScene->curve.positionsBuffer->GetDefaultView(
            Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (positionSrv == nullptr)
        {
            return false;
        }
        positionVar->Set(positionSrv);
    }

    return true;
}

bool CameraSegmentationPass::updatePerDrawConstants(Diligent::IDeviceContext *graphicsContext,
                                                    const ForwardDrawCommand &drawCommand)
{
    PerObjectConstants perObject{};
    perObject.instanceIndex     = drawCommand.instanceIndex;
    perObject.drawListOffset    = drawCommand.drawListOffset;
    perObject.useDrawListBuffer = drawCommand.useDrawListBuffer;
    void *mapped                = nullptr;
    graphicsContext->MapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                               mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &perObject, sizeof(perObject));
    graphicsContext->UnmapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE);
    return true;
}

void CameraSegmentationPass::bindGeometry(Diligent::IDeviceContext *graphicsContext,
                                          const MeshGpuCache::CachedBuffers &meshBuffers) const
{
    const Diligent::Uint64 vertexOffset = 0u;
    Diligent::IBuffer *vertexBuffers[]  = {meshBuffers.vertexBuffer};
    graphicsContext->SetVertexBuffers(0, 1, vertexBuffers, &vertexOffset,
                                      Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                      Diligent::SET_VERTEX_BUFFERS_FLAG_RESET);
    graphicsContext->SetIndexBuffer(meshBuffers.indexBuffer, 0,
                                    Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

bool CameraSegmentationPass::drawIndirect(const gpu::GpuRenderTargetBinding &targetBinding,
                                          const ForwardDrawCommand &drawCommand,
                                          Diligent::IBuffer *indirectArgsBuffer,
                                          Diligent::Uint64 argsOffsetBytes)
{
    DrawSetup setup{};
    if (!prepareDraw(targetBinding, drawCommand, setup) || indirectArgsBuffer == nullptr)
    {
        return false;
    }

    Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_D32_FLOAT;
    Diligent::ITexture *depthTexture     = nullptr;
    if (!mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(targetBinding.target,
                                                                     depthTexture) ||
        depthTexture == nullptr)
    {
        return false;
    }
    depthFormat = depthTexture->GetDesc().Format;

    Diligent::IPipelineState *pipeline = getOrCreatePipeline(
        setup.backendContext.renderDevice,
        PipelineKey{drawCommand.programFamily, drawCommand.materialFeatureFlags,
                    setup.backendContext.activeRenderTargetColorFormat, depthFormat});
    if (pipeline == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceBinding *shaderBinding = getOrCreateShaderBinding(pipeline);
    if (shaderBinding == nullptr)
    {
        return false;
    }
    if (!bindSceneBuffers(shaderBinding, drawCommand.programFamily))
    {
        return false;
    }
    if (!mMaterialHelper.bindMaterialResources(setup.backendContext.renderDevice,
                                               setup.backendContext.graphicsContext, shaderBinding,
                                               drawCommand.materialId))
    {
        return false;
    }
    if (!updatePerDrawConstants(setup.backendContext.graphicsContext, drawCommand))
    {
        return false;
    }
    if (!mMaterialHelper.updateMaterialConstants(setup.backendContext.graphicsContext,
                                                 drawCommand.materialId))
    {
        return false;
    }

    bindGeometry(setup.backendContext.graphicsContext, *setup.meshBuffers);
    setup.backendContext.graphicsContext->SetPipelineState(pipeline);
    setup.backendContext.graphicsContext->CommitShaderResources(
        shaderBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedIndirectAttribs drawAttrs{};
    drawAttrs.IndexType      = Diligent::VT_UINT32;
    drawAttrs.pAttribsBuffer = indirectArgsBuffer;
    drawAttrs.DrawArgsOffset = argsOffsetBytes;
    drawAttrs.Flags          = Diligent::DRAW_FLAG_VERIFY_ALL;
    drawAttrs.AttribsBufferStateTransitionMode =
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
    setup.backendContext.graphicsContext->DrawIndexedIndirect(drawAttrs);
    return true;
}

} // namespace cressim::neo::graphics::detail
