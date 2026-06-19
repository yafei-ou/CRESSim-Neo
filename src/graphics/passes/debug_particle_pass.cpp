#include "graphics/passes/debug_particle_pass.h"

#include "gpu/shader_source_provider.h"
#include "graphics/passes/render_pass_types.h"
#include "physics/physics_gpu_scene_view.h"

#include <cstring>

namespace cressim::neo::graphics::detail
{

namespace
{

constexpr std::uint32_t kUseParticleRadiiFlag         = 1u << 0u;
constexpr std::uint32_t kHighlightStaticParticlesFlag = 1u << 1u;
constexpr std::uint32_t kShowCutEdgesFlag             = 1u << 2u;
constexpr std::uint32_t kShowStrainFlag               = 1u << 3u;
constexpr std::uint32_t kShowDamageFlag               = 1u << 4u;
constexpr std::uint32_t kShapePrimitiveCenters        = 0u;
constexpr std::uint32_t kShapePrimitiveAxes           = 1u;
constexpr std::uint32_t kShapePrimitiveMembers        = 2u;

} // namespace

DebugParticlePass::DebugParticlePass(gpu::GpuDevice &device) : mDevice(device) {}

std::size_t DebugParticlePass::PipelineKeyHasher::operator()(const PipelineKey &key) const noexcept
{
    const std::size_t colorHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat));
    const std::size_t depthHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.depthFormat));
    return colorHash ^ (depthHash << 1u);
}

bool DebugParticlePass::initialize()
{
    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        return false;
    }

    mInitialized = ensureConstants(backendContext.renderDevice);
    return mInitialized;
}

bool DebugParticlePass::ensureConstants(Diligent::IRenderDevice *renderDevice)
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
    constantBufferDesc.Name           = "CRESSimNeo.DebugParticlePass.Constants";
    constantBufferDesc.Size           = sizeof(DrawConstants);
    constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
    constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
    constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
    renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mConstantsBuffer);
    if (mConstantsBuffer == nullptr)
    {
        return false;
    }

    if (mFallbackRadiusBuffer == nullptr)
    {
        const float fallbackRadius = 0.15f;
        Diligent::BufferDesc radiusBufferDesc{};
        radiusBufferDesc.Name              = "CRESSimNeo.DebugParticlePass.FallbackRadius";
        radiusBufferDesc.Size              = sizeof(float);
        radiusBufferDesc.Usage             = Diligent::USAGE_IMMUTABLE;
        radiusBufferDesc.BindFlags         = Diligent::BIND_SHADER_RESOURCE;
        radiusBufferDesc.Mode              = Diligent::BUFFER_MODE_STRUCTURED;
        radiusBufferDesc.ElementByteStride = sizeof(float);
        const Diligent::BufferData radiusData{&fallbackRadius, sizeof(float)};
        renderDevice->CreateBuffer(radiusBufferDesc, &radiusData, &mFallbackRadiusBuffer);
    }

    return mFallbackRadiusBuffer != nullptr;
}

Diligent::IPipelineState *DebugParticlePass::getOrCreatePipeline(
    Diligent::IRenderDevice *renderDevice, const PipelineKey &key)
{
    auto it = mPipelines.find(key);
    if (it != mPipelines.end())
    {
        return it->second;
    }
    if (renderDevice == nullptr || key.colorFormat == Diligent::TEX_FORMAT_UNKNOWN ||
        key.depthFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return nullptr;
    }

    gpu::ShaderSourceProvider shaderSourceProvider(mDevice.shaderSourceConfig());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderSourceProvider.streamFactory();
    if (streamFactory == nullptr)
    {
        return nullptr;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::ShaderMacro layerMacros[] = {
        {"MANUAL_LAYER_EXPORT", "1"},
    };
    shaderCreateInfo.Macros =
        Diligent::ShaderMacroArray{layerMacros, static_cast<Diligent::Uint32>(1u)};

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.DebugParticlePass.VS";
    shaderCreateInfo.FilePath        = "graphics/debug_particles.vs.hlsl";
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
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.DebugParticlePass.PS";
    shaderCreateInfo.FilePath        = "graphics/debug_particles.ps.hlsl";
    if (!mDevice.createShader(shaderCreateInfo, &pixelShader))
    {
        pixelShader = nullptr;
    }
    if (pixelShader == nullptr)
    {
        return nullptr;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                       = "CRESSimNeo.DebugParticlePass.PSO";
    psoCreateInfo.PSODesc.PipelineType               = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]     = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = key.depthFormat;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode           = Diligent::CULL_MODE_NONE;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable      = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = Diligent::True;
    psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0].BlendEnable = Diligent::False;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    constexpr Diligent::ShaderResourceVariableDesc kVars[] = {
        {Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_CameraInputs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticlePositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticleRadii",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticleEnvironmentIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticleShapeMembershipRanges",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ShapeCorrectionMagnitudes",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_CameraInputs",
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

    if (Diligent::IShaderResourceVariable *vsConstants = pipeline->GetStaticVariableByName(
            Diligent::SHADER_TYPE_VERTEX, "GraphicsDebugParticles"))
    {
        vsConstants->Set(mConstantsBuffer);
    }
    if (Diligent::IShaderResourceVariable *psConstants = pipeline->GetStaticVariableByName(
            Diligent::SHADER_TYPE_PIXEL, "GraphicsDebugParticles"))
    {
        psConstants->Set(mConstantsBuffer);
    }

    auto insertResult = mPipelines.emplace(key, pipeline);
    return insertResult.first->second;
}

Diligent::IPipelineState *DebugParticlePass::getOrCreateEdgePipeline(
    Diligent::IRenderDevice *renderDevice, const PipelineKey &key)
{
    auto it = mEdgePipelines.find(key);
    if (it != mEdgePipelines.end())
    {
        return it->second;
    }
    if (renderDevice == nullptr || key.colorFormat == Diligent::TEX_FORMAT_UNKNOWN ||
        key.depthFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return nullptr;
    }

    gpu::ShaderSourceProvider shaderSourceProvider(mDevice.shaderSourceConfig());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderSourceProvider.streamFactory();
    if (streamFactory == nullptr)
    {
        return nullptr;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::ShaderMacro layerMacros[] = {
        {"MANUAL_LAYER_EXPORT", "1"},
    };
    shaderCreateInfo.Macros =
        Diligent::ShaderMacroArray{layerMacros, static_cast<Diligent::Uint32>(1u)};

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.DebugParticleEdgesPass.VS";
    shaderCreateInfo.FilePath        = "graphics/debug_particle_edges.vs.hlsl";
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
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.DebugParticleEdgesPass.PS";
    shaderCreateInfo.FilePath        = "graphics/debug_particle_edges.ps.hlsl";
    if (!mDevice.createShader(shaderCreateInfo, &pixelShader))
    {
        pixelShader = nullptr;
    }
    if (pixelShader == nullptr)
    {
        return nullptr;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                       = "CRESSimNeo.DebugParticleEdgesPass.PSO";
    psoCreateInfo.PSODesc.PipelineType               = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]     = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = key.depthFormat;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_LINE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode           = Diligent::CULL_MODE_NONE;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable      = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = Diligent::False;
    psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0].BlendEnable = Diligent::False;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    constexpr Diligent::ShaderResourceVariableDesc kVars[] = {
        {Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_CameraInputs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticlePositionsInvMass",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_ParticleEnvironmentIndices",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_SoftEdges",
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

    if (Diligent::IShaderResourceVariable *vsConstants = pipeline->GetStaticVariableByName(
            Diligent::SHADER_TYPE_VERTEX, "GraphicsDebugParticles"))
    {
        vsConstants->Set(mConstantsBuffer);
    }
    if (Diligent::IShaderResourceVariable *psConstants = pipeline->GetStaticVariableByName(
            Diligent::SHADER_TYPE_PIXEL, "GraphicsDebugParticles"))
    {
        psConstants->Set(mConstantsBuffer);
    }

    auto insertResult = mEdgePipelines.emplace(key, pipeline);
    return insertResult.first->second;
}

Diligent::IShaderResourceBinding *DebugParticlePass::getOrCreateBinding(
    Diligent::IPipelineState *pipeline)
{
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    auto it = mBindings.find(pipeline);
    if (it != mBindings.end())
    {
        return it->second;
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    pipeline->CreateShaderResourceBinding(&srb, true);
    if (srb == nullptr)
    {
        return nullptr;
    }

    auto insertResult = mBindings.emplace(pipeline, std::move(srb));
    return insertResult.first->second;
}

bool DebugParticlePass::drawConstraintEdges(
    const gpu::GpuRenderTargetDesc &targetDesc, const GpuEntitySceneView &gpuScene,
    const cressim::neo::physics::PhysicsGpuSceneView &physicsScene,
    Diligent::IRenderDevice *renderDevice, Diligent::IDeviceContext *graphicsContext,
    const DrawConstants &constants)
{
    if (physicsScene.soft.edgeCount == 0u || physicsScene.soft.edgesBuffer == nullptr)
    {
        return true;
    }

    const auto &particles = physicsScene.soft.particles;
    if (particles.positionsInvMassBuffer == nullptr ||
        particles.environmentIndicesBuffer == nullptr ||
        gpuScene.preparedCamerasBuffer == nullptr || gpuScene.cameraInputsBuffer == nullptr)
    {
        return false;
    }

    Diligent::IPipelineState *pipeline = getOrCreateEdgePipeline(
        renderDevice, PipelineKey{targetDesc.colorFormat, targetDesc.depthFormat});
    if (pipeline == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceBinding *binding = getOrCreateBinding(pipeline);
    if (binding == nullptr)
    {
        return false;
    }

    const auto setBufferView = [&](const char *name, Diligent::IBuffer *buffer) -> bool
    {
        Diligent::IShaderResourceVariable *variable =
            binding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX, name);
        if (variable == nullptr || buffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *srv = buffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (srv == nullptr)
        {
            return false;
        }
        variable->Set(srv);
        return true;
    };

    if (!setBufferView("g_PreparedCameras", gpuScene.preparedCamerasBuffer) ||
        !setBufferView("g_CameraInputs", gpuScene.cameraInputsBuffer) ||
        !setBufferView("g_ParticlePositionsInvMass", particles.positionsInvMassBuffer) ||
        !setBufferView("g_ParticleEnvironmentIndices", particles.environmentIndicesBuffer) ||
        !setBufferView("g_SoftEdges", physicsScene.soft.edgesBuffer))
    {
        return false;
    }

    graphicsContext->SetPipelineState(pipeline);
    graphicsContext->CommitShaderResources(binding,
                                           Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawAttribs drawAttrs{};
    drawAttrs.NumVertices = physicsScene.soft.edgeCount * 2u;
    drawAttrs.Flags       = Diligent::DRAW_FLAG_VERIFY_ALL;
    graphicsContext->Draw(drawAttrs);
    (void)constants;
    return true;
}

bool DebugParticlePass::draw(const gpu::GpuRenderTargetBinding &targetBinding,
                             const gpu::GpuRenderTargetDesc &targetDesc,
                             const GpuEntitySceneView &gpuScene,
                             const cressim::neo::physics::PhysicsGpuSceneView &physicsScene,
                             const ResolvedCameraView &camera, std::uint32_t targetLayer,
                             const RenderFrameOptions::DebugParticleOptions &options)
{
    if (!mInitialized || !options.enabled)
    {
        return false;
    }

    const auto &particles = physicsScene.soft.particles;
    if (particles.count == 0u)
    {
        return true;
    }
    if (particles.positionsInvMassBuffer == nullptr ||
        particles.environmentIndicesBuffer == nullptr ||
        gpuScene.preparedCamerasBuffer == nullptr || gpuScene.cameraInputsBuffer == nullptr)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr || backendContext.graphicsContext == nullptr ||
        !backendContext.hasActiveRenderTarget ||
        !(backendContext.activeRenderTargetBinding == targetBinding))
    {
        return false;
    }

    Diligent::IPipelineState *pipeline = getOrCreatePipeline(
        backendContext.renderDevice, PipelineKey{targetDesc.colorFormat, targetDesc.depthFormat});
    if (pipeline == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceBinding *binding = getOrCreateBinding(pipeline);
    if (binding == nullptr)
    {
        return false;
    }

    const auto setBufferView = [&](Diligent::SHADER_TYPE shaderType, const char *name,
                                   Diligent::IBuffer *buffer) -> bool
    {
        Diligent::IShaderResourceVariable *variable = binding->GetVariableByName(shaderType, name);
        if (variable == nullptr || buffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *srv = buffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (srv == nullptr)
        {
            return false;
        }
        variable->Set(srv);
        return true;
    };

    if (!setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_PreparedCameras",
                       gpuScene.preparedCamerasBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_CameraInputs",
                       gpuScene.cameraInputsBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_PIXEL, "g_CameraInputs",
                       gpuScene.cameraInputsBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticlePositionsInvMass",
                       particles.positionsInvMassBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticleEnvironmentIndices",
                       particles.environmentIndicesBuffer))
    {
        return false;
    }
    if (!setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticleShapeMembershipRanges",
                       physicsScene.soft.particleShapeMembershipRangesBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ShapeCorrectionMagnitudes",
                       physicsScene.soft.shapeCorrectionMagnitudesBuffer))
    {
        return false;
    }

    const bool useParticleRadii = options.useParticleRadii && particles.radiiBuffer != nullptr;
    Diligent::IBuffer *radiiBuffer =
        useParticleRadii ? particles.radiiBuffer : mFallbackRadiusBuffer;
    if (!setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticleRadii", radiiBuffer))
    {
        return false;
    }

    DrawConstants constants{};
    constants.color               = options.color;
    constants.staticColor         = options.staticColor;
    constants.edgeColor           = options.edgeColor;
    constants.edgeHighStrainColor = options.edgeHighStrainColor;
    constants.edgeDamagedColor    = options.edgeDamagedColor;
    constants.edgeDisabledColor   = options.edgeDisabledColor;
    constants.cameraIndex         = camera.globalCameraIndex;
    constants.targetLayer         = targetLayer;
    constants.envIndex            = camera.envIndex;
    constants.flags               = useParticleRadii ? kUseParticleRadiiFlag : 0u;
    constants.flags |= options.highlightStaticParticles ? kHighlightStaticParticlesFlag : 0u;
    constants.flags |= options.showCutEdges ? kShowCutEdgesFlag : 0u;
    constants.flags |= options.showStrain ? kShowStrainFlag : 0u;
    constants.flags |= options.showDamage ? kShowDamageFlag : 0u;
    constants.shapeModes = options.shapeMatchingModes;
    constants.maxMembershipCount =
        std::max<std::uint32_t>(options.shapeMaxMembershipCount, 1u);
    constants.fallbackRadius         = options.fallbackRadius;
    constants.highStrainThreshold    = options.highStrainThreshold;
    constants.damageDisplayThreshold = options.damageDisplayThreshold;
    constants.shapeCenterRadius      = options.shapeCenterRadius;
    constants.shapeAxisLength        = options.shapeAxisLength;
    constants.shapeCorrectionScale   = options.shapeCorrectionScale;

    void *mapped = nullptr;
    backendContext.graphicsContext->MapBuffer(mConstantsBuffer, Diligent::MAP_WRITE,
                                              Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &constants, sizeof(constants));
    backendContext.graphicsContext->UnmapBuffer(mConstantsBuffer, Diligent::MAP_WRITE);

    if (options.drawConstraintEdges &&
        !drawConstraintEdges(targetDesc, gpuScene, physicsScene, backendContext.renderDevice,
                             backendContext.graphicsContext, constants))
    {
        return false;
    }

    backendContext.graphicsContext->SetPipelineState(pipeline);
    backendContext.graphicsContext->CommitShaderResources(
        binding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawAttribs drawAttrs{};
    drawAttrs.NumVertices = particles.count * 6u;
    drawAttrs.Flags       = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.graphicsContext->Draw(drawAttrs);
    return true;
}

} // namespace cressim::neo::graphics::detail
