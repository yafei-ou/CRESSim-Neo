#include "graphics/passes/debug_strand_frame_pass.h"

#include "gpu/shader_source_provider.h"
#include "graphics/passes/render_pass_types.h"
#include "physics/physics_gpu_scene_view.h"

#include <cstring>

namespace cressim::neo::graphics::detail
{

DebugStrandFramePass::DebugStrandFramePass(gpu::GpuDevice &device) : mDevice(device) {}

std::size_t DebugStrandFramePass::PipelineKeyHasher::operator()(
    const PipelineKey &key) const noexcept
{
    const std::size_t colorHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat));
    const std::size_t depthHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.depthFormat));
    return colorHash ^ (depthHash << 1u);
}

bool DebugStrandFramePass::initialize()
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

bool DebugStrandFramePass::ensureConstants(Diligent::IRenderDevice *renderDevice)
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
    constantBufferDesc.Name           = "CRESSimNeo.DebugStrandFramePass.Constants";
    constantBufferDesc.Size           = sizeof(DrawConstants);
    constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
    constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
    constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
    renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mConstantsBuffer);
    return mConstantsBuffer != nullptr;
}

Diligent::IPipelineState *DebugStrandFramePass::getOrCreatePipeline(
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
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.DebugStrandFramePass.VS";
    shaderCreateInfo.FilePath        = "graphics/debug_strand_frames.vs.hlsl";
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
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.DebugStrandFramePass.PS";
    shaderCreateInfo.FilePath        = "graphics/debug_strand_frames.ps.hlsl";
    if (!mDevice.createShader(shaderCreateInfo, &pixelShader))
    {
        pixelShader = nullptr;
    }
    if (pixelShader == nullptr)
    {
        return nullptr;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                       = "CRESSimNeo.DebugStrandFramePass.PSO";
    psoCreateInfo.PSODesc.PipelineType               = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]     = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = key.depthFormat;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode           = Diligent::CULL_MODE_NONE;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable      = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = Diligent::False;
    psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0].BlendEnable = Diligent::True;
    auto &blend          = psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0];
    blend.SrcBlend       = Diligent::BLEND_FACTOR_SRC_ALPHA;
    blend.DestBlend      = Diligent::BLEND_FACTOR_INV_SRC_ALPHA;
    blend.BlendOp        = Diligent::BLEND_OPERATION_ADD;
    blend.SrcBlendAlpha  = Diligent::BLEND_FACTOR_ONE;
    blend.DestBlendAlpha = Diligent::BLEND_FACTOR_INV_SRC_ALPHA;
    blend.BlendOpAlpha   = Diligent::BLEND_OPERATION_ADD;
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
        {Diligent::SHADER_TYPE_VERTEX, "g_StrandSegments",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_VERTEX, "g_StrandSegmentStates",
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
            Diligent::SHADER_TYPE_VERTEX, "GraphicsDebugStrandFrames"))
    {
        vsConstants->Set(mConstantsBuffer);
    }
    if (Diligent::IShaderResourceVariable *psConstants = pipeline->GetStaticVariableByName(
            Diligent::SHADER_TYPE_PIXEL, "GraphicsDebugStrandFrames"))
    {
        psConstants->Set(mConstantsBuffer);
    }

    auto insertResult = mPipelines.emplace(key, pipeline);
    return insertResult.first->second;
}

Diligent::IShaderResourceBinding *DebugStrandFramePass::getOrCreateBinding(
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

bool DebugStrandFramePass::draw(const gpu::GpuRenderTargetBinding &targetBinding,
                                const gpu::GpuRenderTargetDesc &targetDesc,
                                const GpuEntitySceneView &gpuScene,
                                const cressim::neo::physics::PhysicsGpuSceneView &physicsScene,
                                const ResolvedCameraView &camera, std::uint32_t targetLayer,
                                const RenderFrameOptions::DebugStrandFrameOptions &options)
{
    if (!mInitialized || !options.enabled)
    {
        return false;
    }

    const auto &soft = physicsScene.soft;
    if (soft.strandSegmentCount == 0u)
    {
        return true;
    }
    if (soft.particles.positionsInvMassBuffer == nullptr ||
        soft.particles.environmentIndicesBuffer == nullptr ||
        soft.strandSegmentsBuffer == nullptr || soft.strandSegmentStatesBuffer == nullptr ||
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
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticlePositionsInvMass",
                       soft.particles.positionsInvMassBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_ParticleEnvironmentIndices",
                       soft.particles.environmentIndicesBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_StrandSegments",
                       soft.strandSegmentsBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_StrandSegmentStates",
                       soft.strandSegmentStatesBuffer))
    {
        return false;
    }

    DrawConstants constants{};
    constants.cameraIndex = camera.globalCameraIndex;
    constants.targetLayer = targetLayer;
    constants.envIndex    = camera.envIndex;
    constants.axisLength  = options.axisLength;
    constants.thickness   = options.thickness;
    constants.opacity     = options.opacity;

    void *mapped = nullptr;
    backendContext.graphicsContext->MapBuffer(mConstantsBuffer, Diligent::MAP_WRITE,
                                              Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &constants, sizeof(constants));
    backendContext.graphicsContext->UnmapBuffer(mConstantsBuffer, Diligent::MAP_WRITE);

    backendContext.graphicsContext->SetPipelineState(pipeline);
    backendContext.graphicsContext->CommitShaderResources(
        binding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawAttribs drawAttribs{};
    drawAttribs.NumVertices = soft.strandSegmentCount * 18u;
    drawAttribs.Flags       = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.graphicsContext->Draw(drawAttribs);
    return true;
}

} // namespace cressim::neo::graphics::detail
