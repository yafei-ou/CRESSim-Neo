#include "graphics/renderer/passes/shadow_pass.h"

#include "DiligentEngine/DiligentCore/Primitives/interface/Errors.hpp"

#include <cstring>
#include <string>

namespace cressim::neo::graphics::detail
{

ShadowPass::ShadowPass(gpu::GpuDevice& device)
    : mDevice(device), mShaderLibrary(""), mMeshGpuCache("CRESSimNeo.ShadowPass")
{
}

bool ShadowPass::initialize()
{
    mShaderLibrary = gpu::ShaderLibrary(mDevice.shaderSourceDirectory());
    mInitialized   = true;
    return true;
}

void ShadowPass::setGpuSceneView(const gpu::GpuEntitySceneView& sceneView) noexcept
{
    mSceneView = sceneView;
}

bool ShadowPass::draw(gpu::GpuRenderTargetHandle target, const ForwardDrawCommand& drawCommand,
                      const Diligent::float4x4& lightViewProjectionMatrix,
                      std::uint32_t cascadeIndex)
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
    if (backendContext.renderDevice == nullptr || backendContext.immediateContext == nullptr ||
        !backendContext.activeRenderTargetHasDepth)
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

    const bool useSceneBuffers = drawCommand.instanceIndex != 0xffffffffu &&
                                 mSceneView.poses.positionsBuffer != nullptr &&
                                 mSceneView.poses.orientationsBuffer != nullptr &&
                                 mSceneView.poses.scalesBuffer != nullptr &&
                                 mSceneView.renderableMetadataBuffer != nullptr &&
                                 mSceneView.renderableVisibilityFlagsBuffer != nullptr &&
                                 mSceneView.renderableShadowCascadeMasksBuffer != nullptr;
    if (useSceneBuffers)
    {
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
            // g_RenderableVisibilityFlags is not used and optimized away
            // {"g_RenderableVisibilityFlags", mSceneView.renderableVisibilityFlagsBuffer},
            {"g_RenderableShadowCascadeMasks", mSceneView.renderableShadowCascadeMasksBuffer},
        };
        for (const VariableBinding& binding : bindings)
        {
            Diligent::IShaderResourceVariable* variable =
                mShaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
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
    }

    PerObjectConstants objectConstants{};
    objectConstants.modelMatrix  = drawCommand.modelMatrix.Transpose();
    objectConstants.normalMatrix = drawCommand.normalMatrix.Transpose();
    objectConstants.instanceIndex = drawCommand.instanceIndex;
    objectConstants.useSceneBuffers = useSceneBuffers ? 1u : 0u;

    ShadowPerPassConstants shadowPassConstants{};
    shadowPassConstants.lightViewProjectionMatrix = lightViewProjectionMatrix.Transpose();
    shadowPassConstants.shadowPassParams[0] = cascadeIndex;

    void* mappedConstants = nullptr;
    backendContext.immediateContext->MapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE,
                                               Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &objectConstants, sizeof(objectConstants));
    backendContext.immediateContext->UnmapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE);

    mappedConstants = nullptr;
    backendContext.immediateContext->MapBuffer(mShadowPerPassBuffer, Diligent::MAP_WRITE,
                                               Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &shadowPassConstants, sizeof(shadowPassConstants));
    backendContext.immediateContext->UnmapBuffer(mShadowPerPassBuffer, Diligent::MAP_WRITE);

    const Diligent::Uint64 vertexOffset = 0;
    Diligent::IBuffer* vertexBuffers[]  = {meshBuffers->vertexBuffer};
    backendContext.immediateContext->SetVertexBuffers(
        0, 1, vertexBuffers, &vertexOffset, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        Diligent::SET_VERTEX_BUFFERS_FLAG_RESET);
    backendContext.immediateContext->SetIndexBuffer(
        meshBuffers->indexBuffer, 0, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    backendContext.immediateContext->SetPipelineState(mPipelineState);
    backendContext.immediateContext->CommitShaderResources(
        mShaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedAttribs drawAttrs{};
    drawAttrs.IndexType  = Diligent::VT_UINT32;
    drawAttrs.NumIndices = meshBuffers->indexCount;
    drawAttrs.Flags      = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.immediateContext->DrawIndexed(drawAttrs);
    return true;
}

bool ShadowPass::createPipeline(Diligent::IRenderDevice* renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    constexpr const char* kShadowVsRelativePath = "graphics/shadow_depth.vs.hlsl";

    std::string shadowVsPath;
    if (!mShaderLibrary.resolveShaderPath(kShadowVsRelativePath, shadowVsPath))
    {
        LOG_ERROR_MESSAGE("ShadowPass shader path resolution failed for relative path '",
                          kShadowVsRelativePath, "'.");
        return false;
    }

    Diligent::IShaderSourceInputStreamFactory* streamFactory = mShaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        LOG_ERROR_MESSAGE("ShadowPass could not acquire shader source stream factory.");
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

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    renderDevice->CreateShader(shaderCreateInfo, &vertexShader);
    if (vertexShader == nullptr)
    {
        LOG_ERROR_MESSAGE("ShadowPass failed to compile shader: '", shadowVsPath, "'.");
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
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_VERTEX, "g_EntityOrientations",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_VERTEX, "g_EntityScales",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_VERTEX, "g_RenderableMetadata",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_VERTEX, "g_RenderableVisibilityFlags",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
        {Diligent::SHADER_TYPE_VERTEX, "g_RenderableShadowCascadeMasks",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    };
    psoCreateInfo.PSODesc.ResourceLayout.Variables = kVars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(std::size(kVars));

    constexpr Diligent::LayoutElement kLayoutElements[] = {
        Diligent::LayoutElement{0, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{1, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{2, 0, 2, Diligent::VT_FLOAT32, Diligent::False}};
    psoCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = kLayoutElements;
    psoCreateInfo.GraphicsPipeline.InputLayout.NumElements    = 3;
    psoCreateInfo.pVS                                         = vertexShader;

    renderDevice->CreateGraphicsPipelineState(psoCreateInfo, &mPipelineState);
    if (mPipelineState == nullptr)
    {
        LOG_ERROR_MESSAGE("ShadowPass failed to create PSO.");
        return false;
    }

    if (!ensureConstantBuffers(renderDevice))
    {
        LOG_ERROR_MESSAGE("ShadowPass failed to allocate constant buffers.");
        return false;
    }

    Diligent::IShaderResourceVariable* perObjectVar =
        mPipelineState->GetStaticVariableByName(Diligent::SHADER_TYPE_VERTEX, "GraphicsPerObject");
    Diligent::IShaderResourceVariable* shadowPerPassVar = mPipelineState->GetStaticVariableByName(
        Diligent::SHADER_TYPE_VERTEX, "GraphicsShadowPerPass");
    if (perObjectVar == nullptr || shadowPerPassVar == nullptr)
    {
        LOG_ERROR_MESSAGE(
            "ShadowPass static constant bindings are missing from shader reflection.");
        return false;
    }

    perObjectVar->Set(mPerObjectBuffer);
    shadowPerPassVar->Set(mShadowPerPassBuffer);

    mPipelineState->CreateShaderResourceBinding(&mShaderResourceBinding, true);
    return mShaderResourceBinding != nullptr;
}

bool ShadowPass::ensureConstantBuffers(Diligent::IRenderDevice* renderDevice)
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
