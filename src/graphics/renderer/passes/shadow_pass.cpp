#include "graphics/renderer/passes/shadow_pass.h"

#include "graphics/device/graphics_device_impl.h"
#include "graphics/device/shaders/shader_fallback_sources.h"

#include <cstring>
#include <string>

namespace cressim::neo::graphics::detail
{

ShadowPass::ShadowPass(GraphicsDeviceImpl& device) :
    mDevice(device),
    mShaderSourceProvider("", true)
{
}

bool ShadowPass::initialize()
{
    mShaderSourceProvider = ShaderSourceProvider(mDevice.shaderSourceDirectory(), mDevice.allowShaderFallback());
    mInitialized = true;
    return true;
}

bool ShadowPass::draw(RenderTargetHandle target, const ForwardDrawCommand& drawCommand)
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
    if (backendContext.renderDevice == nullptr || backendContext.immediateContext == nullptr || !backendContext.activeRenderTargetHasDepth)
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

    if (mPipelineState == nullptr || mShaderResourceBinding == nullptr || mConstantBuffer == nullptr)
    {
        if (!createPipeline(backendContext.renderDevice))
        {
            return false;
        }
    }

    DrawConstants constants{};
    std::memcpy(constants.modelMatrix, drawCommand.modelMatrix, sizeof(constants.modelMatrix));
    std::memcpy(constants.lightViewProjectionMatrix, drawCommand.lightViewProjectionMatrix, sizeof(constants.lightViewProjectionMatrix));

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

    backendContext.immediateContext->SetPipelineState(mPipelineState);
    backendContext.immediateContext->CommitShaderResources(mShaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedAttribs drawAttrs{};
    drawAttrs.IndexType = Diligent::VT_UINT32;
    drawAttrs.NumIndices = meshBuffers->indexCount;
    drawAttrs.Flags = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.immediateContext->DrawIndexed(drawAttrs);
    return true;
}

ShadowPass::CachedMeshGpuData* ShadowPass::getOrCreateMeshBuffers(
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
    vertexBufferDesc.Name = "CRESSimNeo.ShadowPass.VertexBuffer";
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
    indexBufferDesc.Name = "CRESSimNeo.ShadowPass.IndexBuffer";
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

bool ShadowPass::createPipeline(Diligent::IRenderDevice* renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    std::string shadowVsSource;
    if (!mShaderSourceProvider.loadSource("shadow_depth.vs.hlsl", shaders::shadowDepthVertex(), shadowVsSource))
    {
        return false;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.CompileFlags = Diligent::SHADER_COMPILE_FLAG_PACK_MATRIX_ROW_MAJOR;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint = "main";
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name = "CRESSimNeo.ShadowPass.VS";
    shaderCreateInfo.Source = shadowVsSource.c_str();

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    renderDevice->CreateShader(shaderCreateInfo, &vertexShader);
    if (vertexShader == nullptr)
    {
        return false;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name = "CRESSimNeo.ShadowPass.PSO";
    psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets = 0;
    psoCreateInfo.GraphicsPipeline.DSVFormat = Diligent::TEX_FORMAT_D32_FLOAT;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = Diligent::CULL_MODE_BACK;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.FrontCounterClockwise = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = Diligent::True;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType = Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    constexpr Diligent::LayoutElement kLayoutElements[] = {
        Diligent::LayoutElement{0, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{1, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{2, 0, 2, Diligent::VT_FLOAT32, Diligent::False}};
    psoCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = kLayoutElements;
    psoCreateInfo.GraphicsPipeline.InputLayout.NumElements = 3;
    psoCreateInfo.pVS = vertexShader;

    renderDevice->CreateGraphicsPipelineState(psoCreateInfo, &mPipelineState);
    if (mPipelineState == nullptr)
    {
        return false;
    }

    if (mConstantBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name = "CRESSimNeo.ShadowPass.Constants";
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
        mPipelineState->GetStaticVariableByName(Diligent::SHADER_TYPE_VERTEX, "ShadowConstants");
    if (vertexConstants == nullptr)
    {
        return false;
    }
    vertexConstants->Set(mConstantBuffer);
    mPipelineState->CreateShaderResourceBinding(&mShaderResourceBinding, true);
    return mShaderResourceBinding != nullptr;
}

} // namespace cressim::neo::graphics::detail
