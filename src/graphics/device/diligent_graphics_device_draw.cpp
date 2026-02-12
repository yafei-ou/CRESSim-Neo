#include "graphics/device/diligent_graphics_device.h"

#include "graphics/device/shaders/shader_fallback_sources.h"

#include <cstring>

namespace cressim::neo::graphics
{

bool DiligentGraphicsDevice::drawPbr(RenderTargetHandle target, const PbrDrawCommand& drawCommand)
{
    if (!mInitialized || mBackend != GraphicsBackend::Vulkan || !mImmediateContext || !mRenderDevice)
    {
        return false;
    }
    if (!mHasActiveRenderTarget || mActiveRenderTarget.id != target.id)
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

    CachedMeshGpuData* meshBuffers = getOrCreateMeshBuffers(drawCommand);
    if (meshBuffers == nullptr || meshBuffers->vertexBuffer == nullptr || meshBuffers->indexBuffer == nullptr || meshBuffers->indexCount == 0)
    {
        return false;
    }

    PbrPipelineResources* pipeline = getOrCreatePbrPipeline(mActiveRenderTargetHasDepth);
    if (pipeline == nullptr || pipeline->pipelineState == nullptr || pipeline->shaderResourceBinding == nullptr || mPbrConstantBuffer == nullptr)
    {
        return false;
    }

    PbrDrawConstants constants{};
    std::memcpy(constants.modelMatrix, drawCommand.modelMatrix, sizeof(constants.modelMatrix));
    std::memcpy(constants.viewProjectionMatrix, drawCommand.viewProjectionMatrix, sizeof(constants.viewProjectionMatrix));
    constants.cameraPositionMetallic[0] = drawCommand.cameraPosition[0];
    constants.cameraPositionMetallic[1] = drawCommand.cameraPosition[1];
    constants.cameraPositionMetallic[2] = drawCommand.cameraPosition[2];
    constants.cameraPositionMetallic[3] = drawCommand.material.metallic;
    constants.lightDirectionIntensity[0] = drawCommand.light.direction[0];
    constants.lightDirectionIntensity[1] = drawCommand.light.direction[1];
    constants.lightDirectionIntensity[2] = drawCommand.light.direction[2];
    constants.lightDirectionIntensity[3] = drawCommand.light.intensity;
    constants.lightColorRoughness[0] = drawCommand.light.color[0];
    constants.lightColorRoughness[1] = drawCommand.light.color[1];
    constants.lightColorRoughness[2] = drawCommand.light.color[2];
    constants.lightColorRoughness[3] = drawCommand.material.roughness;
    constants.baseColor[0] = drawCommand.material.baseColor[0];
    constants.baseColor[1] = drawCommand.material.baseColor[1];
    constants.baseColor[2] = drawCommand.material.baseColor[2];
    constants.baseColor[3] = 1.0f;

    void* mappedConstants = nullptr;
    mImmediateContext->MapBuffer(mPbrConstantBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &constants, sizeof(constants));
    mImmediateContext->UnmapBuffer(mPbrConstantBuffer, Diligent::MAP_WRITE);

    const Diligent::Uint64 vertexOffset = 0;
    Diligent::IBuffer* vertexBuffers[] = {meshBuffers->vertexBuffer};
    mImmediateContext->SetVertexBuffers(
        0,
        1,
        vertexBuffers,
        &vertexOffset,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        Diligent::SET_VERTEX_BUFFERS_FLAG_RESET);
    mImmediateContext->SetIndexBuffer(meshBuffers->indexBuffer, 0, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    mImmediateContext->SetPipelineState(pipeline->pipelineState);
    mImmediateContext->CommitShaderResources(pipeline->shaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedAttribs drawAttrs{};
    drawAttrs.IndexType = Diligent::VT_UINT32;
    drawAttrs.NumIndices = meshBuffers->indexCount;
    drawAttrs.Flags = Diligent::DRAW_FLAG_VERIFY_ALL;
    mImmediateContext->DrawIndexed(drawAttrs);
    return true;
}

DiligentGraphicsDevice::CachedMeshGpuData* DiligentGraphicsDevice::getOrCreateMeshBuffers(const PbrDrawCommand& drawCommand)
{
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
    vertexBufferDesc.Name = "CRESSimNeo.Pbr.VertexBuffer";
    vertexBufferDesc.Usage = Diligent::USAGE_IMMUTABLE;
    vertexBufferDesc.BindFlags = Diligent::BIND_VERTEX_BUFFER;
    vertexBufferDesc.Size = static_cast<Diligent::Uint64>(drawCommand.vertexCount) * drawCommand.vertexStrideBytes;

    Diligent::BufferData vertexData{};
    vertexData.pData = drawCommand.vertexData;
    vertexData.DataSize = vertexBufferDesc.Size;
    mRenderDevice->CreateBuffer(vertexBufferDesc, &vertexData, &mesh.vertexBuffer);
    if (mesh.vertexBuffer == nullptr)
    {
        mCachedMeshes.erase(drawCommand.meshId);
        return nullptr;
    }

    Diligent::BufferDesc indexBufferDesc{};
    indexBufferDesc.Name = "CRESSimNeo.Pbr.IndexBuffer";
    indexBufferDesc.Usage = Diligent::USAGE_IMMUTABLE;
    indexBufferDesc.BindFlags = Diligent::BIND_INDEX_BUFFER;
    indexBufferDesc.Size = static_cast<Diligent::Uint64>(drawCommand.indexCount) * sizeof(std::uint32_t);

    Diligent::BufferData indexData{};
    indexData.pData = drawCommand.indexData;
    indexData.DataSize = indexBufferDesc.Size;
    mRenderDevice->CreateBuffer(indexBufferDesc, &indexData, &mesh.indexBuffer);
    if (mesh.indexBuffer == nullptr)
    {
        mCachedMeshes.erase(drawCommand.meshId);
        return nullptr;
    }

    mesh.version = drawCommand.meshVersion;
    mesh.indexCount = drawCommand.indexCount;
    return &mesh;
}

bool DiligentGraphicsDevice::createPbrPipeline(bool hasDepthTarget, PbrPipelineResources& outResources)
{
    if (!mRenderDevice)
    {
        return false;
    }

    std::string pbrVsSource;
    if (!loadShaderSource("pbr.vs.hlsl", shaders::pbrVertex(), pbrVsSource))
    {
        return false;
    }

    std::string pbrPsSource;
    if (!loadShaderSource("pbr.ps.hlsl", shaders::pbrPixel(), pbrPsSource))
    {
        return false;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.CompileFlags = Diligent::SHADER_COMPILE_FLAG_PACK_MATRIX_ROW_MAJOR;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint = "main";

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name = hasDepthTarget ? "CRESSimNeo.Pbr.VS.Depth" : "CRESSimNeo.Pbr.VS.NoDepth";
    shaderCreateInfo.Source = pbrVsSource.c_str();
    mRenderDevice->CreateShader(shaderCreateInfo, &vertexShader);
    if (vertexShader == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name = hasDepthTarget ? "CRESSimNeo.Pbr.PS.Depth" : "CRESSimNeo.Pbr.PS.NoDepth";
    shaderCreateInfo.Source = pbrPsSource.c_str();
    mRenderDevice->CreateShader(shaderCreateInfo, &pixelShader);
    if (pixelShader == nullptr)
    {
        return false;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name = hasDepthTarget ? "CRESSimNeo.Pbr.PSO.Depth" : "CRESSimNeo.Pbr.PSO.NoDepth";
    psoCreateInfo.PSODesc.PipelineType = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0] = Diligent::TEX_FORMAT_RGBA8_UNORM;
    psoCreateInfo.GraphicsPipeline.DSVFormat = hasDepthTarget ? Diligent::TEX_FORMAT_D32_FLOAT : Diligent::TEX_FORMAT_UNKNOWN;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = Diligent::CULL_MODE_BACK;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.FrontCounterClockwise = Diligent::True;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = hasDepthTarget ? Diligent::True : Diligent::False;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = hasDepthTarget ? Diligent::True : Diligent::False;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType = Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    constexpr Diligent::LayoutElement layoutElements[] = {
        Diligent::LayoutElement{0, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{1, 0, 3, Diligent::VT_FLOAT32, Diligent::False},
        Diligent::LayoutElement{2, 0, 2, Diligent::VT_FLOAT32, Diligent::False}};
    psoCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = layoutElements;
    psoCreateInfo.GraphicsPipeline.InputLayout.NumElements = 3;
    psoCreateInfo.pVS = vertexShader;
    psoCreateInfo.pPS = pixelShader;

    mRenderDevice->CreateGraphicsPipelineState(psoCreateInfo, &outResources.pipelineState);
    if (outResources.pipelineState == nullptr)
    {
        return false;
    }

    if (mPbrConstantBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name = "CRESSimNeo.Pbr.Constants";
        constantBufferDesc.Size = sizeof(PbrDrawConstants);
        constantBufferDesc.Usage = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
        mRenderDevice->CreateBuffer(constantBufferDesc, nullptr, &mPbrConstantBuffer);
        if (mPbrConstantBuffer == nullptr)
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
    vertexConstants->Set(mPbrConstantBuffer);
    pixelConstants->Set(mPbrConstantBuffer);
    outResources.pipelineState->CreateShaderResourceBinding(&outResources.shaderResourceBinding, true);
    return outResources.shaderResourceBinding != nullptr;
}

DiligentGraphicsDevice::PbrPipelineResources* DiligentGraphicsDevice::getOrCreatePbrPipeline(bool hasDepthTarget)
{
    PbrPipelineResources& resources = hasDepthTarget ? mPbrPipelineWithDepth : mPbrPipelineNoDepth;
    if (resources.pipelineState != nullptr && resources.shaderResourceBinding != nullptr)
    {
        return &resources;
    }

    if (!createPbrPipeline(hasDepthTarget, resources))
    {
        return nullptr;
    }

    return &resources;
}

} // namespace cressim::neo::graphics
