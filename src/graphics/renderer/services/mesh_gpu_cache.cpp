#include "graphics/renderer/services/mesh_gpu_cache.h"

#include <utility>

namespace cressim::neo::graphics::detail
{

MeshGpuCache::MeshGpuCache(std::string debugPrefix) : mDebugPrefix(std::move(debugPrefix)) {}

MeshGpuCache::CachedBuffers* MeshGpuCache::getOrCreate(const ForwardDrawCommand& drawCommand,
                                                       Diligent::IRenderDevice* renderDevice)
{
    if (renderDevice == nullptr)
    {
        return nullptr;
    }

    auto& mesh          = mCachedMeshes[drawCommand.meshId];
    const bool recreate = mesh.vertexBuffer == nullptr || mesh.indexBuffer == nullptr ||
                          mesh.version != drawCommand.meshVersion;
    if (!recreate)
    {
        return &mesh;
    }

    Diligent::BufferDesc vertexBufferDesc{};
    const std::string vertexName = mDebugPrefix + ".VertexBuffer";
    vertexBufferDesc.Name        = vertexName.c_str();
    vertexBufferDesc.Usage       = Diligent::USAGE_IMMUTABLE;
    vertexBufferDesc.BindFlags   = Diligent::BIND_VERTEX_BUFFER;
    vertexBufferDesc.Size =
        static_cast<Diligent::Uint64>(drawCommand.vertexCount) * drawCommand.vertexStrideBytes;

    Diligent::BufferData vertexData{};
    vertexData.pData    = drawCommand.vertexData;
    vertexData.DataSize = vertexBufferDesc.Size;
    renderDevice->CreateBuffer(vertexBufferDesc, &vertexData, &mesh.vertexBuffer);
    if (mesh.vertexBuffer == nullptr)
    {
        mCachedMeshes.erase(drawCommand.meshId);
        return nullptr;
    }

    Diligent::BufferDesc indexBufferDesc{};
    const std::string indexName = mDebugPrefix + ".IndexBuffer";
    indexBufferDesc.Name        = indexName.c_str();
    indexBufferDesc.Usage       = Diligent::USAGE_IMMUTABLE;
    indexBufferDesc.BindFlags   = Diligent::BIND_INDEX_BUFFER;
    indexBufferDesc.Size =
        static_cast<Diligent::Uint64>(drawCommand.indexCount) * sizeof(std::uint32_t);

    Diligent::BufferData indexData{};
    indexData.pData    = drawCommand.indexData;
    indexData.DataSize = indexBufferDesc.Size;
    renderDevice->CreateBuffer(indexBufferDesc, &indexData, &mesh.indexBuffer);
    if (mesh.indexBuffer == nullptr)
    {
        mCachedMeshes.erase(drawCommand.meshId);
        return nullptr;
    }

    mesh.version    = drawCommand.meshVersion;
    mesh.indexCount = drawCommand.indexCount;
    return &mesh;
}

} // namespace cressim::neo::graphics::detail
