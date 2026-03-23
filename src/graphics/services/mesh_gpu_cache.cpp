#include "graphics/services/mesh_gpu_cache.h"

#include <utility>

namespace cressim::neo::graphics::detail
{

MeshGpuCache::MeshGpuCache(std::string debugPrefix) : mDebugPrefix(std::move(debugPrefix)) {}

MeshGpuCache::CachedBuffers *MeshGpuCache::getOrCreate(const RenderResourceManager &resources,
                                                       const ForwardDrawCommand &drawCommand,
                                                       Diligent::IRenderDevice *renderDevice)
{
    if (renderDevice == nullptr)
    {
        return nullptr;
    }
    const MeshResourceDesc *meshDesc = resources.tryGetMesh(MeshHandle{drawCommand.meshId});
    if (meshDesc == nullptr || meshDesc->vertices.empty() || meshDesc->indices.size() < 3)
    {
        return nullptr;
    }

    auto &mesh          = mCachedMeshes[drawCommand.meshId];
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
        static_cast<Diligent::Uint64>(meshDesc->vertices.size()) * sizeof(MeshResourceDesc::Vertex);

    Diligent::BufferData vertexData{};
    vertexData.pData    = meshDesc->vertices.data();
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
        static_cast<Diligent::Uint64>(meshDesc->indices.size()) * sizeof(std::uint32_t);

    Diligent::BufferData indexData{};
    indexData.pData    = meshDesc->indices.data();
    indexData.DataSize = indexBufferDesc.Size;
    renderDevice->CreateBuffer(indexBufferDesc, &indexData, &mesh.indexBuffer);
    if (mesh.indexBuffer == nullptr)
    {
        mCachedMeshes.erase(drawCommand.meshId);
        return nullptr;
    }

    mesh.version    = drawCommand.meshVersion;
    mesh.indexCount = static_cast<std::uint32_t>(meshDesc->indices.size());
    return &mesh;
}

} // namespace cressim::neo::graphics::detail
