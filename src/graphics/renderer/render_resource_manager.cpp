#include "graphics/render_resource_manager.h"

#include <algorithm>
#include <utility>

namespace cressim::neo::graphics
{

MeshHandle RenderResourceManager::registerMesh(const MeshResourceDesc &desc)
{
    const common::ResourceId id = mNextMeshId++;
    MeshResource resource{};
    resource.desc = desc;
    if (!desc.vertices.empty())
    {
        resource.localBoundsMin = desc.vertices.front().position;
        resource.localBoundsMax = desc.vertices.front().position;
        for (const MeshResourceDesc::Vertex &vertex : desc.vertices)
        {
            resource.localBoundsMin.x = std::min(resource.localBoundsMin.x, vertex.position.x);
            resource.localBoundsMin.y = std::min(resource.localBoundsMin.y, vertex.position.y);
            resource.localBoundsMin.z = std::min(resource.localBoundsMin.z, vertex.position.z);
            resource.localBoundsMax.x = std::max(resource.localBoundsMax.x, vertex.position.x);
            resource.localBoundsMax.y = std::max(resource.localBoundsMax.y, vertex.position.y);
            resource.localBoundsMax.z = std::max(resource.localBoundsMax.z, vertex.position.z);
        }
        resource.hasLocalBounds = true;
    }
    mMeshes.emplace(id, std::move(resource));
    return MeshHandle{id};
}

MaterialHandle RenderResourceManager::registerMaterial(const MaterialResourceDesc &desc)
{
    const common::ResourceId id = mNextMaterialId++;
    mMaterials.emplace(id, desc);
    return MaterialHandle{id};
}

TextureHandle RenderResourceManager::registerTexture(const TextureResourceDesc &desc)
{
    const common::ResourceId id = mNextTextureId++;
    mTextures.emplace(id, desc);
    return TextureHandle{id};
}

bool RenderResourceManager::isValid(MeshHandle mesh) const
{
    return mMeshes.find(mesh.id) != mMeshes.end();
}

bool RenderResourceManager::isValid(MaterialHandle material) const
{
    return mMaterials.find(material.id) != mMaterials.end();
}

bool RenderResourceManager::isValid(TextureHandle texture) const
{
    return mTextures.find(texture.id) != mTextures.end();
}

const MeshResourceDesc *RenderResourceManager::tryGetMesh(MeshHandle mesh) const noexcept
{
    const auto it = mMeshes.find(mesh.id);
    if (it == mMeshes.end())
    {
        return nullptr;
    }
    return &it->second.desc;
}

const MaterialResourceDesc *RenderResourceManager::tryGetMaterial(
    MaterialHandle material) const noexcept
{
    const auto it = mMaterials.find(material.id);
    if (it == mMaterials.end())
    {
        return nullptr;
    }
    return &it->second;
}

const TextureResourceDesc *RenderResourceManager::tryGetTexture(
    TextureHandle texture) const noexcept
{
    const auto it = mTextures.find(texture.id);
    if (it == mTextures.end())
    {
        return nullptr;
    }
    return &it->second;
}

bool RenderResourceManager::tryGetMeshLocalBounds(MeshHandle mesh, Diligent::float3 &outMin,
                                                  Diligent::float3 &outMax) const noexcept
{
    const auto it = mMeshes.find(mesh.id);
    if (it == mMeshes.end() || !it->second.hasLocalBounds)
    {
        outMin = {};
        outMax = {};
        return false;
    }

    outMin = it->second.localBoundsMin;
    outMax = it->second.localBoundsMax;
    return true;
}

std::uint64_t RenderResourceManager::meshVersion(MeshHandle mesh) const noexcept
{
    const auto it = mMeshes.find(mesh.id);
    if (it == mMeshes.end())
    {
        return 0;
    }
    return it->second.version;
}

} // namespace cressim::neo::graphics
