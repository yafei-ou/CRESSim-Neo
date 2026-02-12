#include "graphics/render_resource_manager.h"

namespace cressim::neo::graphics
{

MeshHandle RenderResourceManager::registerMesh(const MeshResourceDesc& desc)
{
    (void)desc;

    const common::ResourceId id = mNextMeshId++;
    mMeshIds.insert(id);
    return MeshHandle{id};
}

MaterialHandle RenderResourceManager::registerMaterial(const MaterialResourceDesc& desc)
{
    (void)desc;

    const common::ResourceId id = mNextMaterialId++;
    mMaterialIds.insert(id);
    return MaterialHandle{id};
}

TextureHandle RenderResourceManager::registerTexture(const TextureResourceDesc& desc)
{
    (void)desc;

    const common::ResourceId id = mNextTextureId++;
    mTextureIds.insert(id);
    return TextureHandle{id};
}

bool RenderResourceManager::isValid(MeshHandle mesh) const
{
    return mMeshIds.find(mesh.id) != mMeshIds.end();
}

bool RenderResourceManager::isValid(MaterialHandle material) const
{
    return mMaterialIds.find(material.id) != mMaterialIds.end();
}

bool RenderResourceManager::isValid(TextureHandle texture) const
{
    return mTextureIds.find(texture.id) != mTextureIds.end();
}

} // namespace cressim::neo::graphics
