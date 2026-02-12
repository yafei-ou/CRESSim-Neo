#ifndef CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H
#define CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H

#include "common/id.h"
#include "graphics/export.h"

#include <string>
#include <unordered_set>

namespace cressim::neo::graphics
{

struct MeshHandle
{
    common::ResourceId id = common::kInvalidResourceId;
};

struct MaterialHandle
{
    common::ResourceId id = common::kInvalidResourceId;
};

struct TextureHandle
{
    common::ResourceId id = common::kInvalidResourceId;
};

struct MeshResourceDesc
{
    std::string debugName;
};

struct MaterialResourceDesc
{
    std::string debugName;
};

struct TextureResourceDesc
{
    std::string debugName;
};

class CRESSIM_NEO_GRAPHICS_API RenderResourceManager
{
public:
    MeshHandle registerMesh(const MeshResourceDesc& desc);
    MaterialHandle registerMaterial(const MaterialResourceDesc& desc);
    TextureHandle registerTexture(const TextureResourceDesc& desc);

    bool isValid(MeshHandle mesh) const;
    bool isValid(MaterialHandle material) const;
    bool isValid(TextureHandle texture) const;

private:
    common::ResourceId mNextMeshId = 1;
    common::ResourceId mNextMaterialId = 1;
    common::ResourceId mNextTextureId = 1;

    std::unordered_set<common::ResourceId> mMeshIds;
    std::unordered_set<common::ResourceId> mMaterialIds;
    std::unordered_set<common::ResourceId> mTextureIds;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H
