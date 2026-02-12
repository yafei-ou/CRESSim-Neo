#ifndef CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H
#define CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H

#include "common/id.h"
#include "common/math_types.h"
#include "graphics/export.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

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
    struct Vertex
    {
        common::Vec3f position{};
        common::Vec3f normal{0.0f, 1.0f, 0.0f};
        float texCoordU = 0.0f;
        float texCoordV = 0.0f;
    };

    std::string debugName;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct MaterialResourceDesc
{
    std::string debugName;
    common::Vec3f baseColor{1.0f, 1.0f, 1.0f};
    float metallic = 0.0f;
    float roughness = 0.5f;
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

    const MeshResourceDesc* tryGetMesh(MeshHandle mesh) const noexcept;
    const MaterialResourceDesc* tryGetMaterial(MaterialHandle material) const noexcept;
    const TextureResourceDesc* tryGetTexture(TextureHandle texture) const noexcept;

    std::uint64_t meshVersion(MeshHandle mesh) const noexcept;

private:
    struct MeshResource
    {
        MeshResourceDesc desc{};
        std::uint64_t version = 1;
    };

    common::ResourceId mNextMeshId = 1;
    common::ResourceId mNextMaterialId = 1;
    common::ResourceId mNextTextureId = 1;

    std::unordered_map<common::ResourceId, MeshResource> mMeshes;
    std::unordered_map<common::ResourceId, MaterialResourceDesc> mMaterials;
    std::unordered_map<common::ResourceId, TextureResourceDesc> mTextures;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H
