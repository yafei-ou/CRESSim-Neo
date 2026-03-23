#ifndef CRESSIM_NEO_GRAPHICS_SERVICES_MESH_GPU_CACHE_H
#define CRESSIM_NEO_GRAPHICS_SERVICES_MESH_GPU_CACHE_H

#include "graphics/passes/forward_draw_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace cressim::neo::graphics::detail
{

class MeshGpuCache
{
public:
    explicit MeshGpuCache(std::string debugPrefix);

    struct CachedBuffers
    {
        std::uint64_t version    = 0;
        std::uint32_t indexCount = 0;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> vertexBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> indexBuffer;
    };

    CachedBuffers *getOrCreate(const RenderResourceManager &resources,
                               const ForwardDrawCommand &drawCommand,
                               Diligent::IRenderDevice *renderDevice);

private:
    std::string mDebugPrefix;
    std::unordered_map<common::ResourceId, CachedBuffers> mCachedMeshes;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_SERVICES_MESH_GPU_CACHE_H
